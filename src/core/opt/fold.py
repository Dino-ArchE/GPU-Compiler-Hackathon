"""Compile-time evaluation of PTX arithmetic on immediates, with the hardware's semantics.

Integers wrap to the instruction type; signed div/rem truncate toward zero like C. f32 results
are rounded to nearest-even, which is exact for add/sub/mul/div/sqrt because Python computes
in f64 first (f64 has more than 2*24+2 bits). Ops whose hardware result we cannot reproduce
bit-for-bit (.approx, fused mad/fma) are not folded.
"""

import math
import struct

from core.ir import Cmp, Mod, Op, PtxType

from .analysis.ranges import WIDE, is_int, wrap

FLOAT_FOLDABLE = {Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.SQRT, Op.RCP, Op.MIN, Op.MAX, Op.NEG, Op.ABS}


def round_f32(x):
    if not math.isfinite(x):
        return x
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def c_div(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def result_type(inst):
    t = inst.ptx_type
    return WIDE[t] if Mod.WIDE in inst.types else t


def evaluate(inst, values):
    """The result of `inst` on constant source values (Python ints/floats), or None if unknown."""
    op, mods, t = inst.op, inst.types, inst.ptx_type
    if t is None:
        return None
    if t is PtxType.F32 or t is PtxType.F64:
        return _evaluate_float(op, mods, t, values)
    if not is_int(t):
        return None
    vals = [wrap(int(v), t) for v in values]
    n = int(t.value[1:])
    rt = result_type(inst)
    match op:
        case Op.ADD:
            r = vals[0] + vals[1]
        case Op.SUB:
            r = vals[0] - vals[1]
        case Op.MUL:
            r = vals[0] * vals[1]
            if Mod.HI in mods:
                r >>= n
        case Op.MAD:
            if Mod.HI in mods:
                return None
            r = vals[0] * vals[1] + wrap(int(values[2]), rt)
        case Op.DIV:
            if vals[1] == 0:
                return None
            r = c_div(vals[0], vals[1])
        case Op.REM:
            if vals[1] == 0:
                return None
            r = vals[0] - c_div(vals[0], vals[1]) * vals[1]
        case Op.MIN:
            r = min(vals)
        case Op.MAX:
            r = max(vals)
        case Op.NEG:
            r = -vals[0]
        case Op.ABS:
            r = abs(vals[0])
        case Op.AND:
            r = vals[0] & vals[1]
        case Op.OR:
            r = vals[0] | vals[1]
        case Op.XOR:
            r = vals[0] ^ vals[1]
        case Op.NOT:
            r = ~vals[0]
        case Op.SHL:
            r = vals[0] << min(vals[1] & 0xFFFFFFFF, n)
        case Op.SHR:
            k = min(vals[1] & 0xFFFFFFFF, n)
            r = vals[0] >> k          # arithmetic for signed types; unsigned values are already >= 0
        case Op.MOV:
            r = vals[0]
        case _:
            return None
    return wrap(r, rt)


def _evaluate_float(op, mods, t, values):
    if op not in FLOAT_FOLDABLE or Mod.APPROX in mods or Mod.FTZ in mods:
        return None
    if op in (Op.DIV, Op.RCP, Op.SQRT) and Mod.RN not in mods:
        return None
    v = [float(x) for x in values]
    try:
        match op:
            case Op.ADD:
                r = v[0] + v[1]
            case Op.SUB:
                r = v[0] - v[1]
            case Op.MUL:
                r = v[0] * v[1]
            case Op.DIV:
                r = v[0] / v[1]
            case Op.RCP:
                r = 1.0 / v[0]
            case Op.SQRT:
                r = math.sqrt(v[0])
            case Op.MIN:
                r = min(v)
            case Op.MAX:
                r = max(v)
            case Op.NEG:
                r = -v[0]
            case Op.ABS:      # the last of FLOAT_FOLDABLE
                r = abs(v[0])
    except (ZeroDivisionError, ValueError):
        return None          # leave inf/nan cases to the hardware
    if any(math.isnan(x) for x in (*v, r)):
        return None
    return round_f32(r) if t is PtxType.F32 else r


def compare(cmp, t, a, b):
    """setp on constants -> bool, or None."""
    if is_int(t):
        a, b = wrap(int(a), t), wrap(int(b), t)
    elif t not in (PtxType.F32, PtxType.F64) or math.isnan(a) or math.isnan(b):
        return None
    return {Cmp.EQ: a == b, Cmp.NE: a != b, Cmp.LT: a < b, Cmp.LE: a <= b,
            Cmp.GT: a > b, Cmp.GE: a >= b}.get(Cmp(cmp))
