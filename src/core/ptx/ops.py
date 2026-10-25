"""Elementwise ops (Assign.op.name) -> instructions. One entry per op: add a line for a new op.

Each lowering takes (builder, ptx_type, *args) with args as Values and returns the result
Register. The op set is the f32 one Emmy's own renderer accepts (op_to_expr); composite ops
(gelu, silu, softplus) are already decomposed into these before Kernel IR.

Masks: a comparison whose result is typed bool is a predicate register (setp), which `where`
and the logical ops consume directly; typed f32 it is Emmy's 1.0 / 0.0 mask convention, and any
nonzero value counts as true.

Accuracy: arithmetic, sqrt and reciprocal are IEEE-rounded (.rn). PTX has no exact exp/log/sin,
only .approx forms (~2 ulp) on a power-of-two base, so those are built from ex2/lg2 with a
range reduction where the raw form is not accurate enough. tanh.approx (~2^-11) is too coarse
for rtol 1e-4, so tanh and erf are formulas over exp.
"""

import math

from core.ir import Cmp, Immediate, Mod, Op, PtxType, Register

LOG2E = math.log2(math.e)
LN2 = math.log(2.0)
TWO_PI = 2.0 * math.pi

# Abramowitz & Stegun 7.1.26: erf(x) = 1 - (a1 t + ... + a5 t^5) e^(-x^2), t = 1/(1 + p x),
# for x >= 0; max abs error 1.5e-7
ERF_P = 0.3275911
ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _imm(value, t):
    return Immediate(value, t)


def _select(b, t, pred, a, c):
    """pred ? a : c"""
    return b.selp(t, a, c, pred)


def _mask(b, t, pred):
    """A predicate as Emmy's mask value: 1.0 / 0.0 (or 1 / 0 for integers)."""
    return _select(b, t, pred, _imm(1, t), _imm(0, t))


def _value_type(v):
    """The type an operand holds, for comparing it: an immediate's own type, a register's
    declared one (32/64-bit integer registers are declared untyped .b32/.b64; compare signed)."""
    if isinstance(v, Immediate):
        return v.ptx_type
    t = type(v).PTX_TYPE
    return {PtxType.B32: PtxType.S32, PtxType.B64: PtxType.S64}.get(t, t)


def _is_pred(v):
    return isinstance(v, Register) and _value_type(v) is PtxType.PRED


def _truthy(b, x):
    """x as a predicate: a bool is one already, a number is true when nonzero."""
    if _is_pred(x):
        return x
    t = _value_type(x)
    return b.setp(Cmp.NE, t, x, _imm(0, t))


def _result(b, t, pred):
    """A predicate as the Assign's result type: itself for bool, else a 1 / 0 mask."""
    return pred if t is PtxType.PRED else _mask(b, t, pred)


def _exp(b, t, x):
    # PTX has only 2^x: e^x = 2^(x * log2(e))
    return b.unary(Op.EX2, t, b.mul(t, x, _imm(LOG2E, t)), mode=Mod.APPROX)


def _log(b, t, x):
    # ln x = log2(x) * ln(2)
    return b.mul(t, b.unary(Op.LG2, t, x, mode=Mod.APPROX), _imm(LN2, t))


def _sigmoid(b, t, x):
    # 1 / (1 + e^-x): e^-x overflows to inf for very negative x, giving 0 as it should
    denom = b.add(t, _exp(b, t, b.unary(Op.NEG, t, x)), _imm(1.0, t))
    return b.unary(Op.RCP, t, denom, mode=Mod.RN)


def _tanh(b, t, x):
    # tanh x = 1 - 2 / (e^(2x) + 1): saturates cleanly to +-1 at both ends
    e2x = _exp(b, t, b.mul(t, x, _imm(2.0, t)))
    frac = b.binary(Op.DIV, t, _imm(2.0, t), b.add(t, e2x, _imm(1.0, t)), mode=Mod.RN)
    return b.binary(Op.SUB, t, _imm(1.0, t), frac)


def _erf(b, t, x):
    ax = b.unary(Op.ABS, t, x)
    tt = b.unary(Op.RCP, t, b.mad(t, ax, _imm(ERF_P, t), _imm(1.0, t)), mode=Mod.RN)
    poly = _imm(ERF_A[-1], t)
    for a in reversed(ERF_A[:-1]):          # Horner: ((a5 t + a4) t + a3) t ...
        poly = b.mad(t, poly, tt, _imm(a, t))
    poly = b.mul(t, poly, tt)
    gauss = _exp(b, t, b.unary(Op.NEG, t, b.mul(t, ax, ax)))
    y = b.binary(Op.SUB, t, _imm(1.0, t), b.mul(t, poly, gauss))
    return _select(b, t, b.setp(Cmp.LT, t, x, _imm(0.0, t)), b.unary(Op.NEG, t, y), y)


def _reduce_angle(b, t, x):
    """x - 2pi * round(x / 2pi): sin/cos.approx are only accurate on [-pi, pi]."""
    k = b.cvt(t, t, b.mul(t, x, _imm(1.0 / TWO_PI, t)), mode=Mod.RNI)
    return b.mad(t, b.unary(Op.NEG, t, k), _imm(TWO_PI, t), x)


def _sin(b, t, x):
    return b.unary(Op.SIN, t, _reduce_angle(b, t, x), mode=Mod.APPROX)


def _cos(b, t, x):
    return b.unary(Op.COS, t, _reduce_angle(b, t, x), mode=Mod.APPROX)


def _pow(b, t, x, y):
    """x^y = 2^(y log2 |x|), with C powf's rules for the cases that formula gets wrong:
    a negative base is valid only for an integer y (negated when y is odd), and y == 0 -> 1."""
    r = b.unary(Op.EX2, t, b.mul(t, y, b.unary(Op.LG2, t, b.unary(Op.ABS, t, x), mode=Mod.APPROX)), mode=Mod.APPROX)
    yi = b.cvt(t, t, y, mode=Mod.RNI)
    integral = b.setp(Cmp.EQ, t, y, yi)
    half = b.cvt(t, t, b.mul(t, yi, _imm(0.5, t)), mode=Mod.RMI)      # floor(y / 2)
    odd = b.setp(Cmp.NE, t, b.mul(t, half, _imm(2.0, t)), yi)
    negative = b.setp(Cmp.LT, t, x, _imm(0.0, t))
    r = _select(b, t, b.binary(Op.AND, PtxType.PRED, negative, odd), b.unary(Op.NEG, t, r), r)
    nan_case = b.binary(Op.AND, PtxType.PRED, negative, b.unary(Op.NOT, PtxType.PRED, integral))
    r = _select(b, t, nan_case, _imm(math.nan, t), r)
    return _select(b, t, b.setp(Cmp.EQ, t, y, _imm(0.0, t)), _imm(1.0, t), r)


def _compare(cmp):
    # the comparison runs at the operands' type; the result may be bool (a predicate)
    return lambda b, t, x, y: _result(b, t, b.setp(cmp, _value_type(x), x, y))


def _logical(op):
    return lambda b, t, x, y: _result(b, t, b.binary(op, PtxType.PRED, _truthy(b, x), _truthy(b, y)))


ELEMENTWISE = {
    "add":           lambda b, t, x, y: b.add(t, x, y),
    "subtract":      lambda b, t, x, y: b.binary(Op.SUB, t, x, y),
    "multiply":      lambda b, t, x, y: b.mul(t, x, y),
    "divide":        lambda b, t, x, y: b.binary(Op.DIV, t, x, y, mode=None if not t.is_float else Mod.RN),
    "maximum":       lambda b, t, x, y: b.binary(Op.MAX, t, x, y),
    "minimum":       lambda b, t, x, y: b.binary(Op.MIN, t, x, y),
    "pow":           _pow,
    "square":        lambda b, t, x: b.mul(t, x, x),
    "relu":          lambda b, t, x: b.binary(Op.MAX, t, x, _imm(0.0, t)),
    "negative":      lambda b, t, x: b.unary(Op.NEG, t, x),
    "abs":           lambda b, t, x: b.unary(Op.ABS, t, x),
    "reciprocal":    lambda b, t, x: b.unary(Op.RCP, t, x, mode=Mod.RN),
    "sqrt":          lambda b, t, x: b.unary(Op.SQRT, t, x, mode=Mod.RN),
    "rsqrt":         lambda b, t, x: b.unary(Op.RCP, t, b.unary(Op.SQRT, t, x, mode=Mod.RN), mode=Mod.RN),
    "exp":           _exp,
    "exp_fast":      _exp,
    "log":           _log,
    "sigmoid":       _sigmoid,
    "tanh":          _tanh,
    "erf":           _erf,
    "sin":           _sin,
    "cos":           _cos,
    "copy":          lambda b, t, x: b.mov(t, x),
    "pad":           lambda b, t, x: b.mov(t, x),
    # masks: comparisons and logical combines produce 1 / 0
    "equal":         _compare(Cmp.EQ),
    "not_equal":     _compare(Cmp.NE),
    "less":          _compare(Cmp.LT),
    "less_equal":    _compare(Cmp.LE),
    "greater":       _compare(Cmp.GT),
    "greater_equal": _compare(Cmp.GE),
    "bitwise_and":   _logical(Op.AND),
    "bitwise_or":    _logical(Op.OR),
    "bitwise_not":   lambda b, t, x: _result(b, t, b.unary(Op.NOT, PtxType.PRED, _truthy(b, x))),
    "where":         lambda b, t, c, x, y: _select(b, t, _truthy(b, c), x, y),
}


def lower_elementwise(builder, name, ptx_type, args):
    if name not in ELEMENTWISE:
        raise NotImplementedError(f"elementwise op {name!r}")
    return ELEMENTWISE[name](builder, ptx_type, *args)
