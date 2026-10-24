import math

from core.ir import Immediate, Instruction, Op, PtxType, Register

from ..analysis.ranges import is_int, wrap
from ..fold import result_type
from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass

COMMUTATIVE = {Op.ADD, Op.MUL, Op.MIN, Op.MAX, Op.AND, Op.OR, Op.XOR}


@register_pass
class InstCombine(FunctionPass):
    """Algebraic peepholes, one instruction at a time (LLVM's instcombine in miniature).

    canonicalize   commutative ops put the immediate second: add 4, %r -> add %r, 4
                   sub %r, c -> add %r, -c (so one set of add patterns covers both)
    identities     x+0, x*1, x/1, x|0, x^0, x<<0, x&~0, min(x,x) -> x;  x*0, x&0, x%1 -> 0
                   mad a,b,0 -> mul;  mad a,1,c -> add;  mad a,0,c -> c
    floats         x*1.0 -> x, x*-1.0 -> -x, x+(-0.0) -> x, x-0.0 -> x (all bit-exact);
                   x/2^k -> x*2^-k (same correctly rounded result)
    reassociate    (x+c1)+c2 -> x+(c1+c2),  (x*c1)*c2 -> x*(c1*c2),  (x<<a)<<b -> x<<(a+b)
                   when the inner result is a single-def register (x must hold one value too)

    Integer identities hold in wrap-around arithmetic, so no range facts are needed. Float
    rewrites skip .ftz/.sat/.approx forms, whose results these identities don't describe.
    """

    name = "instcombine"
    preserves = CFG_ANALYSES

    def run(self, function, am):
        self.du = am.get("defuse")
        changed = False
        for block in function.blocks:
            for k, inst in enumerate(block.instructions):
                if inst.has_side_effects or len(inst.dsts) != 1:
                    continue
                changed |= self.canonicalize(inst)
                new = self.simplify(inst)
                if new is not None:
                    new.guard, new.negate, new.comment = inst.guard, inst.negate, inst.comment
                    block.instructions[k] = new
                    if self.du.single_def(inst.dsts[0]) is inst:
                        self.du.defs[inst.dsts[0]] = [new]      # later chains see the new form
                    changed = True
        return changed

    def canonicalize(self, inst):
        s = inst.srcs
        if inst.op in COMMUTATIVE and len(s) == 2 and _imm(s[0]) and not _imm(s[1]):
            inst.srcs = (s[1], s[0])
            return True
        if inst.op in (Op.MAD, Op.FMA) and _imm(s[0]) and not _imm(s[1]):
            inst.srcs = (s[1], s[0], s[2])
            return True
        t = inst.ptx_type
        if inst.op is Op.SUB and is_int(t) and _imm(s[1]):
            inst.op, inst.srcs = Op.ADD, (s[0], Immediate(wrap(-int(s[1].value), t), t))
            return True
        return False

    def simplify(self, inst):
        t = inst.ptx_type
        if t is None or not inst.srcs:
            return None
        if is_int(t):
            return self.simplify_int(inst, t)
        if t in (PtxType.F32, PtxType.F64) and not ({"ftz", "sat", "approx"} & set(inst.types)):
            return self.simplify_float(inst, t)
        return None

    def simplify_int(self, inst, t):
        op, s, d = inst.op, inst.srcs, inst.dsts[0]
        rt = result_type(inst)
        mods = inst.types
        b = _ival(s[1], t) if len(s) > 1 else None
        if op is Op.MAD and "hi" not in mods:
            c = s[2]
            if b == 0:
                return _mov(d, rt, c)
            if _ival(c, rt) == 0:
                return Instruction(Op.MUL, mods, [d], s[:2])
            if b == 1 and "lo" in mods:
                return Instruction(Op.ADD, (t,), [d], [s[0], c])
            return None
        if len(s) != 2:
            return None
        x = s[0]
        if b == 0 and op in (Op.ADD, Op.OR, Op.XOR, Op.SHL, Op.SHR):
            return _mov(d, t, x)
        if b == 1 and (op is Op.DIV or op is Op.MUL and "lo" in mods):
            return _mov(d, t, x)
        if op is Op.AND and b == wrap(-1, t):
            return _mov(d, t, x)
        if b == 0 and op in (Op.MUL, Op.AND):
            return _mov(d, rt, Immediate(0, rt))
        if op is Op.REM and (b == 1 or b == -1 and t.value.startswith("s")):
            return _mov(d, t, Immediate(0, t))
        if op in (Op.MIN, Op.MAX) and x is s[1]:
            return _mov(d, t, x)
        if op is Op.SUB and x is s[1] and isinstance(x, Register):
            return _mov(d, t, Immediate(0, t))
        return self.reassociate(inst, t, b)

    def reassociate(self, inst, t, c2):
        op, s, d = inst.op, inst.srcs, inst.dsts[0]
        if c2 is None or not isinstance(s[0], Register):
            return None
        inner = self.du.single_def(s[0])
        if inner is None or inner.op is not op or inner.types != inst.types or len(inner.srcs) != 2:
            return None
        x, c1 = inner.srcs[0], _ival(inner.srcs[1], t)
        if c1 is None or not (isinstance(x, Immediate) or self.du.is_ssa(x)) or x is s[0]:
            return None
        if op is Op.ADD:
            c = wrap(c1 + c2, t)
        elif op is Op.MUL and "lo" in inst.types:
            c = wrap(c1 * c2, t)
        elif op is Op.SHL and c1 + c2 < int(t.value[1:]):
            c = c1 + c2
        else:
            return None
        return Instruction(op, inst.types, [d], [x, Immediate(c, t)])

    def simplify_float(self, inst, t):
        op, s, d = inst.op, inst.srcs, inst.dsts[0]
        if len(s) != 2 or not _imm(s[1]):
            return None
        x, c = s[0], float(s[1].value)
        neg_zero = c == 0.0 and math.copysign(1.0, c) < 0
        if op is Op.MUL and c == 1.0:
            return _mov(d, t, x)
        if op is Op.MUL and c == -1.0:
            return Instruction(Op.NEG, (t,), [d], [x])
        if op is Op.ADD and neg_zero or op is Op.SUB and c == 0.0 and not neg_zero:
            return _mov(d, t, x)
        if op is Op.DIV and "rn" in inst.types and c != 0.0:
            m, e = math.frexp(abs(c))
            if m == 0.5 and -125 <= 1 - e <= 127:          # c = +-2^(e-1); 1/c is a normal f32
                return Instruction(Op.MUL, ("rn", t), [d], [x, Immediate(1.0 / c, t)])
        return None


def _imm(v):
    return isinstance(v, Immediate)


def _ival(v, t):
    """An integer immediate's value as type t holds it, else None."""
    if isinstance(v, Immediate) and not isinstance(v.value, float):
        return wrap(int(v.value), t)
    return None


def _mov(d, t, v):
    return Instruction(Op.MOV, (t,), [d], [v])

