import math

from core.ir import Immediate, Instruction, Mod, Op, PtxType, Register

from ..fold import round_f32
from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass

F32 = PtxType.F32


@register_pass
class FastMath(FunctionPass):
    """IEEE-rounded f32 special functions -> the SFU's approximate forms.

        div.rn  d, a, sqrt(x)   ->  rsqrt.approx t, x;  mul d, a, t     (sqrt read only here)
        rcp.rn  d, sqrt(x)      ->  rsqrt.approx d, x
        div.rn  d, a, c         ->  mul d, a, 1/c            (c an immediate, 1/c a normal f32)
        div.rn  d, 1.0, y       ->  rcp.approx d, y
        div.rn  d, a, y         ->  div.full d, a, y
        rcp.rn / sqrt.rn        ->  rcp.approx / sqrt.approx

    .rn division and sqrt are multi-instruction software sequences; the .approx forms are one
    MUFU op (div.full is rcp + mul with range scaling). Errors stay within 2 ulp (rsqrt: 2^-22.9
    relative), far below the 1e-3 tolerance. No .ftz: subnormals keep their values. Runs after
    const-fold, which does not fold .approx ops, and before fuse-mad, so a*(1/c)+b becomes fma.
    """

    name = "fast-math"
    preserves = CFG_ANALYSES

    def run(self, function, am):
        du = am.get("defuse")
        changed = False
        for block in function.blocks:
            for k, inst in enumerate(block.instructions):
                if inst.op not in (Op.DIV, Op.RCP, Op.SQRT) or inst.ptx_type is not F32 \
                        or Mod.RN not in inst.types:
                    continue
                new = self.rewrite(inst, du)
                if new is inst:
                    changed = True
                elif new is not None:
                    new.guard, new.negate, new.comment = inst.guard, inst.negate, inst.comment
                    block.instructions[k] = new
                    changed = True
        return changed

    def rewrite(self, inst, du):
        d, s = inst.dsts[0], inst.srcs
        if inst.op is Op.SQRT:
            inst.types = (Mod.APPROX, F32)   # in place: a later div/rcp may still turn it into rsqrt
            return inst
        y = s[-1]
        sq = self.sole_sqrt(y, inst, du)
        if sq is not None:
            # the sqrt feeds only this instruction: turn it into rsqrt in place
            sq.op, sq.types = Op.RSQRT, (Mod.APPROX, F32)
            if inst.op is Op.RCP:
                return Instruction(Op.MOV, (F32,), [d], [y])
            return Instruction(Op.MUL, (F32,), [d], [s[0], y])
        if inst.op is Op.RCP:
            return Instruction(Op.RCP, (Mod.APPROX, F32), [d], s)
        a = s[0]
        r = _reciprocal(float(y.value)) if isinstance(y, Immediate) else None
        if r is not None:
            return Instruction(Op.MUL, (F32,), [d], [a, Immediate(r, F32)])
        if isinstance(a, Immediate) and float(a.value) == 1.0:
            return Instruction(Op.RCP, (Mod.APPROX, F32), [d], [y])
        return Instruction(Op.DIV, (Mod.FULL, F32), [d], s)

    @staticmethod
    def sole_sqrt(y, inst, du):
        """The sqrt.rn defining y, when y has that one def and inst is its only reader."""
        if not isinstance(y, Register):
            return None
        sq = du.single_def(y)
        if sq is None or sq.op is not Op.SQRT or sq.ptx_type is not F32:   # .rn, or .approx already
            return None
        if du.uses_of(y) != [inst] or inst.srcs.count(y) != 1:
            return None
        return sq


def _reciprocal(c):
    """1/c rounded to f32 when that is a finite normal number (multiplying by it is then within
    1.5 ulp of dividing by c), else None."""
    if c == 0.0 or not math.isfinite(c):
        return None
    r = round_f32(1.0 / c)
    if not math.isfinite(r) or abs(r) < 2.0 ** -126:
        return None
    return r
