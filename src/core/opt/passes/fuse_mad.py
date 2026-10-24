from core.ir import Instruction, Op, PtxType, Register

from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass



@register_pass
class FuseMultiplyAdd(FunctionPass):
    """add(mul(a, b), c) -> one multiply-add, when the mul result has no other reader.

        mul.lo.s32 t, a, b;   add.s32 d, t, c   ->  mad.lo.s32 d, a, b, c
        mul.wide.s32 t, a, k; add.s64 d, t, c   ->  mad.wide.s32 d, a, k, c   (address math)
        mul.f32 t, a, b;      add.f32 d, t, c   ->  fma.rn.f32 d, a, b, c

    Integer fusion is exact. For f32 it is the contraction ptxas applies anyway to a mul/add
    pair without a rounding modifier (an explicit .rn on either forbids it, so those stay).
    Runs late: addr-fold and vectorize-mem pattern-match the separate mul.wide/add.
    """

    name = "fuse-mad"
    preserves = CFG_ANALYSES

    def run(self, function, am):
        du = am.get("defuse")
        changed = False
        for block in function.blocks:
            for k, add in enumerate(block.instructions):
                if add.op is not Op.ADD or add.guard is not None:
                    continue
                new = self.fuse(add, du)
                if new is not None:
                    new.comment = add.comment
                    block.instructions[k] = new
                    changed = True
        return changed

    def fuse(self, add, du):
        for t_idx in (0, 1):
            t, c = add.srcs[t_idx], add.srcs[1 - t_idx]
            if not isinstance(t, Register):
                continue
            mul = du.single_def(t)
            if mul is None or mul.op is not Op.MUL or len(du.uses_of(t)) != 1:
                continue
            if not all(du.is_ssa(r) for r in mul.uses()):
                continue                   # a or b could change between the mul and the add
            a, b = mul.srcs
            at = add.ptx_type
            if at.is_float:
                if mul.types != (at,) or add.types != (at,):
                    continue
                return Instruction(Op.FMA, ("rn", at), [add.dsts[0]], [a, b, c])
            mt = mul.ptx_type
            if mul.types == ("lo", mt) and mt == at:
                return Instruction(Op.MAD, ("lo", at), [add.dsts[0]], [a, b, c])
            if mul.types == ("wide", mt) and at in (PtxType.S64, PtxType.U64) and mt in (PtxType.S32, PtxType.U32):
                return Instruction(Op.MAD, ("wide", mt), [add.dsts[0]], [a, b, c])
        return None


