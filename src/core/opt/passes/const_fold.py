from core.ir import Immediate, Mov, Op, Vector

from ..fold import compare, evaluate, result_type
from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass


@register_pass
class ConstantFold(FunctionPass):
    """Evaluate instructions whose sources are all immediates, and guards on known predicates.

    `add.s32 %r5, 12, 1` becomes `mov.s32 %r5, 13` (copy-prop then pushes 13 into the users).
    A single-def `setp` on two immediates is a known predicate: every instruction it guards is
    then always executed (guard dropped) or never (deleted), and `selp` on it becomes a mov.
    The setp itself is left for DCE. Resolving a guarded `bra` changes the CFG, so the CFG
    analyses are kept only if no branch changed.
    """

    name = "const-fold"

    def run(self, function, am):
        du = am.get("defuse")
        known = {}                                   # predicate register -> bool
        for _, inst in function.instructions():
            if inst.op is Op.SETP and du.single_def(inst.dsts[0]) is inst and _all_imm(inst):
                res = compare(inst.types[0], inst.ptx_type, *(v.value for v in inst.srcs))
                if res is not None:
                    known[inst.dsts[0]] = res

        changed = branch_changed = False
        for block in function.blocks:
            out = []
            for inst in block.instructions:
                if inst.guard is not None and inst.guard in known:
                    changed = True
                    branch_changed |= inst.is_terminator
                    if known[inst.guard] == inst.negate:     # the guard never holds
                        continue
                    inst.guard, inst.negate = None, False
                folded = self.fold(inst, known)
                if folded is not None:
                    changed = True
                    inst = folded
                out.append(inst)
            block.instructions = out
        self.preserves = frozenset() if branch_changed else CFG_ANALYSES
        return changed

    def fold(self, inst, known):
        """The mov that replaces `inst`, or None."""
        if inst.has_side_effects or len(inst.dsts) != 1 or isinstance(inst.dsts[0], Vector):
            return None
        if inst.op is Op.SELP and inst.srcs[2] in known:
            return _mov(inst, inst.ptx_type, inst.srcs[0] if known[inst.srcs[2]] else inst.srcs[1])
        if inst.op in (Op.MOV, Op.SETP) or not _all_imm(inst):
            return None
        r = evaluate(inst, [v.value for v in inst.srcs])
        if r is None:
            return None
        t = result_type(inst)
        return _mov(inst, t, Immediate(r, t))


def _all_imm(inst):
    return bool(inst.srcs) and all(isinstance(v, Immediate) for v in inst.srcs)


def _mov(inst, t, value):
    return Mov(t, inst.dsts[0], value, guard=inst.guard, negate=inst.negate, comment=inst.comment)
