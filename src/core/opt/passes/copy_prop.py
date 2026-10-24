from core.ir import Immediate, Op, Register, SpecialRegister

from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass


@register_pass
class CopyPropagation(FunctionPass):
    """Replace reads of `d` with `s` after `mov d, s`, leaving the mov dead for DCE.

    Safe without SSA when `d` is written only by this (unguarded) mov and `s` holds one value
    for the whole kernel: an immediate, a special register, or a single-def register. Then
    every read of d sees exactly s. Accumulators (several defs) are never touched.

    PTX limits where the replacement may go: special registers can only be read by mov, a
    float literal only by a float op (and vice versa), and an address base must be a register.
    """

    name = "copy-prop"
    preserves = CFG_ANALYSES

    def run(self, function, am):
        du = am.get("defuse")
        changed = False
        for _, mov in list(function.instructions()):
            if mov.op is not Op.MOV or mov.guard is not None:
                continue
            d, s = mov.dsts[0], mov.srcs[0]
            if du.single_def(d) is not mov or s is d:
                continue
            if isinstance(s, Register) and not (du.is_ssa(s) and type(s) is type(d)) \
                    and not isinstance(s, SpecialRegister):
                continue
            if not isinstance(s, (Register, Immediate)):
                continue                       # symbols (shared array addresses) stay in movs
            for user in du.uses_of(d):
                if user is mov or not _legal(user, s):
                    continue
                changed |= user.replace_use(d, s) > 0
        return changed


def _legal(user, s):
    if isinstance(s, SpecialRegister):
        return user.op is Op.MOV
    if isinstance(s, Immediate):
        t = user.ptx_type
        return t is None or t.is_float == s.ptx_type.is_float
    return True
