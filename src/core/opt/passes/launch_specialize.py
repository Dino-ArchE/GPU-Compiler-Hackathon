from core.ir import NTID_X, Immediate, Op

from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass


@register_pass
class LaunchSpecialize(FunctionPass):
    """Replace %ntid.x with the block size the kernel is launched with.

    meta.json fixes the launch, so `mov %r1, %ntid.x` is really `mov %r1, 256`; after
    copy-prop the thread index becomes `mad.lo.s32 gid, %ctaid.x, 256, %tid.x` and later
    passes see exact multiples (alignment) and bounds. The tid/ctaid bounds themselves are
    used by the `ranges` analysis directly.
    """

    name = "launch-specialize"
    preserves = CFG_ANALYSES

    def run(self, function, am):
        if function.launch_block is None:
            return False
        changed = False
        for _, inst in function.instructions():
            if inst.op is Op.MOV and inst.srcs[0] is NTID_X:
                inst.srcs = (Immediate(function.launch_block[0], inst.ptx_type),)
                changed = True
        return changed
