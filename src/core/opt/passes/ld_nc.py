from core.ir import Op, StateSpace

from ..pass_manager import PRESERVE_ALL, FunctionPass, register_pass


@register_pass
class LoadNonCoherent(FunctionPass):
    """ld.global -> ld.global.nc for buffers the kernel never writes.

    .nc loads go through the read-only (texture) data cache, which has its own bandwidth and
    tolerates scattered access better. Only legal when the data cannot change during the
    kernel: the `memory` analysis proves the buffer is a read-only param nothing stores to.
    """

    name = "ld-nc"
    preserves = PRESERVE_ALL       # a modifier change: no analysis looks at .nc

    def run(self, function, am):
        mem = am.get("memory")
        changed = False
        for _, inst in function.instructions():
            if inst.op is Op.LD and inst.space is StateSpace.GLOBAL and "nc" not in inst.types \
                    and mem.is_invariant_load(inst):
                k = inst.types.index(StateSpace.GLOBAL)
                inst.types = (*inst.types[:k + 1], "nc", *inst.types[k + 1:])
                changed = True
        return changed
