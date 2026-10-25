"""Register liveness (backward dataflow over the CFG).

live_in(B)  = uses(B) | (live_out(B) - kills(B))
live_out(B) = union of live_in(S) over successors S

An unconditional write kills the register; a guarded one (`@p add %r1, ...`) does not, since
where the guard fails the old value survives. Special registers are never tracked.
"""

from core.ir import SpecialRegister

from ..pass_manager import Analysis, register_analysis


def kills(inst):
    return [] if inst.guard is not None else inst.defs()


def reads(inst):
    return [r for r in inst.uses() if not isinstance(r, SpecialRegister)]


class Liveness:
    def __init__(self, function, cfg):
        self.cfg = cfg
        gen, kill = {}, {}
        for block in function.blocks:
            g, k = set(), set()
            for inst in reversed(block.instructions):
                k.update(kills(inst))
                g.difference_update(kills(inst))
                g.update(reads(inst))
            gen[block], kill[block] = g, k

        self.live_in = {b: set() for b in function.blocks}
        self.live_out = {b: set() for b in function.blocks}
        order = list(reversed(cfg.rpo()))
        changed = True
        while changed:
            changed = False
            for block in order:
                out = set().union(*(self.live_in[s] for s in cfg.succs[block]))
                new_in = gen[block] | (out - kill[block])
                if new_in != self.live_in[block] or out != self.live_out[block]:
                    self.live_in[block], self.live_out[block] = new_in, out
                    changed = True

    def walk_backward(self, block):
        """Yield (inst, registers live right after inst), last instruction first."""
        live = set(self.live_out[block])
        for inst in reversed(block.instructions):
            yield inst, set(live)
            live.difference_update(kills(inst))
            live.update(reads(inst))


@register_analysis
class LivenessAnalysis(Analysis):
    name = "liveness"

    def run(self, function, am):
        return Liveness(function, am.get("cfg"))
