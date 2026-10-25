from ..analysis.liveness import kills, reads
from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass


@register_pass
class DeadCodeElimination(FunctionPass):
    """Delete instructions whose results are never read and that have no side effects.

    Liveness-based, so it is exact for non-SSA registers: an accumulator update stays as long as
    some later read (on any path, including around a loop) can see it. Each block is walked
    backward keeping the live set current, so a chain of dead instructions dies in one sweep;
    the outer loop repeats while removals make values in other blocks dead.
    """

    name = "dce"
    preserves = CFG_ANALYSES

    def run(self, function, am):
        changed = False
        while True:
            live_out = am.get("liveness").live_out
            removed = False
            for block in function.blocks:
                live = set(live_out[block])
                keep = []
                for inst in reversed(block.instructions):
                    defs = inst.defs()
                    if defs and not inst.has_side_effects and not any(r in live for r in defs):
                        removed = True
                        continue
                    live.difference_update(kills(inst))
                    live.update(reads(inst))
                    keep.append(inst)
                block.instructions = keep[::-1]
            if not removed:
                return changed
            changed = True
            am.invalidate(self.preserves)
