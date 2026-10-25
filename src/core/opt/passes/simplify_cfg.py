from core.ir import Op

from ..analysis.cfg import CFG
from ..pass_manager import FunctionPass, register_pass


@register_pass
class SimplifyCFG(FunctionPass):
    """Tidy the block graph, repeating until nothing changes:

    - delete blocks no path from the entry reaches
    - delete a bra (guarded or not) to the very next block: falling through does the same
    - thread jumps: `bra T` where T is just `bra T2` becomes `bra T2`
    - delete empty blocks, pointing their branches at the block they fall into
    - merge B into A when A falls through to B and nothing else reaches B

    Fewer blocks means longer straight-line regions for the per-block passes (vectorize-mem)
    and fewer labels in the output. Layout order is kept: removing or merging a block never
    changes where any remaining block falls through to.
    """

    name = "simplify-cfg"

    def run(self, function, am):
        changed = False
        for _ in range(10_000):            # each step shrinks the graph; the cap guards cycles of jumps
            if not self.step(function):
                break
            changed = True
        return changed

    def step(self, function):
        blocks = function.blocks
        cfg = CFG(function)
        live = cfg.reachable()
        if len(live) != len(blocks):
            function.blocks = [b for b in blocks if b in live]
            return True

        for i, block in enumerate(blocks):
            term = block.terminator
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if term is not None and term.op is Op.BRA:
                if term.target is nxt:
                    block.instructions.pop()
                    return True
                target = term.target
                only = target.instructions[0] if len(target.instructions) == 1 else None
                if only is not None and only.op is Op.BRA and only.guard is None and only.target is not target:
                    term.srcs = (only.target,)
                    return True

        for i, block in enumerate(blocks[:-1]):
            if not block.instructions:
                self.retarget(function, block, blocks[i + 1])
                del blocks[i]
                return True

        for i, block in enumerate(blocks[:-1]):
            nxt = blocks[i + 1]
            if block.terminator is None and cfg.preds[nxt] == [block]:
                block.instructions.extend(nxt.instructions)
                del blocks[i + 1]
                return True
        return False

    @staticmethod
    def retarget(function, old, new):
        for _, inst in function.instructions():
            if inst.op is Op.BRA and inst.target is old:
                inst.srcs = (new,)
