"""Control-flow graph: successors/predecessors between basic blocks.

Requires the canonical form made by the `split-blocks` pass: a bra or ret may only be a block's
last instruction. Edges: the branch target, plus fall-through to the next block in layout order
unless the block ends in an unconditional bra/ret.
"""

from core.ir import Op

from ..pass_manager import Analysis, register_analysis


class CFG:
    def __init__(self, function):
        self.function = function
        self.entry = function.blocks[0]
        self.succs = {b: [] for b in function.blocks}
        self.preds = {b: [] for b in function.blocks}
        layout = function.blocks
        for i, block in enumerate(layout):
            for inst in block.instructions[:-1]:
                if inst.is_terminator:
                    raise ValueError(f"{block.name}: {inst!r} is not last; run split-blocks first")
            term = block.terminator
            if term is not None and term.op is Op.BRA:
                self._edge(block, term.target)
            falls_through = term is None or term.guard is not None
            if falls_through and i + 1 < len(layout):
                self._edge(block, layout[i + 1])

    def _edge(self, a, b):
        if b not in self.succs[a]:
            self.succs[a].append(b)
            self.preds[b].append(a)

    def falls_through(self, block):
        term = block.terminator
        return term is None or term.guard is not None

    def reachable(self):
        seen, stack = {self.entry}, [self.entry]
        while stack:
            for s in self.succs[stack.pop()]:
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        return seen

    def rpo(self):
        """Reachable blocks in reverse post-order: every block before its successors, loops aside."""
        order, seen = [], set()

        def dfs(block):
            seen.add(block)
            for s in self.succs[block]:
                if s not in seen:
                    dfs(s)
            order.append(block)

        dfs(self.entry)
        return order[::-1]


@register_analysis
class CFGAnalysis(Analysis):
    name = "cfg"

    def run(self, function, am):
        return CFG(function)
