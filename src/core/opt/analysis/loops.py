"""Natural loops: one per header, found from back edges (latch -> header, header dominates latch)."""

from ..pass_manager import Analysis, register_analysis


class Loop:
    def __init__(self, header):
        self.header = header
        self.blocks = {header}
        self.latches = []
        self.parent = None
        self.children = []

    @property
    def depth(self):
        d, loop = 1, self.parent
        while loop is not None:
            d, loop = d + 1, loop.parent
        return d

    def exits(self, cfg):
        """Blocks outside the loop that a loop block can branch or fall to."""
        return [s for b in self.blocks for s in cfg.succs[b] if s not in self.blocks]

    def preheader(self, cfg):
        """The unique outside predecessor of the header whose only successor is the header."""
        outside = [p for p in cfg.preds[self.header] if p not in self.blocks]
        if len(outside) == 1 and cfg.succs[outside[0]] == [self.header]:
            return outside[0]
        return None

    def __repr__(self):
        return f"Loop({self.header.name}, {sorted(b.name for b in self.blocks)})"


class LoopInfo:
    def __init__(self, cfg, dom):
        self.loops = {}                     # header -> Loop
        for block in dom.order:
            for succ in cfg.succs[block]:
                if dom.dominates(succ, block):          # back edge block -> succ
                    loop = self.loops.setdefault(succ, Loop(succ))
                    loop.latches.append(block)
                    stack = [block]
                    while stack:
                        b = stack.pop()
                        if b not in loop.blocks:
                            loop.blocks.add(b)
                            stack.extend(p for p in cfg.preds[b] if p in dom.idom)
        # nesting: a loop's parent is the smallest other loop containing its header
        for loop in self.loops.values():
            outer = [o for o in self.loops.values() if o is not loop and loop.header in o.blocks]
            if outer:
                loop.parent = min(outer, key=lambda o: len(o.blocks))
                loop.parent.children.append(loop)

    def innermost(self, block):
        """The deepest loop containing `block`, or None."""
        best = None
        for loop in self.loops.values():
            if block in loop.blocks and (best is None or len(loop.blocks) < len(best.blocks)):
                best = loop
        return best

    def depth(self, block):
        loop = self.innermost(block)
        return loop.depth if loop else 0

    def __iter__(self):
        """Loops innermost first, so a pass hoists out of inner loops before outer ones."""
        return iter(sorted(self.loops.values(), key=lambda l: -l.depth))


@register_analysis
class LoopAnalysis(Analysis):
    name = "loops"

    def run(self, function, am):
        return LoopInfo(am.get("cfg"), am.get("dominators"))
