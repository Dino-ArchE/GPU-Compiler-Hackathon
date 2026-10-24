"""Dominator tree (Cooper, Harvey & Kennedy, "A Simple, Fast Dominance Algorithm").

Block a dominates b if every path from the entry to b goes through a. Only reachable blocks
appear; unreachable ones have no idom.
"""

from ..pass_manager import Analysis, register_analysis


class DominatorTree:
    def __init__(self, cfg):
        self.cfg = cfg
        order = cfg.rpo()
        index = {b: i for i, b in enumerate(order)}
        idom = {cfg.entry: cfg.entry}

        def intersect(a, b):
            while a is not b:
                while index[a] > index[b]:
                    a = idom[a]
                while index[b] > index[a]:
                    b = idom[b]
            return a

        changed = True
        while changed:
            changed = False
            for block in order[1:]:
                done = [p for p in cfg.preds[block] if p in idom]
                new = done[0]
                for p in done[1:]:
                    new = intersect(p, new)
                if idom.get(block) is not new:
                    idom[block] = new
                    changed = True

        self.order = order
        self.idom = {b: (None if b is cfg.entry else d) for b, d in idom.items()}
        self.children = {b: [] for b in order}
        for b in order:
            if self.idom[b] is not None:
                self.children[self.idom[b]].append(b)

    def dominates(self, a, b):
        """True if a dominates b (every block dominates itself)."""
        while b is not None:
            if b is a:
                return True
            b = self.idom.get(b)
        return False

    def preorder(self):
        """Dominator-tree preorder: each block after its dominators (for scoped walks like CSE)."""
        out, stack = [], [self.cfg.entry]
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(reversed(self.children[b]))
        return out


@register_analysis
class DominatorAnalysis(Analysis):
    name = "dominators"

    def run(self, function, am):
        return DominatorTree(am.get("cfg"))
