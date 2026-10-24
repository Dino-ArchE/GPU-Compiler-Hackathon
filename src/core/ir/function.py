from .block import BasicBlock
from .reg import RegisterFile
from .value import Param


class Function:
    """One kernel entry: params, its registers, and its blocks in layout order."""

    def __init__(self, name, params):
        self.name = name
        self.params = {p: Param(p) for p in params}
        self.regs = RegisterFile()
        self.blocks = []          # layout order: fall-through goes to the next block
        self._block_count = 0     # one counter for all hints, so block names never collide

    def param(self, name):
        return self.params[name]

    def new_block(self, hint="BB"):
        """Create a uniquely named block; it joins the layout when the builder moves into it."""
        block = BasicBlock(f"{hint}_{self._block_count}")
        self._block_count += 1
        return block

    def emit(self):
        param_str = ",".join([str(param) for param in self.params])
        return f".visible .entry {self.name}({param_str})"