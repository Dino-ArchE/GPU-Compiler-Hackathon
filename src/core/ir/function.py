from .block import BasicBlock
from .reg import RegisterFile
from .value import Symbol


class Function:
    """One kernel entry: params, shared arrays, its registers, and its blocks in layout order."""

    def __init__(self, name, params):
        self.name = name
        # buffers are passed as .u64 pointers; the prefix keeps names like `add` off PTX keywords
        self.params = {p: Symbol(f"param_{p}") for p in params}
        self.shared = []          # (Symbol, align, nbytes)
        self.regs = RegisterFile()
        self.blocks = []          # layout order: fall-through goes to the next block
        self._block_count = 0     # one counter for all hints, so block names never collide

    def param(self, name):
        return self.params[name]

    def add_shared(self, name, align, nbytes):
        sym = Symbol(name)
        self.shared.append((sym, align, nbytes))
        return sym

    def new_block(self, hint="BB"):
        """Create a uniquely named block; it joins the layout when the builder moves into it."""
        block = BasicBlock(f"{hint}_{self._block_count}")
        self._block_count += 1
        return block
