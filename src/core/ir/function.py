from .block import BasicBlock
from .reg import RegisterFile
from .value import Param


class Function:
    """One kernel entry: params, its registers, and its blocks in layout order.

    `launch_grid`/`launch_block` are the fixed launch geometry from meta.json and `readonly` the
    params the kernel only reads; codegen fills them so passes can specialize on them.
    """

    def __init__(self, name, params):
        self.name = name
        self.params = {p: Param(p) for p in params}
        self.regs = RegisterFile()
        self.blocks = []          # layout order: fall-through goes to the next block
        self._block_count = 0     # one counter for all hints, so block names never collide
        self.launch_grid = None
        self.launch_block = None
        self.readonly = set()     # Param symbols never written by this kernel

    def param(self, name):
        return self.params[name]

    def new_block(self, hint="BB"):
        """Create a uniquely named block; it joins the layout when the builder moves into it."""
        block = BasicBlock(f"{hint}_{self._block_count}")
        self._block_count += 1
        return block

    def instructions(self):
        """Every (block, instruction) in layout order. Copy first if you mutate while iterating."""
        for block in self.blocks:
            for inst in block.instructions:
                yield block, inst

    def num_instructions(self):
        return sum(len(b.instructions) for b in self.blocks)

    def replace_all_uses(self, old, new):
        """Replace every read of register `old` with `new` (see Instruction.replace_use)."""
        return sum(inst.replace_use(old, new) for _, inst in self.instructions())

    def emit(self):
        param_str = ",".join([str(param) for param in self.params])
        return f".visible .entry {self.name}({param_str})"
