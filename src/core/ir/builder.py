from .instruction import Instruction


class Builder:
    """IRBuilder: appends instructions at an insertion point inside one Function.

    The visitor decides what to build; the builder only creates registers, blocks and
    instructions. It knows nothing about the Kernel IR or about PTX text.
    """

    def __init__(self, function):
        self.function = function
        self.block = None
        self.set_block(function.new_block("ENTRY"))

    def reg(self, dtype):
        return self.function.regs.new(dtype)

    def new_block(self, hint="BB"):
        return self.function.new_block(hint)

    def set_block(self, block):
        """Move the insertion point to `block`, placing it next in the layout on first visit."""
        if block not in self.function.blocks:
            self.function.blocks.append(block)
        self.block = block

    def inst(self, op, *types, dsts=(), srcs=(), guard=None, negate=False, comment=None):
        """Append `op.types dsts, srcs`, e.g. inst("add", "f32", dsts=[d], srcs=[a, b])."""
        return self.block.append(Instruction(op, types, dsts, srcs, guard, negate, comment))
