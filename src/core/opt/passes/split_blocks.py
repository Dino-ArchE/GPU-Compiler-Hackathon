from ..pass_manager import FunctionPass, register_pass


@register_pass
class SplitBlocks(FunctionPass):
    """Canonicalize: end a block right after every bra/ret, so each block has a single exit.

    Codegen emits `@p bra EXIT` in the middle of a block (the thread guard); the instructions
    after it move to a new block that the old one falls through to. The CFG requires this form.
    """

    name = "split-blocks"

    def run(self, function, am):
        changed = False
        i = 0
        while i < len(function.blocks):
            block = function.blocks[i]
            cut = next((k for k, inst in enumerate(block.instructions[:-1]) if inst.is_terminator), None)
            if cut is not None:
                tail = function.new_block("BB")
                tail.instructions = block.instructions[cut + 1:]
                block.instructions = block.instructions[:cut + 1]
                function.blocks.insert(i + 1, tail)
                changed = True
            i += 1
        return changed
