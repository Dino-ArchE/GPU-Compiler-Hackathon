from .reg import Register
from .value import Address, Vector


class Instruction:
    """One PTX instruction as data: `@guard op.types dsts, srcs;`.

    `op` and `types` are plain strings ("ld", ("global", "v4", "f32")) so passes can match on them
    without a class per opcode. Stores and branches have no dsts.
    """

    def __init__(self, op, types=(), dsts=(), srcs=(), guard=None, negate=False, comment=None):
        self.op = op
        self.types = tuple(types)
        self.dsts = tuple(dsts)
        self.srcs = tuple(srcs)
        self.guard = guard        # predicate Register: the instruction runs only where it holds
        self.negate = negate      # @!%p instead of @%p
        self.comment = comment

    def defs(self):
        """Registers this instruction writes."""
        return [r for v in self.dsts for r in _registers(v)]

    def uses(self):
        """Registers this instruction reads, including the guard and address bases."""
        regs = [r for v in self.srcs for r in _registers(v)]
        return regs + [self.guard] if self.guard is not None else regs

    def __repr__(self):
        return f"Instruction({'.'.join((self.op, *self.types))} {self.dsts} <- {self.srcs})"


def _registers(value):
    if isinstance(value, Register):
        return [value]
    if isinstance(value, Address):
        return _registers(value.base)
    if isinstance(value, Vector):
        return [r for e in value.elems for r in _registers(e)]
    return []
