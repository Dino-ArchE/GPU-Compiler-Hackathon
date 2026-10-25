from .value import Value


class BasicBlock(Value):
    """A labelled run of instructions; as an operand it is the branch target (`bra LOOP_0`)."""

    def __init__(self, name):
        self.name = name
        self.instructions = []

    def append(self, inst):
        self.instructions.append(inst)
        return inst

    # instructions have no __eq__, so list lookups below compare by identity

    def index(self, inst):
        return self.instructions.index(inst)

    def insert_before(self, anchor, inst):
        self.instructions.insert(self.index(anchor), inst)
        return inst

    def insert_after(self, anchor, inst):
        self.instructions.insert(self.index(anchor) + 1, inst)
        return inst

    def remove(self, inst):
        self.instructions.remove(inst)

    @property
    def terminator(self):
        """The trailing bra/ret, or None if the block falls through."""
        if self.instructions and self.instructions[-1].is_terminator:
            return self.instructions[-1]
        return None

    def __str__(self):
        return self.name
