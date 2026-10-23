from .value import Value


class BasicBlock(Value):
    """A labelled run of instructions; as an operand it is the branch target (`bra LOOP_0`)."""

    def __init__(self, name):
        self.name = name
        self.instructions = []

    def append(self, inst):
        self.instructions.append(inst)
        return inst

    def __str__(self):
        return self.name
