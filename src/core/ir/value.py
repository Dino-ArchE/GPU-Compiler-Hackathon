import struct

from .type_enum import PtxType


class Value:
    """Anything an instruction can take as an operand. str() is its PTX spelling.

    Unlike LLVM, an Instruction is not a Value: PTX registers are not SSA (accumulators are
    updated in place), so results live in Registers that instructions write.
    """

    def __str__(self):
        raise NotImplementedError(type(self).__name__)

    def __repr__(self):
        return str(self)


class Immediate(Value):
    """A literal operand; floats print as PTX hex bit patterns (1.0 -> 0f3F800000)."""

    def __init__(self, value, ptx_type):
        self.value = value
        self.ptx_type = PtxType(ptx_type)

    def __str__(self):
        if self.ptx_type is PtxType.F32:
            return "0f%08X" % struct.unpack("<I", struct.pack("<f", self.value))[0]
        if self.ptx_type is PtxType.F64:
            return "0d%016X" % struct.unpack("<Q", struct.pack("<d", self.value))[0]
        return str(int(self.value))


class Symbol(Value):
    """A named address: a kernel param (param_x) or a shared array (acc0_smem)."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class Param(Symbol):
    """A kernel parameter: a slot in .param space, read with ld.param.

    The prefix keeps buffer names like `add` or `max` off PTX keywords.
    """

    def __init__(self, name, ptx_type=PtxType.U64):
        super().__init__(f"param_{name}")
        self.ptx_type = PtxType(ptx_type)   # u64 for buffer pointers


class Address(Value):
    """A memory operand [base+offset]; base is a Register or a Symbol, offset in bytes."""

    def __init__(self, base, offset=0):
        self.base = base
        self.offset = offset

    def __str__(self):
        return f"[{self.base}+{self.offset}]" if self.offset else f"[{self.base}]"


class Vector(Value):
    """A register group {%f0, %f1, ...} for v2/v4 loads and stores."""

    def __init__(self, elems):
        self.elems = tuple(elems)

    def __str__(self):
        return "{" + ", ".join(str(e) for e in self.elems) + "}"
