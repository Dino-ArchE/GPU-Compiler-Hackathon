from .type_enum import PtxType
from .value import Value


class Register(Value):
    """A PTX register operand. str() is its use form, e.g. %f3 or %tid.x."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"%{self.name}"

    __repr__ = __str__


class VirtualRegister(Register):
    """An allocated register: unlimited in PTX, mapped to hardware registers by ptxas."""

    PREFIX = None      # name stem, e.g. "f" -> %f0, %f1, ...
    PTX_TYPE = None    # declared type in `.reg .<PTX_TYPE> %<PREFIX><N>;`

    def __init__(self, idx):
        super().__init__(f"{self.PREFIX}{idx}")
        self.idx = idx


class PRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "p", PtxType.PRED     # setp results, @%p guards


class RSRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "rs", PtxType.B16     # 16-bit integers


class HRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "h", PtxType.F16      # half floats


class RRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "r", PtxType.B32      # 32-bit ints; .b32 serves both .s32 and .u32 ops


class RDRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "rd", PtxType.B64     # 64-bit ints and addresses


class FRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "f", PtxType.F32


class FDRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "fd", PtxType.F64


class SpecialRegister(Register):
    """A built-in read-only register (%tid.x, %ctaid.x, ...). Never allocated or declared."""


TID_X = SpecialRegister("tid.x")
NTID_X = SpecialRegister("ntid.x")
CTAID_X = SpecialRegister("ctaid.x")
LANEID = SpecialRegister("laneid")


# value type -> register class; signedness lives on instructions, not registers
TYPE_TO_CLASS = {
    PtxType.PRED: PRegister,
    PtxType.B16: RSRegister, PtxType.S16: RSRegister, PtxType.U16: RSRegister,
    PtxType.F16: HRegister,
    PtxType.B32: RRegister, PtxType.S32: RRegister, PtxType.U32: RRegister,
    PtxType.F32: FRegister,
    PtxType.B64: RDRegister, PtxType.S64: RDRegister, PtxType.U64: RDRegister,
    PtxType.F64: FDRegister,
}


class RegisterFile:
    """Hands out fresh virtual registers and renders their `.reg` declarations.
    Serves as a register factory"""
    def __init__(self):
        self.counts = {}   # register class -> number allocated so far

    def new(self, ptx_type):
        # PtxType(...) also accepts an exact spelling like "f32" but rejects typos and "i32"
        cls = TYPE_TO_CLASS[PtxType(ptx_type)]
        idx = self.counts.get(cls, 0)
        self.counts[cls] = idx + 1
        return cls(idx)

    def declarations(self):
        # `%f<N>` declares %f0 .. %f(N-1)
        return [f".reg .{cls.PTX_TYPE} %{cls.PREFIX}<{n}>;" for cls, n in self.counts.items()]
