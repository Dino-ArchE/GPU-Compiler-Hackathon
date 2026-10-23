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
    PTX_TYPE = None    # declared type in `.reg <PTX_TYPE> %<PREFIX><N>;`

    def __init__(self, idx):
        super().__init__(f"{self.PREFIX}{idx}")
        self.idx = idx


class PRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "p", ".pred"      # setp results, @%p guards


class RSRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "rs", ".b16"      # 16-bit integers


class HRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "h", ".f16"       # half floats


class RRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "r", ".b32"       # 32-bit ints; .b32 serves both .s32 and .u32 ops


class RDRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "rd", ".b64"      # 64-bit ints and addresses


class FRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "f", ".f32"


class FDRegister(VirtualRegister):
    PREFIX, PTX_TYPE = "fd", ".f64"


class SpecialRegister(Register):
    """A built-in read-only register (%tid.x, %ctaid.x, ...). Never allocated or declared."""


TID_X = SpecialRegister("tid.x")
NTID_X = SpecialRegister("ntid.x")
CTAID_X = SpecialRegister("ctaid.x")
LANEID = SpecialRegister("laneid")


# value dtype (IR / meta.json spelling, or a PTX type suffix) -> register class
DTYPE_TO_CLASS = {
    "pred": PRegister,
    "i16": RSRegister, "s16": RSRegister, "u16": RSRegister, "b16": RSRegister,
    "f16": HRegister,
    "i32": RRegister, "s32": RRegister, "u32": RRegister, "b32": RRegister,
    "i64": RDRegister, "s64": RDRegister, "u64": RDRegister, "b64": RDRegister,
    "f32": FRegister,
    "f64": FDRegister,
}


class RegisterFile:
    """Hands out fresh virtual registers and renders their `.reg` declarations.
    Serves as a register factory"""
    def __init__(self):
        self.counts = {}   # register class -> number allocated so far

    def new(self, dtype):
        cls = DTYPE_TO_CLASS[dtype]
        idx = self.counts.get(cls, 0)
        self.counts[cls] = idx + 1
        return cls(idx)

    def declarations(self):
        # `%f<N>` declares %f0 .. %f(N-1)
        return [f".reg {cls.PTX_TYPE} %{cls.PREFIX}<{n}>;" for cls, n in self.counts.items()]
