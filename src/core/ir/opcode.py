from enum import StrEnum


class Op(StrEnum):
    """PTX opcodes, the first dotted part of an instruction: `ld` in ld.global.f32.

    Modifiers after it (state space, type, rounding, comparison) are separate.
    """

    # data movement
    LD = "ld"
    ST = "st"
    MOV = "mov"
    CVTA = "cvta"
    CVT = "cvt"
    CP = "cp"            # cp.async.* copies into shared memory
    # integer and float arithmetic
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    MAD = "mad"
    FMA = "fma"
    DIV = "div"
    REM = "rem"
    NEG = "neg"
    ABS = "abs"
    MIN = "min"
    MAX = "max"
    # float special functions
    RCP = "rcp"
    SQRT = "sqrt"
    RSQRT = "rsqrt"
    EX2 = "ex2"
    LG2 = "lg2"
    # bitwise
    AND = "and"
    OR = "or"
    XOR = "xor"
    NOT = "not"
    SHL = "shl"
    SHR = "shr"
    # compare and select
    SETP = "setp"
    SELP = "selp"
    # control flow and synchronization
    BRA = "bra"
    RET = "ret"
    BAR = "bar"
    SHFL = "shfl"
    ACTIVEMASK = "activemask"
    # atomics
    RED = "red"
    ATOM = "atom"


class Cmp(StrEnum):
    """setp comparison operators: setp.<cmp>.<type>. Signedness/float-ness come from the type."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
