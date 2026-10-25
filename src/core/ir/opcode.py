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
    SIN = "sin"
    COS = "cos"
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
    # tensor cores
    MMA = "mma"
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


class Mod(StrEnum):
    """Instruction modifiers other than types, state spaces and comparisons: the `rn` in
    div.rn.f32, the `lo` in mul.lo.s32. Passes test for them (`Mod.HI in inst.types`), so a
    misspelled one must fail at import rather than silently never match."""

    # float rounding: to nearest even, toward zero, toward -inf, toward +inf
    RN = "rn"
    RZ = "rz"
    RM = "rm"
    RP = "rp"
    # cvt rounding to an integral value (same four directions); rna = nearest, ties away (tf32)
    RNI = "rni"
    RZI = "rzi"
    RMI = "rmi"
    RPI = "rpi"
    RNA = "rna"
    # which part of an integer product: low half, high half, full double-width result
    LO = "lo"
    HI = "hi"
    WIDE = "wide"
    # float precision: fast approximation, fast non-IEEE divide, flush denormals, clamp to [0, 1]
    APPROX = "approx"
    FULL = "full"
    FTZ = "ftz"
    SAT = "sat"
    # memory: non-coherent (read-only) load, cp.async cache level, vector width, cvta direction
    NC = "nc"
    CA = "ca"
    CG = "cg"
    V2 = "v2"
    V4 = "v4"
    TO = "to"
    # cp.async groups
    ASYNC = "async"
    COMMIT_GROUP = "commit_group"
    WAIT_GROUP = "wait_group"
    # synchronization and warp exchange (shfl modes: bfly = lane ^ b, down, up, idx)
    SYNC = "sync"
    WARP = "warp"
    ALIGNED = "aligned"
    BFLY = "bfly"
    DOWN = "down"
    UP = "up"
    IDX = "idx"
    # red/atom reduction operator
    ADD = "add"
    # mma: tile shape, A/B layouts, and tf32 (an operand format, not a register type)
    M16N8K8 = "m16n8k8"
    ROW = "row"
    COL = "col"
    TF32 = "tf32"

    @classmethod
    def vec(cls, n):
        """The .v2/.v4 modifier for an n-element vector access."""
        return {2: cls.V2, 4: cls.V4}[n]
