from enum import StrEnum


class PtxType(StrEnum):
    """PTX scalar types, spelled as in instruction suffixes: add.f32, ld.param.u64, setp.lt.s32.

    A StrEnum member *is* a str, so it drops into f-strings and joins unchanged, while a typo
    (PtxType.F23) fails at import time and PtxType("f23") raises instead of emitting bad PTX.
    """

    PRED = "pred"
    B16 = "b16"
    S16 = "s16"
    U16 = "u16"
    F16 = "f16"
    B32 = "b32"
    S32 = "s32"
    U32 = "u32"
    F32 = "f32"
    B64 = "b64"
    S64 = "s64"
    U64 = "u64"
    F64 = "f64"

    @property
    def nbytes(self):
        if self is PtxType.PRED:
            raise ValueError("pred has no memory size")
        return int(self.value[1:]) // 8

    @property
    def is_float(self):
        return self.value.startswith("f")

    @classmethod
    def from_meta(cls, dtype):
        """meta.json / runner dtype -> PtxType. Their ints are i32/i64; PTX spells signed ints s32/s64."""
        return cls("s" + dtype[1:] if dtype.startswith("i") else dtype)


class StateSpace(StrEnum):
    """Where a memory instruction reads or writes: ld.param, ld.global, st.shared, ..."""

    PARAM = "param"
    GLOBAL = "global"
    SHARED = "shared"
