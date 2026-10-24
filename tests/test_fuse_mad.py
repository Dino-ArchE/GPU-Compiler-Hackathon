import numpy as np

from core.ir import CTAID_X, NTID_X, TID_X, Address, Immediate, Op, PtxType, StateSpace

from core.opt import Repeat
from helpers import differential, finish, new_function, ops

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32
PIPE = ["split-blocks", "fuse-mad", Repeat(["dce"])]


def build(reuse_mul=False, rn=False):
    """out[g] = x[g] * 3 + 1 (float) and the address math via mul.wide + add."""
    def b_():
        f, b = new_function(grid=(2, 1, 1), block=(32, 1, 1))
        x = b.cvta_to_global(b.load_param(f.param("x")))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        gid = b.add(S, b.mul(S, b.mov(U, CTAID_X), b.mov(U, NTID_X)), b.mov(U, TID_X))
        off = b.mul_wide(gid, Immediate(4, S))
        v = b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, off)))
        p = b.binary(Op.MUL, F, v, Immediate(3.0, F), mode="rn" if rn else None)
        r = b.add(F, p, Immediate(1.0, F))
        if reuse_mul:
            r = b.add(F, r, p)
        b.store(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, out, b.mul_wide(gid, Immediate(4, S)))), r)
        return finish(b)
    return b_


X = np.random.default_rng(6).standard_normal(64).astype(np.float32)
ARGS = ({"x": X}, {"out": np.zeros(64, np.float32)})


def test_fuses_int_wide_and_float():
    o = ops(differential(build(), PIPE, *ARGS))
    assert "mad.lo.s32" in o and o.count("mad.wide.s32") == 2 and "fma.rn.f32" in o
    assert "mul.f32" not in o and "add.s64" not in o


def test_shared_mul_or_explicit_rounding_not_fused():
    assert "fma.rn.f32" not in ops(differential(build(reuse_mul=True), PIPE, *ARGS))
    assert "fma.rn.f32" not in ops(differential(build(rn=True), PIPE, *ARGS))
