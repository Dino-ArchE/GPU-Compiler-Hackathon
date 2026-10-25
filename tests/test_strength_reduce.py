import numpy as np
import pytest

from core.ir import CTAID_X, NTID_X, TID_X, Address, Immediate, Op, PtxType, StateSpace

from core.opt import Repeat
from core.opt.passes.strength_reduce import magic_u32
from helpers import differential, finish, new_function, ops

S, U = PtxType.S32, PtxType.U32
PIPE = ["split-blocks", "launch-specialize", "strength-reduce", Repeat(["copy-prop", "dce"])]


@pytest.mark.parametrize("d", [3, 5, 6, 7, 10, 12, 24, 100, 125, 255, 641, 1000, 4097, 65537, 2**31 - 1])
@pytest.mark.parametrize("hi", [255, 4095, 2**20, 2**31 - 1])
def test_magic_numbers_exact(d, hi):
    mm = magic_u32(d, hi)
    assert mm is not None
    m, s = mm
    rng = np.random.default_rng(d)
    xs = {0, 1, d - 1, d, d + 1, hi, hi - 1, *(k * d + r for k in (1, 2, hi // d) for r in (0, d - 1))}
    xs |= set(int(v) for v in rng.integers(0, hi + 1, 2000))
    for x in xs:
        if 0 <= x <= hi:
            assert (x * m) >> (32 + s) == x // d, (x, d)


def build(op, c, scale):
    """out[g] = op(g * scale, c) for 1024 threads (g*scale is provably >= 0)."""
    def b_():
        f, b = new_function(grid=(4, 1, 1), block=(256, 1, 1))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        gid = b.mad(S, b.mov(U, CTAID_X), b.mov(U, NTID_X), b.mov(U, TID_X))
        x = b.mul(S, gid, Immediate(scale, S))
        r = b.binary(op, S, x, Immediate(c, S)) if op is not Op.MUL else b.mul(S, x, Immediate(c, S))
        b.store(StateSpace.GLOBAL, S, Address(b.add(PtxType.S64, out, b.mul_wide(gid, Immediate(4, S)))), r)
        return finish(b)
    return b_


@pytest.mark.parametrize("op,c,scale,gone", [
    (Op.DIV, 8, 1, "div.s32"), (Op.REM, 8, 1, "rem.s32"), (Op.MUL, 16, 1, None),
    (Op.DIV, 7, 3, "div.s32"), (Op.REM, 7, 3, "rem.s32"),
    (Op.DIV, 1000, 2_000_000, "div.s32"), (Op.REM, 1000, 2_000_000, "rem.s32"),
])
def test_rewrites_match_reference(op, c, scale, gone):
    f = differential(build(op, c, scale), PIPE, {"x": np.zeros(1, np.float32)},
                     {"out": np.zeros(1024, np.int32)}, rtol=0, atol=0)
    if gone:
        assert gone not in ops(f)
    else:
        assert "shl.b32" in ops(f)


def test_possibly_negative_dividend_untouched():
    def b_():
        f, b = new_function(grid=(1, 1, 1), block=(64, 1, 1))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        tid = b.mov(U, TID_X)
        x = b.add(S, tid, Immediate(-32, S))                 # in [-32, 31]
        r = b.binary(Op.DIV, S, x, Immediate(4, S))
        b.store(StateSpace.GLOBAL, S, Address(b.add(PtxType.S64, out, b.mul_wide(tid, Immediate(4, S)))), r)
        return finish(b)
    f = differential(b_, PIPE, {"x": np.zeros(1, np.float32)}, {"out": np.zeros(64, np.int32)}, rtol=0, atol=0)
    assert "div.s32" in ops(f)
