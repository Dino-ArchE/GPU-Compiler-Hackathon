import numpy as np
import pytest

from core.ir import CTAID_X, NTID_X, TID_X, Address, Immediate, Op, PtxType, StateSpace

from helpers import differential, finish, new_function, ops

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32
PIPE = ["split-blocks", "fast-math", "copy-prop", "dce"]


def build(body):
    """out[g] = body(b, x[g]) for 64 threads."""
    def b_():
        f, b = new_function(grid=(2, 1, 1), block=(32, 1, 1))
        x = b.cvta_to_global(b.load_param(f.param("x")))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        gid = b.add(S, b.mul(S, b.mov(U, CTAID_X), b.mov(U, NTID_X)), b.mov(U, TID_X))
        off = b.mul_wide(gid, Immediate(4, S))
        v = b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, off)))
        b.store(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, out, off)), body(b, v))
        return finish(b)
    return b_


def fl(c):
    return Immediate(c, F)


def sqrt(b, v):
    return b.unary(Op.SQRT, F, v, mode="rn")


def div(b, a, y):
    return b.binary(Op.DIV, F, a, y, mode="rn")


X = np.random.default_rng(7).uniform(0.5, 4.0, 64).astype(np.float32)
ARGS = ({"x": X}, {"out": np.zeros(64, np.float32)})


@pytest.mark.parametrize("body, want, gone", [
    (lambda b, v: div(b, v, sqrt(b, v)), ["rsqrt.approx.f32", "mul.f32"], ["sqrt.rn.f32", "div.rn.f32"]),
    (lambda b, v: b.unary(Op.RCP, F, sqrt(b, b.add(F, v, fl(1.0))), mode="rn"),
     ["rsqrt.approx.f32"], ["rcp.rn.f32", "sqrt.rn.f32"]),
    (lambda b, v: div(b, v, fl(3.0)), ["mul.f32"], ["div.rn.f32"]),
    (lambda b, v: div(b, fl(1.0), v), ["rcp.approx.f32"], ["div.rn.f32"]),
    (lambda b, v: div(b, fl(2.0), v), ["div.full.f32"], ["div.rn.f32"]),
    (lambda b, v: b.unary(Op.RCP, F, v, mode="rn"), ["rcp.approx.f32"], ["rcp.rn.f32"]),
    (lambda b, v: sqrt(b, v), ["sqrt.approx.f32"], ["sqrt.rn.f32"]),
])
def test_rewrites(body, want, gone):
    o = ops(differential(build(body), PIPE, *ARGS, rtol=1e-6))
    assert all(w in o for w in want), o
    assert not any(g in o for g in gone), o


def test_shared_sqrt_keeps_sqrt():
    """sqrt(x) read twice: x/sqrt(x) + sqrt(x) must keep a sqrt for the second reader."""
    def body(b, v):
        s = sqrt(b, v)
        return b.add(F, div(b, v, s), s)
    o = ops(differential(build(body), PIPE, *ARGS))
    assert "rsqrt.approx.f32" not in o and "sqrt.approx.f32" in o and "div.full.f32" in o


def test_division_by_zero_or_tiny_constant_stays_a_division():
    for c in (0.0, 1e38):           # 1/1e38 is subnormal: multiplying would lose precision
        o = ops(differential(build(lambda b, v: div(b, v, fl(c))), PIPE, *ARGS))
        assert "mul.f32" not in o and "div.full.f32" in o
