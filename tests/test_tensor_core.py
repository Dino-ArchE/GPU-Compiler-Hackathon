"""Tensor-core and async-copy instructions: mma.sync m16n8k8 tf32 (1x and 3xTF32), cvt.rna.tf32,
cp.async + commit/wait. Each kernel is checked against numpy in the interpreter, survives the
-O2 pipeline unchanged in meaning, and assembles with ptxas for sm_89."""

import numpy as np
import pytest

from core.ir import TID_X, Address, Immediate, Module, Op, PtxType, StateSpace

from core.opt import build_pipeline
from helpers import differential, finish, new_function, ops
from interp import run_kernel

F, S, U, B32 = PtxType.F32, PtxType.S32, PtxType.U32, PtxType.B32
rng = np.random.default_rng(11)
A = rng.standard_normal((16, 8)).astype(np.float32)
B = rng.standard_normal((8, 8)).astype(np.float32)
ARGS = ({"a": A, "b": B}, {"d": np.zeros((16, 8), np.float32)})


def tf32(x):
    """cvt.rna.tf32.f32 in numpy: keep 10 mantissa bits, round half away from zero."""
    bits = x.astype(np.float32).view(np.uint32)
    return ((bits + np.uint32(0x1000)) & np.uint32(0xFFFFE000)).view(np.float32)


def build_mma(split):
    """One warp: D = A @ B with fragments loaded per the m16n8k8 layout (see ir.Mma)."""
    def b_():
        f, b = new_function(params=("a", "b", "d"), grid=(1, 1, 1), block=(32, 1, 1), readonly=("a", "b"))
        base = {p: b.cvta_to_global(b.load_param(f.param(p))) for p in ("a", "b", "d")}
        lane = b.mov(U, TID_X)
        g = b.binary(Op.SHR, U, lane, Immediate(2, U))
        t = b.binary(Op.AND, B32, lane, Immediate(3, B32))

        def elem(buf, row, col):                 # address of buf[row*8 + col]
            idx = b.add(S, b.mul(S, b.add(S, g if row[0] == "g" else t, Immediate(row[1], S)),
                                 Immediate(8, S)),
                        b.add(S, g if col[0] == "g" else t, Immediate(col[1], S)) if col[0] != "2t"
                        else b.add(S, b.mul(S, t, Immediate(2, S)), Immediate(col[1], S)))
            return Address(b.add(PtxType.S64, base[buf], b.mul_wide(idx, Immediate(4, S))))

        def frag(buf, places):
            return [b.load(StateSpace.GLOBAL, F, elem(buf, r, c)) for r, c in places]

        a = frag("a", [(("g", 0), ("t", 0)), (("g", 8), ("t", 0)), (("g", 0), ("t", 4)), (("g", 8), ("t", 4))])
        bb = frag("b", [(("t", 0), ("g", 0)), (("t", 4), ("g", 0))])
        acc = [b.mov(F, Immediate(0.0, F)) for _ in range(4)]
        if split:
            a_hi, a_lo = zip(*(b.split_tf32(x) for x in a))
            b_hi, b_lo = zip(*(b.split_tf32(x) for x in bb))
            # small terms first, so the big one adds onto an accumulator that already holds them
            acc = b.mma_tf32(a_lo, b_hi, acc, d=acc)
            acc = b.mma_tf32(a_hi, b_lo, acc, d=acc)
            acc = b.mma_tf32(a_hi, b_hi, acc, d=acc)
        else:
            acc = b.mma_tf32([b.cvt_tf32(x) for x in a], [b.cvt_tf32(x) for x in bb], acc)
        places = [(("g", 0), ("2t", 0)), (("g", 0), ("2t", 1)), (("g", 8), ("2t", 0)), (("g", 8), ("2t", 1))]
        for v, (r, c) in zip(acc, places):
            b.store(StateSpace.GLOBAL, F, elem("d", r, c), v)
        return finish(b)
    return b_


def test_mma_tf32_matches_numpy():
    got = run_kernel(_module(build_mma(False)()), *ARGS)["d"]
    np.testing.assert_allclose(got, tf32(A).astype(np.float64) @ tf32(B), rtol=1e-6, atol=1e-6)
    assert np.abs(got - A.astype(np.float64) @ B).max() > 1e-4      # plain tf32 really is coarse


def test_3xtf32_is_f32_accurate():
    got = run_kernel(_module(build_mma(True)()), *ARGS)["d"]
    np.testing.assert_allclose(got, A.astype(np.float64) @ B, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("split", [False, True])
def test_mma_through_o2(split):
    f = differential(build_mma(split), build_pipeline(2), *ARGS, rtol=1e-6)
    o = ops(f)
    assert o.count("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32") == (3 if split else 1)
    assert "st.global.v2.f32" in o                  # c0,c1 (and c2,c3) are adjacent columns


def build_cp_async():
    """32 threads: cp.async 16 bytes each into smem, wait, bar.sync, read a neighbour's slot."""
    def b_():
        m = Module()
        f, b = new_function(params=("x", "out"), grid=(1, 1, 1), block=(32, 1, 1))
        m.add_function(f)
        smem = m.add_shared("tile", 16, 32 * 16)
        x = b.cvta_to_global(b.load_param(f.param("x")))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        tid = b.mov(U, TID_X)
        off = b.mul_wide(tid, Immediate(16, S))
        dst = b.add(PtxType.S64, b.mov(PtxType.U64, smem), off)
        b.cp_async(Address(dst), Address(b.add(PtxType.S64, x, off)), 16, cache="cg")
        b.cp_async_commit()
        b.cp_async_wait(0)
        b.sync()
        nb = b.binary(Op.AND, B32, b.add(U, tid, Immediate(1, U)), Immediate(31, B32))
        src = b.add(PtxType.S64, b.mov(PtxType.U64, smem), b.mul_wide(nb, Immediate(16, S)))
        for k in range(4):
            v = b.load(StateSpace.SHARED, F, Address(src, 4 * k))
            b.store(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, out, off), 4 * k), v)
        finish(b)
        return m
    return b_


def test_cp_async_round_trip():
    x = rng.standard_normal(128).astype(np.float32)
    f = differential(build_cp_async(), build_pipeline(2), {"x": x}, {"out": np.zeros(128, np.float32)})
    got = run_kernel(build_cp_async()(), {"x": x}, {"out": np.zeros(128, np.float32)})["out"]
    np.testing.assert_array_equal(got, np.roll(x.reshape(32, 4), -1, axis=0).ravel())
    o = ops(f)
    assert "cp.async.cg.shared.global" in o and "cp.async.wait_group" in o
    assert "ld.shared.v4.f32" in o                  # the 4 reads merge, but stay after the wait


def _module(f):
    if isinstance(f, Module):
        return f
    m = Module()
    m.add_function(f)
    return m
