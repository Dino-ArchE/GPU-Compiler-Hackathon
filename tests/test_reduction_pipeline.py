"""A block reduction shaped like r01 (strided loop, warp butterfly, smem, bar.sync, red.add),
hand-built in our IR, run through the full pipelines. Guards the passes against the control
flow and synchronization that the reduction codegen will produce."""

import numpy as np
import pytest

from core.ir import TID_X, Address, Builder, Cmp, Function, Immediate, Instruction, Mod, Module, Op, PtxType, StateSpace

from core.opt import build_pipeline
from helpers import differential, ops

F, S, U, B32 = PtxType.F32, PtxType.S32, PtxType.U32, PtxType.B32
N, THREADS, BLOCKS = 1024, 128, 2


def build():
    m = Module()
    f = m.add_function(Function("k_sum", ["x", "sum"]))
    f.launch_grid, f.launch_block = (BLOCKS, 1, 1), (THREADS, 1, 1)
    f.readonly = {f.param("x")}
    smem = m.add_shared("acc_smem", 4, 4 * THREADS // 32)
    b = Builder(f)
    exit_ = b.new_block("EXIT")
    x = b.cvta_to_global(b.load_param(f.param("x")))
    out = b.cvta_to_global(b.load_param(f.param("sum")))
    from core.ir import CTAID_X, NTID_X
    tid = b.mov(U, TID_X)
    gid = b.mad(S, b.mov(U, CTAID_X), b.mov(U, NTID_X), tid)

    # for (i = gid; i < N; i += BLOCKS*THREADS) acc += x[i]
    acc = b.mov(F, Immediate(0.0, F))
    i = b.mov(S, gid)
    head, body, done = b.new_block("LOOP"), b.new_block("BODY"), b.new_block("LOOP_END")
    b.set_block(head)
    b.bra(done, guard=b.setp(Cmp.GE, S, i, Immediate(N, S)))
    b.set_block(body)
    stride = b.mul(S, b.mov(U, NTID_X), Immediate(BLOCKS, S))
    v = b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, b.mul_wide(i, Immediate(4, S)))))
    b.inst(Op.ADD, F, dsts=[acc], srcs=[acc, v])
    b.inst(Op.ADD, S, dsts=[i], srcs=[i, stride])
    b.bra(head)
    b.set_block(done)

    # warp butterfly: acc += shfl.bfly(acc, k) for k = 16..1
    for k in (16, 8, 4, 2, 1):
        bits = b.reg(B32)
        b.inst(Op.MOV, B32, dsts=[bits], srcs=[acc])
        other_bits = b.reg(B32)
        b.inst(Op.SHFL, Mod.SYNC, Mod.BFLY, B32, dsts=[other_bits],
               srcs=[bits, Immediate(k, U), Immediate(31, U), Immediate(-1, B32)])
        other = b.reg(F)
        b.inst(Op.MOV, B32, dsts=[other], srcs=[other_bits])
        b.inst(Op.ADD, F, dsts=[acc], srcs=[acc, other])

    # if (lane == 0) smem[warp] = acc;  bar.sync;  if (tid == 0) red.add(sum, sum(smem))
    lane = b.binary(Op.REM, S, tid, Immediate(32, S))
    warp = b.binary(Op.DIV, S, tid, Immediate(32, S))
    after = b.new_block("AFTER_STORE")
    b.bra(after, guard=b.setp(Cmp.NE, S, lane, Immediate(0, S)))
    b.set_block(b.new_block("STORE"))
    base = b.mov(PtxType.U64, smem)
    b.store(StateSpace.SHARED, F, Address(b.add(PtxType.S64, base, b.mul_wide(warp, Immediate(4, S)))), acc)
    b.set_block(after)
    b.sync()
    b.bra(exit_, guard=b.setp(Cmp.NE, U, tid, Immediate(0, U)))
    b.set_block(b.new_block("FOLD"))
    total = None
    for w in range(THREADS // 32):
        base = b.mov(PtxType.U64, smem)
        idx = b.mul(S, b.mov(S, Immediate(w, S)), Immediate(1, S))
        part = b.load(StateSpace.SHARED, F, Address(b.add(PtxType.S64, base, b.mul_wide(idx, Immediate(4, S)))))
        total = part if total is None else b.add(F, total, part)
    b.append(Instruction(Op.RED, (StateSpace.GLOBAL, Mod.ADD, F), srcs=[Address(out), total]))
    b.set_block(exit_)
    b.inst(Op.RET)
    return m


X = np.random.default_rng(7).standard_normal(N).astype(np.float32)


@pytest.mark.parametrize("level", [1, 2])
def test_reduction_survives_pipeline(level):
    f = differential(build, build_pipeline(level), {"x": X}, {"sum": np.zeros(1, np.float32)}, rtol=1e-5)
    o = ops(f)
    assert o.count("shfl.sync.bfly.b32") == 5 and "bar.sync" in o and "red.global.add.f32" in o
    if level == 2:
        assert "rem.s32" not in o and "div.s32" not in o
        assert "ld.shared.v4.f32" in o                 # the 4 warp partials: one 16-byte load
        # the strided loop runs N / (BLOCKS*THREADS) = 4 trips in every thread: fully unrolled
        assert not any(b.name.startswith("BODY") for b in f.blocks)
        assert o.count("ld.global.nc.f32") == N // (BLOCKS * THREADS)


def test_reference_value():
    from interp import run_kernel
    out = run_kernel(build(), {"x": X}, {"sum": np.zeros(1, np.float32)})
    np.testing.assert_allclose(out["sum"][0], X.sum(dtype=np.float64), rtol=1e-4)
