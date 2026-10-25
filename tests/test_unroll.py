import numpy as np
import pytest

from core.ir import CTAID_X, NTID_X, TID_X, Address, Cmp, Immediate, Op, PtxType, StateSpace

from core.opt import Repeat, build_pipeline
from core.opt.pipeline import SCALAR_RANGE
from helpers import differential, finish, new_function, ops

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32
PIPE = ["split-blocks", "unroll", Repeat(SCALAR_RANGE)]
N = 2048


def build(start="gid", stop=N, step=None, threads=128, blocks=1, branchy=False, cleanup_first=False):
    """out[gid] = sum of x[i] for i in start..stop:step (step defaults to the thread count);
    `branchy` adds `if (v > 0)` inside the body, which puts control flow in the loop."""
    step = step or threads * blocks

    def b_():
        f, b = new_function(grid=(blocks, 1, 1), block=(threads, 1, 1))
        x = b.cvta_to_global(b.load_param(f.param("x")))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        gid = b.add(S, b.mul(S, b.mov(U, CTAID_X), b.mov(U, NTID_X)), b.mov(U, TID_X))
        acc = b.mov(F, Immediate(0.0, F))
        i = b.mov(S, gid if start == "gid" else Immediate(start, S))
        head, body, done = b.new_block("LOOP"), b.new_block("BODY"), b.new_block("DONE")
        b.set_block(head)
        b.bra(done, guard=b.setp(Cmp.GE, S, i, Immediate(stop, S)))
        b.set_block(body)
        v = b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, b.mul_wide(i, Immediate(4, S)))))
        if branchy:
            skip = b.new_block("SKIP")
            b.bra(skip, guard=b.setp(Cmp.LE, F, v, Immediate(0.0, F)))
            b.set_block(b.new_block("POS"))
            b.inst(Op.ADD, F, dsts=[acc], srcs=[acc, v])
            b.set_block(skip)
        else:
            b.inst(Op.ADD, F, dsts=[acc], srcs=[acc, v])
        b.inst(Op.ADD, S, dsts=[i], srcs=[i, Immediate(step, S)])
        b.bra(head)
        b.set_block(done)
        b.store(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, out, b.mul_wide(gid, Immediate(4, S)))), acc)
        return finish(b)
    return b_


X = np.random.default_rng(3).standard_normal(N).astype(np.float32)


def args(threads=128, blocks=1):
    return {"x": X}, {"out": np.zeros(threads * blocks, np.float32)}


def loops(f):
    """Backward branches left (a branch to a block at or before its own)."""
    pos = {b: k for k, b in enumerate(f.blocks)}
    return sum(1 for b in f.blocks for i in b.instructions if i.op is Op.BRA and pos[i.target] <= pos[b])


def test_thread_strided_loop_unrolls_completely():
    f = differential(build(), PIPE, *args(), rtol=1e-5)
    o = ops(f)
    assert loops(f) == 0 and o.count("ld.global.f32") == N // 128 and "setp.ge.s32" not in o


def test_constant_loop_folds_to_immediate_offsets():
    f = differential(build(start=0, stop=32, step=1), PIPE + ["addr-fold", Repeat(SCALAR_RANGE)],
                     *args(), rtol=1e-5)
    assert loops(f) == 0 and ops(f).count("ld.global.f32") == 32
    # every copy addresses x + 4k off the one base: no per-copy index math survives
    assert ops(f).count("mul.wide.s32") <= 2


def test_long_loop_is_partially_unrolled():
    f = differential(build(start=0, stop=1024, step=1, threads=32), PIPE, *args(threads=32), rtol=1e-4)
    o = ops(f)
    assert loops(f) == 1 and o.count("ld.global.f32") == 8 and o.count("setp.ge.s32") == 1


def test_body_with_control_flow():
    f = differential(build(branchy=True), PIPE, *args(), rtol=1e-5)
    assert loops(f) == 0 and ops(f).count("ld.global.f32") == N // 128


def test_merged_single_block_loop():
    """simplify-cfg first merges the header into the body: one block that is header and latch."""
    f = differential(build(), ["split-blocks", "simplify-cfg", *PIPE[1:]], *args(), rtol=1e-5)
    assert loops(f) == 0


@pytest.mark.parametrize("kw, a", [
    (dict(stop=2000), args()),                      # threads run 15 or 16 trips: not uniform
    (dict(blocks=4, step=128), args(blocks=4)),     # start up to 511 with step 128: 16..12 trips
])
def test_non_uniform_trip_count_is_left_alone(kw, a):
    f = differential(build(**kw), PIPE, *a, rtol=1e-5)
    assert loops(f) == 1 and ops(f).count("ld.global.f32") == 1


def test_in_full_pipeline():
    differential(build(), build_pipeline(2), *args(), rtol=1e-5)
    differential(build(start=0, stop=1024, step=1, threads=32), build_pipeline(2), *args(threads=32), rtol=1e-4)
