import numpy as np

from core.ir import CTAID_X, NTID_X, TID_X, Address, Cmp, Immediate, Op, PtxType, StateSpace

from core.opt import Repeat
from helpers import differential, finish, new_function, ops

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32
PIPE = ["split-blocks", Repeat(["cse", "copy-prop", "dce"])]


def build(store_to_x=False, readonly=("x",)):
    """out[g] = x[0]*x[g] + x[0]*x[g]; the index math and x[0] are computed twice."""
    def b_():
        f, b = new_function(grid=(2, 1, 1), block=(32, 1, 1), readonly=readonly)
        x = b.cvta_to_global(b.load_param(f.param("x")))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        gid = b.mad(S, b.mov(U, CTAID_X), b.mov(U, NTID_X), b.mov(U, TID_X))
        terms = []
        for _ in range(2):
            c = b.load(StateSpace.GLOBAL, F, Address(x))
            if store_to_x and not terms:
                b.store(StateSpace.GLOBAL, F, Address(x, 4 * 64), c)
            v = b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, b.mul_wide(gid, Immediate(4, S)))))
            terms.append(b.mul(F, c, v))
        r = b.add(F, terms[0], terms[1])
        b.store(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, out, b.mul_wide(gid, Immediate(4, S)))), r)
        return finish(b)
    return b_


X = np.random.default_rng(2).standard_normal(65).astype(np.float32)


def test_merges_pure_math_and_readonly_loads():
    f = differential(build(), PIPE, {"x": X}, {"out": np.zeros(64, np.float32)})
    o = ops(f)
    assert o.count("ld.global.f32") == 2 and o.count("mul.f32") == 1
    assert o.count("mul.wide.s32") == 1          # the out address reuses x's offset


def test_loads_from_written_buffer_not_merged():
    f = differential(build(store_to_x=True, readonly=()), PIPE, {"x": X.copy()}, {"out": np.zeros(64, np.float32)})
    assert ops(f).count("ld.global.f32") >= 3


def test_dominance_respected():
    """An expression computed in one branch arm is not reused in the join block."""
    def b_():
        f, b = new_function(grid=(1, 1, 1), block=(32, 1, 1))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        tid = b.mov(U, TID_X)
        join = b.new_block("JOIN")
        b.bra(join, guard=b.setp(Cmp.LT, U, tid, Immediate(16, U)))
        b.set_block(b.new_block("ARM"))
        b.mul(S, tid, Immediate(3, S))
        b.set_block(join)
        r = b.mul(S, tid, Immediate(3, S))
        b.store(StateSpace.GLOBAL, S, Address(b.add(PtxType.S64, out, b.mul_wide(tid, Immediate(4, S)))), r)
        return finish(b)
    f = differential(b_, ["split-blocks", "cse"], {"x": X}, {"out": np.zeros(32, np.int32)})
    assert ops(f).count("mul.lo.s32") == 2
