import numpy as np

from core.ir import CTAID_X, NTID_X, TID_X, Address, Cmp, Immediate, Op, PtxType, StateSpace

from helpers import differential, finish, new_function, ops, text

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32
PIPE = ["split-blocks", "if-convert", "simplify-cfg"]


def build(with_bar=False, size=1):
    """if (tid % 32 == 0) out[gid / 32] = x[gid] (+1, `size` times); optionally a bar inside."""
    def b_():
        f, b = new_function(grid=(2, 1, 1), block=(64, 1, 1))
        x = b.cvta_to_global(b.load_param(f.param("x")))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        tid = b.mov(U, TID_X)
        gid = b.mad(S, b.mov(U, CTAID_X), b.mov(U, NTID_X), tid)
        v = b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, b.mul_wide(gid, Immediate(4, S)))))
        lane = b.binary(Op.AND, PtxType.B32, tid, Immediate(31, PtxType.B32))
        join = b.new_block("JOIN")
        b.bra(join, guard=b.setp(Cmp.NE, U, lane, Immediate(0, U)))
        b.set_block(b.new_block("THEN"))
        for _ in range(size):
            v = b.add(F, v, Immediate(1.0, F))
        if with_bar:
            b.sync()
        w = b.binary(Op.SHR, U, gid, Immediate(5, U))
        b.store(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, out, b.mul_wide(w, Immediate(4, S)))), v)
        b.set_block(join)
        return finish(b)
    return b_


X = np.random.default_rng(5).standard_normal(128).astype(np.float32)
ARGS = ({"x": X}, {"out": np.zeros(4, np.float32)})


def branches(f):
    return sum(1 for _, i in f.instructions() if i.op is Op.BRA)


def test_triangle_becomes_predicated():
    f = differential(build(), PIPE, *ARGS)
    assert branches(f) == 0 and len(f.blocks) == 1
    st = next(i for _, i in f.instructions() if i.op is Op.ST)
    assert st.guard is not None and st.negate        # @!p: runs where the skip was not taken
    assert "@!%p" in next(l for l in text(f).splitlines() if "st.global" in l)


def test_barrier_or_long_block_keeps_the_branch():
    assert branches(differential(build(with_bar=True), PIPE, *ARGS)) == 1
    assert branches(differential(build(size=12), PIPE, *ARGS)) == 1
