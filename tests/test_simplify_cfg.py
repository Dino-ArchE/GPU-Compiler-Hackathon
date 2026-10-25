import numpy as np

from core.ir import TID_X, Address, Cmp, Immediate, Op, PtxType, StateSpace

from core.opt import Repeat
from helpers import differential, finish, new_function, ops

S, U = PtxType.S32, PtxType.U32
NOX = {"x": np.zeros(1, np.float32)}


def store_tid(b, out, tid, value):
    b.store(StateSpace.GLOBAL, S, Address(b.add(PtxType.S64, out, b.mul_wide(tid, Immediate(4, S)))), value)


def test_known_guard_collapses_to_one_block():
    def b_():
        f, b = new_function(grid=(1, 1, 1), block=(32, 1, 1))
        exit_ = b.new_block("EXIT")
        out = b.cvta_to_global(b.load_param(f.param("out")))
        tid = b.mov(U, TID_X)
        b.bra(exit_, guard=b.setp(Cmp.GE, U, tid, Immediate(32, U)))   # never true
        store_tid(b, out, tid, b.mov(S, Immediate(7, S)))
        return finish(b, exit_)
    pipe = ["split-blocks", Repeat(["range-fold", "const-fold", "dce", "simplify-cfg"])]
    f = differential(b_, pipe, NOX, {"out": np.zeros(32, np.int32)})
    assert len(f.blocks) == 1 and "bra" not in ops(f)


def test_threading_and_unreachable():
    """bra A; (dead block); A: bra B; B: store -> one branch-free path."""
    def b_():
        f, b = new_function(grid=(1, 1, 1), block=(32, 1, 1))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        tid = b.mov(U, TID_X)
        a, dead, bb = b.new_block("A"), b.new_block("DEAD"), b.new_block("B")
        b.bra(a)
        b.set_block(dead)
        store_tid(b, out, tid, b.mov(S, Immediate(1, S)))
        b.set_block(a)
        b.bra(bb)
        b.set_block(b.new_block("PAD"))
        b.inst(Op.RET)
        b.set_block(bb)
        store_tid(b, out, tid, b.mov(S, Immediate(2, S)))
        return finish(b)
    f = differential(b_, ["split-blocks", "simplify-cfg"], NOX, {"out": np.zeros(32, np.int32)})
    assert ops(f).count("st.global.s32") == 1
    assert ops(f).count("bra") <= 1


def test_diamond_kept():
    """A real if/else must survive."""
    def b_():
        f, b = new_function(grid=(1, 1, 1), block=(32, 1, 1))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        tid = b.mov(U, TID_X)
        other, join = b.new_block("ELSE"), b.new_block("JOIN")
        r = b.mov(S, Immediate(1, S))
        b.bra(other, guard=b.setp(Cmp.LT, U, tid, Immediate(10, U)))
        b.inst(Op.MOV, S, dsts=[r], srcs=[Immediate(2, S)])
        b.bra(join)
        b.set_block(other)
        b.inst(Op.MOV, S, dsts=[r], srcs=[Immediate(3, S)])
        b.set_block(join)
        store_tid(b, out, tid, r)
        return finish(b)
    f = differential(b_, ["split-blocks", "simplify-cfg"], NOX, {"out": np.zeros(32, np.int32)})
    assert ops(f).count("bra") == 2
