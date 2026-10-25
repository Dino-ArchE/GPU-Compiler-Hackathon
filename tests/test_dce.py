import numpy as np

from core.ir import Address, Cmp, Immediate, Op, PtxType, StateSpace

from helpers import differential, finish, new_function, ops, run

F, S = PtxType.F32, PtxType.S32


def imm(v, t=S):
    return Immediate(v, t)


def test_removes_dead_chain_keeps_store():
    f, b = new_function()
    out = b.cvta_to_global(b.load_param(f.param("out")))
    a = b.mov(F, imm(1.0, F))
    dead1 = b.add(F, a, a)
    b.mul(F, dead1, dead1)                       # dead chain: dead1 -> mul
    b.store(StateSpace.GLOBAL, F, Address(out), a)
    finish(b)
    assert run(f, "split-blocks", "dce")
    assert ops(f) == ["ld.param.u64", "cvta.to.global.u64", "mov.f32", "st.global.f32", "ret"]


def build_loop():
    """acc = 0; i = 0; do { acc += x[i]; junk = acc * 2; i++ } while (i < 8); out[0] = acc"""
    f, b = new_function(grid=(1, 1, 1), block=(1, 1, 1))
    x = b.cvta_to_global(b.load_param(f.param("x")))
    out = b.cvta_to_global(b.load_param(f.param("out")))
    acc = b.mov(F, imm(0.0, F))
    i = b.mov(S, imm(0))
    loop = b.new_block("LOOP")
    b.set_block(loop)
    v = b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, b.mul_wide(i, imm(4)))))
    b.inst(Op.ADD, F, dsts=[acc], srcs=[acc, v])
    b.mul(F, acc, imm(2.0, F))                   # dead
    b.inst(Op.ADD, S, dsts=[i], srcs=[i, imm(1)])
    b.bra(loop, guard=b.setp(Cmp.LT, S, i, imm(8)))
    b.set_block(b.new_block("DONE"))
    b.store(StateSpace.GLOBAL, F, Address(out), acc)
    return finish(b)


def test_keeps_loop_carried_accumulator():
    x = np.arange(8, dtype=np.float32)
    f = differential(build_loop, ["split-blocks", "dce"], {"x": x}, {"out": np.zeros(1, np.float32)})
    assert ops(f).count("mul.f32") == 0
    assert ops(f).count("add.f32") == 1


def test_guarded_def_is_not_a_kill():
    """r = 1; @p r = 2; out = r: the unguarded def is still needed where p is false."""
    f, b = new_function(grid=(1, 1, 1), block=(2, 1, 1))
    from core.ir import TID_X
    out = b.cvta_to_global(b.load_param(f.param("out")))
    tid = b.mov(PtxType.U32, TID_X)
    r = b.mov(S, imm(1))
    p = b.setp(Cmp.EQ, PtxType.U32, tid, Immediate(1, PtxType.U32))
    b.inst(Op.MOV, S, dsts=[r], srcs=[imm(2)], guard=p)
    addr = b.add(PtxType.S64, out, b.mul_wide(b.mov(S, tid), imm(4)))
    b.store(StateSpace.GLOBAL, S, Address(addr), r)
    finish(b)
    run(f, "split-blocks", "dce")
    assert ops(f).count("mov.s32") == 3
