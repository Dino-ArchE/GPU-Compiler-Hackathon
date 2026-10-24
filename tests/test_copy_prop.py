import numpy as np

from core.ir import CTAID_X, NTID_X, TID_X, Address, Cmp, Immediate, Op, PtxType, StateSpace

from helpers import differential, finish, new_function, ops, run

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32


def build():
    """out[gid] = (gid + c) where c = mov 5; t = mov tid (a copy of a copy)."""
    f, b = new_function(grid=(2, 1, 1), block=(4, 1, 1))
    out = b.cvta_to_global(b.load_param(f.param("out")))
    tid = b.mov(U, TID_X)
    t2 = b.mov(U, tid)
    gid = b.mad(S, b.mov(U, CTAID_X), b.mov(U, NTID_X), t2)
    c = b.mov(S, Immediate(5, S))
    v = b.add(S, gid, c)
    b.store(StateSpace.GLOBAL, S, Address(b.add(PtxType.S64, out, b.mul_wide(gid, Immediate(4, S)))), v)
    return finish(b)


def test_propagates_and_cleans_up():
    f = differential(build, ["split-blocks", "copy-prop", "dce"], {"x": np.zeros(1, np.float32)},
                     {"out": np.zeros(8, np.int32)})
    add = next(i for _, i in f.instructions() if i.op is Op.ADD and i.ptx_type is S)
    assert isinstance(add.srcs[1], Immediate) and add.srcs[1].value == 5
    # mov %r, %tid.x must stay (special registers only feed mov), but its copy is gone
    assert ops(f).count("mov.u32") == 3 and "mov.s32" not in ops(f)


def test_accumulator_copy_not_propagated():
    """acc is updated in a loop; snap = mov acc must keep reading the value at the copy."""
    def build_loop():
        f, b = new_function(grid=(1, 1, 1), block=(1, 1, 1))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        acc = b.mov(S, Immediate(0, S))
        loop = b.new_block("LOOP")
        b.set_block(loop)
        b.inst(Op.ADD, S, dsts=[acc], srcs=[acc, Immediate(3, S)])
        snap = b.mov(S, acc)                    # snapshot of acc after this iteration
        b.inst(Op.ADD, S, dsts=[acc], srcs=[acc, Immediate(1, S)])
        b.bra(loop, guard=b.setp(Cmp.LT, S, acc, Immediate(20, S)))
        b.set_block(b.new_block("DONE"))
        b.store(StateSpace.GLOBAL, S, Address(out), snap)
        return finish(b)

    f = differential(build_loop, ["split-blocks", "copy-prop", "dce"], {"x": np.zeros(1, np.float32)},
                     {"out": np.zeros(1, np.int32)})
    assert ops(f).count("mov.s32") == 2


def test_float_immediate_not_put_into_int_op():
    f, b = new_function()
    c = b.mov(F, Immediate(1.0, F))
    x = b.inst(Op.MOV, PtxType.B32, dsts=[b.reg(PtxType.B32)], srcs=[c])   # bit-cast style user
    finish(b)
    run(f, "split-blocks", "copy-prop")
    assert x.srcs[0] is c
