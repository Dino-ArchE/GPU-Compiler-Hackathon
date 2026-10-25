import numpy as np

from core.ir import CTAID_X, NTID_X, TID_X, Address, Cmp, Immediate, Op, PtxType, StateSpace

from core.opt import Repeat
from helpers import differential, finish, new_function, ops

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32
PIPE = ["split-blocks", "launch-specialize",
        Repeat(["range-fold", "const-fold", "instcombine", "copy-prop", "cse", "dce"])]


def build(n):
    """if (gid >= n) return; out[gid] = (gid % 64) + gid / 64 + min(tid, 512)"""
    def b_():
        f, b = new_function(grid=(4, 1, 1), block=(64, 1, 1))
        exit_ = b.new_block("EXIT")
        out = b.cvta_to_global(b.load_param(f.param("out")))
        tid = b.mov(U, TID_X)
        gid = b.mad(S, b.mov(U, CTAID_X), b.mov(U, NTID_X), tid)
        b.bra(exit_, guard=b.setp(Cmp.GE, S, gid, Immediate(n, S)))
        lane = b.binary(Op.REM, S, tid, Immediate(64, S))
        warp = b.binary(Op.DIV, S, tid, Immediate(64, S))
        m = b.binary(Op.MIN, S, tid, Immediate(512, S))
        v = b.add(S, b.add(S, lane, warp), m)
        b.store(StateSpace.GLOBAL, S, Address(b.add(PtxType.S64, out, b.mul_wide(gid, Immediate(4, S)))), v)
        return finish(b, exit_)
    return b_


def run_case(n):
    return differential(build(n), PIPE, {"x": np.zeros(1, np.float32)}, {"out": np.zeros(256, np.int32)})


def test_redundant_guard_and_trivial_div_rem_vanish():
    f = run_case(256)                                  # 4 x 64 threads: the guard never fires
    o = ops(f)
    assert "setp.ge.s32" not in o and "bra" not in o
    assert "rem.s32" not in o and "div.s32" not in o and "min.s32" not in o
    mad = next(i for _, i in f.instructions() if i.op is Op.MAD)
    assert isinstance(mad.srcs[1], Immediate) and mad.srcs[1].value == 64


def test_needed_guard_stays():
    o = ops(run_case(200))
    assert "setp.ge.s32" in o and "bra" in o
