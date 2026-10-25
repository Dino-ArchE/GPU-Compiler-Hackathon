import numpy as np

from core.ir import TID_X, Address, Cmp, Immediate, Op, PtxType, StateSpace

from helpers import differential, finish, new_function, ops, run
from core.opt.fold import evaluate, round_f32
from core.ir import Binary, Instruction

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32


def test_evaluate_matches_hardware_rules():
    def ev(op, t, *vals, mode=None):
        return evaluate(Binary(op, t, None, None, None, mode=mode), list(vals))
    assert ev(Op.DIV, S, -7, 2) == -3 and ev(Op.REM, S, -7, 2) == -1      # C truncation
    assert ev(Op.ADD, S, 2**31 - 1, 1) == -2**31                          # wraps
    assert ev(Op.SHR, S, -8, 1) == -4 and ev(Op.SHR, U, 2**32 - 8, 1) == 2**31 - 4
    assert ev(Op.MUL, S, 3, 5, mode="wide") == 15
    assert ev(Op.MUL, U, 2**31, 4, mode="hi") == 2
    assert ev(Op.DIV, S, 1, 0) is None
    assert ev(Op.ADD, F, 0.1, 0.2) == round_f32(round_f32(0.1) + round_f32(0.2))
    assert ev(Op.DIV, F, 1.0, 3.0) is None                    # div without .rn is approximate
    assert ev(Op.DIV, F, 1.0, 3.0, mode="rn") == round_f32(1 / 3)
    assert ev(Op.MUL, F, 3e38, 10.0) == float("inf")
    mad = Instruction(Op.MAD, ("lo", S), dsts=[None], srcs=[None] * 3)
    assert evaluate(mad, [3, 4, 5]) == 17


def build_known_guard():
    """p = 3 < 2 (false): @p bra SKIP is never taken; out[tid] = 6 * 7 + tid"""
    f, b = new_function(grid=(1, 1, 1), block=(4, 1, 1))
    skip = b.new_block("SKIP")
    out = b.cvta_to_global(b.load_param(f.param("out")))
    tid = b.mov(U, TID_X)
    p = b.setp(Cmp.LT, S, Immediate(3, S), Immediate(2, S))
    b.bra(skip, guard=p)
    v = b.add(S, b.mul(S, Immediate(6, S), Immediate(7, S)), tid)
    addr = b.add(PtxType.S64, out, b.mul_wide(tid, Immediate(4, S)))
    b.store(StateSpace.GLOBAL, S, Address(addr), v)
    return finish(b, skip)


def test_folds_values_and_branches():
    f = differential(build_known_guard, ["split-blocks", "const-fold", "dce"], {"x": np.zeros(1, np.float32)},
                     {"out": np.zeros(4, np.int32)})
    o = ops(f)
    assert "setp.lt.s32" not in o and "bra" not in o and "mul.lo.s32" not in o
    assert "mov.s32" in o


def test_always_true_guard_becomes_unconditional():
    f, b = new_function(grid=(1, 1, 1), block=(1, 1, 1))
    end = b.new_block("END")
    p = b.setp(Cmp.EQ, S, Immediate(1, S), Immediate(1, S))
    b.bra(end, guard=p, negate=False)
    b.mov(S, Immediate(0, S))
    finish(b, end)
    run(f, "split-blocks", "const-fold")
    bra = next(i for _, i in f.instructions() if i.op is Op.BRA)
    assert bra.guard is None
