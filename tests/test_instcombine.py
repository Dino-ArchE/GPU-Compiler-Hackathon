import numpy as np
import pytest

from core.ir import CTAID_X, NTID_X, TID_X, Address, Immediate, Instruction, Mod, Op, PtxType, StateSpace

from core.opt import Repeat
from helpers import differential, finish, new_function, ops, run

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32
X = np.random.default_rng(1).standard_normal(64).astype(np.float32)
XI = np.arange(-32, 32, dtype=np.int32)


def kernel(body, dtype=S, n=64):
    """out[gid] = body(b, gid, x[gid]) for 64 threads; body returns the stored register."""
    def build():
        f, b = new_function(grid=(2, 1, 1), block=(32, 1, 1))
        x = b.cvta_to_global(b.load_param(f.param("x")))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        gid = b.mad(S, b.mov(U, CTAID_X), b.mov(U, NTID_X), b.mov(U, TID_X))
        off = b.mul_wide(gid, Immediate(4, S))
        v = b.load(StateSpace.GLOBAL, dtype, Address(b.add(PtxType.S64, x, off)))
        r = body(b, gid, v)
        b.store(StateSpace.GLOBAL, dtype, Address(b.add(PtxType.S64, out, off)), r)
        return finish(b)
    return build


def check(body, dtype=S, expect_gone=()):
    x = XI if dtype is S else X
    f = differential(kernel(body, dtype), ["split-blocks", Repeat(["instcombine", "copy-prop", "dce"])],
                     {"x": x}, {"out": np.zeros(64, x.dtype)}, rtol=0, atol=0)
    for o in expect_gone:
        assert o not in ops(f), ops(f)
    return f


def i(v):
    return Immediate(v, S)


def fl(v):
    return Immediate(v, F)


@pytest.mark.parametrize("body,gone", [
    (lambda b, g, v: b.add(S, v, i(0)), ["add.s32"]),
    (lambda b, g, v: b.add(S, i(0), v), ["add.s32"]),
    (lambda b, g, v: b.mul(S, v, i(1)), ["mul.lo.s32"]),
    (lambda b, g, v: b.binary(Op.DIV, S, v, i(1)), ["div.s32"]),
    (lambda b, g, v: b.binary(Op.REM, S, v, i(-1)), ["rem.s32"]),
    (lambda b, g, v: b.binary(Op.AND, S, v, i(-1)), ["and.s32"]),
    (lambda b, g, v: b.binary(Op.SUB, S, v, v), ["sub.s32"]),
    (lambda b, g, v: b.add(S, b.add(S, b.add(S, v, i(3)), i(4)), i(-7)), ["add.s32"]),
    (lambda b, g, v: b.binary(Op.SUB, S, b.add(S, v, i(5)), i(5)), ["add.s32", "sub.s32"]),
    (lambda b, g, v: b.mul(S, b.mul(S, v, i(3)), i(5)), []),
    (lambda b, g, v: b.binary(Op.SHL, PtxType.B32, b.binary(Op.SHL, PtxType.B32, v, i(1)), i(2)), []),
])
def test_int_identities(body, gone):
    check(body, S, gone)


def test_mad_by_zero_is_its_addend():
    f = check(lambda b, g, v: b.mad(S, v, i(0), g))
    assert ops(f).count("mad.lo.s32") == 1           # only the gid computation is left


def test_reassociation_collapses_chain():
    f = check(lambda b, g, v: b.mul(S, b.mul(S, v, i(3)), i(5)))
    assert ops(f).count("mul.lo.s32") == 1
    mul = next(x for _, x in f.instructions() if x.op is Op.MUL and Mod.LO in x.types)
    assert mul.srcs[1].value == 15


@pytest.mark.parametrize("body,gone", [
    (lambda b, g, v: b.mul(F, v, fl(1.0)), ["mul.f32"]),
    (lambda b, g, v: b.add(F, v, fl(-0.0)), ["add.f32"]),
    (lambda b, g, v: b.binary(Op.SUB, F, v, fl(0.0)), ["sub.f32"]),
    (lambda b, g, v: b.mul(F, v, fl(-1.0)), ["mul.f32"]),
    (lambda b, g, v: b.binary(Op.DIV, F, v, fl(4.0), mode=Mod.RN), ["div.rn.f32"]),
    (lambda b, g, v: b.binary(Op.DIV, F, v, fl(0.125), mode=Mod.RN), ["div.rn.f32"]),
])
def test_float_identities_are_bit_exact(body, gone):
    check(body, F, gone)


def test_float_add_zero_kept():
    """-0.0 + 0.0 = +0.0, so x + 0.0 is not x."""
    f = check(lambda b, g, v: b.add(F, v, fl(0.0)), F)
    assert "add.f32" in ops(f)


def test_div_by_non_power_of_two_kept():
    f = check(lambda b, g, v: b.binary(Op.DIV, F, v, fl(3.0), mode=Mod.RN), F)
    assert "div.rn.f32" in ops(f)


def test_constant_hoisted_out_of_inner_add():
    """g + (v + 5) -> (g + v) + 5: the constant ends up outermost, ready for addr-fold."""
    f = check(lambda b, g, v: b.add(S, g, b.add(S, v, i(5))))
    adds = [inst for _, inst in f.instructions() if inst.op is Op.ADD and inst.ptx_type is S]
    assert len(adds) == 2 and isinstance(adds[-1].srcs[1], Immediate) and adds[-1].srcs[1].value == 5


def test_constant_not_hoisted_when_inner_add_is_shared():
    def body(b, g, v):
        inner = b.add(S, v, i(5))
        return b.mul(S, b.add(S, g, inner), inner)
    f = check(body)
    assert sum(1 for o in ops(f) if o == "add.s32") == 2
