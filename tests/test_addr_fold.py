import numpy as np

from core.ir import CTAID_X, NTID_X, TID_X, Address, Immediate, Op, PtxType, StateSpace

from core.opt import Repeat
from helpers import differential, finish, new_function, ops

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32
PIPE = ["split-blocks", "addr-fold", Repeat(["instcombine", "cse", "copy-prop", "dce"])]


def build():
    """out[g] = x[4g] + x[4g+1] + x[4g+2] + x[4g+3], each address computed from scratch."""
    f, b = new_function(grid=(2, 1, 1), block=(32, 1, 1))
    x = b.cvta_to_global(b.load_param(f.param("x")))
    out = b.cvta_to_global(b.load_param(f.param("out")))
    gid = b.mad(S, b.mov(U, CTAID_X), b.mov(U, NTID_X), b.mov(U, TID_X))
    acc = None
    for k in range(4):
        idx = b.add(S, b.mul(S, gid, Immediate(4, S)), Immediate(k, S))
        v = b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, b.mul_wide(idx, Immediate(4, S)))))
        acc = v if acc is None else b.add(F, acc, v)
    b.store(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, out, b.mul_wide(gid, Immediate(4, S))), 0), acc)
    return finish(b)


def test_offsets_share_one_base():
    x = np.random.default_rng(3).standard_normal(256).astype(np.float32)
    f = differential(build, PIPE, {"x": x}, {"out": np.zeros(64, np.float32)})
    loads = [i for _, i in f.instructions() if i.op is Op.LD and i.space is StateSpace.GLOBAL]
    assert len({id(i.srcs[0].base) for i in loads}) == 1
    assert sorted(i.srcs[0].offset for i in loads) == [0, 4, 8, 12]
    assert ops(f).count("mul.wide.s32") == 2      # one for x, one for out


def test_constant_pointer_offset():
    def b_():
        f, b = new_function(grid=(1, 1, 1), block=(1, 1, 1))
        x = b.cvta_to_global(b.load_param(f.param("x")))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        v = b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, Immediate(8, PtxType.S64)), 4))
        b.store(StateSpace.GLOBAL, F, Address(out), v)
        return finish(b)
    x = np.arange(4, dtype=np.float32)
    f = differential(b_, PIPE, {"x": x}, {"out": np.zeros(1, np.float32)})
    ld = next(i for _, i in f.instructions() if i.op is Op.LD and i.space is StateSpace.GLOBAL)
    assert ld.srcs[0].offset == 12 and "add.s64" not in ops(f)
