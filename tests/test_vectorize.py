import numpy as np

from core.ir import CTAID_X, NTID_X, TID_X, Address, Immediate, Op, PtxType, StateSpace

from core.opt import Repeat
from helpers import differential, finish, new_function, ops

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32
PIPE = ["split-blocks", "addr-fold", Repeat(["instcombine", "cse", "copy-prop", "dce"]), "vectorize-mem", "ld-nc"]
X = np.random.default_rng(4).standard_normal(64 * 4 + 8).astype(np.float32)


def build(stride=4, first=0, write_x_between=False):
    """out[stride*g + k] = x[stride*g + first + k] * 2 for k < 4, as scalar ld/st."""
    def b_():
        f, b = new_function(grid=(2, 1, 1), block=(32, 1, 1), readonly=() if write_x_between else ("x",))
        x = b.cvta_to_global(b.load_param(f.param("x")))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        gid = b.mad(S, b.mov(U, CTAID_X), b.mov(U, NTID_X), b.mov(U, TID_X))
        base = b.mul(S, gid, Immediate(stride, S))
        vals = []
        for k in range(4):
            i = b.add(S, base, Immediate(first + k, S))
            vals.append(b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, b.mul_wide(i, Immediate(4, S))))))
            if write_x_between and k == 1:
                b.store(StateSpace.GLOBAL, F, Address(x, 4 * 260), vals[0])
        for k, v in enumerate(vals):
            i = b.add(S, base, Immediate(k, S))
            b.store(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, out, b.mul_wide(i, Immediate(4, S)))),
                    b.mul(F, v, Immediate(2.0, F)))
        return finish(b)
    return b_


def run_case(**kw):
    out = np.zeros(64 * 4 + 8, np.float32)
    return ops(differential(build(**kw), PIPE, {"x": X.copy()}, {"out": out}))


def test_aligned_run_becomes_v4_nc():
    o = run_case()
    assert o.count("ld.global.nc.v4.f32") == 1 and o.count("st.global.v4.f32") == 1
    assert "ld.global.f32" not in o and "st.global.f32" not in o


def test_misaligned_start_uses_smaller_vectors():
    # x[4g+1 .. 4g+4]: 4-byte aligned only at the start -> scalar, v2 (offset 8 is 8-aligned), scalar
    o = run_case(first=1)
    assert "ld.global.nc.v4.f32" not in o and o.count("ld.global.nc.v2.f32") == 1


def test_odd_stride_unknown_alignment_stays_scalar():
    o = run_case(stride=3)
    assert not any(".v" in x for x in o)


def test_store_between_loads_blocks_merge_and_nc():
    o = run_case(write_x_between=True)
    assert not any(x.startswith("ld.global.nc") for x in o)
    assert "ld.global.v4.f32" not in o
