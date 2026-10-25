import numpy as np

from core.ir import TID_X, Address, Immediate, PtxType, StateSpace

from helpers import differential, finish, new_function, ops

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32


def build(readonly):
    def b_():
        f, b = new_function(grid=(1, 1, 1), block=(8, 1, 1), readonly=readonly)
        x = b.cvta_to_global(b.load_param(f.param("x")))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        off = b.mul_wide(b.mov(U, TID_X), Immediate(4, S))
        v = b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, off)))
        o = b.add(PtxType.S64, out, off)
        b.store(StateSpace.GLOBAL, F, Address(o), b.add(F, v, b.load(StateSpace.GLOBAL, F, Address(o))))
        return finish(b)
    return b_


def test_only_readonly_inputs_get_nc():
    x = np.arange(8, dtype=np.float32)
    o = ops(differential(build(("x",)), ["split-blocks", "ld-nc"], {"x": x}, {"out": np.ones(8, np.float32)}))
    assert o.count("ld.global.nc.f32") == 1 and o.count("ld.global.f32") == 1   # out is read-modify-write
    o = ops(differential(build(()), ["split-blocks", "ld-nc"], {"x": x}, {"out": np.ones(8, np.float32)}))
    assert "ld.global.nc.f32" not in o
