import numpy as np

from core.ir import CTAID_X, NTID_X, TID_X, Address, Cmp, Immediate, Op, PtxType, StateSpace

from core.opt import Repeat
from helpers import differential, finish, new_function, ops

F, S, U = PtxType.F32, PtxType.S32, PtxType.U32
N = 256


def build_strided(guarded_entry=True, two_entries=False):
    """acc = 0; for (i = tid; i < N; i += ntid) acc += x[i] * scale[0]; out[tid] = acc

    Loop shape like a strided reduction: HEAD tests i < N, BODY does the work and jumps back.
    The stride (ntid*1), scale[0] and the x base address are invariant.
    """
    def b_():
        f, b = new_function(params=("x", "scale", "out"), grid=(1, 1, 1), block=(32, 1, 1),
                            readonly=("x", "scale"))
        x = b.cvta_to_global(b.load_param(f.param("x")))
        sc = b.cvta_to_global(b.load_param(f.param("scale")))
        out = b.cvta_to_global(b.load_param(f.param("out")))
        tid = b.mov(U, TID_X)
        acc = b.mov(F, Immediate(0.0, F))
        i = b.mov(S, tid)
        head, body, done = b.new_block("HEAD"), b.new_block("BODY"), b.new_block("DONE")
        if two_entries:                          # header gets two outside preds: needs a new PRE
            b.bra(head, guard=b.setp(Cmp.LT, U, tid, Immediate(16, U)))
            b.set_block(b.new_block("MID"))
            b.mov(S, Immediate(0, S))
        elif guarded_entry:
            b.bra(head)
        b.set_block(head)
        s = b.load(StateSpace.GLOBAL, F, Address(sc))              # invariant load in the header
        b.bra(done, guard=b.setp(Cmp.GE, S, i, Immediate(N, S)))
        b.set_block(body)
        stride = b.mul(S, b.mov(U, NTID_X), Immediate(1, S))     # invariant arithmetic
        v = b.load(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, x, b.mul_wide(i, Immediate(4, S)))))
        b.inst(Op.FMA, "rn", F, dsts=[acc], srcs=[v, s, acc])
        b.inst(Op.ADD, S, dsts=[i], srcs=[i, stride])
        b.bra(head)
        b.set_block(done)
        b.store(StateSpace.GLOBAL, F, Address(b.add(PtxType.S64, out, b.mul_wide(tid, Immediate(4, S)))), acc)
        return finish(b)
    return b_


X = np.random.default_rng(5).standard_normal(N).astype(np.float32)
ARGS = ({"x": X, "scale": np.array([0.5], np.float32)}, {"out": np.zeros(32, np.float32)})


def loop_blocks(f):
    return [b for b in f.blocks if b.name.startswith(("HEAD", "BODY"))]


def test_hoists_invariants_out_of_strided_loop():
    f = differential(build_strided(), ["split-blocks", "licm"], *ARGS, rtol=1e-5)
    inside = [i for b in loop_blocks(f) for i in b.instructions]
    assert not any(i.op is Op.MUL and i.types == ("lo", S) for i in inside)
    assert sum(i.op is Op.LD for i in inside) == 1               # only x[i] stays
    assert not any(i.op is Op.MOV and i.srcs[0] is NTID_X for i in inside)


def test_creates_preheader_for_two_entries():
    f = differential(build_strided(two_entries=True), ["split-blocks", "licm"], *ARGS, rtol=1e-5)
    pre = next(b for b in f.blocks if b.name.startswith("PRE"))
    assert any(i.op is Op.LD for i in pre.instructions)
    assert f.blocks.index(pre) + 1 == f.blocks.index(next(b for b in f.blocks if b.name.startswith("HEAD")))


def test_variant_values_stay():
    f = differential(build_strided(), ["split-blocks", "licm", Repeat(["dce"])], *ARGS, rtol=1e-5)
    inside = [i for b in loop_blocks(f) for i in b.instructions]
    assert any(i.op is Op.FMA for i in inside) and any(i.op is Op.ADD and i.ptx_type is S for i in inside)
