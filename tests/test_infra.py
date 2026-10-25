import pytest

from core.ir import (
    CTAID_X, NTID_X, TID_X, Address, Cmp, Immediate, Instruction, Load, Op, PtxType, StateSpace, Vector,
)
from core.opt import PASSES, ANALYSES, PassManager, Repeat, VerifyError, verify
from core.opt.pass_manager import FunctionPass

from helpers import am, finish, new_function, run


def guarded_kernel():
    """gid = ctaid*ntid+tid; if gid >= 1000 exit; out[gid] = x[gid] (one block, mid-block bra)."""
    f, b = new_function()
    exit_ = b.new_block("EXIT")
    x = b.cvta_to_global(b.load_param(f.param("x")))
    out = b.cvta_to_global(b.load_param(f.param("out")))
    c, n, t = (b.mov(PtxType.U32, s) for s in (CTAID_X, NTID_X, TID_X))
    gid = b.mad(PtxType.S32, c, n, t)
    p = b.setp(Cmp.GE, PtxType.S32, gid, Immediate(1000, PtxType.S32))
    b.bra(exit_, guard=p)
    off = b.mul_wide(gid, Immediate(4, PtxType.S32))
    v = b.load(StateSpace.GLOBAL, PtxType.F32, Address(b.add(PtxType.S64, x, off)))
    b.store(StateSpace.GLOBAL, PtxType.F32, Address(b.add(PtxType.S64, out, off)), v)
    return finish(b, exit_), exit_


def test_registry_has_core_entries():
    assert "split-blocks" in PASSES
    for name in ("cfg", "dominators", "loops", "defuse", "liveness", "ranges"):
        assert name in ANALYSES


def test_split_blocks_makes_cfg_valid():
    f, exit_ = guarded_kernel()
    with pytest.raises(ValueError):
        am(f).get("cfg")
    assert run(f, "split-blocks")
    assert len(f.blocks) == 3 and f.blocks[-1] is exit_
    cfg = am(f).get("cfg")
    entry, body = f.blocks[0], f.blocks[1]
    assert set(cfg.succs[entry]) == {exit_, body}
    assert cfg.succs[body] == [exit_]
    assert not run(f, "split-blocks")            # idempotent


def test_dominators_and_liveness():
    f, exit_ = guarded_kernel()
    run(f, "split-blocks")
    a = am(f)
    dom = a.get("dominators")
    entry, body = f.blocks[0], f.blocks[1]
    assert dom.dominates(entry, exit_) and not dom.dominates(body, exit_)
    assert dom.idom[exit_] is entry
    live = a.get("liveness")
    assert live.live_in[entry] == set()
    assert live.live_out[body] == set()
    assert len(live.live_in[body]) >= 2          # x, out bases and gid offset cross the edge


def loop_kernel():
    """acc = 0; i = 0; LOOP: acc += x[i]; i += 1; if i < 8 goto LOOP; out[0] = acc"""
    f, b = new_function()
    x = b.cvta_to_global(b.load_param(f.param("x")))
    out = b.cvta_to_global(b.load_param(f.param("out")))
    acc = b.mov(PtxType.F32, Immediate(0.0, PtxType.F32))
    i = b.mov(PtxType.S32, Immediate(0, PtxType.S32))
    loop = b.new_block("LOOP")
    b.set_block(loop)
    off = b.mul_wide(i, Immediate(4, PtxType.S32))
    v = b.load(StateSpace.GLOBAL, PtxType.F32, Address(b.add(PtxType.S64, x, off)))
    b.inst(Op.ADD, PtxType.F32, dsts=[acc], srcs=[acc, v])
    b.inst(Op.ADD, PtxType.S32, dsts=[i], srcs=[i, Immediate(1, PtxType.S32)])
    p = b.setp(Cmp.LT, PtxType.S32, i, Immediate(8, PtxType.S32))
    b.bra(loop, guard=p)
    b.set_block(b.new_block("DONE"))
    b.store(StateSpace.GLOBAL, PtxType.F32, Address(out), acc)
    return finish(b), loop


def test_loops_and_multi_def_registers():
    f, loop_block = loop_kernel()
    run(f, "split-blocks")
    a = am(f)
    loops = list(a.get("loops"))
    assert len(loops) == 1 and loops[0].header is loop_block and loops[0].blocks == {loop_block}
    assert loops[0].preheader(a.get("cfg")) is f.blocks[0]
    du = a.get("defuse")
    acc = f.blocks[0].instructions[-2].dsts[0]
    assert len(du.defs_of(acc)) == 2 and du.single_def(acc) is None
    assert acc in a.get("liveness").live_in[loop_block]
    # a loop counter has no known range; its byte offset is still a multiple of 4
    rng = a.get("ranges")
    off = loop_block.instructions[0].dsts[0]
    assert rng.align(off, PtxType.S64) == 4 and not rng.nonneg(off, PtxType.S64)


def test_ranges_use_launch():
    f, _ = guarded_kernel()
    run(f, "split-blocks")
    rng = am(f).get("ranges")
    gid = next(i for _, i in f.instructions() if i.op is Op.MAD).dsts[0]
    r = rng.get(gid, PtxType.S32)
    assert (r.lo, r.hi) == (0, 4 * 256 - 1)
    off = next(i for _, i in f.instructions() if "wide" in i.types).dsts[0]
    assert rng.get(off, PtxType.S64).align == 4
    assert rng.constant(NTID_X, PtxType.U32) == 256


def test_replace_use_reaches_addresses_and_vectors():
    f, b = new_function()
    r0, r1, r2 = b.reg(PtxType.U64), b.reg(PtxType.U64), b.reg(PtxType.F32)
    ld = Load(StateSpace.GLOBAL, PtxType.F32, Vector([r2, b.reg(PtxType.F32)]), Address(r0, 8))
    st = Instruction(Op.ST, (StateSpace.GLOBAL, "v2", PtxType.F32), srcs=[Address(r0), Vector([r2, r2])])
    assert ld.replace_use(r0, r1) == 1 and ld.srcs[0].base is r1 and ld.srcs[0].offset == 8
    assert st.replace_use(r2, Immediate(1.0, PtxType.F32)) == 1
    assert st.replace_use(r0, Immediate(0, PtxType.U64)) == 0     # no immediate address base
    assert ld.replace_def(r2, r0) == 1


def test_verifier_catches_bad_ir():
    f, b = new_function()
    r = b.reg(PtxType.S32)
    b.add(PtxType.S32, r, Immediate(1, PtxType.S32))     # r never written
    finish(b)
    with pytest.raises(VerifyError, match="read before"):
        verify(f)


def test_repeat_runs_to_fixed_point():
    calls = []

    class Countdown(FunctionPass):
        name = "countdown-test"

        def run(self, function, am):
            calls.append(1)
            return len(calls) < 3

    PASSES[Countdown.name] = Countdown
    try:
        f, b = new_function()
        finish(b)
        from core.ir import Module
        m = Module()
        m.add_function(f)
        PassManager([Repeat([Countdown.name])]).run(m)
        assert len(calls) == 3
    finally:
        del PASSES[Countdown.name]
