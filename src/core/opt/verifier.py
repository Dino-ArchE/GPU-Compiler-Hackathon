"""Structural checks on our IR, run after every pass that changed something.

A failure names the pass that broke the invariant, which is far easier to debug than wrong
numbers from the GPU or a ptxas error on generated text.
"""

from core.ir import (
    TYPE_TO_CLASS, Address, BasicBlock, Immediate, Op, PtxType, Register, SpecialRegister, Value,
    Vector, VirtualRegister,
)

from .analysis.cfg import CFG
from .analysis.liveness import Liveness

# ops whose single dst has the register class of the op type (add.f32 -> %f, add.s32 -> %r)
SAME_TYPE_DST = {Op.ADD, Op.SUB, Op.MUL, Op.MAD, Op.FMA, Op.DIV, Op.REM, Op.NEG, Op.ABS, Op.MIN,
                 Op.MAX, Op.RCP, Op.SQRT, Op.RSQRT, Op.EX2, Op.LG2, Op.AND, Op.OR, Op.XOR, Op.NOT,
                 Op.SHL, Op.SHR, Op.SELP, Op.MOV}


class VerifyError(Exception):
    pass


def _bitcast_mov(inst, d, t):
    """mov.b32 %f1, %r1 (and back) reinterprets bits between same-width registers: legal PTX."""
    return inst.op is Op.MOV and t.value.startswith("b") and d.PTX_TYPE.nbytes == t.nbytes


def verify(function, after="?"):
    def fail(msg):
        raise VerifyError(f"after {after}: {function.name}: {msg}")

    if not function.blocks:
        fail("no blocks")
    names = set()
    blocks = set(function.blocks)
    for block in function.blocks:
        if block.name in names:
            fail(f"duplicate block name {block.name}")
        names.add(block.name)
        for inst in block.instructions:
            for v in (*inst.dsts, *inst.srcs):
                if not isinstance(v, Value):
                    fail(f"{inst!r}: operand {v!r} is not a Value")
            for d in inst.dsts:
                ok = isinstance(d, VirtualRegister) or (
                    isinstance(d, Vector) and all(isinstance(e, VirtualRegister) for e in d.elems))
                if not ok:
                    fail(f"{inst!r}: cannot write {d!r}")
            if inst.guard is not None and TYPE_TO_CLASS[PtxType.PRED] is not type(inst.guard):
                fail(f"{inst!r}: guard {inst.guard!r} is not a predicate register")
            if inst.op is Op.BRA and inst.target not in blocks:
                fail(f"{inst!r}: target is not a block of this function")
            if inst.op is not Op.BRA and any(isinstance(v, BasicBlock) for v in inst.srcs):
                fail(f"{inst!r}: only bra takes a block operand")
            if inst.op is not Op.MOV and any(isinstance(v, SpecialRegister) for v in inst.srcs):
                fail(f"{inst!r}: special registers can only be read by mov")
            if inst.op in SAME_TYPE_DST and "wide" not in inst.types and len(inst.dsts) == 1:
                d, t = inst.dsts[0], inst.ptx_type
                if t is not None and isinstance(d, Register) and type(d) is not TYPE_TO_CLASS[t] \
                        and not _bitcast_mov(inst, d, t):
                    fail(f"{inst!r}: dst {d} has the wrong register class for .{t}")
            if inst.op in SAME_TYPE_DST and inst.op not in (Op.MOV, Op.SELP) and "wide" not in inst.types:
                t = inst.ptx_type
                srcs = inst.srcs[:1] if inst.op in (Op.SHL, Op.SHR) else inst.srcs
                for v in srcs:
                    if t is not None and isinstance(v, VirtualRegister) and type(v) is not TYPE_TO_CLASS[t]:
                        fail(f"{inst!r}: source {v} has the wrong register class for .{t}")
            for v in inst.srcs:
                if isinstance(v, Address) and isinstance(v.base, Immediate):
                    fail(f"{inst!r}: address base is an immediate")

    last = function.blocks[-1].terminator
    if last is None or last.guard is not None:
        fail(f"last block {function.blocks[-1].name} can fall off the end of the kernel")

    # a register read before any write on some path is a bug, unless every def is guarded
    # (`@p mov r, a; @!p mov r, b` is fine but invisible to liveness)
    live = Liveness(function, CFG(function)).live_in[function.blocks[0]]
    defs = {}
    for _, inst in function.instructions():
        for r in inst.defs():
            defs.setdefault(r, []).append(inst)
    for r in live:
        if not defs.get(r) or any(d.guard is None for d in defs[r]):
            fail(f"{r} may be read before it is written")
