from .opcode import Cmp, Op
from .reg import Register
from .type_enum import PtxType, StateSpace
from .value import Address, Immediate, Vector


# Ops a pass may never delete or reorder freely, even when their results are unused:
# they write memory, synchronize threads, exchange data between lanes, or change control flow.
# mma is a warp collective like shfl: all 32 lanes must execute it together.
SIDE_EFFECT_OPS = frozenset({Op.ST, Op.RED, Op.ATOM, Op.CP, Op.BAR, Op.SHFL, Op.MMA, Op.BRA, Op.RET})
MEMORY_READ_OPS = frozenset({Op.LD, Op.ATOM, Op.CP})
MEMORY_WRITE_OPS = frozenset({Op.ST, Op.RED, Op.ATOM, Op.CP})


class Instruction:
    """One PTX instruction as data: `@guard op.types dsts, srcs;`.

    `op` is an Op; `types` holds the dotted modifiers after it (("global", "v4", "f32")) so passes
    can match on them. Subclasses are typed constructors that fill these same fields, so the
    __str__ and defs()/uses() treat every instruction alike. Stores and branches have no dsts.
    """

    def __init__(self, op, types=(), dsts=(), srcs=(), guard=None, negate=False, comment=None):
        self.op = Op(op)
        self.types = tuple(types)
        self.dsts = tuple(dsts)
        self.srcs = tuple(srcs)
        self.guard = guard        # predicate Register: the instruction runs only where it holds
        self.negate = negate      # @!%p instead of @%p
        self.comment = comment

    def defs(self):
        """Registers this instruction writes."""
        return [r for v in self.dsts for r in _registers(v)]

    def uses(self):
        """Registers this instruction reads, including the guard and address bases."""
        regs = [r for v in self.srcs for r in _registers(v)]
        return regs + [self.guard] if self.guard is not None else regs

    # ------------------------------------------------------------------ properties for passes

    @property
    def has_side_effects(self):
        """True if the instruction must stay even when nothing reads its dsts."""
        return self.op in SIDE_EFFECT_OPS

    @property
    def reads_memory(self):
        """ld.param reads a kernel argument, which never changes: it does not count."""
        return self.op in MEMORY_READ_OPS and self.space is not StateSpace.PARAM

    @property
    def writes_memory(self):
        return self.op in MEMORY_WRITE_OPS

    @property
    def is_terminator(self):
        """bra and ret end a basic block (guarded or not) once blocks are split."""
        return self.op in (Op.BRA, Op.RET)

    @property
    def space(self):
        """The StateSpace a memory instruction touches, or None."""
        for t in self.types:
            if isinstance(t, StateSpace):
                return t
        return None

    @property
    def ptx_type(self):
        """The last type modifier that is a PtxType (the operation type: f32 in add.f32)."""
        for t in reversed(self.types):
            if isinstance(t, PtxType):
                return t
        return None

    @property
    def target(self):
        """The BasicBlock a bra jumps to."""
        return self.srcs[0] if self.op is Op.BRA else None

    # ------------------------------------------------------------------ mutation helpers

    def replace_use(self, old, new):
        """Replace reads of register `old` with `new`; returns how many operands changed.

        Inside an [address] only a Register may take old's place; a plain source, and an element
        of a st {vector}, also accepts an Immediate (mma fragments do not). The guard is replaced
        only by another register.
        """
        count = 0
        srcs = []
        vector_imm = self.op is not Op.MMA
        for v in self.srcs:
            v2 = _substitute(v, old, new, vector_imm)
            count += v2 is not v
            srcs.append(v2)
        self.srcs = tuple(srcs)
        if self.guard is old and isinstance(new, Register):
            self.guard = new
            count += 1
        return count

    def replace_def(self, old, new):
        """Rename a destination register; returns how many dsts changed."""
        count = 0
        dsts = []
        for v in self.dsts:
            v2 = _substitute(v, old, new)
            count += v2 is not v
            dsts.append(v2)
        self.dsts = tuple(dsts)
        return count

    def copy(self):
        """A shallow clone: new tuples, same operand objects."""
        clone = Instruction.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        return clone

    def __str__(self):
        """The PTX line: `@!%p1 ld.global.f32 %f1, [%rd2];  // comment`."""
        guard = f"@{'!' if self.negate else ''}{self.guard} " if self.guard is not None else ""
        opcode = ".".join((self.op, *self.types))
        operands = ", ".join(str(v) for v in (*self.dsts, *self.srcs))
        text = f"{guard}{opcode} {operands};" if operands else f"{guard}{opcode};"
        return f"{text}\t// {self.comment}" if self.comment else text

    def __repr__(self):
        guard = f"@{'!' if self.negate else ''}{self.guard} " if self.guard is not None else ""
        return f"{type(self).__name__}({guard}{'.'.join((self.op, *self.types))} {self.dsts} <- {self.srcs})"


def _substitute(value, old, new, vector_imm=True):
    """`value` with register `old` replaced by `new`, or `value` itself if nothing changed."""
    if value is old:
        return new
    if isinstance(value, Address) and value.base is old and isinstance(new, Register):
        return Address(new, value.offset)
    if isinstance(value, Vector) and any(e is old for e in value.elems) \
            and (vector_imm or isinstance(new, Register)):
        return Vector(new if e is old else e for e in value.elems)
    return value


def _registers(value):
    if isinstance(value, Register):
        return [value]
    if isinstance(value, Address):
        return _registers(value.base)
    if isinstance(value, Vector):
        return [r for e in value.elems for r in _registers(e)]
    return []


def _vec(value):
    """The .v2/.v4 modifier for a Vector operand, nothing for a scalar."""
    return (f"v{len(value.elems)}",) if isinstance(value, Vector) else ()


class Load(Instruction):
    """ld.<space>[.vN].<type> dst, [addr]. The type is the element read: pointers are untyped."""

    def __init__(self, space, ptx_type, dst, addr, **kw):
        super().__init__(Op.LD, (space, *_vec(dst), ptx_type), dsts=[dst], srcs=[addr], **kw)


class Store(Instruction):
    """st.<space>[.vN].<type> [addr], src"""

    def __init__(self, space, ptx_type, addr, src, **kw):
        super().__init__(Op.ST, (space, *_vec(src), ptx_type), srcs=[addr, src], **kw)


class Mov(Instruction):
    """mov.<type> dst, src: copy a register, immediate, special register or symbol address."""

    def __init__(self, ptx_type, dst, src, **kw):
        super().__init__(Op.MOV, (ptx_type,), dsts=[dst], srcs=[src], **kw)


class Sync(Instruction):
    """bar.sync id: block-wide barrier."""

    def __init__(self, barrier_id=0, **kw):
        super().__init__(Op.BAR, ("sync",), srcs=[Immediate(barrier_id, PtxType.U32)], **kw)


class CpAsync(Instruction):
    """cp.async.ca|cg.shared.global [dst], [src], nbytes: copy 4, 8 or 16 bytes from global to
    shared memory without passing through registers (.cg: 16 bytes only, bypasses L1).
    The copy lands at a later cp.async.wait_group; a bar.sync then publishes it to the CTA."""

    def __init__(self, dst_addr, src_addr, nbytes, cache="ca", **kw):
        super().__init__(Op.CP, ("async", cache, StateSpace.SHARED, StateSpace.GLOBAL),
                         srcs=[dst_addr, src_addr, Immediate(nbytes, PtxType.U32)], **kw)


class CpAsyncCommit(Instruction):
    """cp.async.commit_group: close the group of this thread's cp.async issued so far."""

    def __init__(self, **kw):
        super().__init__(Op.CP, ("async", "commit_group"), **kw)


class CpAsyncWait(Instruction):
    """cp.async.wait_group n: wait until at most n of this thread's groups are still in flight."""

    def __init__(self, pending=0, **kw):
        super().__init__(Op.CP, ("async", "wait_group"), srcs=[Immediate(pending, PtxType.U32)], **kw)


class CvtTf32(Instruction):
    """cvt.rna.tf32.f32 dst(.b32), src(.f32): round to tf32 (10 mantissa bits, ties away from
    zero), the operand format of the f32 tensor-core mma."""

    def __init__(self, dst, src, **kw):
        super().__init__(Op.CVT, ("rna", "tf32", PtxType.F32), dsts=[dst], srcs=[src], **kw)


class Mma(Instruction):
    """mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {d0..d3}, {a0..a3}, {b0, b1}, {c0..c3}

    One warp computes D(16x8) = A(16x8) * B(8x8) + C(16x8) on the tensor cores, accumulating in
    f32. Every lane holds a fragment of each matrix (PTX ISA "Matrix Fragments for mma.m16n8k8",
    with g = lane >> 2 and t = lane & 3):

        A (row-major, tf32 in .b32): a0 (g, t)   a1 (g+8, t)   a2 (g, t+4)   a3 (g+8, t+4)
        B (col-major, tf32 in .b32): b0 (t, g)   b1 (t+4, g)
        C, D (f32):                  c0 (g, 2t)  c1 (g, 2t+1)  c2 (g+8, 2t)  c3 (g+8, 2t+1)

    Verified on the hosted RTX 5090 by tools/mma_probe.py (bit-exact; a transposed B fails).
    .sync.aligned: all 32 lanes must execute the same mma, so it is never guarded per lane.
    D may be the same registers as C (accumulate in place).
    """

    SHAPE = "m16n8k8"

    def __init__(self, d, a, b, c, **kw):
        super().__init__(Op.MMA, ("sync", "aligned", self.SHAPE, "row", "col",
                                  PtxType.F32, "tf32", "tf32", PtxType.F32),
                         dsts=[Vector(d)], srcs=[Vector(a), Vector(b), Vector(c)], **kw)


class Cvta(Instruction):
    """cvta[.to].<space>.u64 dst, src: convert an address between generic and a state space.

    to=True is generic -> space (cvta.to.global, for kernel params); to=False is space -> generic.
    The type is the address size (u64 under .address_size 64), not the pointee's type.
    """

    def __init__(self, space, dst, src, to=True, ptx_type=PtxType.U64, **kw):
        types = ("to", space, ptx_type) if to else (space, ptx_type)
        super().__init__(Op.CVTA, types, dsts=[dst], srcs=[src], **kw)


class Binary(Instruction):
    """<op>[.mode].<type> dst, a, b: add, sub, mul, div, min, max, rem, and, or, shl, ...

    `mode` is the extra modifier some ops need, e.g. mul.lo / mul.wide for integers. With
    mode="wide" the type names the sources (s32) while dst is twice as wide (an s64 register).
    """

    def __init__(self, op, ptx_type, dst, a, b, mode=None, **kw):
        types = (mode, ptx_type) if mode else (ptx_type,)
        super().__init__(op, types, dsts=[dst], srcs=[a, b], **kw)


class Unary(Instruction):
    """<op>[.mode].<type> dst, a: neg, abs, rcp, sqrt, ex2, ... (mode: rn, approx, ...)"""

    def __init__(self, op, ptx_type, dst, a, mode=None, **kw):
        types = (mode, ptx_type) if mode else (ptx_type,)
        super().__init__(op, types, dsts=[dst], srcs=[a], **kw)


class Setp(Instruction):
    """setp.<cmp>.<type> p, a, b: compare, writing a predicate register."""

    def __init__(self, cmp, ptx_type, dst, a, b, **kw):
        super().__init__(Op.SETP, (Cmp(cmp), ptx_type), dsts=[dst], srcs=[a, b], **kw)


class Bra(Instruction):
    """[@p] bra target: jump to a BasicBlock; unconditional without a guard."""

    def __init__(self, target, **kw):
        super().__init__(Op.BRA, (), srcs=[target], **kw)
