from .opcode import Cmp, Op
from .reg import Register
from .type_enum import PtxType, StateSpace
from .value import Address, Immediate, Vector


# Ops a pass may never delete or reorder freely, even when their results are unused:
# they write memory, synchronize threads, exchange data between lanes, or change control flow.
SIDE_EFFECT_OPS = frozenset({Op.ST, Op.RED, Op.ATOM, Op.CP, Op.BAR, Op.SHFL, Op.BRA, Op.RET})
MEMORY_READ_OPS = frozenset({Op.LD, Op.ATOM, Op.CP})
MEMORY_WRITE_OPS = frozenset({Op.ST, Op.RED, Op.ATOM, Op.CP})


class Instruction:
    """One PTX instruction as data: `@guard op.types dsts, srcs;`.

    `op` is an Op; `types` holds the dotted modifiers after it (("global", "v4", "f32")) so passes
    can match on them. Subclasses are typed constructors that fill these same fields, so the
    printer and defs()/uses() treat every instruction alike. Stores and branches have no dsts.
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

        Inside an [address] or a {vector} only a Register may take old's place; a plain source
        also accepts an Immediate. The guard is replaced only by another register.
        """
        count = 0
        srcs = []
        for v in self.srcs:
            v2 = _substitute(v, old, new)
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

    def __repr__(self):
        guard = f"@{'!' if self.negate else ''}{self.guard} " if self.guard is not None else ""
        return f"{type(self).__name__}({guard}{'.'.join((self.op, *self.types))} {self.dsts} <- {self.srcs})"


def _substitute(value, old, new):
    """`value` with register `old` replaced by `new`, or `value` itself if nothing changed."""
    if value is old:
        return new
    if isinstance(value, Address) and value.base is old and isinstance(new, Register):
        return Address(new, value.offset)
    if isinstance(value, Vector) and any(e is old for e in value.elems):
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
