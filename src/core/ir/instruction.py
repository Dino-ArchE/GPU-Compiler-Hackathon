from .opcode import Cmp, Op
from .reg import Register
from .type_enum import PtxType
from .value import Address, Immediate, Vector


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

    def __repr__(self):
        return f"{type(self).__name__}({'.'.join((self.op, *self.types))} {self.dsts} <- {self.srcs})"


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
