"""Integer value ranges and alignment: what each index register can hold.

For every integer value it answers Range(lo, hi, align): lo <= value <= hi, and `align` (a
power of two) divides the value. Sources of facts:

  - immediates
  - special registers, using the fixed launch from meta.json when known (ntid.x is then a
    constant, tid.x < block size, ctaid.x < grid size)
  - pointer params: buffers are allocated at least 16-byte aligned (cudaMalloc gives 256)
  - arithmetic on single-def registers (add, mul, mad, shl, shr, and, div, rem, min, max, ...)

A register with several defs or a guarded def (accumulators, loop counters) is unknown. Overflow
past the type's range also gives unknown bounds, but alignment survives wrap-around (modular
arithmetic keeps divisibility by powers of two). Lets passes prove `x >= 0` (signed div/rem ->
shifts and masks) and `addr % 16 == 0` (vector loads).
"""

from core.ir import CTAID_X, LANEID, NTID_X, TID_X, Address, Immediate, Op, Param, PtxType, Register, StateSpace

from ..pass_manager import Analysis, register_analysis

PARAM_ALIGN = 16
HUGE_ALIGN = 1 << 64     # alignment of the value 0: every power of two divides it

SIGNED = {PtxType.S16: 16, PtxType.S32: 32, PtxType.S64: 64}
UNSIGNED = {PtxType.U16: 16, PtxType.U32: 32, PtxType.U64: 64,
            PtxType.B16: 16, PtxType.B32: 32, PtxType.B64: 64}
WIDE = {PtxType.S16: PtxType.S32, PtxType.S32: PtxType.S64, PtxType.U16: PtxType.U32, PtxType.U32: PtxType.U64}


def is_int(t):
    return t in SIGNED or t in UNSIGNED


def bounds(t):
    if t in SIGNED:
        n = SIGNED[t]
        return -(1 << (n - 1)), (1 << (n - 1)) - 1
    n = UNSIGNED[t]
    return 0, (1 << n) - 1


def lowbit(v):
    """Largest power of two dividing v."""
    return HUGE_ALIGN if v == 0 else v & -v


def wrap(v, t):
    """v reduced to type t's range, as the hardware register would hold it."""
    lo, hi = bounds(t)
    span = hi - lo + 1
    return (v - lo) % span + lo


class Range:
    __slots__ = ("lo", "hi", "align")

    def __init__(self, lo, hi, align=1):
        self.lo, self.hi, self.align = lo, hi, align

    @property
    def exact(self):
        return self.lo == self.hi

    @property
    def nonneg(self):
        return self.lo >= 0

    def __repr__(self):
        return f"Range[{self.lo}, {self.hi}] align {self.align}"


def top(t, align=1):
    return Range(*bounds(t), align)


def fit(r, t):
    """The range as read through type t; bounds that don't fit (overflow) become t's full range."""
    lo, hi = bounds(t)
    if lo <= r.lo and r.hi <= hi:
        return r
    return Range(lo, hi, r.align)


def _corners(a, b, f):
    vals = [f(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
    return min(vals), max(vals)


class ValueRanges:
    def __init__(self, function, defuse, module=None):
        self.function = function
        self.du = defuse
        # shared arrays are aligned as declared
        self.symbol_align = {sym: align for sym, align, _ in module.shared} if module else {}
        self.memo = {}
        self.busy = set()
        grid, block = function.launch_grid, function.launch_block
        if block is not None:
            self.special = {
                TID_X: Range(0, block[0] - 1),
                NTID_X: Range(block[0], block[0], lowbit(block[0])),
                CTAID_X: Range(0, grid[0] - 1),
            }
        else:
            self.special = {TID_X: Range(0, 1023), NTID_X: Range(1, 1024), CTAID_X: Range(0, (1 << 31) - 1)}
        self.special[LANEID] = Range(0, 31)

    def get(self, value, t):
        """Range of `value` read as integer type t, or None if t is not an integer type."""
        t = PtxType(t)
        if not is_int(t):
            return None
        if isinstance(value, Immediate):
            v = wrap(int(value.value), t)
            return Range(v, v, lowbit(v))
        if value in self.special:
            return fit(self.special[value], t)
        if not isinstance(value, Register):
            return top(t)
        if value not in self.memo:
            inst = self.du.single_def(value)
            if inst is None or value in self.busy:
                return top(t)
            self.busy.add(value)
            try:
                self.memo[value] = self._eval(inst)
            finally:
                self.busy.discard(value)
        r = self.memo[value]
        return fit(r, t) if r is not None else top(t)

    def nonneg(self, value, t):
        r = self.get(value, t)
        return r is not None and r.nonneg

    def constant(self, value, t):
        """The value if it is known exactly, else None."""
        r = self.get(value, t)
        return r.lo if r is not None and r.exact else None

    def align(self, value, t):
        r = self.get(value, t)
        return r.align if r is not None else 1

    def address_align(self, addr):
        """Power of two dividing the byte address [base+offset] (global/param buffers)."""
        return min(self.base_align(addr.base), lowbit(addr.offset))

    def base_align(self, base):
        if isinstance(base, Param):
            return PARAM_ALIGN
        if base in self.symbol_align:
            return self.symbol_align[base]
        return self.align(base, PtxType.U64)

    def range_of(self, inst):
        """Range of the value this one `inst` writes, even when its dst has other defs (a loop
        counter's initial value), or None."""
        return self._eval(inst)

    # ------------------------------------------------------------------ transfer functions

    def _eval(self, inst):
        """Range of the value `inst` writes (None if not an integer op we understand)."""
        t = inst.ptx_type
        op, mods = inst.op, inst.types
        if t is None:
            return None

        if op is Op.LD:
            if inst.space is StateSpace.PARAM and t in (PtxType.U64, PtxType.B64, PtxType.S64):
                return top(t, PARAM_ALIGN)       # a buffer pointer
            return top(t) if is_int(t) else None
        if op is Op.CVTA:
            return top(t, self.align(inst.srcs[0], t))
        if op is Op.MOV and inst.srcs[0] in self.symbol_align:
            return top(t, self.symbol_align[inst.srcs[0]])
        if not is_int(t):
            return None

        wide = "wide" in mods
        rt = WIDE[t] if wide else t                     # result type
        s = [self.get(v, t) if not isinstance(v, Address) else top(t) for v in inst.srcs]

        if op is Op.MOV:
            return s[0]
        if op is Op.ADD:
            a, b = s
            r = Range(a.lo + b.lo, a.hi + b.hi, min(a.align, b.align))
        elif op is Op.SUB:
            a, b = s
            r = Range(a.lo - b.hi, a.hi - b.lo, min(a.align, b.align))
        elif op is Op.MUL and "hi" not in mods:
            a, b = s
            r = Range(*_corners(a, b, lambda x, y: x * y), a.align * b.align)
        elif op is Op.MAD and "hi" not in mods:
            a, b = s[:2]
            c = self.get(inst.srcs[2], rt)
            lo, hi = _corners(a, b, lambda x, y: x * y)
            r = Range(lo + c.lo, hi + c.hi, min(a.align * b.align, c.align))
        elif op is Op.SHL and s[1].exact:
            a, k = s[0], s[1].lo
            r = Range(a.lo << k, a.hi << k, a.align << k)
        elif op is Op.SHR and s[1].exact and s[0].nonneg:
            a, k = s[0], s[1].lo
            r = Range(a.lo >> k, a.hi >> k, max(1, a.align >> k))
        elif op is Op.AND and (s[0].exact or s[1].exact):
            a, m = (s[1], s[0].lo) if s[0].exact else (s[0], s[1].lo)
            if m >= 0:
                r = Range(0, m if not a.nonneg else min(m, a.hi), max(a.align, lowbit(m)))
            else:
                return top(rt, a.align)
        elif op is Op.DIV and s[1].exact and s[1].lo > 0 and s[0].nonneg:
            a, c = s[0], s[1].lo
            r = Range(a.lo // c, a.hi // c)
        elif op is Op.REM and s[1].exact and s[1].lo > 0 and s[0].nonneg:
            a, c = s[0], s[1].lo
            align = min(a.align, c) if lowbit(c) == c else 1
            r = a if a.hi < c else Range(0, c - 1, align)
        elif op is Op.MIN:
            a, b = s
            r = Range(min(a.lo, b.lo), min(a.hi, b.hi), min(a.align, b.align))
        elif op is Op.MAX:
            a, b = s
            r = Range(max(a.lo, b.lo), max(a.hi, b.hi), min(a.align, b.align))
        elif op is Op.SELP:
            a, b = s[:2]
            r = Range(min(a.lo, b.lo), max(a.hi, b.hi), min(a.align, b.align))
        elif op is Op.NEG:
            r = Range(-s[0].hi, -s[0].lo, s[0].align)
        else:
            return top(rt)
        return fit(r, rt)


@register_analysis
class RangeAnalysis(Analysis):
    name = "ranges"

    def run(self, function, am):
        return ValueRanges(function, am.get("defuse"), am.module)
