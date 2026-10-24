from core.ir import Binary, Immediate, Instruction, Op, PtxType, Register

from ..analysis.ranges import SIGNED, is_int
from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass

UNSIGNED_OF = {PtxType.S32: PtxType.U32, PtxType.S64: PtxType.U64, PtxType.U32: PtxType.U32,
               PtxType.U64: PtxType.U64}
BITS_OF = {PtxType.S32: PtxType.B32, PtxType.U32: PtxType.B32, PtxType.S64: PtxType.B64,
           PtxType.U64: PtxType.B64}


def magic_u32(d, hi):
    """(m, s) with x // d == (x * m) >> (32 + s) for every 0 <= x <= hi, m < 2^32; else None.

    m = ceil(2^(32+s) / d) overshoots 2^(32+s)/d by e/d (e = m*d - 2^(32+s)); writing
    x = q*d + r, the quotient stays q while r + x*e/2^(32+s) < d, i.e. while x*e < 2^(32+s).
    """
    for s in range(32):
        p = 1 << (32 + s)
        m = -(-p // d)
        if m >= 1 << 32:
            return None
        if (m * d - p) * hi < p:
            return m, s
    return None


@register_pass
class StrengthReduce(FunctionPass):
    """Replace expensive integer ops with cheaper ones, using range facts for signed ops.

        mul.lo x, 2^k                  ->  shl.b x, k
        div.s32 x, 2^k   (x >= 0)      ->  shr.u32 x, k
        rem.s32 x, 2^k   (x >= 0)      ->  and.b32 x, 2^k - 1
        div x, c         (x in [0,hi]) ->  mul.hi.u32 t, x, m; shr.u32 q, t, s
        rem x, c         (x in [0,hi]) ->  the quotient as above; mad.lo r, q, -c, x
        div/rem.s32 x, y (x, y >= 0)   ->  div/rem.u32 (no sign fix-ups)

    Signed div/rem by a power of two need a rounding fix-up for negative x, so they are only
    rewritten when the `ranges` analysis proves x >= 0 (true for thread and tile indices).
    PTX has no integer divide instruction at all: div/rem by a constant otherwise expands to
    a long emulation sequence.
    """

    name = "strength-reduce"
    preserves = CFG_ANALYSES

    def run(self, function, am):
        self.rng = am.get("ranges")
        self.function = function
        changed = False
        for block in function.blocks:
            out = []
            for inst in block.instructions:
                new = self.reduce(inst) if inst.guard is None and len(inst.dsts) == 1 else None
                if new is None:
                    out.append(inst)
                else:
                    out.extend(new)
                    changed = True
            block.instructions = out
        return changed

    def reduce(self, inst):
        t, op = inst.ptx_type, inst.op
        if t is None or not is_int(t) or len(inst.srcs) != 2 or t not in UNSIGNED_OF:
            return None
        d, (x, y) = inst.dsts[0], inst.srcs
        c = self.rng.constant(y, t) if isinstance(y, Immediate) else None
        if op is Op.MUL and inst.types == ("lo", t) and c is not None and c > 1 and c & (c - 1) == 0:
            return [Binary(Op.SHL, BITS_OF[t], d, x, Immediate(c.bit_length() - 1, PtxType.U32))]
        if op not in (Op.DIV, Op.REM) or not isinstance(x, Register):
            return None
        xr = self.rng.get(x, t)
        if not xr.nonneg:
            return None
        u = UNSIGNED_OF[t]
        if c is None:
            if t in SIGNED and self.rng.nonneg(y, t):
                return [Binary(op, u, d, x, y)]
            return None
        if c <= 1:
            return None
        if c & (c - 1) == 0:
            k = c.bit_length() - 1
            if op is Op.DIV:
                return [Binary(Op.SHR, u, d, x, Immediate(k, PtxType.U32))]
            return [Binary(Op.AND, BITS_OF[t], d, x, Immediate(c - 1, t))]
        if u is not PtxType.U32:
            return None
        magic = magic_u32(c, xr.hi)
        if magic is None:
            return None
        m, s = magic
        regs = self.function.regs
        q = d if op is Op.DIV and s == 0 else regs.new(PtxType.U32)
        seq = [Binary(Op.MUL, PtxType.U32, q, x, Immediate(m, PtxType.U32), mode="hi")]
        if s:
            q2 = d if op is Op.DIV else regs.new(PtxType.U32)
            seq.append(Binary(Op.SHR, PtxType.U32, q2, q, Immediate(s, PtxType.U32)))
            q = q2
        if op is Op.REM:
            seq.append(Instruction(Op.MAD, ("lo", PtxType.S32), dsts=[d], srcs=[q, Immediate(-c, PtxType.S32), x]))
        return seq
