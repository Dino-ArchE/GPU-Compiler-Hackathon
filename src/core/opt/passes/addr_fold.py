from core.ir import Address, Binary, Immediate, Op, PtxType, Register, Symbol

from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass

ADDR_TYPES = (PtxType.S64, PtxType.U64, PtxType.B64)
MEMORY_OPS = (Op.LD, Op.ST, Op.RED, Op.ATOM)


@register_pass
class AddressFold(FunctionPass):
    """Move constant parts of an address into the [base+offset] immediate.

        add.s64 r, B, 16;            ld [r]            ->  ld [B+16]
        mov.u64 r, smem;             ld [r+8]          ->  ld [smem+8]
        add.s32 i, j, 3; mul.wide.s32 w, i, 4; add.s64 r, B, w; ld [r]
                                                       ->  mul.wide.s32 w', j, 4; add.s64 r', B, w';
                                                           ld [r'+12]

    The last form is what makes x[4*g+0], x[4*g+1], ... share one base register (after CSE),
    which ptxas folds into its addressing and vectorize-mem can merge into one v4 access.
    It assumes the s32 index add does not overflow, as the reference kernel's C `int` index
    math does. All registers involved must be single-def so moving their reads is safe.
    """

    name = "addr-fold"
    preserves = CFG_ANALYSES

    def run(self, function, am):
        self.du = am.get("defuse")
        self.function = function
        changed = False
        for block in function.blocks:
            for inst in list(block.instructions):
                if inst.op not in MEMORY_OPS or not inst.srcs or not isinstance(inst.srcs[0], Address):
                    continue
                addr = self.fold(block, inst, inst.srcs[0])
                if addr is not inst.srcs[0]:
                    inst.srcs = (addr, *inst.srcs[1:])
                    changed = True
        return changed

    def ssa(self, v):
        return isinstance(v, (Immediate, Symbol)) or self.du.is_ssa(v)

    def fold(self, block, inst, addr):
        while isinstance(addr.base, Register):
            d = self.du.single_def(addr.base)
            if d is None:
                return addr
            if d.op is Op.MOV and isinstance(d.srcs[0], Symbol):
                return Address(d.srcs[0], addr.offset)
            if d.op is not Op.ADD or d.ptx_type not in ADDR_TYPES:
                return addr
            base, w = d.srcs
            if isinstance(w, Immediate) and self.ssa(base):
                addr = Address(base, addr.offset + int(w.value))
                continue
            split = self.split_index(w)
            if split is None or not self.ssa(base):
                return addr
            j, k, c = split
            fn = self.function
            w2 = fn.regs.new(PtxType.S64)
            r2 = fn.regs.new(PtxType.S64)
            block.insert_before(inst, Binary(Op.MUL, PtxType.S32, w2, j, Immediate(k, PtxType.S32), mode="wide"))
            block.insert_before(inst, Binary(Op.ADD, PtxType.S64, r2, base, w2))
            return Address(r2, addr.offset + c * k)
        return addr

    def split_index(self, w):
        """w = mul.wide.s32 (add.s32 j, c), k  ->  (j, k, c), else None."""
        m = self.du.single_def(w) if isinstance(w, Register) else None
        if m is None or m.op is not Op.MUL or m.types != ("wide", PtxType.S32):
            return None
        i, k = m.srcs
        a = self.du.single_def(i) if isinstance(i, Register) else None
        if not isinstance(k, Immediate) or a is None or a.op is not Op.ADD or a.ptx_type is not PtxType.S32:
            return None
        j, c = a.srcs
        if not isinstance(c, Immediate) or not self.ssa(j):
            return None
        return j, int(k.value), int(c.value)
