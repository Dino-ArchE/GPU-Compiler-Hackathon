from core.ir import Address, Instruction, Mod, Op, Register, StateSpace, Symbol, Vector

from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass

SPACES = (StateSpace.GLOBAL, StateSpace.SHARED)
MAX_VECTOR_BYTES = 16          # PTX vectors: v4 of 32-bit or v2 of 64-bit elements


@register_pass
class VectorizeMemory(FunctionPass):
    """Merge scalar loads/stores of consecutive elements into one ld/st .v2/.v4.

        ld.global.f32 %f1, [%rd9];  ld.global.f32 %f2, [%rd9+4]; ...  (x4)
        ->  ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd9];

    Needs: one base register, offsets one element apart, and the first address aligned to the
    vector's size (proved by the `ranges` alignment facts, misaligned vector accesses fault).
    A load group is issued at its first load and a store group at its last store, so between
    them there may be no barrier or access to the same state space that could alias (loads from
    read-only buffers excepted), no read of a moved load's result, and no write of a moved
    store's value. Each element register must be single-def.

    A shared array is ours to declare: when its declared alignment is the only obstacle,
    the declaration is raised (`.align 4` -> `.align 16`) instead of giving up.
    """

    name = "vectorize-mem"
    preserves = CFG_ANALYSES

    def run(self, function, am):
        self.du, self.ranges, self.mem = am.get("defuse"), am.get("ranges"), am.get("memory")
        self.module = am.module
        changed = False
        for block in function.blocks:
            while self.vectorize_once(block):
                changed = True
        return changed

    def vectorize_once(self, block):
        groups = {}
        for k, inst in enumerate(block.instructions):
            key = self.candidate(inst)
            if key is not None:
                groups.setdefault(key, []).append((inst.srcs[0].offset, k, inst))
        for key, members in groups.items():
            by_offset = {}
            for off, k, inst in members:
                by_offset.setdefault(off, (k, inst))     # duplicates: keep the first
            for off in sorted(by_offset):
                for width in (4, 2):
                    run = self.try_group(block, key, by_offset, off, width)
                    if run:
                        self.rewrite(block, key, run)
                        return True
        return False

    def candidate(self, inst):
        """(op, types, base) grouping key for a scalar ld/st we may merge, else None."""
        if inst.op not in (Op.LD, Op.ST) or inst.guard is not None or inst.space not in SPACES:
            return None
        value = inst.dsts[0] if inst.op is Op.LD else inst.srcs[1]
        if isinstance(value, Vector):
            return None
        if inst.op is Op.LD and not (isinstance(value, Register) and self.du.single_def(value) is inst):
            return None
        base = inst.srcs[0].base
        if isinstance(base, Register) and not self.du.is_ssa(base):
            return None
        return (inst.op, inst.types, base)

    def try_group(self, block, key, by_offset, off, width):
        op, types, base = key
        nbytes = types[-1].nbytes
        if width * nbytes > MAX_VECTOR_BYTES:
            return None
        offsets = [off + i * nbytes for i in range(width)]
        if any(o not in by_offset for o in offsets):
            return None
        need = width * nbytes
        if self.ranges.address_align(Address(base, off)) < need and not self.raise_align(base, off, need):
            return None
        run = [by_offset[o] for o in offsets]
        positions = sorted(k for k, _ in run)
        lo, hi = positions[0], positions[-1]
        members = {id(inst) for _, inst in run}
        between = [i for i in block.instructions[lo:hi + 1] if id(i) not in members]
        space = types[0]
        for other in between:
            # cp.async (and its commit/wait) is a fence too: copies land at the wait, not the issue
            if other.op in (Op.BAR, Op.CP) or other.op in (Op.ST, Op.RED, Op.ATOM) and other.space is space:
                return None
            if other.op is Op.LD and other.space is space and op is Op.ST and not self.mem.is_invariant_load(other):
                return None
        insts = block.instructions
        for k, inst in run:
            if op is Op.LD:
                # this load now happens at lo: nothing in [lo, k) may read its old value
                if any(inst.dsts[0] in i.uses() for i in insts[lo:k]):
                    return None
            else:
                # this store now happens at hi: its value must not change in (k, hi]
                v = inst.srcs[1]
                if isinstance(v, Register) and any(v in i.defs() for i in insts[k + 1:hi + 1]):
                    return None
        return run

    def raise_align(self, base, off, need):
        """Re-declare shared array `base` with alignment `need` if that makes [base+off] aligned."""
        if self.module is None or not isinstance(base, Symbol) or off % need:
            return False
        for k, (sym, align, nbytes) in enumerate(self.module.shared):
            if sym is base:
                self.module.shared[k] = (sym, max(align, need), nbytes)
                self.ranges.symbol_align[sym] = max(align, need)
                return True
        return False

    def rewrite(self, block, key, run):
        op, types, base = key
        vtypes = (*types[:-1], Mod.vec(len(run)), types[-1])
        addr = run[0][1].srcs[0]
        values = Vector(inst.dsts[0] if op is Op.LD else inst.srcs[1] for _, inst in run)
        if op is Op.LD:
            new = Instruction(Op.LD, vtypes, dsts=[values], srcs=[addr])
            anchor = min(k for k, _ in run)
        else:
            new = Instruction(Op.ST, vtypes, srcs=[addr, values])
            anchor = max(k for k, _ in run)
        members = {id(inst) for _, inst in run}
        out = []
        for k, inst in enumerate(block.instructions):
            if k == anchor:
                out.append(new)
            elif id(inst) not in members:
                out.append(inst)
        block.instructions = out
