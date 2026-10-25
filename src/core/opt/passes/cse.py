import struct

from core.ir import Address, Immediate, Op, Register, Vector

from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass

COMMUTATIVE = {Op.ADD, Op.MUL, Op.MIN, Op.MAX, Op.AND, Op.OR, Op.XOR}


@register_pass
class CommonSubexpressionElimination(FunctionPass):
    """Reuse an earlier identical computation instead of repeating it (dominator-scoped).

    Walks the dominator tree with a table of available expressions; an instruction equal to one
    in a dominating position has its result replaced by the earlier register and is deleted.
    Sound for non-SSA code because only single-def results whose register sources each hold one
    value (single-def or special) take part: equal inputs then mean equal outputs everywhere.

    Loads take part only when their value cannot change: ld.param, and ld.global from a
    read-only buffer that nothing in the kernel writes (see the `memory` analysis).
    """

    name = "cse"
    preserves = CFG_ANALYSES

    def run(self, function, am):
        self.du = am.get("defuse")
        self.mem = am.get("memory")
        dom = am.get("dominators")
        self.removed = set()

        def walk(block, available):
            table = dict(available)
            for inst in block.instructions:
                key = self.key(inst)
                if key is None:
                    continue
                if key in table:
                    earlier, d = table[key], inst.dsts[0]
                    for user in self.du.uses_of(d):
                        user.replace_use(d, earlier)
                    self.removed.add(inst)
                else:
                    table[key] = inst.dsts[0]
            for child in dom.children[block]:
                walk(child, table)

        walk(dom.cfg.entry, {})
        if not self.removed:
            return False
        for block in function.blocks:
            block.instructions = [i for i in block.instructions if i not in self.removed]
        return True

    def key(self, inst):
        if inst.guard is not None or len(inst.dsts) != 1 or isinstance(inst.dsts[0], Vector):
            return None
        if inst.op is Op.LD:
            if not self.mem.is_invariant_load(inst):
                return None
        elif inst.has_side_effects or inst.reads_memory:
            return None
        d = inst.dsts[0]
        if self.du.single_def(d) is not inst:
            return None
        if not all(self.du.is_ssa(r) for r in inst.uses()):
            return None
        srcs = [_vkey(v) for v in inst.srcs]
        if inst.op in COMMUTATIVE and len(srcs) == 2:
            srcs.sort(key=repr)
        return (inst.op, inst.types, type(d), tuple(srcs))


def _vkey(v):
    if isinstance(v, Immediate):
        bits = struct.pack("<d", v.value) if isinstance(v.value, float) else v.value
        return ("imm", str(v.ptx_type), bits)
    if isinstance(v, Address):
        return ("addr", _vkey(v.base), v.offset)
    if isinstance(v, Vector):
        return ("vec", tuple(_vkey(e) for e in v.elems))
    if isinstance(v, Register):
        return ("reg", v.name)
    return ("sym", str(v))
