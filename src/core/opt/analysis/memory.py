"""Pointer provenance: which buffer an address points into, and which buffers get written.

An address's root is the kernel Param it was derived from (ld.param -> cvta -> add offsets,
through single-def registers) or the shared Symbol. Distinct params are assumed not to alias:
each is a separate tensor of the graph, and the reference kernels treat them that way.

A load from a root that nothing in the kernel writes (and that codegen marked read-only)
returns the same value every time, so CSE may merge such loads and ld-nc may route them
through the read-only cache.
"""

from core.ir import Address, Mod, Op, Param, PtxType, Register, StateSpace, Symbol

from ..pass_manager import Analysis, register_analysis

POINTER_ARITH = {Op.ADD, Op.SUB}


class PointerInfo:
    def __init__(self, function, defuse):
        self.function = function
        self.du = defuse
        self.memo = {}
        self.written = set()          # roots stored to (st, red, atom, cp)
        self.unknown_global_write = False
        for _, inst in function.instructions():
            if inst.writes_memory and inst.srcs and isinstance(inst.srcs[0], Address):
                root = self.root(inst.srcs[0])
                if root is not None:
                    self.written.add(root)
                elif inst.space is not StateSpace.SHARED:
                    self.unknown_global_write = True

    def root(self, value):
        """The Param or shared Symbol `value` (a register or an Address) points into, or None."""
        if isinstance(value, Address):
            value = value.base
        if isinstance(value, Symbol):            # Param is a Symbol too
            return value
        if not isinstance(value, Register):
            return None
        if value not in self.memo:
            self.memo[value] = None              # cycles resolve to unknown
            self.memo[value] = self._trace(value)
        return self.memo[value]

    def _trace(self, reg):
        inst = self.du.single_def(reg)
        if inst is None:
            return None
        if inst.op is Op.LD and inst.space is StateSpace.PARAM:
            base = inst.srcs[0].base
            return base if isinstance(base, Param) else None
        if inst.op in (Op.CVTA, Op.MOV):
            return self.root(inst.srcs[0])
        if inst.op in POINTER_ARITH and inst.ptx_type in (PtxType.S64, PtxType.U64, PtxType.B64):
            roots = [r for r in (self.root(v) for v in inst.srcs) if r is not None]
            return roots[0] if len(roots) == 1 else None
        if inst.op is Op.MAD and Mod.WIDE in inst.types:
            return self.root(inst.srcs[2])
        return None

    def is_invariant_load(self, inst):
        """ld whose value cannot change while the kernel runs."""
        if inst.op is not Op.LD:
            return False
        if inst.space is StateSpace.PARAM:
            return True
        if inst.space is not StateSpace.GLOBAL or self.unknown_global_write:
            return False
        root = self.root(inst.srcs[0])
        return isinstance(root, Param) and root in self.function.readonly and root not in self.written


@register_analysis
class MemoryAnalysis(Analysis):
    name = "memory"

    def run(self, function, am):
        return PointerInfo(function, am.get("defuse"))
