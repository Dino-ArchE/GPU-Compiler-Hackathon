"""Def-use chains: for each register, the instructions that write it and those that read it.

Our registers are not SSA: an accumulator has several defs and a guarded def (`@p mov`) may
leave the old value in place. `single_def` is the safe query for SSA-style rewrites: it returns
the def only when there is exactly one and it is unconditional, so the register holds that one
value everywhere it is read (the def dominates its uses in code the visitor emits).
"""

from core.ir import SpecialRegister

from ..pass_manager import Analysis, register_analysis


class DefUse:
    def __init__(self, function):
        self.defs = {}       # Register -> [Instruction]
        self.uses = {}       # Register -> [Instruction] (once per reading instruction)
        self.block_of = {}   # Instruction -> BasicBlock
        for block, inst in function.instructions():
            self.block_of[inst] = block
            for reg in inst.defs():
                self.defs.setdefault(reg, []).append(inst)
            for reg in dict.fromkeys(inst.uses()):
                self.uses.setdefault(reg, []).append(inst)

    def defs_of(self, reg):
        return self.defs.get(reg, [])

    def uses_of(self, reg):
        return self.uses.get(reg, [])

    def single_def(self, reg):
        """The one unconditional instruction defining `reg`, else None. Special registers: None."""
        if isinstance(reg, SpecialRegister):
            return None
        defs = self.defs.get(reg, [])
        if len(defs) == 1 and defs[0].guard is None:
            return defs[0]
        return None

    def is_ssa(self, reg):
        """Holds one value for the whole kernel: a special register or a single-def register."""
        return isinstance(reg, SpecialRegister) or self.single_def(reg) is not None

    def has_uses(self, reg):
        return bool(self.uses.get(reg))


@register_analysis
class DefUseAnalysis(Analysis):
    name = "defuse"

    def run(self, function, am):
        return DefUse(function)
