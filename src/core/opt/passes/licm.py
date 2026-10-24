from core.ir import Op

from ..pass_manager import FunctionPass, register_pass


@register_pass
class LoopInvariantCodeMotion(FunctionPass):
    """Hoist computations whose value is the same on every iteration into the loop preheader.

    An instruction is invariant when it is pure, unguarded, its result register is single-def,
    and each register it reads is either never written inside the loop or written only by an
    instruction already found invariant. Pure arithmetic may run speculatively (a loop that
    runs zero times now computes it once, harmlessly; integer division never traps on the
    GPU). A load is hoisted only if its value cannot change (`memory` analysis) and it sits in
    the header, which runs whenever the loop is entered, so no new address is ever touched.

    Loops are visited innermost first, so code can climb out of a nest one level per visit.
    A missing preheader is created in front of the header when the layout allows it.
    """

    name = "licm"

    def run(self, function, am):
        changed = False
        for _ in range(8):                       # new preheaders change the loop forest: recompute
            if not self.hoist_one_loop(function, am):
                break
            changed = True
            am.invalidate()
        return changed

    def hoist_one_loop(self, function, am):
        du, mem, cfg = am.get("defuse"), am.get("memory"), am.get("cfg")
        for loop in am.get("loops"):
            invariant = self.find_invariant(function, loop, du, mem)
            if not invariant:
                continue
            pre = loop.preheader(cfg) or self.make_preheader(function, loop, cfg)
            if pre is None:
                continue
            for block in loop.blocks:
                block.instructions = [i for i in block.instructions if i not in invariant]
            at = len(pre.instructions) - (1 if pre.terminator is not None else 0)
            pre.instructions[at:at] = invariant                # keeps their relative order
            return True
        return False

    def find_invariant(self, function, loop, du, mem):
        inside = [b for b in function.blocks if b in loop.blocks]       # layout order
        written = {r for b in inside for i in b.instructions for r in i.defs()}
        invariant, found = [], set()
        grew = True
        while grew:
            grew = False
            for block in inside:
                for inst in block.instructions:
                    if inst in found or not self.movable(inst, block, loop, du, mem):
                        continue
                    if all(r not in written or du.single_def(r) in found for r in inst.uses()):
                        found.add(inst)
                        invariant.append(inst)
                        grew = True
        # keep program order so each hoisted instruction follows the ones it reads
        order = {inst: k for k, (_, inst) in enumerate(function.instructions())}
        return sorted(invariant, key=order.__getitem__)

    @staticmethod
    def movable(inst, block, loop, du, mem):
        if inst.guard is not None or len(inst.dsts) != 1 or du.single_def(inst.dsts[0]) is not inst:
            return False
        if inst.op is Op.LD:
            return block is loop.header and mem.is_invariant_load(inst)
        return not inst.has_side_effects and not inst.reads_memory

    @staticmethod
    def make_preheader(function, loop, cfg):
        header = loop.header
        k = function.blocks.index(header)
        prev = function.blocks[k - 1] if k > 0 else None
        if prev is None or (prev in loop.blocks and cfg.falls_through(prev)):
            return None
        pre = function.new_block("PRE")
        for p in cfg.preds[header]:
            if p not in loop.blocks and p.terminator is not None and p.terminator.target is header:
                p.terminator.srcs = (pre,)
        function.blocks.insert(k, pre)
        return pre


