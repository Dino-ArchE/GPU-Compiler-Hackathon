from core.ir import Op

from ..pass_manager import FunctionPass, register_pass

MAX_INSTRUCTIONS = 8
# never predicated: barriers and warp collectives must run on every lane, cp.async groups are
# per thread but their wait/commit bookkeeping is easier to reason about unguarded
NOT_PREDICABLE = frozenset({Op.BAR, Op.SHFL, Op.MMA, Op.CP, Op.BRA, Op.RET})


@register_pass
class IfConvert(FunctionPass):
    """Replace a short skipped block with predicated instructions (a triangle in the CFG):

        A:  ...; @p bra J           A:  ...
        T:  st [x], v;          ->      @!p st [x], v;
        J:  ...                         ...

    T must be A's fall-through, reached only from A, fall through into J itself, and hold at
    most MAX_INSTRUCTIONS unguarded, predicable instructions that do not redefine p. This is
    the `if (lane == 0) smem[warp] = acc` and `if (tid == 0) out = acc` of every reduction:
    a branch per warp becomes one guarded store, and simplify-cfg can then merge A, T and J.

    Predicated defs are not kills, so the SSA-style passes treat T's results conservatively
    afterwards: this runs late, after the scalar clean-up.
    """

    name = "if-convert"

    def run(self, function, am):
        changed = False
        blocks = function.blocks
        cfg = am.get("cfg")
        for i in range(len(blocks) - 2):
            a, t, j = blocks[i], blocks[i + 1], blocks[i + 2]
            br = a.terminator
            if br is None or br.op is not Op.BRA or br.guard is None or br.target is not j:
                continue
            if cfg.preds[t] != [a] or t.terminator is not None or not t.instructions:
                continue
            if len(t.instructions) > MAX_INSTRUCTIONS:
                continue
            if any(x.guard is not None or x.op in NOT_PREDICABLE or br.guard in x.defs()
                   for x in t.instructions):
                continue
            for inst in t.instructions:
                inst.guard, inst.negate = br.guard, not br.negate      # run where bra is not taken
            a.instructions.pop()
            changed = True
        return changed
