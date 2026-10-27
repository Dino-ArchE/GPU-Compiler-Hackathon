from core.ir import Cmp, Immediate, Instruction, Mod, Op, Setp

from ..analysis.ranges import is_int
from ..pass_manager import CFG_ANALYSES, FunctionPass, register_pass


@register_pass
class RangeFold(FunctionPass):
    """Fold integer ops whose outcome the `ranges` analysis already knows.

        setp.ge.s32 p, gid, 4096   with gid in [0, 4095]   ->  p is a known false
        rem.s32 r, x, 32           with x in [0, 31]       ->  mov r, x
        div.s32 q, x, 32           with x in [0, 31]       ->  mov q, 0
        min/max                    with disjoint ranges    ->  mov of the smaller/larger
        any op                     with an exact result    ->  mov of the constant

    A decided setp is rewritten as a setp on two immediates so const-fold resolves the
    guards it controls (this is how a redundant thread guard disappears).
    """

    name = "range-fold"
    preserves = CFG_ANALYSES

    def run(self, function, am):
        rng = am.get("ranges")
        changed = False
        for block in function.blocks:
            for k, inst in enumerate(block.instructions):
                t = inst.ptx_type
                if inst.has_side_effects or len(inst.dsts) != 1 or t is None or not is_int(t):
                    continue
                if all(isinstance(v, Immediate) for v in inst.srcs):
                    continue                                  # const-fold's job
                new = self.fold(inst, t, rng)
                if new is not None:
                    new.guard, new.negate, new.comment = inst.guard, inst.negate, inst.comment
                    block.instructions[k] = new
                    changed = True
        return changed

    def fold(self, inst, t, rng):
        op, d = inst.op, inst.dsts[0]
        # an op-specific rule returns when it applies; otherwise control leaves the match and
        # the general rule below (a known-constant result) still gets its chance
        match op:
            case Op.SETP:
                a, b = (rng.get(v, t) for v in inst.srcs)
                res = _decide(Cmp(inst.types[0]), a, b)
                if res is None:
                    return None
                zero = Immediate(0, t)
                return Setp(Cmp.EQ if res else Cmp.NE, t, d, zero, zero)
            case Op.REM | Op.DIV if len(inst.srcs) == 2:
                x, c = rng.get(inst.srcs[0], t), rng.constant(inst.srcs[1], t)
                if c is not None and c > 0 and x.nonneg and x.hi < c:
                    return _mov(d, t, inst.srcs[0] if op is Op.REM else Immediate(0, t))
            case Op.MIN | Op.MAX:
                a, b = (rng.get(v, t) for v in inst.srcs)
                if a.hi <= b.lo:
                    return _mov(d, t, inst.srcs[0] if op is Op.MIN else inst.srcs[1])
                if b.hi <= a.lo:
                    return _mov(d, t, inst.srcs[1] if op is Op.MIN else inst.srcs[0])
        if op is not Op.MOV and Mod.WIDE not in inst.types:
            c = rng.constant(d, t)
            if c is not None:
                return _mov(d, t, Immediate(c, t))
        return None


def _decide(cmp, a, b):
    """The comparison's value if it is the same for every a in [a.lo, a.hi], b in [b.lo, b.hi]."""
    match cmp:
        case Cmp.LT:
            return True if a.hi < b.lo else False if a.lo >= b.hi else None
        case Cmp.LE:
            return True if a.hi <= b.lo else False if a.lo > b.hi else None
        case Cmp.GT:
            return _decide(Cmp.LT, b, a)
        case Cmp.GE:
            return _decide(Cmp.LE, b, a)
        case Cmp.EQ:
            return True if a.exact and b.exact and a.lo == b.lo else False if a.hi < b.lo or b.hi < a.lo else None
        case Cmp.NE:
            r = _decide(Cmp.EQ, a, b)
            return None if r is None else not r
        case _:
            return None


def _mov(d, t, v):
    return Instruction(Op.MOV, (t,), [d], [v])
