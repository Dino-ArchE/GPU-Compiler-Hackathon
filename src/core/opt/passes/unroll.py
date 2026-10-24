from core.ir import Immediate, Instruction, Op, PtxType, Register

from ..analysis.ranges import bounds
from ..fold import compare
from ..pass_manager import FunctionPass, register_pass

MAX_TRIPS = 4096          # longest loop we bother to count
MAX_SIZE = 1024           # instructions the unrolled loop may grow to
MAX_FULL_TRIPS = 64       # loops up to this many iterations are unrolled completely
FACTORS = (8, 4, 2)


@register_pass
class Unroll(FunctionPass):
    """Unroll counted loops whose trip count is the same for every thread.

    The shape codegen emits for `for i in start..stop:step` (simplify-cfg may have merged H..L):

        H:  ...; setp p, i, stop; @p bra E      the only exit, taken when the count runs out
            ...                                 H..L contiguous in the layout, one latch L
        L:  ...; add i, i, step; bra H

    The trip count T comes from the range of `start` (launch geometry: a thread-strided loop
    from tid with a stop that is a multiple of the block runs the same T everywhere). With
    T <= MAX_FULL_TRIPS the loop becomes T straight copies and its compares disappear;
    otherwise it is unrolled by a factor U dividing T, testing only once per U iterations.

    Each copy reads the induction variable as its own single-def register i + k*step, and the
    registers that live only within one iteration are renamed per copy, so the scalar passes
    can fold each copy's index math (addr-fold turns them into immediate offsets).
    Accumulators and other values carried across iterations keep their registers.
    """

    name = "unroll"

    def run(self, function, am):
        self.done = set()              # headers already unrolled: never unroll a loop twice
        changed = False
        for _ in range(16):            # one loop per round: unrolling invalidates the loop forest
            if not self.unroll_one(function, am):
                break
            am.invalidate()
            changed = True
        return changed

    def unroll_one(self, function, am):
        cfg, loops = am.get("cfg"), am.get("loops")
        for loop in loops:                                   # innermost first
            if loop.children or loop.header in self.done:
                continue
            shape = self.match(function, loop, cfg, am)
            if shape is None:
                continue
            trips, size = shape["trips"], shape["size"]
            if 1 <= trips <= MAX_FULL_TRIPS and trips * size <= MAX_SIZE and shape["can_drop_test"]:
                factor = trips
            else:
                factor = next((u for u in FACTORS if trips % u == 0 and u * size <= MAX_SIZE), None)
            if factor is None:
                continue
            self.expand(function, shape, factor, am.get("liveness"))
            self.done.add(loop.header)
            return True
        return False

    # ------------------------------------------------------------------ recognize the loop

    def match(self, function, loop, cfg, am):
        du, ranges, dom = am.get("defuse"), am.get("ranges"), am.get("dominators")
        header = loop.header
        if len(loop.latches) != 1:
            return None
        latch = loop.latches[0]
        k = function.blocks.index(header)
        body = function.blocks[k:k + len(loop.blocks)]
        if set(body) != loop.blocks or body[-1] is not latch:
            return None
        # exit: a guarded bra out of the header; every other edge stays inside the loop
        test = header.terminator
        if test is None or test.op is not Op.BRA or test.guard is None or test.target in loop.blocks:
            return None
        exit_ = test.target
        if set(loop.exits(cfg)) != {exit_}:
            return None
        back = latch.terminator
        if back.op is not Op.BRA or back.guard is not None or back.target is not header:
            return None
        # the compare: setp.cmp p, i, stop (either order), stop a constant
        setp = du.single_def(test.guard)
        if setp is None or setp.op is not Op.SETP or du.block_of[setp] is not header:
            return None
        cmp, t = setp.types[0], setp.ptx_type
        a, b = setp.srcs
        if isinstance(a, Register) and ranges.constant(b, t) is not None:
            iv, stop, iv_first = a, ranges.constant(b, t), True
        elif isinstance(b, Register) and ranges.constant(a, t) is not None:
            iv, stop, iv_first = b, ranges.constant(a, t), False
        else:
            return None
        # the induction variable: one `add iv, iv, step` in the latch, one def before the loop
        inside = [d for d in du.defs_of(iv) if du.block_of[d] in loop.blocks]
        outside = [d for d in du.defs_of(iv) if du.block_of[d] not in loop.blocks]
        if len(inside) != 1 or len(outside) != 1 or outside[0].guard is not None:
            return None
        inc, init_def = inside[0], outside[0]
        if inc.op is not Op.ADD or inc.guard is not None or du.block_of[inc] is not latch \
                or inc.ptx_type != t or inc.srcs.count(iv) != 1:
            return None
        step = inc.srcs[1] if inc.srcs[0] is iv else inc.srcs[0]
        if not isinstance(step, Immediate) or isinstance(step.value, float) or int(step.value) == 0:
            return None
        if header is latch and header.instructions.index(inc) < header.instructions.index(setp):
            return None                        # the increment must come after the exit test
        if not dom.dominates(du.block_of[init_def], header):
            return None
        init = ranges.range_of(init_def)
        if init is None:
            return None

        def exits(i):
            return compare(cmp, t, *((i, stop) if iv_first else (stop, i))) != test.negate
        trips = self.trip_count(init, int(step.value), exits, t)
        if trips is None:
            return None
        # a full unroll drops the header's last run (the failing test): only if it does nothing
        pre_test = header.instructions[:header.instructions.index(test)]
        live = am.get("liveness")
        can_drop = not any(i.has_side_effects or set(i.defs()) & live.live_in[exit_] for i in pre_test)
        size = sum(len(blk.instructions) for blk in body) - 2
        return dict(header=header, latch=latch, body=body, exit=exit_, test=test, iv=iv, inc=inc,
                    step=int(step.value), t=t, trips=trips, size=max(size, 1),
                    can_drop_test=can_drop)

    @staticmethod
    def trip_count(init, step, exits, t):
        """The iteration count shared by every start value in `init`, else None. Counted
        exactly for each start (a range of a few thousand thread ids, tens of trips)."""
        if init.hi - init.lo > MAX_TRIPS:
            return None
        lo, hi = bounds(t)
        found = None
        first = -(-init.lo // init.align) * init.align          # values are multiples of align
        for start in range(first, init.hi + 1, init.align):
            i, n = start, 0
            while not exits(i):
                i, n = i + step, n + 1
                if n > MAX_TRIPS or not lo <= i <= hi:
                    return None                # too long, or would wrap: leave it alone
            if found is not None and n != found:
                return None
            found = n
        return found

    # ------------------------------------------------------------------ rewrite

    def expand(self, function, s, factor, live):
        body, header, latch, iv, t = s["body"], s["header"], s["latch"], s["iv"], s["t"]
        full = factor == s["trips"]
        defined = {r for blk in body for inst in blk.instructions for r in inst.defs()}
        local = defined - live.live_in[header] - live.live_in[s["exit"]]
        k0 = function.blocks.index(header)

        original = {blk: list(blk.instructions) for blk in body}    # copy 0 reuses the blocks
        back = latch.terminator
        copies, cur = [], iv
        for c in range(factor):
            last = c == factor - 1
            names = {blk: (blk if c == 0 else function.new_block(blk.name.split("_")[0] + "_U"))
                     for blk in body}
            renamed = {r: function.regs.new(r.PTX_TYPE) for r in local}
            for blk in body:
                insts = []
                for inst in original[blk]:
                    if inst is s["test"] and (full or c > 0):
                        continue
                    if inst is s["inc"]:
                        if last and full and iv not in live.live_in[s["exit"]]:
                            continue            # the final count is dead: iv stays single-def
                        nxt = iv if last else function.regs.new(iv.PTX_TYPE)
                        insts.append(Instruction(Op.ADD, (t,), [nxt],
                                                 [iv, Immediate(s["step"] * (c + 1), t)],
                                                 comment=inst.comment))
                        cur = nxt
                        continue
                    if inst is back:
                        if last:
                            target = s["exit"] if full else header
                            insts.append(Instruction(Op.BRA, (), srcs=[target]))
                        continue                # copies fall through into the next one
                    new = inst.copy()
                    if new.op is Op.BRA and new.target in names:
                        new.srcs = (names[new.target],)
                    if cur is not iv:
                        new.replace_use(iv, cur)
                    for old, reg in renamed.items():
                        new.replace_use(old, reg)
                        new.replace_def(old, reg)
                    insts.append(new)
                names[blk].instructions = insts
            copies.extend(names[blk] for blk in body)
        function.blocks[k0:k0 + len(body)] = copies
