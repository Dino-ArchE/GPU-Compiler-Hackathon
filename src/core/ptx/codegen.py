import math

from core.ir import (
    CTAID_X, NTID_X, TID_X, Address, Binary, Builder, Cmp, Function, Immediate, Instruction, Load,
    Module, Mov, Op, PtxType, Register, StateSpace, Store, Vector,
)

from .ops import lower_elementwise
from .symbol import Buffer, SymbolTable


# Smem.dtype uses C spellings
SMEM_TYPES = {"float": PtxType.F32, "half": PtxType.F16, "double": PtxType.F64, "int": PtxType.S32}

# Index arithmetic (Load/Write indices, Cond) is 32-bit signed, like the reference .cu's `int`.
INDEX_TYPE = PtxType.S32
INT32_MAX = 2**31 - 1

# BinaryExpr operators on indices
# indices are non-negative, so floor division // is the same as truncating /
INDEX_ARITH = {"+": Op.ADD, "-": Op.SUB, "*": Op.MUL, "/": Op.DIV, "//": Op.DIV, "%": Op.REM, "^": Op.XOR}
INDEX_LOGIC = {"&&": Op.AND, "||": Op.OR}
INDEX_CMP = {"<": Cmp.LT, "<=": Cmp.LE, ">": Cmp.GT, ">=": Cmp.GE, "==": Cmp.EQ, "!=": Cmp.NE}


# statements that every thread of the block must reach: a guard may not branch around them
# shfl.sync operands: all 32 lanes take part; clamp 0x1f = whole warp
FULL_MASK = Immediate(0xFFFFFFFF, PtxType.B32)
WARP_CLAMP = Immediate(0x1F, PtxType.B32)


def _children(stmt):
    """Nested statement lists of a node (loop/Tile bodies, both arms of a Cond)."""
    for field in ("body", "else_body"):
        child = getattr(stmt, field, None)
        if child:
            yield child


def _walk(body):
    """Every statement in `body`, nested ones included, in order."""
    for stmt in body:
        yield stmt
        for child in _children(stmt):
            yield from _walk(child)


class _Sum:
    """An index sum we build ourselves (CpAsyncCopy smem_index + lane_index); visit__Sum lowers it."""

    def __init__(self, left, right):
        self.left, self.right = left, right


def _compare(op, a, b):
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[op]


def _fold(op, a, b):
    """Constant-fold two int literals with C semantics (/ and % truncate toward zero)."""
    if op in ("/", "//"):
        return int(a / b)
    if op == "^":
        return a ^ b
    if op == "%":
        return a - int(a / b) * b
    return {"+": a + b, "-": a - b, "*": a * b}[op]


class CodeGen:
    """Lowers one kernel's Kernel IR into our IR: a Module holding one Function.

    Emmy's nodes have no accept(), so visit() dispatches on the node's class name
    (Tile -> visit_Tile), the way Python's ast.NodeVisitor does.

    Statement visitors (Tile, Load, Assign, Write, ...) emit instructions and return nothing.
    Expression visitors (Var, Literal, BinaryExpr) return the Value holding their result
    (a Register, or an Immediate for literals) so the parent can use it as an operand.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.module = Module()
        self.function = self.module.add_function(Function(kernel.name, [a.name for a in kernel.args]))
        # facts the optimizer may specialize on: the launch is fixed and inputs are never written
        self.function.launch_grid, self.function.launch_block = kernel.grid, kernel.block
        self.function.readonly = {self.function.param(a.name) for a in kernel.args if not a.is_output}
        self.builder = Builder(self.function)
        self.sym_table = SymbolTable()
        self.accum_ready = set()   # ids of Accum nodes whose identity init is already emitted
        self.explicit_init = set() # accumulators an Init node sets up: never re-initialize them
        self.smem_pool = None      # extern .shared pool, when the launch passes dynamic smem
        self.smem_used = 0         # bytes of the pool handed out so far
        # where guarded-off threads jump; placed last so `ret` is the kernel's only exit
        self.exit = self.builder.new_block("EXIT")

    def gen(self):
        self.prologue()
        self.visit(self.kernel.body)
        self.builder.set_block(self.exit)
        self.builder.inst(Op.RET)
        return self.module

    def __call__(self):
        return self.gen()

    def visit(self, node):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise NotImplementedError(f"no visitor for {type(node).__name__}")
        return method(node)

    def visit_Body(self, body):
        for stmt in body:
            self.visit(stmt)

    def prologue(self):
        """Not driven by an IR node: turn each param into a global base address, in arg_order."""
        for arg in self.kernel.args:
            generic = self.builder.load_param(self.function.param(arg.name))
            base = self.builder.cvta_to_global(generic)
            buffer = Buffer(StateSpace.GLOBAL, base, arg.shape, PtxType.from_meta(arg.dtype))
            self.sym_table.define_buffer(arg.name, buffer)

    # ------------------------------------------------------------------ statements

    def visit_Tile(self, tile):
        """One thread per point of the axes' iteration space.

        1. gid = ctaid.x * ntid.x + tid.x; with an aux (producer) band past the cooperative
           threads, gid = ctaid.x * block_threads + tid.x % block_threads instead.
        2. guard, only if some launched thread has no cell (N < grid*block, or an aux band):
               @(gid >= N) bra EXIT    -- gen() places EXIT (holding `ret`) last.
           Barriers inside stay correct: exited threads no longer count at bar.sync, and
           shuffles use the active-lane mask (the reference .cu does the same).
        3. decode gid into the axes (decode_axes) and define_scalar each axis name.
        4. visit the body.
        """
        b = self.builder
        ctaid, ntid, tid = (b.mov(PtxType.U32, sreg) for sreg in (CTAID_X, NTID_X, TID_X))
        if tile.aux_threads:
            cells = Immediate(tile.block_threads, INDEX_TYPE)
            gid = b.mad(INDEX_TYPE, ctaid, cells, b.binary(Op.REM, PtxType.U32, tid, cells))
        else:
            gid = b.mad(INDEX_TYPE, ctaid, ntid, tid)

        # implicit vars used by Cond/TreeHalve in reductions
        self.sym_table.define_scalar("lane", b.binary(Op.AND, PtxType.B32, tid, Immediate(31, PtxType.B32)))
        self.sym_table.define_scalar("warp", b.binary(Op.SHR, PtxType.U32, tid, Immediate(5, PtxType.U32)))

        extents = [axis.extent.as_static() for axis in tile.axes]
        n = math.prod(extents)
        if n > INT32_MAX:
            raise NotImplementedError(f"Tile of {n} cells needs 64-bit indexing")
        if tile.aux_threads or n < math.prod(self.kernel.grid) * math.prod(self.kernel.block):
            outside = b.setp(Cmp.GE, INDEX_TYPE, gid, Immediate(n, INDEX_TYPE))
            b.bra(self.exit, guard=outside)

        for axis, value in zip(tile.axes, self.decode_axes(tile, gid, extents)):
            self.sym_table.define_scalar(axis.name, value)
        self.visit(tile.body)

    def decode_axes(self, tile, gid, extents):
        """gid -> one index per axis, row-major: axis i = (gid / stride_i) % extent_i, where
        stride_i is the product of the extents to its right (the first axis needs no %).

        Grouped rasterization (raster_group G over the adjacent raster_axes m, n) remaps those
        two axes so G block-tiles of the grouped axis run fastest; consecutive CTAs then reuse
        the same operand slab from L2. With the grouped axis a (extent Ea), the other b (Eb)
        and sub = the (m, n) part of gid:  grp = sub / (G*Eb),  size = min(G, Ea - grp*G),
            a = grp*G + (sub % (G*Eb)) % size,   b = (sub % (G*Eb)) / size
        """
        strides = [math.prod(extents[i + 1:]) for i in range(len(extents))]
        values = []
        for i, (stride, ext) in enumerate(zip(strides, extents)):
            v = self.div_index(gid, stride)
            values.append(v if i == 0 else self.rem_index(v, ext))

        raster = self.raster(tile, extents)
        if raster is not None:
            gi, oi, g = raster
            ea, eb = extents[gi], extents[oi]
            mi, ni = min(gi, oi), max(gi, oi)
            sub = self.div_index(gid, strides[ni])
            if mi != 0:
                sub = self.rem_index(sub, ea * eb)
            grp = self.div_index(sub, g * eb)
            within = self.rem_index(sub, g * eb)
            if ea % g == 0:
                size = Immediate(g, INDEX_TYPE)
            else:
                left = self.builder.binary(Op.SUB, INDEX_TYPE, Immediate(ea, INDEX_TYPE), self.mul_index(grp, g))
                size = self.builder.binary(Op.MIN, INDEX_TYPE, left, Immediate(g, INDEX_TYPE))
            b = self.builder
            values[gi] = self.add_index(self.mul_index(grp, g), b.binary(Op.REM, INDEX_TYPE, within, size))
            values[oi] = b.binary(Op.DIV, INDEX_TYPE, within, size)
        return values

    @staticmethod
    def raster(tile, extents):
        """(grouped axis, other axis, G) when grouped rasterization applies, else None."""
        if not (tile.raster_group and tile.raster_axes):
            return None
        names = [axis.name for axis in tile.axes]
        m, n = tile.raster_axes
        if m not in names or n not in names or names.index(n) != names.index(m) + 1:
            return None
        mi, ni = names.index(m), names.index(n)
        gi, oi = (mi, ni) if tile.raster_orient == "m" else (ni, mi)
        g = min(tile.raster_group, extents[gi])
        return (gi, oi, g) if extents[gi] > 1 and g > 1 else None

    def visit_Load(self, load):
        """names = load input[index]: buffer element(s) -> scalar register(s).

        len(names) == 1 -> ld.<space>.<type>;  2 or 4 -> ld.<space>.v2|v4.<type> into a Vector,
        consecutive elements starting at `index` (p* only has single-name loads).
        Each name in load.names is defined as a scalar.
        """
        if len(load.names) not in (1, 2, 4):
            raise NotImplementedError(f"Load of {len(load.names)} values from {load.input}")
        buffer = self.sym_table.buffer(load.input)
        addr = self.address(buffer, load.index)
        dsts = [self.builder.reg(buffer.elem_type) for _ in load.names]
        dst = dsts[0] if len(dsts) == 1 else Vector(dsts)
        self.builder.append(Load(buffer.space, buffer.elem_type, dst, addr))
        for name, reg in zip(load.names, dsts):
            self.sym_table.define_scalar(name, reg)

    def visit_Assign(self, assign):
        """f32 name = op(args...): elementwise math on scalars.

        assign.op is an ElementwiseImpl; dispatch on assign.op.name through a table
        (add -> add.f32, multiply -> mul.f32, relu -> max.f32 x, 0, ...). Result type from
        assign.dtype. Defines assign.name as a scalar.
        """
        self.assign(assign, assign.dtype)

    def assign(self, assign, dtype):
        """Shared by Assign and combine_states, whose Assigns carry dtype=None (use the state's)."""
        args = [self.operand(arg) for arg in assign.args]
        ptx_type = PtxType.from_meta((assign.dtype or dtype).name)
        result = lower_elementwise(self.builder, assign.op.name, ptx_type, args)
        self.sym_table.define_scalar(assign.name, result)

    def visit_Write(self, write):
        """output[index] = values: scalar register(s) -> buffer element(s).

        len(values) == 1 -> st.<space>.<type>;  2 or 4 -> st.<space>.v2|v4.<type> {..} at `index`
        (p00/p01 store v2, p02/p03 store v4). write.atomic -> red.global.add (reductions, later).
        """
        # the vector form assumes values are consecutive elements; PTX only has v2/v4
        if len(write.values) not in (1, 2, 4):
            raise NotImplementedError(f"Write of {len(write.values)} values to {write.output}")
        if write.swizzle != "NONE":
            raise NotImplementedError(f"swizzled Write ({write.swizzle}) to {write.output}")
        buffer = self.sym_table.buffer(write.output)
        addr = self.address(buffer, write.index)
        srcs = [self.operand(v) for v in write.values]
        if write.atomic:
            # partial results from many blocks: red.add into a buffer meta.json zeroes first
            if len(srcs) != 1:
                raise NotImplementedError(f"vector atomic Write to {write.output}")
            self.builder.append(Instruction(Op.RED, (buffer.space, "add", buffer.elem_type), srcs=[addr, srcs[0]]))
            return
        src = srcs[0] if len(srcs) == 1 else Vector(srcs)
        self.builder.store(buffer.space, buffer.elem_type, addr, src)

    def visit_StridedLoop(self, loop):
        """for axis = start; axis < end; axis += step  (a thread's slice of a reduction)."""
        end = loop.end if loop.end is not None else Immediate(loop.axis.extent.as_static(), INDEX_TYPE)
        self.loop(loop, self.visit(loop.start), self.index_value(end), self.visit(loop.step))

    def visit_Loop(self, loop):
        """for axis = 0; axis < extent; axis++  (unroll is left to ptxas)."""
        one = Immediate(1, INDEX_TYPE)
        self.loop(loop, Immediate(0, INDEX_TYPE), Immediate(loop.axis.extent.as_static(), INDEX_TYPE), one)

    def loop(self, loop, start, end, step):
        """Emit a counted loop over loop.axis:

                axis = start; init accumulators of this loop
            HEAD:   @(axis >= end) bra END
                    body; write back rebound names
                    axis += step; bra HEAD
            END:
        """
        b = self.builder
        self.init_accumulators(loop)
        axis = b.mov(INDEX_TYPE, start)   # own copy: the loop updates it in place
        self.sym_table.define_scalar(loop.axis.name, axis)
        head, end_block = b.new_block("LOOP"), b.new_block("LOOP_END")

        b.set_block(head)
        done = b.setp(Cmp.GE, INDEX_TYPE, axis, end)
        b.bra(end_block, guard=done)
        self.scoped(loop.body)
        b.append(Binary(Op.ADD, INDEX_TYPE, axis, axis, step))
        b.bra(head)
        b.set_block(end_block)

    def visit_Init(self, init):
        """f32 name = identity, set once where it stands (matmul accumulators, which must
        survive the outer k loop although their Accum names only the inner one)."""
        ptx_type = PtxType.from_meta(init.dtype.name)
        self.sym_table.define_scalar(init.name, self.builder.mov(ptx_type, Immediate(init.identity, ptx_type)))
        self.explicit_init.add(init.name)

    def visit_Let(self, let):
        """f32 name = value: a named constant (or expression)."""
        ptx_type = PtxType.from_meta(let.dtype.name)
        value = Immediate(let.value.value, ptx_type) if type(let.value).__name__ == "Literal" else self.visit(let.value)
        self.sym_table.define_scalar(let.name, self.builder.mov(ptx_type, value))

    def visit_CpAsyncCopy(self, copy):
        """cp.async: copy nbytes global -> shared without passing through registers.
        The shared index is smem_index + lane_index, element-wise (lane_index may be None)."""
        if copy.swizzle != "NONE":
            raise NotImplementedError(f"swizzled cp.async ({copy.swizzle}) into {copy.smem}")
        smem = self.sym_table.buffer(copy.smem)
        index = copy.smem_index
        if copy.lane_index:
            index = [_Sum(a, b) for a, b in zip(index, copy.lane_index)]
        dst = self.address(smem, index)
        src = self.address(self.sym_table.buffer(copy.src), copy.src_index)
        # .cg (L2 only) exists for 16-byte copies; smaller ones must use .ca
        level = "cg" if copy.nbytes == 16 else "ca"
        self.builder.cp_async(dst, src, copy.nbytes, cache=level)

    def visit_CpAsyncCommit(self, commit):
        self.builder.cp_async_commit()

    def visit_CpAsyncWait(self, wait):
        """Wait until at most `group` committed copy groups are still in flight."""
        self.builder.cp_async_wait(wait.group)

    def init_accumulators(self, loop):
        """Accum identity goes before the loop its `axes` names, not at the Accum statement:
        online softmax reads the accumulator before its Accum line."""
        for stmt in _walk(loop.body):
            if (type(stmt).__name__ == "Accum" and loop.axis.name in stmt.axes
                    and id(stmt) not in self.accum_ready and stmt.name not in self.explicit_init):
                self.accum_ready.add(id(stmt))
                self.init_accumulator(stmt)

    def init_accumulator(self, accum):
        ptx_type = PtxType.from_meta(accum.dtype.name)
        acc = self.builder.mov(ptx_type, Immediate(accum.op.identity, ptx_type))
        self.sym_table.define_scalar(accum.name, acc)

    def visit_Accum(self, accum):
        """name <- op(base or name, value). Like an Assign to an existing name; the loop's
        write-back (scoped) moves the result into the register the next iteration reads."""
        if accum.name not in self.sym_table.scalars:
            self.init_accumulator(accum)    # no enclosing loop named it
        first = accum.base if accum.base is not None else accum.name
        args = [self.operand(first), self.operand(accum.value)]
        ptx_type = PtxType.from_meta(accum.dtype.name)
        self.sym_table.define_scalar(accum.name, lower_elementwise(self.builder, accum.op.name, ptx_type, args))

    def visit_WarpShuffle(self, shuffle):
        """XOR butterfly over `length` lanes: each round, read every state from lane ^ offset
        into state_b, then run combine_states. Afterwards every lane holds the result."""
        # the lanes still running: a guarded partial warp must not name exited lanes
        mask = self.builder.reg(PtxType.B32)
        self.builder.append(Instruction(Op.ACTIVEMASK, (PtxType.B32,), dsts=[mask]))
        offset = shuffle.length // 2
        while offset >= 1:
            for state, other in zip(shuffle.state, shuffle.state_b):
                src = self.sym_table.scalar(state)
                dst = self.builder.reg(type(src).PTX_TYPE)   # same register kind as the state
                srcs = [src, Immediate(offset, PtxType.U32), WARP_CLAMP, mask]
                self.builder.append(Instruction(Op.SHFL, ("sync", "bfly", PtxType.B32), dsts=[dst], srcs=srcs))
                self.sym_table.define_scalar(other, dst)
            for combine in shuffle.combine_states:
                self.assign(combine, shuffle.dtype)
            offset //= 2

    def visit_TreeHalve(self, halve):
        """Shared-memory tree over `length` slots (one per warp), then broadcast slot 0:

            for s = length/2 .. 1:  if tid < s: buf[tid] = combine(buf[tid], buf[tid + s])
                                    bar.sync
            state = buf[0]

        When the tree is indexed by `warp`, all 32 lanes of a warp share one slot: only lane 0
        stores, so each slot has one writer (a read-modify-write by every lane is a data race
        that only works while the warp happens to run in lockstep). The other lanes' sums are
        dropped anyway, since everyone reloads the root afterwards.
        """
        b = self.builder
        writer = None
        if halve.tid_var == "warp":
            writer = b.setp(Cmp.EQ, INDEX_TYPE, self.sym_table.scalar("lane"), Immediate(0, INDEX_TYPE))
        tid = self.sym_table.scalar(halve.tid_var)
        bufs = [self.sym_table.buffer(name) for name in halve.bufs]
        # segmented layout: slot(t) = t*scale + inner_var, root = inner_var (flat: t and 0)
        if halve.inner is not None:
            inner_name, scale = halve.inner
            inner = self.sym_table.scalar(inner_name)
            slot = lambda t: self.add_index(self.mul_index(t, scale), inner)  # noqa: E731
            root = inner
        else:
            slot = lambda t: t  # noqa: E731
            root = Immediate(0, INDEX_TYPE)
        s = halve.length // 2
        while s >= 1:
            skip = b.new_block("HALVE")
            active = b.setp(Cmp.LT, INDEX_TYPE, tid, Immediate(s, INDEX_TYPE))
            b.bra(skip, guard=active, negate=True)
            partner = b.add(INDEX_TYPE, tid, Immediate(s, INDEX_TYPE))
            for buf, state, other in zip(bufs, halve.state, halve.state_b):
                self.sym_table.define_scalar(state, b.load(buf.space, buf.elem_type, self.element(buf, slot(tid))))
                self.sym_table.define_scalar(other, b.load(buf.space, buf.elem_type, self.element(buf, slot(partner))))
            for combine in halve.combine_states:
                self.assign(combine, halve.dtype)
            for buf, state in zip(bufs, halve.state):
                addr, value = self.element(buf, slot(tid)), self.sym_table.scalar(state)
                b.append(Store(buf.space, buf.elem_type, addr, value, guard=writer))
            b.set_block(skip)
            self.barrier(halve.barrier_id, halve.barrier_count)
            s //= 2
        # every thread reads the total; this also replaces names only set inside the branches
        for buf, state in zip(bufs, halve.state):
            self.sym_table.define_scalar(state, b.load(buf.space, buf.elem_type, self.element(buf, root)))

    def visit_Smem(self, smem):
        """Declares a shared array (printed at module scope); emits no instructions."""
        elem_type = SMEM_TYPES[smem.dtype]
        nbytes = math.prod(smem.extents) * elem_type.nbytes
        align = smem.align or elem_type.nbytes
        if not self.kernel.smem_bytes:
            base = self.module.add_shared(smem.name, align, nbytes)
        else:
            base = self.pool_slice(smem.name, align, nbytes)
        self.sym_table.define_buffer(smem.name, Buffer(StateSpace.SHARED, base, smem.extents, elem_type))

    def pool_slice(self, name, align, nbytes):
        """Dynamic shared memory: static .shared is capped at 48 KB, so when meta.json passes
        smem_bytes every Smem array is carved out of one .extern pool at an aligned offset.
        Returns a register holding the array's shared address."""
        if self.smem_pool is None:
            self.smem_pool = self.module.add_shared("smem_pool", 16)
        offset = -(-self.smem_used // align) * align
        self.smem_used = offset + nbytes
        if self.smem_used > self.kernel.smem_bytes:
            raise ValueError(f"Smem {name} needs {self.smem_used} bytes; launch gives {self.kernel.smem_bytes}")
        base = self.builder.mov(PtxType.U64, self.smem_pool)
        return self.builder.add(PtxType.U64, base, Immediate(offset, PtxType.U64)) if offset else base

    def visit_Cond(self, cond):
        """if cond: body [else: else_body]. Threads that fail branch around the body."""
        b = self.builder
        taken = self.visit(cond.cond)
        if isinstance(taken, Immediate):   # folded at compile time
            self.scoped(cond.body if taken.value else cond.else_body)
            return
        end = b.new_block("ENDIF")
        other = b.new_block("ELSE") if cond.else_body else end
        b.bra(other, guard=taken, negate=True)
        self.scoped(cond.body)
        if cond.else_body:
            b.bra(end)
            b.set_block(other)
            self.scoped(cond.else_body)
        b.set_block(end)

    def visit_Sync(self, sync):
        """bar.sync 0 (whole CTA), bar.warp.sync (one warp), or a named barrier over `count`."""
        if sync.warp:
            self.builder.append(Instruction(Op.BAR, ("warp", "sync"), srcs=[FULL_MASK]))
        else:
            self.barrier(sync.barrier_id, sync.count)

    def barrier(self, barrier_id, count=None):
        if barrier_id == 0:
            self.builder.sync(0)
            return
        if count is None:
            raise ValueError(f"named barrier {barrier_id} needs a thread count")
        ids = [Immediate(barrier_id, PtxType.U32), Immediate(count, PtxType.U32)]
        self.builder.append(Instruction(Op.BAR, ("sync",), srcs=ids))

    def scoped(self, body):
        """Visit a loop body or a Cond arm. A name that existed before and was rebound inside
        (Accum, a combine, an update) gets its new value moved back into its old register, so
        code after the body, or the next iteration, reads one fixed register on every path."""
        before = dict(self.sym_table.scalars)
        self.visit(body)
        for name, old in before.items():
            new = self.sym_table.scalars[name]
            if new is not old:
                self.builder.append(Mov(type(old).PTX_TYPE, old, new))
                self.sym_table.define_scalar(name, old)

    # ------------------------------------------------------------------ expressions

    def visit_Var(self, var):
        """A name in an index expression -> its scalar Register (an axis like a0, or lane/warp)."""
        return self.sym_table.scalar(var.name)

    def visit_Literal(self, literal):
        """An index constant -> Immediate(value, INDEX_TYPE). Emits nothing."""
        if literal.dtype != "int":
            raise NotImplementedError(f"{literal.dtype} literal in an index expression")
        return Immediate(literal.value, INDEX_TYPE)

    def visit_BinaryExpr(self, binary):
        """Integer index math: lower both sides, emit one op, return its result Register.

        Seen so far: + * (p*), / % - == < (later categories). Comparisons return a predicate.
        """
        left, right = self.visit(binary.left), self.visit(binary.right)
        if binary.op in INDEX_CMP:
            if isinstance(left, Immediate) and isinstance(right, Immediate):
                return Immediate(int(_compare(binary.op, left.value, right.value)), PtxType.PRED)
            return self.builder.setp(INDEX_CMP[binary.op], INDEX_TYPE, left, right)
        if binary.op in INDEX_LOGIC:
            return self.logic(binary.op, left, right)
        if binary.op not in INDEX_ARITH:
            raise NotImplementedError(f"index operator {binary.op!r}")
        # PTX needs at least one register operand, so two literals are folded here
        if isinstance(left, Immediate) and isinstance(right, Immediate):
            return Immediate(_fold(binary.op, left.value, right.value), INDEX_TYPE)
        if binary.op == "*":
            return self.builder.mul(INDEX_TYPE, left, right)
        if binary.op == "^":   # bitwise ops are typed by width, not sign
            return self.builder.binary(Op.XOR, PtxType.B32, left, right)
        return self.builder.binary(INDEX_ARITH[binary.op], INDEX_TYPE, left, right)

    def logic(self, op, left, right):
        """&& / || on predicates; a folded (constant) side decides or drops out."""
        for const, other in ((left, right), (right, left)):
            if isinstance(const, Immediate):
                if isinstance(other, Immediate):
                    value = (const.value and other.value) if op == "&&" else (const.value or other.value)
                    return Immediate(int(bool(value)), PtxType.PRED)
                absorbing = not const.value if op == "&&" else const.value
                return Immediate(int(bool(const.value)), PtxType.PRED) if absorbing else other
        return self.builder.binary(INDEX_LOGIC[op], PtxType.PRED, left, right)

    def visit_TernaryExpr(self, ternary):
        """cond ? a : b on indices -> selp (both sides are computed; they are cheap)."""
        cond = self.visit(ternary.cond)
        if isinstance(cond, Immediate):
            return self.visit(ternary.if_true if cond.value else ternary.if_false)
        a, b = self.visit(ternary.if_true), self.visit(ternary.if_false)
        dst = self.builder.reg(INDEX_TYPE)
        self.builder.append(Instruction(Op.SELP, (INDEX_TYPE,), dsts=[dst], srcs=[a, b, cond]))
        return dst

    def visit__Sum(self, node):
        return self.add_index(self.visit(node.left), self.visit(node.right))

    # ------------------------------------------------------------------ helpers

    def address(self, buffer, index):
        """Kernel IR index tuple -> Address of that element in `buffer`.

        flat = sum(idx_i * stride_i) (row-major; an empty index means 0), then
        global: Address(base + mul.wide(flat, elem_bytes));  shared: the same from the Symbol.
        Shared by Load and Write.
        """
        flat = None
        for idx, stride in zip(index, buffer.strides):
            term = self.visit(idx)
            if stride != 1:
                term = self.mul_index(term, stride)
            flat = term if flat is None else self.add_index(flat, term)

        return self.element(buffer, Immediate(0, INDEX_TYPE) if flat is None else flat)

    def element(self, buffer, flat):
        """Address of element `flat` (a flattened index Value) of `buffer`."""
        nbytes = buffer.elem_type.nbytes
        # constant index: the byte offset goes straight into [base+off]
        if isinstance(flat, Immediate):
            return Address(buffer.base, flat.value * nbytes)
        b = self.builder
        # a static shared array is a Symbol; ld/st need its address in a register for [reg]
        base = buffer.base if isinstance(buffer.base, Register) else b.mov(PtxType.U64, buffer.base)
        offset = b.mul_wide(flat, Immediate(nbytes, INDEX_TYPE))
        return Address(b.add(PtxType.S64, base, offset))

    def div_index(self, value, k):
        if k == 1:
            return value
        if isinstance(value, Immediate):
            return Immediate(int(value.value / k), INDEX_TYPE)
        return self.builder.binary(Op.DIV, INDEX_TYPE, value, Immediate(k, INDEX_TYPE))

    def rem_index(self, value, k):
        if isinstance(value, Immediate):
            return Immediate(value.value % k, INDEX_TYPE)
        return self.builder.binary(Op.REM, INDEX_TYPE, value, Immediate(k, INDEX_TYPE))

    def mul_index(self, value, k):
        if isinstance(value, Immediate):
            return Immediate(value.value * k, INDEX_TYPE)
        return self.builder.mul(INDEX_TYPE, value, Immediate(k, INDEX_TYPE))

    def add_index(self, a, b):
        if isinstance(a, Immediate) and isinstance(b, Immediate):
            return Immediate(a.value + b.value, INDEX_TYPE)
        return self.builder.add(INDEX_TYPE, a, b)

    def index_value(self, value):
        """A loop bound given either as an IR expression or already as a Value."""
        return value if isinstance(value, Immediate) else self.visit(value)

    def operand(self, name_or_expr):
        """An Assign/Write argument: a scalar's name, or (defensively) an expression node."""
        if isinstance(name_or_expr, str):
            return self.sym_table.scalar(name_or_expr)
        return self.visit(name_or_expr)
