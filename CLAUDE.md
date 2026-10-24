# Emmy Hackathon 2026 — Kernel IR → PTX compiler

## Essence

Write `compile.py` (repo root): `python compile.py path/to/kernel.ir.json > out.ptx`.
Input is Emmy's Kernel IR (one `KernelOp` per file); output is PTX that `ptxas` assembles and that
produces correct results under the launch in the sibling `meta.json`. Evaluated on a **hidden set**
of the same 5 categories (pointwise, reductions, matmul, fused, attention) — nothing may be
hardcoded to the open problems. Deadline: **2026-10-27 23:59 AoE**. MVP = categories 1 + 2.

**The rule:** the IR → PTX translation must be our own. Emmy may be used only to *load* the IR
(`Graph.from_dict`). Never use emmy's renderer (`render_kernelop`), `emmy compile`, `nvcc`, or any
other compiler in the pipeline. Judges read the code; the author must explain it on video.

## Working mode

The user writes the compiler; Claude pairs (explain, review, debug, answer PTX/IR questions).
Do not write large chunks of the compiler unprompted. Small, targeted suggestions and snippets
are fine; ask before implementing a whole phase.

## Output contract

- Entry name = `KernelOp.name`.
- Params = `meta.json` `kernels[i].arg_order` (buffers as `.param .u64`). Not the IR's input order.
  Organizers confirmed reading it is permitted. Path: `Path(argv[1]).resolve().parent / "meta.json"`;
  pick the kernel whose `entry == KernelOp.name`, not `kernels[0]`. (`KernelOp.inputs` order is
  wrong for p02/f01/m01/m02; the raw JSON graph node's `inputs` + `outputs` matches 15/15.)
- Header: `.version 8.0` / `.target sm_89` / `.address_size 64`. ASCII only, even in comments.
- Correct under `meta.json` grid/block. The thread guard `gid < N` (N = product of Tile axis
  extents) is NOT in the IR. Derive it, and don't let guarded threads skip a `bar.sync`.
- `ConstantOp` nodes arrive as 1-element buffer args.

## Architecture (LLVM-style: visitor -> our IR -> printer)

Emmy is the frontend + middle end; `ptxas` does register allocation and scheduling. We own the
step between: a visitor lowers Kernel IR into our own small PTX-level IR, which a printer renders.
The IR layer exists for clean structure and optional passes, NOT for speed (ptxas already does
CSE/DCE/LICM). Build passes only after categories 1-2 pass.

Optimization plan: a pass manager over our IR (`src/core/opt/`, built; see below), toggled by a
compile.py flag so raw vs optimized output can be diffed. `submit.py` sends no ptxas flags, so
ptxas runs at its default `-O3`; the runner API also accepts `-O0..-O3`. Priority: passes that
need knowledge ptxas lacks (vector ld/st, `ld.global.nc` for read-only inputs, unsigned index
math for cheap div/rem, guard elision, approx math within tolerance) over passes ptxas
already does (CSE/DCE/const-fold). Measure with the runner's `registers` and `median_us`.
Our registers are not SSA: any pass must respect redefinitions (accumulators).

```
compile.py                   argv -> KernelInfo -> visitor -> print_module -> stdout
src/core/frontend/           KernelInfo: name, params (arg_order), buffer shapes/dtypes, body
src/core/visitor/            KernelVisitor: dispatch by class name (visit_Tile); codegen subclass
src/core/ir/                 our IR, data only (+ IRBuilder):
    value.py                 Value base; Immediate, Symbol, Address [base+off], Vector {..}
    reg.py                   Register(Value) hierarchy, RegisterFile allocator
    type_enum.py             PtxType (f32, u64, s32, ...; nbytes, is_float, from_meta), StateSpace
    opcode.py                Op enum (ld, st, mov, add, setp, bra, ...)
    instruction.py           Instruction(op, types, dsts, srcs, guard, negate, comment); defs/uses.
                             Subclasses (Load, Store, Mov, Sync) are typed constructors that fill
                             the same fields; they never print themselves
    block.py                 BasicBlock(Value): branch target + instruction list
    function.py              Function: params, shared, RegisterFile, blocks (layout order)
    module.py                Module: version/target/address_size + functions
    builder.py               Builder: insertion point, reg(), new_block(), set_block(); typed
                             helpers (load, load_param, store, mov, sync) allocate and RETURN the
                             result register, LLVM-style; inst(Op.X, ...) is the generic fallback
src/core/ptx/printer.py      print_module(Module) -> PTX text; the only code that knows PTX syntax
src/core/opt/                optimizer over our IR (compile.py -O0|-O1|-O2, default -O2;
                             --passes a,b / --print-after X|all / --stats). On any optimizer
                             exception compile.py falls back to unoptimized PTX.
    pass_manager.py          FunctionPass/Analysis + @register_pass/@register_analysis (one class
                             per file, auto-discovered), AnalysisManager cache (`preserves`
                             decides what survives a change), Repeat(...) fixed-point groups
    pipeline.py              the -O levels; every pipeline starts with split-blocks
    verifier.py              run after every changing pass (names the pass that broke the IR)
    fold.py                  PTX-exact constant evaluation shared by folding passes
    analysis/                cfg, dominators, loops, defuse (single_def/is_ssa), liveness,
                             ranges (interval + pow2 alignment, launch-aware), memory (pointer
                             roots, read-only buffers; distinct params assumed not to alias)
    passes/                  split-blocks, const-fold, instcombine, copy-prop, cse, dce,
                             simplify-cfg, licm, launch-specialize, range-fold, addr-fold,
                             strength-reduce, vectorize-mem, fuse-mad, ld-nc
tests/                       pytest; interp.py is a reference interpreter for our IR (threads as
                             coroutines, bar.sync/shfl supported). helpers.differential() runs a
                             kernel before/after passes, compares outputs, then ptxas-checks it.
tools/check_opt.sh           -O0 vs -O2 + ptxas over every supported problem
tools/run_runner.sh          submit every supported problem to the runner
```

Optimizer rules: SSA-style rewrites (copy-prop, CSE, LICM, reassociation) only touch registers
whose `defuse.single_def` exists (one unguarded def); accumulators are left alone. A guarded def
is not a kill. Special registers can only feed `mov`. Each new pass gets a unit test through
`differential()`, then `tools/check_opt.sh` + `tools/run_runner.sh`, then its own commit.

Rules: every operand is a `Value` (never a raw str/int). Opcodes, types and state spaces use the
enums (`Op`, `PtxType`, `StateSpace`); meta.json dtypes go through `PtxType.from_meta` (i32 -> s32). Instruction is not a Value: PTX registers
are not SSA (accumulators update in place). Block layout order = `set_block` order, and fall-through
goes to the next block. The visitor emits `ret` itself. `ptx/` never imports frontend or IR-gen code.

Phase order: pointwise (p*) -> reductions (r*) -> fused (f*) -> attention (a*) -> matmul (m*)
-> hidden-set hardening. Attention is easier than matmul: it has no cp.async.

## IR facts (verified on the open set)

- `KernelOp.body` is a `Body` (iterable, not a dataclass); statements are dataclasses.
  `Tile.body`, loop bodies and `Cond.body` are also `Body`s.
- **Tile decode:** `gid = ctaid.x*ntid.x + tid.x`, then the axes are decoded row-major, mixed-radix
  (last axis = `gid % ext`, earlier ones `(gid / stride) % ext`, first one `gid / stride`).
  `raster_axes` doesn't change the decode in any reference `.cu`.
- `lane = tid.x & 31`, `warp = tid.x >> 5` are implicit vars used by `Cond`/`TreeHalve`.
- **Buffer shapes:** `op.inputs`/`op.outputs` tensors report `shape=()`. Take shapes from the
  graph-level JSON nodes and flatten multi-dim indices row-major. `Smem` uses its own `extents`.
  An empty index (`sum_1[]`) means offset 0.
- **Accum:** the accumulator starts at the op's identity (add -> 0, maximum -> -inf) and is
  initialized *before the loop named in `Accum.axes`*, not at the statement. It can be read
  before its `Accum` line (online softmax `maximum(acc0, in0)`), and it is reset per
  outer-loop iteration in attention.
- `WarpShuffle`/`TreeHalve` give their combine as `combine_states` (a list of `Assign`) over
  `state` + `state_b` names. Lower them generically: shuffle/load each state into `state_b`,
  then emit the Assigns. This covers the single sum and the two-state softmax alike.
- Multi-name `Load` / multi-value `Write` = consecutive elements -> `ld/st.global.v2|v4.f32`.
  `Write.atomic` -> `red.global.add.f32` (meta lists the buffer in `zero_outputs`).
- Ops seen: add, subtract, multiply, divide, maximum, reciprocal, sqrt, exp, relu, copy.
  The hidden set may add more, so keep `ops.py` table-driven.
- Float literals in PTX: hex form `0f3F800000` (1.0). exp(x) = `ex2.approx(x * log2e)`.

## Testing

```bash
python submit.py p00_scalar_add --compiler compile.py      # hosted RTX 5090 runner
/usr/local/cuda/bin/ptxas -arch=sm_89 out.ptx -o /dev/null # local syntax check (ptxas 12.1)
```

`*.ir.txt` is the readable IR; `*.ref.cu` is Emmy's CUDA for the same kernel, useful for
semantics (index decode, loop bounds). `examples/k_sum_1_reduce.ptx` is hand-written PTX for
r01. More practice IRs (needs torch): `EMMY_DUMP_DIR=out emmy compile -c "<expr>" --ir kernel`.
