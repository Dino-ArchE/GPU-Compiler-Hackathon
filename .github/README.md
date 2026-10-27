# Kernel IR → PTX, in Python

A compiler for [Emmy](https://github.com/cloudrift-ai/emmy)'s Kernel IR that emits PTX directly,
written for Emmy Hackathon 2026. It lowers the IR into its own small PTX-level IR, optimizes
that with an LLVM-style pass manager, and prints PTX that `ptxas` assembles.

```bash
python compile.py path/to/kernel.ir.json > out.ptx        # -O2 by default
python submit.py p00_scalar_add --compiler compile.py      # run it on the hosted RTX 5090
```

Emmy is used only to load the IR (`Graph.from_dict`). The translation to PTX is ours: no
`nvcc`, no `emmy compile`, no Emmy renderer. The task description is in the root
[README.md](../README.md) and [TASK.md](../TASK.md).

## Results

**All 15 open problems pass, in all five categories**, plus 49 more problems we generated to
look like the hidden set (64/64 on the RTX 5090 runner, at `-O0` and at `-O2`).

| Problem | Category | `-O0` (µs) | `-O2` (µs) |
|---|---|---:|---:|
| p00_scalar_add | pointwise | 22.1 | 20.4 |
| p01_vector_add | pointwise | 20.4 | 20.1 |
| p02_scale_shift | pointwise | 15.1 | 16.3 |
| p03_relu_mul | pointwise | 17.1 | 15.5 |
| r01_array_sum | reduction | 20.5 | 20.5 |
| r02_big_sum | reduction | 38.6 | **24.2** |
| r03_mean | reduction | 28.2 | **22.2** |
| r04_row_sum | reduction | 21.3 | 20.3 |
| m01_matmul_512 | matmul | 109.3 | **44.5** |
| m02_matmul_rect | matmul | 16.3 | 14.8 |
| f01_rmsnorm | fused | 15.4 | 15.3 |
| f02_row_softmax | fused | 14.8 | 15.2 |
| f03_l2norm | fused | 15.6 | 15.1 |
| a01_attention | attention | 100.5 | 104.1 |
| a02_attention_128 | attention | 71.6 | **53.4** |

Median kernel time from the runner, one run each. Kernels near 15–20 µs are at the launch floor,
where differences of a few percent are noise. The optimizer pays off where there is real work:
matmul is 2.5× faster, the large reduction 1.6×.

## How it works

```
kernel.ir.json + meta.json
        │  Graph.from_dict (the only Emmy call)
        ▼
  KernelInfo           src/core/frontend   name, params in meta.json's arg_order, buffer shapes, launch
        │
        ▼
  CodeGen              src/core/ptx        visits each Kernel IR node, emits our IR through a Builder
        │
        ▼
  Module/Function/     src/core/ir         PTX-level IR: blocks, instructions, typed registers;
  BasicBlock/Inst                          str() of any piece is its PTX text
        │
        ▼
  PassManager          src/core/opt        -O0 / -O1 / -O2; verifier after every pass;
        │                                  any optimizer exception falls back to -O0 output
        ▼
  str(module)  →  PTX
```

**Our own IR, LLVM-shaped but not SSA.** A `Module` holds `Function`s, a function holds
`BasicBlock`s, a block holds `Instruction`s, and operands are `Value`s (registers, immediates,
addresses, vectors, blocks as branch targets). Unlike LLVM, an instruction is not its own
result: PTX registers are updated in place (an accumulator is rewritten every loop iteration),
and an instruction can write several registers or none. Opcodes, types, state spaces,
comparisons and modifiers are enums, and an instruction rejects a bare string modifier, so a
typo fails at once instead of silently never matching in a pass.

**Code generation** (`src/core/ptx/codegen.py`) dispatches on the IR node's class name
(`visit_Tile`, `visit_Accum`, ...). The parts that are not spelled out in the IR:

- **Tile → threads.** Each thread's flat id `ctaid.x * ntid.x + tid.x` is decoded into the
  tile's axes, row-major (last axis fastest), so the threads of one cooperative tile share a CTA
  and neighbouring lanes touch neighbouring addresses. Grouped rasterization (`raster_group`)
  reorders the matmul block tiles for L2 reuse.
- **The thread guard.** The IR has no bounds check. When the launch has more threads than
  cells, extra threads branch to the exit; barriers stay correct because exited threads no
  longer count, and warp shuffles use `activemask` instead of a full mask.
- **Accumulators** start at their op's identity (0 for sum, -inf for max) before the loop the
  IR names, unless an explicit `Init` owns them.
- **Registers are rebound, not overwritten.** Each assignment gets a fresh register; at the end
  of a loop body or branch, names that changed are moved back into the register the next
  iteration reads.
- **Reductions:** warp shuffles (XOR butterfly) and a shared-memory tree. **Matmul:** `cp.async`
  copies into shared memory, with dynamic shared memory when a kernel needs more than 48 KB.
- **Elementwise ops** are a table (`src/core/ptx/ops.py`): 34 ops including exp, log, tanh, erf,
  sigmoid, sin/cos with range reduction, pow, comparisons and `where`.

**The optimizer** (`src/core/opt/`) is a pass manager with cached analyses (CFG, dominators,
loops, def-use, liveness, value ranges, pointer roots). `-O2` focuses on facts `ptxas` cannot
know, because it never sees the launch configuration or which buffers are read-only:

| Pass | What it does |
|---|---|
| launch-specialize | `%ntid.x` → the launch's block size |
| range-fold | decide compares and div/rem from value ranges (removes redundant thread guards) |
| unroll | unroll loops with a trip count every thread shares |
| addr-fold | move constant offsets into `[base+offset]` |
| strength-reduce | integer div/rem by constants → multiply-high and shifts |
| vectorize-mem | adjacent scalar loads/stores → `ld/st.v2/.v4` |
| ld-nc | `ld.global.nc` for buffers the kernel never writes |
| fast-math | IEEE div/sqrt/rcp → approximate forms, within 2 ulp (tolerance is 1e-3) |
| const-fold, instcombine, copy-prop, cse, dce, licm, fuse-mad, if-convert, simplify-cfg | the classic clean-ups, run to a fixed point |

Rewrites that assume one definition per register only touch registers that really have one, so
accumulators are left alone. A verifier runs after every pass and names the pass that broke the
IR. If the optimizer ever throws, `compile.py` emits the unoptimized PTX instead: an optimizer
bug cannot cost a correct kernel.

## How we know it is correct

The hidden set is "the same five categories, different shapes, sizes and op mixes", so most of
the testing is about inputs we have not seen.

- **A reference interpreter for our IR** (`tests/interp.py`) runs every thread as a coroutine,
  with barriers, shuffles and shared memory. Each optimizer pass is tested differentially: run
  the kernel before and after the pass, compare outputs, then assemble with `ptxas`.
- **Hidden-set-style problems** (`tests/e2e/`). The organizers' own generator
  (`tools/generate.py`) builds problem packages from PyTorch expressions; it reproduces the open
  problems byte for byte. We wrote 42 new specs: sizes that do not fill the last block, unseen
  ops (gelu, silu, erf, sin, where, pow), broadcasting, column and max reductions, odd and
  rectangular matmuls, layernorm, log-softmax, attention with unequal lengths. Expected outputs
  come from independent numpy code.
- **Mutations** of open problems exercise IR fields no open problem uses (grouped
  rasterization, a guard around barriers, named barriers) while keeping the expected output
  valid.
- **The GPU runner** checks everything for real: `tests/e2e/run_runner.py --open`, plus
  per-op accuracy against numpy (`tests/e2e/op_check.py`).

Two real bugs this found: comparisons (`x > 0` in `torch.where`) crashed the compiler because
the IR types them `bool`, now a predicate register; and the shared-memory reduction tree had all
32 lanes of a warp read-modify-write the same slot, a data race that only worked because a warp
runs in lockstep on current GPUs. Now only lane 0 stores. The interpreter exposed it by running
lanes one at a time.

```bash
uv sync                                   # dependencies (Python 3.12)
python -m pytest tests --ignore=tests/e2e # unit + interpreter tests, ~1 min
python -m pytest tests/e2e                # every generated problem: compile, ptxas, interpret, ~6 min
python tests/e2e/run_runner.py --open     # all 64 problems on the RTX 5090
```

## Layout

```
compile.py                 entry point: argv -> KernelInfo -> CodeGen -> optimizer -> PTX
src/core/frontend/         KernelInfo: the IR and meta.json, in one object
src/core/ir/               our IR: values, registers, instructions, blocks, functions, Builder
src/core/ptx/              CodeGen (Kernel IR -> our IR), elementwise op table, symbol table
src/core/opt/              pass manager, pipelines, verifier, analysis/, passes/
tests/                     unit tests, reference interpreter
tests/e2e/                 generated problems, mutations, GPU sweep, op accuracy
```

`compile.py --help` lists the optimizer controls: `-O0..2`, `--passes`, `--disable`,
`--print-after <pass|all>` and `--stats`.

## Limitations

- Index math is 32-bit: a tile of more than 2³¹ cells is rejected rather than miscompiled.
- IR the hidden set does not use (tensor-core and TMA nodes, swizzled shared memory) raises a
  clear `NotImplementedError` instead of guessing.
- Only f32 data has been tested end to end; the type plumbing is in place for others.
