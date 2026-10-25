# End-to-end tests: hidden-set-style problems

The hidden set is "the same five categories, different shapes, sizes and op mixes". This folder
holds problems built the same way the organizers build theirs, so they stand in for it.

| File | What it does |
|---|---|
| `expressions.py` | Torch-expression specs, in `tools/generate.py`'s format: odd sizes, unseen ops, broadcasting, column reductions, odd matmul shapes, layernorm, unequal attention lengths |
| `problems/e*` | Packages generated from those specs: IR, `meta.json`, `expected.npz`, reference `.cu` |
| `mutations.py` | Writes `problems/mx_*`: open problems mutated to exercise IR fields no open problem uses (raster order, a guard around barriers, named barriers, `&&`) |
| `test_e2e.py` | Local pytest, no GPU needed: every problem compiles at `-O0`/`-O2` and assembles with ptxas; small kernels run in the interpreter on the seeded inputs and are compared with `expected.npz` |
| `run_runner.py` | The evaluation pipeline on the GPU: `submit.py --compiler compile.py` over every problem |
| `op_check.py` | Accuracy of each elementwise op on the GPU against numpy, including overflow ranges |

```bash
python -m pytest tests/e2e                     # local
python tests/e2e/run_runner.py [-O0] [--open]  # GPU runner (all e2e problems; --open adds problems/)
python tests/e2e/op_check.py                   # GPU runner, per-op accuracy
```

## Regenerating the problems

`problems/` is committed, so none of the above needs torch. To change or add a spec, install
torch locally (CPU build; it is not a project dependency, `compile.py` never uses it):

```bash
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
python tools/generate.py --emmy .venv/bin/emmy --out tests/e2e/problems tests/e2e/expressions.py
python tests/e2e/mutations.py
```

`uv sync` removes packages the lockfile does not list; use `uv sync --inexact` to keep torch.

Writing a spec: every `name = ...` statement in `code` becomes a graph *input* (the tracer
treats it as data), so write intermediate values inline. `ref` receives the seeded inputs in
the graph's input order. Emmy cannot trace row-wise `amin` or `log_softmax`, so they cannot be
in the hidden set either. Generation calls Emmy's own CUDA backend to build the reference; that
is test-data generation only, never part of `compile.py`.
