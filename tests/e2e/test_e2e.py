"""Local end-to-end checks over the generated e2e problems (no GPU needed).

For every problem in tests/e2e/problems:
  * compile.py lowers it at -O0 and -O2 and ptxas assembles the result;
  * kernels small enough to interpret are run on the generator's seeded inputs (the ones the
    runner uses) and compared with expected.npz at the problem's tolerance.
The GPU check of the same problems is tests/e2e/run_runner.py.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tests/: interp, helpers

from core.frontend import KernelInfo  # noqa: E402
from core.opt import PassManager, build_pipeline  # noqa: E402
from core.ptx import CodeGen  # noqa: E402

from helpers import PTXAS  # noqa: E402
from interp import NP_TYPES, run_kernel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROBLEMS = sorted(p for p in (Path(__file__).resolve().parent / "problems").iterdir() if (p / "meta.json").exists())
# the interpreter runs every thread in Python: keep to kernels it finishes in seconds
# (tests/e2e/run_runner.py checks every problem, large ones included, on the GPU)
MAX_INTERP_THREADS = 4096


def kernels(problem):
    meta = json.loads((problem / "meta.json").read_text())
    return [(problem / f"{k['entry']}.ir.json", k) for k in meta["kernels"]]


@pytest.mark.parametrize("level", [0, 2])
@pytest.mark.parametrize("problem", PROBLEMS, ids=[p.name for p in PROBLEMS])
def test_compiles_and_assembles(problem, level, tmp_path):
    for ir, _ in kernels(problem):
        r = subprocess.run([sys.executable, str(ROOT / "compile.py"), f"-O{level}", str(ir)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "optimizer failed" not in r.stderr, r.stderr
        if Path(PTXAS).exists():
            out = tmp_path / "k.ptx"
            out.write_text(r.stdout)
            a = subprocess.run([PTXAS, "-arch=sm_89", str(out), "-o", "/dev/null"], capture_output=True, text=True)
            assert a.returncode == 0, a.stderr


def seeded_inputs(entry, kernel):
    """The generator's inputs: arg `seed` s -> default_rng(s).standard_normal; constants carry data."""
    by_name = {a["name"]: a for a in entry["args"]}
    inputs, outputs = {}, {}
    for arg in kernel.args:
        dt = NP_TYPES[{"i32": "s32"}.get(arg.dtype, arg.dtype)]
        shape = tuple(arg.shape) or (1,)
        a = by_name[arg.name]
        if arg.is_output:
            outputs[arg.name] = np.zeros(shape, dt)
        elif "data" in a:
            inputs[arg.name] = np.array(a["data"], dt).reshape(shape)
        else:
            inputs[arg.name] = np.random.default_rng(a["seed"]).standard_normal(shape).astype(dt)
    return inputs, outputs


SMALL = [p for p in PROBLEMS
         if all(np.prod(k["grid"]) * np.prod(k["block"]) <= MAX_INTERP_THREADS for _, k in kernels(p))
         and len(kernels(p)) == 1]


# -O2 only: that -O1/-O2 compute what -O0 computes is tests/test_problems.py's job
@pytest.mark.parametrize("level", [2])
@pytest.mark.parametrize("problem", SMALL, ids=[p.name for p in SMALL])
def test_matches_expected(problem, level):
    meta = json.loads((problem / "meta.json").read_text())
    expected = dict(np.load(problem / "expected.npz"))
    (ir, entry), = kernels(problem)
    kernel = KernelInfo(ir)
    module = CodeGen(kernel).gen()
    PassManager(build_pipeline(level), verify=True).run(module)
    str(module)                                           # the module prints (ASCII, every operand)
    inputs, outputs = seeded_inputs(entry, kernel)
    try:
        got = run_kernel(module, inputs, outputs)
    except RuntimeError as e:
        if "step limit" in str(e):
            pytest.skip("too long to interpret; covered by run_runner.py")
        raise
    for name, ref in expected.items():
        np.testing.assert_allclose(got[name].reshape(ref.shape), ref, rtol=meta["rtol"], atol=meta["atol"])
