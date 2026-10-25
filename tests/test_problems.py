"""Every open problem codegen supports: the optimized kernel computes what -O0 computes."""

import os

import numpy as np
import pytest

from core.opt import PassManager, build_pipeline

from problems import fresh, run, supported_kernels

# attention and matmul take minutes in the Python interpreter: opt in with EMMY_SLOW=1
# (the runner checks them on the GPU: tools/run_runner.sh)
SLOW = ("a", "m")
KERNELS = [k for k in supported_kernels() if os.environ.get("EMMY_SLOW") or not k[0].startswith(SLOW)]


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("kid,ir,kernel", KERNELS, ids=[k[0] for k in KERNELS])
def test_optimized_matches_O0(kid, ir, kernel, level):
    expected = run(fresh(kernel), ir, kernel)
    module = fresh(kernel)
    PassManager(build_pipeline(level), verify=True).run(module)
    got = run(module, ir, kernel)
    for name in expected:
        np.testing.assert_allclose(got[name], expected[name], rtol=1e-6, atol=1e-6)
