"""Build small functions by hand and run passes on them."""

import os
import subprocess
import tempfile

from core.ir import Builder, Function, Module, Op
from core.opt import AnalysisManager, PassManager
from core.opt import verify
from core.ptx.printer import print_function, print_module


def new_function(params=("x", "out"), grid=(4, 1, 1), block=(256, 1, 1), readonly=("x",)):
    f = Function("k", list(params))
    f.launch_grid, f.launch_block = grid, block
    f.readonly = {f.param(p) for p in readonly}
    return f, Builder(f)


def finish(b, exit_block=None):
    """Append `ret` (in exit_block if given) and return the function."""
    if exit_block is not None:
        b.set_block(exit_block)
    b.inst(Op.RET)
    return b.function


def run(function, *names):
    pm = PassManager(list(names), verify=True)
    m = Module()
    m.add_function(function)
    changed = pm.run(m)
    verify(function, after="test")
    return changed


def am(function):
    return AnalysisManager(function)


def text(function):
    return print_function(function)


def ops(function):
    """Opcodes with modifiers, in layout order: ['mov.u32', 'add.s32', ...]."""
    return [".".join((i.op, *i.types)) for _, i in function.instructions()]


def differential(build, passes, inputs, outputs, rtol=1e-6, atol=1e-6):
    """Build a fresh function twice, run `passes` on one copy, interpret both, compare outputs.
    `build()` returns a Function (or a Module, when it declares shared arrays). Returns the
    optimized function for further assertions."""
    import numpy as np
    from interp import run_kernel

    def module_of(f):
        if isinstance(f, Module):
            return f
        m = Module()
        m.add_function(f)
        return m

    ref = run_kernel(module_of(build()), inputs, {k: v.copy() for k, v in outputs.items()})
    opt_module = module_of(build())
    opt = opt_module.functions[0]
    PassManager(list(passes), verify=True).run(opt_module)
    verify(opt, after="test")
    got = run_kernel(opt_module, inputs, {k: v.copy() for k, v in outputs.items()})
    for name in ref:
        np.testing.assert_allclose(got[name], ref[name], rtol=rtol, atol=atol)
    assemble(opt_module)
    return opt


PTXAS = os.environ.get("PTXAS", "/usr/local/cuda/bin/ptxas")


def assemble(module):
    """Run ptxas on the module (skipped when ptxas is not installed)."""
    if not os.path.exists(PTXAS):
        return
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as fh:
        fh.write(print_module(module))
    res = subprocess.run([PTXAS, "-arch=sm_89", fh.name, "-o", os.devnull], capture_output=True, text=True)
    os.unlink(fh.name)
    assert res.returncode == 0, res.stderr + print_module(module)
