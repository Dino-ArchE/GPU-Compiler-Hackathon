"""Build small functions by hand and run passes on them."""

from core.ir import Builder, Function, Module, Op
from core.opt import AnalysisManager, PassManager
from core.opt import verify
from core.ptx.printer import print_function


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
