"""Load the open problems through codegen and run them in the interpreter on random inputs."""

import json
from pathlib import Path

import numpy as np

from core.frontend import KernelInfo
from core.ptx import CodeGen

from interp import NP_TYPES, run_kernel

ROOT = Path(__file__).resolve().parents[1]


def supported_kernels():
    """[(id, ir_path, KernelInfo)] for every open-problem kernel codegen can lower today."""
    out = []
    for ir in sorted(ROOT.glob("problems/*/*.ir.json")):
        kernel = KernelInfo(ir)
        try:
            CodeGen(kernel).gen()
        except Exception:          # not lowered by codegen yet
            continue
        out.append((f"{ir.parent.name}/{kernel.name}", ir, kernel))
    return out


def random_args(ir_path, kernel, seed=0):
    """Inputs from meta.json (constants keep their data, tensors get random normals), zeroed outputs."""
    rng = np.random.default_rng(seed)
    meta = json.loads((Path(ir_path).parent / "meta.json").read_text())
    entry = next(k for k in meta["kernels"] if k["entry"] == kernel.name)
    by_name = {a["name"]: a for a in entry["args"]}
    inputs, outputs = {}, {}
    for arg in kernel.args:
        dt = NP_TYPES[{"i32": "s32"}.get(arg.dtype, arg.dtype)]
        shape = tuple(arg.shape) or (1,)
        if arg.is_output:
            outputs[arg.name] = np.zeros(shape, dt)
        elif "data" in by_name[arg.name]:
            inputs[arg.name] = np.array(by_name[arg.name]["data"], dt).reshape(shape)
        else:
            inputs[arg.name] = rng.standard_normal(shape).astype(dt)
    return inputs, outputs


def run(module, ir_path, kernel, seed=0):
    inputs, outputs = random_args(ir_path, kernel, seed)
    return run_kernel(module, inputs, outputs)


def fresh(kernel):
    return CodeGen(kernel).gen()
