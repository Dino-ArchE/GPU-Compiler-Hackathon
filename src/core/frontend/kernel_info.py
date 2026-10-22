import json
from pathlib import Path

from emmy.compiler.graph import Graph


class Arg:
    """One kernel parameter: a global buffer passed as a .u64 pointer."""

    def __init__(self, name, dtype, shape, is_output):
        self.name = name
        self.dtype = dtype          # "f32", "f16", "i32", ... (meta.json spelling)
        self.shape = tuple(shape)   # () or (1,) for constants; used to flatten indices
        self.is_output = is_output

    def __repr__(self):
        kind = "out" if self.is_output else "in"
        return f"Arg({self.name}: {self.dtype}{list(self.shape)} {kind})"


class KernelInfo:
    """Everything codegen needs about one kernel: the IR body plus the launch contract."""

    def __init__(self, ir_path):
        ir_path = Path(ir_path).resolve()

        with open(ir_path) as f:
            graph = Graph.from_dict(json.load(f))
        kernels = [n.op for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]
        if len(kernels) != 1:
            raise ValueError(f"{ir_path}: expected one KernelOp, found {len(kernels)}")
        op = kernels[0]

        with open(ir_path.parent / "meta.json") as f:
            meta = json.load(f)
        entries = [k for k in meta["kernels"] if k["entry"] == op.name]
        if not entries:
            raise ValueError(f"{ir_path}: no kernel {op.name!r} in meta.json")
        launch = entries[0]

        self.name = op.name
        self.body = op.body
        self.grid = tuple(launch["grid"])
        self.block = tuple(launch["block"])

        # meta.json's arg_order is the parameter contract; args carries dtype/shape per name
        arg_meta = {a["name"]: a for a in launch["args"]}
        self.args = []
        for name in launch["arg_order"]:
            a = arg_meta[name]
            self.args.append(Arg(name, a["dtype"], a["shape"], a.get("output", False)))
        self.arg_by_name = {a.name: a for a in self.args}

    def __repr__(self):
        return f"KernelInfo({self.name}, args={self.args}, grid={self.grid}, block={self.block})"
