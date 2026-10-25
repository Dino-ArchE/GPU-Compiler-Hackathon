"""Our IR -> PTX text. The only place that knows PTX file layout and instruction syntax."""


def print_module(module):
    out = [
        f".version {module.version}",
        f".target {module.target}",
        f".address_size {module.address_size}",
    ]
    # shared arrays are module-scope globals, declared ahead of the entries that use them
    if module.shared:
        out.append("")
        for sym, align, nbytes in module.shared:
            if nbytes is None:   # dynamic shared memory: size comes from the launch
                out.append(f".extern .shared .align {align} .b8 {sym}[];")
            else:
                out.append(f".shared .align {align} .b8 {sym}[{nbytes}];")
    for function in module.functions:
        out += ["", *_function(function)]
    text = "\n".join(out) + "\n"
    if not text.isascii():
        bad = sorted({c for c in text if not c.isascii()})
        raise ValueError(f"PTX must be ASCII; found {bad}")
    return text


def _function(function):
    out = []
    params = [f"\t.param .{sym.ptx_type} {sym}" for sym in function.params.values()]
    if params:
        out += [f".visible .entry {function.name}(", ",\n".join(params), ")"]
    else:
        out.append(f".visible .entry {function.name}()")

    out.append("{")
    out += [f"\t{decl}" for decl in function.regs.declarations()]
    for block in function.blocks:
        out.append(f"{block.name}:")
        out += [f"\t{_instruction(inst)}" for inst in block.instructions]
    out.append("}")
    return out


def _instruction(inst):
    guard = f"@{'!' if inst.negate else ''}{inst.guard} " if inst.guard is not None else ""
    opcode = ".".join((inst.op, *inst.types))
    operands = ", ".join(str(v) for v in (*inst.dsts, *inst.srcs))
    text = f"{guard}{opcode} {operands};" if operands else f"{guard}{opcode};"
    return f"{text}\t// {inst.comment}" if inst.comment else text


def print_function(function):
    """One function's PTX text without the module header; for debug dumps."""
    return "\n".join(_function(function))
