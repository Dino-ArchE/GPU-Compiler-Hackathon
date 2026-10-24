"""Elementwise ops (Assign.op.name) -> instructions. One entry per op: add a line for a new op.

Each lowering takes (builder, ptx_type, *args) with args as Values and returns the result
Register. Float results use IEEE-rounded forms (.rn) where PTX has a choice, so outputs match
the reference; swapping in .approx variants is an optimizer decision, not a codegen one.
"""

import math

from core.ir import Immediate, Op

LOG2E = math.log2(math.e)


def _exp(b, t, x):
    # PTX has only 2^x: e^x = 2^(x * log2(e))
    return b.unary(Op.EX2, t, b.mul(t, x, Immediate(LOG2E, t)), mode="approx")


ELEMENTWISE = {
    "add":        lambda b, t, x, y: b.add(t, x, y),
    "subtract":   lambda b, t, x, y: b.binary(Op.SUB, t, x, y),
    "multiply":   lambda b, t, x, y: b.mul(t, x, y),
    "divide":     lambda b, t, x, y: b.binary(Op.DIV, t, x, y, mode="rn"),
    "maximum":    lambda b, t, x, y: b.binary(Op.MAX, t, x, y),
    "minimum":    lambda b, t, x, y: b.binary(Op.MIN, t, x, y),
    "relu":       lambda b, t, x: b.binary(Op.MAX, t, x, Immediate(0.0, t)),
    "negative":   lambda b, t, x: b.unary(Op.NEG, t, x),
    "abs":        lambda b, t, x: b.unary(Op.ABS, t, x),
    "reciprocal": lambda b, t, x: b.unary(Op.RCP, t, x, mode="rn"),
    "sqrt":       lambda b, t, x: b.unary(Op.SQRT, t, x, mode="rn"),
    "exp":        _exp,
    "copy":       lambda b, t, x: b.mov(t, x),
}


def lower_elementwise(builder, name, ptx_type, args):
    if name not in ELEMENTWISE:
        raise NotImplementedError(f"elementwise op {name!r}")
    return ELEMENTWISE[name](builder, ptx_type, *args)
