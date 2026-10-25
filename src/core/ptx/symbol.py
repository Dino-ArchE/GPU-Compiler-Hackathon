import math


class Buffer:
    """An indexable array in Kernel IR terms: a global param buffer or a shared array.

    Load/Write handle both the same way: flatten the index against `shape`, scale by
    the element size, and address it from `base` in `space`.
    """

    def __init__(self, space, base, shape, elem_type):
        self.space = space            # StateSpace.GLOBAL or StateSpace.SHARED
        self.base = base              # global: Register holding the address; shared: its Symbol
        self.shape = tuple(shape)
        self.elem_type = elem_type    # PtxType of one element

    @property
    def strides(self):
        """Row-major element strides: shape (64, 512) -> (512, 1)."""
        return tuple(math.prod(self.shape[i + 1:]) for i in range(len(self.shape)))

    def __repr__(self):
        return f"Buffer({self.space} {self.base} {list(self.shape)} {self.elem_type})"


class SymbolTable:
    """Resolves Kernel IR names to what the visitor built for them.

    Flat, with no scopes: Emmy's names never shadow. A name defined again is the same variable
    updated in place (Accum, combine_states, Init) or a name reused after its last use
    (sibling loops), so redefining simply overwrites. Only a lookup of an undefined name is an error.
    """

    def __init__(self):
        self.scalars = {}   # axis vars, SSA values, accumulators, lane/warp -> Register
        self.buffers = {}   # global params and shared arrays -> Buffer

    def define_scalar(self, name, reg):
        self.scalars[name] = reg
        return reg

    def scalar(self, name):
        if name not in self.scalars:
            raise LookupError(f"unknown scalar {name!r}: used before any statement defined it")
        return self.scalars[name]

    def define_buffer(self, name, buffer):
        self.buffers[name] = buffer
        return buffer

    def buffer(self, name):
        if name not in self.buffers:
            raise LookupError(f"unknown buffer {name!r}: not a kernel param or declared Smem")
        return self.buffers[name]
