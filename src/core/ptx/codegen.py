import math

from core.ir import Builder, Function, Module, Op, PtxType, StateSpace
from core.visitor import Buffer, KernelVisitor, SymbolTable


# Smem.dtype uses C spellings
SMEM_TYPES = {"float": PtxType.F32, "half": PtxType.F16, "double": PtxType.F64, "int": PtxType.S32}


class CodeGen(KernelVisitor):
    """Lowers one kernel's Kernel IR into our IR: a Module holding one Function."""

    def __init__(self, kernel):
        super().__init__(kernel)
        self.kernel = kernel
        self.module = Module()
        self.function = self.module.add_function(Function(kernel.name, [a.name for a in kernel.args]))
        self.builder = Builder(self.function)
        self.sym_table = SymbolTable()

    def gen(self):
        self.prologue()
        self.visit(self.kernel.body)
        self.builder.inst(Op.RET)
        return self.module

    def __call__(self):
        return self.gen()

    def prologue(self):
        """Not driven by an IR node: turn each param into a global base address, in arg_order."""
        for arg in self.kernel.args:
            generic = self.builder.load_param(self.function.param(arg.name))
            base = self.builder.cvta_to_global(generic)
            buffer = Buffer(StateSpace.GLOBAL, base, arg.shape, PtxType.from_meta(arg.dtype))
            self.sym_table.define_buffer(arg.name, buffer)

    def visit_Tile(self, tile):
        axes = tile.axes

        # We need to populate each axis as a entry
        for child in tile.body:
            self.visit(child)

    def visit_StridedLoop(self, loop):
        for body in loop.body:
            self.visit(body)

    def visit_Var(self, var):
        name = var.name
        val = self.sym_table.buffer(name)
        return val

    def visit_Load(self, load):
        input_ = load.input
        # Find the value of input_ in the symbol table
        val = self.sym_table.scalar(input_)
        reg = self.builder.reg(PtxType.from_meta(load.dtype))
        dest = self.sym_table.define_scalar(load.name, reg)


    def visit_Write(self, write):
        ...

    def visit_Accum(self, accum):
        ...

    def visit_WarpShuffle(self, shuffle):
        ...

    def visit_Smem(self, smem):
        """Declares a shared array (printed at module scope); emits no instructions."""
        elem_type = SMEM_TYPES[smem.dtype]
        nbytes = math.prod(smem.extents) * elem_type.nbytes
        sym = self.module.add_shared(smem.name, smem.align or elem_type.nbytes, nbytes)
        self.sym_table.define_buffer(smem.name, Buffer(StateSpace.SHARED, sym, smem.extents, elem_type))

    def visit_Cond(self, cond):
        self.visit_BinaryExpr(cond.cond)

    def visit_BinaryExpr(self, binary):
        left_val = self.visit(binary.left)
        right_val = self.visit(binary.right)
        match binary.op:
            case '+':
                ...
            case '*':
                ...
            case _:
                ...


    def visit_Literal(self, literal):
        self.builder

    def visit_Sync(self, sync):
        self.builder.sync(sync.barrier_id)

    def visit_TreeHalve(self, halve):
        ...

    def visit_Assign(self, assign):
        name = assign.name # This is the target
