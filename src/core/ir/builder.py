from .instruction import Binary, Cvta, Instruction, Load, Mov, Store, Sync
from .opcode import Op
from .type_enum import PtxType, StateSpace
from .value import Address


class Builder:
    """IRBuilder: appends instructions at an insertion point inside one Function.

    The visitor decides what to build; the builder only creates registers, blocks and
    instructions. It knows nothing about the Kernel IR or about PTX text. Like LLVM's
    IRBuilder, methods that produce a value allocate its register and return it.
    """

    def __init__(self, function):
        self.function = function
        self.block = None
        self.set_block(function.new_block("ENTRY"))

    def reg(self, ptx_type):
        return self.function.regs.new(ptx_type)

    def new_block(self, hint="BB"):
        return self.function.new_block(hint)

    def set_block(self, block):
        """Move the insertion point to `block`, placing it next in the layout on first visit."""
        if block not in self.function.blocks:
            self.function.blocks.append(block)
        self.block = block

    def append(self, inst):
        return self.block.append(inst)

    def inst(self, op, *types, dsts=(), srcs=(), guard=None, negate=False, comment=None):
        """Fallback for opcodes without a class: inst(Op.ADD, "f32", dsts=[d], srcs=[a, b])."""
        return self.append(Instruction(op, types, dsts, srcs, guard, negate, comment))

    def load(self, space, ptx_type, addr):
        dst = self.reg(ptx_type)
        self.append(Load(space, ptx_type, dst, addr))
        return dst

    def load_param(self, param):
        return self.load(StateSpace.PARAM, param.ptx_type, Address(param))

    def store(self, space, ptx_type, addr, src):
        return self.append(Store(space, ptx_type, addr, src))

    def mov(self, ptx_type, src):
        dst = self.reg(ptx_type)
        self.append(Mov(ptx_type, dst, src))
        return dst

    def sync(self, barrier_id=0):
        return self.append(Sync(barrier_id))

    def cvta_to_global(self, src):
        """Generic address (as passed in a param) -> global address for ld/st.global."""
        dst = self.reg(PtxType.U64)
        self.append(Cvta(StateSpace.GLOBAL, dst, src))
        return dst

    def binary(self, op, ptx_type, a, b, mode=None):
        """Generic two-source op; a and b are Registers or Immediates."""
        dst = self.reg(ptx_type)
        self.append(Binary(op, ptx_type, dst, a, b, mode))
        return dst

    def add(self, ptx_type, a, b):
        return self.binary(Op.ADD, ptx_type, a, b)

    def mul(self, ptx_type, a, b):
        """Integer mul must say which half it keeps: mul.lo (same width). Float mul has no mode."""
        return self.binary(Op.MUL, ptx_type, a, b, mode=None if ptx_type.is_float else "lo")

    def mul_wide(self, a, b, ptx_type=PtxType.S32):
        """32 x 32 -> 64 bit: turns an element index into a 64-bit byte offset."""
        dst = self.reg(PtxType.S64 if ptx_type is PtxType.S32 else PtxType.U64)
        self.append(Binary(Op.MUL, ptx_type, dst, a, b, mode="wide"))
        return dst
