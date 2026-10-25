from .instruction import (
    Binary, Bra, CpAsync, CpAsyncCommit, CpAsyncWait, Cvta, CvtTf32, Instruction, Load, Mma, Mov,
    Setp, Store, Sync, Unary,
)
from .opcode import Mod, Op
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
        """Fallback for opcodes without a class: inst(Op.ADD, PtxType.F32, dsts=[d], srcs=[a, b])."""
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

    # ------------------------------------------------------------------ async copy, tensor cores

    def cp_async(self, dst_addr, src_addr, nbytes, cache=Mod.CA):
        return self.append(CpAsync(dst_addr, src_addr, nbytes, cache))

    def cp_async_commit(self):
        return self.append(CpAsyncCommit())

    def cp_async_wait(self, pending=0):
        return self.append(CpAsyncWait(pending))

    def cvt_tf32(self, x):
        """f32 -> tf32 bits in a .b32 register (an mma A/B operand)."""
        dst = self.reg(PtxType.B32)
        self.append(CvtTf32(dst, x))
        return dst

    def split_tf32(self, x):
        """x ~= hi + lo with both tf32 (3xTF32: hi*hi + hi*lo + lo*hi recovers ~f32 accuracy)."""
        hi = self.cvt_tf32(x)
        hi_f = self.reg(PtxType.F32)
        self.append(Mov(PtxType.B32, hi_f, hi))                 # bit-cast back to read it as f32
        return hi, self.cvt_tf32(self.binary(Op.SUB, PtxType.F32, x, hi_f))

    def mma_tf32(self, a, b, c, d=None):
        """D = A*B + C for one warp (see Mma for the fragment layout). a: 4 .b32, b: 2 .b32,
        c: 4 .f32 registers; d defaults to fresh registers. Returns d."""
        d = list(d) if d is not None else [self.reg(PtxType.F32) for _ in range(4)]
        self.append(Mma(d, a, b, c))
        return d

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
        return self.binary(Op.MUL, ptx_type, a, b, mode=None if ptx_type.is_float else Mod.LO)

    def mad(self, ptx_type, a, b, c):
        """a * b + c. Integers keep the low half (mad.lo); floats are one fused, rounded
        operation (fma.rn: PTX requires a rounding mode on float multiply-add)."""
        dst = self.reg(ptx_type)
        if ptx_type.is_float:
            self.append(Instruction(Op.FMA, (Mod.RN, ptx_type), dsts=[dst], srcs=[a, b, c]))
        else:
            self.append(Instruction(Op.MAD, (Mod.LO, ptx_type), dsts=[dst], srcs=[a, b, c]))
        return dst

    def cvt(self, dst_type, src_type, a, mode=None):
        """cvt[.mode].<dst>.<src>: convert a value; mode rounds (rni = to nearest integer)."""
        dst = self.reg(dst_type)
        types = (mode, dst_type, src_type) if mode else (dst_type, src_type)
        self.append(Instruction(Op.CVT, types, dsts=[dst], srcs=[a]))
        return dst

    def unary(self, op, ptx_type, a, mode=None):
        dst = self.reg(ptx_type)
        self.append(Unary(op, ptx_type, dst, a, mode))
        return dst

    def mul_wide(self, a, b, ptx_type=PtxType.S32):
        """32 x 32 -> 64 bit: turns an element index into a 64-bit byte offset."""
        dst = self.reg(PtxType.S64 if ptx_type is PtxType.S32 else PtxType.U64)
        self.append(Binary(Op.MUL, ptx_type, dst, a, b, mode=Mod.WIDE))
        return dst

    def setp(self, cmp, ptx_type, a, b):
        """Compare a and b; returns the predicate register holding the result."""
        dst = self.reg(PtxType.PRED)
        self.append(Setp(cmp, ptx_type, dst, a, b))
        return dst

    def bra(self, target, guard=None, negate=False):
        """Branch to `target`; with a guard, only where it holds (or fails, with negate)."""
        return self.append(Bra(target, guard=guard, negate=negate))
