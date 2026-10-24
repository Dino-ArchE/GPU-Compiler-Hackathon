"""A reference interpreter for our IR, used to differential-test passes (never by compile.py).

Runs every thread of the launch as a coroutine; threads of a block meet at bar.sync and lanes
of a warp at shfl.sync and mma.sync (fragments gathered per the PTX layout), so reductions and
tensor-core tiles run too. cp.async copies complete at issue, a legal schedule of the hardware's. Registers hold raw bit patterns and each
instruction reinterprets them through its type, like the hardware does. Floats are rounded to
f32 after every op; .approx ops are computed exactly (compare with a tolerance).
"""

import math
import struct

import numpy as np

from core.ir import Address, BasicBlock, Immediate, Op, Param, PtxType, Register, SpecialRegister, StateSpace, Symbol, Vector

BITS = {"pred": 1, "b16": 16, "s16": 16, "u16": 16, "f16": 16, "b32": 32, "s32": 32, "u32": 32,
        "f32": 32, "b64": 64, "s64": 64, "u64": 64, "f64": 64}
WIDE = {"s16": "s32", "s32": "s64", "u16": "u32", "u32": "u64"}
NP_TYPES = {"f32": np.float32, "s32": np.int32, "u32": np.uint32, "f64": np.float64, "s64": np.int64}
SHARED_BASE = 0xFF << 32


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0] if math.isfinite(x) else x


def to_bits(value, t):
    n = BITS[t]
    if t == "f32":
        return struct.unpack("<I", struct.pack("<f", value))[0]
    if t == "f64":
        return struct.unpack("<Q", struct.pack("<d", value))[0]
    return int(value) & ((1 << n) - 1)


def from_bits(bits, t):
    n = BITS[t]
    if t == "f32":
        return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]
    if t == "f64":
        return struct.unpack("<d", struct.pack("<Q", bits))[0]
    bits &= (1 << n) - 1
    if t.startswith("s") and bits >> (n - 1):
        return bits - (1 << n)
    return bits


def f_div(a, b):
    """IEEE division: x/0 is a signed infinity, 0/0 is nan."""
    if b == 0.0:
        return math.nan if a == 0.0 or math.isnan(a) else math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def c_div(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


class Memory:
    """Byte-addressed buffers; a pointer's high bits pick the buffer."""

    def __init__(self):
        self.buffers = {}

    def add(self, key, data):
        self.buffers[key] = data
        return key << 32

    def locate(self, addr):
        buf = self.buffers[addr >> 32]
        off = addr & 0xFFFFFFFF
        return buf, off

    def read(self, addr, t):
        buf, off = self.locate(addr)
        n = BITS[t] // 8
        if off < 0 or off + n > len(buf):
            raise IndexError(f"read of {n} bytes at offset {off} outside a {len(buf)}-byte buffer")
        return int.from_bytes(buf[off:off + n], "little")

    def write(self, addr, t, bits):
        buf, off = self.locate(addr)
        n = BITS[t] // 8
        if off < 0 or off + n > len(buf):
            raise IndexError(f"write of {n} bytes at offset {off} outside a {len(buf)}-byte buffer")
        buf[off:off + n] = int(bits).to_bytes(n, "little")


class Thread:
    def __init__(self, function, memory, params, shared_syms, tid, ctaid, ntid, nctaid):
        self.f = function
        self.mem = memory
        self.params = params          # Param -> pointer value
        self.shared = shared_syms     # Symbol -> address
        self.regs = {}
        self.special = {"tid.x": tid, "ntid.x": ntid, "ctaid.x": ctaid, "nctaid.x": nctaid,
                        "laneid": tid % 32}
        self.tid = tid

    # ---------------------------------------------------------------- operand access

    def read(self, v, t):
        """Operand value typed as t (Python int / float / bool)."""
        if isinstance(v, Immediate):
            if t in ("f32", "f64"):
                return float(v.value)
            return from_bits(to_bits(int(v.value), t), t)
        if isinstance(v, SpecialRegister):
            return self.special[v.name]
        if isinstance(v, Symbol):
            return self.params[v] if isinstance(v, Param) else self.shared[v]
        if isinstance(v, Register):
            if v not in self.regs:
                raise RuntimeError(f"thread {self.tid}: {v} read before written")
            bits = self.regs[v]
            return bits if t == "pred" else from_bits(bits, t)
        raise TypeError(f"cannot read {v!r}")

    def write(self, reg, t, value):
        self.regs[reg] = bool(value) if t == "pred" else to_bits(value, t)

    def addr(self, a):
        base = a.base
        if isinstance(base, Symbol):
            return self.shared[base] + a.offset
        return self.read(base, "u64") + a.offset

    # ---------------------------------------------------------------- execution

    def run(self):
        """Coroutine: yields ('bar',) or ('shfl', mode, value, b, c) and receives shfl results."""
        blocks = self.f.blocks
        index = {b: i for i, b in enumerate(blocks)}
        bi, ii = 0, 0
        steps = 0
        while True:
            steps += 1
            if steps > 1_000_000:
                raise RuntimeError("interpreter step limit (infinite loop?)")
            if ii >= len(blocks[bi].instructions):
                bi, ii = bi + 1, 0
                if bi >= len(blocks):
                    raise RuntimeError("fell off the end of the kernel")
                continue
            inst = blocks[bi].instructions[ii]
            ii += 1
            if inst.guard is not None and self.regs[inst.guard] == inst.negate:
                continue
            op = inst.op
            if op is Op.RET:
                return
            if op is Op.BRA:
                bi, ii = index[inst.target], 0
                continue
            if op is Op.BAR:
                yield ("bar",)
                continue
            if op is Op.MMA:
                a, b, c = (v.elems for v in inst.srcs)
                frag = ([self.read(r, "b32") for r in a], [self.read(r, "b32") for r in b],
                        [self.read(r, "f32") for r in c])
                result = yield ("mma", frag)
                for r, v in zip(inst.dsts[0].elems, result):
                    self.write(r, "f32", v)
                continue
            if op is Op.SHFL:
                mode = inst.types[1]
                t = inst.types[-1]
                a, b, c = (self.read(v, t if k == 0 else "u32") for k, v in enumerate(inst.srcs[:3]))
                result = yield ("shfl", mode, a, b, c)
                self.write(inst.dsts[0], t, result)
                continue
            self.execute(inst)

    def execute(self, inst):
        op, types = inst.op, inst.types
        t = inst.ptx_type
        if op is Op.LD and inst.space is StateSpace.PARAM:
            self.write(inst.dsts[0], t, self.params[inst.srcs[0].base])   # the buffer's pointer
            return
        if op is Op.LD:
            dst, addr = inst.dsts[0], self.addr(inst.srcs[0])
            regs = dst.elems if isinstance(dst, Vector) else (dst,)
            n = BITS[t] // 8
            for k, r in enumerate(regs):
                self.regs[r] = self.mem.read(addr + k * n, t)
            return
        if op is Op.ST:
            addr, src = self.addr(inst.srcs[0]), inst.srcs[1]
            vals = src.elems if isinstance(src, Vector) else (src,)
            n = BITS[t] // 8
            for k, v in enumerate(vals):
                self.mem.write(addr + k * n, t, to_bits(self.read(v, t), t))
            return
        if op is Op.RED:
            addr = self.addr(inst.srcs[0])
            old = from_bits(self.mem.read(addr, t), t)
            new = old + self.read(inst.srcs[1], t)
            self.mem.write(addr, t, to_bits(f32(new) if t == "f32" else new, t))
            return

        if op is Op.CP:
            if "async" in types and len(inst.srcs) == 3:
                dst_a, src_a = self.addr(inst.srcs[0]), self.addr(inst.srcs[1])
                for k in range(0, int(inst.srcs[2].value), 4):
                    self.mem.write(dst_a + k, "b32", self.mem.read(src_a + k, "b32"))
            return                    # commit_group / wait_group: every copy is already done

        dst = inst.dsts[0]
        if op is Op.CVT and "tf32" in types:
            bits = to_bits(self.read(inst.srcs[0], "f32"), "f32")
            if (bits >> 23) & 0xFF != 0xFF:              # round to 10 mantissa bits, ties away
                bits = (bits + 0x1000) & 0xFFFFE000
            self.regs[dst] = bits
            return
        if op is Op.SETP:
            cmp = types[0]
            a, b = (self.read(v, t) for v in inst.srcs)
            res = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[cmp]
            self.write(dst, "pred", res)
            return
        if op is Op.SELP:
            a, b = (self.read(v, t) for v in inst.srcs[:2])
            self.write(dst, t, a if self.read(inst.srcs[2], "pred") else b)
            return
        if op is Op.CVTA or op is Op.MOV:
            self.write(dst, t, self.read(inst.srcs[0], t))
            return

        wide = "wide" in types
        rt = WIDE[t] if wide else t
        srcs = [self.read(v, t if not (op is Op.MAD and k == 2 and wide) else rt)
                for k, v in enumerate(inst.srcs)]
        is_f = t in ("f32", "f64")
        n = BITS[t]
        if op is Op.ADD:
            r = srcs[0] + srcs[1]
        elif op is Op.SUB:
            r = srcs[0] - srcs[1]
        elif op is Op.MUL:
            r = srcs[0] * srcs[1]
            if "hi" in types:
                r = r >> n
        elif op in (Op.MAD, Op.FMA):
            r = srcs[0] * srcs[1]
            if "hi" in types:
                r = r >> n
            r = r + srcs[2]
        elif op is Op.DIV:
            r = f_div(srcs[0], srcs[1]) if is_f else c_div(srcs[0], srcs[1])
        elif op is Op.REM:
            r = srcs[0] - c_div(srcs[0], srcs[1]) * srcs[1]
        elif op is Op.MIN:
            r = min(srcs)
        elif op is Op.MAX:
            r = max(srcs)
        elif op is Op.NEG:
            r = -srcs[0]
        elif op is Op.ABS:
            r = abs(srcs[0])
        elif op is Op.AND:
            r = srcs[0] & srcs[1]
        elif op is Op.OR:
            r = srcs[0] | srcs[1]
        elif op is Op.XOR:
            r = srcs[0] ^ srcs[1]
        elif op is Op.NOT:
            r = ~srcs[0]
        elif op is Op.SHL:
            r = srcs[0] << min(srcs[1], n)
        elif op is Op.SHR:
            a = srcs[0] if t.startswith("s") else srcs[0] & ((1 << n) - 1)
            r = a >> min(srcs[1], n)
        elif op is Op.RCP:
            r = 1.0 / srcs[0]
        elif op is Op.SQRT:
            r = math.sqrt(srcs[0])
        elif op is Op.RSQRT:
            r = 1.0 / math.sqrt(srcs[0])
        elif op is Op.EX2:
            r = 2.0 ** srcs[0]
        elif op is Op.LG2:
            r = math.log2(srcs[0])
        else:
            raise NotImplementedError(f"interp: {inst!r}")
        if is_f and t == "f32":
            r = f32(r)
        self.write(dst, rt, r)


def run_kernel(module, inputs, outputs, grid=None, block=None):
    """Run the module's kernel. inputs/outputs: {param name: np.ndarray} (outputs are filled and
    returned as new arrays). grid/block default to the function's launch."""
    f = module.functions[0]
    grid = grid or f.launch_grid
    block = block or f.launch_block
    mem = Memory()
    params = {}
    arrays = {**inputs, **outputs}
    for k, (name, param) in enumerate(f.params.items(), start=1):
        params[param] = mem.add(k, bytearray(np.ascontiguousarray(arrays[name]).tobytes()))

    shared_layout, off = {}, 0
    for sym, align, nbytes in module.shared:
        off = (off + align - 1) // align * align
        shared_layout[sym] = off
        off += nbytes

    nthreads = block[0]
    for cta in range(grid[0]):
        shared_key = 0xFF0000 + cta
        base = mem.add(shared_key, bytearray(max(off, 1)))
        syms = {s: base + o for s, o in shared_layout.items()}
        threads = [Thread(f, mem, params, syms, t, cta, nthreads, grid[0]) for t in range(nthreads)]
        _run_cta(threads)
        del mem.buffers[shared_key]

    result = {}
    for k, (name, param) in enumerate(f.params.items(), start=1):
        if name in outputs:
            buf = mem.buffers[k]
            result[name] = np.frombuffer(bytes(buf), dtype=outputs[name].dtype).reshape(outputs[name].shape).copy()
    return result


def _run_cta(threads):
    gens = {t.tid: t.run() for t in threads}
    pending = {}          # tid -> event
    resume = {tid: None for tid in gens}
    while gens:
        for tid in list(resume):
            value = resume.pop(tid)
            try:
                pending[tid] = gens[tid].send(value)
            except StopIteration:
                del gens[tid]
        if not gens:
            break
        # warps whose live lanes all reached a shfl exchange values
        warps = {}
        for tid in gens:
            warps.setdefault(tid // 32, []).append(tid)
        progressed = False
        for lanes in warps.values():
            if all(pending.get(t, ("",))[0] == "mma" for t in lanes):
                if len(lanes) != 32:
                    raise RuntimeError(f"mma.sync with only {len(lanes)} live lanes")
                frags = {t % 32: pending.pop(t)[1] for t in lanes}
                for lane, d in _mma_m16n8k8(frags).items():
                    resume[lanes[0] - lanes[0] % 32 + lane] = d
                progressed = True
                continue
            if all(pending.get(t, ("",))[0] == "shfl" for t in lanes):
                values = {t % 32: pending[t][2] for t in lanes}
                for t in lanes:
                    _, mode, _, b, c = pending.pop(t)
                    lane = t % 32
                    src = {"bfly": lane ^ b, "down": lane + b, "up": lane - b, "idx": b}[mode]
                    if not (0 <= src < 32) or src not in values or (mode == "down" and src > (c & 31)):
                        src = lane
                    resume[t] = values[src]
                progressed = True
        if not progressed and all(pending.get(t, ("",))[0] == "bar" for t in gens):
            for t in gens:
                pending.pop(t)
                resume[t] = None
            progressed = True
        if not progressed:
            raise RuntimeError(f"deadlock: {set(e[0] for e in pending.values())}")


def _tf32(bits):
    return from_bits(bits & 0xFFFFE000, "f32")      # the tensor core ignores the low 13 bits


def _mma_m16n8k8(frags):
    """frags: lane -> (a[4], b[2], c[4]) -> lane -> d[4], for the layout documented on Mma."""
    A, B, C = np.zeros((16, 8)), np.zeros((8, 8)), np.zeros((16, 8))
    for lane, (a, b, c) in frags.items():
        g, t = lane >> 2, lane & 3
        A[g, t], A[g + 8, t], A[g, t + 4], A[g + 8, t + 4] = map(_tf32, a)
        B[t, g], B[t + 4, g] = map(_tf32, b)
        C[g, 2 * t], C[g, 2 * t + 1], C[g + 8, 2 * t], C[g + 8, 2 * t + 1] = c
    D = (A @ B + C).astype(np.float32)
    out = {}
    for lane in frags:
        g, t = lane >> 2, lane & 3
        out[lane] = [float(D[g, 2 * t]), float(D[g, 2 * t + 1]), float(D[g + 8, 2 * t]), float(D[g + 8, 2 * t + 1])]
    return out
