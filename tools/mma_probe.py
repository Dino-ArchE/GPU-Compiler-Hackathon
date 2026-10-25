"""Hardware check of the m16n8k8 tf32 fragment layout, disguised as p01 (out = x0 + x1, exact).

Each warp takes a 16x8 tile of x0 as A (columns permuted by s(k) = (3k+1) & 7), multiplies by
the permutation matrix P (P[k][s(k)] = 1), so D[m][s(k)] = A[m][k] = x0[m*8 + s(k)] exactly when
x0 is split into three exact tf32 pieces. Any mistake in the A, B or C/D layout gives wrong
numbers. Run it on the hosted GPU:

    python tools/mma_probe.py > /tmp/probe.ptx && python submit.py p01_vector_add /tmp/probe.ptx

PASS with max abs err 0 means the layout documented on core.ir.Mma matches the hardware.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.ir import (  # noqa: E402
    CTAID_X, NTID_X, TID_X, Address, Builder, Cmp, Function, Immediate, Instruction, Module, Mov,
    Op, PtxType, StateSpace,
)

F, S, U, B32 = PtxType.F32, PtxType.S32, PtxType.U32, PtxType.B32


def build():
    m = Module()
    f = m.add_function(Function("k_add_pointwise", ["x0", "x1", "add"]))
    f.launch_grid, f.launch_block = (2, 1, 1), (256, 1, 1)
    b = Builder(f)
    x0, x1, out = (b.cvta_to_global(b.load_param(f.param(p))) for p in ("x0", "x1", "add"))
    tid = b.mov(U, TID_X)
    gid = b.mad(S, b.mov(U, CTAID_X), b.mov(U, NTID_X), tid)
    lane = b.binary(Op.AND, B32, tid, Immediate(31, B32))
    tile = b.binary(Op.AND, B32, b.binary(Op.SHR, U, gid, Immediate(5, U)), Immediate(7, B32))
    base = b.mul(S, tile, Immediate(128, S))
    g = b.binary(Op.SHR, U, lane, Immediate(2, U))
    t = b.binary(Op.AND, B32, lane, Immediate(3, B32))

    def sigma(k):
        return b.binary(Op.AND, B32, b.mad(S, k, Immediate(3, S), Immediate(1, S)), Immediate(7, B32))

    def addr(buf, row, col):
        idx = b.add(S, base, b.add(S, b.mul(S, row, Immediate(8, S)), col))
        return Address(b.add(PtxType.S64, buf, b.mul_wide(idx, Immediate(4, S))))

    g8 = b.add(S, g, Immediate(8, S))
    t4 = b.add(S, t, Immediate(4, S))
    a_vals = [b.load(StateSpace.GLOBAL, F, addr(x0, r, sigma(k))) for r, k in ((g, t), (g8, t), (g, t4), (g8, t4))]

    def as_f32(bits):
        v = b.reg(F)
        b.append(Mov(B32, v, bits))
        return v

    pieces = [[], [], []]                    # hi, mid, lo fragments
    for v in a_vals:
        r = v
        for p in range(3):
            bits = b.cvt_tf32(r)
            pieces[p].append(bits)
            if p < 2:
                r = b.binary(Op.SUB, F, r, as_f32(bits))

    one = Immediate(0x3F800000, B32)
    zero = Immediate(0, B32)
    bfrag = []
    for k in (t, t4):                        # b0 = P[t][g], b1 = P[t+4][g]
        hit = b.setp(Cmp.EQ, U, sigma(k), g)
        bfrag.append(_selp(b, one, zero, hit))

    acc = [b.mov(F, Immediate(0.0, F)) for _ in range(4)]
    for p in range(3):
        acc = b.mma_tf32(pieces[p], bfrag, acc, d=acc)

    two_t = b.mul(S, t, Immediate(2, S))
    two_t1 = b.add(S, two_t, Immediate(1, S))
    for v, (r, c) in zip(acc, ((g, two_t), (g, two_t1), (g8, two_t), (g8, two_t1))):
        y = b.load(StateSpace.GLOBAL, F, addr(x1, r, c))
        b.store(StateSpace.GLOBAL, F, addr(out, r, c), b.add(F, v, y))
    b.inst(Op.RET)
    return m


def _selp(b, x, y, p):
    d = b.reg(B32)
    b.append(Instruction(Op.SELP, (B32,), [d], [x, y, p]))
    return d


if __name__ == "__main__":
    sys.stdout.write(str(build()))
