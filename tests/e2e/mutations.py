#!/usr/bin/env python3
"""Build mutated copies of open problems that exercise IR fields no open problem uses.

Each mutation keeps expected.npz valid, so the runner still judges the result:
  * raster_group G on m01: grouped CTA order permutes which block computes which tile;
    G = 3 makes the last group ragged (the min(G, Ea - grp*G) path), G = 4 divides evenly.
  * a larger block on r01/r03: the extra threads have no cell, so the Tile guard must
    branch around barriers and shuffles must use the active-lane mask.
  * named barriers (bar.sync id, count) and an && condition on r01.

    python tests/e2e/mutations.py      # writes tests/e2e/problems/mx_*
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "problems"

LANE0 = "cond=BinaryExpr(op='==', left=Var(name='lane'), right=Literal(value=0, dtype='int'))"
LANE0_AND_WARP = ("cond=BinaryExpr(op='&&', left=BinaryExpr(op='==', left=Var(name='lane'), "
                  "right=Literal(value=0, dtype='int')), right=BinaryExpr(op='<', left=Var(name='warp'), "
                  "right=Literal(value=4, dtype='int')))")


def mutate(src, name, ir_edits=(), block=None):
    dst = OUT / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ROOT / "problems" / src, dst)
    for f in list(dst.glob("*.ref.cu")):
        f.unlink()                         # the reference .cu describes the unmutated kernel
    for ir in dst.glob("*.ir.json"):
        text = ir.read_text()
        for old, new in ir_edits:
            assert text.count(old) >= 1, f"{name}: {old!r} not in {ir.name}"
            text = text.replace(old, new)
        ir.write_text(text)
    meta = json.loads((dst / "meta.json").read_text())
    meta["id"] = name
    if block:
        for k in meta["kernels"]:
            k["block"] = block
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", dst.relative_to(ROOT))


def main():
    for g in (3, 4):
        for orient in ("m", "n"):
            mutate("m01_matmul_512", f"mx_raster_g{g}_{orient}",
                   [("raster_group=None, raster_orient='m'", f"raster_group={g}, raster_orient='{orient}'")])
    mutate("r01_array_sum", "mx_guard_r01", block=[256, 1, 1])
    mutate("r03_mean", "mx_guard_r03", block=[256, 1, 1])
    mutate("r01_array_sum", "mx_named_barriers", [
        ("Sync(barrier_id=0, count=None, warp=False)", "Sync(barrier_id=1, count=128, warp=False)"),
        ("barrier_id=0, barrier_count=None", "barrier_id=2, barrier_count=128"),
        (LANE0, LANE0_AND_WARP),
    ])


if __name__ == "__main__":
    main()
