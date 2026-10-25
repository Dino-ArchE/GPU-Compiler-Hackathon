"""Kernel IR -> PTX.  Usage: python compile.py [-O0|-O1|-O2] path/to/kernel.ir.json > out.ptx

Optimizer flags (all optional; the default is -O2):
    -O0 / -O1 / -O2         pipeline from src/core/opt/pipeline.py (-O0 = codegen output as is)
    --passes a,b,c          run exactly these passes instead of a level's pipeline
    --disable a,b           drop these passes from the pipeline (to find a pass that hurts)
    --print-after a,b|all   dump the function to stderr after these passes (when they change it)
    --stats                 per-pass table on stderr: runs, changes, instructions removed
    --no-verify             skip the IR verifier between passes
"""

import argparse
import sys
import traceback
from pathlib import Path

# run from a fresh clone with no install: find our package under src/
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from core.frontend import KernelInfo  # noqa: E402
from core.opt import PassManager, build_pipeline  # noqa: E402
from core.ptx import CodeGen, print_module  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Kernel IR -> PTX")
    ap.add_argument("ir", help="path to <kernel>.ir.json (meta.json must sit next to it)")
    ap.add_argument("-O", dest="level", type=int, choices=(0, 1, 2), default=2)
    ap.add_argument("--passes", help="comma-separated pass list, overrides -O")
    ap.add_argument("--disable", default="", help="comma-separated passes to leave out")
    ap.add_argument("--print-after", default="", help="comma-separated pass names, or 'all'")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    kernel = KernelInfo(args.ir)
    module = CodeGen(kernel).gen()

    passes = args.passes.split(",") if args.passes else None
    pm = PassManager(build_pipeline(args.level, passes, [p for p in args.disable.split(",") if p]), verify=not args.no_verify,
                     print_after=[p for p in args.print_after.split(",") if p], stats=args.stats)
    try:
        pm.run(module)
    except Exception:
        # an optimizer bug must not cost a correct kernel: fall back to the unoptimized code
        traceback.print_exc()
        print("compile.py: optimizer failed, emitting unoptimized PTX", file=sys.stderr)
        module = CodeGen(kernel).gen()
    pm.report()
    sys.stdout.write(print_module(module))


if __name__ == "__main__":
    main()
