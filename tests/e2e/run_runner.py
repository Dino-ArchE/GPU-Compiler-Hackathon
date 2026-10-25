#!/usr/bin/env python3
"""Run every e2e problem (and optionally the open set) through submit.py on the GPU runner.

This is the evaluation pipeline itself: compile.py on each IR, ptxas on the runner, launch under
meta.json, compare with expected.npz. Usage:

    python tests/e2e/run_runner.py                 # all tests/e2e/problems at the default -O2
    python tests/e2e/run_runner.py -O0 --open      # unoptimized, plus problems/
    python tests/e2e/run_runner.py ep_gelu em_odd  # just these
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E2E = Path(__file__).resolve().parent / "problems"


def problem_dirs(names, include_open):
    dirs = sorted(p for p in E2E.iterdir() if (p / "meta.json").exists())
    if include_open:
        dirs = sorted(p for p in (ROOT / "problems").iterdir() if (p / "meta.json").exists()) + dirs
    if names:
        dirs = [p for p in dirs if p.name in names]
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="problem ids (default: all)")
    ap.add_argument("-O", dest="level", type=int, choices=(0, 1, 2), default=None)
    ap.add_argument("--open", action="store_true", help="also run the open problems/")
    args = ap.parse_args()

    compiler = [str(ROOT / "compile.py")]
    wrapper = None
    if args.level is not None:
        # submit.py runs `python <compiler> <ir>`: pin the level through a tiny wrapper
        wrapper = Path(__file__).resolve().parent / f".compile_O{args.level}.py"
        wrapper.write_text(f"import runpy, sys\nsys.argv[1:1] = ['-O{args.level}']\n"
                           f"runpy.run_path({str(ROOT / 'compile.py')!r}, run_name='__main__')\n")
        compiler = [str(wrapper)]

    failed = []
    dirs = problem_dirs(set(args.names), args.open)
    for p in dirs:
        r = subprocess.run([sys.executable, str(ROOT / "submit.py"), str(p), "--compiler", *compiler],
                           capture_output=True, text=True)
        lines = [ln for ln in r.stdout.splitlines() if ("PASS" in ln or "FAIL" in ln or "rejected" in ln
                                                        or "failed" in ln or "launch" in ln)]
        summary = lines[-1] if lines else (r.stdout + r.stderr).strip().splitlines()[-1:]
        status = "ok  " if r.returncode == 0 else "FAIL"
        if r.returncode:
            failed.append(p.name)
        print(f"{status} {p.name:<22} {str(summary)[:110]}", flush=True)
    if wrapper:
        wrapper.unlink()
    print(f"\n{len(dirs) - len(failed)}/{len(dirs)} passed" + (f"; failed: {' '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
