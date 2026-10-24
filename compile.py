"""Kernel IR -> PTX.  Usage: python compile.py path/to/kernel.ir.json > out.ptx"""

import sys
from pathlib import Path

# run from a fresh clone with no install: find our package under src/
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from core.frontend import KernelInfo  # noqa: E402
from core.ptx import CodeGen, print_module  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python compile.py path/to/kernel.ir.json > out.ptx")
    kernel = KernelInfo(sys.argv[1])
    module = CodeGen(kernel).gen()
    sys.stdout.write(print_module(module))


if __name__ == "__main__":
    main()
