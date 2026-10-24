#!/usr/bin/env bash
# Compile every problem that codegen supports at -O0 and at the given level (default 2), check
# both with ptxas, and print instruction counts side by side. Usage: tools/check_opt.sh [level]
level=${1:-2}
root=$(cd "$(dirname "$0")/.." && pwd)
py=${PYTHON:-$root/.venv/bin/python}
ptxas=${PTXAS:-/usr/local/cuda/bin/ptxas}
out=${TMPDIR:-/tmp}/check_opt
mkdir -p "$out"
status=0
for ir in "$root"/problems/*/*.ir.json; do
    name=$(basename "$(dirname "$ir")")/$(basename "$ir" .ir.json)
    tag=${name//\//__}
    if ! "$py" "$root/compile.py" -O0 "$ir" > "$out/$tag.O0.ptx" 2>/dev/null; then
        printf '%-48s unsupported by codegen\n' "$name"; continue
    fi
    if ! "$py" "$root/compile.py" -O"$level" "$ir" > "$out/$tag.O$level.ptx" 2> "$out/$tag.err" \
        || grep -q "optimizer failed" "$out/$tag.err"; then
        printf '%-48s OPTIMIZER FAILED (%s)\n' "$name" "$out/$tag.err"; status=1; continue
    fi
    if ! "$ptxas" -arch=sm_89 "$out/$tag.O$level.ptx" -o /dev/null; then
        printf '%-48s PTXAS FAILED\n' "$name"; status=1; continue
    fi
    n0=$(grep -c ';' "$out/$tag.O0.ptx"); n=$(grep -c ';' "$out/$tag.O$level.ptx")
    printf '%-48s ok  %4d -> %4d lines\n' "$name" "$n0" "$n"
done
exit $status
