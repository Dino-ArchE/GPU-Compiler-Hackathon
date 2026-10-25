#!/usr/bin/env bash
# Submit every problem codegen supports to the runner with compile.py (default -O2).
# Usage: tools/run_runner.sh [problem ...]
root=$(cd "$(dirname "$0")/.." && pwd)
py=${PYTHON:-$root/.venv/bin/python}
problems=("$@")
if [ ${#problems[@]} -eq 0 ]; then
    for d in "$root"/problems/*/; do
        ir=$(ls "$d"*.ir.json | head -1)
        "$py" "$root/compile.py" -O0 "$ir" > /dev/null 2>&1 && problems+=("$(basename "$d")")
    done
fi
status=0
for p in "${problems[@]}"; do
    res=$("$py" "$root/submit.py" "$p" --compiler "$root/compile.py" 2>&1)
    code=$?
    printf '%-20s %s\n' "$p" "$(echo "$res" | grep -E 'registers|PASS|FAIL|rror' | tr '\n' ' ')"
    [ $code -ne 0 ] && status=1
done
exit $status
