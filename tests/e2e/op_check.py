#!/usr/bin/env python3
"""Accuracy of each elementwise op on the GPU runner, against numpy in float64.

Takes p03 (`y * relu(x)`), swaps `relu` (unary ops, with y = 1) or `multiply` (binary ops) for
the op under test, sends our own input data (the runner accepts `data` in place of `seed`), and
compares at rtol 1e-4 / atol 1e-5. Input ranges include overflow regions (exp of 120, ...).

    python tests/e2e/op_check.py            # every op
    python tests/e2e/op_check.py tanh pow   # just these
"""
import json, shutil, subprocess, sys, tempfile, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from submit import post, RUNNER  # noqa: E402

P03 = ROOT / 'problems' / 'p03_relu_mul'
_erf = np.vectorize(math.erf)
rng = np.random.default_rng(0)
N = 2048

def ref_pow(x, y):
    with np.errstate(all='ignore'):
        return np.power(x.astype(np.float64), y.astype(np.float64))

UNARY = {  # name: (input domain sampler, numpy reference)
    'sigmoid': (lambda: rng.uniform(-200, 200, N), lambda x: 1 / (1 + np.exp(-x))),
    'tanh':    (lambda: rng.uniform(-100, 100, N), np.tanh),
    'erf':     (lambda: rng.uniform(-30, 30, N), _erf),
    'log':     (lambda: rng.uniform(1e-3, 1e3, N), np.log),
    'rsqrt':   (lambda: rng.uniform(1e-3, 1e3, N), lambda x: 1 / np.sqrt(x)),
    'sin':     (lambda: rng.uniform(-50, 50, N), np.sin),
    'cos':     (lambda: rng.uniform(-50, 50, N), np.cos),
    'exp':     (lambda: rng.uniform(-120, 120, N), np.exp),
    'exp_fast':(lambda: rng.uniform(-20, 20, N), np.exp),
    'square':  (lambda: rng.uniform(-10, 10, N), np.square),
    'negative':(lambda: rng.uniform(-10, 10, N), np.negative),
    'abs':     (lambda: rng.uniform(-10, 10, N), np.abs),
    'sqrt':    (lambda: rng.uniform(0, 1e3, N), np.sqrt),
    'reciprocal': (lambda: rng.uniform(0.1, 10, N) * rng.choice([-1, 1], N), lambda x: 1 / x),
    'bitwise_not': (lambda: rng.choice([0.0, 1.0, 2.5], N), lambda x: (x == 0).astype(float)),
}
BINARY = {
    'pow':   (lambda: rng.uniform(-3, 3, N), lambda: np.round(rng.uniform(-3, 3, N)) * rng.choice([1, 1, 0.5], N), ref_pow),
    'greater': (lambda: rng.integers(0, 4, N).astype(float), lambda: rng.integers(0, 4, N).astype(float), lambda a, b: (a > b).astype(float)),
    'equal':   (lambda: rng.integers(0, 4, N).astype(float), lambda: rng.integers(0, 4, N).astype(float), lambda a, b: (a == b).astype(float)),
    'less_equal': (lambda: rng.integers(0, 4, N).astype(float), lambda: rng.integers(0, 4, N).astype(float), lambda a, b: (a <= b).astype(float)),
    'bitwise_and': (lambda: rng.choice([0.0, 1.0], N), lambda: rng.choice([0.0, 2.0], N), lambda a, b: ((a != 0) & (b != 0)).astype(float)),
    'bitwise_or':  (lambda: rng.choice([0.0, 1.0], N), lambda: rng.choice([0.0, 2.0], N), lambda a, b: ((a != 0) | (b != 0)).astype(float)),
    'minimum': (lambda: rng.normal(size=N), lambda: rng.normal(size=N), np.minimum),
    'subtract': (lambda: rng.normal(size=N), lambda: rng.normal(size=N), np.subtract),
    'divide': (lambda: rng.normal(size=N), lambda: rng.uniform(0.5, 2, N), np.divide),
}

def run(op):
    d = Path(tempfile.mkdtemp())
    shutil.copytree(P03, d / 'p', dirs_exist_ok=True)
    ir = next((d / 'p').glob('*.ir.json'))
    meta = json.loads((d / 'p' / 'meta.json').read_text())['kernels'][0]
    t = ir.read_text()
    if op in UNARY:
        gen, ref = UNARY[op]
        t = t.replace("ElementwiseImpl('relu')", f"ElementwiseImpl('{op}')")
        x, y = gen(), np.ones(N)
        expect = ref(x.astype(np.float32).astype(np.float64))
    else:
        gx, gy, ref = BINARY[op]
        t = t.replace("ElementwiseImpl('relu')", "ElementwiseImpl('copy')")
        t = t.replace("ElementwiseImpl('multiply')", f"ElementwiseImpl('{op}')")
        x, y = gx(), gy()
        # p03: v1 = multiply(in1=y, v0=copy(x))  -> op(y, x)
        expect = ref(y.astype(np.float32).astype(np.float64), x.astype(np.float32).astype(np.float64))
    ir.write_text(t)
    r = subprocess.run([sys.executable, str(ROOT / 'compile.py'), str(ir)], capture_output=True, text=True)
    if r.returncode:
        return f'compile error: {r.stderr.strip().splitlines()[-1]}'
    c = post(RUNNER, '/compile', {'ptx': r.stdout})
    if not c.get('ok'):
        return 'ptxas: ' + c.get('stderr', '')[:300]
    args = []
    for a in meta['args']:
        a = {k: v for k, v in a.items() if k != 'seed'}
        if a['name'] == 'x': a['data'] = x.astype(np.float32).tolist()
        if a['name'] == 'y': a['data'] = y.astype(np.float32).tolist()
        args.append(a)
    out = post(RUNNER, '/run', {'cubin_id': c['cubin_id'], 'entry': meta['entry'], 'grid': meta['grid'],
              'block': meta['block'], 'smem_bytes': 0, 'zero_outputs': [], 'args': args, 'max_return': N})
    if not out.get('ok'):
        return 'run: ' + str(out)[:300]
    # the runner's JSON spells both inf and NaN as null
    got = np.array([np.nan if v is None else v for v in out['outputs']['mul']['values']], dtype=np.float64)
    expect = np.where(np.isfinite(expect.astype(np.float32)), expect, np.nan)
    nan_ok = np.isnan(expect) == np.isnan(got)
    fin = ~np.isnan(expect)
    rel = np.abs(got[fin] - expect[fin]) / np.maximum(np.abs(expect[fin]), 1e-30)
    ok = nan_ok.all() and np.allclose(got[fin], expect[fin], rtol=1e-4, atol=1e-5)
    if not ok:
        bad = np.where(~nan_ok | (fin & ~np.isclose(got, np.nan_to_num(expect), rtol=1e-4, atol=1e-5)))[0][:6]
        base, ex = (y, x) if op in BINARY else (x, None)
        print('   bad:', [(float(np.float32(base[i])), None if ex is None else float(np.float32(ex[i])), float(expect[i]), float(got[i])) for i in bad])
    return f"{'PASS' if ok else 'FAIL'}  max abs {np.max(np.abs(got[fin]-expect[fin])):.2e}  max rel {np.max(rel):.2e}  nan-match {nan_ok.all()}"

for op in (sys.argv[1:] or [*UNARY, *BINARY]):
    print(f'{op:12s}', run(op), flush=True)
