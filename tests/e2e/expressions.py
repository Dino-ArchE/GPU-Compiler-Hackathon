"""Hidden-set-style variations of the five categories, in tools/generate.py's spec format.

The hidden set has "the same categories, different shapes, sizes and op mixes". These specs
probe exactly that: sizes that are not multiples of the block, ops the open set never uses,
broadcasting, column reductions, odd matmul shapes, unequal attention lengths. Each spec is
{id, code, ref, rtol, atol[, knobs]}; `ref` gets the seeded inputs in the traced graph's
input order and returns the expected outputs.

Regenerate the packages (needs torch, see tests/e2e/README.md):
    python tools/generate.py --emmy .venv/bin/emmy --out tests/e2e/problems tests/e2e/expressions.py
"""

import math

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"   # the schedule the open reductions pin
_erf = np.vectorize(math.erf)


def d(x):
    return x.astype(np.float64)


def f(x):
    return np.asarray(x).astype(np.float32)


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _attention(q, kt, v, scale):
    return f(_softmax((d(q) @ d(kt)) / scale) @ d(v))


PW = dict(rtol=1e-4, atol=1e-5)          # pointwise with transcendental ops
RED = dict(rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS)
MM = dict(rtol=1e-3, atol=1e-3)
FUSED = dict(rtol=1e-3, atol=1e-4)

OPEN = [
    # ---- category 1: pointwise -------------------------------------------
    dict(id="ep_odd_add", code="torch.randn(1000) + torch.randn(1000)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="ep_tiny_mul", code="torch.randn(33) * torch.randn(33)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="ep_three_inputs",
         code="a = torch.randn(4096); b = torch.randn(4096); c = torch.randn(4096); a * b + c",
         ref=lambda a, b, c: [f(d(a) * d(b) + d(c))], rtol=1e-6, atol=1e-6),
    dict(id="ep_sigmoid", code="torch.sigmoid(torch.randn(4096))",
         ref=lambda x: [f(1 / (1 + np.exp(-d(x))))], **PW),
    dict(id="ep_tanh_mul", code="x = torch.randn(2048); y = torch.randn(2048); torch.tanh(x) * y",
         ref=lambda x, y: [f(np.tanh(d(x)) * d(y))], **PW),
    dict(id="ep_gelu", code="F.gelu(torch.randn(3000))",
         ref=lambda x: [f(0.5 * d(x) * (1 + _erf(d(x) / math.sqrt(2))))], **PW),
    dict(id="ep_silu", code="F.silu(torch.randn(2500))",
         ref=lambda x: [f(d(x) / (1 + np.exp(-d(x))))], **PW),
    dict(id="ep_softplus_like", code="torch.log(torch.exp(torch.randn(2048)) + 1.0)",
         ref=lambda x: [f(np.log(np.exp(d(x)) + 1.0))], **PW),
    dict(id="ep_sqrt_abs", code="torch.sqrt(torch.abs(torch.randn(4096)))",
         ref=lambda x: [f(np.sqrt(np.abs(d(x))))], **PW),
    dict(id="ep_rsqrt", code="torch.rsqrt(torch.abs(torch.randn(4096)) + 1.0)",
         ref=lambda x: [f(1 / np.sqrt(np.abs(d(x)) + 1.0))], **PW),
    dict(id="ep_leaky_where", code="x = torch.randn(4096); torch.where(x > 0, x, x * 0.1)",
         ref=lambda x: [f(np.where(x > 0, d(x), d(x) * 0.1))], **PW),
    dict(id="ep_clamp", code="torch.clamp(torch.randn(4096), -0.5, 0.5)",
         ref=lambda x: [np.clip(x, -0.5, 0.5)], rtol=0, atol=0),
    dict(id="ep_maximum", code="torch.maximum(torch.randn(3000), torch.randn(3000))",
         ref=lambda a, b: [np.maximum(a, b)], rtol=0, atol=0),
    dict(id="ep_div_const", code="torch.randn(4096) / 3.0 - 1.5",
         ref=lambda x: [f(d(x) / 3.0 - 1.5)], **PW),
    dict(id="ep_cube", code="torch.pow(torch.randn(4096), 3.0)",
         ref=lambda x: [f(d(x) ** 3)], **PW),
    dict(id="ep_sin_cos", code="x = torch.randn(4096); y = torch.randn(4096); torch.sin(x) + torch.cos(y)",
         ref=lambda x, y: [f(np.sin(d(x)) + np.cos(d(y)))], **PW),
    dict(id="ep_row_bias", code="x = torch.randn(64, 256); b = torch.randn(256); x + b",
         ref=lambda x, b: [x + b], rtol=0, atol=0),
    dict(id="ep_col_scale", code="x = torch.randn(64, 256); s = torch.randn(64, 1); x * s",
         ref=lambda x, s: [x * s], rtol=0, atol=0),

    # ---- category 2: reductions ------------------------------------------
    dict(id="er_odd_sum", code="torch.randn(1000).sum()",
         ref=lambda x: [f(d(x).sum())], **RED),
    dict(id="er_max", code="torch.randn(4096).amax()",
         ref=lambda x: [x.max()], **RED),
    dict(id="er_row_max", code="torch.randn(32, 1000).amax(-1)",
         ref=lambda x: [x.max(-1)], **RED),
    dict(id="er_row_mean", code="torch.randn(128, 256).mean(-1)",
         ref=lambda x: [f(d(x).mean(-1))], **RED),
    dict(id="er_sum_squares", code="x = torch.randn(8192); (x * x).sum()",
         ref=lambda x: [f((d(x) ** 2).sum())], rtol=1e-4, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="er_abs_sum", code="torch.randn(5000).abs().sum()",
         ref=lambda x: [f(np.abs(d(x)).sum())], rtol=1e-4, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="er_col_sum", code="torch.randn(256, 64).sum(0)",
         ref=lambda x: [f(d(x).sum(0))], **RED),
    dict(id="er_long_rows", code="torch.randn(16, 8192).sum(-1)",
         ref=lambda x: [f(d(x).sum(-1))], rtol=1e-4, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="er_many_rows", code="torch.randn(1024, 64).sum(-1)",
         ref=lambda x: [f(d(x).sum(-1))], **RED),

    # ---- category 3: matmul ----------------------------------------------
    dict(id="em_256", code="torch.randn(256, 256) @ torch.randn(256, 256)",
         ref=lambda a, b: [f(d(a) @ d(b))], **MM),
    dict(id="em_small", code="torch.randn(64, 128) @ torch.randn(128, 32)",
         ref=lambda a, b: [f(d(a) @ d(b))], **MM),
    dict(id="em_tall", code="torch.randn(512, 64) @ torch.randn(64, 128)",
         ref=lambda a, b: [f(d(a) @ d(b))], **MM),
    dict(id="em_deep_k", code="torch.randn(64, 1024) @ torch.randn(1024, 64)",
         ref=lambda a, b: [f(d(a) @ d(b))], **MM),
    dict(id="em_odd", code="torch.randn(96, 200) @ torch.randn(200, 72)",
         ref=lambda a, b: [f(d(a) @ d(b))], **MM),

    # ---- category 4: fused -----------------------------------------------
    dict(id="ef_softmax_512", code="torch.softmax(torch.randn(64, 512), dim=-1)",
         ref=lambda x: [f(_softmax(d(x)))], rtol=1e-3, atol=1e-5),
    dict(id="ef_softmax_odd", code="torch.softmax(torch.randn(16, 1000), dim=-1)",
         ref=lambda x: [f(_softmax(d(x)))], rtol=1e-3, atol=1e-5),
    dict(id="ef_log_softmax", code="torch.log(torch.softmax(torch.randn(64, 256), dim=-1))",
         ref=lambda x: [f(np.log(_softmax(d(x))))], **FUSED),
    dict(id="ef_layernorm",
         code="x = torch.randn(32, 512); w = torch.randn(512); b = torch.randn(512); "
              "(x - x.mean(-1, keepdim=True)) / torch.sqrt(((x - x.mean(-1, keepdim=True)) ** 2)"
              ".mean(-1, keepdim=True) + 1e-5) * w + b",
         ref=lambda x, w, b: [f((d(x) - d(x).mean(-1, keepdims=True))
                                / np.sqrt(d(x).var(-1, keepdims=True) + 1e-5) * d(w) + d(b))], **FUSED),
    dict(id="ef_rmsnorm_2048",
         code="x = torch.randn(16, 2048); w = torch.randn(2048); "
              "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w",
         ref=lambda x, w: [f(d(x) / np.sqrt((d(x) ** 2).mean(-1, keepdims=True) + 1e-6) * d(w))], **FUSED),
    dict(id="ef_center", code="x = torch.randn(64, 384); x - x.mean(-1, keepdim=True)",
         ref=lambda x: [f(d(x) - d(x).mean(-1, keepdims=True))], **FUSED),
    dict(id="ef_matmul_relu", code="torch.relu(torch.randn(128, 128) @ torch.randn(128, 128))",
         ref=lambda a, b: [f(np.maximum(d(a) @ d(b), 0))], **MM),

    # ---- category 5: attention-like --------------------------------------
    dict(id="ea_small",
         code="q = torch.randn(32, 16); kt = torch.randn(16, 32); v = torch.randn(32, 16); "
              "torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: [_attention(q, kt, v, 4.0)], **FUSED),
    dict(id="ea_256",
         code="q = torch.randn(256, 64); kt = torch.randn(64, 256); v = torch.randn(256, 64); "
              "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [_attention(q, kt, v, 8.0)], **FUSED),
    dict(id="ea_rect",
         code="q = torch.randn(64, 32); kt = torch.randn(32, 128); v = torch.randn(128, 32); "
              "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: [_attention(q, kt, v, 5.656854)], **FUSED),
]
