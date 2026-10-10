"""Minimal NPU functional tests for prefix_attn_dn.

Four cases against an independent fp64 golden (dedup prefix-in-Q causal
semantics): one single-batch smoke shape, one GQA multi-batch shape, and the
two production uniform shapes (p20/130x10 fp16, p25/193x10 bf16).
Tolerances match the full regression suite: fp16 0.003 / bf16 0.025.
"""
import math

import pytest
import torch

from prefix_attn import prefix_attn_dn

D = 64


def golden_f64(q, k, v, bounds, scale):
    out = torch.zeros(q.shape, dtype=torch.float64)
    p = bounds[0]
    group = q.shape[1] // k.shape[1]
    for h in range(q.shape[1]):
        kh = k[:p, h // group].double()
        vh = v[:p, h // group].double()
        spans = [(0, p, 0)] + [(bounds[i], bounds[i + 1], p)
                               for i in range(len(bounds) - 1)]
        for lo, hi, row_offset in spans:
            qh = q[lo:hi, h].double()
            if lo:
                kh_use = torch.cat((kh, k[lo:hi, h // group].double()))
                vh_use = torch.cat((vh, v[lo:hi, h // group].double()))
            else:
                kh_use, vh_use = kh, vh
            score = qh @ kh_use.T * scale
            row = torch.arange(hi - lo)[:, None] + row_offset
            col = torch.arange(kh_use.shape[0])[None, :]
            score = score.masked_fill(col > row, -float("inf"))
            prob = torch.softmax(score, -1).nan_to_num(0.0)
            out[lo:hi, h] = prob @ vh_use
    return out


def make_bounds(prefix, privates):
    bounds, acc = [prefix], prefix
    for length in privates:
        acc += length
        bounds.append(acc)
    return bounds


CASES = [
    ("smoke-single-h1x1-float16", [127, 128], 1, 1, torch.float16, 11),
    ("multibatch-h4x2-float16", [64, 96, 160, 200, 260], 4, 2,
     torch.float16, 4360),
    ("focusu-p20-130x10-h14x2-float16", make_bounds(20, [130] * 10), 14, 2,
     torch.float16, 71000),
    ("focusu-p25-193x10-h14x2-bfloat16", make_bounds(25, [193] * 10), 14, 2,
     torch.bfloat16, 71003),
]


@pytest.mark.parametrize("bounds,n1,n2,dtype,seed",
                         [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_prefix_attn_dn(bounds, n1, n2, dtype, seed):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is unavailable")
    gen = torch.Generator().manual_seed(seed)
    t = bounds[-1]
    q = torch.randn((t, n1, D), generator=gen).to(dtype)
    k = torch.randn((t, n2, D), generator=gen).to(dtype)
    v = torch.randn((t, n2, D), generator=gen).to(dtype)
    ref = golden_f64(q, k, v, bounds, 1 / math.sqrt(D))
    got = prefix_attn_dn(q.npu(), k.npu(), v.npu(),
                         torch.tensor(bounds, dtype=torch.int32,
                                      device="npu")).cpu().double()
    assert torch.isfinite(got).all()
    tolerance = 0.003 if dtype == torch.float16 else 0.025
    torch.testing.assert_close(got, ref, atol=tolerance, rtol=tolerance)
