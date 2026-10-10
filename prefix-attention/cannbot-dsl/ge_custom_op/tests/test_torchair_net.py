#!/usr/bin/env python3
"""torchair end-to-end test: a small attention block network using
torch.ops.prefix_attention.forward, compiled (下沉) through torchair onto the GE
custom op PrefixAttention, compared against the eager reference.

Env (set by tests/run_torchair.sh):
  ASCEND_CUSTOM_OPP_PATH  opp root with libcust_opapi.so + artifacts
"""
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).resolve().parent
GE_DIR = HERE.parent
sys.path.insert(0, str(GE_DIR.parent))  # cannbot-dsl root (prefix_attn pkg)

# registers torch.ops.prefix_attention.forward + mounts OPP + torchair converter
import prefix_attn.torch_ops  # noqa: E402,F401

from torch_npu.dynamo import torchair  # noqa: E402

T, N1, N2, D = 1320, 14, 2, 64
HIDDEN = N1 * D
BOUNDS = [20] + [20 + 130 * (i + 1) for i in range(10)]


class AttnBlock(torch.nn.Module):
    """norm-free qkv -> prefix attention -> o_proj block (fp16)."""

    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(HIDDEN, (N1 + 2 * N2) * D, bias=False,
                                   dtype=torch.float16)
        self.o = torch.nn.Linear(N1 * D, HIDDEN, bias=False,
                                 dtype=torch.float16)

    def forward(self, x, bounds):
        qkv = self.qkv(x)
        q = qkv[:, : N1 * D].view(-1, N1, D).contiguous()
        k = qkv[:, N1 * D: (N1 + N2) * D].view(-1, N2, D).contiguous()
        v = qkv[:, (N1 + N2) * D:].view(-1, N2, D).contiguous()
        attn = torch.ops.prefix_attention.forward(q, k, v, bounds, 0.125)
        return self.o(attn.view(-1, N1 * D))


def main():
    assert torch.npu.is_available(), "NPU is unavailable"
    torch.manual_seed(7)
    model = AttnBlock().npu().eval()
    x = torch.randn(T, HIDDEN, dtype=torch.float16).npu()
    bounds = torch.tensor(BOUNDS, dtype=torch.int32).npu()

    with torch.no_grad():
        ref = model(x, bounds)          # eager: DSL python host path
        compiled = torch.compile(model, backend=torchair.get_npu_backend())
        # Sabotage the python host path: if the compiled call graph-breaks and
        # falls back to eager, the PrivateUse1 impl re-resolves prefix_attention
        # at call time and hits this; only the GE C++ Execute path survives.
        import prefix_attn as _pkg

        def _no_eager(*a, **kw):
            raise AssertionError("compiled path fell back to python eager")

        _pkg.prefix_attention = _no_eager
        got = compiled(x, bounds)       # torchair: GE PrefixAttention node
    torch.npu.synchronize()

    diff = (got.float() - ref.float()).abs()
    max_abs = diff.max().item()
    print(f"output shape={tuple(got.shape)} dtype={got.dtype} "
          f"max_abs_diff={max_abs:.6f}")
    tol = 3e-3  # fp16 attention + two GEMMs, eager vs GE path
    if max_abs <= tol:
        print("TORCHAIR_NET_PASS", flush=True)
        return 0
    print(f"TORCHAIR_NET_FAIL (max_abs_diff={max_abs} > {tol})", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
