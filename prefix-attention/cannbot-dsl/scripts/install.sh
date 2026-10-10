#!/bin/bash
# Install the python frontend (prefix_attn package + torch integration).
#
#   bash scripts/install.sh
#
# This is the DEVELOP/VERIFY side: it needs cannbotdsl (the eager path runs
# the DSL host).  For the production model-integration side (GE deliverable,
# no cannbotdsl needed at build/run), use scripts/install_ge.sh.
#
# Steps (idempotent):
#   1. pip install -e the prefix_attn package
#   2. eager smoke test through torch.ops.ascendc_ops.prefix_attention
#
# Afterwards, from any python:
#   import torch, torch_npu
#   import prefix_attn.torch_ops        # registers op + mounts OPP + converter
#   out = torch.ops.ascendc_ops.prefix_attention(q, k, v, bounds)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")

echo "[1/2] pip install -e $ROOT"
pip install -e "$ROOT" --no-deps -q
echo "  done"

echo "[2/2] eager smoke test (torch.ops.ascendc_ops.prefix_attention)"
python - <<'EOF'
import torch
import torch_npu  # noqa: F401
import prefix_attn.torch_ops as tops

q = torch.randn(128, 1, 64, dtype=torch.float16).npu()
k = torch.randn(128, 1, 64, dtype=torch.float16).npu()
v = torch.randn(128, 1, 64, dtype=torch.float16).npu()
b = torch.tensor([127, 128], dtype=torch.int32).npu()
out = torch.ops.ascendc_ops.prefix_attention(q, k, v, b)
torch.npu.synchronize()
assert out.shape == q.shape and out.dtype == q.dtype
assert torch.isfinite(out.float()).all()
print("SMOKE_PASS eager (DSL host path)")
print("  opp_root           =", tops.OPP_ROOT)
print("  torchair converter =", tops.TORCHAIR_CONVERTER_REGISTERED)
EOF

cat <<'USAGE'
[DONE] python 前端已装。使用(任意 python):
  import torch, torch_npu
  import prefix_attn.torch_ops
  out = torch.ops.ascendc_ops.prefix_attention(q, k, v, bounds)   # eager(需 cannbotdsl)
GE 交付件(模型接入,无需 cannbotdsl)另跑: bash scripts/install_ge.sh
USAGE
