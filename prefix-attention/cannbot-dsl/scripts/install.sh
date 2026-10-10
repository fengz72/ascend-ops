#!/bin/bash
# One-shot installer for the prefix_attention operator (per board).
#
#   bash scripts/install.sh
#
# Steps (idempotent):
#   1. export kernel artifacts (kernel fp16/bf16 AOT .so + AICPU .so)  [needs NPU]
#   2. build + install GE custom op deliverable (libcust_opapi.so -> opp/)
#   3. pip install -e the prefix_attn package
#   4. eager smoke test through torch.ops.ascendc_ops.prefix_attention
#
# Afterwards, from any python:
#   import torch, torch_npu
#   import prefix_attn.torch_ops        # registers op + mounts OPP + converter
#   out = torch.ops.ascendc_ops.prefix_attention(q, k, v, bounds)
#
# Env overrides: ASCEND_HOME_PATH, PREFIX_ATTN_FORCE_EXPORT=1 (re-export
# artifacts even if present), PREFIX_ATTN_AIC_CORES.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
GE_DIR="$ROOT/ge_custom_op"
ART_DIR="$GE_DIR/opp/op_graph/lib/linux/aarch64/artifacts"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"

echo "[1/4] kernel artifacts (kernel fp16/bf16 + AICPU metadata)"
if [ -f "$ART_DIR/kernel_fp16.so" ] && [ -f "$ART_DIR/kernel_bf16.so" ] \
   && [ -f "$ART_DIR/aicpu_metadata.so" ] && [ "${PREFIX_ATTN_FORCE_EXPORT:-0}" != "1" ]; then
  echo "  present, skipping (set PREFIX_ATTN_FORCE_EXPORT=1 to re-export)"
else
  (cd "$ROOT" && python "$GE_DIR/scripts/export_artifacts.py" | tail -4)
fi

echo "[2/4] GE custom op deliverable (libcust_opapi.so)"
cmake -S "$GE_DIR" -B "$GE_DIR/build" -DCMAKE_BUILD_TYPE=Release \
  > "$GE_DIR/build_configure.log" 2>&1
cmake --build "$GE_DIR/build" -j"$(nproc)" > "$GE_DIR/build/build.log" 2>&1
cmake --install "$GE_DIR/build" > /dev/null
echo "  installed: $GE_DIR/opp"

echo "[3/4] pip install -e $ROOT"
pip install -e "$ROOT" --no-deps -q
echo "  done"

echo "[4/4] eager smoke test (torch.ops.ascendc_ops.prefix_attention)"
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
print("SMOKE_PASS eager")
print("  opp_root           =", tops.OPP_ROOT)
print("  torchair converter =", tops.TORCHAIR_CONVERTER_REGISTERED)
EOF

cat <<'USAGE'
[DONE] 使用方式(任意 python):
  import torch, torch_npu
  import prefix_attn.torch_ops
  out = torch.ops.ascendc_ops.prefix_attention(q, k, v, bounds)        # eager
  # 图模式: forward 内用 torch.ops.ascendc_ops.prefix_attention,然后
  # torch.compile(model, backend=torchair.get_npu_backend())
USAGE
