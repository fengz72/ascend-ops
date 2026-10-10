#!/bin/bash
# Build + install the GE deliverable (custom_op_v2 PrefixAttention).
#
#   bash scripts/install_ge.sh
#
# This is the MODEL-INTEGRATION side.  It compiles the checked-in codegen'd
# AscendC sources (../codegen/*.asc, ../codegen/aicpu/*) with bisheng into
# kernel_fp16.so / kernel_bf16.so / aicpu_metadata.so, builds libcust_opapi.so
# (GE custom op + launcher), and installs the OPP package.  NO cannbotdsl and
# NO python DSL runtime are needed here -- only CANN/bisheng + cmake.
#
# (Regenerating the .asc sources from the DSL kernel DOES need cannbotdsl:
#  python ge_custom_op/scripts/dump_kernel_src.py -- run only when the kernel
#  changes; the sources are checked in.)
#
# Env: ASCEND_HOME_PATH, NPU_ARCH (default dav-3510), PREFIX_ATTN_AIC_CORES.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
GE_DIR="$ROOT/ge_custom_op"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"

if [ ! -f "$ROOT/codegen/launch_prefix_macro_fp16.asc" ]; then
  echo "[ERROR] codegen sources missing. Run (needs cannbotdsl):" >&2
  echo "        python ge_custom_op/scripts/dump_kernel_src.py" >&2
  exit 1
fi

echo "[1/3] cmake configure (NPU_ARCH=${NPU_ARCH:-dav-3510})"
cmake -S "$GE_DIR" -B "$GE_DIR/build" -DCMAKE_BUILD_TYPE=Release \
  ${NPU_ARCH:+-DNPU_ARCH=$NPU_ARCH} > "$GE_DIR/build_configure.log" 2>&1

echo "[2/3] build (bisheng compiles .asc + aicpu; CXX builds libcust_opapi.so)"
cmake --build "$GE_DIR/build" -j"$(nproc)" > "$GE_DIR/build/build.log" 2>&1 || {
  tail -30 "$GE_DIR/build/build.log"; exit 1; }

echo "[3/3] install OPP package"
cmake --install "$GE_DIR/build" > /dev/null
echo "  installed: $GE_DIR/opp"
ls "$GE_DIR/opp/op_graph/lib/linux/aarch64/artifacts/"*.so

cat <<'USAGE'
[DONE] GE 交付件已装。挂载后即可入图(无需 cannbotdsl):
  export ASCEND_CUSTOM_OPP_PATH="<...>/ge_custom_op/opp:$ASCEND_CUSTOM_OPP_PATH"
  # torchair: import prefix_attn.torch_ops 会自动挂载 OPP + 注册 converter
  # 验证: bash ge_custom_op/tests/run_online.sh   (GE Session 端到端)
USAGE
