#!/bin/bash
# Build + install the GE deliverable (custom_op_v2 PrefixAttention).
#
#   bash ge/scripts/install.sh
#
# This is the MODEL-INTEGRATION side.  It compiles the checked-in codegen'd
# AscendC sources (ge/codegen/*.asc, UNMODIFIED) with bisheng into
# kernel_fp16.so (fp16 only -- the pipeline runs fp16 and a single kernel
# .so makes the dtype symbol question moot), builds the AICPU planner
# (aicpu_metadata.so, per-thread TLS scratch) and libcust_opapi.so (GE custom
# op + host planner + device-task launcher), and installs the OPP package
# DIRECTLY INTO THE CANN DIRECTORY ($ASCEND_HOME_PATH/opp/op_graph).  NO
# cannbotdsl and NO python DSL runtime are needed here -- only CANN/bisheng
# + cmake.
#
# (Regenerating the .asc sources from the DSL kernel DOES need cannbotdsl:
#  python ge/scripts/dump_kernel_src.py -- run only when the kernel
#  changes; the sources are checked in.)
#
# Env: ASCEND_HOME_PATH, NPU_ARCH (default dav-3510).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
GE_DIR=$(dirname "$HERE")
# CANN path: $ASCEND_HOME_PATH, else toolkit/latest, else newest cann-* dir
# (keeps working across CANN upgrades without editing this script).
if [ -z "$ASCEND_HOME_PATH" ]; then
  CANN_CAND="/usr/local/Ascend/ascend-toolkit/latest"
  if [ ! -d "$CANN_CAND" ]; then
    CANN_CAND=$(ls -d /usr/local/Ascend/cann-*/ 2>/dev/null | sort -V | tail -1)
  fi
  if [ -n "$CANN_CAND" ]; then
    export ASCEND_HOME_PATH="${CANN_CAND%/}"
  fi
fi

if [ -z "$ASCEND_HOME_PATH" ]; then
  echo "[ERROR] CANN not found. Source set_env.sh or set ASCEND_HOME_PATH." >&2
  exit 1
fi

if [ ! -f "$GE_DIR/codegen/launch_prefix_macro_fp16.asc" ]; then
  echo "[ERROR] codegen sources missing. Run (needs cannbotdsl):" >&2
  echo "        python ge/scripts/dump_kernel_src.py" >&2
  exit 1
fi

echo "[1/3] cmake configure (NPU_ARCH=${NPU_ARCH:-dav-3510})"
if ! cmake -S "$GE_DIR" -B "$GE_DIR/build" -DCMAKE_BUILD_TYPE=Release \
     ${NPU_ARCH:+-DNPU_ARCH=$NPU_ARCH} > "$GE_DIR/build_configure.log" 2>&1; then
  tail -30 "$GE_DIR/build_configure.log"
  exit 1
fi

echo "[2/3] build (bisheng compiles .asc/.aicpu; CXX builds libcust_opapi.so)"
cmake --build "$GE_DIR/build" -j"$(nproc)" > "$GE_DIR/build/build.log" 2>&1 || {
  tail -30 "$GE_DIR/build/build.log"; exit 1; }

echo "[3/3] install OPP package into CANN directory"
cmake --install "$GE_DIR/build" > /dev/null
echo "  installed: $ASCEND_HOME_PATH/opp/op_graph"
ls "$ASCEND_HOME_PATH/opp/op_graph/lib/linux/aarch64/artifacts/"*.so

cat <<USAGE
[DONE] GE 交付件已装入 CANN 目录。挂载后即可入图(无需 cannbotdsl):
  export ASCEND_CUSTOM_OPP_PATH="\$ASCEND_HOME_PATH/opp\${ASCEND_CUSTOM_OPP_PATH:+:\$ASCEND_CUSTOM_OPP_PATH}"
  # torchair: import prefix_attn_torch 会自动挂载 + 注册 converter,无需手动 export
  # 注意: CANN 升级会清空 $ASCEND_HOME_PATH/opp,需重跑本脚本
USAGE
