#!/bin/bash
# torchair end-to-end: small attention-block network compiled onto the GE
# custom op PrefixAttention, compared against the eager DSL path.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
GE_DIR=$(dirname "$HERE")
BUILD="$GE_DIR/build"

if [ ! -f "$GE_DIR/opp/op_graph/lib/linux/aarch64/libcust_opapi.so" ]; then
  echo "[INFO] build + install GE custom op first"
  cmake -S "$GE_DIR" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build "$BUILD" -j"$(nproc)" > /dev/null
  cmake --install "$BUILD" > /dev/null
fi
if [ ! -f "$GE_DIR/opp/op_graph/lib/linux/aarch64/artifacts/kernel_fp16.so" ]; then
  echo "[ERROR] artifacts missing; run: bash scripts/install_ge.sh" >&2
  exit 1
fi

export ASCEND_CUSTOM_OPP_PATH="$GE_DIR/opp${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
echo "[INFO] run torchair net test (ASCEND_CUSTOM_OPP_PATH=$GE_DIR/opp)"
python "$HERE/test_torchair_net.py" 2>&1 | tee "$BUILD/torchair_net.log" | tail -6
grep -Fq "TORCHAIR_NET_PASS" "$BUILD/torchair_net.log"
echo "[INFO] torchair net sample PASS"
