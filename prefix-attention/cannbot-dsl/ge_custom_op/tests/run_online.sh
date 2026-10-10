#!/bin/bash
# Build the GE custom op, stage the ES python wrapper, and run the online
# GE Session end-to-end test (single PrefixAttnDn node vs python reference).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
GE_DIR=$(dirname "$HERE")
ASCEND=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}
BUILD="$GE_DIR/build"
PKG="$BUILD/python_package"

echo "[INFO] build + install (OPP layout)"
cmake -S "$GE_DIR" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release > "$BUILD/cmake_configure.log" 2>&1 || {
  cmake -S "$GE_DIR" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release; }
cmake --build "$BUILD" -j"$(nproc)" 2>&1 | tail -2
cmake --install "$BUILD" > /dev/null

echo "[INFO] stage ES python wrapper"
rm -rf "$PKG"
mkdir -p "$PKG/es_prefix_attn"
cp "$HERE/es_package_template/__init__.py" "$PKG/es_prefix_attn/"
cp "$BUILD"/es_prefix_attn_build/generated_code/*.py "$PKG/es_prefix_attn/"

echo "[INFO] case data"
[ -f "$GE_DIR/probe/case_data/meta.txt" ] || python "$GE_DIR/probe/dump_case.py"

export PYTHONPATH="$PKG${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="$BUILD${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ASCEND_CUSTOM_OPP_PATH="$GE_DIR/opp${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"

echo "[INFO] run GE online session (ASCEND_CUSTOM_OPP_PATH=$GE_DIR/opp)"
python "$HERE/run_ge_session.py" 2>&1 | tee "$BUILD/ge_session.log" | tail -5
grep -Fq "GE_SESSION_PASS" "$BUILD/ge_session.log"
echo "[INFO] PrefixAttnDn GE custom op online sample PASS"
