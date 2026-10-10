#!/bin/bash
# P0 ABI probe: dump one case from the python host path, then launch the
# prebuilt artifacts from pure C++/ACL and verify bit-exact output.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
GE_DIR=$(dirname "$HERE")
ASCEND=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}
INC="$ASCEND/aarch64-linux/include"
ARTIFACTS="$GE_DIR/opp/op_graph/lib/linux/aarch64/artifacts"

cd "$HERE"
python dump_case.py
g++ -std=c++17 -O2 -Wall \
    -I"$GE_DIR/src" -I"$INC" \
    frame_probe.cpp "$GE_DIR/src/dsl_launcher.cpp" \
    -L"$ASCEND/lib64" -lascendcl -ldl \
    -o frame_probe
PREFIX_ATTN_ARTIFACTS="$ARTIFACTS" ./frame_probe case_data
