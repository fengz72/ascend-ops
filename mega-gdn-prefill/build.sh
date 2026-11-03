#!/usr/bin/env bash
set -euo pipefail

TOP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${ASCEND_HOME_PATH:-}" ] && [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
  echo "error: CANN environment not loaded; source set_env.sh first" >&2
  exit 1
fi

BUILD_DIR="${TOP}/build"
THIRD_PARTY_DIR="${TOP}/.cache/cann_3rd_lib"
mkdir -p "${BUILD_DIR}" "${THIRD_PARTY_DIR}"

OPS="${OPS:-mega_gdn_prefill_op,causal_conv1d,mega_chunk_gdn}"

cd "${TOP}/src"
bash build.sh --pkg --ops="${OPS}" --soc="${SOC_VERSION:-ascend910_93}" \
  --build-dir="${BUILD_DIR}" --cann_3rd_lib_path="${THIRD_PARTY_DIR}" "$@"

echo
echo "installer:"
ls -1 "${BUILD_DIR}"/cann_ops*.run 2>/dev/null || true
