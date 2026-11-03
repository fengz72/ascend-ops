#!/usr/bin/env bash
set -euo pipefail

TOP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  echo "error: ASCEND_HOME_PATH not set; source CANN set_env.sh first" >&2
  exit 1
fi

BUILD_DIR="${TOP}/build"
THIRD_PARTY_DIR="${TOP}/.cache/cann_3rd_lib"
mkdir -p "${BUILD_DIR}" "${THIRD_PARTY_DIR}"

OPS="${OPS:-mega_gdn_prefill_op,causal_conv1d,mega_chunk_gdn}"

cd "${TOP}/src"
bash build.sh --pkg --ops="${OPS}" --soc="${SOC_VERSION:-ascend910_93}" \
  --build-dir="${BUILD_DIR}" --cann_3rd_lib_path="${THIRD_PARTY_DIR}" "$@"

# 安装算子到 CANN（幂等）：图模式 kernel 编译走标准 tbe-custom store，
# 全局 opp 必须有本包；多 vendor 环境需 load_priority（见 README）。
RUNS=("${BUILD_DIR}"/cann_ops-mega_gdn_prefill*.run)
[ "${#RUNS[@]}" -eq 1 ] || { echo "error: expected exactly one installer, got ${#RUNS[@]}" >&2; exit 1; }
echo "installing operator into \$ASCEND_HOME_PATH/opp"
bash "${RUNS[0]}" --install-path="${ASCEND_HOME_PATH}/opp" >/dev/null
VENDORS_CFG="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}/vendors/config.ini"
if [ ! -f "${VENDORS_CFG}" ]; then
  echo "load_priority=mega_gdn_prefill" > "${VENDORS_CFG}"
elif ! grep -q "load_priority=mega_gdn_prefill" "${VENDORS_CFG}"; then
  echo "warning: ${VENDORS_CFG} carries other load_priority entries; merge 'load_priority=mega_gdn_prefill' manually" >&2
fi

# 注册 torch 绑定：pip editable 指向 vendor 包 op_api/python（随 .run 的
# op_api 整目录拷贝安装），脱离源码树（部署机只需 .run + pip 一条命令）。
# 开发机如需改 python/ 即时生效，可改用源码树安装：
# pip install -e <repo>/mega-gdn-prefill/python --no-deps（二选一，后装者生效）。
# GE_PYTHON 由 core.dependencies.prepare 传入以定向 python 环境。
VENDOR_PY="$(dirname "${VENDORS_CFG}")/mega_gdn_prefill/op_api/python"
if [ ! -f "${VENDOR_PY}/pyproject.toml" ]; then
  echo "error: ${VENDOR_PY}/pyproject.toml missing; the .run was built without the python install rule" >&2
  exit 1
fi
echo "registering torch binding (pip editable -> ${VENDOR_PY})"
"${GE_PYTHON:-python3}" -m pip install -e "${VENDOR_PY}" --no-deps --no-build-isolation >/dev/null

echo
echo "installer:"
ls -1 "${BUILD_DIR}"/cann_ops*.run 2>/dev/null || true
