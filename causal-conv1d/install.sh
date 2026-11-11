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

OPS="${OPS:-causal_conv1d}"

cd "${TOP}"
bash build.sh --pkg --ops="${OPS}" --soc="${SOC_VERSION:-ascend910_93}" \
  --cann_3rd_lib_path="${THIRD_PARTY_DIR}" "$@"

# 安装算子到 CANN（幂等）：图模式 kernel 编译走标准 tbe-custom store，
# 全局 opp 必须有本包。同名算子类型 CausalConv1d 也存在于 mega_gdn_prefill
# vendor 包（vendored xllm 旧谱系，本仓库无运行时调用方）；config.ini 里把
# 本包放在 load_priority 首位即可让 GE/aclnn 解析到本包的 fla 谱系版本。
RUNS=("${BUILD_DIR}"/causal_conv1d_linux*.run)
[ "${#RUNS[@]}" -eq 1 ] || { echo "error: expected exactly one installer, got ${#RUNS[@]}" >&2; exit 1; }
echo "installing operator into \$ASCEND_HOME_PATH/opp"
bash "${RUNS[0]}" --install-path="${ASCEND_HOME_PATH}/opp" >/dev/null
VENDORS_CFG="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}/vendors/config.ini"
if [ ! -f "${VENDORS_CFG}" ]; then
  echo "load_priority=causal_conv1d" > "${VENDORS_CFG}"
elif grep -q "^load_priority=" "${VENDORS_CFG}"; then
  # 幂等置首：剔除列表中已有的 causal_conv1d 项再前插，避免重复条目
  # （mega 包等后装者可能把本包挤到非首位）。
  old="$(grep -m1 "^load_priority=" "${VENDORS_CFG}" | cut -d= -f2-)"
  rest="$(printf '%s' "${old}" | awk -v FS=, '{out=""; for (i = 1; i <= NF; i++) if ($i != "" && $i != "causal_conv1d") out = (out == "" ? $i : out "," $i); print out}')"
  new="causal_conv1d${rest:+,${rest}}"
  if [ "${old}" != "${new}" ]; then
    sed -i -E "s/^load_priority=.*/load_priority=${new}/" "${VENDORS_CFG}"
    echo "set ${VENDORS_CFG} load_priority=${new}"
  fi
else
  echo "load_priority=causal_conv1d" >> "${VENDORS_CFG}"
fi

# 注册 torch 绑定：pip editable 指向 vendor 包 op_api/python（随 .run 的
# op_api 整目录拷贝安装），脱离源码树。开发机如需改 python/ 即时生效，可改用
# 源码树安装：pip install -e <repo>/causal-conv1d/python --no-deps（二选一，
# 后装者生效）。GE_PYTHON 由 core.dependencies.prepare 传入以定向 python 环境。
VENDOR_PY="$(dirname "${VENDORS_CFG}")/causal_conv1d/op_api/python"
if [ ! -f "${VENDOR_PY}/pyproject.toml" ]; then
  echo "error: ${VENDOR_PY}/pyproject.toml missing; the .run was built without the python install rule" >&2
  exit 1
fi
echo "registering torch binding (pip editable -> ${VENDOR_PY})"
"${GE_PYTHON:-python3}" -m pip install -e "${VENDOR_PY}" --no-deps --no-build-isolation >/dev/null

echo
echo "installer:"
ls -1 "${BUILD_DIR}"/causal_conv1d_linux*.run 2>/dev/null || true
