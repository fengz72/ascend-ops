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

OPS="${OPS:-causal_conv1d_qkv}"

cd "${TOP}/src"
bash build.sh --pkg --ops="${OPS}" --soc="${SOC_VERSION:-ascend910_93}" \
  --build-dir="${BUILD_DIR}" --cann_3rd_lib_path="${THIRD_PARTY_DIR}" \
  --output-dir="${BUILD_DIR}/output" --build-out-dir="${BUILD_DIR}/build_out" "$@"

# 安装算子到 CANN（幂等）：图模式 kernel 编译走标准 tbe-custom store，
# 全局 opp 必须有本包。本包只注册 CausalConv1dQkv；不会覆盖宿主已有的
# fla 谱系 CausalConv1d（见 cmake/custom_build.cmake / tiling shim）。
# OPP_ROOT 固定为 .run 实际写入的树，断言与 config.ini 不得改看去处。
OPP_ROOT="${ASCEND_HOME_PATH}/opp"
# nullglob: 没有产物时数组必须为空，否则字面量会骗过下面的 -eq 1 断言。
shopt -s nullglob
RUNS=("${BUILD_DIR}"/causal_conv1d_qkv_linux*.run)
[ "${#RUNS[@]}" -eq 1 ] || { echo "error: expected exactly one installer, got ${#RUNS[@]}" >&2; exit 1; }
echo "installing operator into ${OPP_ROOT}"
bash "${RUNS[0]}" --install-path="${OPP_ROOT}" >/dev/null

# 断言：安装后的 ops-info 必须含 CausalConv1dQkvGE + aclnnSupport。该 entry 由
# cmake/inject_causal_conv1d_qkv_ge_info.py 在构建期注入；缺失时 eager 仍可用，
# 只有图模式编译报 EZ3003，所以必须在这里 fail-loud，不能留给模型侧发现。
GE_INFO_POSTPROC="${TOP}/cmake/inject_causal_conv1d_qkv_ge_info.py"
INFO_JSONS=("${OPP_ROOT}"/vendors/causal_conv1d_qkv/op_impl/ai_core/tbe/config/*/aic-*-ops-info.json)
if [ ! -f "${INFO_JSONS[0]:-}" ] || [ ! -f "${GE_INFO_POSTPROC}" ]; then
  echo "error: cannot assert CausalConv1dQkvGE ops-info (missing installed JSON under ${OPP_ROOT}/vendors/causal_conv1d_qkv or ${GE_INFO_POSTPROC})" >&2
  exit 1
fi
"${GE_PYTHON:-python3}" "${GE_INFO_POSTPROC}" --check "${INFO_JSONS[@]}" || {
  echo "error: installed ops-info lacks CausalConv1dQkvGE aclnnSupport; graph mode would fail with EZ3003. Check the cmake injection hook." >&2
  exit 1
}

# load_priority: 仅追加，保持既有顺序（causal_conv1d 必须原位不动）。
VENDORS_CFG="${OPP_ROOT}/vendors/config.ini"
if [ ! -f "${VENDORS_CFG}" ]; then
  echo "load_priority=causal_conv1d_qkv" > "${VENDORS_CFG}"
elif grep -q "^load_priority=" "${VENDORS_CFG}"; then
  old="$(grep -m1 "^load_priority=" "${VENDORS_CFG}" | cut -d= -f2-)"
  case ",${old}," in
    *,causal_conv1d_qkv,*) : ;;
    *)
      new="${old:+${old},}causal_conv1d_qkv"
      sed -i -E "s/^load_priority=.*/load_priority=${new}/" "${VENDORS_CFG}"
      echo "appended causal_conv1d_qkv: load_priority=${new}"
      ;;
  esac
else
  echo "load_priority=causal_conv1d_qkv" >> "${VENDORS_CFG}"
fi

# 注册 torch 绑定：pip editable 指向 vendor 包 op_api/python（随 .run 的
# op_api 整目录拷贝安装），脱离源码树。开发机如需改 python/ 即时生效，可改用
# 源码树安装：pip install -e <repo>/causal-conv1d-qkv/python --no-deps
# （二选一，后装者生效）。GE_PYTHON 由 core.dependencies.prepare 传入以定向
# python 环境。SKIP_TORCH_BINDING=1 用于只装算子不注册绑定的场景，允许
# python/ 缺失，但给出醒目告警。
VENDOR_PY="$(dirname "${VENDORS_CFG}")/causal_conv1d_qkv/op_api/python"
if [ "${SKIP_TORCH_BINDING:-0}" = "1" ]; then
  echo "WARNING: SKIP_TORCH_BINDING=1 -> torch binding NOT registered; torch.ops.ascend_ops.causal_conv1d_qkv is unavailable" >&2
elif [ ! -f "${VENDOR_PY}/pyproject.toml" ]; then
  echo "error: ${VENDOR_PY}/pyproject.toml missing; the .run was built without the python install rule (or python/ is absent from the source tree). Set SKIP_TORCH_BINDING=1 to install the operator only." >&2
  exit 1
else
  echo "registering torch binding (pip editable -> ${VENDOR_PY})"
  "${GE_PYTHON:-python3}" -m pip install -e "${VENDOR_PY}" --no-deps --no-build-isolation >/dev/null
fi

echo
echo "installer:"
ls -1 "${BUILD_DIR}"/causal_conv1d_qkv_linux*.run 2>/dev/null || true
