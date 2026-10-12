"""PrefixAttention torch接入GE交付件(独立交付, 无需 cannbotdsl / prefix_attn 包)。

import本包即完成: torch schema注册 + Meta支持 + PrivateUse1(eager) dispatch走
GE交付件 libcust_opapi.so + OPP自动挂载 + torchair GE converter注册(图模式)。
本包是 torch.ops 接入的唯一入口(eager 与图模式)。

用法:
    import prefix_attn_torch  # noqa
    out = torch.ops.ascendc_ops.prefix_attention(q, k, v, bounds, 0.125)
    # 图模式: torch.compile(model, backend=torchair.get_npu_backend())

前置条件:
    - GE交付件已装入CANN目录(bash ge/scripts/install.sh)
    - import前source了CANN set_env.sh(或自行设置ASCEND_HOME_PATH)
    - 首次import会JIT编译torch扩展(~1min), 产物缓存在torch extension cache
"""

import inspect
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401  PrivateUse1后端注册须先加载torch_npu
from torch.library import Library, impl

__version__ = "1.0.0"

# --- torch schema + Meta ---
AS_LIBRARY = Library("ascendc_ops", "DEF")
OP_NAME = "prefix_attention"
AS_LIBRARY.define(
    f"{OP_NAME}(Tensor query, Tensor key, Tensor value, "
    "Tensor actual_seq_lengths, float scale=0.0) -> Tensor")


@impl(AS_LIBRARY, OP_NAME, "Meta")
def _prefix_attention_meta(query, key, value, actual_seq_lengths, scale=0.0):
    torch._check(query.dim() == 3,
                 lambda: "query must be 3D TND tensor [T, N, D]")
    return query.new_empty(query.size())


# --- OPP root locate / mount ---
_DEFAULT_CANN = "/usr/local/Ascend/ascend-toolkit/latest"


def _opp_lib_dir(root):
    return Path(root) / "op_graph" / "lib" / "linux" / "aarch64"


def _deliverable(root):
    return _opp_lib_dir(root) / "libcust_opapi.so"


def _cann_home():
    """CANN 根目录: $ASCEND_HOME_PATH, 否则标准 toolkit 安装位。"""
    env = os.environ.get("ASCEND_HOME_PATH")
    if env and os.path.exists(env):
        return env
    if os.path.exists(_DEFAULT_CANN):
        return _DEFAULT_CANN
    return None


def find_opp_root():
    """定位GE交付件OPP根(未安装返回None)。

    $ASCEND_CUSTOM_OPP_PATH 各条目,否则 install.sh 的标准安装位
    ($ASCEND_HOME_PATH/opp, 含未设env时的toolkit默认路径)。
    """
    candidates = []
    for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if entry:
            candidates.append(Path(entry))
    ascend_home = _cann_home()
    if ascend_home:
        candidates.append(Path(ascend_home) / "opp")   # install.sh 安装位
    for root in candidates:
        if _deliverable(root).exists():
            return root
    return None


def mount_opp():
    """把OPP根挂到ASCEND_CUSTOM_OPP_PATH(幂等), 图模式GE初始化时可见。"""
    root = find_opp_root()
    if root is None:
        raise RuntimeError(
            "GE deliverable not found; run ge/scripts/install.sh first "
            "(looked in $ASCEND_CUSTOM_OPP_PATH, $ASCEND_HOME_PATH/opp "
            f"and {_DEFAULT_CANN}/opp)")
    root = str(root)
    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if root not in current.split(":"):
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
            f"{root}:{current}" if current else root)
    return root


OPP_ROOT = mount_opp()


# --- JIT eager bridge: torch tensors -> C ABI of libcust_opapi.so ---
def _resolve_cann_path():
    env = _cann_home()
    if env:
        return env
    raise RuntimeError(
        "CANN toolkit path not found, set ASCEND_HOME_PATH "
        "(e.g. source set_env.sh) and retry")


def load_extension():
    """JIT编译并加载eager桥接扩展(带进程级缓存), 产物缓存在torch extension cache。"""
    from torch.utils.cpp_extension import load

    pkg_dir = Path(__file__).resolve().parent
    torch_npu_path = Path(torch_npu.__file__).resolve().parent
    cann_path = _resolve_cann_path()
    root = find_opp_root()
    if root is None:
        raise RuntimeError(
            "GE deliverable not found; run ge/scripts/install.sh first")
    lib_dir = _opp_lib_dir(root)
    include_paths = [
        str(torch_npu_path / "include"),
        str(torch_npu_path / "include/third_party/acl/inc"),
        str(Path(cann_path) / "include"),
    ]
    cxx_args = [
        "-fstack-protector-all", "-Wl,-z,relro,-z,now,-z,noexecstack", "-fPIC", "-pie",
        "-s", "-fvisibility=hidden", "-D_FORTIFY_SOURCE=2", "-O2", "-w",
        "-D_GLIBCXX_USE_CXX11_ABI=" + ("1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"),
    ]
    ldflags = [
        f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}", "-lcust_opapi",
        f"-L{cann_path}/lib64", "-lascendcl",
        f"-L{torch_npu_path}/lib", "-ltorch_npu",
    ]
    return load(name="prefix_attn_torch",
                sources=[str(pkg_dir / "prefix_attn_bridge.cpp")],
                extra_include_paths=include_paths,
                extra_cflags=cxx_args,
                extra_ldflags=ldflags,
                verbose=False)


op_module = load_extension()


@impl(AS_LIBRARY, OP_NAME, "PrivateUse1")
def _prefix_attention_eager(query, key, value, actual_seq_lengths, scale=0.0):
    return op_module.prefix_attention(query, key, value, actual_seq_lengths,
                                      scale)


# --- torchair GE converter (graph mode) ---
def _assert_defaults_match_schema(converter):
    """torch.export会把==schema默认值的实参从fx节点kwargs规约掉, converter以
    自身Python签名默认值兜底——两层不一致即图模式静默丢参, 故断言。"""
    op = torch.ops.ascendc_ops.prefix_attention
    schema_defaults = {a.name: a.default_value
                       for a in op.default._schema.arguments
                       if a.default_value is not None}
    sig = inspect.signature(converter)
    for name, sval in schema_defaults.items():
        if name not in sig.parameters:
            continue
        cval = sig.parameters[name].default
        if cval is inspect.Parameter.empty:
            continue
        if cval != sval:
            raise RuntimeError(
                f"converter默认值与算子schema默认值不一致: {name} schema={sval!r} "
                f"converter={cval!r}")


def register_converter():
    try:
        import torchair  # noqa: F401
        from torchair.ge import attr as ge_attr
        from torchair.ge import custom_op as ge_custom_op
        from torchair._ge_concrete_graph.ge_converter.converter_utils import (
            Tensor, TensorSpec,
        )
    except ImportError as e:
        import warnings
        warnings.warn(f"torchair不可用({e}), 图模式converter未注册, 仅支持eager调用")
        return

    @torchair.register_fx_node_ge_converter(
        torch.ops.ascendc_ops.prefix_attention.default)
    def _convert_prefix_attention(
        query: Tensor, key: Tensor, value: Tensor,
        actual_seq_lengths: Tensor, scale: float = 0.0,
        meta_outputs: TensorSpec = None,
    ):
        return ge_custom_op(
            "PrefixAttention",
            inputs={
                "query": query, "key": key, "value": value,
                "actual_seq_lengths": actual_seq_lengths,
            },
            attrs={"scale": ge_attr.Float(float(scale))},
            outputs=["attention_out"],
        )

    _assert_defaults_match_schema(_convert_prefix_attention)


register_converter()
