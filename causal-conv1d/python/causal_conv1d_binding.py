"""JIT loader for the causal_conv1d eager bridge.

Compiles python/causal_conv1d_bridge.cpp against the installed
libcust_opapi.so (located via the activator's find_op_lib) on first eager
call; cached in the torch extension cache. Graph-mode flows never load this
module.
"""
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

_module = None


def load():
    """JIT-compile and load the eager bridge (process-cached)."""
    global _module
    if _module is not None:
        return _module
    from causal_conv1d import find_op_lib
    lib_dir = os.path.dirname(find_op_lib())
    cann = os.environ.get("ASCEND_HOME_PATH")
    if not cann:
        raise RuntimeError("ASCEND_HOME_PATH not set; source CANN set_env.sh first")
    from torch.utils.cpp_extension import load
    torch_npu_path = Path(torch_npu.__file__).resolve().parent
    here = Path(__file__).resolve().parent
    _module = load(
        name="causal_conv1d_bridge",
        sources=[str(here / "causal_conv1d_bridge.cpp")],
        extra_include_paths=[
            str(torch_npu_path / "include"),
            str(torch_npu_path / "include/third_party/acl/inc"),
            str(Path(cann) / "include"),
        ],
        extra_cflags=[
            "-fstack-protector-all", "-Wl,-z,relro,-z,now,-z,noexecstack", "-fPIC", "-pie",
            "-s", "-fvisibility=hidden", "-D_FORTIFY_SOURCE=2", "-O2", "-w",
            "-D_GLIBCXX_USE_CXX11_ABI=" + ("1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"),
        ],
        extra_ldflags=[
            f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}", "-lcust_opapi",
            f"-L{cann}/lib64", "-lascendcl",
            f"-L{torch_npu_path}/lib", "-ltorch_npu",
        ],
        verbose=False,
    )
    return _module
