"""Activator for the CausalConv1dGE custom-op-v2 wrapper.

On import: locates the installed libcust_opapi.so (CANN vendor package),
dlopens it (RTLD_GLOBAL) so GE sees the REG_OP, mounts the vendor package
into ASCEND_CUSTOM_OPP_PATH, then registers
torch.ops.ascend_ops.causal_conv1d (Meta + PrivateUse1) and the torchair
GE converter. With the binding pip-installed the operator imports from
anywhere; no shell exports are needed.

Contract: dense FN prefill only — x dim-last (B,S,D) or (T,D), weight (W,D),
bias (D,) optional, activation "none"/"silu"/"swish". The varlen/cache routes
of the upstream operator stay reachable through its aclnn API directly.

The torch.library.Library handle must stay module-level alive — a
function-local would be GC'd, unregistering the schema and crashing dynamo.
torch_npu must be imported before torchair: torch_npu registers the
top-level torchair alias (its bundled copy) in sys.modules.
"""
import ctypes
import os

import torch_npu  # noqa: F401  PrivateUse1 dispatch + torchair alias injection

LIB_NAME = "libcust_opapi.so"
VENDOR_NAME = "causal_conv1d"

SCHEMA = ("causal_conv1d(Tensor x, Tensor weight, Tensor? bias=None, "
          "str activation=\"silu\", int head_num=0) -> Tensor")

ACTIVATIONS = ("none", "silu", "swish")

_LOADED = False
_REGISTERED = False
_TORCH_LIB = None


def _vendor_lib(root):
    """Candidate .so under an opp root: the vendor package itself
    (<pkg>/op_api/lib) or an opp tree containing vendors/<pkg>."""
    root = os.path.normpath(root)
    for path in (os.path.join(root, "op_api", "lib", LIB_NAME),
                 os.path.join(root, "vendors", VENDOR_NAME, "op_api", "lib",
                              LIB_NAME)):
        if os.path.isfile(path):
            return path
    return None


def find_op_lib():
    """Locate the installed libcust_opapi.so.

    Search order: the canonical CANN opp tree ($ASCEND_OPP_PATH, where
    install.sh installs the vendor package) first, then $ASCEND_CUSTOM_OPP_PATH
    roots. The opp tree goes first because other packages (e.g. the fla wheel)
    mount their own libcust_opapi.so into ASCEND_CUSTOM_OPP_PATH and would
    otherwise shadow this package's lib; the loaded lib is verified to export
    causal_conv1d_run so a wrong same-named lib fails loud.
    """
    candidates = []
    opp = os.environ.get("ASCEND_OPP_PATH")
    if opp:
        candidates.append(opp)
    candidates.extend(
        root for root in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
        if root)
    for root in candidates:
        found = _vendor_lib(root)
        if found:
            return found
    raise FileNotFoundError(
        f"{LIB_NAME} not found (looked in $ASCEND_OPP_PATH and "
        "$ASCEND_CUSTOM_OPP_PATH); install the operator first: "
        "bash causal-conv1d/install.sh with CANN set_env.sh sourced")


def _mount_opp(op_lib):
    """Expose the vendor package to GE discovery (idempotent): graph-mode
    kernel compile resolves the operator through ASCEND_CUSTOM_OPP_PATH."""
    vendor = os.path.dirname(os.path.dirname(os.path.dirname(op_lib)))
    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if vendor not in current.split(":"):
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
            f"{vendor}:{current}" if current else vendor)


def _eager_impl(x, weight, bias=None, activation="silu", head_num=0):
    from causal_conv1d_binding import load
    return load().causal_conv1d(x, weight, bias, activation, head_num)


def _meta_impl(x, weight, bias=None, activation="silu", head_num=0):
    import torch
    return torch.empty_like(x)


def _register_torch_op():
    global _REGISTERED
    if _REGISTERED:
        return
    import torch
    import torchair
    global _TORCH_LIB
    lib = torch.library.Library("ascend_ops", "FRAGMENT")
    _TORCH_LIB = lib
    lib.define(SCHEMA)
    lib.impl("causal_conv1d", _eager_impl, "PrivateUse1")
    lib.impl("causal_conv1d", _meta_impl, "Meta")

    def _convert(x, weight, bias=None, activation="silu", head_num=0,
                 meta_outputs=None):
        return torchair.ge.custom_op(
            "CausalConv1dGE",
            inputs={"x": x, "weight": weight, "bias": bias},
            outputs=["y"],
            attrs={"activation": torchair.ge.attr.Str(activation),
                   "head_num": torchair.ge.attr.Int(head_num)})

    torchair.register_fx_node_ge_converter(
        torch.ops.ascend_ops.causal_conv1d.default)(_convert)
    _REGISTERED = True


def ensure_loaded():
    global _LOADED
    if _LOADED:
        return
    op_lib = find_op_lib()
    lib = ctypes.CDLL(op_lib, mode=ctypes.RTLD_GLOBAL)
    if not hasattr(lib, "causal_conv1d_run"):
        raise ImportError(
            f"{op_lib} does not export causal_conv1d_run; another vendor's "
            "libcust_opapi.so with the same name was found — check "
            "ASCEND_CUSTOM_OPP_PATH shadowing and reinstall this package")
    _mount_opp(op_lib)
    _register_torch_op()
    _LOADED = True


ensure_loaded()
