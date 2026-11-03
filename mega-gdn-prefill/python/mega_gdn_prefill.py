"""Activator for the MegaGdnPrefill GE custom-op-v2 wrapper.

On import: locates the installed libcust_opapi.so (CANN vendor package),
dlopens it (RTLD_GLOBAL) so GE sees the REG_OP, mounts the vendor package
into ASCEND_CUSTOM_OPP_PATH, then registers
torch.ops.ascend_ops.mega_gdn_prefill (Meta + PrivateUse1) and the torchair
GE converter. With the binding pip-installed (pip install -e python) the
operator imports from anywhere; no shell exports are needed.

Contract: z is overwritten in place and returned as an alias. The schema
declares the mutation (Tensor(a!) z) so dynamo keeps z-reads ordered after
the op; the return stays a plain Tensor because AOT functionalization
rejects alias annotations on custom-op outputs.

The torch.library.Library handle must stay module-level alive — a
function-local would be GC'd, unregistering the schema and crashing dynamo.
torch_npu must be imported before torchair: torch_npu registers the
top-level torchair alias (its bundled copy) in sys.modules.
"""
import ctypes
import os

import torch_npu  # noqa: F401  PrivateUse1 dispatch + torchair alias injection

LIB_NAME = "libcust_opapi.so"
VENDOR_NAME = "mega_gdn_prefill"

NAMES = ["mixed_qkv", "b", "a", "z", "conv_weight", "conv_state", "a_log", "dt_bias",
         "conv_state_read_indices", "conv_state_write_indices", "ssm_state_read_indices",
         "ssm_state_write_indices", "ssm_cache", "mask_lower", "mask_full",
         "minus_identity", "cu_seqlens", "norm_weight"]

SCHEMA = ("mega_gdn_prefill(Tensor mixed_qkv, Tensor b, Tensor a, Tensor(a!) z, Tensor conv_weight, "
          "Tensor conv_state, Tensor a_log, Tensor dt_bias, Tensor conv_state_read_indices, "
          "Tensor conv_state_write_indices, Tensor ssm_state_read_indices, "
          "Tensor ssm_state_write_indices, Tensor ssm_cache, Tensor mask_lower, "
          "Tensor mask_full, Tensor minus_identity, Tensor cu_seqlens, Tensor norm_weight) "
          "-> Tensor")

_LOADED = False
_REGISTERED = False
_TORCH_LIB = None


def _vendor_lib(root):
    """Candidate .so under an opp root: the vendor package itself
    (<pkg>/op_api/lib) or an opp tree containing vendors/<pkg>."""
    root = os.path.normpath(root)
    for path in (os.path.join(root, "op_api", "lib", LIB_NAME),
                 os.path.join(root, "vendors", VENDOR_NAME, "op_api", "lib", LIB_NAME)):
        if os.path.isfile(path):
            return path
    return None


def find_op_lib():
    """Locate the installed libcust_opapi.so.

    Search order: ASCEND_CUSTOM_OPP_PATH roots, then the CANN opp tree
    ($ASCEND_OPP_PATH, where install.sh installs the vendor package).
    """
    for root in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if root:
            found = _vendor_lib(root)
            if found:
                return found
    opp = os.environ.get("ASCEND_OPP_PATH")
    if opp:
        found = _vendor_lib(opp)
        if found:
            return found
    raise FileNotFoundError(
        f"{LIB_NAME} not found (looked in $ASCEND_CUSTOM_OPP_PATH and "
        "$ASCEND_OPP_PATH); install the operator first: "
        "bash mega-gdn-prefill/install.sh with CANN set_env.sh sourced")


def _mount_opp(op_lib):
    """Expose the vendor package to GE discovery (idempotent): graph-mode
    kernel compile resolves the operator through ASCEND_CUSTOM_OPP_PATH."""
    vendor = os.path.dirname(os.path.dirname(os.path.dirname(op_lib)))
    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if vendor not in current.split(":"):
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{vendor}:{current}" if current else vendor


def _eager_impl(mixed_qkv, b, a, z, conv_weight, conv_state, a_log, dt_bias,
                conv_state_read_indices, conv_state_write_indices, ssm_state_read_indices,
                ssm_state_write_indices, ssm_cache, mask_lower, mask_full, minus_identity,
                cu_seqlens, norm_weight):
    from mega_gdn_prefill_binding import load
    return load().mega_gdn_prefill(
        mixed_qkv, b, a, z, conv_weight, conv_state, a_log, dt_bias,
        conv_state_read_indices, conv_state_write_indices, ssm_state_read_indices,
        ssm_state_write_indices, ssm_cache, mask_lower, mask_full, minus_identity,
        cu_seqlens, norm_weight)


def _meta_impl(mixed_qkv, b, a, z, conv_weight, conv_state, a_log, dt_bias,
               conv_state_read_indices, conv_state_write_indices, ssm_state_read_indices,
               ssm_state_write_indices, ssm_cache, mask_lower, mask_full, minus_identity,
               cu_seqlens, norm_weight):
    import torch
    return torch.empty_like(z)


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
    lib.impl("mega_gdn_prefill", _eager_impl, "PrivateUse1")
    lib.impl("mega_gdn_prefill", _meta_impl, "Meta")

    def _convert(mixed_qkv, b, a, z, conv_weight, conv_state, a_log, dt_bias,
                 conv_state_read_indices, conv_state_write_indices, ssm_state_read_indices,
                 ssm_state_write_indices, ssm_cache, mask_lower, mask_full, minus_identity,
                 cu_seqlens, norm_weight, meta_outputs=None):
        values = [mixed_qkv, b, a, z, conv_weight, conv_state, a_log, dt_bias,
                  conv_state_read_indices, conv_state_write_indices, ssm_state_read_indices,
                  ssm_state_write_indices, ssm_cache, mask_lower, mask_full, minus_identity,
                  cu_seqlens, norm_weight]
        return torchair.ge.custom_op(
            "MegaGdnPrefill",
            inputs={name: value for name, value in zip(NAMES, values)},
            outputs=["z"])

    torchair.register_fx_node_ge_converter(torch.ops.ascend_ops.mega_gdn_prefill.default)(_convert)
    _REGISTERED = True


def ensure_loaded():
    global _LOADED
    if _LOADED:
        return
    op_lib = find_op_lib()
    _mount_opp(op_lib)
    ctypes.CDLL(op_lib, mode=ctypes.RTLD_GLOBAL)
    _register_torch_op()
    _LOADED = True


ensure_loaded()
