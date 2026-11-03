"""Activator for the MegaGdnPrefill GE custom-op-v2 wrapper.

On import: dlopens libcust_opapi.so (RTLD_GLOBAL) so GE sees the REG_OP,
then registers torch.ops.ascend_ops.mega_gdn_prefill (Meta + PrivateUse1)
and the torchair GE converter.

The torch.library.Library handle must stay module-level alive — a
function-local would be GC'd, unregistering the schema and crashing dynamo.

Remove this dependency when CANN ships a graph-usable MegaGdnPrefillOp.
"""
import ctypes
import importlib
import math
from pathlib import Path

LIB_NAME = "libcust_opapi.so"
INSTALL_RELPATH = Path("dependencies") / "mega_gdn_prefill" / "install"

NAMES = ["mixed_qkv", "b", "a", "z", "conv_weight", "conv_state", "a_log", "dt_bias",
         "conv_state_read_indices", "conv_state_write_indices", "ssm_state_read_indices",
         "ssm_state_write_indices", "ssm_cache", "mask_lower", "mask_full",
         "minus_identity", "cu_seqlens", "norm_weight"]

SCHEMA = ("mega_gdn_prefill(Tensor mixed_qkv, Tensor b, Tensor a, Tensor z, Tensor conv_weight, "
          "Tensor conv_state, Tensor a_log, Tensor dt_bias, Tensor conv_state_read_indices, "
          "Tensor conv_state_write_indices, Tensor ssm_state_read_indices, "
          "Tensor ssm_state_write_indices, Tensor ssm_cache, Tensor mask_lower, "
          "Tensor mask_full, Tensor minus_identity, Tensor cu_seqlens, Tensor norm_weight) "
          "-> Tensor")

_LOADED = False
_REGISTERED = False
_TORCH_LIB = None


def _install_dir(work_dir=None):
    here = Path(__file__).resolve().parent
    candidate = here.parent
    if (candidate / "lib" / LIB_NAME).is_file() or (candidate / "opp").is_dir():
        return candidate
    if work_dir is not None:
        candidate = Path(work_dir) / INSTALL_RELPATH
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"mega_gdn_prefill install not found next to {here}; run "
        "core.dependencies.prepare for the mega_gdn_prefill dependency first")


def library_path(work_dir=None):
    path = _install_dir(work_dir) / "lib" / LIB_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{LIB_NAME} not found at {path}")
    return path


def _eager_impl(mixed_qkv, b, a, z, conv_weight, conv_state, a_log, dt_bias,
                conv_state_read_indices, conv_state_write_indices, ssm_state_read_indices,
                ssm_state_write_indices, ssm_cache, mask_lower, mask_full, minus_identity,
                cu_seqlens, norm_weight):
    custom_ops_lib = importlib.import_module("custom_ops_lib")
    cu_host = cu_seqlens.detach().to("cpu").tolist()
    num_matrices = int(b.shape[1]) * sum(
        (cu_host[i + 1] - cu_host[i] + 127) // 128 for i in range(len(cu_host) - 1))
    out = custom_ops_lib.mega_gdn_prefill_op(
        mixed_qkv, b, a, z, conv_weight, conv_state, a_log, dt_bias,
        conv_state_read_indices, conv_state_write_indices, ssm_state_read_indices,
        ssm_state_write_indices, ssm_cache, mask_lower, mask_full, minus_identity,
        cu_seqlens, norm_weight, num_matrices)
    return out


def _eager_unavailable(*args, **kwargs):
    raise RuntimeError("eager MegaGdnPrefill requires the xllm custom_ops_lib "
                       "binding (graph-mode export only needs the Meta impl)")


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
    try:
        import custom_ops_lib  # noqa: F401
        lib.impl("mega_gdn_prefill", _eager_impl, "PrivateUse1")
    except ImportError:
        lib.impl("mega_gdn_prefill", _eager_unavailable, "PrivateUse1")
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


def ensure_loaded(work_dir=None):
    global _LOADED
    if _LOADED:
        return
    ctypes.CDLL(str(library_path(work_dir)), mode=ctypes.RTLD_GLOBAL)
    _register_torch_op()
    _LOADED = True


ensure_loaded()
