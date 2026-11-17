"""Activator for the CausalConv1dQkvGE custom-op-v2 wrapper.

On import: locates the installed libcust_opapi.so (CANN vendor package),
dlopens it (RTLD_GLOBAL) so GE sees the REG_OP, mounts the vendor package
into ASCEND_CUSTOM_OPP_PATH, then registers
torch.ops.ascend_ops.causal_conv1d_qkv (Meta + PrivateUse1) and the torchair
GE converter. With the binding pip-installed the operator imports from
anywhere; no shell exports are needed.

Contract: packed-QKV prefill only — x BF16 (T,D) varlen, weight BF16 (4,D),
no bias, conv_states BF16 (slots, 3, D) mutated in place, metadata (1-D
int64, host or NPU resident) selecting the sequence split / cache slots.
activation_mode=2 is the upstream packed cadence; other upstream modes are
forwarded as-is to the aclnn operator.

The schema keeps the scalars BEFORE the optional tensors: this repo hit a
torch 2.12 boxed-call pitfall (see causal-conv1d/README.md "坑位记录") where
scalar args following an optional Tensor? were dropped. `Tensor(a!)` on
conv_states is what orders dynamo's later reads after the in-place state
update.

The torch.library.Library handle must stay module-level alive — a
function-local would be GC'd, unregistering the schema and crashing dynamo.
torch_npu must be imported before torchair: torch_npu registers the
top-level torchair alias (its bundled copy) in sys.modules.
"""
import ctypes
import os

import torch_npu  # noqa: F401  PrivateUse1 dispatch + torchair alias injection

LIB_NAME = "libcust_opapi.so"
VENDOR_NAME = "causal_conv1d_qkv"
OP_NAME = "causal_conv1d_qkv"
GE_OP_NAME = "CausalConv1dQkvGE"
RUN_SYMBOL = "causal_conv1d_qkv_run"

SCHEMA = (
    "causal_conv1d_qkv(Tensor x, Tensor weight, Tensor(a!) conv_states, "
    "int q_dim, int k_dim, int v_dim, int head_dim=128, "
    "int activation_mode=2, int pad_slot_id=-1, int run_mode=0, "
    "ScalarType? out_dtype=None, Tensor? query_start_loc=None, "
    "Tensor? cache_indices=None, Tensor? initial_state_mode=None, "
    "Tensor? num_accepted_tokens=None) -> Tensor")

_LOADED = False
_TORCH_LIB = None
_BRIDGE_CALL = None


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
    roots. Other packages (the fla wheel, causal-conv1d) mount their own
    libcust_opapi.so into ASCEND_CUSTOM_OPP_PATH; the opp tree goes first so a
    same-named lib cannot shadow this package's. The loaded lib is verified to
    export causal_conv1d_qkv_run so a wrong lib fails loud.
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
        "bash causal-conv1d-qkv/install.sh with CANN set_env.sh sourced")


def _mount_opp(op_lib):
    """Expose the vendor package to GE discovery (idempotent): graph-mode
    kernel compile resolves the operator through ASCEND_CUSTOM_OPP_PATH."""
    vendor = os.path.dirname(os.path.dirname(os.path.dirname(op_lib)))
    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if vendor not in current.split(":"):
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
            f"{vendor}:{current}" if current else vendor)


def _eager_impl(x, weight, conv_states, q_dim, k_dim, v_dim, head_dim=128,
                activation_mode=2, pad_slot_id=-1, run_mode=0, out_dtype=None,
                query_start_loc=None, cache_indices=None,
                initial_state_mode=None, num_accepted_tokens=None):
    # This runs ~18x per request; resolve the JIT-loaded bridge once instead of
    # re-importing the loader (and re-reading its module global) on every call.
    global _BRIDGE_CALL
    if _BRIDGE_CALL is None:
        from causal_conv1d_qkv_binding import load
        _BRIDGE_CALL = load().causal_conv1d_qkv
    return _BRIDGE_CALL(
        x, weight, conv_states, q_dim, k_dim, v_dim, head_dim, activation_mode,
        pad_slot_id, run_mode, out_dtype, query_start_loc, cache_indices,
        initial_state_mode, num_accepted_tokens)


def _meta_impl(x, weight, conv_states, q_dim, k_dim, v_dim, head_dim=128,
               activation_mode=2, pad_slot_id=-1, run_mode=0, out_dtype=None,
               query_start_loc=None, cache_indices=None,
               initial_state_mode=None, num_accepted_tokens=None):
    import torch
    dtype = x.dtype if out_dtype is None else out_dtype
    return torch.empty(x.shape, dtype=dtype, device="meta")


def _out_dtype_code(out_dtype):
    """GE attr encoding (0 = BF16, 1 = FP16); None means "same as x" = BF16."""
    import torch
    if out_dtype is None or out_dtype == torch.bfloat16:
        return 0
    if out_dtype == torch.float16:
        return 1
    raise ValueError(f"CausalConv1dQkv: unsupported out_dtype {out_dtype} "
                     "(bf16/fp16 only)")


def _register_torch_op():
    import torch
    import torchair
    global _TORCH_LIB
    lib = torch.library.Library("ascend_ops", "FRAGMENT")
    _TORCH_LIB = lib
    lib.define(SCHEMA)
    lib.impl(OP_NAME, _eager_impl, "PrivateUse1")
    lib.impl(OP_NAME, _meta_impl, "Meta")

    def _convert(x, weight, conv_states, q_dim, k_dim, v_dim, head_dim=128,
                 activation_mode=2, pad_slot_id=-1, run_mode=0, out_dtype=None,
                 query_start_loc=None, cache_indices=None,
                 initial_state_mode=None, num_accepted_tokens=None,
                 meta_outputs=None):
        # Dict order is the GE node's input order, pinned both here and by the
        # build-time-injected ops-info entry; it must not drift. bias sits last
        # because it is the only optional input the packed cadence never
        # supplies. The wrapper reads OpExecuteContext inputs by IR index, so a
        # missing MIDDLE optional (e.g. cache_indices) does not shift its
        # neighbours.
        return torchair.ge.custom_op(
            GE_OP_NAME,
            inputs={"x": x, "weight": weight, "conv_states": conv_states,
                    "query_start_loc": query_start_loc,
                    "cache_indices": cache_indices,
                    "initial_state_mode": initial_state_mode,
                    "num_accepted_tokens": num_accepted_tokens,
                    "bias": None},
            outputs=["y"],
            attrs={"q_dim": torchair.ge.attr.Int(q_dim),
                   "k_dim": torchair.ge.attr.Int(k_dim),
                   "v_dim": torchair.ge.attr.Int(v_dim),
                   "head_dim": torchair.ge.attr.Int(head_dim),
                   "activation_mode": torchair.ge.attr.Int(activation_mode),
                   "pad_slot_id": torchair.ge.attr.Int(pad_slot_id),
                   "run_mode": torchair.ge.attr.Int(run_mode),
                   "out_dtype": torchair.ge.attr.Int(
                       _out_dtype_code(out_dtype))})

    torchair.register_fx_node_ge_converter(
        torch.ops.ascend_ops.causal_conv1d_qkv.default)(_convert)


def ensure_loaded():
    global _LOADED
    if _LOADED:
        return
    op_lib = find_op_lib()
    lib = ctypes.CDLL(op_lib, mode=ctypes.RTLD_GLOBAL)
    if not hasattr(lib, RUN_SYMBOL):
        raise ImportError(
            f"{op_lib} does not export {RUN_SYMBOL}; another vendor's "
            "libcust_opapi.so with the same name was found — check "
            "ASCEND_CUSTOM_OPP_PATH shadowing and reinstall this package")
    _mount_opp(op_lib)
    _register_torch_op()
    _LOADED = True


ensure_loaded()
