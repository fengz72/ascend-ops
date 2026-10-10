"""PyTorch frontend for prefix_attn_dn: torch.ops registration + GE/torchair
wiring.  One import makes the operator usable in torch:

    import torch, torch_npu
    import prefix_attn.torch_ops            # <- this module

    out = torch.ops.prefix_attn.dn(q, k, v, bounds)          # eager
    compiled = torch.compile(model, backend=torchair.get_npu_backend())
                                                             # graph mode

What happens at import:
  1. torch.ops.prefix_attn.dn is defined (PrivateUse1 -> DSL python host
     path, Meta -> shape/dtype propagation).
  2. The bundled GE custom-op OPP package (PrefixAttnDn) is located and
     prepended to ASCEND_CUSTOM_OPP_PATH so GE/torchair can find it at
     initialize time.  Override with PREFIX_ATTN_DN_OPP=<opp root>.
  3. The torchair FX->GE converter is registered when torchair is
     importable (bundled with torch_npu as torch_npu.dynamo.torchair).

Eager mode needs nothing else.  Graph (torchair) mode additionally needs
the GE deliverables built: run scripts/install.sh once per board.
"""
import os
from pathlib import Path

import torch

from .dn import prefix_attn_dn

_PKG_DIR = Path(__file__).resolve().parent

torch.library.define(
    "prefix_attn::dn",
    "(Tensor query, Tensor key, Tensor value, Tensor actual_seq_lengths, "
    "float scale=0.0) -> Tensor",
)


@torch.library.impl("prefix_attn::dn", "PrivateUse1")
def _dn_privateuse1(query, key, value, actual_seq_lengths, scale=0.0):
    return prefix_attn_dn(query, key, value, actual_seq_lengths,
                          scale=None if scale <= 0.0 else scale)


@torch.library.impl("prefix_attn::dn", "Meta")
def _dn_meta(query, key, value, actual_seq_lengths, scale=0.0):
    return torch.empty_like(query)


def _opp_deliverable(root):
    return root / "op_graph/lib/linux/aarch64/libcust_opapi.so"


def find_opp_root():
    """Locate the GE custom-op OPP root (None when not built yet)."""
    env = os.environ.get("PREFIX_ATTN_DN_OPP")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(_PKG_DIR / "ge" / "opp")               # wheel layout
    candidates.append(_PKG_DIR.parent / "ge_custom_op" / "opp")  # source tree
    for root in candidates:
        if _opp_deliverable(root).exists():
            return root
    return None


def mount_opp():
    """Prepend the bundled OPP root to ASCEND_CUSTOM_OPP_PATH (idempotent).

    GE reads the variable at initialize time, so setting it at import is
    sufficient as long as this module is imported before compiling/running
    a graph.  Returns the mounted root, or None when deliverables are
    missing (eager mode still works; graph mode will not).
    """
    root = find_opp_root()
    if root is None:
        return None
    root = str(root)
    current = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if root not in current.split(":"):
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
            f"{root}:{current}" if current else root)
    return root


def register_torchair_converter():
    """Register the FX->GE converter; returns True on success.

    Safe to call multiple times only once per process (torchair keeps a
    registry); import-time failure to import torchair is not an error —
    eager mode does not need it.
    """
    try:
        from torch_npu.dynamo import torchair
        from torchair.ge import attr as ge_attr
        from torchair.ge import custom_op as ge_custom_op
    except ImportError:
        return False

    @torchair.register_fx_node_ge_converter(torch.ops.prefix_attn.dn.default)
    def _convert_prefix_attn_dn(query, key, value, actual_seq_lengths,
                                scale=0.0, meta_outputs=None):
        return ge_custom_op(
            "PrefixAttnDn",
            inputs={
                "query": query,
                "key": key,
                "value": value,
                "actual_seq_lengths": actual_seq_lengths,
            },
            attrs={"scale": ge_attr.Float(float(scale))},
            outputs=["attention_out"],
        )

    return True


OPP_ROOT = mount_opp()
TORCHAIR_CONVERTER_REGISTERED = register_torchair_converter()
