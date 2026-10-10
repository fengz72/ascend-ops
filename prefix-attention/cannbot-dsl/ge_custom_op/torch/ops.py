"""torch.ops registration for the prefix_attn DN operator.

Schema:  prefix_attn::dn(query, key, value, actual_seq_lengths, scale=0.0)
  - PrivateUse1 (NPU eager): calls the cannbotdsl python host path
  - Meta: shape/dtype propagation for dynamo tracing / export

Import this module once before using torch.ops.prefix_attn.dn or compiling
with torchair (together with converter.py for graph lowering).
"""
import sys
from pathlib import Path

import torch

_DSL_ROOT = Path(__file__).resolve().parents[2]  # cannbot-dsl/
if str(_DSL_ROOT) not in sys.path:
    sys.path.insert(0, str(_DSL_ROOT))

torch.library.define(
    "prefix_attn::dn",
    "(Tensor query, Tensor key, Tensor value, Tensor actual_seq_lengths, "
    "float scale=0.0) -> Tensor",
)


@torch.library.impl("prefix_attn::dn", "PrivateUse1")
def _dn_privateuse1(query, key, value, actual_seq_lengths, scale=0.0):
    from prefix_attn import prefix_attn_dn
    return prefix_attn_dn(query, key, value, actual_seq_lengths,
                          scale=None if scale <= 0.0 else scale)


@torch.library.impl("prefix_attn::dn", "Meta")
def _dn_meta(query, key, value, actual_seq_lengths, scale=0.0):
    return torch.empty_like(query)
