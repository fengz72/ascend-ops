"""torchair FX -> GE converter for torch.ops.prefix_attn.dn.

Maps the FX node onto the GE custom op "PrefixAttnDn" (custom_op_v2 eager
execute op delivered by ge_custom_op/opp).  Requires ASCEND_CUSTOM_OPP_PATH
to include the opp root so torchair's IR registry knows the op type.
"""
import torch  # noqa: F401  (ensures torch.ops.prefix_attn is populated)
from torch_npu.dynamo import torchair

from torchair.ge import attr as ge_attr
from torchair.ge import custom_op as ge_custom_op


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
