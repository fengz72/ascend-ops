// GE custom op prototype for the cannbotdsl DN prefix-in-Q attention kernel.
// Op type name must match the REG_AUTO_MAPPING_OP class and any frontend
// converter (three-way consistency per custom_op_v2 guide).
#ifndef PREFIX_ATTN_DN_GE_CUSTOM_OP_INCLUDE_PREFIX_ATTN_DN_OP_H_
#define PREFIX_ATTN_DN_GE_CUSTOM_OP_INCLUDE_PREFIX_ATTN_DN_OP_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {
REG_OP(PrefixAttnDn)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(actual_seq_lengths, TensorType({DT_INT32}))
    .ATTR(scale, Float, 0.0)
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(PrefixAttnDn);
}  // namespace ge

#endif  // PREFIX_ATTN_DN_GE_CUSTOM_OP_INCLUDE_PREFIX_ATTN_DN_OP_H_
