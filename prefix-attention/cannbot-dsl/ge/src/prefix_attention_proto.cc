// Shape/dtype inference registration for PrefixAttention (compiled into both the
// ES-generation proto library and the runtime libcust_opapi.so).
#include "prefix_attention_op.h"

namespace {

ge::graphStatus InferShapeForPrefixAttention(gert::InferShapeContext *ctx) {
  const auto *input_shape = ctx->GetInputShape(0U);
  auto *output_shape = ctx->GetOutputShape(0U);
  if ((input_shape == nullptr) || (output_shape == nullptr)) {
    return ge::GRAPH_FAILED;
  }
  *output_shape = *input_shape;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForPrefixAttention(gert::InferDataTypeContext *ctx) {
  return ctx->SetOutputDataType(0U, ctx->GetInputDataType(0U));
}

IMPL_OP(PrefixAttention)
    .InferShape(InferShapeForPrefixAttention)
    .InferDataType(InferDataTypeForPrefixAttention);

}  // namespace
