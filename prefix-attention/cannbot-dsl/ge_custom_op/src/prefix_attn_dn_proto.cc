// Shape/dtype inference registration for PrefixAttnDn (compiled into both the
// ES-generation proto library and the runtime libcust_opapi.so).
#include "prefix_attn_dn_op.h"

namespace {

ge::graphStatus InferShapeForPrefixAttnDn(gert::InferShapeContext *ctx) {
  const auto *input_shape = ctx->GetInputShape(0U);
  auto *output_shape = ctx->GetOutputShape(0U);
  if ((input_shape == nullptr) || (output_shape == nullptr)) {
    return ge::GRAPH_FAILED;
  }
  *output_shape = *input_shape;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForPrefixAttnDn(gert::InferDataTypeContext *ctx) {
  return ctx->SetOutputDataType(0U, ctx->GetInputDataType(0U));
}

IMPL_OP(PrefixAttnDn)
    .InferShape(InferShapeForPrefixAttnDn)
    .InferDataType(InferDataTypeForPrefixAttnDn);

}  // namespace
