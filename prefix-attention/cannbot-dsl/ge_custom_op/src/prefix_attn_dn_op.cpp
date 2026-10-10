// PrefixAttnDn GE custom op (custom_op_v2, eager execute / scenario A).
//
// One graph node wraps the full DN host sequence: zero-init metadata
// workspace -> AICPU metadata planner launch -> DN macro attention kernel
// launch (schema-driven ABI v9 frame), both async on the GE execution
// stream.  Kernels are the prebuilt cannbotdsl artifacts under
// <this .so dir>/artifacts/ (see scripts/export_artifacts.py); the launcher
// was verified bit-exact against the python host path (probe/).
#include <cmath>
#include <iostream>

#include "acl/acl.h"
#include "graph/custom_op.h"
#include "register/op_impl_registry.h"

#include "artifacts.h"
#include "dsl_launcher.h"
#include "prefix_attn_dn_op.h"

namespace {

constexpr int64_t kHeadDim = 64;
constexpr size_t kMetadataBytes = PREFIX_ATTN_METADATA_ELEMS * sizeof(int32_t);

#define PREFIX_ATTN_DN_FAIL(msg)                                        \
  do {                                                                  \
    std::cerr << "[PrefixAttnDn] " << msg << std::endl;                 \
    return ge::GRAPH_FAILED;                                            \
  } while (0)

}  // namespace

class PrefixAttnDn : public ge::EagerExecuteOp {
 public:
  ge::graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    const gert::Tensor *query = ctx->GetInputTensor(0);
    const gert::Tensor *key = ctx->GetInputTensor(1);
    const gert::Tensor *value = ctx->GetInputTensor(2);
    const gert::Tensor *bounds = ctx->GetInputTensor(3);
    if (query == nullptr || key == nullptr || value == nullptr ||
        bounds == nullptr) {
      PREFIX_ATTN_DN_FAIL("null input tensor");
    }

    // Contract mirror of prefix_attn/validate.py (host-side, no D2H).
    const gert::Shape &q_shape = query->GetShape().GetStorageShape();
    const gert::Shape &k_shape = key->GetShape().GetStorageShape();
    const gert::Shape &v_shape = value->GetShape().GetStorageShape();
    const gert::Shape &b_shape = bounds->GetShape().GetStorageShape();
    if (q_shape.GetDimNum() != 3 || k_shape.GetDimNum() != 3 ||
        v_shape.GetDimNum() != 3) {
      PREFIX_ATTN_DN_FAIL("query/key/value must be rank-3 TND tensors");
    }
    if (b_shape.GetDimNum() != 1 || b_shape.GetDim(0) < 2) {
      PREFIX_ATTN_DN_FAIL("actual_seq_lengths must be 1-D [B+1], B>=1");
    }
    const int64_t total = q_shape.GetDim(0);
    const int64_t n1 = q_shape.GetDim(1);
    const int64_t head_dim = q_shape.GetDim(2);
    const int64_t n2 = k_shape.GetDim(1);
    if (head_dim != kHeadDim || k_shape.GetDim(2) != kHeadDim ||
        v_shape.GetDim(2) != kHeadDim) {
      PREFIX_ATTN_DN_FAIL("only D=64 is supported");
    }
    if (k_shape.GetDim(0) != total || v_shape.GetDim(0) != total ||
        v_shape.GetDim(1) != n2) {
      PREFIX_ATTN_DN_FAIL("Q/K/V must have matching T and equal K/V shapes");
    }
    if (n2 <= 0 || n1 < n2 || n1 % n2 != 0) {
      PREFIX_ATTN_DN_FAIL("N1 must be a positive multiple of N2");
    }
    const ge::DataType q_dtype = query->GetDataType();
    if ((q_dtype != ge::DT_FLOAT16 && q_dtype != ge::DT_BF16) ||
        key->GetDataType() != q_dtype || value->GetDataType() != q_dtype) {
      PREFIX_ATTN_DN_FAIL("query/key/value must all be fp16 or all bf16");
    }
    if (bounds->GetDataType() != ge::DT_INT32) {
      PREFIX_ATTN_DN_FAIL("actual_seq_lengths must be int32");
    }

    float scale = 0.0F;
    const gert::RuntimeAttrs *attrs = ctx->GetAttrs();
    if (attrs != nullptr) {
      const float *scale_attr = attrs->GetFloat(0);
      if (scale_attr != nullptr) {
        scale = *scale_attr;
      }
    }
    if (!(scale > 0.0F)) {
      scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    }

    gert::Tensor *output = ctx->MallocOutputTensor(0, query->GetShape(),
                                                   query->GetFormat(),
                                                   q_dtype);
    if (output == nullptr) {
      PREFIX_ATTN_DN_FAIL("failed to malloc output tensor");
    }
    void *metadata = ctx->MallocWorkSpace(kMetadataBytes);
    if (metadata == nullptr) {
      PREFIX_ATTN_DN_FAIL("failed to malloc metadata workspace");
    }
    aclrtStream stream = ctx->GetStream();
    // The planner assumes a zero-initialized buffer (torch.zeros on the
    // python side); keep the memset stream-ordered before the launches.
    if (aclrtMemsetAsync(metadata, kMetadataBytes, 0, kMetadataBytes,
                         stream) != ACL_SUCCESS) {
      PREFIX_ATTN_DN_FAIL("aclrtMemsetAsync failed on metadata workspace");
    }

    prefix_attn::DnLaunchParams params{};
    params.out = output->GetAddr();
    params.query = const_cast<void *>(query->GetAddr());
    params.key = const_cast<void *>(key->GetAddr());
    params.value = const_cast<void *>(value->GetAddr());
    params.total_tokens = total;
    params.n1 = n1;
    params.n2 = n2;
    params.bounds = const_cast<void *>(bounds->GetAddr());
    params.bounds_len = b_shape.GetDim(0);
    params.metadata = metadata;
    params.scale = scale;
    params.stream = stream;
    params.bf16 = q_dtype == ge::DT_BF16;
    const int status = prefix_attn::LauncherRun(params);
    if (status != 0) {
      std::cerr << "[PrefixAttnDn] launch failed: status=" << status
                << " err=" << prefix_attn::LauncherError() << std::endl;
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }
};

REG_AUTO_MAPPING_OP(PrefixAttnDn);
