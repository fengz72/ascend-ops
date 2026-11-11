// CausalConv1dGE — GE custom-op-v2 wrapper for the vendored CausalConv1d.
// Compiled into the same libcust_opapi.so (hooked by cmake/custom_build.cmake);
// Execute calls aclnnCausalConv1d by direct linkage.
//
// Scope: dense FN prefill only — x is dim-last (B,S,D) or (T,D), weight (W,D),
// bias (D,) optional, activation "none"/"silu"/"swish", headNum for optional
// head-split output. runMode=0 (FN), no conv states, no metadata tensors
// (the upstream kernel rejects queryStartLoc on dense (B,S,D) input).
// The varlen/cache routes stay reachable through the upstream aclnn API.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "aclnn/acl_meta.h"
#include "acl/acl_rt.h"
#include "graph/custom_op.h"
#include "graph/operator_reg.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/infer_shape_context.h"

// Generated aclnn declarations (autogen header rides the vendor package
// include dir; declared here to keep the wrapper self-contained).
extern "C" aclnnStatus aclnnCausalConv1dGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *biasOptional,
    const aclTensor *convStatesOptional, const aclTensor *queryStartLocOptional,
    const aclTensor *cacheIndicesOptional, const aclTensor *hasInitialStateOptional,
    const aclTensor *numAcceptedTokensOptional,
    const aclIntArray *queryStartLocCpuOptional,
    const aclIntArray *cacheIndicesCpuOptional,
    const aclIntArray *hasInitialStateCpuOptional,
    const aclIntArray *numAcceptedTokensCpuOptional,
    char *activationOptional, int64_t padSlotId, int64_t nullBlockId,
    int64_t runMode, int64_t headNum, int64_t maxQueryLen, const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnCausalConv1d(void *workspace, uint64_t workspaceSize,
                                         aclOpExecutor *executor,
                                         aclrtStream stream);

namespace ge {
REG_OP(CausalConv1dGE)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .ATTR(activation, String, "silu")
    .ATTR(head_num, Int, 0)
    .OP_END_FACTORY_REG(CausalConv1dGE)
}  // namespace ge

namespace {

aclDataType ToAclDataType(ge::DataType dataType) {
  switch (dataType) {
    case ge::DT_BF16:
      return ACL_BF16;
    case ge::DT_FLOAT16:
      return ACL_FLOAT16;
    default:
      return ACL_DT_UNDEFINED;
  }
}

aclTensor *MakeAclTensor(const gert::Tensor *tensor) {
  const auto &shape = tensor->GetOriginShape();
  const size_t dimNum = shape.GetDimNum();
  std::vector<int64_t> dims(dimNum);
  std::vector<int64_t> strides(dimNum, 1);
  for (size_t i = 0; i < dimNum; ++i) {
    dims[i] = shape.GetDim(i);
  }
  for (size_t i = dimNum - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * dims[i];
  }
  return aclCreateTensor(dims.data(), dimNum, ToAclDataType(tensor->GetDataType()),
                         strides.data(), 0, ACL_FORMAT_ND, dims.data(), dimNum,
                         const_cast<void *>(tensor->GetAddr()));
}

constexpr int kNumInputs = 3;  // x, weight, bias(optional)
constexpr int kIdxBias = 2;

}  // namespace

class CausalConv1dGE : public ge::EagerExecuteOp, public ge::ShapeInferOp {
 public:
  ge::graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    const gert::Tensor *inputs[kNumInputs];
    for (int i = 0; i < kNumInputs; ++i) {
      inputs[i] = ctx->GetInputTensor(i);  // optional slots may be null
    }
    if (inputs[0] == nullptr || inputs[1] == nullptr) {
      std::fprintf(stderr, "CausalConv1dGE: fail x_or_weight_null\n");
      return ge::GRAPH_FAILED;
    }
    const char *activation = ctx->GetAttrs()->GetStr(0);
    const int64_t headNum = *ctx->GetAttrs()->GetInt(1);

    gert::Tensor *output = ctx->MallocOutputTensor(
        0, CopyStorageShape(inputs[0]), inputs[0]->GetFormat(),
        inputs[0]->GetDataType());
    if (output == nullptr) {
      std::fprintf(stderr, "CausalConv1dGE: fail malloc_output\n");
      return ge::GRAPH_FAILED;
    }

    aclTensor *aclInputs[kNumInputs] = {
        MakeAclTensor(inputs[0]), MakeAclTensor(inputs[1]),
        inputs[kIdxBias] != nullptr ? MakeAclTensor(inputs[kIdxBias]) : nullptr};
    for (int i = 0; i < kNumInputs; ++i) {
      if (i != kIdxBias && aclInputs[i] == nullptr) {
        for (int j = 0; j < i; ++j) {
          aclDestroyTensor(aclInputs[j]);
        }
        std::fprintf(stderr, "CausalConv1dGE: fail make_acl %d\n", i);
        return ge::GRAPH_FAILED;
      }
    }
    aclTensor *aclOutput = MakeAclTensor(output);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = aclnnCausalConv1dGetWorkspaceSize(
        aclInputs[0], aclInputs[1], aclInputs[2],
        nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        const_cast<char *>(activation),
        -1, 0, 0, headNum, -1,
        aclOutput, &workspaceSize, &executor);
    if (status == 0) {
      void *workspace = workspaceSize > 0
          ? ctx->MallocWorkSpace(static_cast<size_t>(workspaceSize))
          : nullptr;
      if (workspaceSize > 0 && workspace == nullptr) {
        status = 1;
      } else {
        status = aclnnCausalConv1d(workspace, workspaceSize, executor, ctx->GetStream());
      }
    }
    for (int i = 0; i < kNumInputs; ++i) {
      if (aclInputs[i] != nullptr) {
        aclDestroyTensor(aclInputs[i]);
      }
    }
    aclDestroyTensor(aclOutput);
    if (status != 0) {
      std::fprintf(stderr, "CausalConv1dGE: aclnn status %d\n",
                   static_cast<int>(status));
    }
    return status == 0 ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
  }

  ge::graphStatus InferShape(gert::InferShapeContext *ctx) override {
    const gert::Shape *xShape = ctx->GetInputShape(0);
    if (xShape == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *ctx->GetOutputShape(0) = *xShape;
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    if (ctx->SetOutputDataType(0, ctx->GetInputDataType(0)) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }

 private:
  gert::StorageShape CopyStorageShape(const gert::Tensor *tensor) {
    gert::StorageShape shape;
    const auto &origin = tensor->GetOriginShape();
    auto &originOut = shape.MutableOriginShape();
    auto &storageOut = shape.MutableStorageShape();
    originOut.SetDimNum(origin.GetDimNum());
    storageOut.SetDimNum(origin.GetDimNum());
    for (size_t i = 0; i < origin.GetDimNum(); ++i) {
      originOut.SetDim(i, origin.GetDim(i));
      storageOut.SetDim(i, origin.GetDim(i));
    }
    return shape;
  }
};

REG_AUTO_MAPPING_OP(CausalConv1dGE);
