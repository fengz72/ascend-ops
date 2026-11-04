// MegaGdnPrefill — GE custom-op-v2 wrapper for the vendored MegaGdnPrefillOp.
// Compiled into the same libcust_opapi.so (hooked by cmake/custom_build.cmake);
// Execute calls aclnnMegaGdnPrefillOp by direct linkage.
//
// num_matrices is derived from cu_seqlens at Execute time (keeps B/S dynamic;
// the upstream INT64 attribute would bake shapes into the graph).
// Output norm_output is a separately allocated tensor (ctx->MallocOutputTensor,
// upstream contract: z stays a pure input). An earlier design aliased the
// output onto z's buffer (MakeOutputRefInput) to dodge the GE output-pool
// small->large treading, but that defect is fixed by the fix_9.1.0 executor
// package, and the nonstandard alias was the prime suspect for the ge_runtime
// (default allocator) cross-execution corruption.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "graph/custom_op.h"
#include "graph/operator_reg.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/infer_shape_context.h"

#include "mega_gdn_prefill_common.h"

namespace ge {
REG_OP(MegaGdnPrefill)
    .INPUT(mixed_qkv, TensorType({DT_BF16}))
    .INPUT(b, TensorType({DT_BF16}))
    .INPUT(a, TensorType({DT_BF16}))
    .INPUT(z, TensorType({DT_BF16}))
    .INPUT(conv_weight, TensorType({DT_BF16}))
    .INPUT(conv_state, TensorType({DT_BF16}))
    .INPUT(a_log, TensorType({DT_FLOAT}))
    .INPUT(dt_bias, TensorType({DT_FLOAT}))
    .INPUT(conv_state_read_indices, TensorType({DT_INT32}))
    .INPUT(conv_state_write_indices, TensorType({DT_INT32}))
    .INPUT(ssm_state_read_indices, TensorType({DT_INT32}))
    .INPUT(ssm_state_write_indices, TensorType({DT_INT32}))
    .INPUT(ssm_cache, TensorType({DT_FLOAT}))
    .INPUT(mask_lower, TensorType({DT_FLOAT}))
    .INPUT(mask_full, TensorType({DT_FLOAT}))
    .INPUT(minus_identity, TensorType({DT_BF16}))
    .INPUT(cu_seqlens, TensorType({DT_INT32}))
    .INPUT(norm_weight, TensorType({DT_BF16}))
    .OUTPUT(norm_output, TensorType({DT_BF16}))
    .OP_END_FACTORY_REG(MegaGdnPrefill)
}  // namespace ge

namespace {

aclDataType ToAclDataType(ge::DataType dataType) {
  switch (dataType) {
    case ge::DT_BF16:
      return ACL_BF16;
    case ge::DT_FLOAT16:
      return ACL_FLOAT16;
    case ge::DT_FLOAT:
      return ACL_FLOAT;
    case ge::DT_INT32:
      return ACL_INT32;
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

size_t TensorBytes(const gert::Tensor *tensor) {
  const ge::DataType dataType = tensor->GetDataType();
  size_t elementSize = 0;
  switch (dataType) {
    case ge::DT_BF16:
    case ge::DT_FLOAT16:
    case ge::DT_INT16:
    case ge::DT_UINT16:
      elementSize = 2;
      break;
    case ge::DT_FLOAT:
    case ge::DT_INT32:
    case ge::DT_UINT32:
      elementSize = 4;
      break;
    case ge::DT_INT64:
    case ge::DT_UINT64:
    case ge::DT_DOUBLE:
      elementSize = 8;
      break;
    case ge::DT_INT8:
    case ge::DT_UINT8:
    case ge::DT_BOOL:
      elementSize = 1;
      break;
    default:
      elementSize = 0;
      break;
  }
  return static_cast<size_t>(tensor->GetShapeSize()) * elementSize;
}

aclTensor *MakeScratchAclTensor(const gert::Tensor *tensor, void *addr) {
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
                         strides.data(), 0, ACL_FORMAT_ND, dims.data(), dimNum, addr);
}

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

constexpr int kNumInputs = 18;
constexpr int kIdxB = 1;
constexpr int kIdxZ = 3;
constexpr int kIdxConvState = 5;
constexpr int kIdxSsmCache = 12;
constexpr int kIdxCuSeqlens = 16;

}  // namespace

class MegaGdnPrefill : public ge::EagerExecuteOp, public ge::ShapeInferOp {
 public:
  ge::graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    const gert::Tensor *inputs[kNumInputs];
    for (int i = 0; i < kNumInputs; ++i) {
      inputs[i] = ctx->GetInputTensor(i);
      if (inputs[i] == nullptr) {
        { std::fprintf(stderr, "MegaGdnPrefill: fail input_null\n"); return ge::GRAPH_FAILED; }
      }
    }
    const auto &cuShape = inputs[kIdxCuSeqlens]->GetOriginShape();
    if (cuShape.GetDimNum() != 1) {
      { std::fprintf(stderr, "MegaGdnPrefill: fail cu_shape\n"); return ge::GRAPH_FAILED; }
    }
    const int64_t batch = cuShape.GetDim(0) - 1;
    const auto &bShape = inputs[kIdxB]->GetOriginShape();
    if (batch < 1 || bShape.GetDimNum() != 2) {
      { std::fprintf(stderr, "MegaGdnPrefill: fail batch_b_shape\n"); return ge::GRAPH_FAILED; }
    }
    const int64_t heads = bShape.GetDim(1);
    std::vector<int32_t> cuHost(static_cast<size_t>(batch + 1));
    const size_t cuBytes = cuHost.size() * sizeof(int32_t);
    if (aclrtMemcpyAsync(cuHost.data(), cuBytes, inputs[kIdxCuSeqlens]->GetAddr(), cuBytes,
                         ACL_MEMCPY_DEVICE_TO_HOST, ctx->GetStream()) != ACL_SUCCESS ||
        aclrtSynchronizeStream(ctx->GetStream()) != ACL_SUCCESS) {
      { std::fprintf(stderr, "MegaGdnPrefill: fail memcpy\n"); return ge::GRAPH_FAILED; }
    }
    int64_t numMatrices = 0;
    for (int64_t i = 0; i < batch; ++i) {
      const int64_t length = static_cast<int64_t>(cuHost[i + 1]) - cuHost[i];
      if (length < 0) {
        { std::fprintf(stderr, "MegaGdnPrefill: fail num_matrices\n"); return ge::GRAPH_FAILED; }
      }
      numMatrices += (length + 127) / 128;
    }
    numMatrices *= heads;
    if (numMatrices <= 0) {
      { std::fprintf(stderr, "MegaGdnPrefill: fail norm_output\n"); return ge::GRAPH_FAILED; }
    }
    const uint64_t fftsAddr = ResolveFftsAddr();

    gert::Tensor *normOutput = ctx->MallocOutputTensor(
        0, CopyStorageShape(inputs[kIdxZ]), inputs[kIdxZ]->GetFormat(),
        inputs[kIdxZ]->GetDataType());
    if (normOutput == nullptr) {
      std::fprintf(stderr, "MegaGdnPrefill: fail malloc_norm_output\n");
      return ge::GRAPH_FAILED;
    }
    const size_t convStateBytes = TensorBytes(inputs[kIdxConvState]);
    const size_t ssmCacheBytes = TensorBytes(inputs[kIdxSsmCache]);
    const size_t ssmCacheOffset = (convStateBytes + 63) / 64 * 64;
    const size_t scratchBytes = ssmCacheOffset + ssmCacheBytes;
    uint8_t *scratch = scratchBytes > 0
        ? static_cast<uint8_t *>(ctx->MallocWorkSpace(scratchBytes))
        : nullptr;
    if (scratchBytes > 0 && scratch == nullptr) {
      std::fprintf(stderr, "MegaGdnPrefill: fail malloc_state_scratch\n");
      return ge::GRAPH_FAILED;
    }
    aclTensor *aclConvStateOut = MakeScratchAclTensor(
        inputs[kIdxConvState], scratch);
    aclTensor *aclSsmCacheOut = MakeScratchAclTensor(
        inputs[kIdxSsmCache], scratch + ssmCacheOffset);

    aclTensor *aclInputs[kNumInputs];
    for (int i = 0; i < kNumInputs; ++i) {
      aclInputs[i] = MakeAclTensor(inputs[i]);
      if (aclInputs[i] == nullptr) {
        for (int j = 0; j < i; ++j) {
          aclDestroyTensor(aclInputs[j]);
        }
        std::fprintf(stderr, "MegaGdnPrefill: fail make_acl %d\n", i);
        return ge::GRAPH_FAILED;
      }
    }
    aclTensor *aclNormOutput = MakeAclTensor(normOutput);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = aclnnMegaGdnPrefillOpGetWorkspaceSize(
        aclInputs[0], aclInputs[1], aclInputs[2], aclInputs[3], aclInputs[4], aclInputs[5],
        aclInputs[6], aclInputs[7], aclInputs[8], aclInputs[9], aclInputs[10], aclInputs[11],
        aclInputs[12], aclInputs[13], aclInputs[14], aclInputs[15], aclInputs[16],
        aclInputs[17], static_cast<int64_t>(fftsAddr), numMatrices, aclNormOutput,
        aclConvStateOut, aclSsmCacheOut, &workspaceSize, &executor);
    if (status == 0) {
      void *workspace = workspaceSize > 0
          ? ctx->MallocWorkSpace(static_cast<size_t>(workspaceSize))
          : nullptr;
      if (workspaceSize > 0 && workspace == nullptr) {
        status = 1;
      } else {
        status = aclnnMegaGdnPrefillOp(workspace, workspaceSize, executor, ctx->GetStream());
      }
    }
    for (int i = 0; i < kNumInputs; ++i) {
      aclDestroyTensor(aclInputs[i]);
    }
    aclDestroyTensor(aclNormOutput);
    aclDestroyTensor(aclConvStateOut);
    aclDestroyTensor(aclSsmCacheOut);
    if (status != 0) {
      std::fprintf(stderr, "MegaGdnPrefill: aclnn status %d (workspaceSize=%llu numMatrices=%lld ffts=%llu)\n",
                   static_cast<int>(status), static_cast<unsigned long long>(workspaceSize),
                   static_cast<long long>(numMatrices), static_cast<unsigned long long>(fftsAddr));
    }
    return status == 0 ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
  }

  ge::graphStatus InferShape(gert::InferShapeContext *ctx) override {
    const gert::Shape *zShape = ctx->GetInputShape(kIdxZ);
    if (zShape == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *ctx->GetOutputShape(0) = *zShape;
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    if (ctx->SetOutputDataType(0, ctx->GetInputDataType(kIdxZ)) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }
};

REG_AUTO_MAPPING_OP(MegaGdnPrefill);
