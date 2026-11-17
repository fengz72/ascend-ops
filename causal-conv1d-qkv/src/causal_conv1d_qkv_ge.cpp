// CausalConv1dQkvGE — GE aclnn-lowered wrapper for the vendored CausalConv1dQkv
// (packed-QKV prefill variant of CausalConv1d: depthwise conv over the state
// history -> silu -> split last dim into [q|k|v] -> per-head L2 normalize of
// q/k). Compiled into the same libcust_opapi.so (hooked by
// cmake/custom_build.cmake); Execute calls aclnnCausalConv1dQkv by direct
// linkage.
//
// Registered through IMPL_OP + OpExecuteFunc/HostInputs instead of
// REG_AUTO_MAPPING_OP: that is the only CANN 9.1.0 mechanism that lets the
// framework materialize the four ValueDepend metadata inputs on HOST
// (aclnn_node_converter.cc: `functions->IsHostInput(ir_index) ? kOnHost : ...`),
// so Execute reads their values directly instead of doing its own D2H.
//
// conv_states is a plain INPUT because the kernel rewrites its last width-1
// rows in place (the sliding-window history); there is no separate state
// output (precedent: MegaGdnPrefill's in-place conv_state/ssm_cache contract).
//
// out_dtype encodes the output dtype (0 = BF16, 1 = FP16) instead of deriving
// it from x: inputs stay BF16 while y is FP16 or BF16, so InferDataType must
// follow the attr (this is where this wrapper differs from CausalConv1dGE).

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "causal_conv1d_qkv_aclnn.h"
#include "causal_conv1d_qkv_common.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {
// Input order is frozen: the torchair converter in python/ and the ops-info
// entry injected at build time both pin these names and positions. bias stays
// last because it is the one optional input that is never present in the packed
// cadence, so putting it behind the always-present tensors keeps the natural
// reading order for everything that is. Middle-omission safety does not depend
// on this order: the OpExecuteContext accessors below are IR-indexed, so an
// absent slot (e.g. cache_indices) is reported as null without shifting its
// neighbours.
REG_OP(CausalConv1dQkvGE)
    .INPUT(x, TensorType({DT_BF16}))
    .INPUT(weight, TensorType({DT_BF16}))
    .INPUT(conv_states, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(cache_indices, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(initial_state_mode, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(num_accepted_tokens, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(q_dim, Int, 1024)
    .ATTR(k_dim, Int, 1024)
    .ATTR(v_dim, Int, 3072)
    .ATTR(head_dim, Int, 128)
    .ATTR(activation_mode, Int, 2)
    .ATTR(pad_slot_id, Int, -1)
    .ATTR(run_mode, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .OP_END_FACTORY_REG(CausalConv1dQkvGE)
}  // namespace ge

namespace {

constexpr char kTag[] = "CausalConv1dQkvGE";

// TensorType({DT_FLOAT16, DT_BF16}) above is the GE-visible dtype list; the
// concrete output dtype comes from the out_dtype attr (see InferDataType).

// IR input indices (REG_OP order above): x, weight, conv_states, then the four
// metadata slots, then bias.
constexpr int kIdxX = 0;
constexpr int kIdxWeight = 1;
constexpr int kIdxConvStates = 2;
constexpr int kIdxQueryStartLoc = 3;
constexpr int kIdxCacheIndices = 4;
constexpr int kIdxInitialStateMode = 5;
constexpr int kIdxNumAcceptedTokens = 6;
constexpr int kIdxBias = 7;
constexpr int kNumTensorInputs = 4;  // aclnn order: x, weight, bias, conv_states

// aclnn argument order (independent of the GE input order above).
constexpr int kAclX = 0;
constexpr int kAclWeight = 1;
constexpr int kAclBias = 2;
constexpr int kAclConvStates = 3;

constexpr size_t kAttrQDim = 0;
constexpr size_t kAttrKDim = 1;
constexpr size_t kAttrVDim = 2;
constexpr size_t kAttrHeadDim = 3;
constexpr size_t kAttrActivationMode = 4;
constexpr size_t kAttrPadSlotId = 5;
constexpr size_t kAttrRunMode = 6;
constexpr size_t kAttrOutDtype = 7;

constexpr int64_t kOutDtypeBf16 = 0;
constexpr int64_t kOutDtypeFp16 = 1;

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

ge::DataType ToGeDataType(int64_t outDtypeCode) {
  switch (outDtypeCode) {
    case kOutDtypeBf16:
      return ge::DT_BF16;
    case kOutDtypeFp16:
      return ge::DT_FLOAT16;
    default:
      return ge::DT_UNDEFINED;
  }
}

int64_t AttrInt(const gert::RuntimeAttrs *attrs, size_t index, int64_t fallback) {
  const int64_t *value = attrs->GetInt(index);
  return value != nullptr ? *value : fallback;
}

aclTensor *MakeAclTensor(const gert::Tensor *tensor) {
  // Strides/offset assume a contiguous layout: the vendored aclnn def marks all
  // tensors AutoContiguous and gert::Tensor exposes no strides to check.
  const auto &shape = tensor->GetOriginShape();
  const size_t dimNum = shape.GetDimNum();
  if (dimNum == 0) {
    // Guards the stride loop below (dimNum - 1 would wrap to SIZE_MAX).
    return nullptr;
  }
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

// HostInputs contract probe: with CAUSAL_CONV1D_QKV_TRACE=1, report where the
// metadata inputs actually landed, whether the D2H fallback fired, and the
// output handle. Inert when unset.
bool TraceEnabled() {
  static const bool enabled = std::getenv("CAUSAL_CONV1D_QKV_TRACE") != nullptr;
  return enabled;
}

int PlacementCode(const gert::Tensor *tensor) {
  return tensor == nullptr ? -1 : static_cast<int>(tensor->GetPlacement());
}

}  // namespace

UINT32 CausalConv1dQkvExecute(gert::OpExecuteContext *ctx) {
  if (ctx == nullptr) {
    std::fprintf(stderr, "%s: fail ctx_null\n", kTag);
    return ge::GRAPH_FAILED;
  }
  // IR-indexed accessors: an absent optional slot yields null in its own slot
  // and does not shift its neighbours.
  const gert::Tensor *x = ctx->GetRequiredInputTensor(kIdxX);
  const gert::Tensor *weight = ctx->GetRequiredInputTensor(kIdxWeight);
  const gert::Tensor *convStates = ctx->GetRequiredInputTensor(kIdxConvStates);
  const gert::Tensor *bias = ctx->GetOptionalInputTensor(kIdxBias);
  const gert::Tensor *metaSrc[kNumMetaInputs] = {
      ctx->GetOptionalInputTensor(kIdxQueryStartLoc),
      ctx->GetOptionalInputTensor(kIdxCacheIndices),
      ctx->GetOptionalInputTensor(kIdxInitialStateMode),
      ctx->GetOptionalInputTensor(kIdxNumAcceptedTokens)};
  if (x == nullptr || weight == nullptr || convStates == nullptr) {
    std::fprintf(stderr, "%s: fail required_input_null\n", kTag);
    return ge::GRAPH_FAILED;
  }
  const gert::RuntimeAttrs *attrs = ctx->GetAttrs();
  if (attrs == nullptr) {
    std::fprintf(stderr, "%s: fail attrs_null\n", kTag);
    return ge::GRAPH_FAILED;
  }
  const int64_t qDim = AttrInt(attrs, kAttrQDim, 1024);
  const int64_t kDim = AttrInt(attrs, kAttrKDim, 1024);
  const int64_t vDim = AttrInt(attrs, kAttrVDim, 3072);
  const int64_t headDim = AttrInt(attrs, kAttrHeadDim, 128);
  const int64_t activationMode = AttrInt(attrs, kAttrActivationMode, 2);
  const int64_t padSlotId = AttrInt(attrs, kAttrPadSlotId, -1);
  const int64_t runMode = AttrInt(attrs, kAttrRunMode, 0);

  const ge::DataType outDtype =
      ToGeDataType(AttrInt(attrs, kAttrOutDtype, kOutDtypeBf16));
  if (outDtype == ge::DT_UNDEFINED) {
    std::fprintf(stderr, "%s: fail out_dtype attr\n", kTag);
    return ge::GRAPH_FAILED;
  }

  const auto &xShape = x->GetOriginShape();
  const auto &wShape = weight->GetOriginShape();
  const int64_t xDimNum = static_cast<int64_t>(xShape.GetDimNum());
  if (xDimNum < 1 || wShape.GetDimNum() != 2) {
    std::fprintf(stderr, "%s: fail x_or_weight_shape\n", kTag);
    return ge::GRAPH_FAILED;
  }
  const int64_t dim = xShape.GetDim(xShape.GetDimNum() - 1);
  const int64_t width = wShape.GetDim(0);
  const bool hasBias = bias != nullptr;
  if (!CheckPackedLayout(kTag, activationMode, runMode, hasBias, xDimNum, dim,
                         width, qDim, kDim, vDim, headDim)) {
    return ge::GRAPH_FAILED;
  }

  // Output is allocated by the framework from InferShape/InferDataType.
  const gert::Tensor *output = ctx->GetOutputTensor(0);
  if (output == nullptr || output->GetAddr() == nullptr) {
    std::fprintf(stderr, "%s: fail output_tensor\n", kTag);
    return ge::GRAPH_FAILED;
  }
  if (output->GetDataType() != outDtype) {
    std::fprintf(stderr, "%s: fail output_dtype %d (attr %d)\n", kTag,
                 static_cast<int>(output->GetDataType()),
                 static_cast<int>(outDtype));
    return ge::GRAPH_FAILED;
  }

  // Metadata: HostInputs normally lands these on host (zero copy). A slot that
  // still arrives device-placed goes through the shared batched D2H fallback.
  const void *metaAddrs[kNumMetaInputs] = {nullptr, nullptr, nullptr, nullptr};
  int64_t metaCounts[kNumMetaInputs] = {0, 0, 0, 0};
  bool metaHost[kNumMetaInputs] = {false, false, false, false};
  for (int k = 0; k < kNumMetaInputs; ++k) {
    if (metaSrc[k] == nullptr) {
      continue;
    }
    const int64_t count = metaSrc[k]->GetShapeSize();
    if (count <= 0 || metaSrc[k]->GetAddr() == nullptr) {
      std::fprintf(stderr, "%s: fail meta_shape %d\n", kTag, k);
      return ge::GRAPH_FAILED;
    }
    metaAddrs[k] = metaSrc[k]->GetAddr();
    metaCounts[k] = count;
    metaHost[k] = gert::TensorPlacementUtils::IsOnHost(metaSrc[k]->GetPlacement());
  }
  MetaIntArrays metaArrays;
  if (!BuildMetaIntArrays(kTag, metaAddrs, metaCounts, metaHost,
                          static_cast<aclrtStream>(ctx->GetStream()),
                          &metaArrays)) {
    return ge::GRAPH_FAILED;
  }
  if (metaArrays.deviceStaged) {
    // Contract drift must be loud even without CAUSAL_CONV1D_QKV_TRACE: the
    // build-time ops-info entry (aclnnSupport) is what makes HostInputs deliver
    // host metadata, and without it every call pays this drain again. Once per
    // process keeps it readable under a benchmark.
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
      std::fprintf(stderr,
                   "%s: HostInputs not honoured for metadata; falling back to "
                   "stream-ordered D2H (check the vendor ops-info aclnnSupport "
                   "entry)\n",
                   kTag);
    }
  }

  aclTensor *aclInputs[kNumTensorInputs] = {
      MakeAclTensor(x), MakeAclTensor(weight),
      bias != nullptr ? MakeAclTensor(bias) : nullptr,
      MakeAclTensor(convStates)};
  aclTensor *aclOutput = MakeAclTensor(output);
  auto destroyAll = [&aclInputs, &aclOutput]() {
    for (int i = 0; i < kNumTensorInputs; ++i) {
      if (aclInputs[i] != nullptr) {
        aclDestroyTensor(aclInputs[i]);
      }
    }
    if (aclOutput != nullptr) {
      aclDestroyTensor(aclOutput);
    }
  };
  const gert::Tensor *tensorSrc[kNumTensorInputs] = {x, weight, bias, convStates};
  for (int i = 0; i < kNumTensorInputs; ++i) {
    // A present input whose wrap failed must not be treated as "absent": a
    // dropped bias or conv-state history would silently change results.
    if (aclInputs[i] == nullptr && tensorSrc[i] != nullptr) {
      destroyAll();
      std::fprintf(stderr, "%s: fail make_acl %d\n", kTag, i);
      return ge::GRAPH_FAILED;
    }
  }
  if (aclOutput == nullptr) {
    destroyAll();
    std::fprintf(stderr, "%s: fail make_acl out\n", kTag);
    return ge::GRAPH_FAILED;
  }

  if (TraceEnabled()) {
    static std::atomic<long long> calls{0};
    const long long call = calls.fetch_add(1);
    if (call < 3 || call % 200 == 0) {
      std::fprintf(
          stderr,
          "[qkv-trace] call=%lld x_place=%d meta_place=[%d,%d,%d,%d] "
          "fallback_d2h=%d out=%p out_place=%d stream=%p\n",
          call, PlacementCode(x), PlacementCode(metaSrc[0]),
          PlacementCode(metaSrc[1]), PlacementCode(metaSrc[2]),
          PlacementCode(metaSrc[3]), metaArrays.deviceStaged ? 1 : 0,
          const_cast<void *>(output->GetAddr()), PlacementCode(output),
          ctx->GetStream());
    }
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus status = aclnnCausalConv1dQkvGetWorkspaceSize(
      aclInputs[kAclX], aclInputs[kAclWeight], aclInputs[kAclBias],
      aclInputs[kAclConvStates], metaArrays.array[0], metaArrays.array[1],
      metaArrays.array[2], metaArrays.array[3], activationMode, padSlotId,
      runMode, qDim, kDim, vDim, headDim, aclOutput, &workspaceSize, &executor);
  if (status == 0) {
    // Framework-managed device workspace: no aclrtMalloc/Free and no drain.
    void *workspace = workspaceSize > 0 ? ctx->MallocWorkspace(
                                              static_cast<size_t>(workspaceSize))
                                        : nullptr;
    if (workspaceSize > 0 && workspace == nullptr) {
      status = 1;
    } else {
      status = aclnnCausalConv1dQkv(workspace, workspaceSize, executor,
                                    static_cast<aclrtStream>(ctx->GetStream()));
      if (workspaceSize > 0) {
        ctx->FreeWorkspace();
      }
    }
  }
  destroyAll();
  if (status != 0) {
    std::fprintf(stderr, "%s: aclnn status %d\n", kTag,
                 static_cast<int>(status));
  }
  return status == 0 ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

UINT32 CausalConv1dQkvInferShape(gert::InferShapeContext *ctx) {
  const gert::Shape *xShape = ctx->GetInputShape(kIdxX);
  if (xShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  *ctx->GetOutputShape(0) = *xShape;
  return ge::GRAPH_SUCCESS;
}

UINT32 CausalConv1dQkvInferDataType(gert::InferDataTypeContext *ctx) {
  const gert::RuntimeAttrs *attrs = ctx->GetAttrs();
  if (attrs == nullptr) {
    std::fprintf(stderr, "%s: fail infer_dtype_attrs_null\n", kTag);
    return ge::GRAPH_FAILED;
  }
  const ge::DataType outDtype =
      ToGeDataType(AttrInt(attrs, kAttrOutDtype, kOutDtypeBf16));
  if (outDtype == ge::DT_UNDEFINED) {
    std::fprintf(stderr, "%s: fail infer_dtype attr\n", kTag);
    return ge::GRAPH_FAILED;
  }
  if (ctx->SetOutputDataType(0, outDtype) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// HostInputs({3,4,5,6}) = the four ValueDepend metadata slots; the aclnn
// lowering then hands them over as host tensors (kOnHost) instead of device
// ones.
IMPL_OP(CausalConv1dQkvGE)
    .OpExecuteFunc(&CausalConv1dQkvExecute)
    .HostInputs({3, 4, 5, 6})
    .InferShape(&CausalConv1dQkvInferShape)
    .InferDataType(&CausalConv1dQkvInferDataType);
