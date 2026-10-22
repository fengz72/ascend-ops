// PrefixAttention custom op: q/k/v/bounds -> out (fp16).
//
// The whole GE surface in one file: REG_OP prototype, IMPL_OP shape/dtype
// inference, and the Execute class. The memset and AICPU planner run on a
// GE-managed auxiliary stream. Preloaded bounds allow it to skip the
// main-to-aux wait; the AIC kernel stays on the graph stream. CANN >= 9.3.0.
//
// The planner reads live bounds on every run, with no host round trip.
// bounds_preloaded requires an int32 graph input ready before execution. Each model
// instance must complete its preceding execution before another invocation.
//
// With dynamic input descriptors, T, batch and bounds contents may change
// between executions; N1/N2/D and dtype remain specialized (see artifacts.h).
// Growing outputs requires GE's MallocOutputTensor capacity-check/reallocation
// fix and matching GE compiler/executor components (see README.md).
//
// Device launchers: kernel_launcher.cc.  Eager entry: prefix_attn_eager.cc.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include "acl/acl_rt.h"
#include "graph/custom_op.h"
#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

#include "kernel_launcher.h"
#include "artifacts.h"

namespace {

constexpr const char *kPlannerStreamKey = "prefix_attention.aicpu";
struct PlannerResources {
  aclrtEvent input_ready = nullptr;
  aclrtEvent metadata_ready = nullptr;
  void *metadata = nullptr;
};

using ResourceKey = std::tuple<uintptr_t, uintptr_t, std::string>;
std::mutex resources_mutex;
// No ACL calls from a static destructor: ACL may already be finalized.
std::map<ResourceKey, PlannerResources> planner_resources;
constexpr size_t kMetadataBytes =
    sizeof(int32_t) * static_cast<size_t>(prefix_attn::kMetadataElems);

aclError CurrentContext(aclrtContext &context) {
  const auto ret = aclrtGetCurrentContext(&context);
  return ret == ACL_SUCCESS && context == nullptr ? ACL_ERROR_INVALID_PARAM : ret;
}

aclError ReleasePlannerResources(PlannerResources &resources) {
  int first_error = ACL_SUCCESS;
  for (auto *event : {&resources.input_ready, &resources.metadata_ready}) {
    if (*event != nullptr) {
      const auto ret = aclrtDestroyEvent(*event);
      if (ret == ACL_SUCCESS) {
        *event = nullptr;
      } else if (first_error == ACL_SUCCESS) {
        first_error = ret;
      }
    }
  }
  if (resources.metadata != nullptr) {
    const auto ret = aclrtFree(resources.metadata);
    if (ret == ACL_SUCCESS) {
      resources.metadata = nullptr;
    } else if (first_error == ACL_SUCCESS) {
      first_error = ret;
    }
  }
  return first_error;
}

bool HasPlannerResources(const PlannerResources &resources) {
  return resources.input_ready != nullptr || resources.metadata_ready != nullptr ||
         resources.metadata != nullptr;
}

aclError GetPlannerResources(aclrtStream auxiliary, const char *node,
                            bool needs_input_event, PlannerResources &resources) {
  if (auxiliary == nullptr || node == nullptr || node[0] == '\0') {
    return ACL_ERROR_INVALID_PARAM;
  }
  aclrtContext context = nullptr;
  auto ret = CurrentContext(context);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  const ResourceKey key{reinterpret_cast<uintptr_t>(context),
                        reinterpret_cast<uintptr_t>(auxiliary), node};
  const std::lock_guard<std::mutex> lock(resources_mutex);
  const auto found = planner_resources.find(key);
  if (found != planner_resources.end()) {
    if (found->second.metadata_ready == nullptr || found->second.metadata == nullptr ||
        (needs_input_event && found->second.input_ready == nullptr)) {
      return ACL_ERROR_INVALID_PARAM;
    }
    resources = found->second;
    return ACL_SUCCESS;
  }
  constexpr uint32_t flags = ACL_EVENT_SYNC |
                            ACL_EVENT_CAPTURE_STREAM_PROGRESS | ACL_EVENT_TIME_LINE;
  PlannerResources created;
  if (needs_input_event) {
    ret = aclrtCreateEventWithFlag(&created.input_ready, flags);
  }
  if (ret == ACL_SUCCESS) {
    ret = aclrtCreateEventWithFlag(&created.metadata_ready, flags);
  }
  if (ret == ACL_SUCCESS) {
    // RT2 recycles workspaces on the main stream. An early auxiliary writer
    // needs a private buffer, rather than memory still used by earlier tasks.
    ret = aclrtMalloc(&created.metadata, kMetadataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
  }
  if (ret != ACL_SUCCESS) {
    (void)ReleasePlannerResources(created);
    if (HasPlannerResources(created)) {
      planner_resources.emplace(key, created); // Explicit cleanup can retry.
    }
    return ret;
  }
  planner_resources.emplace(key, created);
  resources = created;
  return ACL_SUCCESS;
}

constexpr size_t kInputQuery = 0U;
constexpr size_t kInputKey = 1U;
constexpr size_t kInputValue = 2U;
constexpr size_t kInputBorder = 3U;
constexpr size_t kOutputOut = 0U;

// Shape specialization baked into the artifacts; fp16-only (ge/README.md).
constexpr int64_t kN1 = PREFIX_ATTN_N1;
constexpr int64_t kN2 = PREFIX_ATTN_N2;
constexpr int64_t kHeadDim = 64;

ge::graphStatus InferShapeForPrefixAttention(gert::InferShapeContext *ctx) {
  const auto *input_shape = ctx->GetInputShape(0U);
  auto *output_shape = ctx->GetOutputShape(0U);
  if ((input_shape == nullptr) || (output_shape == nullptr)) {
    return ge::GRAPH_FAILED;
  }
  *output_shape = *input_shape;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForPrefixAttention(
    gert::InferDataTypeContext *ctx) {
  return ctx->SetOutputDataType(0U, ctx->GetInputDataType(0U));
}

}  // namespace

// Historical ABI name: releases both events and private metadata buffers.
// The caller must finish and unload all models before invoking this hook.
extern "C" int prefix_attn_release_attached_events() {
  aclrtContext context = nullptr;
  auto ret = CurrentContext(context);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  const std::lock_guard<std::mutex> lock(resources_mutex);
  int first_error = ACL_SUCCESS;
  for (auto it = planner_resources.begin(); it != planner_resources.end();) {
    if (std::get<0>(it->first) != reinterpret_cast<uintptr_t>(context)) {
      ++it;
      continue;
    }
    ret = ReleasePlannerResources(it->second);
    if (first_error == ACL_SUCCESS && ret != ACL_SUCCESS) {
      first_error = ret;
    }
    if (!HasPlannerResources(it->second)) {
      it = planner_resources.erase(it);
    } else {
      ++it;
    }
  }
  return first_error;
}

namespace ge {

// Op prototype (fp16-only, ge/README.md).  The type name must match the
// class and any frontend converter.
REG_OP(PrefixAttention)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(actual_seq_lengths, TensorType({DT_INT32}))
    .ATTR(scale, Float, 0.0)
    .ATTR(bounds_preloaded, Bool, false)
    .OUTPUT(attention_out, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(PrefixAttention);

// Graph-compile-time shape/dtype inference.
IMPL_OP(PrefixAttention)
    .InferShape(InferShapeForPrefixAttention)
    .InferDataType(InferDataTypeForPrefixAttention);

// Execute only: shape inference is the IMPL_OP above; eager calls bypass
// this class (they go via prefix_attn_eager.cc).
class PrefixAttention final : public EagerExecuteOp, public ResourceUsageReporter {
 public:
  graphStatus DeclareResourceUsage(gert::ResourceUsageContext &ctx) override {
    return ctx.ReportAttachedStream({AscendString(kPlannerStreamKey)});
  }
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    const gert::Tensor *query = ctx->GetInputTensor(kInputQuery);
    const gert::Tensor *key = ctx->GetInputTensor(kInputKey);
    const gert::Tensor *value = ctx->GetInputTensor(kInputValue);
    const gert::Tensor *bounds = ctx->GetInputTensor(kInputBorder);
    if (query == nullptr || key == nullptr || value == nullptr ||
        bounds == nullptr) {
      std::cerr << "[PrefixAttention] null input tensor" << std::endl;
      return GRAPH_FAILED;
    }

    const gert::Shape &q_shape = query->GetShape().GetStorageShape();
    const gert::Shape &k_shape = key->GetShape().GetStorageShape();
    const gert::Shape &b_shape = bounds->GetShape().GetStorageShape();
    const int64_t total = q_shape.GetDimNum() >= 1 ? q_shape.GetDim(0) : 0;
    const int64_t n1 = q_shape.GetDimNum() >= 2 ? q_shape.GetDim(1) : 0;
    const int64_t head_dim = q_shape.GetDimNum() >= 3 ? q_shape.GetDim(2) : 0;
    const int64_t n2 = k_shape.GetDimNum() >= 2 ? k_shape.GetDim(1) : 0;
    if (q_shape.GetDimNum() != 3 || k_shape.GetDimNum() != 3 ||
        head_dim != kHeadDim || n1 != kN1 || n2 != kN2 ||
        k_shape.GetDim(0) != total || key->GetDataType() != query->GetDataType() ||
        value->GetDataType() != query->GetDataType() ||
        query->GetDataType() != DT_FLOAT16 ||
        bounds->GetDataType() != DT_INT32 || b_shape.GetDimNum() != 1 ||
        b_shape.GetDim(0) < 2) {
      std::cerr << "[PrefixAttention] shape/dtype contract violated "
                    "(fp16-only specialization)" << std::endl;
      return GRAPH_FAILED;
    }
    const int64_t bounds_len = b_shape.GetDim(0);

    gert::Tensor *output = ctx->MallocOutputTensor(
        kOutputOut, query->GetShape(), query->GetFormat(), query->GetDataType());
    if (output == nullptr) {
      std::cerr << "[PrefixAttention] MallocOutputTensor failed" << std::endl;
      return GRAPH_FAILED;
    }

    float scale = 0.0F;
    bool bounds_preloaded = false;
    const gert::RuntimeAttrs *attrs = ctx->GetAttrs();
    if (attrs != nullptr) {
      const float *scale_attr = attrs->GetFloat(0);
      if (scale_attr != nullptr) {
        scale = *scale_attr;
      }
      const bool *preloaded_attr = attrs->GetBool(1U);
      if (preloaded_attr != nullptr) {
        bounds_preloaded = *preloaded_attr;
      }
    }
    if (!(scale > 0.0F)) {
      scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    }

    aclrtStream stream = ctx->GetStream();
    const aclrtStream planner_stream =
        ctx->RequestAttachedStream(kPlannerStreamKey);
    PlannerResources resources;
    if (planner_stream == nullptr || planner_stream == stream ||
        GetPlannerResources(planner_stream, ctx->GetNodeName(), !bounds_preloaded, resources) != ACL_SUCCESS) {
      std::cerr << "[PrefixAttention] attached stream/resource allocation failed" << std::endl;
      return GRAPH_FAILED;
    }

    // Static GE models may copy Data inputs inside the model. Only a caller
    // that established a direct, preloaded input contract may skip this wait.
    if (!bounds_preloaded &&
        (aclrtRecordEvent(resources.input_ready, stream) != ACL_SUCCESS ||
         aclrtStreamWaitEvent(planner_stream, resources.input_ready) != ACL_SUCCESS ||
         aclrtResetEvent(resources.input_ready, planner_stream) != ACL_SUCCESS)) {
      return GRAPH_FAILED;
    }
    // One model instance must finish its preceding invocation before replay.
    void *metadata_dev = resources.metadata;
    constexpr size_t meta_bytes = kMetadataBytes;
    if (aclrtMemsetAsync(metadata_dev, meta_bytes, 0, meta_bytes, planner_stream) !=
        ACL_SUCCESS) {
      std::cerr << "[PrefixAttention] metadata memset failed" << std::endl;
      return GRAPH_FAILED;
    }
    if (prefix_attn::LaunchAicpuPlanner(planner_stream, bounds->GetAddr(), metadata_dev,
                                        static_cast<uint32_t>(bounds_len), kN1,
                                        kN2, prefix_attn::kAicCores) != 0) {
      return GRAPH_FAILED;
    }
    if (aclrtRecordEvent(resources.metadata_ready, planner_stream) != ACL_SUCCESS ||
        aclrtStreamWaitEvent(stream, resources.metadata_ready) != ACL_SUCCESS ||
        aclrtResetEvent(resources.metadata_ready, stream) != ACL_SUCCESS) {
      std::cerr << "[PrefixAttention] planner-to-main synchronization failed" << std::endl;
      return GRAPH_FAILED;
    }
    if (prefix_attn::LaunchAttnKernel(stream, output->GetAddr(),
                                      query->GetAddr(), key->GetAddr(),
                                      value->GetAddr(), bounds->GetAddr(),
                                      metadata_dev, total, bounds_len,
                                      scale) != 0) {
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
};

REG_AUTO_MAPPING_OP(PrefixAttention);
}  // namespace ge
