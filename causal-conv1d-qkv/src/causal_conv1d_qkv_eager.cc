// Eager flat entry for the JIT torch bridge: torch tensors flattened to raw
// pointer/shape arrays -> aclnnCausalConv1dQkv, the same sequence as the GE
// Execute path. Output is a fresh device buffer allocated by the caller side.
// No torch headers on this side.
//
// addrs/dims/strides/dtypes/dim_nums describe the 8 aclnn inputs IN ORDER:
//   0 x, 1 weight, 2 bias (nullable), 3 conv_states,
//   4 query_start_loc (nullable), 5 cache_indices (nullable),
//   6 initial_state_mode (nullable), 7 num_accepted_tokens (nullable)
// dims/strides use 4 int64 slots per tensor (tensor i lives at dims + 4*i).
// dtypes are ACL_* enum values. out* describe y, a fresh device buffer
// allocated by the caller. Returns 0 on success.
//
// Inputs 4..7 are tiling metadata: aclIntArray is a host-side tiling input read
// inside GetWorkspaceSize (the upstream torch call site passes at::IntArrayRef,
// i.e. the same host path). Host storage is used in place; device tensors are
// drained with ONE batched pinned copy + ONE stream drain, so the bridge may
// pass either. A drain is a driver round trip (~30us here), hence never one per
// tensor.
#include <cstdint>
#include <cstdio>

#include "causal_conv1d_qkv_aclnn.h"
#include "causal_conv1d_qkv_common.h"

namespace {

constexpr char kTag[] = "CausalConv1dQkv eager";

constexpr int kNumTensorInputs = 4;  // x, weight, bias(optional), conv_states
constexpr int kIdxX = 0;
constexpr int kIdxWeight = 1;
constexpr int kIdxBias = 2;
constexpr int kIdxConvStates = 3;

aclTensor *MakeAclTensor(const int64_t *dims, const int64_t *strides,
                         int32_t dtype, int32_t dimNum, void *addr) {
  if (dimNum < 1 || dimNum > 4) {
    return nullptr;
  }
  return aclCreateTensor(dims, static_cast<uint64_t>(dimNum),
                         static_cast<aclDataType>(dtype), strides, 0,
                         ACL_FORMAT_ND, dims, static_cast<uint64_t>(dimNum),
                         addr);
}

int64_t NumElements(const int64_t *dims, int32_t dimNum) {
  if (dimNum < 1) {
    return 0;
  }
  int64_t count = 1;
  for (int32_t d = 0; d < dimNum; ++d) {
    if (dims[d] <= 0) {
      return 0;
    }
    count *= dims[d];
  }
  return count;
}

// aclrtFree is not stream-ordered, so the naive workspace route drains the
// stream after every call. The workspace is opaque kernel scratch of a few tens
// of KB: one grow-only buffer per thread removes that drain. It is only reusable
// on the stream that produced it (a consumer on the old stream, e.g. under
// torch.npu.stream(s), may still be reading it) and only on the device it was
// allocated on, so both are part of the key. aclrtGetDevice is a ~0.07us driver
// query, cheap enough to keep in the key path. Deliberately never freed: a
// thread_local destructor would run on whatever device is current at teardown
// (precedent: mega_gdn_prefill_eager.cc's leak-only scratch).
class WorkspaceCache {
 public:
  void *Get(uint64_t size, aclrtStream stream) {
    if (size == 0) {
      return nullptr;
    }
    int32_t device = -1;
    if (aclrtGetDevice(&device) != ACL_SUCCESS) {
      return nullptr;
    }
    if (stream != stream_) {
      Release(stream_);
      stream_ = stream;
    }
    if (ptr_ != nullptr && size <= capacity_ && device == device_) {
      return ptr_;
    }
    Release(stream);
    void *ptr = nullptr;
    if (aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      return nullptr;
    }
    ptr_ = ptr;
    capacity_ = size;
    device_ = device;
    return ptr_;
  }

 private:
  // aclrtFree is not stream-ordered: drain the stream that produced the buffer.
  void Release(aclrtStream stream) {
    if (ptr_ == nullptr) {
      return;
    }
    (void)aclrtSynchronizeStream(stream);
    (void)aclrtFree(ptr_);
    ptr_ = nullptr;
    capacity_ = 0;
  }

  void *ptr_ = nullptr;
  uint64_t capacity_ = 0;
  aclrtStream stream_ = nullptr;
  int32_t device_ = -1;
};

WorkspaceCache &CachedWorkspace() {
  static thread_local WorkspaceCache cache;
  return cache;
}

}  // namespace

// hidden by default (-fvisibility=hidden); the JIT bridge dlsyms this name.
extern "C" __attribute__((visibility("default"))) int causal_conv1d_qkv_run(
    void *const *addrs, const int64_t *dims, const int64_t *strides,
    const int32_t *dtypes, const int32_t *dim_nums, int64_t activation_mode,
    int64_t pad_slot_id, int64_t run_mode, int64_t q_dim, int64_t k_dim,
    int64_t v_dim, int64_t head_dim, void *out_addr, const int64_t *out_dims,
    const int64_t *out_strides, int32_t out_dtype, int32_t out_dim_num,
    void *stream) {
  if (addrs == nullptr || dims == nullptr || strides == nullptr ||
      dtypes == nullptr || dim_nums == nullptr || out_addr == nullptr ||
      out_dims == nullptr || out_strides == nullptr || stream == nullptr) {
    return 2;
  }
  if (out_dtype != ACL_FLOAT16 && out_dtype != ACL_BF16) {
    std::fprintf(stderr, "%s: bad out dtype %d\n", kTag, out_dtype);
    return 2;
  }
  aclrtStream aclStream = static_cast<aclrtStream>(stream);

  aclTensor *aclInputs[kNumTensorInputs] = {nullptr, nullptr, nullptr, nullptr};
  aclTensor *out = MakeAclTensor(out_dims, out_strides, out_dtype, out_dim_num,
                                 out_addr);
  auto destroyAll = [&aclInputs, &out]() {
    for (int i = 0; i < kNumTensorInputs; ++i) {
      if (aclInputs[i] != nullptr) {
        aclDestroyTensor(aclInputs[i]);
      }
    }
    if (out != nullptr) {
      aclDestroyTensor(out);
    }
  };

  for (int i = 0; i < kNumTensorInputs; ++i) {
    if (addrs[i] == nullptr) {
      continue;  // addrs[kIdxBias] may be null: no bias
    }
    aclInputs[i] = MakeAclTensor(dims + 4 * i, strides + 4 * i, dtypes[i],
                                 dim_nums[i], addrs[i]);
    // A present input whose wrap failed must not be treated as "absent": a
    // dropped conv-state history or bias would silently change results.
    if (aclInputs[i] == nullptr) {
      destroyAll();
      std::fprintf(stderr, "%s: fail make_acl input %d\n", kTag, i);
      return 3;
    }
  }
  const bool hasBias = aclInputs[kIdxBias] != nullptr;
  if (aclInputs[kIdxX] == nullptr || aclInputs[kIdxWeight] == nullptr ||
      aclInputs[kIdxConvStates] == nullptr || out == nullptr) {
    destroyAll();
    std::fprintf(stderr, "%s: fail required_tensor\n", kTag);
    return 3;
  }

  // Packed QKV is BF16-in only, needs width-4 weights and a 2D varlen x; x's
  // last dim carries dim, weight's first dim carries the tap count.
  const int32_t xDimNum = dim_nums[kIdxX];
  const int64_t dim = xDimNum >= 1 ? dims[xDimNum - 1] : 0;
  const int64_t width = dims[4 * kIdxWeight] >= 1 ? dims[4 * kIdxWeight] : 0;
  if (activation_mode == kPackedQkvActivationMode && dtypes[kIdxX] != ACL_BF16) {
    destroyAll();
    std::fprintf(stderr, "%s: packed qkv needs bf16 x\n", kTag);
    return 3;
  }
  if (!CheckPackedLayout(kTag, activation_mode, run_mode, hasBias, xDimNum, dim,
                         width, q_dim, k_dim, v_dim, head_dim)) {
    destroyAll();
    return 3;
  }

  // Metadata classification. The shape/dtype checks below stay: this extern-C
  // symbol is a contract boundary reachable from non-torch callers.
  const void *metaAddrs[kNumMetaInputs] = {nullptr, nullptr, nullptr, nullptr};
  int64_t metaCounts[kNumMetaInputs] = {0, 0, 0, 0};
  bool metaHost[kNumMetaInputs] = {false, false, false, false};
  for (int k = 0; k < kNumMetaInputs; ++k) {
    const int i = kNumTensorInputs + k;
    if (addrs[i] == nullptr) {
      continue;
    }
    const int64_t size = NumElements(dims + 4 * i, dim_nums[i]);
    if (dim_nums[i] != 1 || size <= 0 || dtypes[i] != ACL_INT64) {
      destroyAll();
      std::fprintf(stderr, "%s: fail metadata %d (must be 1-D int64)\n", kTag,
                   i);
      return 4;
    }
    metaAddrs[k] = addrs[i];
    metaCounts[k] = size;
    metaHost[k] = !IsDevicePtr(addrs[i]);
  }
  MetaIntArrays metaArrays;
  if (!BuildMetaIntArrays(kTag, metaAddrs, metaCounts, metaHost, aclStream,
                          &metaArrays)) {
    destroyAll();
    return 4;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus status = aclnnCausalConv1dQkvGetWorkspaceSize(
      aclInputs[kIdxX], aclInputs[kIdxWeight],
      hasBias ? aclInputs[kIdxBias] : nullptr, aclInputs[kIdxConvStates],
      metaArrays.array[0], metaArrays.array[1], metaArrays.array[2],
      metaArrays.array[3], activation_mode, pad_slot_id, run_mode, q_dim, k_dim,
      v_dim, head_dim, out, &workspaceSize, &executor);
  if (status == 0) {
    void *workspace =
        workspaceSize > 0 ? CachedWorkspace().Get(workspaceSize, aclStream)
                          : nullptr;
    if (workspaceSize > 0 && workspace == nullptr) {
      status = 5;
    } else {
      status = aclnnCausalConv1dQkv(workspace, workspaceSize, executor,
                                    aclStream);
    }
  }
  destroyAll();
  if (status != 0) {
    std::fprintf(stderr, "%s: aclnn status %d\n", kTag,
                 static_cast<int>(status));
  }
  return status == 0 ? 0 : static_cast<int>(status);
}
