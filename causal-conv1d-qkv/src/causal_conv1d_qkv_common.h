// Shared internals of the CausalConv1dQkv GE wrapper and eager flat entry (both
// compiled into libcust_opapi.so): the host-side mirror of the upstream
// packed-QKV tiling gates and the host-value plumbing for the four ValueDepend
// metadata inputs. Project addition, not vendored operator code.
#ifndef CAUSAL_CONV1D_QKV_COMMON_H_
#define CAUSAL_CONV1D_QKV_COMMON_H_

#include <cstdint>
#include <cstdio>
#include <vector>

#include "aclnn/acl_meta.h"
#include "acl/acl_rt.h"

constexpr int kNumMetaInputs = 4;  // query_start_loc .. num_accepted_tokens
constexpr int64_t kPackedQkvActivationMode = 2;
constexpr int64_t kPackedHeadDim = 128;
constexpr int64_t kPackedWidth = 4;
constexpr int64_t kDimAlignElems = 16;

// Host-side mirror of the upstream packed-QKV tiling gates. It exists to turn a
// bare aclnn 561002 into a named-values diagnostic; the kernel stays the
// authority on every one of these conditions. Non-packed activation modes keep
// the upstream per-mode semantics and are forwarded as-is.
inline bool CheckPackedLayout(const char *tag, int64_t activationMode,
                              int64_t runMode, bool hasBias, int64_t xDimNum,
                              int64_t dim, int64_t width, int64_t qDim,
                              int64_t kDim, int64_t vDim, int64_t headDim) {
  if (activationMode != kPackedQkvActivationMode) {
    return true;
  }
  // 2D varlen x is the only cadence the packed kernel tiles.
  const bool ok = runMode == 0 && !hasBias && xDimNum == 2 &&
                  width == kPackedWidth && qDim > 0 && qDim == kDim &&
                  headDim == kPackedHeadDim && qDim % headDim == 0 &&
                  kDim % headDim == 0 && vDim > 0 && vDim % headDim == 0 &&
                  qDim + kDim + vDim == dim && dim % kDimAlignElems == 0;
  if (!ok) {
    std::fprintf(stderr,
                 "%s: packed qkv layout invalid (runMode=%lld hasBias=%d "
                 "xDimNum=%lld width=%lld q/k/v=%lld/%lld/%lld head=%lld dim=%lld)\n",
                 tag, static_cast<long long>(runMode), hasBias ? 1 : 0,
                 static_cast<long long>(xDimNum), static_cast<long long>(width),
                 static_cast<long long>(qDim), static_cast<long long>(kDim),
                 static_cast<long long>(vDim), static_cast<long long>(headDim),
                 static_cast<long long>(dim));
  }
  return ok;
}

// True when addr must be copied out before host tiling can read it. Unknown
// placement counts as device: a redundant D2H is recoverable, a host read of
// device memory is not (the aclnn *Tensor metadata variant does exactly that
// and segfaults on device metadata).
inline bool IsDevicePtr(const void *addr) {
  aclrtPtrAttributes attributes{};
  return aclrtPointerGetAttributes(addr, &attributes) != ACL_SUCCESS ||
         attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE ||
         attributes.location.type == ACL_MEM_LOCATION_TYPE_MANAGED;
}

// Pinned host staging for device-resident metadata: aclrtMemcpyAsync into
// pageable memory bounces through a driver-side copy, and aclrtMallocHost is
// itself a driver round trip, so one buffer is kept per thread and grown on
// demand. Deliberately never freed: a thread_local destructor would run on
// whatever device happens to be current at thread teardown, where aclrtFreeHost
// is at best an error (precedent: mega_gdn_prefill_eager.cc's leak-only scratch).
class MetaStaging {
 public:
  int64_t *Get(size_t count) {
    if (pinned_ != nullptr && count <= capacity_) {
      return pinned_;
    }
    void *pinned = nullptr;
    if (aclrtMallocHost(&pinned, count * sizeof(int64_t)) == ACL_SUCCESS &&
        pinned != nullptr) {
      pinned_ = static_cast<int64_t *>(pinned);
      capacity_ = count;
      return pinned_;
    }
    pageable_.resize(count);
    return pageable_.data();
  }

 private:
  int64_t *pinned_ = nullptr;
  size_t capacity_ = 0;
  std::vector<int64_t> pageable_;
};

inline MetaStaging &Staging() {
  static thread_local MetaStaging staging;
  return staging;
}

// Owns the aclIntArrays built from the metadata slots; destroyed on scope exit
// so every early-return path in the callers stays clean.
struct MetaIntArrays {
  aclIntArray *array[kNumMetaInputs] = {nullptr, nullptr, nullptr, nullptr};
  bool deviceStaged = false;  // at least one slot had to be copied D2H

  MetaIntArrays() = default;
  MetaIntArrays(const MetaIntArrays &) = delete;
  MetaIntArrays &operator=(const MetaIntArrays &) = delete;
  ~MetaIntArrays() {
    for (int k = 0; k < kNumMetaInputs; ++k) {
      if (array[k] != nullptr) {
        aclDestroyIntArray(array[k]);
      }
    }
  }
};

// Turn the four ValueDepend metadata inputs into host aclIntArrays. The order is
// load-bearing: classify placement (isHost comes from the caller, which is the
// only part that differs between the GE and eager legs), issue every D2H copy
// into one pinned staging buffer, drain the stream ONCE, and only then
// aclCreateIntArray — the array snapshots the values it is handed.
// addrs[k] == nullptr marks an absent slot; counts[k] is its element count.
inline bool BuildMetaIntArrays(const char *tag, const void *const *addrs,
                               const int64_t *counts, const bool *isHost,
                               aclrtStream stream, MetaIntArrays *out) {
  size_t stagedElems = 0;
  for (int k = 0; k < kNumMetaInputs; ++k) {
    if (addrs[k] != nullptr && !isHost[k]) {
      stagedElems += static_cast<size_t>(counts[k]);
    }
  }
  out->deviceStaged = stagedElems > 0;
  int64_t *staged = stagedElems > 0 ? Staging().Get(stagedElems) : nullptr;
  const int64_t *values[kNumMetaInputs] = {nullptr, nullptr, nullptr, nullptr};
  size_t offset = 0;
  for (int k = 0; k < kNumMetaInputs; ++k) {
    if (addrs[k] == nullptr) {
      continue;
    }
    values[k] = static_cast<const int64_t *>(addrs[k]);
    if (isHost[k]) {
      continue;
    }
    const size_t bytes = static_cast<size_t>(counts[k]) * sizeof(int64_t);
    if (aclrtMemcpyAsync(staged + offset, bytes, addrs[k], bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST, stream) != ACL_SUCCESS) {
      std::fprintf(stderr, "%s: fail metadata copy %d\n", tag, k);
      return false;
    }
    values[k] = staged + offset;
    offset += static_cast<size_t>(counts[k]);
  }
  if (stagedElems > 0 && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    std::fprintf(stderr, "%s: fail metadata drain\n", tag);
    return false;
  }
  for (int k = 0; k < kNumMetaInputs; ++k) {
    if (addrs[k] == nullptr) {
      continue;
    }
    out->array[k] =
        aclCreateIntArray(values[k], static_cast<uint64_t>(counts[k]));
    if (out->array[k] == nullptr) {
      std::fprintf(stderr, "%s: fail metadata int_array %d\n", tag, k);
      return false;
    }
  }
  return true;
}

#endif  // CAUSAL_CONV1D_QKV_COMMON_H_
