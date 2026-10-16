// Eager entry for the torch bridge (prefix_attn_torch/prefix_attn_bridge.cpp):
// [memset -> AICPU planner -> AIC kernel] on the caller's stream, same
// sequence as the GE path.  One thread = one scratch buffer; reuse is
// stream-ordered, so a thread alternating streams must sync in between.
#include <cstddef>
#include <cstdint>

#include "acl/acl_rt.h"

#include "artifacts.h"
#include "kernel_launcher.h"

namespace {

// Shape specialization baked into the artifacts; fp16-only (ge/README.md).
constexpr int64_t kN1 = PREFIX_ATTN_N1;
constexpr int64_t kN2 = PREFIX_ATTN_N2;

}  // namespace

extern "C" int prefix_attn_run_v2(void *out, const void *query, const void *key,
                                  const void *value, const void *bounds,
                                  long long total_tokens, long long n1,
                                  long long n2, long long bounds_len,
                                  float scale, void *stream) {
  if (out == nullptr || query == nullptr || key == nullptr || value == nullptr ||
      bounds == nullptr || stream == nullptr || bounds_len < 2 ||
      n1 != kN1 || n2 != kN2) {
    return 10;
  }
  auto *s = static_cast<aclrtStream>(stream);

  // per-thread scratch, reallocated on device change; the destructor's
  // aclrtFree must not run after aclFinalize (torch owns teardown)
  struct Scratch {
    void *dev = nullptr;
    int device = -1;
    ~Scratch() {
      if (dev != nullptr) {
        aclrtFree(dev);
      }
    }
  };
  static thread_local Scratch scratch;
  const size_t meta_bytes =
      sizeof(int32_t) * static_cast<size_t>(prefix_attn::kMetadataElems);
  int cur_device = -1;
  if (aclrtGetDevice(&cur_device) != ACL_SUCCESS || cur_device < 0) {
    return 12;
  }
  if (scratch.dev != nullptr && scratch.device != cur_device) {
    aclrtFree(scratch.dev);
    scratch.dev = nullptr;
  }
  if (scratch.dev == nullptr) {
    if (aclrtMalloc(&scratch.dev, meta_bytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS) {
      return 12;
    }
    scratch.device = cur_device;
  }

  // zero-init: the planner leaves unused core records untouched
  if (aclrtMemsetAsync(scratch.dev, meta_bytes, 0, meta_bytes, s) !=
      ACL_SUCCESS) {
    return 13;
  }
  if (prefix_attn::LaunchAicpuPlanner(s, bounds, scratch.dev,
                                      static_cast<uint32_t>(bounds_len), kN1,
                                      kN2, prefix_attn::kAicCores) != 0) {
    return 14;
  }
  return prefix_attn::LaunchAttnKernel(s, out, query, key, value, bounds,
                                       scratch.dev, total_tokens, bounds_len,
                                       scale) == 0
             ? 0
             : 17;
}
