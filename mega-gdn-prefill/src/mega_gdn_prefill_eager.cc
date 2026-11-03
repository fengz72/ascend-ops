// Eager flat entry for the JIT torch bridge (python/mega_gdn_prefill_bridge.cpp):
// torch tensors flattened to raw pointer/shape arrays -> aclnnMegaGdnPrefillOp,
// the same sequence as the GE Execute path. z (index 3) doubles as the norm
// output — the GE wrapper's MakeOutputRefInput in-place contract. The
// conv_state/ssm_cache outputs are discarded into thread-local scratch (the
// caller consumes only z). No torch headers on this side.
#include <cstdio>
#include <cstdint>

#include "mega_gdn_prefill_common.h"

namespace {

constexpr int kNumInputs = 18;
constexpr int kIdxZ = 3;
constexpr int kIdxConvState = 5;
constexpr int kIdxSsmCache = 12;

size_t ElementBytes(int32_t dtype) {
  switch (dtype) {
    case ACL_BF16:
    case ACL_FLOAT16:
      return 2;
    case ACL_FLOAT:
    case ACL_INT32:
      return 4;
    default:
      return 0;
  }
}

size_t ShapeBytes(const int64_t *dims, int32_t dimNum, int32_t dtype) {
  const size_t elem = ElementBytes(dtype);
  if (elem == 0 || dimNum < 1 || dimNum > 4) {
    return 0;
  }
  int64_t count = 1;
  for (int32_t i = 0; i < dimNum; ++i) {
    if (dims[i] < 0) {
      return 0;
    }
    count *= dims[i];
  }
  return static_cast<size_t>(count) * elem;
}

aclTensor *MakeAclTensor(const int64_t *dims, const int64_t *strides,
                         int32_t dtype, int32_t dimNum, void *addr) {
  if (dimNum < 1 || dimNum > 4) {
    return nullptr;
  }
  return aclCreateTensor(dims, static_cast<uint64_t>(dimNum),
                         static_cast<aclDataType>(dtype), strides, 0,
                         ACL_FORMAT_ND, dims, static_cast<uint64_t>(dimNum), addr);
}

// Discarded-output scratch, leak-only: never freed — aclrtFree is not
// stream-ordered (an in-flight kernel may still be writing the discarded
// conv/ssm outputs) and cross-device frees are undefined. At most one stale
// block per thread per device/shape-growth, reclaimed at process exit.
struct Scratch {
  void *dev;
  uint64_t bytes;
  int32_t device;
};
thread_local Scratch g_scratch{nullptr, 0, -1};

}  // namespace

extern "C" int mega_gdn_prefill_run_v2(void *const *addrs, const int64_t *dims,
                                       const int64_t *strides,
                                       const int32_t *dtypes,
                                       const int32_t *dimNums,
                                       int64_t numMatrices, void *stream) {
  if (addrs == nullptr || dims == nullptr || strides == nullptr ||
      dtypes == nullptr || dimNums == nullptr || stream == nullptr ||
      numMatrices <= 0) {
    return 2;
  }
  for (int i = 0; i < kNumInputs; ++i) {
    if (addrs[i] == nullptr) {
      return 2;
    }
  }

  const size_t convBytes = ShapeBytes(
      dims + kIdxConvState * 4, dimNums[kIdxConvState], dtypes[kIdxConvState]);
  const size_t ssmBytes = ShapeBytes(
      dims + kIdxSsmCache * 4, dimNums[kIdxSsmCache], dtypes[kIdxSsmCache]);
  if (convBytes == 0 || ssmBytes == 0) {
    return 3;
  }
  const size_t ssmOffset = (convBytes + 63) / 64 * 64;
  const uint64_t scratchBytes = static_cast<uint64_t>(ssmOffset) + ssmBytes;
  int32_t device = -1;
  if (aclrtGetDevice(&device) != ACL_SUCCESS) {
    return 4;
  }
  if (g_scratch.dev == nullptr || g_scratch.device != device ||
      g_scratch.bytes < scratchBytes) {
    void *dev = nullptr;
    if (aclrtMalloc(&dev, scratchBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      return 4;
    }
    g_scratch.dev = dev;
    g_scratch.bytes = scratchBytes;
    g_scratch.device = device;
  }

  aclTensor *aclInputs[kNumInputs];
  for (int i = 0; i < kNumInputs; ++i) {
    aclInputs[i] = MakeAclTensor(dims + i * 4, strides + i * 4, dtypes[i],
                                 dimNums[i], addrs[i]);
    if (aclInputs[i] == nullptr) {
      for (int j = 0; j < i; ++j) {
        aclDestroyTensor(aclInputs[j]);
      }
      return 5;
    }
  }
  // norm output aliases z in place (GE MakeOutputRefInput contract);
  // conv/ssm outputs are discarded into scratch
  aclTensor *aclNormOutput = MakeAclTensor(
      dims + kIdxZ * 4, strides + kIdxZ * 4, dtypes[kIdxZ], dimNums[kIdxZ],
      addrs[kIdxZ]);
  aclTensor *aclConvOut = MakeAclTensor(
      dims + kIdxConvState * 4, strides + kIdxConvState * 4,
      dtypes[kIdxConvState], dimNums[kIdxConvState], g_scratch.dev);
  aclTensor *aclSsmOut = MakeAclTensor(
      dims + kIdxSsmCache * 4, strides + kIdxSsmCache * 4, dtypes[kIdxSsmCache],
      dimNums[kIdxSsmCache], static_cast<char *>(g_scratch.dev) + ssmOffset);
  if (aclNormOutput == nullptr || aclConvOut == nullptr || aclSsmOut == nullptr) {
    for (int i = 0; i < kNumInputs; ++i) {
      aclDestroyTensor(aclInputs[i]);
    }
    if (aclNormOutput != nullptr) aclDestroyTensor(aclNormOutput);
    if (aclConvOut != nullptr) aclDestroyTensor(aclConvOut);
    if (aclSsmOut != nullptr) aclDestroyTensor(aclSsmOut);
    return 5;
  }

  const uint64_t fftsAddr = ResolveFftsAddr();
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus status = aclnnMegaGdnPrefillOpGetWorkspaceSize(
      aclInputs[0], aclInputs[1], aclInputs[2], aclInputs[3], aclInputs[4], aclInputs[5],
      aclInputs[6], aclInputs[7], aclInputs[8], aclInputs[9], aclInputs[10], aclInputs[11],
      aclInputs[12], aclInputs[13], aclInputs[14], aclInputs[15], aclInputs[16],
      aclInputs[17], static_cast<int64_t>(fftsAddr), numMatrices, aclNormOutput,
      aclConvOut, aclSsmOut, &workspaceSize, &executor);
  if (status == 0) {
    void *workspace = nullptr;
    if (workspaceSize > 0) {
      if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        status = 6;
      }
    }
    if (status == 0) {
      status = aclnnMegaGdnPrefillOp(workspace, workspaceSize, executor,
                                     static_cast<aclrtStream>(stream));
    }
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
  }
  for (int i = 0; i < kNumInputs; ++i) {
    aclDestroyTensor(aclInputs[i]);
  }
  aclDestroyTensor(aclNormOutput);
  aclDestroyTensor(aclConvOut);
  aclDestroyTensor(aclSsmOut);
  if (status != 0) {
    std::fprintf(stderr,
                 "MegaGdnPrefill eager: aclnn status %d (workspaceSize=%llu "
                 "numMatrices=%lld ffts=%llu)\n",
                 static_cast<int>(status),
                 static_cast<unsigned long long>(workspaceSize),
                 static_cast<long long>(numMatrices),
                 static_cast<unsigned long long>(fftsAddr));
  }
  return status == 0 ? 0 : static_cast<int>(status);
}
