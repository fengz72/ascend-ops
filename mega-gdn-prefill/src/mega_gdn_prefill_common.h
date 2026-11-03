// Shared internals of the MegaGdnPrefill GE wrapper and the eager flat entry
// (both compiled into libcust_opapi.so): the aclnn declarations and the FFTS
// address resolver. Project addition, not vendored operator code.
#ifndef MEGA_GDN_PREFILL_COMMON_H_
#define MEGA_GDN_PREFILL_COMMON_H_

#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>

#include "aclnn/acl_meta.h"
#include "acl/acl_rt.h"

extern "C" aclnnStatus aclnnMegaGdnPrefillOpGetWorkspaceSize(
    const aclTensor *mixedQkv, const aclTensor *b, const aclTensor *a, const aclTensor *z,
    const aclTensor *convWeight, const aclTensor *convState, const aclTensor *aLog,
    const aclTensor *dtBias, const aclTensor *convStateReadIndices,
    const aclTensor *convStateWriteIndices, const aclTensor *ssmStateReadIndices,
    const aclTensor *ssmStateWriteIndices, const aclTensor *ssmCache,
    const aclTensor *maskLower, const aclTensor *maskFull, const aclTensor *minusIdentity,
    const aclTensor *cuSeqlens, const aclTensor *normWeight, int64_t fftsAddr,
    int64_t numMatrices, const aclTensor *normOutputOut, const aclTensor *convStateOutOut,
    const aclTensor *ssmCacheOutOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnMegaGdnPrefillOp(void *workspace, uint64_t workspaceSize,
                                             aclOpExecutor *executor, aclrtStream stream);

// A2/A3 FFTS address resolved at runtime; 0 disables the C2C path (env var or
// resolver failure).
inline uint64_t ResolveFftsAddr() {
  if (std::getenv("MEGA_GDN_PREFILL_FFTS_DISABLE") != nullptr) {
    return 0;
  }
  using RtGetC2cCtrlAddr = int32_t (*)(uint64_t *, uint32_t *);
  static RtGetC2cCtrlAddr getC2cCtrlAddr = []() -> RtGetC2cCtrlAddr {
    void *runtime = dlopen("libruntime.so", RTLD_LAZY | RTLD_LOCAL);
    if (runtime == nullptr) {
      return nullptr;
    }
    return reinterpret_cast<RtGetC2cCtrlAddr>(dlsym(runtime, "rtGetC2cCtrlAddr"));
  }();
  if (getC2cCtrlAddr == nullptr) {
    return 0;
  }
  uint64_t fftsAddr = 0;
  uint32_t fftsLen = 0;
  if (getC2cCtrlAddr(&fftsAddr, &fftsLen) != 0 || fftsLen == 0) {
    return 0;
  }
  return fftsAddr;
}

#endif  // MEGA_GDN_PREFILL_COMMON_H_
