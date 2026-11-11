// Eager flat entry for the JIT torch bridge (python/causal_conv1d_bridge.cpp):
// torch tensors flattened to raw pointer/shape arrays -> aclnnCausalConv1d, the
// same sequence as the GE Execute path (dense FN prefill: runMode=0, no
// metadata). Output is a fresh device buffer allocated by the caller side.
// No torch headers on this side.
#include <cstdint>
#include <cstdio>

#include "aclnn/acl_meta.h"
#include "acl/acl_rt.h"

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

namespace {

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

}  // namespace

extern "C" int causal_conv1d_run(void *const *addrs, const int64_t *dims,
                                 const int64_t *strides,
                                 const int32_t *dtypes,
                                 const int32_t *dimNums,
                                 const char *activation, int64_t headNum,
                                 void *outAddr, const int64_t *outDims,
                                 const int64_t *outStrides, int32_t outDtype,
                                 int32_t outDimNum, void *stream) {
  if (addrs == nullptr || dims == nullptr || strides == nullptr ||
      dtypes == nullptr || dimNums == nullptr || outAddr == nullptr ||
      stream == nullptr) {
    return 2;
  }
  aclTensor *x = MakeAclTensor(dims, strides, dtypes[0], dimNums[0], addrs[0]);
  aclTensor *weight = MakeAclTensor(dims + 4, strides + 4, dtypes[1],
                                    dimNums[1], addrs[1]);
  // addrs[2] may be null: no bias
  aclTensor *bias = addrs[2] != nullptr
      ? MakeAclTensor(dims + 8, strides + 8, dtypes[2], dimNums[2], addrs[2])
      : nullptr;
  aclTensor *out = MakeAclTensor(outDims, outStrides, outDtype, outDimNum,
                                 outAddr);
  if (x == nullptr || weight == nullptr || out == nullptr) {
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    return 3;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus status = aclnnCausalConv1dGetWorkspaceSize(
      x, weight, bias,
      nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr,
      const_cast<char *>(activation),
      -1, 0, 0, headNum, -1,
      out, &workspaceSize, &executor);
  if (status == 0) {
    void *workspace = nullptr;
    if (workspaceSize > 0) {
      if (aclrtMalloc(&workspace, workspaceSize,
                      ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        status = 4;
      }
    }
    if (status == 0) {
      status = aclnnCausalConv1d(workspace, workspaceSize, executor,
                                  static_cast<aclrtStream>(stream));
    }
    if (workspace != nullptr) {
      // aclrtFree is not stream-ordered; the conv kernel may still be reading
      // the workspace. Drain the stream before freeing (mega wrapper pattern).
      if (aclrtSynchronizeStream(static_cast<aclrtStream>(stream)) != ACL_SUCCESS
          && status == 0) {
        status = 5;
      }
      aclrtFree(workspace);
    }
  }
  if (x != nullptr) aclDestroyTensor(x);
  if (weight != nullptr) aclDestroyTensor(weight);
  if (bias != nullptr) aclDestroyTensor(bias);
  if (out != nullptr) aclDestroyTensor(out);
  if (status != 0) {
    std::fprintf(stderr, "CausalConv1d eager: aclnn status %d\n",
                 static_cast<int>(status));
  }
  return status == 0 ? 0 : static_cast<int>(status);
}
