// Generated aclnn entry points of the vendored CausalConv1dQkv (the autogen
// header rides the vendor package include dir; declared locally to keep the
// wrappers self-contained and the call sites from drifting).
//
// Only the aclIntArray metadata variant is declared: the four ValueDepend
// metadata inputs are consumed by host tiling, and the upstream torch call
// site feeds at::IntArrayRef through this same variant. The *Tensor variant
// exists in the autogen header but is *not* usable with device-resident
// metadata (measured: nnopbase reads the value-depend data as host memory and
// the process segfaults), so it is not declared or called here.
#ifndef CAUSAL_CONV1D_QKV_ACLNN_H_
#define CAUSAL_CONV1D_QKV_ACLNN_H_

#include <cstdint>

#include "aclnn/acl_meta.h"
#include "acl/acl_rt.h"

extern "C" aclnnStatus aclnnCausalConv1dQkvGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *biasOptional,
    const aclTensor *convStates, const aclIntArray *queryStartLocOptional,
    const aclIntArray *cacheIndicesOptional,
    const aclIntArray *initialStateModeOptional,
    const aclIntArray *numAcceptedTokensOptional, int64_t activationMode,
    int64_t padSlotId, int64_t runMode, int64_t qDim, int64_t kDim, int64_t vDim,
    int64_t headDim, const aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" aclnnStatus aclnnCausalConv1dQkv(void *workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor *executor,
                                            aclrtStream stream);

#endif  // CAUSAL_CONV1D_QKV_ACLNN_H_
