// Generated aclnn entry points of the vendored CausalConv1d (the autogen
// header rides the vendor package include dir; declared locally to keep the
// wrappers self-contained and the two call sites from drifting).
#ifndef CAUSAL_CONV1D_ACLNN_H_
#define CAUSAL_CONV1D_ACLNN_H_

#include <cstdint>

#include "aclnn/acl_meta.h"
#include "acl/acl_rt.h"

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

#endif  // CAUSAL_CONV1D_ACLNN_H_
