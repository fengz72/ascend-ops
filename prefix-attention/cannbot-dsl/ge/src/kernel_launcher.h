// Stream-ordered launchers for the two device kernels, via the codegen'd
// host entries declared in artifacts.h (link-time DT_NEEDED, found via
// RUNPATH $ORIGIN/artifacts):
//
//   AICPU planner  aicpu_metadata.so
//   AIC  kernel    kernel_fp16.so  (ABI v9 call frame)
//
// Both must go through the bisheng <<<>>> layer: aclrtLaunchKernelV2 does
// NOT wait for in-flight AICPU tasks, so a plain launch can race the
// planner.
#ifndef PREFIX_ATTN_GE_SRC_KERNEL_LAUNCHER_H_
#define PREFIX_ATTN_GE_SRC_KERNEL_LAUNCHER_H_

#include <cstdint>

namespace prefix_attn {

// Metadata buffer length (bound to artifacts.h by static_assert in the .cc).
constexpr int64_t kMetadataElems = 4096;

// Block count pinned to the process-wide GEInitialize core limit:
// custom_op_v2 has no interface to query it inside Execute.
constexpr int kAicCores = 16;

// AICPU planner: reads bounds (int32[bounds_len]), writes metadata.
int LaunchAicpuPlanner(void *stream, const void *bounds, void *metadata,
                       uint32_t bounds_len, uint32_t n1, uint32_t n2,
                       uint32_t aic_core_num);

// AIC macro kernel via the ABI v9 host entry.  Returns 0 on success.
int LaunchAttnKernel(void *stream, void *out, const void *query,
                     const void *key, const void *value, const void *bounds,
                     void *metadata, int64_t total_tokens, int64_t bounds_len,
                     float scale);

}  // namespace prefix_attn

#endif  // PREFIX_ATTN_GE_SRC_KERNEL_LAUNCHER_H_
