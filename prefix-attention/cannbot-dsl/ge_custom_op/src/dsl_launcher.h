// Schema-driven launcher for the prebuilt cannbotdsl artifacts.
//
// Host ABI v9 (self-describing): each kernel artifact .so exports
//   <base>__cannbot_schema_v2() -> const CannBotHostSchemaV2*
//   <base>__cannbot_invoke_v9(aclrtStream, const void* frame, uint64_t bytes)
// The schema carries frame size/alignment, contract digest and per-slot
// offsets, so the call frame is built generically here (no hardcoded layout).
// The AICPU metadata artifact exports a flat C launch stub.
#pragma once

#include <cstdint>

namespace prefix_attn {

struct LaunchParams {
  void *out;             // device addr, [T, N1, 64], same dtype as query
  void *query;           // device addr, [T, N1, 64] fp16/bf16
  void *key;             // device addr, [T, N2, 64]
  void *value;           // device addr, [T, N2, 64]
  int64_t total_tokens;  // T
  int64_t n1;            // query heads
  int64_t n2;            // kv heads
  void *bounds;          // device addr, int32 [B+1]
  int64_t bounds_len;    // B+1
  void *metadata;        // device addr, int32 [PREFIX_ATTN_METADATA_ELEMS],
                         // MUST be zero-initialized before LauncherRun
  float scale;
  void *stream;          // aclrtStream
  bool bf16;             // false -> fp16 artifact
};

// dlopen the three artifact .so files (idempotent).  Directory resolution:
// $PREFIX_ATTN_ARTIFACTS, else <dir of this .so>/artifacts, else
// <dir of this .so> (probe builds sit next to the artifacts).
// Returns 0 on success, non-zero error code otherwise (see LauncherError).
int LauncherInit();

// Effective AIC core budget: min(device default from export manifest, cap),
// with $PREFIX_ATTN_AIC_CORES override (same semantics as the python host).
int QueryAicCores();

// Launch sequence (both async on params.stream, order guaranteed):
//   1. AICPU metadata planner into params.metadata
//   2. macro attention kernel via invoke_v9
// Returns 0 on success; positive values are invoke_v9 status codes.
int LauncherRun(const LaunchParams &params);

// Last error message (thread-local).
const char *LauncherError();

}  // namespace prefix_attn
