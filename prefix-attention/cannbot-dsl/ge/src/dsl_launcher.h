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
// <dir of this .so>/artifacts (the OPP layout installed by install.sh), else
// <dir of this .so> (some layouts place artifacts directly next to the .so).
// Returns 0 on success, non-zero error code otherwise (see LauncherError).
int LauncherInit();

// Effective AIC core budget: fixed 16 for now (the python host queries the
// device via torch_npu instead).
int QueryAicCores();

// Launch sequence (both async on params.stream, order guaranteed):
//   1. AICPU metadata planner into params.metadata
//   2. macro attention kernel via invoke_v9
// Returns 0 on success; positive values are invoke_v9 status codes.
int LauncherRun(const LaunchParams &params);

// Last error message (thread-local).
const char *LauncherError();

}  // namespace prefix_attn

// Flat C ABI for eager frontends (e.g. ge/prefix_attn_torch): the
// extern "C" symbols are stable across C++ ABI variants, so consumers link
// libcust_opapi.so and declare these prototypes without shipping headers.
#ifdef __cplusplus
extern "C" {
#endif

// One eager launch (async on `stream`); `metadata` must be a
// zero-initialized device buffer of prefix_attn_metadata_elems() int32.
// Returns 0 on success; see prefix_attn_last_error().
int prefix_attn_run(void *out, const void *query, const void *key,
                    const void *value, const void *bounds,
                    long long total_tokens, long long n1, long long n2,
                    long long bounds_len, void *metadata, float scale,
                    void *stream, int bf16);

// int32 element count of the metadata workspace.
long long prefix_attn_metadata_elems(void);

// Last launcher error message (thread-local).
const char *prefix_attn_last_error(void);

#ifdef __cplusplus
}
#endif
