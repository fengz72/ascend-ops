// Implementation of the schema-driven DSL artifact launcher.  The host ABI
// structs below mirror the definitions embedded in the cannbotdsl codegen
// dump (codegen/launch_prefix_macro.asc, host ABI v9 / schema v2) and are
// validated against the artifact's own schema entry at init.
#include "dsl_launcher.h"

#include <dlfcn.h>
#include <unistd.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "artifacts.h"

namespace prefix_attn {
namespace {

constexpr uint32_t kFrameMagic = 0x43424639u;   // "CBF9"
constexpr uint32_t kSchemaMagic = 0x43425332u;  // "CBS2"
constexpr uint16_t kHostAbiVersion = 9;
constexpr uint16_t kSchemaVersion = 2;
constexpr uint32_t kSlotKindTensor = 1;
constexpr uint32_t kSlotKindScalar = 2;

// Host ABI v9 frame header (56 bytes, digest at offset 24).
struct FrameHeaderV9 {
  uint32_t magic;
  uint16_t host_abi_version;
  uint16_t schema_version;
  uint32_t slot_count;
  uint32_t reserved;
  uint64_t frame_bytes;
  uint8_t contract_digest[32];
};

// Per-slot layout descriptor (128 bytes).
struct SlotSchemaV2 {
  uint32_t kind;
  uint32_t flags;
  uint32_t dtype_code;
  uint32_t dtype_bits;
  uint32_t dtype_lanes;
  uint32_t encoding;
  uint32_t logical_rank;
  uint32_t physical_rank;
  uint64_t offset;
  uint64_t size;
  uint64_t alignment;
  uint64_t pointer_offset;
  uint64_t shape_offset;
  uint64_t stride_offset;
  uint64_t physical_shape_offset;
  uint64_t physical_stride_offset;
  uint8_t abi_type_digest[32];
};

// Artifact-exported schema (returned by <base>__cannbot_schema_v2()).
struct SchemaV2 {
  uint32_t magic;
  uint16_t host_abi_version;
  uint16_t schema_version;
  uint32_t slot_count;
  uint32_t reserved;
  uint64_t frame_bytes;
  uint64_t frame_alignment;
  uint8_t contract_digest[32];
  const SlotSchemaV2 *slots;
};

static_assert(sizeof(FrameHeaderV9) == 56, "frame header ABI mismatch");
static_assert(offsetof(FrameHeaderV9, contract_digest) == 24, "header layout");
static_assert(sizeof(SlotSchemaV2) == 128, "slot schema ABI mismatch");

typedef const SchemaV2 *(*SchemaFn)();
typedef int32_t (*InvokeV9Fn)(void *stream, const void *frame, uint64_t bytes);
typedef void (*AicpuDoFn)(void *stream, uint64_t bounds, uint64_t metadata,
                          uint32_t bounds_len, uint32_t num_heads_q,
                          uint32_t num_heads_kv, uint32_t aic_core_num,
                          uint32_t macro_cost);

thread_local std::string g_error;

void SetError(const std::string &msg) { g_error = msg; }

struct Artifacts {
  void *kernel_fp16 = nullptr;
  void *kernel_bf16 = nullptr;
  void *aicpu = nullptr;
  const SchemaV2 *schema_fp16 = nullptr;
  const SchemaV2 *schema_bf16 = nullptr;
  InvokeV9Fn invoke_fp16 = nullptr;
  InvokeV9Fn invoke_bf16 = nullptr;
  AicpuDoFn aicpu_do = nullptr;
  int aic_cores = 0;
};

Artifacts g_art;
std::once_flag g_once;
int g_init_status = -1;

std::string DirOfThisLibrary() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(&DirOfThisLibrary), &info) != 0 &&
      info.dli_fname != nullptr) {
    std::string path = info.dli_fname;
    size_t pos = path.rfind('/');
    if (pos != std::string::npos) return path.substr(0, pos);
  }
  return ".";
}

void *OpenArtifact(const std::string &dir, const char *name) {
  std::string path = dir + "/" + name;
  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    // probe builds may sit directly next to the artifacts
    handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  }
  if (handle == nullptr) SetError(std::string("dlopen failed: ") + dlerror());
  return handle;
}

void *Resolve(void *handle, const char *symbol) {
  dlerror();
  void *fn = dlsym(handle, symbol);
  if (fn == nullptr) {
    SetError(std::string("dlsym ") + symbol + " failed: " + dlerror());
  }
  return fn;
}

int ValidateSchema(const SchemaV2 *schema, const char *tag) {
  if (schema == nullptr) return 1;
  if (schema->magic != kSchemaMagic || schema->host_abi_version != kHostAbiVersion ||
      schema->schema_version != kSchemaVersion) {
    SetError(std::string(tag) + ": schema magic/version mismatch");
    return 1;
  }
  if (schema->slot_count != 8 || schema->slots == nullptr) {
    SetError(std::string(tag) + ": expected 8 slots");
    return 1;
  }
  // Slot contract of _launch_prefix_macro(out, query, key, value, scale,
  // bounds, metadata, block_dim): tensors [3d,3d,3d,3d,-,1d,1d,-].
  const uint32_t expect_kind[8] = {kSlotKindTensor, kSlotKindTensor, kSlotKindTensor,
                                   kSlotKindTensor, kSlotKindScalar, kSlotKindTensor,
                                   kSlotKindTensor, kSlotKindScalar};
  const uint32_t expect_rank[8] = {3, 3, 3, 3, 0, 1, 1, 0};
  for (uint32_t i = 0; i < 8; ++i) {
    const SlotSchemaV2 &s = schema->slots[i];
    if (s.kind != expect_kind[i] || s.logical_rank != expect_rank[i]) {
      char buf[128];
      snprintf(buf, sizeof(buf), "%s: slot %u kind/rank %" PRIu32 "/%" PRIu32
               " unexpected", tag, i, s.kind, s.logical_rank);
      SetError(buf);
      return 1;
    }
  }
  if (schema->slots[0].dtype_bits != 16 || schema->slots[4].dtype_bits != 32 ||
      schema->slots[5].dtype_bits != 32 || schema->slots[7].dtype_bits != 64) {
    SetError(std::string(tag) + ": slot dtype widths unexpected");
    return 1;
  }
  return 0;
}

int QueryAicCoresLocked() {
  int aic = PREFIX_ATTN_AIC_DEFAULT;
  const char *env = getenv("PREFIX_ATTN_AIC_CORES");
  if (env != nullptr) {
    int requested = atoi(env);
    if (requested <= 0) {
      SetError("PREFIX_ATTN_AIC_CORES must be a positive integer");
      return -1;
    }
    aic = requested < aic ? requested : aic;
  }
  if (aic > PREFIX_ATTN_AIC_CAP) aic = PREFIX_ATTN_AIC_CAP;
  return aic;
}

void InitOnce() {
  std::string dir = ".";
  const char *env_dir = getenv("PREFIX_ATTN_ARTIFACTS");
  if (env_dir != nullptr && env_dir[0] != '\0') {
    dir = env_dir;
  } else {
    std::string self = DirOfThisLibrary();
    // OPP layout: artifacts/ sits next to this .so; probe layout: same dir.
    dir = self + "/artifacts";
    if (access((dir + "/" + PREFIX_ATTN_SO_FP16).c_str(), R_OK) != 0) {
      dir = self;
    }
  }
  g_art.kernel_fp16 = OpenArtifact(dir, PREFIX_ATTN_SO_FP16);
  if (g_art.kernel_fp16 == nullptr) { g_init_status = 1; return; }
  g_art.kernel_bf16 = OpenArtifact(dir, PREFIX_ATTN_SO_BF16);
  if (g_art.kernel_bf16 == nullptr) { g_init_status = 1; return; }
  g_art.aicpu = OpenArtifact(dir, PREFIX_ATTN_AICPU_SO);
  if (g_art.aicpu == nullptr) { g_init_status = 1; return; }

  auto schema_fp16 = reinterpret_cast<SchemaFn>(
      Resolve(g_art.kernel_fp16, PREFIX_ATTN_SCHEMA_FP16));
  auto schema_bf16 = reinterpret_cast<SchemaFn>(
      Resolve(g_art.kernel_bf16, PREFIX_ATTN_SCHEMA_BF16));
  if (schema_fp16 == nullptr || schema_bf16 == nullptr) { g_init_status = 1; return; }
  g_art.schema_fp16 = schema_fp16();
  g_art.schema_bf16 = schema_bf16();
  if (ValidateSchema(g_art.schema_fp16, "kernel_fp16") != 0 ||
      ValidateSchema(g_art.schema_bf16, "kernel_bf16") != 0) {
    g_init_status = 1;
    return;
  }
  g_art.invoke_fp16 = reinterpret_cast<InvokeV9Fn>(
      Resolve(g_art.kernel_fp16, PREFIX_ATTN_INVOKE_FP16));
  g_art.invoke_bf16 = reinterpret_cast<InvokeV9Fn>(
      Resolve(g_art.kernel_bf16, PREFIX_ATTN_INVOKE_BF16));
  g_art.aicpu_do = reinterpret_cast<AicpuDoFn>(
      Resolve(g_art.aicpu, PREFIX_ATTN_AICPU_LAUNCH));
  if (g_art.invoke_fp16 == nullptr || g_art.invoke_bf16 == nullptr ||
      g_art.aicpu_do == nullptr) {
    g_init_status = 1;
    return;
  }
  g_art.aic_cores = QueryAicCoresLocked();
  g_init_status = g_art.aic_cores > 0 ? 0 : 1;
}

void WriteTensorSlot(uint8_t *frame, const SlotSchemaV2 &slot, void *ptr,
                     const int64_t *shape, const int64_t *stride) {
  *reinterpret_cast<void **>(frame + slot.offset + slot.pointer_offset) = ptr;
  memcpy(frame + slot.offset + slot.shape_offset, shape,
         slot.logical_rank * sizeof(int64_t));
  memcpy(frame + slot.offset + slot.stride_offset, stride,
         slot.logical_rank * sizeof(int64_t));
}

}  // namespace

int LauncherInit() {
  std::call_once(g_once, InitOnce);
  return g_init_status;
}

int QueryAicCores() {
  if (LauncherInit() != 0) return -1;
  return g_art.aic_cores;
}

int LauncherRun(const LaunchParams &p) {
  if (LauncherInit() != 0) return 1;
  if (p.out == nullptr || p.query == nullptr || p.key == nullptr ||
      p.value == nullptr || p.bounds == nullptr || p.metadata == nullptr ||
      p.stream == nullptr) {
    SetError("null device pointer or stream in LaunchParams");
    return 1;
  }
  const SchemaV2 *schema = p.bf16 ? g_art.schema_bf16 : g_art.schema_fp16;
  InvokeV9Fn invoke = p.bf16 ? g_art.invoke_bf16 : g_art.invoke_fp16;

  // 1. AICPU metadata planner (metadata buffer must be zeroed by the caller).
  g_art.aicpu_do(p.stream, reinterpret_cast<uint64_t>(p.bounds),
                 reinterpret_cast<uint64_t>(p.metadata),
                 static_cast<uint32_t>(p.bounds_len),
                 static_cast<uint32_t>(p.n1), static_cast<uint32_t>(p.n2),
                 static_cast<uint32_t>(g_art.aic_cores), /*macro_cost=*/1u);

  // 2. Build the ABI v9 call frame per the artifact's own schema.
  uint64_t frame_bytes = schema->frame_bytes;
  uint64_t alignment = schema->frame_alignment ? schema->frame_alignment : 8;
  uint64_t alloc_bytes = (frame_bytes + alignment - 1) / alignment * alignment;
  void *raw = aligned_alloc(alignment, alloc_bytes);
  if (raw == nullptr) {
    SetError("aligned_alloc failed for call frame");
    return 1;
  }
  memset(raw, 0, alloc_bytes);
  uint8_t *frame = reinterpret_cast<uint8_t *>(raw);
  FrameHeaderV9 *header = reinterpret_cast<FrameHeaderV9 *>(frame);
  header->magic = kFrameMagic;
  header->host_abi_version = kHostAbiVersion;
  header->schema_version = kSchemaVersion;
  header->slot_count = schema->slot_count;
  header->reserved = 0;
  header->frame_bytes = frame_bytes;
  memcpy(header->contract_digest, schema->contract_digest, 32);

  const int64_t q_shape[3] = {p.total_tokens, p.n1, 64};
  const int64_t q_stride[3] = {p.n1 * 64, 64, 1};
  const int64_t kv_shape[3] = {p.total_tokens, p.n2, 64};
  const int64_t kv_stride[3] = {p.n2 * 64, 64, 1};
  const int64_t bounds_shape[1] = {p.bounds_len};
  const int64_t bounds_stride[1] = {1};
  const int64_t meta_shape[1] = {PREFIX_ATTN_METADATA_ELEMS};
  const int64_t meta_stride[1] = {1};
  const SlotSchemaV2 *slots = schema->slots;
  WriteTensorSlot(frame, slots[0], p.out, q_shape, q_stride);
  WriteTensorSlot(frame, slots[1], p.query, q_shape, q_stride);
  WriteTensorSlot(frame, slots[2], p.key, kv_shape, kv_stride);
  WriteTensorSlot(frame, slots[3], p.value, kv_shape, kv_stride);
  *reinterpret_cast<float *>(frame + slots[4].offset) = p.scale;
  WriteTensorSlot(frame, slots[5], p.bounds, bounds_shape, bounds_stride);
  WriteTensorSlot(frame, slots[6], p.metadata, meta_shape, meta_stride);
  *reinterpret_cast<int64_t *>(frame + slots[7].offset) =
      static_cast<int64_t>(g_art.aic_cores);

  int32_t status = invoke(p.stream, frame, frame_bytes);
  free(raw);
  if (status != 0) {
    char buf[96];
    snprintf(buf, sizeof(buf), "invoke_v9 returned %" PRId32, status);
    SetError(buf);
  }
  return status;
}

const char *LauncherError() { return g_error.c_str(); }

}  // namespace prefix_attn
