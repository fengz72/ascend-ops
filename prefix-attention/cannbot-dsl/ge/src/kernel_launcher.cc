// ABI v9 call-frame construction for the AIC kernel launch (the frame
// layout is the artifact's own schema_v2 declaration).
#include "kernel_launcher.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

#include "artifacts.h"

// kMetadataElems must match the generated artifacts.h: the workspace the op
// allocates from it is the buffer whose shape LaunchAttnKernel reports to
// the kernel.
static_assert(PREFIX_ATTN_METADATA_ELEMS == prefix_attn::kMetadataElems,
              "artifacts.h (generated) and kernel_launcher.h drifted -- "
              "regenerate artifacts.h via ge/scripts/dump_kernel_src.py");

namespace prefix_attn {
namespace {

constexpr uint32_t kFrameMagic = 0x43424639u;   // "CBF9"
constexpr uint32_t kSchemaMagic = 0x43425332u;  // "CBS2"
constexpr uint16_t kHostAbiVersion = 9;
constexpr uint16_t kSchemaVersion = 2;

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

constexpr size_t kMaxFrameBytes = 1024;

// One-time schema validation (thread-safe static init); nullptr on mismatch.
const SchemaV2 *LoadSchema() {
  static const SchemaV2 *schema = []() -> const SchemaV2 * {
    const auto *s = static_cast<const SchemaV2 *>(PREFIX_ATTN_SCHEMA_FP16());
    if (s == nullptr || s->magic != kSchemaMagic ||
        s->host_abi_version != kHostAbiVersion ||
        s->schema_version != kSchemaVersion ||
        s->slot_count != 8 || s->slots == nullptr ||
        s->frame_bytes == 0 ||
        s->frame_bytes > kMaxFrameBytes ||
        s->frame_alignment == 0 || s->frame_alignment > 16) {
      std::fprintf(stderr,
                   "[PrefixAttention] kernel artifact schema mismatch\n");
      return nullptr;
    }
    return s;
  }();
  return schema;
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

int LaunchAicpuPlanner(void *stream, const void *bounds, void *metadata,
                       uint32_t bounds_len, uint32_t n1, uint32_t n2,
                       uint32_t aic_core_num) {
  PREFIX_ATTN_AICPU_LAUNCH(stream, reinterpret_cast<uint64_t>(bounds),
                           reinterpret_cast<uint64_t>(metadata), bounds_len, n1,
                           n2, aic_core_num, /*macro_cost=*/1U);
  return 0;
}

int LaunchAttnKernel(void *stream, void *out, const void *query,
                     const void *key, const void *value, const void *bounds,
                     void *metadata, int64_t total_tokens, int64_t bounds_len,
                     float scale) {
  const SchemaV2 *schema = LoadSchema();
  if (schema == nullptr) {
    return 1;
  }

  // Per-call stack frame; the invoke consumes it synchronously.
  alignas(16) uint8_t frame[kMaxFrameBytes];
  const uint64_t frame_bytes = schema->frame_bytes;
  memset(frame, 0, frame_bytes);

  FrameHeaderV9 *header = reinterpret_cast<FrameHeaderV9 *>(frame);
  header->magic = kFrameMagic;
  header->host_abi_version = kHostAbiVersion;
  header->schema_version = kSchemaVersion;
  header->slot_count = schema->slot_count;
  header->reserved = 0;
  header->frame_bytes = frame_bytes;
  memcpy(header->contract_digest, schema->contract_digest, 32);

  const int64_t n1 = PREFIX_ATTN_N1;
  const int64_t n2 = PREFIX_ATTN_N2;
  const int64_t q_shape[3] = {total_tokens, n1, 64};
  const int64_t q_stride[3] = {n1 * 64, 64, 1};
  const int64_t kv_shape[3] = {total_tokens, n2, 64};
  const int64_t kv_stride[3] = {n2 * 64, 64, 1};
  const int64_t bounds_shape[1] = {bounds_len};
  const int64_t bounds_stride[1] = {1};
  const int64_t meta_shape[1] = {PREFIX_ATTN_METADATA_ELEMS};
  const int64_t meta_stride[1] = {1};
  const SlotSchemaV2 *slots = schema->slots;
  WriteTensorSlot(frame, slots[0], out, q_shape, q_stride);
  WriteTensorSlot(frame, slots[1], const_cast<void *>(query), q_shape, q_stride);
  WriteTensorSlot(frame, slots[2], const_cast<void *>(key), kv_shape, kv_stride);
  WriteTensorSlot(frame, slots[3], const_cast<void *>(value), kv_shape, kv_stride);
  *reinterpret_cast<float *>(frame + slots[4].offset) = scale;
  WriteTensorSlot(frame, slots[5], const_cast<void *>(bounds), bounds_shape,
                  bounds_stride);
  WriteTensorSlot(frame, slots[6], metadata, meta_shape, meta_stride);
  *reinterpret_cast<int64_t *>(frame + slots[7].offset) = kAicCores;

  const int32_t status = PREFIX_ATTN_INVOKE_FP16(stream, frame, frame_bytes);
  if (status != 0) {
    std::fprintf(stderr, "[PrefixAttention] invoke_v9 returned %d\n", status);
    return 1;
  }
  return 0;
}

}  // namespace prefix_attn
