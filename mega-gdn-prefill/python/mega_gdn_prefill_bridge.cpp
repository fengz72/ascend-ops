// Eager bridge, JIT-compiled by python/mega_gdn_prefill_binding.py: torch
// tensors -> mega_gdn_prefill_run_v2 (flat C ABI of libcust_opapi.so).
// Contract mirrors the GE wrapper: z is overwritten in place (Tensor(a!)) and
// returned as an alias.
#include <array>
#include <cstdint>

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>

extern "C" int mega_gdn_prefill_run_v2(void *const *addrs, const int64_t *dims,
                                       const int64_t *strides,
                                       const int32_t *dtypes,
                                       const int32_t *dim_nums,
                                       int64_t num_matrices, void *stream);

namespace {

constexpr int kNumInputs = 18;
constexpr int kIdxB = 1;
constexpr int kIdxZ = 3;
constexpr int kIdxCuSeqlens = 16;
constexpr int kHeadChunk = 128;

int32_t AclDtype(const at::Tensor &t) {
  switch (t.scalar_type()) {
    case at::kBFloat16: return ACL_BF16;
    case at::kFloat: return ACL_FLOAT;
    case at::kHalf: return ACL_FLOAT16;
    case at::kInt: return ACL_INT32;
    default: return -1;
  }
}

// dtype contract of the GE REG_OP prototype
const at::ScalarType kDtypes[kNumInputs] = {
    at::kBFloat16, at::kBFloat16, at::kBFloat16, at::kBFloat16, at::kBFloat16,
    at::kBFloat16, at::kFloat, at::kFloat, at::kInt, at::kInt, at::kInt,
    at::kInt, at::kFloat, at::kFloat, at::kFloat, at::kBFloat16, at::kInt,
    at::kBFloat16};

const char *const kNames[kNumInputs] = {
    "mixed_qkv", "b", "a", "z", "conv_weight", "conv_state", "a_log",
    "dt_bias", "conv_state_read_indices", "conv_state_write_indices",
    "ssm_state_read_indices", "ssm_state_write_indices", "ssm_cache",
    "mask_lower", "mask_full", "minus_identity", "cu_seqlens", "norm_weight"};

}  // namespace

at::Tensor mega_gdn_prefill(at::Tensor mixed_qkv, at::Tensor b, at::Tensor a,
                            at::Tensor z, at::Tensor conv_weight,
                            at::Tensor conv_state, at::Tensor a_log,
                            at::Tensor dt_bias,
                            at::Tensor conv_state_read_indices,
                            at::Tensor conv_state_write_indices,
                            at::Tensor ssm_state_read_indices,
                            at::Tensor ssm_state_write_indices,
                            at::Tensor ssm_cache, at::Tensor mask_lower,
                            at::Tensor mask_full, at::Tensor minus_identity,
                            at::Tensor cu_seqlens, at::Tensor norm_weight) {
  const at::Tensor *const ts[kNumInputs] = {
      &mixed_qkv, &b, &a, &z, &conv_weight, &conv_state, &a_log, &dt_bias,
      &conv_state_read_indices, &conv_state_write_indices,
      &ssm_state_read_indices, &ssm_state_write_indices, &ssm_cache,
      &mask_lower, &mask_full, &minus_identity, &cu_seqlens, &norm_weight};
  for (int i = 0; i < kNumInputs; ++i) {
    const at::Tensor &t = *ts[i];
    TORCH_CHECK(t.device() == z.device(), kNames[i], " must be on ", z.device());
    TORCH_CHECK(t.scalar_type() == kDtypes[i], kNames[i], " must be ",
                toString(kDtypes[i]), " (got ", toString(t.scalar_type()), ")");
    TORCH_CHECK(t.is_contiguous(), kNames[i], " must be contiguous");
    TORCH_CHECK(t.dim() >= 1 && t.dim() <= 4, kNames[i], " must be 1-4D");
  }
  TORCH_CHECK(z.dim() == 3, "z must be [T, NV, DV]");
  TORCH_CHECK(b.dim() == 2, "b must be [T, NV]");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
              "cu_seqlens must be 1-D [B+1] with B >= 1");
  // Bind the calling thread to the tensors' device: python worker threads do
  // not inherit torch.npu.set_device, and both getCurrentNPUStream and the
  // thread-local scratch inside the .so must follow the input device.
  const c10::DeviceGuard device_guard(z.device());

  // num_matrices from cu_seqlens on host (GE Execute parity)
  const at::Tensor cuHost = cu_seqlens.to(at::kCPU);
  const int32_t *const cu = cuHost.data_ptr<int32_t>();
  const int64_t batch = cu_seqlens.numel() - 1;
  int64_t numMatrices = 0;
  for (int64_t i = 0; i < batch; ++i) {
    const int64_t length = static_cast<int64_t>(cu[i + 1]) - cu[i];
    TORCH_CHECK(length >= 0, "cu_seqlens must be non-decreasing");
    numMatrices += (length + kHeadChunk - 1) / kHeadChunk;
  }
  numMatrices *= b.size(1);
  TORCH_CHECK(numMatrices > 0, "derived num_matrices must be positive");

  std::array<void *, kNumInputs> addrs{};
  std::array<int64_t, kNumInputs * 4> dims{}, strides{};
  std::array<int32_t, kNumInputs> dtypes{}, dimNums{};
  for (int i = 0; i < kNumInputs; ++i) {
    const at::Tensor &t = *ts[i];
    addrs[i] = t.data_ptr();
    dimNums[i] = static_cast<int32_t>(t.dim());
    dtypes[i] = AclDtype(t);
    for (int64_t d = 0; d < 4; ++d) {
      dims[i * 4 + d] = d < t.dim() ? t.size(d) : 1;
      strides[i * 4 + d] = d < t.dim() ? t.stride(d) : 1;
    }
  }
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  const int status = mega_gdn_prefill_run_v2(
      addrs.data(), dims.data(), strides.data(), dtypes.data(), dimNums.data(),
      numMatrices, static_cast<void *>(stream));
  TORCH_CHECK(status == 0, "mega_gdn_prefill eager launch failed, status=", status);
  return z;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mega_gdn_prefill", &mega_gdn_prefill,
        "MegaGdnPrefillOp eager: overwrites z in place, returns it as alias");
}
