// Eager bridge for the prefix_attn_torch torch binding: torch tensors ->
// prefix_attn_run_v2 (host planning + flat-args kernel launch in
// libcust_opapi.so).  No AICPU submission anywhere on this path.
#include <cmath>

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>

extern "C" {
int prefix_attn_run_v2(void *out, const void *query, const void *key,
                       const void *value, const void *bounds,
                       long long total_tokens, long long n1,
                       long long n2, long long bounds_len,
                       float scale, void *stream);
}

at::Tensor prefix_attention(const at::Tensor &query, const at::Tensor &key,
                            const at::Tensor &value,
                            const at::Tensor &actual_seq_lengths,
                            double scale) {
  constexpr int64_t kHeadDim = 64;
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "query/key/value must be rank-3 TND tensors");
  TORCH_CHECK(actual_seq_lengths.dim() == 1 &&
                  actual_seq_lengths.numel() >= 2,
              "actual_seq_lengths must be 1-D [B+1] with B >= 1");
  TORCH_CHECK(query.scalar_type() == at::kHalf,
              "query must be fp16 (bf16 is not supported by this "
              "deliverable -- see ge/README.md)");
  TORCH_CHECK(key.scalar_type() == query.scalar_type() &&
                  value.scalar_type() == query.scalar_type(),
              "key/value must share the query dtype (fp16)");
  TORCH_CHECK(actual_seq_lengths.scalar_type() == at::kInt,
              "actual_seq_lengths must be int32");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                  value.is_contiguous() &&
                  actual_seq_lengths.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(query.size(2) == kHeadDim, "only D=64 is supported");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "key and value must have identical shapes");
  TORCH_CHECK(key.size(0) == query.size(0),
              "key/value must match query T");
  // Bind the calling thread to the tensors' device: python worker threads
  // do not inherit torch.npu.set_device, and the raw aclrt calls below
  // (malloc/launch) would otherwise land on the thread's default device
  // while the stream/tensors live on another -> SVM pagefault at launch.
  const c10::DeviceGuard device_guard(query.device());
  const float s = scale > 0.0 ? static_cast<float>(scale)
                              : static_cast<float>(
                                    1.0 / std::sqrt(static_cast<double>(kHeadDim)));
  at::Tensor out = at::empty_like(query);
  // Drain TorchNPU host submissions before issuing raw ACL work on the same
  // device stream. stream(false) can overtake pending Q/K/V production,
  // especially on the first inference; this is not a device synchronization.
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  const int status = prefix_attn_run_v2(
      out.data_ptr(), query.data_ptr(), key.data_ptr(), value.data_ptr(),
      actual_seq_lengths.data_ptr(), query.size(0), query.size(1), key.size(1),
      actual_seq_lengths.numel(), s, static_cast<void *>(stream));
  TORCH_CHECK(status == 0, "prefix_attention launch failed, status=", status);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefix_attention", &prefix_attention,
        "prefix-in-Q attention (host planner + flat-args AIC kernel)");
}
