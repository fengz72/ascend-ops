// Eager bridge for the prefix_attn_torch torch binding: torch tensors -> flat C
// ABI of libcust_opapi.so (prefix_attn_run) -> dsl_launcher -> kernel .so.
#include <cmath>

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>

extern "C" {
int prefix_attn_run(void *out, const void *query, const void *key,
                    const void *value, const void *bounds,
                    long long total_tokens, long long n1, long long n2,
                    long long bounds_len, void *metadata, float scale,
                    void *stream, int bf16);
long long prefix_attn_metadata_elems(void);
const char *prefix_attn_last_error(void);
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
  TORCH_CHECK(query.scalar_type() == at::kHalf ||
                  query.scalar_type() == at::kBFloat16,
              "query must be fp16 or bf16");
  TORCH_CHECK(key.scalar_type() == query.scalar_type() &&
                  value.scalar_type() == query.scalar_type(),
              "key/value must share the query dtype (fp16 or bf16)");
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
  const float s = scale > 0.0 ? static_cast<float>(scale)
                              : static_cast<float>(
                                    1.0 / std::sqrt(static_cast<double>(kHeadDim)));
  at::Tensor out = at::empty_like(query);
  // The AICPU metadata planner assumes a zero-initialized buffer; at::zeros
  // memsets on the current NPU stream, ordered before both launches below.
  at::Tensor metadata = at::zeros({prefix_attn_metadata_elems()},
                                  query.options().dtype(at::kInt));
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
  const int status = prefix_attn_run(
      out.data_ptr(), query.data_ptr(), key.data_ptr(), value.data_ptr(),
      actual_seq_lengths.data_ptr(), query.size(0), query.size(1),
      key.size(1), actual_seq_lengths.numel(), metadata.data_ptr(), s,
      static_cast<void *>(stream),
      query.scalar_type() == at::kBFloat16 ? 1 : 0);
  TORCH_CHECK(status == 0, "prefix_attn launch failed: ",
              prefix_attn_last_error());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefix_attention", &prefix_attention);
}
