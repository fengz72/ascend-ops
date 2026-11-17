// Eager bridge, JIT-compiled by python/causal_conv1d_qkv_binding.py: torch
// tensors -> causal_conv1d_qkv_run (flat C ABI of libcust_opapi.so), the same
// sequence as the GE Execute path. Output is a fresh device buffer.
//
// conv_states is handed over as the caller's own buffer because the kernel
// rewrites its last width-1 rows in place (sliding-window history); the schema
// marks it Tensor(a!) so dynamo orders later reads after the op.
//
// The four metadata tensors feed host tiling: host storage is used in place
// (the fast path), while NPU-resident metadata is drained stream-ordered by
// the entry. NPU-resident metadata must live on x's device, otherwise that
// drain would read a foreign-device buffer on x's stream.
#include <array>
#include <cstdint>
#include <optional>

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>

extern "C" int causal_conv1d_qkv_run(
    void *const *addrs, const int64_t *dims, const int64_t *strides,
    const int32_t *dtypes, const int32_t *dim_nums, int64_t activation_mode,
    int64_t pad_slot_id, int64_t run_mode, int64_t q_dim, int64_t k_dim,
    int64_t v_dim, int64_t head_dim, void *out_addr, const int64_t *out_dims,
    const int64_t *out_strides, int32_t out_dtype, int32_t out_dim_num,
    void *stream);

namespace {

int32_t AclDtype(const at::Tensor &t) {
  switch (t.scalar_type()) {
    case at::kBFloat16: return ACL_BF16;
    case at::kHalf: return ACL_FLOAT16;
    case at::kLong: return ACL_INT64;  // tiling metadata
    default: return -1;
  }
}

// Input order of the flat ABI; tensor i's dims/strides live at +4*i.
constexpr int kNumInputs = 8;
constexpr int kIdxQueryStartLoc = 4;  // first metadata slot (4..7)

const char *const kNames[kNumInputs] = {
    "x", "weight", "bias", "conv_states", "query_start_loc", "cache_indices",
    "initial_state_mode", "num_accepted_tokens"};

}  // namespace

at::Tensor causal_conv1d_qkv(
    at::Tensor x, at::Tensor weight, at::Tensor conv_states, int64_t q_dim,
    int64_t k_dim, int64_t v_dim, int64_t head_dim, int64_t activation_mode,
    int64_t pad_slot_id, int64_t run_mode,
    std::optional<at::ScalarType> out_dtype,
    std::optional<at::Tensor> query_start_loc,
    std::optional<at::Tensor> cache_indices,
    std::optional<at::Tensor> initial_state_mode,
    std::optional<at::Tensor> num_accepted_tokens) {
  // bias is absent by contract (packed QKV has no bias); slot 2 stays null.
  const at::Tensor *const ts[kNumInputs] = {
      &x, &weight, nullptr, &conv_states,
      query_start_loc.has_value() ? &query_start_loc.value() : nullptr,
      cache_indices.has_value() ? &cache_indices.value() : nullptr,
      initial_state_mode.has_value() ? &initial_state_mode.value() : nullptr,
      num_accepted_tokens.has_value() ? &num_accepted_tokens.value() : nullptr};
  for (int i = 0; i < kNumInputs; ++i) {
    if (ts[i] == nullptr) {
      continue;
    }
    const at::Tensor &t = *ts[i];
    // Metadata may live on host (the caller's cu_seqlens) or on the NPU.
    if (i < kIdxQueryStartLoc) {
      TORCH_CHECK(t.device() == x.device(), kNames[i], " must be on ", x.device());
      // The vendored OpDef declares these BF16-only; reject early so the
      // message names the offending tensor instead of an opaque aclnn code.
      TORCH_CHECK(t.scalar_type() == at::kBFloat16,
                  kNames[i], " must be bf16 (got ", toString(t.scalar_type()), ")");
    }
    TORCH_CHECK(t.is_contiguous(), kNames[i], " must be contiguous");
    if (i >= kIdxQueryStartLoc) {
      TORCH_CHECK(t.scalar_type() == at::kLong,
                  kNames[i], " must be int64 (got ", toString(t.scalar_type()), ")");
      TORCH_CHECK(t.dim() == 1, kNames[i], " must be 1-D");
      if (t.device().is_privateuseone()) {
        TORCH_CHECK(t.device() == x.device(), kNames[i],
                    " must be on ", x.device(), " when NPU-resident (got ",
                    t.device(), ")");
      }
    } else {
      TORCH_CHECK(AclDtype(t) != -1, kNames[i], " has unsupported dtype ",
                  toString(t.scalar_type()));
    }
  }
  TORCH_CHECK(x.dim() == 2, "x must be (T,D) in packed mode");
  TORCH_CHECK(weight.dim() == 2, "weight must be (width,D)");
  TORCH_CHECK(conv_states.dim() == 3,
              "conv_states must be (slots,width-1,D)");
  TORCH_CHECK(weight.size(1) == x.size(1), "weight D must match x last dim");
  TORCH_CHECK(conv_states.size(2) == x.size(1),
              "conv_states D must match x last dim");

  const at::ScalarType yType = out_dtype.value_or(x.scalar_type());
  TORCH_CHECK(yType == at::kBFloat16 || yType == at::kHalf,
              "out_dtype must be bf16/fp16 (got ", toString(yType), ")");
  // Bind the calling thread to the tensors' device: python worker threads do
  // not inherit torch.npu.set_device, and getCurrentNPUStream must follow the
  // input device.
  const c10::DeviceGuard device_guard(x.device());

  at::Tensor out = at::empty(x.sizes(), x.options().dtype(yType));

  std::array<void *, kNumInputs> addrs{};
  std::array<int64_t, kNumInputs * 4> dims{}, strides{};
  std::array<int32_t, kNumInputs> dtypes{}, dimNums{};
  for (int i = 0; i < kNumInputs; ++i) {
    if (ts[i] == nullptr) {
      dimNums[i] = 0;
      continue;
    }
    addrs[i] = ts[i]->data_ptr();
    dimNums[i] = static_cast<int32_t>(ts[i]->dim());
    dtypes[i] = AclDtype(*ts[i]);
    for (int64_t d = 0; d < ts[i]->dim(); ++d) {
      dims[i * 4 + d] = ts[i]->size(d);
      strides[i * 4 + d] = ts[i]->stride(d);
    }
  }
  // query_start_loc defines the varlen split; without it the kernel would
  // treat the whole token run as one sequence.
  TORCH_CHECK(ts[kIdxQueryStartLoc] != nullptr,
              "query_start_loc is required in packed mode");

  std::array<int64_t, 4> outDims{}, outStrides{};
  const int32_t outDimNum = static_cast<int32_t>(out.dim());
  for (int64_t d = 0; d < out.dim(); ++d) {
    outDims[d] = out.size(d);
    outStrides[d] = out.stride(d);
  }
  const int32_t outDtype = AclDtype(out);
  // torch_npu 2.12 exposes the stream helper in namespace c10_npu
  void *stream = c10_npu::getCurrentNPUStream().stream();

  const int rc = causal_conv1d_qkv_run(
      addrs.data(), dims.data(), strides.data(), dtypes.data(), dimNums.data(),
      activation_mode, pad_slot_id, run_mode, q_dim, k_dim, v_dim, head_dim,
      out.data_ptr(), outDims.data(), outStrides.data(), outDtype, outDimNum,
      stream);
  TORCH_CHECK(rc == 0, "causal_conv1d_qkv_run failed with code ", rc);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("causal_conv1d_qkv", &causal_conv1d_qkv,
        "CausalConv1dQkv packed-QKV prefill eager entry");
}
