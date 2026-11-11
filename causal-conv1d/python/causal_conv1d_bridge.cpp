// Eager bridge, JIT-compiled by python/causal_conv1d_binding.py: torch
// tensors -> causal_conv1d_run (flat C ABI of libcust_opapi.so).
// Contract mirrors the GE wrapper: dense FN prefill, fresh output tensor.
#include <array>
#include <cstdint>
#include <string>

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <acl/acl.h>

extern "C" int causal_conv1d_run(void *const *addrs, const int64_t *dims,
                                 const int64_t *strides,
                                 const int32_t *dtypes,
                                 const int32_t *dim_nums,
                                 const char *activation, int64_t head_num,
                                 void *out_addr, const int64_t *out_dims,
                                 const int64_t *out_strides, int32_t out_dtype,
                                 int32_t out_dim_num, void *stream);

namespace {

int32_t AclDtype(const at::Tensor &t) {
  switch (t.scalar_type()) {
    case at::kBFloat16: return ACL_BF16;
    case at::kHalf: return ACL_FLOAT16;
    default: return -1;
  }
}

constexpr int kNumInputs = 3;  // x, weight, bias(optional)

const char *const kNames[kNumInputs] = {"x", "weight", "bias"};

}  // namespace

at::Tensor causal_conv1d(at::Tensor x, at::Tensor weight,
                         std::optional<at::Tensor> bias,
                         std::string activation, int64_t head_num) {
  const at::Tensor *const ts[kNumInputs] = {
      &x, &weight, bias.has_value() ? &bias.value() : nullptr};
  for (int i = 0; i < kNumInputs; ++i) {
    if (ts[i] == nullptr) {
      continue;
    }
    const at::Tensor &t = *ts[i];
    TORCH_CHECK(t.device() == x.device(), kNames[i], " must be on ", x.device());
    TORCH_CHECK(t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kHalf,
                kNames[i], " must be bf16/fp16 (got ",
                toString(t.scalar_type()), ")");
    TORCH_CHECK(t.is_contiguous(), kNames[i], " must be contiguous");
    TORCH_CHECK(t.dim() >= 1 && t.dim() <= 4, kNames[i], " must be 1-4D");
  }
  TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be (T,D) or (B,S,D)");
  TORCH_CHECK(weight.dim() == 2, "weight must be (W,D)");
  TORCH_CHECK(weight.size(1) == x.size(-1), "weight D must match x last dim");
  if (ts[kNumInputs - 1] != nullptr) {
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == x.size(-1),
                "bias must be (D,)");
  }
  // Bind the calling thread to the tensors' device: python worker threads do
  // not inherit torch.npu.set_device, and getCurrentNPUStream must follow the
  // input device.
  const c10::DeviceGuard device_guard(x.device());

  at::Tensor out = at::empty_like(x);

  std::array<void *, kNumInputs> addrs{x.data_ptr(), weight.data_ptr(),
                                       ts[2] != nullptr ? ts[2]->data_ptr()
                                                        : nullptr};
  std::array<int64_t, kNumInputs * 4> dims{}, strides{};
  std::array<int32_t, kNumInputs> dtypes{}, dimNums{};
  for (int i = 0; i < kNumInputs; ++i) {
    if (ts[i] == nullptr) {
      dimNums[i] = 0;
      continue;
    }
    dimNums[i] = static_cast<int32_t>(ts[i]->dim());
    dtypes[i] = AclDtype(*ts[i]);
    for (int64_t d = 0; d < ts[i]->dim(); ++d) {
      dims[i * 4 + d] = ts[i]->size(d);
      strides[i * 4 + d] = ts[i]->stride(d);
    }
  }
  std::array<int64_t, 4> outDims{}, outStrides{};
  const int32_t outDimNum = static_cast<int32_t>(out.dim());
  for (int64_t d = 0; d < out.dim(); ++d) {
    outDims[d] = out.size(d);
    outStrides[d] = out.stride(d);
  }
  const int32_t outDtype = AclDtype(out);
  // torch_npu 2.12 exposes the stream helper in namespace c10_npu
  void *stream = c10_npu::getCurrentNPUStream().stream();

  const int rc = causal_conv1d_run(
      addrs.data(), dims.data(), strides.data(), dtypes.data(), dimNums.data(),
      activation.c_str(), head_num, out.data_ptr(), outDims.data(),
      outStrides.data(), outDtype, outDimNum, stream);
  TORCH_CHECK(rc == 0, "causal_conv1d_run failed with code ", rc);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("causal_conv1d", &causal_conv1d,
        "CausalConv1d dense FN prefill eager entry");
}
