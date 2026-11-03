"""Minimal functional test: CPU golden vs eager NPU vs torchair compiled graph.

Usage:
    python3 test/test_mega_gdn_prefill.py [device]

Requires: CANN env sourced (set_env.sh) and `bash install.sh` (builds,
installs the operator and registers the binding). The activator locates
the .so and mounts the vendor package itself; no shell exports are needed.
"""
import sys

NV, DK, DV, C = 16, 128, 128, 6144
NAMES = ["mixed_qkv", "b", "a", "z", "conv_weight", "conv_state", "a_log",
         "dt_bias", "conv_state_read_indices", "conv_state_write_indices",
         "ssm_state_read_indices", "ssm_state_write_indices", "ssm_cache",
         "mask_lower", "mask_full", "minus_identity", "cu_seqlens",
         "norm_weight"]


def build_inputs(tokens, seed=42):
    gen = torch.Generator().manual_seed(seed)
    return {
        "mixed_qkv": (torch.randn(tokens, C, generator=gen) * 0.3).bfloat16(),
        "b": (torch.randn(tokens, NV, generator=gen) * 0.4).bfloat16(),
        "a": (torch.randn(tokens, NV, generator=gen) * 0.4).bfloat16(),
        "z": (torch.randn(tokens, NV, DV, generator=gen) * 0.3).bfloat16(),
        "conv_weight": (torch.randn(4, C, generator=gen) * 0.05).bfloat16(),
        "conv_state": torch.zeros(4, 3, C, dtype=torch.bfloat16),
        "a_log": torch.linspace(-2.0, 0.25, NV),
        "dt_bias": torch.linspace(-0.75, 0.5, NV),
        "conv_state_read_indices": torch.tensor([-1], dtype=torch.int32),
        "conv_state_write_indices": torch.tensor([0], dtype=torch.int32),
        "ssm_state_read_indices": torch.tensor([-1], dtype=torch.int32),
        "ssm_state_write_indices": torch.tensor([0], dtype=torch.int32),
        "ssm_cache": torch.zeros(4, NV, DV, DK),
        "mask_lower": torch.tril(torch.ones(128, 128), diagonal=-1),
        "mask_full": torch.tril(torch.ones(128, 128)),
        "minus_identity": torch.zeros(128, 128, dtype=torch.bfloat16).fill_diagonal_(-1),
        "cu_seqlens": torch.tensor([0, tokens], dtype=torch.int32),
        "norm_weight": (1.0 + torch.randn(DK, generator=gen) * 0.05).bfloat16(),
    }


def cpu_golden(case):
    """CPU FP32 golden: causal conv + L2 norm + gates + delta rule + RMSNorm."""
    T = case["mixed_qkv"].shape[0]
    x = case["mixed_qkv"].float().t().unsqueeze(0)
    w = case["conv_weight"].float().t().reshape(C, 1, 4)
    mixed = torch.nn.functional.silu(
        torch.nn.functional.conv1d(
            torch.nn.functional.pad(x, (3, 0)), w, groups=C)).squeeze(0).t()

    q = mixed[:, :C // 3].reshape(T, NV, DK)
    k = mixed[:, C // 3:2 * C // 3].reshape(T, NV, DK)
    v = mixed[:, 2 * C // 3:].reshape(T, NV, DV)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(case["b"].float())
    g = -torch.exp(case["a_log"].float()) * torch.nn.functional.softplus(
        case["a"].float() + case["dt_bias"].float(), threshold=20.0)

    S = torch.zeros(NV, DK, DV)
    out = torch.zeros(T, NV, DV)
    for t in range(T):
        S = S * torch.exp(g[t]).view(-1, 1, 1)
        pred = torch.einsum("hdv,hd->hv", S, k[t])
        S = S + beta[t].view(-1, 1, 1) * k[t].unsqueeze(-1) * \
            (v[t] - pred).unsqueeze(1)
        out[t] = torch.einsum("hd,hdv->hv", q[t], S)

    normed = out / torch.sqrt((out * out).mean(-1, keepdim=True) + 1e-6)
    normed = normed * case["norm_weight"].float()
    return (normed * torch.nn.functional.silu(case["z"].float())).to(torch.bfloat16)


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dev = f"npu:{device}"

    import torch
    import torch_npu
    torch.npu.set_device(device)
    import mega_gdn_prefill  # noqa: F401
    from torch_npu.dynamo.torchair import CompilerConfig, get_npu_backend
    globals()["torch"] = torch

    case = build_inputs(tokens=64)
    dev_inputs = {n: case[n].to(dev).contiguous() for n in NAMES}

    golden = cpu_golden(case)
    # eager leg through the torch op (PrivateUse1 -> JIT bridge): z is
    # overwritten in place and returned as an alias (schema Tensor(a!))
    z_pristine = dev_inputs["z"].clone()
    z_in = z_pristine.clone()
    args = [z_in if n == "z" else dev_inputs[n] for n in NAMES]
    eager_t = torch.ops.ascend_ops.mega_gdn_prefill(*args)
    assert eager_t.data_ptr() == z_in.data_ptr(), \
        "eager output must alias z (in-place contract)"
    assert not torch.equal(z_in, z_pristine), \
        "eager must overwrite z in place (Tensor(a!) contract)"
    eager = eager_t.cpu()

    config = CompilerConfig()
    config.mode = "max-autotune"
    model = torch.nn.Module()
    model.forward = lambda *a: torch.ops.ascend_ops.mega_gdn_prefill(*a)
    compiled = torch.compile(model.eval(),
                             backend=get_npu_backend(compiler_config=config),
                             dynamic=True)
    with torch.no_grad():
        graph = compiled(*[dev_inputs[n] for n in NAMES]).cpu()

    r_graph = rel_l2(graph, eager)
    c_eager = torch.dot(eager.float().flatten(), golden.float().flatten()) / \
        (eager.float().norm() * golden.float().norm()).item().__float__()
    print(f"eager vs CPU golden: cos = {c_eager:.6f} (kernel GDR is FP16)")
    print(f"graph vs eager:      rel_l2 = {r_graph:.8f}")
    assert c_eager > 0.95, f"eager vs golden FAIL: cos = {c_eager}"
    assert r_graph < 0.001, f"graph vs eager FAIL: {r_graph}"
    print("PASS")


if __name__ == "__main__":
    main()
