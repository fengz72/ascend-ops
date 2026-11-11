"""Minimal functional test: CPU golden vs eager NPU vs torchair compiled graph.

Usage:
    python3 test/test_causal_conv1d.py [device]

Requires: CANN env sourced (set_env.sh) and `bash install.sh` (builds,
installs the operator and registers the binding). The activator locates
the .so and mounts the vendor package itself; no shell exports are needed.

Gates (mega-gdn-prefill 口径): graph-vs-eager rel_l2 < 0.001, eager-vs-golden
cosine > 0.95 and rel_l2 <= 0.02. The bf16 kernel's global cosine vs the fp32
golden dips slowly with batch (0.9997@B1, 0.9985@B3) while rel_l2 stays at
0.0017 — verified bit-identical to the upstream fla-npu wheel on the same
inputs, i.e. kernel-inherent bf16 behavior, not a wrapper defect. 2D (T,D)
input is not covered: the upstream kernel rejects it with 561002 (same from
the fla wheel); the model integration only feeds 3D (B,S,D). Cases cover the
startlux decision model's GDN conv shapes (D=6144, W=4, silu, bf16), a
small-D variant, an activation="none" + bias variant and an fp16 variant.
"""
import sys

import torch

W, D = 4, 6144


def build_inputs(shape, seed=42, dtype="bf16"):
    gen = torch.Generator().manual_seed(seed)
    torch_type = torch.bfloat16 if dtype == "bf16" else torch.float16
    return (torch.randn(*shape, generator=gen) * 0.3).to(torch_type), \
        (torch.randn(W, shape[-1], generator=gen) * 0.05).to(torch_type)


def cpu_golden(x, weight, activation, bias=None):
    """Model's causal conv math in fp32: depthwise conv, left context zero."""
    import torch.nn.functional as F
    xf = x.float()
    keep_2d = xf.dim() == 2
    if keep_2d:
        xf = xf.unsqueeze(0)
    xf = xf.transpose(1, 2)  # (B,S,D) -> (B,D,S)
    channels, seq = xf.shape[1], xf.shape[2]
    wf = weight.float().t().reshape(channels, 1, W)
    y = F.conv1d(F.pad(xf, (W - 1, 0)), wf, groups=channels,
                 bias=None if bias is None else bias.float())
    if activation == "silu":
        y = F.silu(y)
    y = y.transpose(1, 2)
    return y.squeeze(0) if keep_2d else y


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def cosine(a, b):
    u, v = a.float().flatten(), b.float().flatten()
    return (torch.dot(u, v) / (u.norm() * v.norm())).item()


def main():
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dev = f"npu:{device}"

    import torch_npu
    torch.npu.set_device(device)
    import causal_conv1d  # noqa: F401
    from torch_npu.dynamo.torchair import CompilerConfig, get_npu_backend

    cases = [
        # (name, x_shape, activation, with_bias, dtype)
        ("startlux-team (1,99,6144)", (1, 99, D), "silu", False, "bf16"),
        ("startlux-b3 (3,99,6144)", (3, 99, D), "silu", False, "bf16"),
        ("startlux-image (1,130,6144)", (1, 130, D), "silu", False, "bf16"),
        ("startlux-urgent (1,85,6144)", (1, 85, D), "silu", False, "bf16"),
        ("small-D (2,64,2048)", (2, 64, 2048), "silu", False, "bf16"),
        ("no activation + bias", (2, 64, 2048), "none", True, "bf16"),
        ("fp16", (1, 99, D), "silu", False, "fp16"),
    ]

    config = CompilerConfig()
    config.mode = "max-autotune"
    model = torch.nn.Module()
    model.forward = lambda *a: torch.ops.ascend_ops.causal_conv1d(*a)
    compiled = torch.compile(model.eval(),
                             backend=get_npu_backend(compiler_config=config),
                             dynamic=True)

    for name, shape, activation, with_bias, dtype in cases:
        torch.manual_seed(42)
        x, weight = build_inputs(shape, dtype=dtype)
        bias = None
        if with_bias:
            bias = (torch.randn(shape[-1]) * 0.05).to(
                torch.bfloat16 if dtype == "bf16" else torch.float16)
        golden = cpu_golden(x, weight, activation, bias)

        dx = x.to(dev)
        dw = weight.to(dev)
        db = bias.to(dev) if bias is not None else None
        args = (dx, dw, db, activation, 0)
        eager = torch.ops.ascend_ops.causal_conv1d(*args).cpu()
        with torch.no_grad():
            graph = compiled(*args).cpu()

        r_graph = rel_l2(graph, eager)
        r_eager = rel_l2(eager, golden)
        c_eager = cosine(eager, golden)
        print(f"{name:<32} eager: cos={c_eager:.6f} rel_l2={r_eager:.6f} | "
              f"graph-vs-eager rel_l2={r_graph:.8f}")
        assert c_eager > 0.95, f"{name}: eager vs golden cosine FAIL {c_eager}"
        assert r_eager <= 0.02, f"{name}: eager vs golden rel_l2 FAIL {r_eager}"
        assert r_graph < 0.001, f"{name}: graph vs eager FAIL {r_graph}"

    print("PASS")


if __name__ == "__main__":
    main()
