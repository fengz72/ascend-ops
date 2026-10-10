#!/usr/bin/env python3
"""Dump one DN case (inputs + python reference output) for the C++ frame probe.

Writes probe/case_data/{q,k,v,bounds,out_py}.bin + meta.txt (KEY VALUE lines).
Case: focusu p20/130x10 fp16, seed 71000 (same as tests/production perf case).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DSL_ROOT = HERE.parents[1]
sys.path.insert(0, str(DSL_ROOT))

import torch
import torch_npu  # noqa: F401

from prefix_attn import prefix_attn_dn
from prefix_attn.metadata import get_effective_core_counts


def main():
    out_dir = HERE / "case_data"
    out_dir.mkdir(exist_ok=True)
    bounds = [20]
    acc = 20
    for _ in range(10):
        acc += 130
        bounds.append(acc)
    n1, n2, d = 14, 2, 64
    gen = torch.Generator().manual_seed(71000)
    t = bounds[-1]
    q = torch.randn((t, n1, d), generator=gen).half()
    k = torch.randn((t, n2, d), generator=gen).half()
    v = torch.randn((t, n2, d), generator=gen).half()
    bn = torch.tensor(bounds, dtype=torch.int32, device="npu")
    out = prefix_attn_dn(q.npu(), k.npu(), v.npu(), bn)
    torch.npu.synchronize()
    aic, _ = get_effective_core_counts()
    (out_dir / "q.bin").write_bytes(q.numpy().tobytes())
    (out_dir / "k.bin").write_bytes(k.numpy().tobytes())
    (out_dir / "v.bin").write_bytes(v.numpy().tobytes())
    (out_dir / "bounds.bin").write_bytes(
        torch.tensor(bounds, dtype=torch.int32).numpy().tobytes())
    (out_dir / "out_py.bin").write_bytes(out.cpu().numpy().tobytes())
    (out_dir / "meta.txt").write_text(
        f"T {t}\nN1 {n1}\nN2 {n2}\nD {d}\nBOUNDS_LEN {len(bounds)}\n"
        f"SCALE {1.0 / 8:.17g}\nAIC {aic}\nDTYPE float16\n")
    print(f"dumped case to {out_dir} (T={t}, aic={aic})")


if __name__ == "__main__":
    main()
