#!/usr/bin/env python3
"""Dump the codegen'd AscendC kernel sources for the GE deliverable build.

This is the ONLY step that needs cannbotdsl: it triggers the DSL compile of
the macro kernel (fp16 + bf16 specializations) with CANNBOTDSL_DUMP_ASCENDC
and copies the generated .asc into codegen/, plus the AICPU metadata planner
sources.  The GE deliverable (CMake + bisheng) then compiles these checked-in
sources WITHOUT cannbotdsl.

Run on the target board from the cannbot-dsl directory:

    python ge_custom_op/scripts/dump_kernel_src.py

Outputs (relative to cannbot-dsl/):
  codegen/launch_prefix_macro_fp16.asc   macro kernel, float16 specialization
  codegen/launch_prefix_macro_bf16.asc   macro kernel, bfloat16 specialization
  codegen/aicpu/*                        AICPU metadata planner sources (4)
  ge_custom_op/src/artifacts.h           generated: .so names + host ABI symbols
  codegen/SOURCE_INFO.json               provenance (sha256, versions, commands)

fp16/bf16 device kernel symbols collide (same kernel__PrefixMacroKernel name),
so each dtype is a separate .so; the host invoke/schema symbols carry a
per-specialization hash recorded in artifacts.h.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # ge_custom_op/scripts
GE_DIR = HERE.parent                            # ge_custom_op/
DSL_ROOT = GE_DIR.parent                        # cannbot-dsl/
sys.path.insert(0, str(DSL_ROOT))

CODEGEN = DSL_ROOT / "codegen"
AICPU = CODEGEN / "aicpu"
HEADER = GE_DIR / "src" / "artifacts.h"

N1, N2, HEAD_DIM = 14, 2, 64
BOUNDS = [20] + [20 + 130 * (i + 1) for i in range(10)]  # p20/130x10, T=1320
SEED = 71000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_macro(dtype_name, torch_dtype):
    """Compile one dtype specialization with DUMP_ASCENDC and return its .asc."""
    import torch
    import torch_npu  # noqa: F401
    from prefix_attn import prefix_attention
    with tempfile.TemporaryDirectory(prefix="dump_asc_") as td:
        os.environ["CANNBOTDSL_DUMP_ASCENDC"] = "1"
        os.environ["CANNBOTDSL_DUMP_DIR"] = td
        gen = torch.Generator().manual_seed(SEED)
        t = BOUNDS[-1]
        q = torch.randn((t, N1, HEAD_DIM), generator=gen).to(torch_dtype)
        k = torch.randn((t, N2, HEAD_DIM), generator=gen).to(torch_dtype)
        v = torch.randn((t, N2, HEAD_DIM), generator=gen).to(torch_dtype)
        bn = torch.tensor(BOUNDS, dtype=torch.int32, device="npu")
        prefix_attention(q.npu(), k.npu(), v.npu(), bn)
        torch.npu.synchronize()
        os.environ.pop("CANNBOTDSL_DUMP_ASCENDC", None)
        os.environ.pop("CANNBOTDSL_DUMP_DIR", None)
        dumped = list(Path(td).glob("launch_prefix_macro*.asc"))
        assert len(dumped) == 1, dumped
        dst = CODEGEN / f"launch_prefix_macro_{dtype_name}.asc"
        shutil.copyfile(dumped[0], dst)
    text = dst.read_text()
    invoke = re.search(r"(_launch_prefix_macro_[0-9a-f]+__cannbot_invoke_v9)", text)
    schema = re.search(r"(_launch_prefix_macro_[0-9a-f]+__cannbot_schema_v2)", text)
    assert invoke and schema, f"host ABI symbols not found in {dst}"
    return {"file": dst.name, "invoke_symbol": invoke.group(1),
            "schema_symbol": schema.group(1), "sha256": sha256(dst)}


def dump_aicpu():
    from prefix_attn.metadata import _prefix_attn_metadata_kernel as k
    AICPU.mkdir(parents=True, exist_ok=True)
    for old in AICPU.iterdir():
        old.unlink()
    files = k.generate_sources(launch_mode="interface")
    kept = []
    for fname, text in files.items():
        if fname.endswith("_x86.cpp"):   # host-sim variant, not built
            continue
        (AICPU / fname).write_text(text)
        kept.append(fname)
    launch = [f for f in kept if f.endswith("_shim.asc")][0]
    sym = re.search(r"extern \"C\" void (\w+_do)\(", (AICPU / launch).read_text())
    return {"files": sorted(kept), "launch_symbol": sym.group(1),
            "sha256": {f: sha256(AICPU / f) for f in sorted(kept)}}


def main():
    import torch
    import torch_npu  # noqa: F401
    import cannbotdsl
    from prefix_attn.metadata import (AIC_CORE_NUM, get_effective_core_counts,
                                      metadata_padded_count)
    CODEGEN.mkdir(parents=True, exist_ok=True)
    # remove the old single-dtype snapshot name if present
    (CODEGEN / "launch_prefix_macro.asc").unlink(missing_ok=True)

    info = {"kernel": {}, "aicpu": None}
    for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        print(f"dumping macro kernel {name} ...", flush=True)
        info["kernel"][name] = dump_macro(name, dt)
    print("dumping AICPU metadata sources ...", flush=True)
    info["aicpu"] = dump_aicpu()

    props = torch.npu.get_device_properties(0)
    aic = get_effective_core_counts()[0]
    source_info = {
        "description": ("codegen'd AscendC sources for the GE deliverable build. "
                        "The macro kernel (fp16/bf16 .asc) and the AICPU metadata "
                        "planner (aicpu/) are compiled by CMake+bisheng WITHOUT "
                        "cannbotdsl; this dump step is the only cannbotdsl user."),
        "generated_by": {
            "cannbotdsl_version": getattr(cannbotdsl, "__version__", "0.7.0"),
            "build_info": json.loads((Path(cannbotdsl.__file__).parent
                                      / "_build_info.json").read_text()),
            "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "command": "python ge_custom_op/scripts/dump_kernel_src.py",
        },
        "device": {"name": props.name, "cube_core_num": props.cube_core_num,
                   "vector_core_num": props.vector_core_num,
                   "effective_aic": aic},
        "kernel_config": {"n1": N1, "n2": N2, "head_dim": HEAD_DIM,
                          "aic_core_cap": AIC_CORE_NUM,
                          "metadata_padded_count": metadata_padded_count()},
        "compile_case": {"dtype": "float16/bfloat16", "prefix": 20,
                         "private_length": 130, "batches": 10,
                         "total_tokens": BOUNDS[-1], "seed": SEED},
        "sources_sha256": {p.name: sha256(p) for p in
                           sorted((DSL_ROOT / "prefix_attn").glob("*.py"))},
        "kernel": info["kernel"], "aicpu": info["aicpu"],
        "notes": [
            "fp16/bf16 device kernel symbols collide (kernel__PrefixMacroKernel___call___0); each dtype compiles to a separate .so.",
            "The .asc tail carries the host ABI v9 invoke/schema entries; dsl_launcher dlopens them by the hashed symbols in artifacts.h.",
            "AICPU snapshot keeps the 4 production files (no _x86.cpp host-sim variant).",
        ],
    }
    (CODEGEN / "SOURCE_INFO.json").write_text(
        json.dumps(source_info, indent=2, ensure_ascii=False) + "\n")

    kf16, kb16, aicpu = info["kernel"]["fp16"], info["kernel"]["bf16"], info["aicpu"]
    HEADER.write_text(
        "// Generated by ge_custom_op/scripts/dump_kernel_src.py - DO NOT EDIT.\n"
        "// Codegen'd AscendC source file names + host ABI symbols of the macro\n"
        "// kernel specializations and the AICPU metadata planner.\n"
        "#pragma once\n\n"
        f'#define PREFIX_ATTN_ASC_FP16   "{kf16["file"]}"\n'
        f'#define PREFIX_ATTN_ASC_BF16   "{kb16["file"]}"\n'
        f'#define PREFIX_ATTN_SO_FP16    "kernel_fp16.so"\n'
        f'#define PREFIX_ATTN_SO_BF16    "kernel_bf16.so"\n'
        f'#define PREFIX_ATTN_INVOKE_FP16 "{kf16["invoke_symbol"]}"\n'
        f'#define PREFIX_ATTN_INVOKE_BF16 "{kb16["invoke_symbol"]}"\n'
        f'#define PREFIX_ATTN_SCHEMA_FP16 "{kf16["schema_symbol"]}"\n'
        f'#define PREFIX_ATTN_SCHEMA_BF16 "{kb16["schema_symbol"]}"\n'
        f'#define PREFIX_ATTN_AICPU_SO     "aicpu_metadata.so"\n'
        f'#define PREFIX_ATTN_AICPU_LAUNCH "{aicpu["launch_symbol"]}"\n'
        f'#define PREFIX_ATTN_METADATA_ELEMS {source_info["kernel_config"]["metadata_padded_count"]}\n'
        f'#define PREFIX_ATTN_AIC_CAP {AIC_CORE_NUM}\n'
        f'#define PREFIX_ATTN_AIC_DEFAULT {props.cube_core_num}\n')
    print(f"done. codegen sources -> {CODEGEN}")
    print(f"  fp16 invoke: {kf16['invoke_symbol']}")
    print(f"  bf16 invoke: {kb16['invoke_symbol']}")
    print(f"  aicpu launch: {aicpu['launch_symbol']}")


if __name__ == "__main__":
    main()
