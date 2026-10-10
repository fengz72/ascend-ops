#!/usr/bin/env python3
"""Export prebuilt DSL kernel artifacts for the GE custom op (PrefixAttention).

Run on the target board from the cannbot-dsl directory:

    python ge_custom_op/scripts/export_artifacts.py

Outputs (paths relative to ge_custom_op/):
  opp/op_graph/lib/linux/aarch64/artifacts/
    kernel_fp16.so / kernel_bf16.so  macro attention kernel AOT artifacts (@host compile)
    aicpu_metadata.so            AICPU metadata planner (interface launch mode)
    MANIFEST.json                sha256, symbols, versions, export command
  src/artifacts.h                generated C header (file names + symbol names)

The kernel artifacts expose the self-describing host ABI v9:
  <base>__cannbot_schema_v2()  -> CannBotHostSchemaV2* (frame layout/digest)
  <base>__cannbot_invoke_v9(stream, frame, bytes) -> int32 status
The AICPU .so exposes the flat C ABI:
  _prefix_attn_metadata_kernel_do(stream, bounds, metadata, len, n1, n2, aic, macro_cost)
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # ge_custom_op/scripts
GE_DIR = HERE.parent                            # ge_custom_op/
DSL_ROOT = GE_DIR.parent                        # cannbot-dsl/
sys.path.insert(0, str(DSL_ROOT))

ART_DIR = GE_DIR / "opp/op_graph/lib/linux/aarch64/artifacts"
HEADER = GE_DIR / "src/artifacts.h"

N1, N2, HEAD_DIM = 14, 2, 64


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exported_symbols(so_path, pattern):
    out = subprocess.check_output(["nm", "-D", "--defined-only", str(so_path)],
                                  text=True)
    return sorted(l.split()[-1] for l in out.splitlines()
                  if pattern in l and " T " in l)


def export_kernel(dtype_name, dtype):
    from prefix_attn.attention import _compiled_launch
    compiled = _compiled_launch(N1, N2, dtype)
    so = ART_DIR / f"kernel_{dtype_name}.so"
    shutil.copyfile(compiled.so_path, so)
    symbols = exported_symbols(so, "__cannbot_")
    invoke = [s for s in symbols if s.endswith("__cannbot_invoke_v9")]
    schema = [s for s in symbols if s.endswith("__cannbot_schema_v2")]
    assert len(invoke) == 1 and len(schema) == 1, symbols
    deps = subprocess.check_output(["ldd", str(so)], text=True)
    return {"file": so.name, "invoke_symbol": invoke[0],
            "schema_symbol": schema[0], "sha256": sha256(so),
            "provider_symbol_name": compiled.symbol_name,
            "host_abi_version": compiled.host_abi_version,
            "ldd": [l.strip() for l in deps.splitlines() if "=>" in l]}


def export_aicpu():
    from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel
    from prefix_attn.metadata import _prefix_attn_metadata_kernel
    with tempfile.TemporaryDirectory(prefix="export_aicpu_") as td:
        compiled = compile_aicpu_kernel(_prefix_attn_metadata_kernel,
                                        workdir=td, launch_mode="interface")
        so = ART_DIR / "aicpu_metadata.so"
        shutil.copyfile(compiled.so_path, so)
    symbols = exported_symbols(so, "_do")
    assert len(symbols) == 1, symbols
    return {"file": so.name, "launch_symbol": symbols[0], "sha256": sha256(so)}


def main():
    import cannbotdsl
    import torch
    import torch_npu  # noqa: F401
    from cannbotdsl import dtypes
    from prefix_attn.metadata import (AIC_CORE_NUM, get_effective_core_counts,
                                      metadata_padded_count)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    HEADER.parent.mkdir(parents=True, exist_ok=True)
    for old in ART_DIR.glob("*.so"):
        old.unlink()

    manifest = {
        "generated_by": {
            "cannbotdsl_version": getattr(cannbotdsl, "__version__", "0.7.0"),
            "build_info": json.loads(
                (Path(cannbotdsl.__file__).parent / "_build_info.json")
                .read_text()) if (Path(cannbotdsl.__file__).parent
                                  / "_build_info.json").exists() else {},
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "command": "python ge_custom_op/scripts/export_artifacts.py",
        },
        "kernel_config": {"n1": N1, "n2": N2, "head_dim": HEAD_DIM,
                          "aic_core_cap": AIC_CORE_NUM,
                          "metadata_padded_count": metadata_padded_count()},
        "sources_sha256": {p.name: sha256(p) for p in
                           sorted((DSL_ROOT / "prefix_attn").glob("*.py"))},
        "kernel": {},
        "aicpu": None,
    }
    props = torch.npu.get_device_properties(0)
    manifest["device"] = {"name": props.name, "cube_core_num": props.cube_core_num,
                          "vector_core_num": props.vector_core_num}
    for name, dt in (("fp16", dtypes.float16), ("bf16", dtypes.bfloat16)):
        print(f"exporting kernel {name} ...", flush=True)
        manifest["kernel"][name] = export_kernel(name, dt)
    print("exporting AICPU metadata ...", flush=True)
    manifest["aicpu"] = export_aicpu()
    (ART_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    k16, kb, aicpu = manifest["kernel"]["fp16"], manifest["kernel"]["bf16"], manifest["aicpu"]
    HEADER.write_text(
        "// Generated by ge_custom_op/scripts/export_artifacts.py - DO NOT EDIT.\n"
        "// Artifact file names and symbol names of the prebuilt DSL kernels.\n"
        "#pragma once\n\n"
        f'#define PREFIX_ATTN_SO_FP16   "{k16["file"]}"\n'
        f'#define PREFIX_ATTN_SO_BF16   "{kb["file"]}"\n'
        f'#define PREFIX_ATTN_INVOKE_FP16 "{k16["invoke_symbol"]}"\n'
        f'#define PREFIX_ATTN_INVOKE_BF16 "{kb["invoke_symbol"]}"\n'
        f'#define PREFIX_ATTN_SCHEMA_FP16 "{k16["schema_symbol"]}"\n'
        f'#define PREFIX_ATTN_SCHEMA_BF16 "{kb["schema_symbol"]}"\n'
        f'#define PREFIX_ATTN_AICPU_SO     "{aicpu["file"]}"\n'
        f'#define PREFIX_ATTN_AICPU_LAUNCH "{aicpu["launch_symbol"]}"\n'
        f'#define PREFIX_ATTN_METADATA_ELEMS {manifest["kernel_config"]["metadata_padded_count"]}\n'
        f'#define PREFIX_ATTN_AIC_CAP {manifest["kernel_config"]["aic_core_cap"]}\n'
        f'#define PREFIX_ATTN_AIC_DEFAULT {manifest["device"]["cube_core_num"]}\n')
    print(f"done: {ART_DIR}")
    for name in ("fp16", "bf16"):
        print(f"  kernel_{name}: {manifest['kernel'][name]['invoke_symbol']}")
    print(f"  aicpu: {manifest['aicpu']['launch_symbol']}")
    print("  ldd(kernel_fp16):")
    for line in manifest["kernel"]["fp16"]["ldd"]:
        print("   ", line)


if __name__ == "__main__":
    main()
