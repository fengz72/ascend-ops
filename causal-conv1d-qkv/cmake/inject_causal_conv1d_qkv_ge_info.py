#!/usr/bin/env python3
"""Add the CausalConv1dQkvGE entry to a generated aic-<soc>-ops-info.json.

CausalConv1dQkvGE is registered through IMPL_OP + OpExecuteFunc (the aclnn
lowering route), and that route resolves its engine from the per-SoC ops-info
JSON. The generator only emits entries for the vendored OpDefs, so without this
entry GE graph compile fails with EZ3003 ("No supported Ops kernel and engine
are found") while eager keeps working — a graph-mode-only breakage.

The entry must keep aclnnSupport (without it GE ignores the whole entry) and one
dtype/format element per output dtype configuration (single-element lists are
rejected for these inputs).

Usage:
    python3 inject_causal_conv1d_qkv_ge_info.py <ops-info.json> [more.json ...]
    python3 inject_causal_conv1d_qkv_ge_info.py --check <ops-info.json> [...]
"""

import json
import os
import sys

OP_NAME = "CausalConv1dQkvGE"

# Kept literally as verified on chip: descriptors mirror REG_OP(CausalConv1dQkvGE)
# input/output order, with the two dtype configs the vendored OpDef declares.
_GE_ENTRY = {
    "coreType": {"value": "AiCore"},
    "dynamicCompileStatic": {"flag": "true"},
    "dynamicFormat": {"flag": "false"},
    "dynamicRankSupport": {"flag": "true"},
    "dynamicShapeSupport": {"flag": "true"},
    "needCheckSupport": {"flag": "false"},
    "precision_reduce": {"flag": "true"},
    "aclnnSupport": {"value": "support_aclnn"},
    "input0": {"dtype": "bfloat16,bfloat16", "format": "ND,ND", "name": "x",
               "paramType": "required", "shape": "all"},
    "input1": {"dtype": "bfloat16,bfloat16", "format": "ND,ND", "name": "weight",
               "paramType": "required", "shape": "all"},
    "input2": {"dtype": "bfloat16,bfloat16", "format": "ND,ND",
               "name": "conv_states", "paramType": "required", "shape": "all"},
    "input3": {"dtype": "int64,int64", "format": "ND,ND",
               "name": "query_start_loc", "paramType": "optional",
               "shape": "all", "valueDepend": "optional"},
    "input4": {"dtype": "int64,int64", "format": "ND,ND", "name": "cache_indices",
               "paramType": "optional", "shape": "all",
               "valueDepend": "optional"},
    "input5": {"dtype": "int64,int64", "format": "ND,ND",
               "name": "initial_state_mode", "paramType": "optional",
               "shape": "all", "valueDepend": "optional"},
    "input6": {"dtype": "int64,int64", "format": "ND,ND",
               "name": "num_accepted_tokens", "paramType": "optional",
               "shape": "all", "valueDepend": "optional"},
    "input7": {"dtype": "bfloat16,bfloat16", "format": "ND,ND", "name": "bias",
               "paramType": "optional", "shape": "all"},
    "output0": {"dtype": "float16,bfloat16", "format": "ND,ND", "name": "y",
                "paramType": "required", "shape": "all"},
    "attr": {"list": "qDim,kDim,vDim,headDim,activationMode,padSlotId,runMode,"
                     "out_dtype"},
    "attr_qDim": {"defaultValue": "1024", "paramType": "optional", "type": "int",
                  "value": "all"},
    "attr_kDim": {"defaultValue": "1024", "paramType": "optional", "type": "int",
                  "value": "all"},
    "attr_vDim": {"defaultValue": "3072", "paramType": "optional", "type": "int",
                  "value": "all"},
    "attr_headDim": {"defaultValue": "128", "paramType": "optional",
                     "type": "int", "value": "all"},
    "attr_activationMode": {"defaultValue": "2", "paramType": "optional",
                            "type": "int", "value": "all"},
    "attr_padSlotId": {"defaultValue": "-1", "paramType": "optional",
                       "type": "int", "value": "all"},
    "attr_runMode": {"defaultValue": "0", "paramType": "optional",
                     "type": "int", "value": "all"},
    "attr_out_dtype": {"defaultValue": "0", "paramType": "optional",
                       "type": "int", "value": "all"},
}


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _dump(path, obj):
    # Match the generator (sort_keys/indent 4/no separator spaces) so a diff
    # against the unmodified JSON shows only the added entry.
    mode = os.stat(path).st_mode if os.path.exists(path) else 0o640
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, sort_keys=True, indent=4, separators=(",", ":"))
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _check_entry(path):
    try:
        obj = _load(path)
    except (OSError, ValueError) as error:
        raise SystemExit(f"[qkv-ops-info] error: cannot read {path}: {error}")
    entry = obj.get(OP_NAME)
    if not isinstance(entry, dict):
        raise SystemExit(
            f"[qkv-ops-info] error: {path} has no {OP_NAME} entry; graph mode "
            "would fail with EZ3003. Is the cmake injection hook disabled?")
    if entry.get("aclnnSupport", {}).get("value") != "support_aclnn":
        raise SystemExit(
            f"[qkv-ops-info] error: {path} {OP_NAME} lacks "
            "aclnnSupport=support_aclnn; GE would ignore the entry.")


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    paths = argv[1:]
    check_only = paths[0] == "--check"
    if check_only:
        paths = paths[1:]
        if not paths:
            raise SystemExit(__doc__)
        for path in paths:
            _check_entry(path)
            print(f"[qkv-ops-info] ok: {path}")
        return 0

    for path in paths:
        if not os.path.isfile(path):
            raise SystemExit(
                f"[qkv-ops-info] error: {path} missing; the ops-info "
                "generator did not run")
        obj = _load(path)
        obj[OP_NAME] = _GE_ENTRY
        _dump(path, obj)
        _check_entry(path)
        print(f"[qkv-ops-info] injected {OP_NAME} into {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
