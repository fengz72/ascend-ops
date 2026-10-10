#!/usr/bin/env python3
"""GE online Session end-to-end test for the PrefixAttention custom op.

Builds a single-op graph (4 inputs -> PrefixAttention -> 1 output) with the
generated ES wrapper, runs it on the NPU through ge.Session, and compares
against the python host-path reference output dumped by probe/dump_case.py
(same kernels, same inputs -> expected bit-exact).

Env (set by tests/run_online.sh):
  ASCEND_CUSTOM_OPP_PATH  opp root with libcust_opapi.so + artifacts
  PYTHONPATH              staged es_prefix_attn package dir
  LD_LIBRARY_PATH         dir containing libes_prefix_attn.so
"""
import sys
import traceback
from pathlib import Path

import numpy as np

from ge.es.graph_builder import GraphBuilder
from ge.ge_global import GeApi
from ge.graph import Tensor
from ge.graph.types import DataType, Format
from ge.session import Session

HERE = Path(__file__).resolve().parent
GE_DIR = HERE.parent
CASE_DATA = GE_DIR / "probe" / "case_data"

GRAPH_ID = 0
DEVICE_ID = 0


def read_meta():
    meta = {}
    for line in (CASE_DATA / "meta.txt").read_text().splitlines():
        k, v = line.split()
        meta[k] = v
    return meta


def build_graph(meta):
    from es_prefix_attn import PrefixAttention

    t = int(meta["T"])
    n1 = int(meta["N1"])
    n2 = int(meta["N2"])
    blen = int(meta["BOUNDS_LEN"])
    dtype = (DataType.DT_BF16 if meta["DTYPE"] == "bfloat16"
             else DataType.DT_FLOAT16)
    builder = GraphBuilder("prefix_attention_graph")
    q = builder.create_input(index=0, name="query", data_type=dtype,
                             format=Format.FORMAT_ND, shape=[t, n1, 64])
    k = builder.create_input(index=1, name="key", data_type=dtype,
                             format=Format.FORMAT_ND, shape=[t, n2, 64])
    v = builder.create_input(index=2, name="value", data_type=dtype,
                             format=Format.FORMAT_ND, shape=[t, n2, 64])
    b = builder.create_input(index=3, name="actual_seq_lengths",
                             data_type=DataType.DT_INT32,
                             format=Format.FORMAT_ND, shape=[blen])
    out = PrefixAttention(q, k, v, b, scale=float(meta["SCALE"]))
    out.set_shape([t, n1, 64]).set_format(Format.FORMAT_ND).set_data_type(dtype)
    builder.set_graph_output(out, 0)
    return builder.build_and_reset()


def input_tensors(meta):
    t, n1, n2 = int(meta["T"]), int(meta["N1"]), int(meta["N2"])
    blen = int(meta["BOUNDS_LEN"])
    dtype = (DataType.DT_BF16 if meta["DTYPE"] == "bfloat16"
             else DataType.DT_FLOAT16)

    def raw(name, dt, shape):
        return Tensor(None, str(CASE_DATA / name), dt, Format.FORMAT_ND, shape)

    return [raw("q.bin", dtype, [t, n1, 64]),
            raw("k.bin", dtype, [t, n2, 64]),
            raw("v.bin", dtype, [t, n2, 64]),
            raw("bounds.bin", DataType.DT_INT32, [blen])]


def run() -> int:
    meta = read_meta()
    options = {"ge.exec.deviceId": str(DEVICE_ID), "ge.graphRunMode": "1"}
    ge_api = GeApi()
    session = None
    ge_initialized = False
    graph_added = False
    try:
        ge_api.ge_initialize(options)
        ge_initialized = True
        session = Session(options)
        session.add_graph(GRAPH_ID, build_graph(meta))
        graph_added = True
        print("GE_GRAPH_ADDED=1", flush=True)
        outputs = session.run_graph(GRAPH_ID, input_tensors(meta))
        if len(outputs) != 1:
            raise RuntimeError(f"expected 1 output, got {len(outputs)}")

        # NOTE: ge Tensor.get_data() returns values round-tripped through a
        # text serialization (fp16 -> decimal string -> float), which adds
        # ~1e-7 noise on every element; device-side output is bit-exact (see
        # probe/). Compare with the fp16/bf16 tolerance instead of bit-equal.
        got = np.asarray(outputs[0].get_data()).astype(np.float32).ravel()
        np_dtype = (np.dtype("V2") if meta["DTYPE"] == "bfloat16"
                    else np.float16)
        ref_raw = np.fromfile(CASE_DATA / "out_py.bin", dtype=np_dtype)
        if meta["DTYPE"] == "bfloat16":
            ref = (ref_raw.view(np.uint16).astype(np.uint32) << 16).view(
                np.float32)
        else:
            ref = ref_raw.astype(np.float32)
        if got.size != ref.size:
            raise RuntimeError(f"size mismatch: got {got.size}, ref {ref.size}")
        diff = np.abs(got - ref)
        max_abs = float(diff.max())
        mismatch = int((diff != 0).sum())
        print(f"elements={got.size} mismatch={mismatch} max_abs_diff={max_abs:.6g}")
        tol = 0.0 if mismatch == 0 else (0.025 if meta["DTYPE"] == "bfloat16"
                                          else 0.003)
        if mismatch == 0:
            print("GE_SESSION_PASS (bit-exact vs python host path)", flush=True)
        elif max_abs <= tol:
            print(f"GE_SESSION_PASS (within tol {tol})", flush=True)
        else:
            print(f"GE_SESSION_FAIL (max_abs_diff={max_abs} > {tol})",
                  flush=True)
            return 1
        return 0
    except Exception as error:
        print(f"GE_SESSION_FAIL: {error}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        try:
            if graph_added and session is not None:
                session.remove_graph(GRAPH_ID)
        finally:
            session = None
            if ge_initialized:
                ge_api.ge_finalize()


if __name__ == "__main__":
    raise SystemExit(run())
