#!/usr/bin/env python3
"""Benchmark CausalConv1dQkv on the real GDN conv shapes of the
startlux-decision-v1.1-0.8b-instant model.

The fused op (conv+silu + q/k per-head L2 norm over a packed [q|k|v] layout)
is compared against the model's current per-layer sub-chain (modeling_text.py:
causal_conv1d + 3 slices/reshapes + 2 folded L2 rms_norms), in eager and in
torchair graph mode.  Cases are the model's true (B, S) GDN prefill shapes:
18 GDN layers, conv_dim 6144, weight (4, 6144) bf16, q_dim=k_dim=v_dim=2048,
head_dim 128, no conv history (initial_state_mode=0).

Usage (from the repo root or the package dir):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  npu-smi info          # pick a chip with AICore 0% and no running process
  python3 causal-conv1d-qkv/test/bench_causal_conv1d_qkv.py --device <idle chip>
  python3 causal-conv1d-qkv/test/bench_causal_conv1d_qkv.py --device <idle chip> --check-only

Timing plumbing follows work/fla-npu-bench/bench_gdn_ops.py: warmup calls ->
one synchronize -> per-call NPU Event start/stop pairs over the timed loop ->
one synchronize; host wall clock around the loop.  Reduction keys mirror that
harness (kernel_ms_*).  A failing arm is recorded as {"error": ...} and the
run continues, so a broken arm can never be mistaken for a fast one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

# Activators: dlopen the vendor libs, mount the opp trees and register
# torch.ops.ascend_ops.{causal_conv1d_qkv, causal_conv1d} + GE converters.
import causal_conv1d_qkv  # noqa: F401
import causal_conv1d  # noqa: F401
from torch_npu.dynamo.torchair import CompilerConfig, get_npu_backend

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reference math shared with the functional gate (conv_reference is the
# upstream CPU golden; packed_reference is its q/k L2 normalization half).
from test_causal_conv1d_qkv import conv_reference, packed_reference  # noqa: E402

# --- model constants (startlux-decision-v1.1-0.8b-instant, Qwen3_5 GDN) ---
DIM = 6144
Q_DIM = K_DIM = V_DIM = 2048
HEAD_DIM = 128
WIDTH = 4
DTYPE = torch.bfloat16
GDN_LAYERS = 18

# Real per-case (B, S) shapes from the model's input pool.  `team` and
# `severity` share a shape; both ids stay so the table lines up with the
# model's published per-case numbers.
CASES = [
    ("official-ticket-team-r0", 1, 99),
    ("official-ticket-severity-r0", 1, 99),
    ("official-ticket-urgent-r0", 1, 85),
    ("official-ticket-b3-r0", 3, 99),
    ("official-image-damaged-r0", 1, 130),
    ("long-seq-1001", 1, 1001),
]

# Smallest case: used for the CPU golden correctness block and the
# run-to-run spread repeat.
SMALLEST = ("official-ticket-urgent-r0", 1, 85)
REPEAT_CASE = SMALLEST

# One mapping drives the table columns, the JSON report and the error list.
# short: stdout column label; eager/graph: which tables the arm belongs to;
# floor: the launch-floor probe, which is not a real arm.
ARMS = {
    "fused_eager": {
        "short": "fused", "eager": True, "graph": False,
        "desc": "CausalConv1dQkv eager, bf16 out, NPU-resident metadata "
                "(state write-back included)"},
    "fused_eager_fp16": {
        "short": "fused_fp16", "eager": True, "graph": False,
        "desc": "CausalConv1dQkv eager, fp16 out, NPU-resident metadata"},
    "fused_eager_hostmeta": {
        "short": "fused_hmeta", "eager": True, "graph": False,
        "desc": "CausalConv1dQkv eager, bf16 out, HOST-resident metadata "
                "(the eager contract's cheap route: no per-call D2H + stream "
                "sync)"},
    "conv_only_eager": {
        "short": "conv", "eager": True, "graph": False,
        "desc": "causal_conv1d(x, w, None, 'silu') alone (eager)"},
    "chain_current_eager": {
        "short": "cur_eager", "eager": True, "graph": False,
        "desc": "current model sub-chain eager: conv + 3 slices + reshape + "
                "2 rms_norm + flatten"},
    "chain_fused_eager": {
        "short": "fus_eager", "eager": True, "graph": False,
        "desc": "replacement sub-chain eager (NPU metadata): fused op + the "
                "same 3 slices/reshape/flatten (no rms_norm)"},
    "chain_fused_eager_hostmeta": {
        "short": "fus_eager_hmeta", "eager": True, "graph": False,
        "desc": "replacement sub-chain eager with HOST metadata"},
    "fused_graph": {
        "short": "fused_graph", "eager": True, "graph": True,
        "desc": "CausalConv1dQkv inside torch.compile(dynamic=True), bf16 out"},
    "chain_current_graph": {
        "short": "cur_graph", "eager": False, "graph": True,
        "desc": "current sub-chain in torch.compile(dynamic=True)"},
    "chain_fused_graph": {
        "short": "fus_graph", "eager": False, "graph": True,
        "desc": "replacement sub-chain in torch.compile(dynamic=True)"},
    "graph_launch_floor": {
        "short": "floor", "eager": False, "graph": True, "floor": True,
        "desc": "trivial 1-node compiled graph, the per-call launch floor"},
}


def arm_names(*, eager=False, graph=False, include_floor=True):
    """Arm ids for a table, in mapping order."""
    names = []
    for name, info in ARMS.items():
        if not ((eager and info["eager"]) or (graph and info["graph"])):
            continue
        if include_floor or not info.get("floor"):
            names.append(name)
    return names


METRIC_KEYS = ("kernel_ms_mean", "kernel_ms_min", "kernel_ms_p50",
               "kernel_ms_p99", "wall_ms_mean")


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def time_call(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    events = []
    t0 = time.perf_counter()
    for _ in range(iters):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        events.append((s, e))
    torch.npu.synchronize()
    wall = (time.perf_counter() - t0) / iters * 1e3
    times = sorted(s.elapsed_time(e) for s, e in events)
    n = len(times)
    return {
        "kernel_ms_mean": sum(times) / n,
        "kernel_ms_min": times[0],
        "kernel_ms_p50": times[n // 2],
        "kernel_ms_p99": times[min(n - 1, math.ceil(0.99 * n) - 1)],
        "wall_ms_mean": wall,
    }


def metrics(out, ref):
    o = out.detach().float().cpu().reshape(-1)
    r = ref.detach().float().cpu().reshape(-1)
    cosine = float(torch.dot(o, r) /
                   (o.norm() * r.norm() + torch.finfo(torch.float32).tiny))
    rel_l2 = float((o - r).norm() / (r.norm() + torch.finfo(torch.float32).tiny))
    return cosine, rel_l2


# --------------------------------------------------------------------------
# inputs / chains
# --------------------------------------------------------------------------
class ChainCurrent(torch.nn.Module):
    """modeling_text.py:142-155 without the later GDN call."""

    def forward(self, x_bsd, w, gamma):
        batch, width, _ = x_bsd.shape
        mixed = torch.ops.ascend_ops.causal_conv1d(x_bsd, w, None, "silu")
        q = mixed[..., :Q_DIM].reshape(batch, width, -1, HEAD_DIM)
        k = mixed[..., Q_DIM:2 * K_DIM].reshape(batch, width, -1, HEAD_DIM)
        v = mixed[..., 2 * K_DIM:].reshape(batch, width, -1, HEAD_DIM)
        eps = 1e-6 / HEAD_DIM
        q, _ = torch.ops.npu.npu_rms_norm(q, gamma, eps)
        k, _ = torch.ops.npu.npu_rms_norm(k, gamma, eps)
        return q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)


class ChainFused(torch.nn.Module):
    """Replacement: fused op + the same 3 slices/reshape/flatten."""

    def forward(self, x_bsd, w, conv_states, qsl, ci, ism):
        batch, width, _ = x_bsd.shape
        y = torch.ops.ascend_ops.causal_conv1d_qkv(
            x_bsd.reshape(-1, DIM), w, conv_states,
            Q_DIM, K_DIM, V_DIM, HEAD_DIM, 2, -1, 0, None, qsl, ci, ism)
        y = y.reshape(batch, width, DIM)
        q = y[..., :Q_DIM].reshape(batch, width, -1, HEAD_DIM)
        k = y[..., Q_DIM:2 * K_DIM].reshape(batch, width, -1, HEAD_DIM)
        v = y[..., 2 * K_DIM:].reshape(batch, width, -1, HEAD_DIM)
        return q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)


def fused_forward(x_view, w, conv_states, qsl, ci, ism, out_dtype=None):
    return torch.ops.ascend_ops.causal_conv1d_qkv(
        x_view, w, conv_states, Q_DIM, K_DIM, V_DIM, HEAD_DIM,
        2, -1, 0, out_dtype, qsl, ci, ism)


def _plus_one(a):
    """Trivial one-node graph: measures the compiled-graph per-call floor."""
    return a + 1.0


def make_inputs(B, S, device):
    """One seed for every case so repeated runs are bit-identical."""
    torch.manual_seed(42)
    x = torch.randn(B, S, DIM, dtype=DTYPE, device=device)
    w = torch.randn(WIDTH, DIM, dtype=DTYPE, device=device)
    conv_states = torch.zeros(B, WIDTH - 1, DIM, dtype=DTYPE, device=device)
    qsl = torch.tensor([i * S for i in range(B + 1)], dtype=torch.int64,
                       device=device)
    ci = torch.arange(B, dtype=torch.int64, device=device)
    ism = torch.zeros(B, dtype=torch.int64, device=device)
    gamma = torch.full((HEAD_DIM,), HEAD_DIM ** -0.5, dtype=DTYPE,
                       device=device)
    return x, w, conv_states, qsl, ci, ism, gamma


# --------------------------------------------------------------------------
# one case: all arms
# --------------------------------------------------------------------------
def run_case(case_id, B, S, device, iters, warmup):
    from torch._dynamo.utils import counters as dynamo_counters

    x, w, conv_states, qsl, ci, ism, gamma = make_inputs(B, S, device)
    x_view = x.reshape(-1, DIM)
    record = {"case": case_id, "B": B, "S": S, "T": B * S,
              "graph_trace": {}}
    cur_mod = ChainCurrent().eval()
    fus_mod = ChainFused().eval()

    def frames_ok():
        return dynamo_counters["frames"].get("ok", 0)

    def compile_and_trace(name, fn, args):
        """Compile fn, run it once, and require that a new graph was traced:
        a silent recompile-limit fallback to eager would invalidate the arm."""
        compiled = torch.compile(fn, backend=get_npu_backend(
            compiler_config=CompilerConfig()), dynamic=True)
        before = frames_ok()
        compiled(*args)
        torch.npu.synchronize()
        new_frames = frames_ok() - before
        record["graph_trace"][name] = {"new_frames": new_frames,
                                       "traced": new_frames > 0}
        if new_frames <= 0:
            raise RuntimeError(
                f"{name}: no new graph frame traced for {case_id} "
                f"(B={B}, S={S}) -> the arm fell back to eager")
        return compiled

    # Eager graph-arm callables are compiled inside their own try block below
    # so a compile failure cannot take the other arms down.
    def arm_fused_graph():
        compiled = compile_and_trace(
            "fused_graph", fused_forward,
            (x_view, w, conv_states, qsl, ci, ism))
        return lambda: compiled(x_view, w, conv_states, qsl, ci, ism)

    def arm_chain_current_graph():
        compiled = compile_and_trace(
            "chain_current_graph", cur_mod, (x, w, gamma))
        return lambda: compiled(x, w, gamma)

    def arm_chain_fused_graph():
        compiled = compile_and_trace(
            "chain_fused_graph", fus_mod, (x, w, conv_states, qsl, ci, ism))
        return lambda: compiled(x, w, conv_states, qsl, ci, ism)

    # Host-resident metadata: the eager entry then feeds the aclnn host tiling
    # directly instead of doing a stream-ordered D2H + aclrtSynchronizeStream.
    qsl_h, ci_h, ism_h = qsl.cpu(), ci.cpu(), ism.cpu()

    arms = [
        ("fused_eager",
         lambda: fused_forward(x_view, w, conv_states, qsl, ci, ism)),
        ("fused_eager_fp16",
         lambda: fused_forward(x_view, w, conv_states, qsl, ci, ism,
                               torch.float16)),
        ("fused_eager_hostmeta",
         lambda: fused_forward(x_view, w, conv_states, qsl_h, ci_h, ism_h)),
        ("fused_graph", arm_fused_graph),
        ("conv_only_eager",
         lambda: torch.ops.ascend_ops.causal_conv1d(x, w, None, "silu")),
        ("chain_current_eager", lambda: cur_mod(x, w, gamma)),
        ("chain_fused_eager",
         lambda: fus_mod(x, w, conv_states, qsl, ci, ism)),
        ("chain_fused_eager_hostmeta",
         lambda: fus_mod(x, w, conv_states, qsl_h, ci_h, ism_h)),
        ("chain_current_graph", arm_chain_current_graph),
        ("chain_fused_graph", arm_chain_fused_graph),
    ]

    for name, thunk in arms:
        try:
            fn = thunk() if name.endswith("_graph") else thunk
            r = time_call(fn, iters, warmup)
            r["iters"] = iters
            r["warmup"] = warmup
            record[name] = r
        except Exception as exc:  # fail loud, keep going
            record[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  ARM ERROR {case_id}/{name}: {type(exc).__name__}: {exc}")

    # Per-call floor of the compiled-graph route on this host (trivial one-node
    # graph).  It is NOT an arm: it bounds how much of a graph number is graph
    # launch rather than operator work.
    try:
        floor = compile_and_trace("graph_launch_floor", _plus_one, (x_view,))
        f = time_call(lambda: floor(x_view), iters, warmup)
        f["iters"] = iters
        f["warmup"] = warmup
        record["graph_launch_floor"] = f
    except Exception as exc:
        record["graph_launch_floor"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  ARM ERROR {case_id}/graph_launch_floor: "
              f"{type(exc).__name__}: {exc}")

    return record


# --------------------------------------------------------------------------
# correctness
# --------------------------------------------------------------------------
def check_correctness(device):
    """Smallest case vs the CPU fp32 golden (test_m... conv_reference +
    packed_reference), plus eager vs graph agreement."""
    from torch._dynamo.utils import counters as dynamo_counters

    case_id, B, S = SMALLEST
    T = B * S
    print("=" * 100)
    print(f"Correctness (smallest case {case_id}, B={B} S={S}) vs CPU fp32 "
          f"golden; package gate cos>0.95, relL2<0.02")
    print("=" * 100)

    x, w, _, qsl, ci, ism, _ = make_inputs(B, S, device)
    x_view = x.reshape(-1, DIM)

    # Golden on its own state clone (conv_reference mutates it in place).
    state_ref = torch.zeros(B, WIDTH - 1, DIM, dtype=DTYPE)
    conv_out = conv_reference(x_view.cpu(), w.cpu(), state_ref, [S], [0], [0])
    golden = packed_reference(conv_out, Q_DIM, K_DIM, V_DIM, DTYPE)

    state = torch.zeros(B, WIDTH - 1, DIM, dtype=DTYPE, device=device)
    eager = fused_forward(x_view, w, state, qsl, ci, ism)
    eager_state_exact = torch.equal(state.cpu(), state_ref)

    # Fresh state for the graph leg: the eager call above already mutated the
    # shared buffer, so reusing it would not be an independent check.
    graph_state = torch.zeros(B, WIDTH - 1, DIM, dtype=DTYPE, device=device)
    before = dynamo_counters["frames"].get("ok", 0)
    compiled = torch.compile(fused_forward, backend=get_npu_backend(
        compiler_config=CompilerConfig()), dynamic=True)
    graph = compiled(x_view, w, graph_state, qsl, ci, ism)
    torch.npu.synchronize()
    graph_state_exact = torch.equal(graph_state.cpu(), state_ref)
    traced = dynamo_counters["frames"].get("ok", 0) - before

    c_eager, r_eager = metrics(eager, golden)
    c_graph, r_graph = metrics(graph, golden)
    _, r_eager_graph = metrics(graph, eager)
    dtype_ok = eager.dtype == DTYPE and graph.dtype == DTYPE
    gate = (c_eager >= 0.95 and r_eager <= 0.02
            and c_graph >= 0.95 and r_graph <= 0.02)

    result = {
        "case": case_id, "B": B, "S": S, "T": T,
        "golden": "conv_reference + packed_reference (CPU fp32)",
        "eager": {"cosine": c_eager, "relL2": r_eager},
        "graph": {"cosine": c_graph, "relL2": r_graph},
        "graph_vs_eager_relL2": r_eager_graph,
        "graph_new_frames": traced,
        "dtype_ok": dtype_ok,
        "conv_states_exact_eager": eager_state_exact,
        "conv_states_exact_graph": graph_state_exact,
        "gate": "PASS" if gate else "FAIL",
    }
    print(f"fused_eager  cos={c_eager:.6f} relL2={r_eager:.6f}")
    print(f"fused_graph  cos={c_graph:.6f} relL2={r_graph:.6f} "
          f"(vs eager relL2={r_eager_graph:.8f}, new frames={traced})")
    print(f"conv_states bit-exact vs golden: eager={eager_state_exact} "
          f"graph={graph_state_exact}   dtype ok: {dtype_ok}   "
          f"gate: {result['gate']}")
    return result


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def _mean_us(rec, arm):
    r = rec.get(arm)
    if not r or "error" in r:
        return None
    return r["kernel_ms_mean"] * 1e3


def _p50_us(rec, arm):
    r = rec.get(arm)
    if not r or "error" in r:
        return None
    return r["kernel_ms_p50"] * 1e3


def _fmt(v):
    return f"{v:>10.1f}" if v is not None else f"{'ERR':>10}"


def _arm_table(title, perf, arms, getter):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    head = f"{'case':<28}{'B':>2}{'S':>5}{'T':>6} |"
    head += "".join(f"{ARMS[a]['short']:>16}" for a in arms)
    print(head)
    for rec in perf:
        line = (f"{rec['case']:<28}{rec['B']:>2}{rec['S']:>5}{rec['T']:>6} |")
        line += "".join(f"{_fmt(getter(rec, a)):>16}" for a in arms)
        print(line)


def print_tables(perf):
    eager_cols = arm_names(eager=True)
    graph_cols = arm_names(graph=True)
    graph_arms = arm_names(graph=True, include_floor=False)
    _arm_table("Mean kernel time per call (us); ERR = arm raised (see JSON)",
               perf, eager_cols, _mean_us)
    _arm_table("P50 kernel time per call (us)", perf, eager_cols, _p50_us)
    _arm_table("Graph-mode arms: mean / P50 per call (us) and the compiled-"
               "graph launch floor (trivial 1-node graph)",
               perf, graph_cols, _mean_us)
    _arm_table("Graph-mode arms: P50 per call (us)", perf, graph_cols, _p50_us)

    _arm_table(f"Per-request projection: mean us/call x {GDN_LAYERS} GDN "
               "layers (graph-mode arms)", perf, graph_arms,
               lambda r, a: (None if _mean_us(r, a) is None
                             else _mean_us(r, a) * GDN_LAYERS))

    print()
    print("=" * 100)
    print("Verdict per case: chain_current_graph - chain_fused_graph "
          "(graph mode, the model's deployment route)")
    print("=" * 100)
    print(f"{'case':<28}{'B':>2}{'S':>5}{'cur us':>9}{'fus us':>9}"
          f"{'floor':>8}{'d us/layer':>12}{'d us/req':>11}{'% faster':>10}")
    for rec in perf:
        cur = _mean_us(rec, "chain_current_graph")
        fus = _mean_us(rec, "chain_fused_graph")
        floor = _mean_us(rec, "graph_launch_floor")
        if cur is None or fus is None:
            print(f"{rec['case']:<28}{rec['B']:>2}{rec['S']:>5}"
                  f"{'ERR':>9}{'ERR':>9}{'n/a':>8}{'n/a':>12}{'n/a':>11}"
                  f"{'n/a':>10}")
            continue
        d = cur - fus
        pct = 100.0 * d / cur if cur else float("nan")
        fl = f"{floor:>8.1f}" if floor is not None else f"{'ERR':>8}"
        print(f"{rec['case']:<28}{rec['B']:>2}{rec['S']:>5}{cur:>9.1f}"
              f"{fus:>9.1f}{fl}{d:>12.1f}{d * GDN_LAYERS:>11.1f}{pct:>9.1f}%")

    print()
    print("Eager decomposition (us/call, smallest + largest shapes):")
    for rec in (perf[2], perf[-1]):
        fe = _mean_us(rec, "fused_eager")
        fh = _mean_us(rec, "fused_eager_hostmeta")
        cf = _mean_us(rec, "chain_fused_eager")
        cfh = _mean_us(rec, "chain_fused_eager_hostmeta")
        ce = _mean_us(rec, "chain_current_eager")
        d2h = (fe - fh) if (fe is not None and fh is not None) else None
        print(f"  {rec['case']:<26} fused NPU-meta={_fmt(fe)} "
              f"fused host-meta={_fmt(fh)} "
              f"D2H cost={_fmt(d2h)} | chain cur={_fmt(ce)} "
              f"chain fus NPU-meta={_fmt(cf)} host-meta={_fmt(cfh)}")

    print()
    errs = [(rec["case"], a, rec[a]["error"]) for rec in perf
            for a in ARMS if rec.get(a, {}).get("error")]
    if errs:
        print("FAILED ARMS (NOT zero - these produced no number):")
        for case, arm, err in errs:
            print(f"  {case}/{arm}: {err}")
    else:
        print("Failed arms: none")


def summarize(perf, repeat):
    out = {}
    for rec in perf:
        cur = _mean_us(rec, "chain_current_graph")
        fus = _mean_us(rec, "chain_fused_graph")
        if cur is not None and fus is not None:
            entry = {
                "chain_current_graph_us": cur,
                "chain_fused_graph_us": fus,
                "delta_us_per_layer": cur - fus,
                "delta_us_per_request": (cur - fus) * GDN_LAYERS,
                "delta_pct": 100.0 * (cur - fus) / cur,
            }
            floor = _mean_us(rec, "graph_launch_floor")
            if floor is not None:
                entry["graph_launch_floor_us"] = floor
                entry["chain_current_graph_minus_floor_us"] = cur - floor
                entry["chain_fused_graph_minus_floor_us"] = fus - floor
            fe = _mean_us(rec, "fused_eager")
            fh = _mean_us(rec, "fused_eager_hostmeta")
            if fe is not None and fh is not None:
                entry["fused_eager_npu_metadata_d2h_cost_us"] = fe - fh
            out[rec["case"]] = entry
    if repeat is not None:
        spreads = {}
        for arm in (name for name, info in ARMS.items()
                    if not info.get("floor")):
            a = _mean_us(next(r for r in perf if r["case"] == REPEAT_CASE[0]),
                         arm)
            b = _mean_us(repeat, arm)
            if a is not None and b is not None:
                spreads[arm] = {"first_us": a, "repeat_us": b,
                                "delta_pct": 100.0 * (b - a) / a}
        out["run_to_run_spread"] = {
            "case": REPEAT_CASE[0],
            "note": "smallest case re-run end to end (graph arms recompiled)",
            "arms": spreads}
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, required=True,
                    help="NPU chip id; pick an idle chip (npu-smi info: "
                         "AICore 0%, no running process)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--out", default=str(HERE / "bench_result.json"))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"

    # Every case traces its own graph (fresh compile, dynamic shapes per case);
    # the default limit would silently fall back to eager and fake the number.
    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = 64
    if hasattr(dynamo.config, "cache_size_limit"):
        dynamo.config.cache_size_limit = 64

    try:
        soc = torch.npu.get_device_name(args.device)
    except Exception:  # pragma: no cover - only a reporting field
        soc = "unknown"
    cann = os.environ.get("ASCEND_HOME_PATH", "unknown")

    report = {
        "device": device,
        "dtype": str(DTYPE),
        "iters": args.iters,
        "warmup": args.warmup,
        "soc": soc,
        "cann": cann,
        "arms": {name: info["desc"] for name, info in ARMS.items()
                 if not info.get("floor")},
        "geometry": {"conv_dim": DIM, "q_dim": Q_DIM, "k_dim": K_DIM,
                     "v_dim": V_DIM, "head_dim": HEAD_DIM, "width": WIDTH,
                     "no_bias": True, "gdn_layers": GDN_LAYERS,
                     "initial_state_mode": "zeros (model has no conv history)"},
        "state_side_effect": "conv_states is written back in place by every "
                             "fused arm, so the state write is inside the "
                             "timed region (initial_state_mode=0 makes the "
                             "values irrelevant to the output)",
        "notes": {
            "metadata_residence": "eager arms mirror the graph arms with "
                                  "NPU-resident metadata (query_start_loc / "
                                  "cache_indices / initial_state_mode), which "
                                  "makes the eager entry do a stream-ordered "
                                  "D2H + aclrtSynchronizeStream per call; the "
                                  "*_hostmeta arms keep the same values on the "
                                  "host, the contract's cheap eager route, and "
                                  "exist only to price that D2H.",
            "graph_launch_floor": "graph_launch_floor is a trivial 1-node "
                                  "compiled graph (a + 1.0) measured with the "
                                  "same harness; it bounds how much of a "
                                  "graph-mode number is per-call graph launch "
                                  "on this host, not operator work.",
        },
    }
    report["correctness"] = check_correctness(device)
    if args.check_only:
        print(json.dumps(report, indent=2))
        return 0

    print()
    print("=" * 100)
    print(f"Performance: warmup={args.warmup} iters={args.iters} per arm, "
          f"model has {GDN_LAYERS} GDN layers")
    print("=" * 100)
    perf = []
    for case_id, B, S in CASES:
        print(f"running {case_id} (B={B}, S={S}) ...")
        perf.append(run_case(case_id, B, S, device, args.iters, args.warmup))
    report["perf"] = perf

    print(f"\nrepeat (run-to-run spread) {REPEAT_CASE[0]} ...")
    repeat = run_case(*REPEAT_CASE, device, args.iters, args.warmup)
    report["repeat"] = repeat

    print_tables(perf)
    report["summary"] = summarize(perf, repeat)

    print()
    print("Run-to-run spread (smallest case, repeated once):")
    for arm, sp in report["summary"]["run_to_run_spread"]["arms"].items():
        print(f"  {arm:<22} {sp['first_us']:>8.1f} -> {sp['repeat_us']:>8.1f} us"
              f"  ({sp['delta_pct']:+.1f}%)")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
