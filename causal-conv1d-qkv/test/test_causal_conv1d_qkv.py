"""Functional gate for CausalConv1dQkv: CPU golden vs eager NPU vs compiled graph.

Usage:
    python3 test/test_causal_conv1d_qkv.py [device]

Requires: CANN env sourced (set_env.sh) and `bash install.sh` (builds,
installs the operator and registers the binding). The activator locates the
.so and mounts the vendor package itself; no shell exports are needed.

Gates:
  * graph vs eager            rel_l2 < 0.001
  * eager vs CPU fp32 golden  cos > 0.95 and rel_l2 < 0.02
  * conv_states after each leg EXACTLY equal to the golden (atol=rtol=0):
    the op's state update is a bit-preserving copy, so any rounding or
    one-step-off window is a defect. Each leg runs on its own state clone.

Beyond the main sweep this gate also covers: host-resident metadata (the
cheap eager route recommended by the activator/README/bench), a middle
optional omission (cache_indices absent, initial_state_mode present) that
pins the wrapper's IR-index binding, a two-step decode chain that reuses
conv_states in place, and negative legs that must fail loud.

Cases mirror the upstream parametrization (head_dim=128, width=4, q_dim=k_dim
=q_heads*128, v_dim=v_heads*128), each in bf16 and fp16 output. The varying
q/k/v dims prove non-default scalars reach the kernel (they change the output
width); the ragged cases mix initial_state_mode 0 and 1 inside one batch.

The graph leg compiles with dynamic=True on purpose: with dynamic=False
torchair stages every input into GE-owned static buffers and the 1-D int64
metadata buffers arrive zero-filled, so host tiling rejects the call
("queryStartLoc[last] must equal cuSeqlen, got 0 vs N"). Eager is unaffected.
"""
import sys

import torch
import torch.nn.functional as F

HEAD_DIM = 128
WIDTH = 4

CASES = [
    # (q_heads, v_heads, lengths, initial_state_mode)
    (16, 48, (64,), (1,)),          # tp1
    (8, 24, (127, 129), (0, 1)),    # tp2-ragged
    (4, 12, (256,), (0,)),          # tp4
    (2, 6, (63, 65), (1, 0)),       # tp8-ragged
    (1, 3, (129,), (1,)),           # tp16
    (16, 16, (22,), (0,)),          # qwen3.5-2b prefill
]
OUT_DTYPES = [("bf16", torch.bfloat16), ("fp16", torch.float16)]


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def cosine(a, b):
    u, v = a.float().flatten(), b.float().flatten()
    return (torch.dot(u, v) / (u.norm() * v.norm())).item()


def cumulative_lengths(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def conv_reference(x, weight, conv_state, lengths, cache_indices,
                   initial_state_mode):
    """Upstream golden: per-sequence causal conv over the state history."""
    outputs = []
    token_offset = 0
    weight_fp32 = weight.float()
    for batch_idx, length in enumerate(lengths):
        slot = cache_indices[batch_idx]
        sequence = x[token_offset:token_offset + length]
        history = (conv_state[slot].clone() if initial_state_mode[batch_idx]
                   else torch.zeros((WIDTH - 1, x.size(1)), dtype=x.dtype))
        padded = torch.cat((history, sequence), dim=0)
        accumulator = torch.zeros_like(sequence, dtype=torch.float32)
        for tap in range(WIDTH):
            accumulator.add_(padded[tap:tap + length].float() * weight_fp32[tap])
        outputs.append(F.silu(accumulator).to(x.dtype))
        conv_state[slot].copy_(padded[-(WIDTH - 1):])
        token_offset += length
    return torch.cat(outputs, dim=0)


def packed_reference(conv_output, q_dim, k_dim, v_dim, output_dtype):
    """Split [q|k|v], per-head L2 normalize q/k in fp32, cast through bf16."""
    q, k, v = torch.split(conv_output, (q_dim, k_dim, v_dim), dim=-1)

    def normalize_qk(value):
        heads = value.view(value.size(0), -1, HEAD_DIM).float()
        norm = torch.sqrt((heads * heads).sum(dim=-1, keepdim=True) + 1.0e-6)
        return (heads / norm).to(torch.bfloat16).to(output_dtype)

    return torch.cat(
        (normalize_qk(q).reshape(-1), normalize_qk(k).reshape(-1),
         v.to(output_dtype).reshape(-1))).view_as(conv_output)


def build_case(q_heads, v_heads, lengths, initial_state_mode):
    gen = torch.Generator().manual_seed(20260716 + q_heads + v_heads)
    q_dim = q_heads * HEAD_DIM
    k_dim = q_dim
    v_dim = v_heads * HEAD_DIM
    dim = q_dim + k_dim + v_dim
    total_tokens = sum(lengths)
    num_slots = len(lengths) + 2
    cache_indices = list(range(1, len(lengths) + 1))
    x = (torch.randn((total_tokens, dim), generator=gen) * 0.25).to(torch.bfloat16)
    weight = (torch.randn((WIDTH, dim), generator=gen) * 0.25).to(torch.bfloat16)
    conv_state = (torch.randn((num_slots, WIDTH - 1, dim), generator=gen) * 0.1
                  ).to(torch.bfloat16)
    return {
        "x": x, "weight": weight, "conv_state": conv_state,
        "q_dim": q_dim, "k_dim": k_dim, "v_dim": v_dim,
        "query_start_loc": torch.tensor(cumulative_lengths(lengths),
                                        dtype=torch.int64),
        "cache_indices": torch.tensor(cache_indices, dtype=torch.int64),
        "initial_state_mode": torch.tensor(initial_state_mode, dtype=torch.int64),
    }


def main():
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dev = f"npu:{device}"

    import torch_npu
    torch.npu.set_device(device)
    import causal_conv1d_qkv  # noqa: F401  registers the op + converter
    from torch_npu.dynamo.torchair import CompilerConfig, get_npu_backend
    from torch._dynamo.utils import counters as dynamo_counters

    # Every case has its own (q_dim, k_dim, v_dim, out_dtype) signature, so a
    # fresh graph is traced per case. The default limit (8) would silently fall
    # back to eager on the later cases and the graph gate would test nothing.
    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = 64
    if hasattr(dynamo.config, "cache_size_limit"):
        dynamo.config.cache_size_limit = 64

    # Default CompilerConfig on purpose: the gate must certify the route the
    # bench and deployment use, not a stricter one.
    config = CompilerConfig()
    model = torch.nn.Module()
    model.forward = lambda *a: torch.ops.ascend_ops.causal_conv1d_qkv(*a)
    compiled = torch.compile(model.eval(),
                             backend=get_npu_backend(compiler_config=config),
                             dynamic=True)

    def frames_ok():
        return dynamo_counters["frames"].get("ok", 0)

    def op_call(invoke, state, x, weight, q_dim, k_dim, v_dim, out_dtype,
                qsl, ci, ism, run_mode=0, nat=None):
        return invoke(x, weight, state, q_dim, k_dim, v_dim, HEAD_DIM, 2, -1,
                      run_mode, out_dtype, qsl, ci, ism, nat)

    failures = 0

    def record(label, ok, detail):
        nonlocal failures
        if not ok:
            failures += 1
            print(f"  FAIL {label}: {detail}")

    for q_heads, v_heads, lengths, initial_state_mode in CASES:
        for tag, out_dtype in OUT_DTYPES:
            name = (f"q{q_heads}/v{v_heads} len={lengths} "
                    f"ism={initial_state_mode} {tag}")
            case = build_case(q_heads, v_heads, lengths, initial_state_mode)
            q_dim, k_dim, v_dim = (case["q_dim"], case["k_dim"],
                                   case["v_dim"])

            # Golden runs on its own state clone and returns the mutated state.
            state_ref = case["conv_state"].clone()
            conv_output = conv_reference(case["x"], case["weight"], state_ref,
                                         lengths, case["cache_indices"].tolist(),
                                         initial_state_mode)
            golden = packed_reference(conv_output, q_dim, k_dim, v_dim,
                                      out_dtype)

            dev_x = case["x"].to(dev)
            dev_w = case["weight"].to(dev)
            dev_meta = {k: case[k].to(dev) for k in
                        ("query_start_loc", "cache_indices", "initial_state_mode")}
            dtype_arg = None if out_dtype == torch.bfloat16 else torch.float16

            def run(state, invoke):
                return invoke(dev_x, dev_w, state, q_dim, k_dim, v_dim, HEAD_DIM,
                              2, -1, 0, dtype_arg, dev_meta["query_start_loc"],
                              dev_meta["cache_indices"],
                              dev_meta["initial_state_mode"])

            eager_state = case["conv_state"].clone().to(dev)
            eager = run(eager_state, torch.ops.ascend_ops.causal_conv1d_qkv).cpu()

            graph_state = case["conv_state"].clone().to(dev)
            graphs_before = frames_ok()
            with torch.no_grad():
                graph = run(graph_state, compiled).cpu()
            torch.npu.synchronize()
            graph_compiled = frames_ok() > graphs_before

            c_eager = cosine(eager, golden)
            r_eager = rel_l2(eager, golden)
            r_graph = rel_l2(graph, eager)
            eager_state_ok = torch.equal(eager_state.cpu(), state_ref)
            graph_state_ok = torch.equal(graph_state.cpu(), state_ref)
            dtype_ok = eager.dtype == out_dtype and graph.dtype == out_dtype

            print(f"{name:<48} dtype={str(eager.dtype).replace('torch.',''):<8} "
                  f"eager cos={c_eager:.6f} rel_l2={r_eager:.6f} | "
                  f"graph-vs-eager rel_l2={r_graph:.8f} | "
                  f"state eager={'ok' if eager_state_ok else 'MISMATCH'} "
                  f"graph={'ok' if graph_state_ok else 'MISMATCH'}")

            for label, ok, detail in (
                    ("dtype", dtype_ok,
                     f"eager={eager.dtype} graph={graph.dtype} want={out_dtype}"),
                    ("eager cos", c_eager > 0.95, f"cos={c_eager:.6f}"),
                    ("eager rel_l2", r_eager < 0.02, f"rel_l2={r_eager:.6f}"),
                    ("graph rel_l2", r_graph < 0.001, f"rel_l2={r_graph:.8f}"),
                    ("eager conv_states", eager_state_ok, "state mismatch"),
                    ("graph conv_states", graph_state_ok, "state mismatch"),
                    ("graph compiled", graph_compiled,
                     "no new graph traced -> the graph leg fell back to eager")):
                if not ok:
                    failures += 1
                    print(f"  FAIL {label}: {detail}")

    # --- host-resident metadata eager leg ----------------------------------
    # The activator/README/bench's *_hostmeta arms recommend keeping the three
    # metadata tensors on CPU; this was the contract's cheap route with zero
    # correctness coverage.
    q_heads, v_heads, lengths, ism = CASES[0]
    case = build_case(q_heads, v_heads, lengths, ism)
    q_dim, k_dim, v_dim = case["q_dim"], case["k_dim"], case["v_dim"]
    state_ref = case["conv_state"].clone()
    golden = packed_reference(
        conv_reference(case["x"], case["weight"], state_ref, lengths,
                       case["cache_indices"].tolist(), ism),
        q_dim, k_dim, v_dim, torch.bfloat16)
    host_state = case["conv_state"].clone().to(dev)
    host = op_call(torch.ops.ascend_ops.causal_conv1d_qkv, host_state,
                   case["x"].to(dev), case["weight"].to(dev), q_dim, k_dim,
                   v_dim, None, case["query_start_loc"], case["cache_indices"],
                   case["initial_state_mode"]).cpu()
    torch.npu.synchronize()
    c_host = cosine(host, golden)
    r_host = rel_l2(host, golden)
    host_state_ok = torch.equal(host_state.cpu(), state_ref)
    print(f"{'host-metadata eager len=' + str(lengths):<48} "
          f"cos={c_host:.6f} rel_l2={r_host:.6f} | "
          f"conv_states {'ok' if host_state_ok else 'MISMATCH'}")
    record("hostmeta eager cos", c_host > 0.95, f"cos={c_host:.6f}")
    record("hostmeta eager rel_l2", r_host < 0.02, f"rel_l2={r_host:.6f}")
    record("hostmeta eager conv_states", host_state_ok, "state mismatch")

    # --- middle-omission graph regression ----------------------------------
    # The GE wrapper reads OpExecuteContext inputs by IR index, so an absent
    # cache_indices must not shift initial_state_mode into its slot. With
    # cache_indices absent upstream maps to identity slots 0..batch-1.
    q_heads, v_heads, lengths, ism = (3, 9, (48, 50), (1, 0))
    case = build_case(q_heads, v_heads, lengths, ism)
    q_dim, k_dim, v_dim = case["q_dim"], case["k_dim"], case["v_dim"]
    identity = list(range(len(lengths)))
    state_ref = case["conv_state"].clone()
    golden = packed_reference(
        conv_reference(case["x"], case["weight"], state_ref, lengths, identity,
                       ism), q_dim, k_dim, v_dim, torch.bfloat16)
    x_dev, w_dev = case["x"].to(dev), case["weight"].to(dev)
    qsl_dev = case["query_start_loc"].to(dev)
    ism_dev = case["initial_state_mode"].to(dev)
    e_state = case["conv_state"].clone().to(dev)
    g_state = case["conv_state"].clone().to(dev)
    e_out = op_call(torch.ops.ascend_ops.causal_conv1d_qkv, e_state, x_dev,
                    w_dev, q_dim, k_dim, v_dim, None, qsl_dev, None, ism_dev)
    torch.npu.synchronize()
    before = frames_ok()
    with torch.no_grad():
        g_out = op_call(compiled, g_state, x_dev, w_dev, q_dim, k_dim, v_dim,
                        None, qsl_dev, None, ism_dev)
    torch.npu.synchronize()
    middle_traced = frames_ok() > before
    e_out, g_out = e_out.cpu(), g_out.cpu()
    middle_y_equal = torch.equal(g_out, e_out)
    middle_state_equal = torch.equal(g_state.cpu(), e_state.cpu())
    c_mid, r_mid = cosine(e_out, golden), rel_l2(e_out, golden)
    print(f"{'middle-omission (ci absent) len=' + str(lengths):<48} "
          f"eager cos={c_mid:.6f} rel_l2={r_mid:.6f} | "
          f"graph==eager y={middle_y_equal} state={middle_state_equal} | "
          f"traced={middle_traced}")
    record("middle-omission eager cos", c_mid > 0.95, f"cos={c_mid:.6f}")
    record("middle-omission eager rel_l2", r_mid < 0.02, f"rel_l2={r_mid:.6f}")
    record("middle-omission eager conv_states",
           torch.equal(e_state.cpu(), state_ref), "state mismatch")
    record("middle-omission graph==eager y", middle_y_equal, "y mismatch")
    record("middle-omission graph==eager conv_states", middle_state_equal,
           "state mismatch")
    record("middle-omission graph conv_states",
           torch.equal(g_state.cpu(), state_ref), "state mismatch")
    record("middle-omission graph compiled", middle_traced,
           "no new graph traced -> the graph leg fell back to eager")

    # --- multi-step decode: the state written by step 1 is step 2's history --
    # This is the in-place contract the whole design rests on; the chained
    # golden reuses the state mutation between calls.
    q_heads, v_heads = 3, 3
    q_dim = k_dim = q_heads * HEAD_DIM
    v_dim = v_heads * HEAD_DIM
    dim = q_dim + k_dim + v_dim
    step1, step2 = 40, 1
    gen = torch.Generator().manual_seed(20260717 + dim)
    x1 = (torch.randn((step1, dim), generator=gen) * 0.25).to(torch.bfloat16)
    x2 = (torch.randn((step2, dim), generator=gen) * 0.25).to(torch.bfloat16)
    weight = (torch.randn((WIDTH, dim), generator=gen) * 0.25
              ).to(torch.bfloat16)
    state0 = (torch.randn((3, WIDTH - 1, dim), generator=gen) * 0.1
              ).to(torch.bfloat16)
    state_ref = state0.clone()
    out1 = conv_reference(x1, weight, state_ref, [step1], [1], [0])
    out2 = conv_reference(x2, weight, state_ref, [step2], [1], [1])
    golden1 = packed_reference(out1, q_dim, k_dim, v_dim, torch.bfloat16)
    golden2 = packed_reference(out2, q_dim, k_dim, v_dim, torch.bfloat16)
    x1d, x2d, wd = x1.to(dev), x2.to(dev), weight.to(dev)
    qsl1 = torch.tensor([0, step1], dtype=torch.int64, device=dev)
    qsl2 = torch.tensor([0, step2], dtype=torch.int64, device=dev)
    ci_d = torch.tensor([1], dtype=torch.int64, device=dev)
    ism0 = torch.tensor([0], dtype=torch.int64, device=dev)
    ism1 = torch.tensor([1], dtype=torch.int64, device=dev)
    e_state = state0.clone().to(dev)
    e1 = op_call(torch.ops.ascend_ops.causal_conv1d_qkv, e_state, x1d, wd,
                 q_dim, k_dim, v_dim, None, qsl1, ci_d, ism0)
    e2 = op_call(torch.ops.ascend_ops.causal_conv1d_qkv, e_state, x2d, wd,
                 q_dim, k_dim, v_dim, None, qsl2, ci_d, ism1)
    torch.npu.synchronize()
    g_state = state0.clone().to(dev)
    before = frames_ok()
    with torch.no_grad():
        g1 = op_call(compiled, g_state, x1d, wd, q_dim, k_dim, v_dim, None,
                     qsl1, ci_d, ism0)
        g2 = op_call(compiled, g_state, x2d, wd, q_dim, k_dim, v_dim, None,
                     qsl2, ci_d, ism1)
    torch.npu.synchronize()
    chain_traced = frames_ok() > before
    e1, e2, g1, g2 = e1.cpu(), e2.cpu(), g1.cpu(), g2.cpu()
    c_e1, r_e1 = cosine(e1, golden1), rel_l2(e1, golden1)
    c_e2, r_e2 = cosine(e2, golden2), rel_l2(e2, golden2)
    c_g1, r_g1 = cosine(g1, golden1), rel_l2(g1, golden1)
    c_g2, r_g2 = cosine(g2, golden2), rel_l2(g2, golden2)
    e_state_ok = torch.equal(e_state.cpu(), state_ref)
    g_state_ok = torch.equal(g_state.cpu(), state_ref)
    print(f"{'multi-step decode ' + str(step1) + '+' + str(step2):<48} "
          f"eager cos={c_e1:.6f}/{c_e2:.6f} rel_l2={r_e1:.6f}/{r_e2:.6f} | "
          f"graph cos={c_g1:.6f}/{c_g2:.6f} | "
          f"state eager={'ok' if e_state_ok else 'MISMATCH'} "
          f"graph={'ok' if g_state_ok else 'MISMATCH'} | "
          f"traced={chain_traced}")
    for label, ok, detail in (
            ("multi-step step1 eager cos", c_e1 > 0.95, f"cos={c_e1:.6f}"),
            ("multi-step step2 eager cos", c_e2 > 0.95, f"cos={c_e2:.6f}"),
            ("multi-step step1 eager rel_l2", r_e1 < 0.02, f"rel_l2={r_e1:.6f}"),
            ("multi-step step2 eager rel_l2", r_e2 < 0.02, f"rel_l2={r_e2:.6f}"),
            ("multi-step step1 graph rel_l2", r_g1 < 0.001,
             f"rel_l2={r_g1:.8f}"),
            ("multi-step step2 graph rel_l2", r_g2 < 0.001,
             f"rel_l2={r_g2:.8f}"),
            ("multi-step eager conv_states", e_state_ok, "state mismatch"),
            ("multi-step graph conv_states", g_state_ok, "state mismatch"),
            ("multi-step graph compiled", chain_traced,
             "no new graph traced -> the graph leg fell back to eager")):
        record(label, ok, detail)

    # --- negative legs: every one must fail loud ---------------------------
    # Upstream owns the validation text/code (the bridge mirror was removed),
    # so only assert that an exception is raised, never a specific message.
    case = build_case(1, 3, (16,), (0,))
    nq, nk, nv = case["q_dim"], case["k_dim"], case["v_dim"]
    nx, nw = case["x"].to(dev), case["weight"].to(dev)
    nqsl = case["query_start_loc"].to(dev)
    nci = case["cache_indices"].to(dev)
    nism = case["initial_state_mode"].to(dev)

    def ncall(state, *, x=nx, w=nw, q=nq, k=nk, v=nv, run_mode=0, nat=None):
        return op_call(torch.ops.ascend_ops.causal_conv1d_qkv, state, x, w,
                       q, k, v, None, nqsl, nci, nism, run_mode, nat)

    def nstate():
        return case["conv_state"].clone().to(dev)

    def expect_fail(label, thunk):
        try:
            thunk()
            torch.npu.synchronize()
        except Exception as exc:  # generic on purpose: upstream owns the text
            print(f"{'neg ' + label:<48} raised {type(exc).__name__}")
        else:
            record("neg " + label, False, "accepted silently")

    expect_fail("run_mode=1", lambda: ncall(nstate(), run_mode=1))
    expect_fail("fp16 x", lambda: ncall(nstate(), x=nx.to(torch.float16)))
    expect_fail("3-D x", lambda: ncall(nstate(), x=nx.unsqueeze(0)))
    expect_fail("q_dim != k_dim",
                lambda: ncall(nstate(), q=nq, k=2 * nk,
                              v=nx.size(1) - nq - 2 * nk))
    bad_dim = nq + nk + 320  # 320 is not a multiple of head_dim
    bad_gen = torch.Generator().manual_seed(4242)
    bad_x = (torch.randn((16, bad_dim), generator=bad_gen) * 0.25
             ).to(torch.bfloat16).to(dev)
    bad_w = (torch.randn((WIDTH, bad_dim), generator=bad_gen) * 0.25
             ).to(torch.bfloat16).to(dev)
    bad_state = (torch.randn((3, WIDTH - 1, bad_dim), generator=bad_gen) * 0.1
                 ).to(torch.bfloat16).to(dev)
    expect_fail("dim not multiple of head_dim",
                lambda: ncall(bad_state, x=bad_x, w=bad_w, v=320))
    nat = torch.tensor([1], dtype=torch.int64, device=dev)
    expect_fail("num_accepted_tokens run_mode=0",
                lambda: ncall(nstate(), nat=nat))

    if failures:
        print(f"FAIL ({failures} checks)")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
