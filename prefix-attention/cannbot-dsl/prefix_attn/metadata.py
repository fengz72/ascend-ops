"""AICPU compact-box metadata generator for prefix_attn.

The metadata ABI follows the arena flash_attn style: a flat int32 buffer
with a fixed physical layout.  AICPU plans a contiguous per-core
``(bN2, m)`` box over the unit/row space on the current NPU stream; each
AIC core walks its box hierarchically (flash_attn's cursor scheme) and
decodes every work item from ``(unit, row)`` with one div/mod pair and a
single per-unit bounds read - no per-item flat-index scan.

Unit space (the flash_attn bN2 axis with the KV-head dimension folded
into rows, since KV is never split): unit ``u=0`` is the shared prefix
``[0, bounds[0])`` and unit ``u>=1`` is private batch ``u-1``
``[bounds[u-1], bounds[u])``.  Unit ``u`` owns ``n1*bl_u`` rows
``m = q_head*bl_u + q_tile``, so the ``(unit, row)`` walk enumerates the
same flat work-item ordering as before (all PREFIX items q_head-major
then q_tile, then PRIVATE items batch-major, then q_head, then q_tile).

Layout (int32 elements)::

    HEAD[16]  + CORE_RECORD[36][16]

HEAD fields: [0]=section_num (always 1), [1]=aic_core_num, [2]=used_cores,
[3]=total_work_items.  Core record ``c`` lives at ``16 + c*16`` with
[0]=bn2_start, [1]=bn2_end (exclusive unit range), [2]=m_start (first
unit's first row), [3]=m_end (last unit's exclusive row bound, >= 1 for
a nonempty box; a chunk boundary landing on a unit start closes the
previous unit with its full extent).  An unused core keeps an all-zero
(empty) box.

This module requires the cannbotdsl AICPU frontend (remote NPU host only).
"""

import tempfile
from functools import lru_cache

from cannbotdsl.aicpu import GmIn, GmOut, I32, U32, I64, aicpu_kernel

AIC_CORE_NUM = 36
HEAD_STRIDE = 16
CORE_RECORD_STRIDE = 16

HEAD_SECTION_NUM_INDEX = 0
HEAD_CORE_NUM_INDEX = 1
HEAD_USED_CORES_INDEX = 2
HEAD_TOTAL_WORK_INDEX = 3

REC_BN2_START_INDEX = 0
REC_BN2_END_INDEX = 1
REC_M_START_INDEX = 2
REC_M_END_INDEX = 3

TILE = 128
# Row-equivalent fixed cost of one KV tile-step: the padded 128-row cube
# matmuls plus the per-tile channel transactions cost roughly 512 AIV
# cycles, i.e. ~32 rows of vector work at ~16.5 cycles per row.
COST_FIXED_ROWS = 32


def get_effective_core_counts(stream=None):
    """Return the AIC/AIV counts available to the launch stream."""
    import torch

    props = torch.npu.get_device_properties(torch.npu.current_device())
    cube = int(getattr(props, "cube_core_num", 0))
    vector = int(getattr(props, "vector_core_num", 0)) or 2 * cube
    try:
        from cannbotdsl import get_platform_info

        info = get_platform_info(stream=stream)
        cube = int(info.cube_core_num) or cube
        vector = int(info.vector_core_num) or vector
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        pass
    cube = min(cube, AIC_CORE_NUM)
    if cube <= 0 or vector < cube:
        raise RuntimeError(
            f"invalid effective NPU core counts: AIC={cube}, AIV={vector}"
        )
    return cube, vector


def metadata_element_count():
    return HEAD_STRIDE + AIC_CORE_NUM * CORE_RECORD_STRIDE


def metadata_padded_count():
    """Allocation length of the metadata GM buffer (4096-element aligned)."""
    return ((metadata_element_count() + 4095) // 4096) * 4096


class _PrefixAttnMetadataArgs:
    bounds: GmIn(I32)
    metadata: GmOut(U32)
    bounds_len: U32
    num_heads_q: U32
    num_heads_kv: U32
    aic_core_num: U32
    macro_cost: U32


@aicpu_kernel
def _prefix_attn_metadata_kernel(a: _PrefixAttnMetadataArgs):
    class Ctx:
        costs: array(I64)
        pref: array(I64)
        bound: array(U32)
        blen: array(I64)
        bfirst: array(I64)
        bn2s: array(I64)
        bn2e: array(I64)
        ms: array(I64)
        me: array(I64)

    n1 = a.num_heads_q
    B = a.bounds_len - 1
    p = a.bounds[0]
    ptiles = ceil_div(p, TILE)
    bl_sum = 0
    for b in range(B):
        bl_sum += ceil_div(a.bounds[b + 1] - a.bounds[b], TILE)
    total = n1 * ptiles + n1 * bl_sum

    ctx = Ctx()
    ctx.costs = zeros(I64, total + 1)
    wi = 0
    # Cost model mirrors the kernel's causally clamped scan: each work item
    # pays visible_tiles * (q_rows + COST_FIXED_ROWS).  Prefix items end at
    # tile qt+1; private items see all ptiles prefix tiles plus qt+1 private
    # tiles.
    for _qh in range(n1):
        for qt in range(ptiles):
            rows = min(TILE, p - qt * TILE)
            ctx.costs[wi] = (qt + 1) * (rows + COST_FIXED_ROWS)
            if a.macro_cost:
                ctx.costs[wi] = (qt + 1) * (TILE + COST_FIXED_ROWS)
            wi += 1
    for b in range(B):
        start = a.bounds[b]
        length = a.bounds[b + 1] - start
        bl = ceil_div(length, TILE)
        for _qh in range(n1):
            for qt in range(bl):
                rows = min(TILE, length - qt * TILE)
                ctx.costs[wi] = (ptiles + qt + 1) * (rows + COST_FIXED_ROWS)
                if a.macro_cost:
                    # Match the merged macro cursor: prefix and private rows
                    # share the boundary KV tile. Keep the same owner count
                    # and contiguous box ABI; change only the cost estimate.
                    visible = ceil_div(p + qt * TILE + rows, TILE)
                    ctx.costs[wi] = visible * (TILE + COST_FIXED_ROWS)
                wi += 1

    ctx.pref = zeros(I64, total + 1)
    for i in range(total):
        ctx.pref[i + 1] = ctx.pref[i] + ctx.costs[i]

    # The macro delay line is sensitive to a one- or two-item final owner when
    # a partial private tile is present.  Keep the launch at the requested
    # block count, but reserve one block for that tail in the macro planner.
    owner_limit = a.aic_core_num
    if a.macro_cost and owner_limit > 1:
        owner_limit = owner_limit - 1
    nowners = min(owner_limit, total)

    # Binary-search the smallest limit whose greedy chunk count fits in
    # nowners contiguous owners: the optimal contiguous min-max split.
    lo = 0
    for i in range(total):
        if ctx.costs[i] > lo:
            lo = ctx.costs[i]
    hi = ctx.pref[total]
    while lo < hi:
        mid = (lo + hi) // 2
        count = 1
        acc = 0
        for i in range(total):
            if acc + ctx.costs[i] > mid:
                count += 1
                acc = ctx.costs[i]
            else:
                acc += ctx.costs[i]
        if count <= nowners:
            hi = mid
        else:
            lo = mid + 1
    limit = lo

    # Emit the greedy chunks with that limit as sorted boundary points.
    ctx.bound = zeros(U32, nowners + 1)
    k = 0
    acc = 0
    for i in range(total):
        if acc + ctx.costs[i] > limit:
            k += 1
            ctx.bound[k] = i
            acc = ctx.costs[i]
        else:
            acc += ctx.costs[i]
    k += 1
    ctx.bound[k] = total

    # Refine: while owners are unused, split the first most expensive chunk
    # with at least two items at its most balanced point.  total >= nowners
    # guarantees such a chunk exists while k < nowners.
    while k < nowners:
        best_ci = -1
        best_cost = -1
        for ci in range(k):
            a0 = ctx.bound[ci]
            b0 = ctx.bound[ci + 1]
            if b0 - a0 >= 2:
                cc = ctx.pref[b0] - ctx.pref[a0]
                if cc > best_cost:
                    best_cost = cc
                    best_ci = ci
        a0 = ctx.bound[best_ci]
        b0 = ctx.bound[best_ci + 1]
        best_m = a0 + 1
        v1 = ctx.pref[a0 + 1] - ctx.pref[a0]
        v2 = ctx.pref[b0] - ctx.pref[a0 + 1]
        if v2 > v1:
            best_v = v2
        else:
            best_v = v1
        for m in range(a0 + 2, b0):
            v1 = ctx.pref[m] - ctx.pref[a0]
            v2 = ctx.pref[b0] - ctx.pref[m]
            if v2 > v1:
                v = v2
            else:
                v = v1
            if v < best_v:
                best_v = v
                best_m = m
        # Insert best_m after best_ci, shifting the tail right by one.
        tmax = k - best_ci - 1
        for t in range(tmax + 1):
            ctx.bound[k + 1 - t] = ctx.bound[k - t]
        ctx.bound[best_ci + 1] = best_m
        k += 1

    # Convert the flat chunk boundaries into per-core (bN2, m) boxes over
    # the unit/row space (flash_attn cursor scheme, KV axis unsplit).  A
    # chunk start lands on its first row's unit at m_start; the exclusive
    # end lands on the unit holding the chunk's LAST row, so m_end is that
    # unit's exclusive row bound (>= 1) and a boundary on a unit start
    # closes the previous unit with its full extent.
    units = 1 + B
    ctx.blen = zeros(I64, units)
    ctx.blen[0] = ptiles
    for b in range(B):
        ctx.blen[b + 1] = ceil_div(a.bounds[b + 1] - a.bounds[b], TILE)
    ctx.bfirst = zeros(I64, units + 1)
    for u in range(units):
        ctx.bfirst[u + 1] = ctx.bfirst[u] + n1 * ctx.blen[u]

    ctx.bn2s = zeros(I64, nowners)
    ctx.bn2e = zeros(I64, nowners)
    ctx.ms = zeros(I64, nowners)
    ctx.me = zeros(I64, nowners)
    for r in range(nowners):
        w0 = ctx.bound[r]
        w1 = ctx.bound[r + 1]
        u0 = 0
        u1 = 0
        for u in range(units):
            if w0 >= ctx.bfirst[u + 1]:
                u0 = u + 1
            if w1 > ctx.bfirst[u + 1]:
                u1 = u + 1
        ctx.bn2s[r] = u0
        ctx.bn2e[r] = u1 + 1
        ctx.ms[r] = w0 - ctx.bfirst[u0]
        ctx.me[r] = w1 - ctx.bfirst[u1]

    a.metadata[HEAD_SECTION_NUM_INDEX] = 1
    a.metadata[HEAD_CORE_NUM_INDEX] = a.aic_core_num
    a.metadata[HEAD_USED_CORES_INDEX] = nowners
    a.metadata[HEAD_TOTAL_WORK_INDEX] = total
    for r in range(nowners):
        base = HEAD_STRIDE + r * CORE_RECORD_STRIDE
        a.metadata[base + REC_BN2_START_INDEX] = ctx.bn2s[r]
        a.metadata[base + REC_BN2_END_INDEX] = ctx.bn2e[r]
        a.metadata[base + REC_M_START_INDEX] = ctx.ms[r]
        a.metadata[base + REC_M_END_INDEX] = ctx.me[r]
    return 0


@lru_cache(maxsize=1)
def _compiled_metadata():
    from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel

    directory = tempfile.TemporaryDirectory(prefix="cannbot_prefix_attn_metadata_")
    compiled = compile_aicpu_kernel(
        _prefix_attn_metadata_kernel,
        workdir=directory.name,
        launch_mode="interface",
    )
    return directory, compiled


def prefix_attn_metadata(bounds, num_heads_q, num_heads_kv, *, macro_cost=False):
    """Generate compact-range metadata on AICPU using the current NPU stream.

    ``macro_cost`` selects the padded-tile cost model of the macro pipeline;
    the default keeps the planner contract used by the ND path.
    """
    import torch
    from cannbotdsl.aicpu import current_raw_stream

    if not isinstance(bounds, torch.Tensor):
        raise TypeError("bounds must be a torch.Tensor")
    if bounds.ndim != 1 or bounds.numel() < 2:
        raise ValueError("bounds must be a 1-D [B+1] tensor")
    if bounds.dtype != torch.int32:
        bounds = bounds.to(torch.int32)
    if bounds.device.type != "npu":
        bounds = bounds.npu()
    if not bounds.is_contiguous():
        bounds = bounds.contiguous()
    if num_heads_kv <= 0 or num_heads_q < num_heads_kv or num_heads_q % num_heads_kv:
        raise ValueError("num_heads_q must be a positive multiple of num_heads_kv")

    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    stream = torch.npu.current_stream(device_id)
    _, compiled = _compiled_metadata()
    aic, _ = get_effective_core_counts(stream=stream)
    size = metadata_padded_count()
    metadata = torch.zeros(size, dtype=torch.int32, device=device)
    compiled.launch(
        current_raw_stream(device_id),
        bounds=bounds.data_ptr(),
        metadata=metadata.data_ptr(),
        bounds_len=bounds.numel(),
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        aic_core_num=aic,
        macro_cost=int(macro_cost),
    )
    bounds.record_stream(stream)
    return metadata

__all__ = [
    "AIC_CORE_NUM",
    "CORE_RECORD_STRIDE",
    "HEAD_STRIDE",
    "get_effective_core_counts",
    "metadata_element_count",
    "metadata_padded_count",
    "prefix_attn_metadata",
]
