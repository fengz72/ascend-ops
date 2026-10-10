"""Prefix-in-Q attention, DN (K@Q^T) variant.

Same host contract and work-item/metadata machinery as ``prefix_attn``
(the ND production kernel); the cube/vector data path is rotated into the
DN orientation validated by ``prob/probe_dn_bisect.py`` (V2):

  - MM1 computes S^T = K @ Q^T (l0a<-K, l0b<-Q); FIXPIPE uses an
    ``split_axis=1`` engine (dual destination), so each AIV slot receives
    the [128 KV, 64 Q] half for its FIXED query range [64*subblock, +64).
  - The lane-wise two-pass softmax keeps max/sum as packed 64-lane state
    (lane == query).  Pass A accumulates the per-query max along KV with a
    bounds mask (``varange`` for causal modes, ``vdups(p)`` for the fully
    visible prefix) and deferred scale; pass B re-reads the scores, applies
    exp, accumulates four independent sum chains, and packs cast results
    in registers before writing P^T directly in NZ order.
  - The Vec softmax writes P^T directly in NZ physical order, then each AIV
    stores its 64 query columns of the shared L1 P tile (``split_axis=1``
    copy).  Keeping the cast and NZ store in one VF avoids a second UB read.
  - PV loads P^T into a dedicated L0A channel with ``transpose=True`` (the
    QK's K view and the transposed P^T view must NOT share one L0A channel
    on this wheel - the shared channel taints the QK matmul's L0C view).
  - O stays [Q, D]; the O emit uses a fixed 64-row split-M engine so
    AIV i's output rows match its fixed query range.  Tail Q tiles clamp
    each AIV's live rows (``qrows - 64*subblock``).

K/V use two GM spans per Channel transaction to assemble a fixed [128, 64]
L1 tile in logical prefix-plus-private order using two ordinary mem_copy
calls into row windows of one selected slot. Empty spans are guarded before
constructing views, and unused rows are initialized directly in L1. Softmax rewrites all 128 P^T rows and masks by the
logical query position and the tile's live KV extent.
"""

import math

import torch
from cannbotdsl import (
    ChannelKind, Dim, MemLoc, TensorSpec, compile, dtypes, permute,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.ops.reg.reg_defs import PackMode
from cannbotdsl.ops.sync import PIPE, cube_fill_l1_zero, cube_sync_pipe
from cannbotdsl.tensor import ceil_div, reinterpret, tile_slice
from cannbotdsl.types.delay_line import DelayLineGroup
from .validate import DesignError, _validate
from .metadata import (
    CORE_RECORD_STRIDE, HEAD_STRIDE, REC_BN2_END_INDEX, REC_BN2_START_INDEX,
    REC_M_END_INDEX, REC_M_START_INDEX, metadata_padded_count,
)

HEAD_DIM = 64


@jit
def _dn_pass_a(qk_ch, scale, sm_max, sm_exp, first, causal, row_base, p_len,
               kv_valid, full_visible=False):
    """DN pass A: per-query max along KV, deferred scale, packed merge."""
    with vf(mode="simd"):
        mask32 = rr.full_mask()
        neg = rr.vdups(-1.0e30, dtypes.float32, mask=mask32)
        if const_expr(full_visible):
            bounds = rr.vdups(0, dtypes.int32, mask=mask32)
        elif const_expr(causal):
            # Causal bound clamped at the tile's live rows: zero-padded K
            # rows would otherwise contribute exp(0 - max) mass (and can
            # overflow fp16 when the real max is very negative).
            bounds = rr.varange(row_base + 1, dtypes.int32)
            bounds = rr.vmins(bounds, kv_valid, mask=mask32)
        else:
            # Fully-visible prefix tile: the row limit IS the tile's live
            # rows (row >= valid means key >= p on this tile).
            bounds = rr.vdups(kv_valid, dtypes.int32, mask=mask32)
        stride = qk_ch.physical_stride[0]
        block_max = neg
        for row in range(0, qk_ch.shape[0], 4):
            v0 = rr.vload(qk_ch, (row + 0) * stride)
            v1 = rr.vload(qk_ch, (row + 1) * stride)
            v2 = rr.vload(qk_ch, (row + 2) * stride)
            v3 = rr.vload(qk_ch, (row + 3) * stride)
            if not const_expr(full_visible):
                v0 = rr.vselect(v0, neg, cond_mask=rr.vgts(bounds, row + 0, mask=mask32))
                v1 = rr.vselect(v1, neg, cond_mask=rr.vgts(bounds, row + 1, mask=mask32))
                v2 = rr.vselect(v2, neg, cond_mask=rr.vgts(bounds, row + 2, mask=mask32))
                v3 = rr.vselect(v3, neg, cond_mask=rr.vgts(bounds, row + 3, mask=mask32))
            m01 = rr.vmax(v0, v1, mask=mask32)
            m23 = rr.vmax(v2, v3, mask=mask32)
            block_max = rr.vmax(block_max, rr.vmax(m01, m23, mask=mask32),
                                mask=mask32)
        block_max = rr.vmuls(block_max, scale, mask=mask32)
        if const_expr(first):
            new_max = block_max
        else:
            old_max = rr.vload(sm_max, 0)
            new_max = rr.vmax(old_max, block_max, mask=mask32)
            corr = rr.vexp_sub(old_max, new_max, mask=mask32)
            rr.vstore(sm_exp, 0, corr, mask32)
        rr.vstore(sm_max, 0, new_max, mask32)


@jit
def _dn_pass_b(qk_ch, p_nz, scale, sm_max, sm_sum, sm_exp, first, causal,
               row_base, p_len, kv_valid, is_fp16, full_visible=False):
    """DN pass B: exp vs merged max, 4-chain sums, direct NZ P stores.

    The FIA kernel casts and writes the P tile in NZ order inside the same
    vector pass.  Do the same here instead of materializing an ND tile and
    reading it back in a second ``_dn_pack_p`` pass.
    """
    with vf(mode="simd"):
        mask32 = rr.full_mask()
        mask16, _ = rr.update_mask(64, elem_bits=16)
        if const_expr(full_visible):
            bounds = rr.vdups(0, dtypes.int32, mask=mask32)
        elif const_expr(causal):
            bounds = rr.varange(row_base + 1, dtypes.int32)
            bounds = rr.vmins(bounds, kv_valid, mask=mask32)
        else:
            bounds = rr.vdups(kv_valid, dtypes.int32, mask=mask32)
        stride = qk_ch.physical_stride[0]
        new_max = rr.vload(sm_max, 0)
        corr = rr.vdups(1.0, dtypes.float32, mask=mask32)
        if not const_expr(first):
            corr = rr.vload(sm_exp, 0)
        zero = rr.vdups(0.0, dtypes.float32, mask=mask32)
        sum0, sum1, sum2, sum3 = zero, zero, zero, zero
        for row in range(0, qk_ch.shape[0], 4):
            v0 = rr.vmuls(rr.vload(qk_ch, (row + 0) * stride), scale, mask=mask32)
            v1 = rr.vmuls(rr.vload(qk_ch, (row + 1) * stride), scale, mask=mask32)
            v2 = rr.vmuls(rr.vload(qk_ch, (row + 2) * stride), scale, mask=mask32)
            v3 = rr.vmuls(rr.vload(qk_ch, (row + 3) * stride), scale, mask=mask32)
            if const_expr(full_visible):
                m0, m1, m2, m3 = mask32, mask32, mask32, mask32
            else:
                m0 = rr.vgts(bounds, row + 0, mask=mask32)
                m1 = rr.vgts(bounds, row + 1, mask=mask32)
                m2 = rr.vgts(bounds, row + 2, mask=mask32)
                m3 = rr.vgts(bounds, row + 3, mask=mask32)
            # Masked vexp_sub zeros inactive lanes; the separate four
            # vselect(-inf) instructions are unnecessary here.
            e0 = rr.vexp_sub(v0, new_max, mask=m0)
            e1 = rr.vexp_sub(v1, new_max, mask=m1)
            e2 = rr.vexp_sub(v2, new_max, mask=m2)
            e3 = rr.vexp_sub(v3, new_max, mask=m3)
            sum0 = rr.vadd(sum0, e0, mask=mask32)
            sum1 = rr.vadd(sum1, e1, mask=mask32)
            sum2 = rr.vadd(sum2, e2, mask=mask32)
            sum3 = rr.vadd(sum3, e3, mask=mask32)
            if const_expr(is_fp16):
                c0 = rr.vcast(e0, dtypes.float16, mask=mask32)
                c1 = rr.vcast(e1, dtypes.float16, mask=mask32)
                c2 = rr.vcast(e2, dtypes.float16, mask=mask32)
                c3 = rr.vcast(e3, dtypes.float16, mask=mask32)
            else:
                c0 = rr.vcast(e0, dtypes.bfloat16, mask=mask32)
                c1 = rr.vcast(e1, dtypes.bfloat16, mask=mask32)
                c2 = rr.vcast(e2, dtypes.bfloat16, mask=mask32)
                c3 = rr.vcast(e3, dtypes.bfloat16, mask=mask32)
            # Narrow casts occupy even halfword lanes. Compact them in
            # registers before writing the first 64 halfwords to NZ.
            c0, unused0 = rr.vdeinterleave(c0, c0)
            c1, unused1 = rr.vdeinterleave(c1, c1)
            c2, unused2 = rr.vdeinterleave(c2, c2)
            c3, unused3 = rr.vdeinterleave(c3, c3)
            # p_nz is declared as NZ, but the declaration alone does not
            # change a vector store's layout.  Emit the same block/stride
            # pattern as FIA's StoreAlign(DATA_BLOCK_COPY): each logical row
            # starts a 16-element NZ block and successive blocks advance by
            # the physical N1 stride.
            dst_stride = p_nz.physical_stride[0]
            block_stride = dst_stride * 16 // 256
            rr.vstore_strided(p_nz, (row + 0) * 16, c0, mask16,
                              block_stride=block_stride, repeat_stride=0)
            rr.vstore_strided(p_nz, (row + 1) * 16, c1, mask16,
                              block_stride=block_stride, repeat_stride=0)
            rr.vstore_strided(p_nz, (row + 2) * 16, c2, mask16,
                              block_stride=block_stride, repeat_stride=0)
            rr.vstore_strided(p_nz, (row + 3) * 16, c3, mask16,
                              block_stride=block_stride, repeat_stride=0)
        block_sum = rr.vadd(
            rr.vadd(sum0, sum1, mask=mask32),
            rr.vadd(sum2, sum3, mask=mask32), mask=mask32,
        )
        if const_expr(first):
            rr.vstore(sm_sum, 0, block_sum, mask32)
        else:
            rr.vstore(sm_sum, 0,
                      rr.vmadd(rr.vload(sm_sum, 0), corr, block_sum,
                               mask=mask32), mask32)
        # Preserve store visibility before the P channel's L1 handoff.
        # Removing this barrier passed precision but regressed throughput
        # for the longer UNIFORM shapes on the 28-core profile.
        rr.vmem_bar("vst_vld")


@jit
def _dn_store_p(p_l1, p_ub, engine, subblock):
    # Axis-1 split over the statically declared (128, 128) L1 P tile: the
    # full/actual extents are both static, so the engine's runtime plan is
    # inferred from the channel's declared shape.
    mem_copy(p_l1, p_ub, engine=engine, part_id=subblock)


@jit
def _dn_native_row_tile(gm, head, token, rows, capacity, width):
    """Drop the singleton head axis without packing the GM token stride."""
    span = gm[head, token:token + rows, 0:width]
    # A rank-2 tile selects the trailing axes and preserves their strides.
    # Tensor.view(shape) instead promises packed storage and is invalid for
    # the interleaved heads in the permuted [N, T, D] GM layout.
    return tile_slice(span, (capacity, width), (0, 0))


@jit
def _dn_load_segments(slot, gm, head, base, valid, prefix_rows, private_base, engine):
    """Assemble K/V in one fixed NZ L1 frame, without UB or a merge API."""
    # PV consumes both 64-row halves. Zero the physical tail as well: a
    # masked probability of zero must not multiply stale NaN/Inf V data.
    if valid < 128:
        cube_fill_l1_zero(slot, repeat=4, blk_num=128, dst_gap=0)
        cube_sync_pipe(PIPE.MTE2)
    if prefix_rows > 0:
        prefix = _dn_native_row_tile(gm, head, base, prefix_rows, 128, 64)
        mem_copy(slot[0:prefix_rows, None], prefix, engine=engine)
    if valid > prefix_rows:
        private = _dn_native_row_tile(gm, head, private_base,
                                      valid - prefix_rows, 128, 64)
        mem_copy(slot[prefix_rows:valid, None], private, engine=engine)


@jit
def _dn_update(pv_ch, old_scale_buf, sum_buf, res_o, finalize):
    """``new_o = old_o*old_scale + tile_pv`` (optionally / sum)."""
    with vf(mode="simd"):
        for row in range(pv_ch.shape[0]):
            old_scale = rr.vload_broadcast(old_scale_buf, row)
            mask, _ = rr.update_mask(64, elem_bits=32)
            base = row * 64
            old_part = rr.vmul(rr.vload(res_o, base), old_scale, mask=mask)
            value = rr.vadd(old_part, rr.vload(pv_ch, base), mask=mask)
            if const_expr(finalize):
                denom = rr.vload_broadcast(sum_buf, row)
                value = rr.vdiv(value, denom, mask=mask)
            rr.vstore(res_o, base, value, mask)


@jit
def _dn_init_o(pv_ch, res_o):
    with vf(mode="simd"):
        for row in range(pv_ch.shape[0]):
            base = row * 64
            mask, _ = rr.update_mask(64, elem_bits=32)
            rr.vstore(res_o, base, rr.vload(pv_ch, base), mask)


@jit
def _dn_normalize_o(sum_buf, res_o, rows):
    with vf(mode="simd"):
        for row in range(rows):
            denom = rr.vload_broadcast(sum_buf, row)
            mask, _ = rr.update_mask(64, elem_bits=32)
            base = row * 64
            rr.vstore(res_o, base,
                      rr.vdiv(rr.vload(res_o, base), denom, mask=mask),
                      mask)


@jit
def _dn_cast_o(o_ub, res_o, rows):
    """Cast the live res_o rows into the dtype16 output channel slot.

    Mirrors the arena flash_attn ``_cast_nd_rows``: dynamic row count,
    physical strides from each operand, and a trailing ``vst_vld``
    barrier so the consuming ``mem_copy`` observes the packed stores.
    """
    with vf(mode="simd"):
        full, _ = rr.update_mask(64, elem_bits=32)
        src_stride = res_o.physical_stride[0]
        dst_stride = o_ub.physical_stride[0]
        for row in range(rows):
            value = rr.vload(res_o, row * src_stride)
            narrowed = rr.vcast(value, o_ub.dtype, mask=full, rounding="rn")
            rr.vstore_pack(o_ub, row * dst_stride, narrowed, full,
                           pack_mode=PackMode.B32_TO_B16)
        rr.vmem_bar("vst_vld")


@jit
def _dn_finish_o(pv, old_scale_buf, sum_buf, res_o, o_ub, rows, first):
    """Finish live rows directly into one selected output slot."""
    with vf(mode="simd"):
        mask, _ = rr.update_mask(64, elem_bits=32)
        for row in range(rows):
            value = rr.vload(pv, row * pv.physical_stride[0])
            if not const_expr(first):
                correction = rr.vload_broadcast(old_scale_buf, row)
                old = rr.vload(res_o, row * res_o.physical_stride[0])
                value = rr.vadd(rr.vmul(old, correction, mask=mask), value, mask=mask)
            denom = rr.vload_broadcast(sum_buf, row)
            value = rr.vdiv(value, denom, mask=mask)
            narrowed = rr.vcast(value, o_ub.dtype, mask=mask, rounding="rn")
            rr.vstore_pack(o_ub, row * o_ub.physical_stride[0], narrowed,
                           mask, pack_mode=PackMode.B32_TO_B16)
        rr.vmem_bar("vst_vld")


@jit
def _dn_write_output(out_gm, qhead, qstart, q_offset, live, res_o, o_ub):
    # One AIV may have no live query rows; do not construct an empty GM view.
    if live > 0:
        _dn_cast_o(o_ub, res_o, live)
        o_tile = _dn_native_row_tile(out_gm, qhead, qstart + q_offset, live,
                                     64, 64)
        mem_copy(o_tile, reinterpret(o_ub, shape=(live, 64), stride=(64, 1)))


class _DnMatmul:
    """DN cube path: S^T = K@Q^T (dual-dst split-N) and O = P@V."""

    def __init__(self, dtype16):
        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        # L0C -> UB dual-destination fixpipes: split_axis selects the
        # two-part routing axis (1 = N/query columns for S^T, 0 = M/output
        # rows for O); the hardware delivers each half to its sub-block.
        self.fixpipe_s = make_copy_engine(split_axis=1)
        self.fixpipe_o = make_copy_engine(split_axis=0)
        self.q_l1 = Channel(MemLoc.L1, (128, 64), dtype16, depth=2)
        self.k_l1 = Channel(MemLoc.L1, (128, 64), dtype16, depth=2)
        self.v_l1 = Channel(MemLoc.L1, (128, 64), dtype16, depth=2)
        # The QK's K view and the PV's transposed P^T view must not share an
        # L0A channel: the wheel's view unification then rewrites the QK
        # matmul's L0C destination as the (64,128) emit piece and the
        # verifier rejects it (probe_dn_bisect V2 vs V5).
        self.l0a_qk = Channel(MemLoc.L0A, (128, 64), dtype16, depth=2)
        # PV splits S2 in halves so its A operand fits a depth-2 channel: a
        # depth-1 (128,128) slot is released at MMAD issue time and the
        # next tile's transposed load can overwrite data the MMAD is still
        # reading (timing-dependent corruption, confirmed in the generated
        # .asc).  A transposed-load L0A operand is declared (K, M) in the
        # "zn" fractal orientation (arena flash_kda / quant_block_sparse
        # pattern), so the matmul reads K from dim 0.
        self.l0a_pv = Channel(
            MemLoc.L0A, (64, 128), dtype16, depth=2, data_format="zn",
        )
        self.l0b = Channel(MemLoc.L0B, (128, 64), dtype16, depth=2)
        self.l0b_pv = Channel(MemLoc.L0B, (64, 64), dtype16, depth=2)
        self.l0c = Channel(MemLoc.L0C, (128, 128), dtypes.float32, depth=2)

    @jit
    def load_q(self, q):
        slot = self.q_l1.produce()
        if q.shape[0] < 128:
            cube_fill_l1_zero(slot, repeat=4, blk_num=128, dst_gap=0)
            cube_sync_pipe(PIPE.MTE2)
        mem_copy(slot[0:q.shape[0], None], q, engine=self.nd2nz)
        self.q_ready = self.q_l1.consume()

    def qk(self):
        # S^T = K @ Q^T with the result view threaded to the S emit
        # (flash_attn pattern: the matmul destination and the fixpipe
        # source are the same packed view).  The copy destination and the
        # matmul lhs are the same L0A view.
        k_l0a = reinterpret(self.l0a_qk.produce(), shape=(128, 64))
        s_l0c = reinterpret(self.l0c.produce(), shape=(128, 128))
        mem_copy(k_l0a, self.k_l1.consume())
        q_l0b = self.l0b.produce()
        mem_copy(q_l0b, self.q_ready)
        matmul(s_l0c, k_l0a, q_l0b, init=True)
        return s_l0c

    def pv(self, p_l1):
        # O = P @ V accumulated over two S2 halves in one L0C epoch: each
        # half is a static sub-tile view of the shared L1 channels, and
        # the dynamic-init matmul lowers to the DSL's K-loop accumulate
        # form (both MMADs share one channel transaction).  The transposed
        # P^T load lands in the (K, M) zn L0A channel, and the result view
        # is also the O emit's source (flash_attn pattern).
        o_l0c = reinterpret(self.l0c.produce(), shape=(128, 64))
        v_l1 = self.v_l1.consume()
        for half in range(2):
            p_piece = tile_slice(p_l1, (64, 128), (half, 0))
            v_piece = tile_slice(v_l1, (64, 64), (half, 0))
            p_l0a = self.l0a_pv.produce()
            v_l0b = self.l0b_pv.produce()
            mem_copy(p_l0a, p_piece, transpose=True)
            mem_copy(v_l0b, v_piece, transpose=True)
            matmul(o_l0c, p_l0a, v_l0b, init=(half == 0))
        return o_l0c


class _DnVector:
    """Packed online-softmax state and P staging for one vector half."""

    def __init__(self, dtype16):
        self.p_ub = Channel(
            MemLoc.UB, (128, 64), dtype16, depth=2,
            data_format="nz", n1_pad=16,
        )
        self.o_ub = Channel(MemLoc.UB, (64, 64), dtype16, depth=2)
        self.res_o = Buffer(MemLoc.UB, (64, 64), dtypes.float32)


@kernel
class _PrefixMacroKernelDn:
    """Metadata-driven DN pipeline: one launch per core."""

    def __init__(self, dtype16=dtypes.float16):
        self.is_fp16 = dtype16 == dtypes.float16
        self.subblock = get_subblock_id()
        self.block_idx = get_block_idx()
        self.preload_num = 3
        # The split-N emit has two AIV consumers.  Depth 2 is sufficient with
        # the four-stage delay line and keeps the cross-core UB/L1 footprint
        # within the channel budget; depth 3 was the source of intermittent
        # stale-slot reads in the long multi-batch path.
        self.qk_ub = Channel(
            MemLoc.UB, (128, 64), dtypes.float32, depth=2,
            kind=ChannelKind.CrossCore,
        )
        self.p_l1 = Channel(
            MemLoc.L1, (128, 128), dtype16, depth=2,
            kind=ChannelKind.CrossCore,
        )
        self.pv_ub = Channel(
            MemLoc.UB, (64, 64), dtypes.float32, depth=2,
            kind=ChannelKind.CrossCore,
        )
        self.matmul = _DnMatmul(dtype16)
        self.vector = _DnVector(dtype16)
        # Packed state per in-flight work item (m_seq % 3): lane == query.
        self.max_slots = [
            Buffer(MemLoc.UB, (64, 1), dtypes.float32)
            for _ in range(self.preload_num)
        ]
        self.sum_slots = [
            Buffer(MemLoc.UB, (64, 1), dtypes.float32)
            for _ in range(self.preload_num)
        ]
        # old_scale written by pass A at tick t, consumed by update at t+2.
        self.exp_slots = [
            Buffer(MemLoc.UB, (64, 1), dtypes.float32)
            for _ in range(self.preload_num)
        ]
        # Static split plans: QK emit splits L0C N (fixed 64 per AIV,
        # actual inferred from the reaching matmul); the P store splits
        # the L1 tile's columns per AIV; the O emit splits L0C M with the
        # SAME fixed 64 rows so output rows match query lanes.
        self.split_cols = make_copy_engine(split_axis=1)

    @jit
    def _stage_qk(self, key, kv_head, base, valid, prefix_rows, private_base):
        slot = self.matmul.k_l1.produce()
        _dn_load_segments(slot, key, kv_head, base, valid, prefix_rows,
                          private_base, self.matmul.nd2nz)
        result = self.matmul.qk()
        # Static axis-1 dual-dst emit over the full (128, 128) S^T view:
        # each AIV slot receives its fixed 64 query columns.
        mem_copy(self.qk_ub.produce(), result, engine=self.matmul.fixpipe_s)

    @jit
    def _stage_softmax(self, scale, base, valid, first, slot, mslot,
                       row0, qrows):
        # row0/base 都属于 prefix + private 的逻辑序列；每个 AIV 固定负责 64 列。
        causal_row0 = row0 + self.subblock * 64 - base
        # 所有 P 行都重写，避免尾 tile 或不可见行残留上一轮的数据。
        qk = reinterpret(self.qk_ub.consume(), shape=(128, 64), stride=(64, 1))
        p_ub = self.vector.p_ub.produce()
        sm_max = self.max_slots[mslot]
        sm_sum = self.sum_slots[mslot]
        sm_exp = self.exp_slots[slot]
        kv_valid = dtypes.int32(valid)
        full_visible = (valid >= 128 if causal_row0 >= 127 else False)
        if full_visible:
            if first:
                _dn_pass_a(qk, scale, sm_max, sm_exp, True, 1,
                           causal_row0, 0, kv_valid, True)
                _dn_pass_b(qk, p_ub, scale, sm_max,
                           sm_sum, sm_exp, True, 1, causal_row0, 0,
                           kv_valid, self.is_fp16, True)
            else:
                _dn_pass_a(qk, scale, sm_max, sm_exp, False, 1,
                           causal_row0, 0, kv_valid, True)
                _dn_pass_b(qk, p_ub, scale, sm_max,
                           sm_sum, sm_exp, False, 1, causal_row0, 0,
                           kv_valid, self.is_fp16, True)
        else:
            if first:
                _dn_pass_a(qk, scale, sm_max, sm_exp, True, 1,
                           causal_row0, 0, kv_valid)
                _dn_pass_b(qk, p_ub, scale, sm_max,
                           sm_sum, sm_exp, True, 1, causal_row0, 0,
                           kv_valid, self.is_fp16)
            else:
                _dn_pass_a(qk, scale, sm_max, sm_exp, False, 1,
                           causal_row0, 0, kv_valid)
                _dn_pass_b(qk, p_ub, scale, sm_max,
                           sm_sum, sm_exp, False, 1, causal_row0, 0,
                           kv_valid, self.is_fp16)
        _dn_store_p(self.p_l1.produce(), p_ub, self.split_cols,
                    self.subblock)

    @jit
    def _stage_pv(self, value, kvhead, base, valid, prefix_rows, private_base):
        slot = self.matmul.v_l1.produce()
        _dn_load_segments(slot, value, kvhead, base, valid, prefix_rows,
                          private_base, self.matmul.nd2nz)
        result = self.matmul.pv(self.p_l1.consume())
        # Static axis-0 dual-dst emit over the packed (128, 64) result
        # view: AIV i's slot receives the fixed 64 rows matching its
        # query range.
        mem_copy(self.pv_ub.produce(), result, engine=self.matmul.fixpipe_o)

    @jit
    def _stage_update(self, out, first, last, slot, mslot, qstart, qrows,
                      qhead):
        q_offset = self.subblock * 64
        live = qrows - q_offset
        live = (0 if live < 0 else live)
        live = (64 if live > 64 else live)
        pv = reinterpret(self.pv_ub.consume(), shape=(live, 64), stride=(64, 1))
        if last:
            o_ub = self.vector.o_ub.produce()
            # Keep the PV wait/free handoff even for an AIV with zero live rows.
            # The VF loop is empty in that case; only GM output is guarded.
            if first:
                _dn_finish_o(pv, self.exp_slots[slot], self.sum_slots[mslot],
                             self.vector.res_o, o_ub, live, True)
            else:
                _dn_finish_o(pv, self.exp_slots[slot], self.sum_slots[mslot],
                             self.vector.res_o, o_ub, live, False)
            if live > 0:
                o_tile = _dn_native_row_tile(out, qhead, qstart + q_offset,
                                             live, 64, 64)
                mem_copy(o_tile, reinterpret(o_ub, shape=(live, 64), stride=(64, 1)))
        elif first:
            _dn_init_o(pv, self.vector.res_o)
        else:
            _dn_update(pv, self.exp_slots[slot], self.sum_slots[mslot],
                       self.vector.res_o, False)

    @jit
    def _gates(self, tick, issued, dl, scale, value, out):
        if tick >= 1 and tick - 1 < issued:
            self._stage_softmax(
                scale, dl.base.tap(1), dl.valid.tap(1),
                dl.first.tap(1), dl.slot.tap(1), dl.mslot.tap(1),
                dl.row0.tap(1), dl.qrows.tap(1),
            )
        if tick >= 2 and tick - 2 < issued:
            self._stage_pv(
                value, dl.kvhead.tap(2), dl.base.tap(2), dl.valid.tap(2),
                dl.prefix_rows.tap(2), dl.private_base.tap(2),
            )
        if tick >= 3 and tick - 3 < issued:
            self._stage_update(
                out, dl.first.tap(3), dl.last.tap(3), dl.slot.tap(3),
                dl.mslot.tap(3), dl.qstart.tap(3), dl.qrows.tap(3),
                dl.qhead.tap(3),
            )
        return 0

    @jit
    def _emit(self, dl, key, value, out, scale, base, valid, prefix_rows,
              private_base, first, last, mslot, row0, q_start, q_rows, q_head, kv_head,
              tick, issued, tile_seq):
        dl.push(base=base, valid=valid, prefix_rows=prefix_rows,
                private_base=private_base, first=first, last=last,
                slot=tile_seq % self.preload_num, mslot=mslot, row0=row0,
                qstart=q_start, qrows=q_rows, qhead=q_head, kvhead=kv_head)
        self._stage_qk(key, kv_head, base, valid, prefix_rows, private_base)
        issued = issued + 1
        self._gates(tick, issued, dl, scale, value, out)
        dl.advance()
        return tick + 1, issued, tile_seq + 1

    def __call__(self, out, query, key, value, scale, bounds, metadata):
        n1 = query.shape[0]
        n2 = key.shape[0]
        group = n1 // n2
        p = dtypes.int64(bounds[0])
        rec = HEAD_STRIDE + self.block_idx * CORE_RECORD_STRIDE
        bn2_start = dtypes.int64(metadata[rec + REC_BN2_START_INDEX])
        bn2_end = dtypes.int64(metadata[rec + REC_BN2_END_INDEX])
        m_start = dtypes.int64(metadata[rec + REC_M_START_INDEX])
        m_end = dtypes.int64(metadata[rec + REC_M_END_INDEX])
        dl = DelayLineGroup(
            4, "base", "valid", "prefix_rows", "private_base",
            "first", "last", "slot", "mslot", "row0", "qstart",
            "qrows", "qhead", "kvhead",
        )
        tick = 0
        issued = 0
        m_seq = 0
        tile_seq = 0
        # Hierarchical (bN2, m) box walk mirroring flash_attn's cursor
        # scheme: the outer loop advances whole units (unit 0 = shared
        # prefix segment, unit u >= 1 = private batch u-1) reading one
        # bounds pair per unit, and the inner loop walks the unit's
        # q_head-major rows m = q_head*bl + q_tile; the box's first/last
        # units clamp to m_start/m_end.  Channel transactions must stay
        # on one control-flow path so the FIFO ledger can resolve epochs;
        # an if/else split or two sequential inner loops makes the ledger
        # ambiguous ("no legal FIFO solution").  The drain must share the
        # nonempty-box guard with its producers (flash_attn does the
        # same): a core with no work must not advance the delay line at
        # all, or the AIC/AIV channel handshake deadlocks on the task.
        if bn2_start < bn2_end:
            for u in range(bn2_start, bn2_end):
                is_prefix = u == 0
                prev = u - 1
                prev = (0 if prev < 0 else prev)
                start_v = (0 if is_prefix else dtypes.int64(bounds[prev]))
                end_v = dtypes.int64(bounds[u])
                length_v = end_v - start_v
                bl = ceil_div(length_v, 128)
                m_lo = (m_start if u == bn2_start else 0)
                m_hi = (m_end if u == bn2_end - 1 else n1 * bl)
                for m in range(m_lo, m_hi):
                    q_head = m // bl
                    q_tile = m % bl
                    q_start = start_v + q_tile * 128
                    q_rows = min(128, length_v - q_tile * 128)
                    row0 = q_tile * 128 + (0 if is_prefix else p)
                    logical_length = (p if is_prefix else p + length_v)
                    ntiles = ceil_div(row0 + q_rows, 128)
                    kv_head = q_head // group
                    q_tile_gm = _dn_native_row_tile(
                        query, q_head, q_start, q_rows, 128, 64
                    )
                    self.matmul.load_q(q_tile_gm)
                    m_seq = m_seq + 1
                    mslot = m_seq % self.preload_num
                    for j in range(ntiles):
                        base = j * 128
                        valid = min(128, logical_length - base)
                        prefix_rows = max(0, min(valid, p - base))
                        private_base = start_v + max(0, base - p)
                        tick, issued, tile_seq = self._emit(
                            dl, key, value, out, scale, base, valid,
                            prefix_rows, private_base, 1 if j == 0 else 0,
                            1 if j == ntiles - 1 else 0,
                            mslot, row0, q_start, q_rows,
                            q_head, kv_head, tick, issued, tile_seq,
                        )
            for _ in range(self.preload_num):
                self._gates(tick, issued, dl, scale, value, out)
                dl.advance()
                tick = tick + 1


@host
def _launch_prefix_macro_dn(out, query, key, value, scale, bounds, metadata,
                            block_dim: int):
    qv = permute(query, (1, 0, 2))
    kv = permute(key, (1, 0, 2))
    vv = permute(value, (1, 0, 2))
    ov = permute(out, (1, 0, 2))
    if const_expr(query.dtype == dtypes.float16):
        op = _PrefixMacroKernelDn(dtype16=dtypes.float16)
    else:
        op = _PrefixMacroKernelDn(dtype16=dtypes.bfloat16)
    op[block_dim](ov, qv, kv, vv, scale, bounds, metadata)


_AOT_LAUNCH_CACHE_DN: dict = {}


def _compiled_launch_dn(n1, n2, dtype16):
    """Compile the DN macro launch once per (N1, N2, dtype) with dynamic T/B.

    All input lengths use bounded GM row views and zero-initialized fixed
    L1 frames, including T < 128; no legacy ND fallback is needed.
    """
    key = (n1, n2, dtype16)
    compiled = _AOT_LAUNCH_CACHE_DN.get(key)
    if compiled is None:
        t_dim = Dim("T", min=1)
        bounds_dim = Dim("B1", min=2)
        compiled = compile(_launch_prefix_macro_dn,
            TensorSpec((t_dim, n1, HEAD_DIM), dtype16),
            TensorSpec((t_dim, n1, HEAD_DIM), dtype16),
            TensorSpec((t_dim, n2, HEAD_DIM), dtype16),
            TensorSpec((t_dim, n2, HEAD_DIM), dtype16),
            float,
            TensorSpec((bounds_dim,), dtypes.int32),
            TensorSpec((metadata_padded_count(),), dtypes.int32),
            int,
        )
        _AOT_LAUNCH_CACHE_DN[key] = compiled
    return compiled


def prefix_attn_dn(query, key, value, actual_seq_lengths,
                   scale=None):
    """DN (K@Q^T) variant of :func:`prefix_attn` with the same contract."""
    _validate(query, key, value, actual_seq_lengths)
    if query.device.type != "npu":
        raise ValueError("prefix_attn_dn requires NPU-resident Q/K/V tensors")
    scale = 1.0 / math.sqrt(query.shape[2]) if scale is None else float(scale)
    from .metadata import (
        get_effective_core_counts, prefix_attn_metadata,
    )
    n1, n2 = query.shape[1], key.shape[1]
    if n1 < n2 or n1 % n2:
        raise DesignError("DESIGN_ERROR: N1 must be divisible by N2")
    stream = torch.npu.current_stream(query.device.index or 0)
    aic, _ = get_effective_core_counts(stream=stream)
    metadata = prefix_attn_metadata(actual_seq_lengths, n1, n2, dn_cost=True)
    out = torch.empty_like(query)
    bounds = actual_seq_lengths
    if (bounds.device.type != "npu" or bounds.dtype != torch.int32
            or not bounds.is_contiguous()):
        bounds = bounds.to(device="npu", dtype=torch.int32).contiguous()
    dtype16 = dtypes.float16 if query.dtype == torch.float16 else dtypes.bfloat16
    _compiled_launch_dn(n1, n2, dtype16)(
        out, query, key, value, float(scale), bounds, metadata, aic,
    )
    return out


__all__ = ["prefix_attn_dn"]
