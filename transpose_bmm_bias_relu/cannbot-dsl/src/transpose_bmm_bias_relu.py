# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fused BMM, bias and ReLU with runtime-dynamic M, batch, K and N.

A uses [M, batch, K]. B uses [batch, K, N] or [batch, N, K].
Each dtype / transpose_b combination compiles once per Python process.
"""

__all__ = ["transpose_bmm_bias_relu"]

import dataclasses
from functools import lru_cache
from typing import TYPE_CHECKING, TypeAlias

import torch
from cannbotdsl import (
    Dim,
    TensorSpec,
    compile as compile_program,
    dtypes,
    get_mem_size,
    get_platform_info,
    make_layout,
)
from cannbotdsl.aot.spec import _DimExpr
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import copy_options, cube_fill_l1_zero, mem_copy
from cannbotdsl.ops.sync import cube_sync_all
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import MemLoc, Tensor, ceil_div, from_tensor, tile_slice

if TYPE_CHECKING:
    from cannbotdsl.core.compiler.function import ProviderCallable


_INPUT_DIM = 3
_L2_CACHE_POLICY = 1
_PROGRAM_CACHE_SIZE = 4  # Two input dtypes and two B layouts.

Dimension: TypeAlias = int | Dim | _DimExpr


def _ceil_align(a: Dimension, b: int) -> Dimension:
    """Round up an integer, dynamic dimension or dimension expression."""
    if b <= 0:
        raise ValueError(f"alignment divisor must be positive; got {b}")
    return ((a + b - 1) // b) * b


_TORCH_DTYPE_TO_DSL = {
    torch.float16: (dtypes.float16, (dtypes.float16.bits // 8)),
    torch.bfloat16: (dtypes.bfloat16, (dtypes.bfloat16.bits // 8)),
}


def _validate_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    transpose_b: bool,
) -> tuple[int, int, int, int]:
    """Validate metadata and return (batch, M, N, K).

    A has shape [m, batch, k]; bias has shape [batch, 1, n].
    B has shape [batch, k, n], or [batch, n, k] if transpose_b is True.
    """
    if not isinstance(transpose_b, bool):
        raise TypeError("The 'transpose_b' argument must be a Python bool.")
    for name, value in (("a", a), ("b", b), ("bias", bias)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"The '{name}' argument must be a torch.Tensor.")

    if a.dtype != b.dtype or a.dtype != bias.dtype:
        raise TypeError(
            "The 'a', 'b' and 'bias' tensors must have the same dtype; "
            f"got {a.dtype}, {b.dtype} and {bias.dtype}."
        )
    if a.dtype not in _TORCH_DTYPE_TO_DSL:
        raise TypeError(
            f"Only torch.float16 and torch.bfloat16 are supported; got {a.dtype}."
        )

    if a.device != b.device or a.device != bias.device:
        raise ValueError(
            "The 'a', 'b' and 'bias' tensors must be on the same device; "
            f"got {a.device}, {b.device} and {bias.device}."
        )
    if a.device.type != "npu":
        raise ValueError(f"Inputs must be NPU tensors; got device {a.device}.")

    for name, value in (("a", a), ("b", b), ("bias", bias)):
        if value.dim() != _INPUT_DIM:
            raise ValueError(
                f"The '{name}' tensor must have rank {_INPUT_DIM}; got rank {value.dim()}."
            )
    for name, value in (("a", a), ("b", b), ("bias", bias)):
        if not value.is_contiguous():
            raise ValueError(
                f"The '{name}' tensor must use contiguous ND storage; got stride={value.stride()}."
            )

    a_m, a_batch, a_k = a.shape
    b_batch = b.size(0)
    b_k, b_n = (b.size(2), b.size(1)) if transpose_b else (b.size(1), b.size(2))
    if b_batch != a_batch:
        raise ValueError(
            f"The batch dimension of 'b' must match 'a' ({a_batch}); got {b_batch}."
        )
    if b_k != a_k:
        raise ValueError(f"K mismatch: 'a' has {a_k}, 'b' has {b_k}.")

    if bias.size(1) != 1:
        raise ValueError(
            f"The second dimension of 'bias' must be 1; got {bias.size(1)}."
        )
    if bias.size(0) != a_batch:
        raise ValueError(
            f"The batch dimension of 'bias' must match 'a' ({a_batch}); got {bias.size(0)}."
        )
    if bias.size(2) != b_n:
        raise ValueError(
            f"The N dimension of 'bias' must match 'b' ({b_n}); got {bias.size(2)}."
        )

    return a_batch, a_m, b_n, a_k


@dataclasses.dataclass
class _TransposeBmmBiasReluContext:
    """Launch context for A=[m, batch, k] and bias=[batch, 1, n].

    B is [batch, k, n] when transpose_b=False, otherwise [batch, n, k].
    """

    a: torch.Tensor
    b: torch.Tensor
    bias: torch.Tensor
    transpose_b: bool
    batch: int
    m: int
    n: int
    k: int


def _degenerate_result(ctx: _TransposeBmmBiasReluContext) -> torch.Tensor | None:
    """Return [m, batch, n] for zero-sized dimensions, or None for normal inputs."""
    if ctx.m == 0 or ctx.n == 0 or ctx.batch == 0:
        return torch.zeros(
            ctx.m,
            ctx.batch,
            ctx.n,
            dtype=ctx.a.dtype,
            device=ctx.a.device,
        )
    if ctx.k == 0:
        bias = ctx.bias.transpose(0, 1)
        out = bias.expand(ctx.m, ctx.batch, ctx.n)
        return torch.relu(out).contiguous()
    return None


def _reshape_output(
    c: torch.Tensor,
    c_m: int,
    c_n: int,
    ctx: _TransposeBmmBiasReluContext,
) -> torch.Tensor:
    """Remove output padding while retaining the [M, batch, N] view."""
    if c_m != ctx.m:
        c = c.narrow(0, 0, ctx.m)
    if c_n != ctx.n:
        c = c.narrow(2, 0, ctx.n)
    return c.reshape(ctx.m, ctx.batch, ctx.n)


def _run_kernel(ctx: _TransposeBmmBiasReluContext) -> torch.Tensor:
    """Bind A through a zero-copy [batch, M, K] view with explicit strides.

    Physical storage stays [M, batch, K]; no transpose kernel is launched.
    C uses a packed [M_pad, batch*N_pad] binding for direct strided writes.
    """
    c_m = _ceil_align(ctx.m, TransposeBmmBiasReluTiling.BASE_M)
    c_n = _ceil_align(ctx.n, TransposeBmmBiasReluTiling.BASE_N)
    c = torch.empty(c_m, ctx.batch, c_n, dtype=ctx.a.dtype, device=ctx.a.device)
    program = _get_dynamic_program(ctx.a.dtype, ctx.transpose_b)
    program(
        ctx.a.permute(1, 0, 2), ctx.b,
        c.view(c_m, ctx.batch * c_n), ctx.bias.view(ctx.batch, ctx.n),
    )
    return _reshape_output(c, c_m, c_n, ctx)


class TransposeBmmBiasReluTiling:
    """Fixed physical tile sizes; logical dimensions are runtime values."""

    # M alignment for the Cube compute paths.
    BASIC_BLOCK_16 = 16
    BASE_M = 128
    BASE_N = 256
    BASE_K = 64
    STEP_K = 4
    # cube_fill_l1_zero counts 32-byte blocks.
    L1_ZERO_BLOCK_BYTES = 32
    N_TILES_PER_GROUP = 2
    L0C_BUFFER_DEPTH = 1
    # Fixed capacities support double buffering for all runtime shapes.
    DB_SIZE = 2
    AIC_NUM = get_platform_info().cube_core_num

    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        transpose_b: bool = False,
    ) -> None:
        """Set tile sizes and check on-chip memory capacity."""
        self.dtype, self.dtype_size = _TORCH_DTYPE_TO_DSL[dtype]
        self.transpose_b = transpose_b
        self.base_m = self.BASE_M
        self.base_n = self.BASE_N
        self.base_k = self.BASE_K
        self.k_l1 = self.BASE_K * self.STEP_K
        self.l1_buffer_num = self.DB_SIZE
        self.l0c_db = self.DB_SIZE
        l1_bytes = ((self.base_m + self.base_n) * self.k_l1 + self.base_n) * self.dtype_size * self.DB_SIZE
        if l1_bytes > get_mem_size('l1'):
            raise ValueError('Fixed tile exceeds the L1 memory capacity')
        if self.base_m * self.base_k * self.dtype_size * self.DB_SIZE > get_mem_size('l0a'):
            raise ValueError('Fixed tile exceeds the L0A memory capacity')
        if self.base_n * self.base_k * self.dtype_size * self.DB_SIZE > get_mem_size('l0b'):
            raise ValueError('Fixed tile exceeds the L0B memory capacity')
        l0c_bytes = (
            self.base_m * self.base_n * (dtypes.float32.bits // 8)
            * self.N_TILES_PER_GROUP * self.L0C_BUFFER_DEPTH
        )
        if l0c_bytes > get_mem_size('l0c'):
            raise ValueError('Fixed tile exceeds the L0C memory capacity')


class TransposeBmmBiasReluKernel:
    """Runtime M/batch/K/N scheduling with fixed L1 and L0 capacities."""

    def __init__(self, tiling: TransposeBmmBiasReluTiling) -> None:
        """Store the tiling, copy options, and buffer placeholders."""
        self.t = tiling
        self.opt_a = copy_options(l2_cache_ctl=_L2_CACHE_POLICY)
        self.opt_b = copy_options(l2_cache_ctl=_L2_CACHE_POLICY)
        self.opt_bias = copy_options(l2_cache_ctl=_L2_CACHE_POLICY)
        self.opt_c = copy_options(l2_cache_ctl=_L2_CACHE_POLICY)
        self.l1_a = self.l1_b = None
        self.l0a = self.l0b = self.l0c_first = None
        self.l0c_second = None
        self.l1_bias = self.bias_ring = None

    @jit
    def _create_buffers(self) -> None:
        """Allocate the L1, L0, and bias buffers for this kernel."""
        t = self.t
        b_shape = (t.base_n, t.k_l1) if t.transpose_b else (t.k_l1, t.base_n)
        self.l1_a = RingBuffer(
            MemLoc.L1, t.dtype, make_layout((t.base_m, t.k_l1)),
            depth=t.l1_buffer_num, data_format="nz",
        )
        self.l1_b = RingBuffer(
            MemLoc.L1, t.dtype, make_layout(b_shape),
            depth=t.l1_buffer_num, data_format="nz",
        )
        self.l0a = RingBuffer(
            MemLoc.L0A, t.dtype, make_layout((t.base_m, t.base_k)), depth=t.DB_SIZE,
        )
        self.l0b = RingBuffer(
            MemLoc.L0B, t.dtype, make_layout((t.base_n, t.base_k)), depth=t.DB_SIZE,
        )
        self.l0c_first = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((t.base_m, t.base_n)),
            depth=t.L0C_BUFFER_DEPTH,
        )
        self.l0c_second = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((t.base_m, t.base_n)),
            depth=t.L0C_BUFFER_DEPTH,
        )
        self.l1_bias = RingBuffer(
            MemLoc.L1, t.dtype, make_layout((1, t.base_n)),
            depth=t.DB_SIZE, data_format="nd",
        )
        self.bias_ring = RingBuffer(
            MemLoc.BIAS, dtypes.float32, make_layout((t.base_n,)),
            depth=t.DB_SIZE, data_format="nd",
        )

    @kernel
    def transpose_bmm_bias_relu_kernel(self, gm_a: Tensor, gm_b: Tensor, gm_c: Tensor, gm_bias: Tensor) -> None:
        """Distribute groups of adjacent N tiles across Cube cores."""
        self._create_buffers()
        batch, m, k = gm_a.shape
        n = gm_bias.shape[1]
        m_tiles = ceil_div(m, self.t.base_m)
        n_tiles = ceil_div(n, self.t.base_n)
        n_groups = ceil_div(n_tiles, self.t.N_TILES_PER_GROUP)
        total_groups = batch * m_tiles * n_groups
        for group_idx in range(get_block_idx(), total_groups, get_block_num()):
            b_idx = group_idx // (m_tiles * n_groups)
            m_idx = (group_idx // n_groups) % m_tiles
            n_idx = (group_idx % n_groups) * self.t.N_TILES_PER_GROUP
            tile_idx = (b_idx * m_tiles + m_idx) * n_tiles + n_idx
            self._compute_output_tile(gm_a, gm_b, gm_c, gm_bias, tile_idx)

    @jit
    def _compute_output_tile(self, gm_a, gm_b, gm_c, gm_bias, tile_idx) -> None:
        """Decode the tile coordinates and select a static M extent."""
        t = self.t
        batch, m, k = gm_a.shape
        n = gm_bias.shape[1]
        m_tiles = ceil_div(m, t.base_m)
        n_tiles = ceil_div(n, t.base_n)
        mn_tiles = m_tiles * n_tiles
        b_idx = tile_idx // mn_tiles
        mn_idx = tile_idx % mn_tiles
        m_idx = mn_idx // n_tiles
        n_idx = mn_idx % n_tiles
        # Expand the static M extents into one binary. The runtime bucket
        # selects the smallest extent that covers the current tile.
        remaining_m = m - m_idx * t.base_m
        block_m = t.BASIC_BLOCK_16
        m_bucket = (remaining_m - 1) // block_m
        last_bucket = t.base_m // block_m - 1
        if m_bucket > last_bucket:
            m_bucket = last_bucket
        for bucket in range_constexpr(t.base_m // block_m):
            if m_bucket == bucket:
                self._compute_sized_tile(
                    gm_a, gm_b, gm_c, gm_bias,
                    (m_idx, n_idx, b_idx, n_tiles, (bucket + 1) * block_m),
                )

    @jit
    def _compute_sized_tile(self, gm_a, gm_b, gm_c, gm_bias, position) -> None:
        """Dispatch to the paired N tile path or the final single N tile."""
        m_idx, n_idx, b_idx, n_tiles, m_eff = position
        if n_idx + 1 < n_tiles:
            self._compute_pair(gm_a, gm_b, gm_c, gm_bias, position)
        else:
            self._compute_single_tile(gm_a, gm_b, gm_c, gm_bias, position)

    @jit
    def _load_pair_bias(self, gm_bias, b_idx, n_idx) -> Tensor:
        """Load one bias tile through L1 and return its BIAS buffer view."""
        t = self.t
        bias_l1 = self.l1_bias.next_write()
        if (n_idx + 1) * t.base_n > gm_bias.shape[1]:
            cube_fill_l1_zero(
                bias_l1,
                blk_num=t.base_n * t.dtype_size // t.L1_ZERO_BLOCK_BYTES,
            )
            cube_sync_all()
        mem_copy(
            bias_l1,
            tile_slice(gm_bias, (1, t.base_n), (b_idx, n_idx)),
            options=self.opt_bias,
        )
        mem_copy(
            self.bias_ring.next_write(),
            self.l1_bias.next_read().reshape(t.base_n),
        )
        return self.bias_ring.next_read()

    @jit
    def _compute_pair(self, gm_a, gm_b, gm_c, gm_bias, position) -> None:
        """Accumulate two adjacent N tiles and write both through ReLU."""
        m_idx, n_idx, b_idx, n_tiles, m_eff = position
        t = self.t
        a2d = gm_a[b_idx, None, None]
        b2d = gm_b[b_idx, None, None]
        bias_first = self._load_pair_bias(gm_bias, b_idx, n_idx)
        bias_second = self._load_pair_bias(gm_bias, b_idx, n_idx + 1)
        c_first = self.l0c_first.next_write()
        c_second = self.l0c_second.next_write()
        accumulators = (c_first, c_second, bias_first, bias_second)
        self._reduce_pair_window(
            a2d, b2d, accumulators, (m_idx, n_idx, 0, True, m_eff),
        )
        for window_idx in range(1, ceil_div(a2d.shape[-1], t.k_l1)):
            self._reduce_pair_window(
                a2d, b2d, accumulators,
                (m_idx, n_idx, window_idx, False, m_eff),
            )
        first_tile = tile_slice(
            gm_c, (t.base_m, t.base_n), (m_idx, b_idx * n_tiles + n_idx),
        )
        second_tile = tile_slice(
            gm_c, (t.base_m, t.base_n), (m_idx, b_idx * n_tiles + n_idx + 1),
        )
        mem_copy(
            tile_slice(first_tile, (m_eff, t.base_n), (0, 0)),
            from_tensor(self.l0c_first.next_read(), layout=make_layout((m_eff, t.base_n))).relu(),
            options=self.opt_c,
        )
        mem_copy(
            tile_slice(second_tile, (m_eff, t.base_n), (0, 0)),
            from_tensor(self.l0c_second.next_read(), layout=make_layout((m_eff, t.base_n))).relu(),
            options=self.opt_c,
        )

    @jit
    def _reduce_pair_window(self, a2d, b2d, accumulators, position) -> None:
        """Load one A window and reuse it for both N tiles."""
        m_idx, n_idx, window_idx, first_window, m_eff = position
        t = self.t
        a_write = self.l1_a.next_write()
        if (window_idx + 1) * t.k_l1 > a2d.shape[-1]:
            cube_fill_l1_zero(
                a_write,
                blk_num=t.base_m * t.k_l1 * t.dtype_size // t.L1_ZERO_BLOCK_BYTES,
            )
            cube_sync_all()
        mem_copy(
            a_write,
            tile_slice(a2d, (t.base_m, t.k_l1), (m_idx, window_idx)),
            options=self.opt_a,
        )
        a_read = self.l1_a.next_read()
        c_first, c_second, bias_first, bias_second = accumulators
        self._reduce_shared_a(
            a_read, b2d, c_first, bias_first,
            (n_idx, window_idx, first_window, m_eff),
        )
        self._reduce_shared_a(
            a_read, b2d, c_second, bias_second,
            (n_idx + 1, window_idx, first_window, m_eff),
        )

    @jit
    def _reduce_shared_a(self, a_read, b2d, c_view, bias, position) -> None:
        """Accumulate one B window using the A window already in L1."""
        n_idx, window_idx, first_window, m_eff = position
        t = self.t
        c_view = from_tensor(c_view, layout=make_layout((m_eff, t.base_n)))
        b_write = self.l1_b.next_write()
        if const_expr(t.transpose_b):
            n, k = b2d.shape[-2], b2d.shape[-1]
        else:
            k, n = b2d.shape[-2], b2d.shape[-1]
        if ((window_idx + 1) * t.k_l1 > k) or ((n_idx + 1) * t.base_n > n):
            cube_fill_l1_zero(
                b_write,
                blk_num=t.base_n * t.k_l1 * t.dtype_size // t.L1_ZERO_BLOCK_BYTES,
            )
            cube_sync_all()
        if const_expr(t.transpose_b):
            b_source = tile_slice(b2d, (t.base_n, t.k_l1), (n_idx, window_idx))
        else:
            b_source = tile_slice(b2d, (t.k_l1, t.base_n), (window_idx, n_idx))
        mem_copy(b_write, b_source, options=self.opt_b)
        b_read = self.l1_b.next_read()
        for k_block in range_constexpr(t.STEP_K):
            a_l0 = from_tensor(self.l0a.next_write(), layout=make_layout((m_eff, t.base_k)))
            b_l0 = self.l0b.next_write()
            mem_copy(
                a_l0,
                tile_slice(a_read, (m_eff, t.base_k), (0, k_block)),
            )
            if const_expr(t.transpose_b):
                b_source_l0 = tile_slice(b_read, (t.base_n, t.base_k), (0, k_block))
            else:
                b_source_l0 = tile_slice(b_read, (t.base_k, t.base_n), (k_block, 0)).permute(1, 0)
            mem_copy(b_l0, b_source_l0)
            if const_expr(first_window and k_block == 0):
                dsl_matmul(c_view, a_l0, b_l0, init=True, bias=bias)
            else:
                dsl_matmul(c_view, a_l0, b_l0, init=False)

    @jit
    def _compute_single_tile(self, gm_a, gm_b, gm_c, gm_bias, position) -> None:
        """Accumulate a single N tile and write it through ReLU."""
        m_idx, n_idx, b_idx, n_tiles, m_eff = position
        t = self.t
        batch, m, k = gm_a.shape
        n = gm_bias.shape[1]
        a2d = gm_a[b_idx, None, None]
        b2d = gm_b[b_idx, None, None]
        c2d = gm_c
        bias_l1 = self.l1_bias.next_write()
        if (n_idx + 1) * t.base_n > n:
            cube_fill_l1_zero(
                bias_l1,
                blk_num=t.base_n * t.dtype_size // t.L1_ZERO_BLOCK_BYTES,
            )
            cube_sync_all()
        mem_copy(bias_l1, tile_slice(gm_bias, (1, t.base_n), (b_idx, n_idx)), options=self.opt_bias)
        mem_copy(self.bias_ring.next_write(), self.l1_bias.next_read().reshape(t.base_n))
        bias = self.bias_ring.next_read()
        c_view = self.l0c_first.next_write()
        # Separate the first window so MMAD init/bias stay compile-time choices.
        self._reduce_window(a2d, b2d, c_view, bias, (m_idx, n_idx, 0, True, b_idx, m, batch, m_eff))
        for window_idx in range(1, ceil_div(k, t.k_l1)):
            self._reduce_window(a2d, b2d, c_view, bias, (m_idx, n_idx, window_idx, False, b_idx, m, batch, m_eff))
        c_tile = tile_slice(c2d, (t.base_m, t.base_n), (m_idx, b_idx * n_tiles + n_idx))
        mem_copy(
            tile_slice(c_tile, (m_eff, t.base_n), (0, 0)),
            from_tensor(self.l0c_first.next_read(), layout=make_layout((m_eff, t.base_n))).relu(),
            options=self.opt_c,
        )

    @jit
    def _reduce_window(self, a2d, b2d, c_view, bias, position) -> None:
        """Load A and B for one K window and accumulate their K blocks."""
        t = self.t
        m_idx, n_idx, window_idx, first_window, b_idx, m, batch, m_eff = position
        # Reinterpret C with the compact fractal strides required by this M extent.
        c_view = from_tensor(c_view, layout=make_layout((m_eff, t.base_n)))
        a_write = self.l1_a.next_write()
        b_write = self.l1_b.next_write()
        # Full P0 windows overwrite the consumed domain and need no clearing.
        # A partial K window consumes fixed-size K blocks, so clear its tail.
        # B also needs clearing when the N tile is partial.
        if (window_idx + 1) * t.k_l1 > a2d.shape[-1]:
            cube_fill_l1_zero(
                a_write,
                blk_num=t.base_m * t.k_l1 * t.dtype_size // t.L1_ZERO_BLOCK_BYTES,
            )
        if const_expr(t.transpose_b):
            n = b2d.shape[-2]
        else:
            n = b2d.shape[-1]
        if ((window_idx + 1) * t.k_l1 > a2d.shape[-1]) or ((n_idx + 1) * t.base_n > n):
            cube_fill_l1_zero(
                b_write,
                blk_num=t.base_n * t.k_l1 * t.dtype_size // t.L1_ZERO_BLOCK_BYTES,
            )
        if ((window_idx + 1) * t.k_l1 > a2d.shape[-1]) or ((n_idx + 1) * t.base_n > n):
            cube_sync_all()
        mem_copy(a_write, tile_slice(a2d, (t.base_m, t.k_l1), (m_idx, window_idx)), options=self.opt_a)
        if const_expr(t.transpose_b):
            b_source = tile_slice(b2d, (t.base_n, t.k_l1), (n_idx, window_idx))
        else:
            b_source = tile_slice(b2d, (t.k_l1, t.base_n), (window_idx, n_idx))
        mem_copy(b_write, b_source, options=self.opt_b)
        a_read = self.l1_a.next_read()
        b_read = self.l1_b.next_read()
        for k_block in range_constexpr(t.STEP_K):
            a_l0 = from_tensor(self.l0a.next_write(), layout=make_layout((m_eff, t.base_k)))
            b_l0 = self.l0b.next_write()
            mem_copy(a_l0, tile_slice(a_read, (m_eff, t.base_k), (0, k_block)))
            if const_expr(t.transpose_b):
                b_source_l0 = tile_slice(b_read, (t.base_n, t.base_k), (0, k_block))
            else:
                b_source_l0 = tile_slice(b_read, (t.base_k, t.base_n), (k_block, 0)).permute(1, 0)
            mem_copy(b_l0, b_source_l0)
            if const_expr(first_window and k_block == 0):
                dsl_matmul(c_view, a_l0, b_l0, init=True, bias=bias)
            else:
                dsl_matmul(c_view, a_l0, b_l0, init=False)

    @host
    def run(self, gm_a: Tensor, gm_b: Tensor, gm_c: Tensor, gm_bias: Tensor) -> None:
        """Launch the device kernel across the available Cube cores."""
        self.transpose_bmm_bias_relu_kernel[self.t.AIC_NUM](gm_a, gm_b, gm_c, gm_bias)


@lru_cache(maxsize=_PROGRAM_CACHE_SIZE)
def _get_dynamic_program(dtype: torch.dtype, transpose_b: bool) -> "ProviderCallable":
    """Compile one shape-dynamic program per dtype and B layout."""
    t = TransposeBmmBiasReluTiling(dtype, transpose_b)
    # A is a strided metadata view of the original contiguous [M, batch, K].
    m = Dim('M', min=1)
    batch = Dim('BATCH', min=1)
    k = Dim('K', min=1)
    n = Dim('N', min=1)
    a_spec = TensorSpec((batch, m, k), t.dtype, stride=(k, batch * k, 1))
    b_spec = TensorSpec((batch, n, k) if transpose_b else (batch, k, n), t.dtype)
    c_spec = TensorSpec(
        (
            _ceil_align(m, t.base_m),
            batch * _ceil_align(n, t.base_n),
        ),
        t.dtype,
    )
    bias_spec = TensorSpec((batch, n), t.dtype)
    return compile_program(TransposeBmmBiasReluKernel(t).run, a_spec, b_spec, c_spec, bias_spec)


def transpose_bmm_bias_relu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    transpose_b: bool = False,
) -> torch.Tensor:
    """Fused [M,batch,K] x B + bias + ReLU with runtime-dynamic dimensions.

    B uses [batch,K,N], or [batch,N,K] when transpose_b=True.
    Inputs are contiguous NPU FP16/BF16 tensors; output is [M,batch,N].
    """
    batch, m, n, k = _validate_inputs(a, b, bias, transpose_b)
    ctx = _TransposeBmmBiasReluContext(a, b, bias, transpose_b, batch, m, n, k)
    out = _degenerate_result(ctx)
    if out is not None:
        return out
    return _run_kernel(ctx)
