"""Host-side input validation for the prefix-in-Q attention contract.

Extracted verbatim from the prefix_attn reference implementation so the
kernel package stays self-contained.
"""

import torch

HEAD_DIM = 64


class DesignError(ValueError):
    """A planner input cannot satisfy a frozen design-time invariant."""

    code = "DESIGN_ERROR"


def _bounds(actual_seq_lengths: torch.Tensor, total: int) -> list[int]:
    if not isinstance(actual_seq_lengths, torch.Tensor):
        raise TypeError("actual_seq_lengths must be a torch.Tensor")
    if actual_seq_lengths.ndim != 1 or actual_seq_lengths.numel() < 2:
        raise ValueError("actual_seq_lengths must be a 1-D [B+1] tensor")
    if actual_seq_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("actual_seq_lengths must use int32 or int64")
    b = [int(x) for x in actual_seq_lengths.detach().cpu().tolist()]
    if b[0] <= 0 or b[-1] != total or any(x >= y for x, y in zip(b, b[1:])):
        raise ValueError("sequence boundaries must be strictly increasing and end at T")
    return b


def _validate(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              actual_seq_lengths: torch.Tensor) -> list[int]:
    if not all(isinstance(x, torch.Tensor) for x in (query, key, value)):
        raise TypeError("query, key and value must be torch.Tensor objects")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query/key/value must be rank-3 TND tensors")
    t, n1, d = query.shape
    if key.shape != value.shape or key.shape[0] != t or key.shape[2] != d:
        raise ValueError("Q/K/V must have matching T and D and equal K/V shapes")
    n2 = key.shape[1]
    if d != HEAD_DIM or n2 <= 0 or n1 < n2 or n1 % n2:
        raise ValueError("expected D=64 and N1 >= N2 with N1 divisible by N2")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("only float16 and bfloat16 inputs are supported")
    if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
        raise ValueError("Q/K/V must be contiguous")
    return _bounds(actual_seq_lengths, t)
