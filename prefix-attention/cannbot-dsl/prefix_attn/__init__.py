"""prefix_attn — prefix-in-Q attention (CANNBotDSL implementation).

The kernel implementation layer: ``prefix_attn.prefix_attention`` runs the
DSL host path.  cannbotdsl is imported lazily on first attribute access, so
importing the package works without it; only eager direct calls require it.

Torch integration (torch.ops registration + torchair converter) lives in the
separate ``prefix_attn_torch`` package (ge/prefix_attn_torch), which runs the GE
deliverable without cannbotdsl.
"""

__all__ = ["prefix_attention"]


def __getattr__(name):
    if name == "prefix_attention":
        try:
            from .attention import prefix_attention
        except ImportError as exc:  # cannbotdsl missing at eager call time
            raise ImportError(
                "prefix_attn.prefix_attention (the eager DSL host path) needs "
                "cannbotdsl. Install it, or use the prefix_attn_torch torch "
                "binding (ge/prefix_attn_torch), which runs the GE deliverable "
                "without cannbotdsl."
            ) from exc
        return prefix_attention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
