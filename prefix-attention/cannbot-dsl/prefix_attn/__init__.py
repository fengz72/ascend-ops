"""prefix_attn — prefix-in-Q attention (CANNBotDSL implementation).

The package separates two layers by import time:

  - torch integration layer (``prefix_attn.torch_ops``): schema/dispatch
    registration + torchair converter + OPP mount.  Depends only on torch, so
    ``import prefix_attn.torch_ops`` works in a production environment without
    cannbotdsl (graph lowering runs the GE C++ deliverable, no DSL needed).
  - kernel implementation layer (``prefix_attn.prefix_attention``): the DSL
    host path.  cannbotdsl is imported lazily on first attribute access, so
    only eager direct calls require it.

``prefix_attention`` is exposed via PEP 562 module ``__getattr__`` to keep the
top-level import free of cannbotdsl.
"""

__all__ = ["prefix_attention"]


def __getattr__(name):
    if name == "prefix_attention":
        try:
            from .attention import prefix_attention
        except ImportError as exc:  # cannbotdsl missing at eager call time
            raise ImportError(
                "prefix_attn.prefix_attention (the eager DSL host path) needs "
                "cannbotdsl. Install it, or use graph lowering (torchair) which "
                "runs the GE C++ deliverable without cannbotdsl."
            ) from exc
        return prefix_attention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
