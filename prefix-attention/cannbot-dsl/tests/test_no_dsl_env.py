#!/usr/bin/env python3
"""Verify the torch integration layer works WITHOUT cannbotdsl installed.

Blocks the cannbotdsl import via a meta_path finder, then checks:
  1. import prefix_attn.torch_ops succeeds (schema/Meta/converter/OPP ready)
  2. torch.ops.ascendc_ops.prefix_attention is registered with a Meta impl
  3. eager direct call raises a clear ImportError mentioning cannbotdsl

Run: python tests/test_no_dsl_env.py   (expects PASS lines; exit 0)
"""
import sys


class _BlockCannbotdsl:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name == "cannbotdsl" or name.startswith("cannbotdsl."):
            raise ImportError(f"No module named '{name}' (blocked by test)")
        return None


sys.meta_path.insert(0, _BlockCannbotdsl())
for mod in list(sys.modules):
    if mod == "cannbotdsl" or mod.startswith("cannbotdsl."):
        del sys.modules[mod]

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

# 1. integration layer imports without cannbotdsl
import prefix_attn.torch_ops as tops  # noqa: E402
print("IMPORT_OK opp_mounted=%s converter=%s"
      % (bool(tops.OPP_ROOT), tops.TORCHAIR_CONVERTER_REGISTERED))

# 2. op registered + Meta impl present (shape inference works w/o DSL)
assert hasattr(torch.ops.ascendc_ops, "prefix_attention"), "op not registered"
q = torch.empty(1320, 14, 64, dtype=torch.float16, device="meta")
k = torch.empty(1320, 2, 64, dtype=torch.float16, device="meta")
v = torch.empty(1320, 2, 64, dtype=torch.float16, device="meta")
b = torch.empty(11, dtype=torch.int32, device="meta")
meta_out = torch.ops.ascendc_ops.prefix_attention(q, k, v, b)
print("META_OK out_shape=%s dtype=%s" % (tuple(meta_out.shape), meta_out.dtype))
assert tuple(meta_out.shape) == (1320, 14, 64)

# 3. eager (PrivateUse1) call must fail with a clear cannbotdsl message
try:
    import prefix_attn
    prefix_attn.prefix_attention  # attribute access triggers the lazy import
    print("EAGER_UNEXPECTED_OK (cannbotdsl was importable?)")
    sys.exit(1)
except ImportError as exc:
    msg = str(exc)
    assert "cannbotdsl" in msg, msg
    print("EAGER_CLEAR_ERROR_OK")

print("NO_DSL_ENV_PASS")
