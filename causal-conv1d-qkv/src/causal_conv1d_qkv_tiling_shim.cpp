// Project-owned build glue (not operator code).
//
// CausalConv1dQkv reuses the shared causal_conv1d host tiling implementation.
// Compiling the vendored causal_conv1d_tiling.cpp as-is would also emit
// IMPL_OP_OPTILING(CausalConv1d), registering a second CausalConv1d tiling in
// this vendor package. The host already ships a different-lineage CausalConv1d,
// so this package must register only CausalConv1dQkv. Neutralize just that
// registration macro while compiling the unmodified upstream TU.

#include "register/op_impl_registry.h"

namespace {

// Stand-in for gert::OpImplRegisterV2 that accepts the chained
// .Tiling(...)/.TilingParse<T>(...) calls and registers nothing.
struct NoopOptilingRegistrar {
    template <typename Func>
    NoopOptilingRegistrar &Tiling(Func)
    {
        return *this;
    }

    template <typename CompileInfo, typename Func>
    NoopOptilingRegistrar &TilingParse(Func)
    {
        return *this;
    }
};

} // namespace

#undef IMPL_OP_OPTILING
#define IMPL_OP_OPTILING(op_type) \
    [[maybe_unused]] static NoopOptilingRegistrar op_impl_register_optiling_##op_type = NoopOptilingRegistrar()

#include "causal_conv1d/op_host/causal_conv1d_tiling.cpp"
