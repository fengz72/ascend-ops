#include <cstdio>
#include <cstdarg>
namespace AscendC {
void printf(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
}
}
#include "_prefix_attn_metadata_kernel_args.h"
#include "_prefix_attn_metadata_kernel_body.inc"
extern "C" int64_t _prefix_attn_metadata_kernel_run_x86(uint64_t a_bounds, uint64_t a_metadata, uint32_t a_bounds_len, uint32_t a_num_heads_q, uint32_t a_num_heads_kv, uint32_t a_aic_core_num, uint32_t a_dn_cost)
{
    _PrefixAttnMetadataArgs s;
    s.bounds = (const int32_t*)a_bounds;
    s.metadata = (uint32_t*)a_metadata;
    s.bounds_len = a_bounds_len;
    s.num_heads_q = a_num_heads_q;
    s.num_heads_kv = a_num_heads_kv;
    s.aic_core_num = a_aic_core_num;
    s.dn_cost = a_dn_cost;
    return _prefix_attn_metadata_kernel_body(&s);
}
