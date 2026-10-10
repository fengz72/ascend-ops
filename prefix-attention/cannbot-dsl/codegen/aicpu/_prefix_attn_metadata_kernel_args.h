#pragma once
#include <cstdint>
struct _PrefixAttnMetadataArgs {
    const int32_t* bounds;
    uint32_t* metadata;
    uint32_t bounds_len;
    uint32_t num_heads_q;
    uint32_t num_heads_kv;
    uint32_t aic_core_num;
    uint32_t macro_cost;
};
