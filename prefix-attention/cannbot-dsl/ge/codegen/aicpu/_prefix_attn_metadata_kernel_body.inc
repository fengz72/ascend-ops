template <typename T> struct AArr { T* data; int64_t len; int64_t cap; };
static thread_local uint8_t g_arena[1 << 22];
static thread_local int64_t g_arena_off = 0;
static void* arena_alloc(int64_t bytes) {
    int64_t base = (g_arena_off + 15) & ~int64_t(15);
    g_arena_off = base + bytes;
    return g_arena + base;
}
struct Ctx {
    AArr<int64_t> costs;
    AArr<int64_t> pref;
    AArr<uint32_t> bound;
    AArr<int64_t> blen;
    AArr<int64_t> bfirst;
    AArr<int64_t> bn2s;
    AArr<int64_t> bn2e;
    AArr<int64_t> ms;
    AArr<int64_t> me;
};

int64_t _prefix_attn_metadata_kernel_body(const _PrefixAttnMetadataArgs* a)
{
int64_t n1 = 0;
int64_t B = 0;
int64_t p = 0;
int64_t ptiles = 0;
int64_t bl_sum = 0;
int64_t b = 0;
int64_t total = 0;
int64_t wi = 0;
int64_t _qh = 0;
int64_t qt = 0;
int64_t rows = 0;
int64_t start = 0;
int64_t length = 0;
int64_t bl = 0;
int64_t visible = 0;
int64_t i = 0;
int64_t owner_limit = 0;
int64_t nowners = 0;
int64_t lo = 0;
int64_t hi = 0;
int64_t mid = 0;
int64_t count = 0;
int64_t acc = 0;
int64_t limit = 0;
int64_t k = 0;
int64_t best_ci = 0;
int64_t best_cost = 0;
int64_t ci = 0;
int64_t a0 = 0;
int64_t b0 = 0;
int64_t cc = 0;
int64_t best_m = 0;
int64_t v1 = 0;
int64_t v2 = 0;
int64_t best_v = 0;
int64_t m = 0;
int64_t v = 0;
int64_t tmax = 0;
int64_t t = 0;
int64_t units = 0;
int64_t u = 0;
int64_t r = 0;
int64_t w0 = 0;
int64_t w1 = 0;
int64_t u0 = 0;
int64_t u1 = 0;
int64_t base = 0;
n1 = ((int64_t)a->num_heads_q);
B = (((int64_t)a->bounds_len) - 1);
p = ((int64_t)a->bounds[(int64_t)0]);
ptiles = (((p) / 128 + (((p) % 128) != 0)));
bl_sum = 0;
for (int64_t b = 0; b < (B); b += 1) {
    bl_sum = ((bl_sum) + (((((int64_t)a->bounds[(int64_t)((b) + 1)]) - ((int64_t)a->bounds[(int64_t)(b)])) / 128 + (((((int64_t)a->bounds[(int64_t)((b) + 1)]) - ((int64_t)a->bounds[(int64_t)(b)])) % 128) != 0))));
}
total = (((n1) * (ptiles)) + ((n1) * (bl_sum)));
Ctx ctx = {};
AArr<int64_t> _av1;
_av1.cap = (int64_t)((total) + 1);
_av1.len = (int64_t)((total) + 1);
_av1.data = (int64_t*)arena_alloc((int64_t)((total) + 1) * sizeof(int64_t));
for (int64_t _z2 = 0; _z2 < (int64_t)((total) + 1); _z2 += 1) { _av1.data[_z2] = 0; }
ctx.costs = _av1;
wi = 0;
for (int64_t _qh = 0; _qh < (n1); _qh += 1) {
    for (int64_t qt = 0; qt < (ptiles); qt += 1) {
        rows = ((128 < ((p) - ((qt) * 128)) ? 128 : ((p) - ((qt) * 128))));
        ctx.costs.data[(int64_t)(wi)] = (int64_t)(((qt) + 1) * ((rows) + 32));
        if ((((int64_t)a->macro_cost) != 0)) {
            ctx.costs.data[(int64_t)(wi)] = (int64_t)(((qt) + 1) * (128 + 32));
        }
        wi = ((wi) + 1);
    }
}
for (int64_t b = 0; b < (B); b += 1) {
    start = ((int64_t)a->bounds[(int64_t)(b)]);
    length = (((int64_t)a->bounds[(int64_t)((b) + 1)]) - (start));
    bl = (((length) / 128 + (((length) % 128) != 0)));
    for (int64_t _qh = 0; _qh < (n1); _qh += 1) {
        for (int64_t qt = 0; qt < (bl); qt += 1) {
            rows = ((128 < ((length) - ((qt) * 128)) ? 128 : ((length) - ((qt) * 128))));
            ctx.costs.data[(int64_t)(wi)] = (int64_t)((((ptiles) + (qt)) + 1) * ((rows) + 32));
            if ((((int64_t)a->macro_cost) != 0)) {
                visible = (((((p) + ((qt) * 128)) + (rows)) / 128 + (((((p) + ((qt) * 128)) + (rows)) % 128) != 0)));
                ctx.costs.data[(int64_t)(wi)] = (int64_t)((visible) * (128 + 32));
            }
            wi = ((wi) + 1);
        }
    }
}
AArr<int64_t> _av3;
_av3.cap = (int64_t)((total) + 1);
_av3.len = (int64_t)((total) + 1);
_av3.data = (int64_t*)arena_alloc((int64_t)((total) + 1) * sizeof(int64_t));
for (int64_t _z4 = 0; _z4 < (int64_t)((total) + 1); _z4 += 1) { _av3.data[_z4] = 0; }
ctx.pref = _av3;
for (int64_t i = 0; i < (total); i += 1) {
    ctx.pref.data[(int64_t)((i) + 1)] = (int64_t)(((int64_t)ctx.pref.data[(int64_t)(i)]) + ((int64_t)ctx.costs.data[(int64_t)(i)]));
}
owner_limit = ((int64_t)a->aic_core_num);
if ((((((int64_t)a->macro_cost) != 0) && (((owner_limit) > 1) != 0)) != 0)) {
    owner_limit = ((owner_limit) - 1);
}
nowners = (((owner_limit) < (total) ? (owner_limit) : (total)));
lo = 0;
for (int64_t i = 0; i < (total); i += 1) {
    if (((((int64_t)ctx.costs.data[(int64_t)(i)]) > (lo)) != 0)) {
        lo = ((int64_t)ctx.costs.data[(int64_t)(i)]);
    }
}
hi = ((int64_t)ctx.pref.data[(int64_t)(total)]);
while ((((lo) < (hi)) != 0)) {
    mid = (((lo) + (hi)) / 2);
    count = 1;
    acc = 0;
    for (int64_t i = 0; i < (total); i += 1) {
        if (((((acc) + ((int64_t)ctx.costs.data[(int64_t)(i)])) > (mid)) != 0)) {
            count = ((count) + 1);
            acc = ((int64_t)ctx.costs.data[(int64_t)(i)]);
        }
        else {
            acc = ((acc) + ((int64_t)ctx.costs.data[(int64_t)(i)]));
        }
    }
    if ((((count) <= (nowners)) != 0)) {
        hi = (mid);
    }
    else {
        lo = ((mid) + 1);
    }
}
limit = (lo);
AArr<uint32_t> _av5;
_av5.cap = (int64_t)((nowners) + 1);
_av5.len = (int64_t)((nowners) + 1);
_av5.data = (uint32_t*)arena_alloc((int64_t)((nowners) + 1) * sizeof(uint32_t));
for (int64_t _z6 = 0; _z6 < (int64_t)((nowners) + 1); _z6 += 1) { _av5.data[_z6] = 0; }
ctx.bound = _av5;
k = 0;
acc = 0;
for (int64_t i = 0; i < (total); i += 1) {
    if (((((acc) + ((int64_t)ctx.costs.data[(int64_t)(i)])) > (limit)) != 0)) {
        k = ((k) + 1);
        ctx.bound.data[(int64_t)(k)] = (uint32_t)(i);
        acc = ((int64_t)ctx.costs.data[(int64_t)(i)]);
    }
    else {
        acc = ((acc) + ((int64_t)ctx.costs.data[(int64_t)(i)]));
    }
}
k = ((k) + 1);
ctx.bound.data[(int64_t)(k)] = (uint32_t)(total);
while ((((k) < (nowners)) != 0)) {
    best_ci = (-1);
    best_cost = (-1);
    for (int64_t ci = 0; ci < (k); ci += 1) {
        a0 = ((int64_t)ctx.bound.data[(int64_t)(ci)]);
        b0 = ((int64_t)ctx.bound.data[(int64_t)((ci) + 1)]);
        if (((((b0) - (a0)) >= 2) != 0)) {
            cc = (((int64_t)ctx.pref.data[(int64_t)(b0)]) - ((int64_t)ctx.pref.data[(int64_t)(a0)]));
            if ((((cc) > (best_cost)) != 0)) {
                best_cost = (cc);
                best_ci = (ci);
            }
        }
    }
    a0 = ((int64_t)ctx.bound.data[(int64_t)(best_ci)]);
    b0 = ((int64_t)ctx.bound.data[(int64_t)((best_ci) + 1)]);
    best_m = ((a0) + 1);
    v1 = (((int64_t)ctx.pref.data[(int64_t)((a0) + 1)]) - ((int64_t)ctx.pref.data[(int64_t)(a0)]));
    v2 = (((int64_t)ctx.pref.data[(int64_t)(b0)]) - ((int64_t)ctx.pref.data[(int64_t)((a0) + 1)]));
    if ((((v2) > (v1)) != 0)) {
        best_v = (v2);
    }
    else {
        best_v = (v1);
    }
    for (int64_t m = ((a0) + 2); m < (b0); m += 1) {
        v1 = (((int64_t)ctx.pref.data[(int64_t)(m)]) - ((int64_t)ctx.pref.data[(int64_t)(a0)]));
        v2 = (((int64_t)ctx.pref.data[(int64_t)(b0)]) - ((int64_t)ctx.pref.data[(int64_t)(m)]));
        if ((((v2) > (v1)) != 0)) {
            v = (v2);
        }
        else {
            v = (v1);
        }
        if ((((v) < (best_v)) != 0)) {
            best_v = (v);
            best_m = (m);
        }
    }
    tmax = (((k) - (best_ci)) - 1);
    for (int64_t t = 0; t < ((tmax) + 1); t += 1) {
        ctx.bound.data[(int64_t)(((k) + 1) - (t))] = (uint32_t)((int64_t)ctx.bound.data[(int64_t)((k) - (t))]);
    }
    ctx.bound.data[(int64_t)((best_ci) + 1)] = (uint32_t)(best_m);
    k = ((k) + 1);
}
units = (1 + (B));
AArr<int64_t> _av7;
_av7.cap = (int64_t)(units);
_av7.len = (int64_t)(units);
_av7.data = (int64_t*)arena_alloc((int64_t)(units) * sizeof(int64_t));
for (int64_t _z8 = 0; _z8 < (int64_t)(units); _z8 += 1) { _av7.data[_z8] = 0; }
ctx.blen = _av7;
ctx.blen.data[(int64_t)0] = (int64_t)(ptiles);
for (int64_t b = 0; b < (B); b += 1) {
    ctx.blen.data[(int64_t)((b) + 1)] = (int64_t)(((((int64_t)a->bounds[(int64_t)((b) + 1)]) - ((int64_t)a->bounds[(int64_t)(b)])) / 128 + (((((int64_t)a->bounds[(int64_t)((b) + 1)]) - ((int64_t)a->bounds[(int64_t)(b)])) % 128) != 0)));
}
AArr<int64_t> _av9;
_av9.cap = (int64_t)((units) + 1);
_av9.len = (int64_t)((units) + 1);
_av9.data = (int64_t*)arena_alloc((int64_t)((units) + 1) * sizeof(int64_t));
for (int64_t _z10 = 0; _z10 < (int64_t)((units) + 1); _z10 += 1) { _av9.data[_z10] = 0; }
ctx.bfirst = _av9;
for (int64_t u = 0; u < (units); u += 1) {
    ctx.bfirst.data[(int64_t)((u) + 1)] = (int64_t)(((int64_t)ctx.bfirst.data[(int64_t)(u)]) + ((n1) * ((int64_t)ctx.blen.data[(int64_t)(u)])));
}
AArr<int64_t> _av11;
_av11.cap = (int64_t)(nowners);
_av11.len = (int64_t)(nowners);
_av11.data = (int64_t*)arena_alloc((int64_t)(nowners) * sizeof(int64_t));
for (int64_t _z12 = 0; _z12 < (int64_t)(nowners); _z12 += 1) { _av11.data[_z12] = 0; }
ctx.bn2s = _av11;
AArr<int64_t> _av13;
_av13.cap = (int64_t)(nowners);
_av13.len = (int64_t)(nowners);
_av13.data = (int64_t*)arena_alloc((int64_t)(nowners) * sizeof(int64_t));
for (int64_t _z14 = 0; _z14 < (int64_t)(nowners); _z14 += 1) { _av13.data[_z14] = 0; }
ctx.bn2e = _av13;
AArr<int64_t> _av15;
_av15.cap = (int64_t)(nowners);
_av15.len = (int64_t)(nowners);
_av15.data = (int64_t*)arena_alloc((int64_t)(nowners) * sizeof(int64_t));
for (int64_t _z16 = 0; _z16 < (int64_t)(nowners); _z16 += 1) { _av15.data[_z16] = 0; }
ctx.ms = _av15;
AArr<int64_t> _av17;
_av17.cap = (int64_t)(nowners);
_av17.len = (int64_t)(nowners);
_av17.data = (int64_t*)arena_alloc((int64_t)(nowners) * sizeof(int64_t));
for (int64_t _z18 = 0; _z18 < (int64_t)(nowners); _z18 += 1) { _av17.data[_z18] = 0; }
ctx.me = _av17;
for (int64_t r = 0; r < (nowners); r += 1) {
    w0 = ((int64_t)ctx.bound.data[(int64_t)(r)]);
    w1 = ((int64_t)ctx.bound.data[(int64_t)((r) + 1)]);
    u0 = 0;
    u1 = 0;
    for (int64_t u = 0; u < (units); u += 1) {
        if ((((w0) >= ((int64_t)ctx.bfirst.data[(int64_t)((u) + 1)])) != 0)) {
            u0 = ((u) + 1);
        }
        if ((((w1) > ((int64_t)ctx.bfirst.data[(int64_t)((u) + 1)])) != 0)) {
            u1 = ((u) + 1);
        }
    }
    ctx.bn2s.data[(int64_t)(r)] = (int64_t)(u0);
    ctx.bn2e.data[(int64_t)(r)] = (int64_t)((u1) + 1);
    ctx.ms.data[(int64_t)(r)] = (int64_t)((w0) - ((int64_t)ctx.bfirst.data[(int64_t)(u0)]));
    ctx.me.data[(int64_t)(r)] = (int64_t)((w1) - ((int64_t)ctx.bfirst.data[(int64_t)(u1)]));
}
a->metadata[(int64_t)0] = (uint32_t)1;
a->metadata[(int64_t)1] = (uint32_t)((int64_t)a->aic_core_num);
a->metadata[(int64_t)2] = (uint32_t)(nowners);
a->metadata[(int64_t)3] = (uint32_t)(total);
for (int64_t r = 0; r < (nowners); r += 1) {
    base = (16 + ((r) * 16));
    a->metadata[(int64_t)((base) + 0)] = (uint32_t)((int64_t)ctx.bn2s.data[(int64_t)(r)]);
    a->metadata[(int64_t)((base) + 1)] = (uint32_t)((int64_t)ctx.bn2e.data[(int64_t)(r)]);
    a->metadata[(int64_t)((base) + 2)] = (uint32_t)((int64_t)ctx.ms.data[(int64_t)(r)]);
    a->metadata[(int64_t)((base) + 3)] = (uint32_t)((int64_t)ctx.me.data[(int64_t)(r)]);
}
return (0);
}

