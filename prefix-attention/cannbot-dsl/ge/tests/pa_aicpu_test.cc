// AICPU planner test: frozen-golden equivalence + concurrency hammers for
// the per-thread TLS scratch arena.
//
//   pa_aicpu_test <artifacts_dir> [threads] [iters] [device]
//     defaults: threads=8 iters=60 device=0
//   pa_aicpu_test <artifacts_dir> --regen [device]
//     re-baseline: print a fresh pa_aicpu_golden.h from the CURRENT AICPU
//     output (only for a deliberate planner change -- see the provenance
//     note in pa_aicpu_golden.h).
//
// Phases:
//   1. equivalence       frozen golden vectors (independent PlanHost
//                        snapshot, see pa_aicpu_golden.h) + zero-tail check
//   2. sync hammer       per-launch device sync; structural invariants
//   3. deep-queue hammer pure-enqueue bursts (in-flight queue genuinely
//                        deepens: no malloc/memcpy/sync between launches);
//                        structural invariants
//   4. same-bounds       identical bounds on every stream (deterministic
//                        offset-0 collision probe for a broken/shared TLS
//                        arena); structural invariants
//
// The structural check validates the planner's output contract without an
// oracle: HEAD fields recomputed from bounds, the used cores' (bN2, m) boxes
// must partition the flat work space [0, total) exactly, and unused core
// records must stay all-zero.  Exact plan values are anchored by phase 1
// (frozen golden) and by the e2e reference gate (original HF forward).
#include <dlfcn.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "pa_aicpu_golden.h"

namespace {

using DoFn = void (*)(void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t,
                      uint32_t, uint32_t);

// Allocation length of the metadata GM buffer (PREFIX_ATTN_METADATA_ELEMS /
// metadata_padded_count() of the kernel package).
constexpr int kMetaElems = 4096;
constexpr int kTile = 128;

int g_failures = 0;

#define CHECK(cond, msg)                                                     \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("  [FAIL] %s (%s:%d)\n", msg, __FILE__, __LINE__);         \
      g_failures++;                                                          \
    }                                                                        \
  } while (0)

#define ACL_OK(expr)                                                        \
  do {                                                                       \
    const aclError _rc = (expr);                                             \
    if (_rc != ACL_SUCCESS) {                                                \
      std::printf("  [FAIL] %s ret=%d (%s:%d)\n", #expr, (int)_rc, __FILE__, \
                  __LINE__);                                                 \
      g_failures++;                                                          \
    }                                                                        \
  } while (0)

int64_t CeilDiv(int64_t a, int64_t b) { return a / b + ((a % b) != 0); }

// Structural validation of one metadata buffer against its bounds.
// Returns 0 when the planner contract holds, else a phase-specific code.
//
// Contract (see prefix_attn/metadata.py): unit u=0 is the shared prefix
// [0, bounds[0]) with ptiles rows per q_head; unit u>=1 is private batch
// u-1; unit u owns n1*bl_u rows m = q_head*bl_u + q_tile.  Core r's record
// (bn2_start, bn2_end, m_start, m_end) covers the flat range
// [bfirst[bn2_start]+m_start, bfirst[bn2_end-1]+m_end), and the used cores'
// ranges partition [0, total) exactly.  Unused records are all-zero.
int CheckStructure(const std::vector<int32_t> &m,
                   const std::vector<int32_t> &bounds) {
  const int64_t n1 = kGoldenN1;
  const int64_t units = static_cast<int64_t>(bounds.size());
  std::vector<int64_t> blen(static_cast<size_t>(units));
  std::vector<int64_t> bfirst(static_cast<size_t>(units) + 1, 0);
  blen[0] = CeilDiv(bounds[0], kTile);
  for (int64_t u = 1; u < units; ++u) {
    blen[u] = CeilDiv(bounds[u] - bounds[u - 1], kTile);
  }
  for (int64_t u = 0; u < units; ++u) {
    bfirst[u + 1] = bfirst[u] + n1 * blen[u];
  }
  const int64_t total = bfirst[units];

  if (m[0] != 1 || m[1] != kGoldenAicCores) return 1;
  const int64_t used = m[2];
  if (used < 1 || used > kGoldenAicCores || used > total) return 2;
  if (static_cast<int64_t>(m[3]) != total) return 3;

  int64_t expect_w0 = 0;
  for (int64_t r = 0; r < used; ++r) {
    const int32_t *rec = m.data() + 16 + r * 16;
    const int64_t bn2s = rec[0], bn2e = rec[1], ms = rec[2], me = rec[3];
    if (bn2s < 0 || bn2s >= bn2e || bn2e > units) return 4;
    if (ms < 0 || ms >= n1 * blen[bn2s]) return 5;
    if (me < 1 || me > n1 * blen[bn2e - 1]) return 6;
    const int64_t w0 = bfirst[bn2s] + ms;
    const int64_t w1 = bfirst[bn2e - 1] + me;
    if (w1 <= w0 || w0 != expect_w0) return 7;
    expect_w0 = w1;
  }
  if (expect_w0 != total) return 8;
  for (int64_t r = used; r < kGoldenAicCores; ++r) {
    const int32_t *rec = m.data() + 16 + r * 16;
    for (int k = 0; k < 4; ++k) {
      if (rec[k] != 0) return 9;
    }
  }
  return 0;
}

// One deferred-verification launch of the deep-queue hammers.
struct BurstLaunch {
  void *bounds_dev = nullptr;
  void *meta_dev = nullptr;
  std::vector<int32_t> bounds;
};

std::vector<int32_t> ReadMeta(void *meta_dev) {
  std::vector<int32_t> out(kMetaElems);
  ACL_OK(aclrtMemcpy(out.data(), out.size() * 4, meta_dev, out.size() * 4,
                     ACL_MEMCPY_DEVICE_TO_HOST));
  return out;
}

std::vector<int32_t> RunAicpu(DoFn do_fn, aclrtStream stream,
                              const std::vector<int32_t> &bounds) {
  const size_t n = bounds.size();
  void *bounds_dev = nullptr;
  void *meta_dev = nullptr;
  ACL_OK(aclrtMalloc(&bounds_dev, n * 4, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&meta_dev, kMetaElems * 4, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMemcpy(bounds_dev, n * 4, bounds.data(), n * 4,
                     ACL_MEMCPY_HOST_TO_DEVICE));
  // zero-init (planner contract: unused core records must read as zero)
  std::vector<int32_t> zero(kMetaElems, 0);
  ACL_OK(aclrtMemcpy(meta_dev, kMetaElems * 4, zero.data(), kMetaElems * 4,
                     ACL_MEMCPY_HOST_TO_DEVICE));
  do_fn(stream, reinterpret_cast<uint64_t>(bounds_dev),
        reinterpret_cast<uint64_t>(meta_dev), static_cast<uint32_t>(n),
        kGoldenN1, 2, kGoldenAicCores, 1U);
  // NOTE: stream sync does not reliably cover AICPU tasks (see the
  // concurrency bug report); the device-wide sync is the safe mode.  The
  // GRAPH path relies on in-stream queue ordering instead, validated by the
  // e2e tests.
  ACL_OK(aclrtSynchronizeDevice());
  auto out = ReadMeta(meta_dev);
  aclrtFree(bounds_dev);
  aclrtFree(meta_dev);
  return out;
}

// Deep-queue hammer core: prep ALL buffers first (malloc + h2d), then burst
// pure async enqueues back-to-back (no host work between launches, so the
// device queue actually piles up), one device sync, then structurally
// verify every output.
int DeepQueueBurst(DoFn do_fn, int tid, int iters, uint32_t seed, int device,
                   bool same_bounds, const std::vector<int32_t> &fixed_bounds) {
  if (aclrtSetDevice(device) != ACL_SUCCESS) return iters;
  aclrtStream s;
  if (aclrtCreateStream(&s) != ACL_SUCCESS) return iters;
  std::mt19937 rng(seed);
  std::vector<BurstLaunch> launches(iters);
  int prepped = 0;
  for (int i = 0; i < iters; ++i) {
    BurstLaunch &L = launches[i];
    if (same_bounds) {
      L.bounds = fixed_bounds;
    } else {
      // bigger T / more segments than the sync hammer: meatier planner tasks
      // (total ~ n1 * T/128 work units) so the pure enqueue burst outpaces
      // the device and the in-flight queue genuinely deepens
      L.bounds.assign(1, 1 + static_cast<int>(rng() % 200));
      const int64_t T = 2000 + static_cast<int>(rng() % 14000);
      const int segs = 2 + static_cast<int>(rng() % 80);
      for (int k = 0; k < segs; ++k) {
        L.bounds.push_back(L.bounds.back() +
                           std::max<int32_t>(1, static_cast<int32_t>(
                                                    (T - L.bounds.back()) /
                                                    (segs - k))));
      }
    }
    const size_t n = L.bounds.size();
    if (aclrtMalloc(&L.bounds_dev, n * 4, ACL_MEM_MALLOC_HUGE_FIRST) ||
        aclrtMalloc(&L.meta_dev, kMetaElems * 4,
                    ACL_MEM_MALLOC_HUGE_FIRST)) {
      break;
    }
    std::vector<int32_t> zero(kMetaElems, 0);
    if (aclrtMemcpy(L.bounds_dev, n * 4, L.bounds.data(), n * 4,
                    ACL_MEMCPY_HOST_TO_DEVICE) ||
        aclrtMemcpy(L.meta_dev, kMetaElems * 4, zero.data(), kMetaElems * 4,
                    ACL_MEMCPY_HOST_TO_DEVICE)) {
      break;
    }
    ++prepped;
  }
  // pure enqueue burst: no malloc/memcpy/sync between launches
  const auto t_enq0 = std::chrono::steady_clock::now();
  for (int i = 0; i < prepped; ++i) {
    const BurstLaunch &L = launches[i];
    do_fn(s, reinterpret_cast<uint64_t>(L.bounds_dev),
          reinterpret_cast<uint64_t>(L.meta_dev),
          static_cast<uint32_t>(L.bounds.size()), kGoldenN1, 2,
          kGoldenAicCores, 1U);
  }
  const auto t_enq1 = std::chrono::steady_clock::now();
  // one device-wide sync covers the whole burst (and other threads' tasks);
  // the wait after the last enqueue is the backlog drain time -- proof the
  // queue actually piled up behind the burst
  aclrtSynchronizeDevice();
  const auto t_drain1 = std::chrono::steady_clock::now();
  const double enq_ms =
      std::chrono::duration<double, std::milli>(t_enq1 - t_enq0).count();
  const double drain_ms =
      std::chrono::duration<double, std::milli>(t_drain1 - t_enq1).count();
  std::printf("  [t%d] enqueue %d in %.1f ms (%.1f us/enq), backlog drain "
              "%.1f ms\n",
              tid, prepped, enq_ms, 1000.0 * enq_ms / prepped, drain_ms);
  int fails = 0;
  char tag[64];
  for (int i = 0; i < prepped; ++i) {
    const BurstLaunch &L = launches[i];
    const int rc = CheckStructure(ReadMeta(L.meta_dev), L.bounds);
    if (rc != 0) {
      std::snprintf(tag, sizeof(tag), "t%d deep it%d", tid, i);
      std::printf("  [%s] structural check rc=%d (T=%d P=%d)\n", tag, rc,
                  L.bounds.back(), L.bounds.front());
      fails++;
    }
  }
  for (int i = 0; i < prepped; ++i) {
    aclrtFree(launches[i].bounds_dev);
    aclrtFree(launches[i].meta_dev);
  }
  aclrtDestroyStream(s);
  return fails;
}

int RegenGolden(DoFn do_fn, int device) {
  std::printf("// Re-baselined from the AICPU kernel's own output on this "
              "device (%d).\n", device);
  std::printf("// Only legitimate for a DELIBERATE planner change -- see the "
              "provenance\n// note this replaces.\n");
  aclrtStream s;
  ACL_OK(aclrtCreateStream(&s));
  for (size_t c = 0; c < sizeof(kGoldenCases) / sizeof(kGoldenCases[0]); ++c) {
    const GoldenCase &g = kGoldenCases[c];
    const std::vector<int32_t> b(g.bounds, g.bounds + g.bounds_len);
    const auto m = RunAicpu(do_fn, s, b);
    std::printf("static const int32_t g_bounds_c%zu[] = {", c);
    for (size_t i = 0; i < b.size(); ++i) {
      std::printf("%s%d", i ? ", " : "", b[i]);
    }
    std::printf("};\n");
    std::printf("static const int32_t g_meta_c%zu[kGoldenElems] = {\n", c);
    for (size_t i = 0; i < kGoldenElems; ++i) {
      std::printf("%s%d", (i % 12) ? ", " : (i ? ",\n" : ""), m[i]);
    }
    std::printf("};\n\n");
  }
  ACL_OK(aclrtDestroyStream(s));
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 5) {
    std::printf("usage: %s <artifacts_dir> [threads] [iters] [device]\n"
                "       %s <artifacts_dir> --regen [device]\n",
                argv[0], argv[0]);
    return 2;
  }
  const bool regen = (argc >= 3 && std::strcmp(argv[2], "--regen") == 0);
  const int kThreads = (argc >= 3 && !regen) ? std::atoi(argv[2]) : 8;
  const int kIters = (argc >= 4 && !regen) ? std::atoi(argv[3]) : 60;
  const int kDevice = regen ? ((argc >= 4) ? std::atoi(argv[3]) : 0)
                            : ((argc >= 5) ? std::atoi(argv[4]) : 0);
  if (!regen) {
    std::printf("config: threads=%d iters=%d device=%d\n", kThreads, kIters,
                kDevice);
  }

  const std::string so_path = std::string(argv[1]) + "/aicpu_metadata.so";
  void *so = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (so == nullptr) {
    std::printf("dlopen(%s) failed: %s\n", so_path.c_str(), dlerror());
    return 2;
  }
  auto do_fn = reinterpret_cast<DoFn>(
      dlsym(so, "_prefix_attn_metadata_kernel_do"));
  if (do_fn == nullptr) {
    std::printf("dlsym failed\n");
    return 2;
  }

  ACL_OK(aclInit(nullptr));
  ACL_OK(aclrtSetDevice(kDevice));

  if (regen) {
    const int rc = RegenGolden(do_fn, kDevice);
    ACL_OK(aclrtResetDevice(kDevice));
    ACL_OK(aclFinalize());
    return rc;
  }

  // ---- phase 1: equivalence against the frozen golden vectors ----
  {
    aclrtStream eq_stream;
    ACL_OK(aclrtCreateStream(&eq_stream));
    for (const auto &g : kGoldenCases) {
      const std::vector<int32_t> b(g.bounds, g.bounds + g.bounds_len);
      const auto aicpu = RunAicpu(do_fn, eq_stream, b);
      bool eq = std::equal(aicpu.begin(), aicpu.begin() + kGoldenElems,
                           g.metadata);
      for (int i = kGoldenElems; i < kMetaElems; ++i) {
        eq = eq && (aicpu[i] == 0);
      }
      CHECK(eq, "aicpu planner == frozen golden");
      if (!eq) {
        int d = 0;
        while (d < kGoldenElems && aicpu[d] == g.metadata[d]) d++;
        std::printf("  first diff @%d: aicpu=%d golden=%d\n", d, aicpu[d],
                    g.metadata[d]);
      }
      std::printf("T=%d P=%d: %s\n", b.back(), b.front(),
                  eq ? "MATCH" : "MISMATCH");
    }
    ACL_OK(aclrtDestroyStream(eq_stream));
  }

  // ---- phase 2: sync hammer (per-launch device sync) ----
  {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int> thread_fail(kThreads, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        if (aclrtSetDevice(kDevice) != ACL_SUCCESS) {
          thread_fail[t] = 1;
          return;
        }
        aclrtStream s;
        if (aclrtCreateStream(&s) != ACL_SUCCESS) {
          thread_fail[t] = 1;
          return;
        }
        std::mt19937 rng(1000 + t);
        for (int i = 0; i < kIters; ++i) {
          // same total T, varying P/own split -> different plans every launch
          std::vector<int32_t> b{1 + static_cast<int>(rng() % 40)};
          const int64_t T = 300 + static_cast<int>(rng() % 1500);
          for (int k = 0; k < 10; ++k) {
            b.push_back(b.back() + 1 + static_cast<int>(rng() % 200));
          }
          const auto aicpu = RunAicpu(do_fn, s, b);
          const int rc = CheckStructure(aicpu, b);
          if (rc != 0) {
            thread_fail[t]++;
            std::printf("  [t%d it%d] structural check rc=%d (T=%d P=%d)\n", t,
                        i, rc, b.back(), b.front());
          }
        }
        aclrtSynchronizeStream(s);
        aclrtDestroyStream(s);
      });
    }
    for (auto &th : threads) {
      th.join();
    }
    int reentr_fail = 0;
    for (int f : thread_fail) {
      reentr_fail += f;
    }
    CHECK(reentr_fail == 0, "sync hammer");
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    std::printf("phase2 sync hammer: %d/%d mismatches (%.0f ms, %.1f us/launch)\n",
                reentr_fail, kThreads * kIters, ms,
                1000.0 * ms / (kThreads * kIters));
  }

  // ---- phase 3: deep-queue hammer (no intermediate sync) ----
  {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int> thread_fail(kThreads, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        if (aclrtSetDevice(kDevice) != ACL_SUCCESS) {
          thread_fail[t] = kIters;
          return;
        }
        thread_fail[t] =
            DeepQueueBurst(do_fn, t, kIters, 2000 + t, kDevice, false, {});
      });
    }
    for (auto &th : threads) {
      th.join();
    }
    int fails = 0;
    for (int f : thread_fail) {
      fails += f;
    }
    CHECK(fails == 0, "deep-queue hammer");
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    std::printf("phase3 deep-queue hammer: %d/%d mismatches (%.0f ms, %.1f us/launch)\n",
                fails, kThreads * kIters, ms, 1000.0 * ms / (kThreads * kIters));
  }

  // ---- phase 4: same-bounds deep-queue (deterministic offset collision
  //      probe for a shared/broken TLS arena) ----
  {
    // T~8k / 41 segments: meatier task, identical on every stream
    std::vector<int32_t> fixed{200};
    for (int k = 0; k < 40; ++k) {
      fixed.push_back(fixed.back() + 200);
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int> thread_fail(kThreads, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        if (aclrtSetDevice(kDevice) != ACL_SUCCESS) {
          thread_fail[t] = kIters;
          return;
        }
        thread_fail[t] = DeepQueueBurst(do_fn, t, kIters, 3000 + t, kDevice,
                                        true, fixed);
      });
    }
    for (auto &th : threads) {
      th.join();
    }
    int fails = 0;
    for (int f : thread_fail) {
      fails += f;
    }
    CHECK(fails == 0, "same-bounds deep-queue hammer");
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    std::printf("phase4 same-bounds deep-queue: %d/%d mismatches (%.0f ms, %.1f us/launch)\n",
                fails, kThreads * kIters, ms, 1000.0 * ms / (kThreads * kIters));
  }

  ACL_OK(aclrtResetDevice(kDevice));
  ACL_OK(aclFinalize());
  if (g_failures != 0) {
    std::printf("AICPU TEST FAIL: %d failures\n", g_failures);
    return 1;
  }
  std::printf("AICPU TEST PASS\n");
  return 0;
}
