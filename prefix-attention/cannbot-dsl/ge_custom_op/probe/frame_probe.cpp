// P0 ABI probe: launch the prebuilt DSL artifacts from pure C++/ACL and
// verify bit-exactness against the python host path (probe/dump_case.py).
//
// Build & run: see probe/run_probe.sh
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <map>

#include "acl/acl.h"
#include "artifacts.h"
#include "dsl_launcher.h"

namespace {

std::vector<uint8_t> ReadFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

float HalfToFloat(uint16_t h) {
  uint32_t sign = (h >> 15) & 1u;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) bits = sign << 31;
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400u)) { man <<= 1; --exp; }
      man &= 0x3ffu;
      bits = (sign << 31) | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = (sign << 31) | (255u << 23) | (man << 13);
  } else {
    bits = (sign << 31) | ((exp + 112) << 23) | (man << 13);
  }
  float out;
  memcpy(&out, &bits, 4);
  return out;
}

#define ACL_CHECK(expr)                                                      \
  do {                                                                       \
    aclError err__ = (expr);                                                 \
    if (err__ != ACL_SUCCESS) {                                              \
      fprintf(stderr, "ACL error %d at %s:%d (%s)\n", (int)err__, __FILE__,  \
              __LINE__, #expr);                                              \
      exit(2);                                                               \
    }                                                                        \
  } while (0)

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: frame_probe <case_data_dir>\n");
    return 2;
  }
  std::string dir = argv[1];

  std::map<std::string, std::string> meta;
  {
    std::ifstream f(dir + "/meta.txt");
    std::string k, v;
    while (f >> k >> v) meta[k] = v;
  }
  const int64_t T = atoll(meta["T"].c_str());
  const int64_t n1 = atoll(meta["N1"].c_str());
  const int64_t n2 = atoll(meta["N2"].c_str());
  const int64_t bounds_len = atoll(meta["BOUNDS_LEN"].c_str());
  const float scale = strtof(meta["SCALE"].c_str(), nullptr);
  const bool bf16 = meta["DTYPE"] == "bfloat16";

  std::vector<uint8_t> hq = ReadFile(dir + "/q.bin");
  std::vector<uint8_t> hk = ReadFile(dir + "/k.bin");
  std::vector<uint8_t> hv = ReadFile(dir + "/v.bin");
  std::vector<uint8_t> hb = ReadFile(dir + "/bounds.bin");
  std::vector<uint8_t> href = ReadFile(dir + "/out_py.bin");

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  void *dq, *dk, *dv, *db, *dout, *dmeta;
  ACL_CHECK(aclrtMalloc(&dq, hq.size(), ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK(aclrtMalloc(&dk, hk.size(), ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK(aclrtMalloc(&dv, hv.size(), ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK(aclrtMalloc(&db, hb.size(), ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK(aclrtMalloc(&dout, href.size(), ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK(aclrtMalloc(&dmeta, PREFIX_ATTN_METADATA_ELEMS * 4,
                        ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK(aclrtMemcpy(dq, hq.size(), hq.data(), hq.size(),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dk, hk.size(), hk.data(), hk.size(),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dv, hv.size(), hv.data(), hv.size(),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(db, hb.size(), hb.data(), hb.size(),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemset(dmeta, PREFIX_ATTN_METADATA_ELEMS * 4, 0,
                        PREFIX_ATTN_METADATA_ELEMS * 4));
  ACL_CHECK(aclrtMemset(dout, href.size(), 0, href.size()));

  if (prefix_attn::LauncherInit() != 0) {
    fprintf(stderr, "LauncherInit failed: %s\n", prefix_attn::LauncherError());
    return 1;
  }
  int aic = prefix_attn::QueryAicCores();
  printf("aic_cores=%d (python used %s)\n", aic, meta["AIC"].c_str());

  prefix_attn::LaunchParams p{};
  p.out = dout; p.query = dq; p.key = dk; p.value = dv;
  p.total_tokens = T; p.n1 = n1; p.n2 = n2;
  p.bounds = db; p.bounds_len = bounds_len;
  p.metadata = dmeta;
  p.scale = scale; p.stream = stream; p.bf16 = bf16;
  int status = prefix_attn::LauncherRun(p);
  if (status != 0) {
    fprintf(stderr, "LauncherRun failed: status=%d err=%s\n", status,
            prefix_attn::LauncherError());
    return 1;
  }
  ACL_CHECK(aclrtSynchronizeStream(stream));

  std::vector<uint8_t> hout(href.size());
  ACL_CHECK(aclrtMemcpy(hout.data(), hout.size(), dout, href.size(),
                        ACL_MEMCPY_DEVICE_TO_HOST));

  size_t n = href.size() / 2;
  const uint16_t *got = reinterpret_cast<const uint16_t *>(hout.data());
  const uint16_t *ref = reinterpret_cast<const uint16_t *>(href.data());
  size_t mismatch = 0;
  double max_abs = 0.0;
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != ref[i]) ++mismatch;
    double d = std::fabs(HalfToFloat(got[i]) - HalfToFloat(ref[i]));
    if (d > max_abs) max_abs = d;
  }
  printf("elements=%zu bit_mismatch=%zu max_abs_diff=%.6g\n", n, mismatch,
         max_abs);
  bool pass = mismatch == 0;
  printf(pass ? "PROBE_PASS (bit-exact vs python host path)\n"
              : "PROBE_DIFF (not bit-exact; check max_abs_diff vs tolerance)\n");

  aclrtFree(dq); aclrtFree(dk); aclrtFree(dv);
  aclrtFree(db); aclrtFree(dout); aclrtFree(dmeta);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return pass ? 0 : 1;
}
