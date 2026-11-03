# MegaGdnPrefill — Qwen3.5 GDN 线性注意力整块融合算子（vendored）

`MegaGdnPrefillOp` 在一次设备调用中融合宽度 4 的 causal conv（含 cache 更新）、
packed QKV 拆分、Q/K L2 归一化、decay/beta gate（含 softplus）、chunk GDN 递推
（KKT/下三角逆/WY/H-O）、SSM cache 读写与 gated RMSNorm×SiLU(z)。
本目录是该算子（vendored 源码）与项目自建的 GE 图模式包装（MegaGdnPrefill）。

## 来源

| 项 | 值 |
|---|---|
| 上游 | https://gitcode.com/jd-opensource/xllm_ops @ `3271763` |
| 许可 | Apache-2.0（`LICENSE`） |

算子代码（`src/{mega_gdn_prefill_op,mega_chunk_gdn,causal_conv1d}/`）与上游逐字节一致。

## 目录

```
src/          算子源码 + mega_gdn_prefill_ge.cpp（GE 包装）
python/       activator（dlopen + torch op 注册 + torchair converter）
test/         精度测试（CPU golden vs eager vs torchair）
cmake/        构建机器（3 个文件有修改，其余与上游一致）
common/       共享代码（与上游一致）
scripts/      构建脚本（与上游一致）
```

## 构建

```bash
source set_env.sh
bash build.sh                # 默认 ascend910_93，三算子
SOC_VERSION=ascend910b bash build.sh
```

产物 `build/cann_ops-mega_gdn_prefill_*.run`。安装到 CANN：

```bash
bash build/cann_ops-mega_gdn_prefill_*.run --install-path=$ASCEND_HOME_PATH/opp
echo "load_priority=mega_gdn_prefill" > $ASCEND_OPP_PATH/vendors/config.ini
```

## 相对上游的改动

约定：算子代码零修改，构建机器文件允许修改。

1. **新增 `build.sh`**：包装 `src/build.sh`，重定向 `--build-dir`/`--cann_3rd_lib_path`
   到 `build/`/`.cache/cann_3rd_lib`，保持源码树干净。
2. **三个 cmake 文件**：
   - `cmake/third_party/abseil-cpp.cmake` + `ascend_protobuf.cmake`：
     第三方解压落点重定向到 `${CANN_3RD_LIB_PATH}`（上游写进源码树）。
   - `cmake/custom_build.cmake`：
     (a) CPack makeself 路径重定向；
     (b) EXISTS 守卫钩子把 `src/mega_gdn_prefill_ge.cpp` 编进 `cust_opapi` 目标
     （aclnn 实现与 GE 包装编译为单一 `libcust_opapi.so`）。
3. **`src/attention/`** 仅保留 `CMakeLists.txt`（装配过滤器，上游 `add_subdirectory`
   无条件引用它，空目录即可跳过无关算子）。
4. **`.gitignore`** 追加 `.cache/`、`test/_ge_out/`。

## GE 包装（MegaGdnPrefill）

上游 `num_matrices` 是编译期 INT64 属性，直接进图会把动态形状烧死。
`MegaGdnPrefill`（custom-op-v2 机制：`REG_OP` + `EagerExecuteOp::Execute` 回调）
在每次执行时从 `cu_seqlens` 现算 `num_matrices`，B/S 保持符号化。

关键设计决策：
- **输出别名 z**（`MakeOutputRefInput`）：绕开 GE 执行器输出内存跨执行复用池
  （小→大会踩踏，实测复现为内核 MTE 无效地址）。z 在本算子之后不得再被消费。
- **scratch 走 `ctx->MallocWorkSpace`**（框架管理，不受复用缺陷影响——
  实测该缺陷仅命中 `MallocOutputTensor` 路径）。
- **A2/A3 FFTS 地址** 经 `dlsym("libruntime.so", "rtGetC2cCtrlAddr")` 现取。

加载链：`ASCEND_CUSTOM_OPP_PATH` 指向 `install/opp`（v2 发现）；eager 绑定读同一
`install/opp/op_api/lib/`。依赖安装（`models/.../scripts/build_mega_gdn.sh`）把
单一 `.so` 硬链接到三处入口（同 inode，一次 dlopen）。

## 约束

- head dim / chunk 固定 128，conv 宽度固定 4，`NV % NK == 0`。
- chunk GDN 流水内部 FP16（`GDN_PREFILL_COMPUTE_DTYPE`），公共 ABI BF16/FP32。
- 以 `ascend910_93` 为验证目标；A5 源码保留但未验证。
- 上游持续演进，本副本不自动跟进；同步时以上述 rev 为基线 diff。
