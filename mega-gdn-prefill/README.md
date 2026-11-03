# MegaGdnPrefill — Qwen3.5 GDN 线性注意力整块融合算子（vendored）

`MegaGdnPrefillOp` 在一次设备调用中融合宽度 4 的 causal conv（含 cache 更新）、
packed QKV 拆分、Q/K L2 归一化、decay/beta gate（含 softplus）、chunk GDN 递推
（KKT/下三角逆/WY/H-O）、SSM cache 读写与 gated RMSNorm×SiLU(z)。本目录是该算子
（vendored 源码）与项目自建的 GE 图模式包装（MegaGdnPrefill）。

上游 <https://gitcode.com/jd-opensource/xllm_ops> @ `3271763`（Apache-2.0）；
算子代码 `src/{mega_gdn_prefill_op,mega_chunk_gdn,causal_conv1d}/` 与上游逐字节一致。

## 构建 + 安装

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh   # CANN 环境脚本（本机路径）
bash install.sh      # 一条命令：构建 .run + 安装算子到 $ASCEND_HOME_PATH/opp
                     # + pip 注册 torch 绑定（editable -> vendor op_api/python/）
# 覆盖项：SOC_VERSION=<soc> / OPS=<ops> / GE_PYTHON=<python> bash install.sh
```

`.so` 与 python 绑定统一由 CANN opp 的 vendor 包承载
（`$ASCEND_OPP_PATH/vendors/mega_gdn_prefill/{op_api/lib, op_api/python}`；图模式
kernel 编译走标准 tbe-custom store，故必须装入全局 opp）。activator import 时经
`$ASCEND_CUSTOM_OPP_PATH`/`$ASCEND_OPP_PATH` 自发现 `.so`，dlopen（RTLD_GLOBAL，
GE 可见 REG_OP）并挂载 vendor 包；eager 桥 JIT 链接同一 `.so`。pip 注册后随处
import，零 shell 导出。多 vendor 环境需 `load_priority=mega_gdn_prefill` 写入
`$ASCEND_OPP_PATH/vendors/config.ini`（install.sh 在该文件缺失时写入，已有其它
条目时告警）。

- **部署机（无仓库）**：`bash cann_ops-mega_gdn_prefill_*.run --install-path=$ASCEND_HOME_PATH/opp`
  后 `pip install -e $ASCEND_OPP_PATH/vendors/mega_gdn_prefill/op_api/python --no-deps --no-build-isolation`
- **开发机备选**：`pip install -e <repo>/mega-gdn-prefill/python --no-deps`（改
  `python/` 即时生效；同包名与 vendor 安装二选一，后装者生效）

## 精度门禁

```bash
python3 test/test_mega_gdn_prefill.py [device]    # 前置：bash install.sh（含绑定注册）
```

门禁：graph vs eager `rel_l2 < 0.001`、eager vs CPU golden `cos > 0.95`，外加
z 契约断言（输出别名 z、原地覆盖）。eager 腿 torch.ops → JIT pybind 桥
（torch extension cache，需 ninja）→ `.so` 内 flat 入口 `mega_gdn_prefill_run_v2`。

## 目录

```
src/          算子源码 + 项目自建接入件（GE 包装 / eager flat 入口 / common.h，见改动 5）
python/       torch 绑定源（activator + JIT binding + bridge.cpp + pyproject.toml）
install.sh    构建 + 安装 + 注册，一条命令
test/         精度门禁
cmake/ common/ scripts/   构建机器与共享代码（有修改与裁剪，见"相对上游的改动"）
```

## 相对上游的改动

约定：算子代码零修改，构建机器文件允许修改。

1. **新增 `install.sh`**（原名 `build.sh`）：包装 `src/build.sh`，重定向
   `--build-dir`/`--cann_3rd_lib_path` 到 `build/`/`.cache/cann_3rd_lib`，保持源码树
   干净；构建后自动安装 `.run` 到 CANN 全局 opp 并 pip 注册绑定。
2. **cmake 文件**：
   - `cmake/third_party/abseil-cpp.cmake` + `ascend_protobuf.cmake`：第三方解压
     落点重定向到 `${CANN_3RD_LIB_PATH}`（上游写进源码树）；
   - `cmake/custom_build.cmake`：(a) CPack makeself 路径重定向；(b) EXISTS 守卫钩子
     把 `src/mega_gdn_prefill_ge.cpp` 与 `src/mega_gdn_prefill_eager.cc` 编进
     `cust_opapi`（aclnn 实现、GE 包装与 eager flat 入口为单一
     `libcust_opapi.so`）；(c) python 四件 install 进 vendor `op_api/python`
     （部署机脱离源码树安装）。
3. **`src/attention/`** 仅保留 `CMakeLists.txt`（装配过滤器，上游
   `add_subdirectory` 无条件引用它）。
4. **`.gitignore`** 追加 `.cache/`、`test/_ge_out/`。
5. **项目自建接入件（非算子代码）**：`src/mega_gdn_prefill_common.h`（aclnn 声明
   与 FFTS 解析，GE/eager 共享）、`src/mega_gdn_prefill_eager.cc`（`extern "C"
   mega_gdn_prefill_run_v2`，零 torch 头依赖）、`python/mega_gdn_prefill_binding.py`
   + `mega_gdn_prefill_bridge.cpp`（JIT eager 桥）、`python/pyproject.toml`（pip
   editable 元数据）；删除旧 `python/custom_ops_lib.py`（ctypes）与
   `python/torchair.py`（死 shim，torch_npu 注入顶层 torchair 别名）。
6. **裁剪与三算子无关的上游遗留**（约 100 个文件），构建产物逐字节等价（仅
   `libcust_opmaster_rt2.0.so` 少 4 个 fallback 死代码导出符号，CANN built-in
   自带同名符号）：
   - `common/`：onnx 插件、fallback、旧 tiling 副本、`aicpu/`、`include/static/`；
   - `cmake/`：UT/示例测试链、RTY 流程、built-in/static 流程、无引用死文件；
   - 同步修改引用处：`src/CMakeLists.txt`、`cmake/{config,func,intf,ut,package,
     build_empty_package}.cmake`、`common/CMakeLists.txt`、`src/build.sh`（删 CI
     脚本调用行）；
   - **保留易误判项**：`common/stub/`（glob 引用，提供 `optiling`/`opapi_math`
     导入目标）、`cmake/scripts/fix_format.sh`（`generate_compile_cmd` ALL 目标
     调用）、`cmake/ut.cmake`（`src/CMakeLists.txt` 无条件 include）。

## GE 包装（MegaGdnPrefill）

上游 `num_matrices` 是编译期 INT64 属性，直接进图会把动态形状烧死；custom-op-v2
机制（`REG_OP` + `EagerExecuteOp::Execute`）每次执行从 `cu_seqlens` 现算，B/S 保持
符号化。关键决策：

- **输出别名 z**（`MakeOutputRefInput`）：绕开 GE 执行器输出内存跨执行复用池
  （小→大会踩踏）；z 在本算子之后不得再被消费。torch 侧 schema 为
  `Tensor(a!) z ... -> Tensor`——mutation 注解使 dynamo 保持 z 读序在算子之后，
  返回值不带别名注解（AOT functionalization 拒绝自定义算子输出别名注解）；
  eager 桥同语义原地写 z 并返回之。
- **scratch 走 `ctx->MallocWorkSpace`**（框架管理，不受上述复用缺陷影响）。
- **A2/A3 FFTS 地址**经 `dlsym("libruntime.so", "rtGetC2cCtrlAddr")` 现取。

## 约束

- head dim / chunk 固定 128，conv 宽度固定 4，`NV % NK == 0`。
- chunk GDN 流水内部 FP16（`GDN_PREFILL_COMPUTE_DTYPE`），公共 ABI BF16/FP32。
- 以 `ascend910_93` 为验证目标；A5 源码保留但未验证。
- 上游持续演进，本副本不自动跟进；同步时以上述 rev 为基线 diff。
