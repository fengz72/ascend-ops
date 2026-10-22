# PrefixAttention — GE 自定义算子(custom_op_v2)

**最低环境要求：CANN ≥ 9.3.0。GE 图模式仅提供 AICPU planner 辅流实现。**

单节点 `PrefixAttention`:Execute 内双 launch(AICPU metadata planner + macro
attention kernel),kernel 由 CMake 调 bisheng 从签入的 AscendC 源码
(`codegen/`)编译,构建与运行均**无需 cannbotdsl**。

## 构建

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh   # 或任一 CANN 环境

bash ge/scripts/install.sh    # bisheng 编 .asc/aicpu + CXX 编 libcust_opapi.so,
                              # 装入 $ASCEND_OPP_PATH/op_graph/
```

- (仅 kernel 改动时)先重新 dump AscendC 源码,需 cannbotdsl + NPU:
  `python ge/scripts/dump_kernel_src.py`
- 换板:`NPU_ARCH=<arch> bash ge/scripts/install.sh`(默认 dav-3510/Ascend950PR)
- 安装目录直接使用 CANN 提供的 `$ASCEND_OPP_PATH`，需先 source `set_env.sh`。
  CANN 升级后重新 source 环境并重跑 `install.sh`。
- 每次 CMake 配置均重新编译检查辅流接口，支持切换 CANN 或原路径重装后重试。
- 非 torch 侧(GE Session/ATC)需:
  `export ASCEND_CUSTOM_OPP_PATH="$ASCEND_OPP_PATH${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"`

## 为什么只支持 fp16(不支持 bf16)

- **没有消费方**:本交付件服务的 qwen2.5-0.5b GE 管线全程 fp16(`dtype: float16`),
  bf16 需求为 0——为不存在的需求维护第二套内核纯属负担。
- **dtype 即独立 .so**:fp16/bf16 设备内核符号同名(`kernel__PrefixMacroKernel`),
  只能各编一个 .so。多交付一个 dtype = 多一份 ~4300 行签入 .asc、多一个
  kernel_bf16.so、DT_NEEDED 与安装面翻倍。
- **bf16 链路从未构建验证**:CMake 只编 fp16;bf16 的 bisheng 编译 + ABI v9 启动 +
  GE 注册从未在板验证过。交付未验证的路径违背 fail-loudly 原则(proto 已收窄为
  `DT_FLOAT16`,bridge 同步拒绝)。

将来需要 bf16:上游 DSL 内核源码(`prefix_attn/attention.py`)本身 dtype 泛型,
恢复 `dump_kernel_src.py` 的 bf16 dump + CMake 加 `kernel_bf16.so` 目标 + 放开
proto/bridge 的 dtype 校验即可。

## 使用(PyTorch)

```bash
pip install -e ge --no-deps    # torch binding(prefix_attn_torch)
```

```python
import torch, torch_npu
import prefix_attn_torch       # 注册 torch.ops + 自动挂载 OPP + torchair converter

out = torch.ops.ascendc_ops.prefix_attention(q, k, v, bounds, 0.125)  # eager

# 图模式(下沉到 GE PrefixAttention C++ 节点)
from torch_npu.dynamo import torchair
compiled = torch.compile(model, backend=torchair.get_npu_backend())
out = compiled(x, bounds)
```

- eager 与图模式均直走 GE 交付件 .so,无需 cannbotdsl
- kernel 为 qwen2.5-0.5b 特化:N1=14、N2=2 固定(`artifacts.h` 记录,launch 时校验);T/B 动态
- 输入:`q [T,N1,64]`、`k/v [T,N2,64]`(npu, contiguous, **fp16 only**)、
  `bounds` int32 `[B+1]`(首项=prefix 长度>0,末项=T,严格递增);`scale<=0` 时取
  `1/sqrt(64)`;输出 shape/dtype 同 query
- 核数:GE 侧固定 16;python 侧运行时经 torch_npu 查询设备(上限 36)

## GE AICPU 辅流版本

GE 图模式的 metadata 清零和 AICPU planner 固定放在
`RequestAttachedStream("prefix_attention.aicpu")` 上，Attention kernel 仍在主流。
CMake 必须通过辅流与资源申报接口检查；已在 CANN 9.3.0 B140 验证。

`bounds` 的 dtype 为 int32。内部属性 `bounds_preloaded` 默认 false，
保持输入就绪同步，兼容静态图内的输入搬运和图内生成的 bounds。
TorchAir converter 仅对带原生适配导出器输入标记的 bounds、直接 Data 输入、
bounds 与 query 均有符号动态维度时设置该属性为 true；当前 Qwen 动态执行路径已通过实际地址跟踪确认直接读取外部输入。
调用方必须先完成同步 H2D，再启动图执行，不能仅凭 dtype 就假定输入已经就绪。
普通未标记的 torch.compile 输入保持默认同步，不自动推断异步生产者已完成。

```text
Qwen 动态直接输入（bounds_preloaded=true）：
执行前：同步 H2D(int32 bounds) 完成
主流：Q/K/V、RoPE及其他计算 -> wait/reset(metadata_ready) -> Attention
辅流：memset -> planner -> record(metadata_ready)

其他路径（bounds_preloaded=false，默认）：
主流：输入搬运或生产者完成 -> record(input_ready) -> wait/reset(metadata_ready) -> Attention
辅流：wait/reset(input_ready) -> memset -> planner -> record(metadata_ready)
```

planner 不读取 Q/K/V。已预加载路径删除每层主流到辅流等待，
所有路径保留 metadata_ready 的 wait/reset，保证 Attention 读取本轮规划结果。
静态 GE 图可能把外部输入搬到模型内部地址，不能无条件删除 input_ready。
V1 下沉路径重放已录制任务，RT2 动态路径由 Execute 实际下发任务；
提前执行的幅度取决于任务下发和调度，需通过 profiling 确认。

metadata 使用按 ACL context、模型辅流、节点名隔离的独立 16 KiB 缓冲区，
避免辅流提前清零时覆盖主流分配器正在复用的 workspace。
已预加载路径每节点只申请一个 metadata_ready event，其他路径另有 input_ready。
24层模型每实例有效 metadata 共384 KiB，不含分配器对齐开销。
全局锁仅保护主机侧资源表，不包围 kernel 发射或设备等待。
提前规划路径同一模型实例必须完成上一次执行后再调用；多个实例各自持有资源。

GE 管理辅流。event 和 metadata 按 ACL context、模型辅流、节点名隔离，支持多个模型使用
相同节点名。**调用方必须在当前 context 中所有相关模型完成执行并卸载之后、
`GEFinalize` 之前，调用 `prefix_attn_release_attached_events()`**；此函数返回
ACL 状态码，并释放 event 与 metadata 缓冲区。清理期间不能并发加载或执行相关模型。不会在静态析构中调用 ACL。
C++ 调用方在使用处声明 `extern "C" int prefix_attn_release_attached_events();`。
资源管理与清理实现集中在 `src/prefix_attention_op.cc`，不再额外提供资源头文件或测试计数接口。
在接入现有模型服务前，需将此清理调用接到服务的模型卸载流程中。

以下命令在仓库根目录执行，安装到当前 `$ASCEND_OPP_PATH`：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
GE_SRC=prefix-attention/cannbot-dsl/ge
cmake -S "$GE_SRC" -B "$GE_SRC/build-aux" -DCMAKE_BUILD_TYPE=Release
cmake --build "$GE_SRC/build-aux" -j4
cmake --install "$GE_SRC/build-aux"
export ASCEND_CUSTOM_OPP_PATH="$ASCEND_OPP_PATH${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
```

基础正确性验证可使用固定 `T=2048, batch=16, N1=14, N2=2, D=64, FP16`，
轮换输入数值与 bounds 内容，逐元素对比独立 FP32 causal-prefix/GQA 参考结果，
并在模型卸载后确认 event 清理成功。

### 动态 shape 与输出扩容

同一动态 shape 模型可在执行间改变 `T`、`batch` 和 bounds 内容；
`N1=14、N2=2、D=64、FP16` 仍为内核特化约束。
动态输出增大要求 GE 包含 `EagerOpExecutionContext::MallocOutputTensor`
按本轮逻辑字节数检查已有容量、不足时重新分配的修复，并使用配套的
GE Compiler / Executor。旧说明中的内存踩踏来自未扩容就复用旧输出缓冲区；
以下已修复环境中未复现该问题。

2026-10-21 在 A5 Ascend950PR、逻辑卡 1 上验证：
CANN `9.3.0~weekly.20261014.01`，GE `upstream/9.3.0-beta.1`
@ `cec5518fe3eb39dd7eec7b1fd124280b2ff6eaf2`，Compiler / Executor 由同一源码构建。
两条路径各在同一个动态模型中依次执行下表 12 组 shape，包含增大、缩小、再增大，
轮换输入数值与 bounds 内容。

| 步骤 | T | batch | 输出字节数 | GE 自动分配输出 | 外部输出与哨兵 |
|---|---|---|---|---|---|
| 1 | 256 | 1 | 458752 | PASS | PASS |
| 2 | 512 | 4 | 917504 | PASS | PASS |
| 3 | 1024 | 8 | 1835008 | PASS | PASS |
| 4 | 2048 | 16 | 3670016 | PASS | PASS |
| 5 | 4096 | 32 | 7340032 | PASS | PASS |
| 6 | 1024 | 16 | 1835008 | PASS | PASS |
| 7 | 6144 | 32 | 11010048 | PASS | PASS |
| 8 | 257 | 4 | 460544 | PASS | PASS |
| 9 | 8192 | 32 | 14680064 | PASS | PASS |
| 10 | 512 | 8 | 917504 | PASS | PASS |
| 11 | 8192 | 32 | 14680064 | PASS | PASS |
| 12 | 2048 | 16 | 3670016 | PASS | PASS |

共 24 次执行、61,474,560 个元素均通过独立 FP32 causal-prefix/GQA 参考校验
（容差 `abs_error <= 0.001 + 0.01 * abs(reference)`）。
最大绝对误差 `0.000179022551`，两条路径的 RMSE 均为 `7.13565412e-06`。

分配跟踪确认每条路径均在同一个节点 context 中实际触发 6 次容量不足后的扩容，
输出从 448 KiB 增至 14 MiB；每次分配后的容量均足够，缩小时正确复用已有缓冲区。
Q/K/V 和 bounds 回读未被改写；外部输出缓冲区前后各 4096 字节哨兵均完整；
模型卸载后的 event 清理成功。验证范围为上述单模型串行 shape 切换，
哨兵检查覆盖外部输出边界。

### 2026-10-22 输入直传与提前规划验证

A5 Qwen的bounds输入改为int32后，100个case、2400次planner的地址跟踪确认
直接读取同步H2D完成的外部输入，无额外bounds搬运；24层均开启bounds_preloaded。
100-case端到端输出共12,800个值与保留入口同步的基线逐位一致，case-0000
通过独立FP32基线。静态图基础用例保留必要输入同步，3次执行、5,505,024个元素
通过独立FP32参考；四个动态图实例开启提前规划，12次执行、22,020,096个元素
通过参考，最大绝对误差0.000186502934，清理接口均返回成功。

固定实际shape为T=2048、batch16、N1=14、N2=2、D=64、FP16，
执行间轮换Q/K/V和bounds内容。同为int32的静态图仍可能发生模型内部输入搬运，
不能据此删除默认输入同步。当前版本的自动提前规划仅针对原生适配导出器的已验证动态直接输入。
