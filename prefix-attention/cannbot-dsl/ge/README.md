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

```text
主流：前序任务 -> record(input_ready) -> wait/reset(metadata_ready) -> Attention
辅流：           wait/reset(input_ready) -> memset -> planner -> record(metadata_ready)
```

两个方向的 wait 后都需要 reset，避免模型重复重放时使用上一次执行的 event 信号。

GE 管理辅流。event 按 ACL context、模型辅流、节点名隔离，支持多个模型使用
相同节点名。**调用方必须在当前 context 中所有相关模型完成执行并卸载之后、
`GEFinalize` 之前，调用 `prefix_attn_release_attached_events()`**；此函数返回
ACL 状态码。清理期间不能并发加载或执行相关模型。不会在静态析构中调用 ACL。
C++ 调用方在使用处声明 `extern "C" int prefix_attn_release_attached_events();`。
event 管理与清理实现集中在 `src/prefix_attention_op.cc`，不再额外提供资源头文件或测试计数接口。
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

已知限制：output size 增大可能踩踏内存；验证时应保持 shape 不变或变小。
