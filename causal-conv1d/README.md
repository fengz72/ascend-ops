# causal-conv1d — fla-npu CausalConv1d vendored + GE custom-op 入图

vendored [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)
`gdn_preprocess/causal_conv1d` 算子（AscendC）与项目自建的 GE custom-op-v2
图模式包装（`CausalConv1dGE`）。算子源码
`fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/` 与上游逐字节一致（main @
`56a21dc`）；`cmake/`、`common/`、`scripts/`、`build.sh`、`CMakeLists.txt`
为 fla-npu 构建机制快照（含本项目适配改动，见文末改动清单）。许可见包内
`LICENSE`：算子源码 BSD-3-Clause，构建机制文件 CANN OSL v2.0。

性能背景（startlux-decision-v1.1-0.8b-instant 模型 GDN conv 路径，A3）：
fla CausalConv1d 单算子约 56 µs/层（NPU event，含 silu、dim-last 直连），
替换模型现状 7 算子链（Transpose×2 + TransData×2 + Conv2D + Slice + Swish，
128.9 µs/层）后 conv 全路径 −57%，每请求 −1.31 ms（18 层）。测量见模型目录
`work/fla-npu-bench/`。本包 eager 腿实测 82–91 µs/次（含 torch op 派发 +
JIT 桥 host 开销，与 fla wheel stable launcher 84–89 µs 同量级）。

## 算子语义

GE 包装只覆盖 **dense FN prefill**（runMode=0）：

```
y = activation(sum_j weight[j] * x[t-j])   # 左上下文补零，depthwise
```

- `x`：dim-last `(B,S,D)`，dtype bf16/fp16；`D` 需为 16 的倍数
- `weight`：`(W,D)`，W ∈ {2,3,4}；`bias`：`(D,)` 可选
- `activation`：`"none"` / `"silu"`（默认）/ `"swish"`
- 输出 `y` 与 `x` 同 shape/dtype
- 2D `(T,D)` 输入上游 kernel 报 561002 不支持（fla wheel 同样），不覆盖；
  varlen（queryStartLoc）/cache（UPDATE 模式）路径不随 GE 包装暴露，
  如有需要直接走上游 aclnn API

## 构建 + 安装

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # CANN 环境脚本
bash install.sh      # 构建 .run + 安装算子到 $ASCEND_HOME_PATH/opp
                     # + 写 load_priority + pip 注册 torch 绑定
# 覆盖项：SOC_VERSION=<soc> / OPS=<ops> bash install.sh
```

- vendor 目录 `$ASCEND_OPP_PATH/vendors/causal_conv1d/`，图模式 kernel 编译
  走标准 tbe-custom store，全局 opp 必须有本包
- **同名算子类型 `CausalConv1d` 冲突披露**：vendored mega-gdn-prefill 包
  （xllm 旧谱系）也携带一个 `CausalConv1d`（本仓库无运行时调用方）。
  install.sh 将本包写进 `vendors/config.ini` 的 `load_priority` 首位，
  GE/aclnn 解析以本包为准；mega 包自身算子不受影响
- torch 绑定随 vendor 包 `op_api/python/` 交付（pip editable 注册，
  包名 `causal-conv1d-torch`），import 即生效，零 shell 导出

## torch 接入

```python
import causal_conv1d                       # dlopen + 挂 opp + 注册三件套
y = torch.ops.ascend_ops.causal_conv1d(x, weight, bias=None,
                                      activation="silu", head_num=0)
```

- eager：`torch.ops.ascend_ops.causal_conv1d`（PrivateUse1 → JIT pybind 桥
  → libcust_opapi.so 内 flat 入口 `causal_conv1d_run` → aclnnCausalConv1d）
- 图模式：activator 注册 torchair converter，`torch.compile`（npu backend）
  时 fx 节点映射为 `torchair.ge.custom_op("CausalConv1dGE")`，GE 侧
  REG_OP `CausalConv1dGE`（EagerExecuteOp 直调 aclnnCausalConv1d，
  kernel 二进制/tiling 复用 vendored 算子）
- OPP 挂载形态：`_mount_opp` 把 vendor 包目录（`<opp>/vendors/causal_conv1d`，
  含 `op_api/`+`op_impl/`+`op_proto/`）prepend 进 `ASCEND_CUSTOM_OPP_PATH`。
  官方 custom-op-v2 文档的 `op_graph/` 最小布局只装 GE 包装 .so、不携带
  AscendC kernel/tiling；本包是 vendored 算子 + GE wrapper 混合交付，必须走
  标准自定义算子 vendor 包布局，vendor 目录形态与 fla-npu wheel/torch_npu
  生态约定一致（GE 注册经进程内 dlopen，不依赖 op_graph 目录扫描）。已验证
  全局 opp `load_priority` 不含本包时，仅凭该挂载 eager+graph 均可解析
  （graph 与 eager 逐位一致）
- `torch.library.Library` 句柄模块级存活（局部变量会被 GC 注销 schema）；
  torch_npu 先于 torchair import（torchair 别名注入）

**坑位记录**（torch 2.12）：schema 含 `Tensor? bias`（无默认值）且注册了
Meta impl 时，dispatcher 以 boxed 调用约定调 python kernel，会丢弃 bias
之后的标量参数——`Tensor? bias=None` 必须显式带默认值，且 eager/meta impl
参数自带同语义的 python 默认值兜底短调用。

## 精度门禁

```bash
python3 test/test_causal_conv1d.py [device]    # 前置：bash install.sh
```

门禁（mega-gdn-prefill 口径）：graph vs eager `rel_l2 < 0.001`（实测
0.00000000，逐位一致）、eager vs CPU fp32 golden `cos > 0.95` 且
`rel_l2 < 0.02`（实测 bf16 rel_l2 ≈ 0.0017、fp16 ≈ 0.0002）。

bf16 kernel 相对 fp32 golden 的全局 cosine 随 batch 缓慢下降（B1 0.9997、
B3 0.9985）而 rel_l2 恒为 0.0017——同输入下与上游 fla-npu wheel 逐位一致，
属 kernel 固有 bf16 行为而非包装缺陷，故门禁取 rel_l2 为主。

覆盖 startlux 模型全部 4 个 GDN conv 真实 shape（(1,99,6144)、(3,99,6144)、
(1,130,6144)、(1,85,6144)）、小 D、`activation="none"`+bias、fp16 变体。

## 目录

```
fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/   vendored 算子（零修改）
fla/ops/ascendc/common/   fla 通用 kernel_utils（构建机制经 config.cmake 随包
                           安装，本算子不引用，勿删）
src/                  项目自建：CausalConv1dGE GE 包装 + eager flat 入口
python/               torch 绑定（activator + JIT binding + bridge）
cmake/ common/ scripts/ build.sh CMakeLists.txt     fla-npu 构建机制快照
test/                 精度门禁
```

## 构建机制改动清单（相对 fla-npu main @ 56a21dc）

- `CMakeLists.txt` / `cmake/*.cmake` / `build.sh`：VENDOR_NAME 默认
  `causal_conv1d`，vendor 目录去掉 `_transformer` 后缀；`.run` 更名
  `causal_conv1d_linux-*.run`
- `cmake/custom_build.cmake`：cust_opapi 目标后挂 EXISTS 守卫钩子，把
  `src/causal_conv1d_ge.cpp`、`src/causal_conv1d_eager.cc` 编进单一
  `libcust_opapi.so`，并把 `python/` 四件随 vendor 包 `op_api/python`
  安装；删除 fla wheel 专用的 CPack ABI-manifest 前置脚本（本包不随
  launcher 交付）
- `build.sh`：fla Stable-ABI launcher 门禁跳过（torch_custom 不随本包
  交付）
- `version.cmake` / `version.info`：随根 CMakeLists 引入（机制文件）
- 其余（opbuild/tiling/runtimeKB/依赖获取等）与上游一致；算子源码零修改

## 已知限制

- 部署环境同时安装 fla-npu wheel（`fla_npu_transformer` vendor 包）时，
  两份 `libcust_opapi.so` 同名共存：activator 以 `$ASCEND_OPP_PATH` 优先
  定位本包并校验 `causal_conv1d_run` 符号 fail-loud；同进程先后 dlopen
  两份库时退出阶段可能报 `double free or corruption`（进程退出清理顺序，
  不影响计算结果；生产 runtime 只装本包即无此现象）
