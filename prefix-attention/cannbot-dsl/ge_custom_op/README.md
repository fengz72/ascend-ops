# PrefixAttention — GE 自定义算子(custom_op_v2,形态1)

把 prefix-attention 接入 GE 图:单个算子节点 `PrefixAttention`,其 `Execute()`
内部完成完整 host 编排(零初始化 metadata workspace → launch AICPU metadata
planner → launch macro attention kernel),两次 launch 同 stream 串行,图上不
可见第二个节点。

**kernel 来源**:GE 交付件由 CMake 调 **bisheng 从签入的 codegen AscendC 源码**
(`../codegen/*.asc`、`../codegen/aicpu/*`)编译出 `kernel_fp16.so` /
`kernel_bf16.so` / `aicpu_metadata.so`,`libcust_opapi.so` 运行时 dlopen 它们。
**GE 构建与运行均不需要 cannbotdsl**——cannbotdsl 仅在 `dump_kernel_src.py`
重新生成 .asc 源码时用到(kernel 改动时才跑)。

参考:CANN GE `custom_op_v2` 自定义算子开发指南(场景 A:动态图在线执行)。

## 两个交付件(解耦)

| 交付件 | 安装 | 依赖 | 用途 |
|---|---|---|---|
| python 前端 | `../scripts/install.sh` | cannbotdsl | 开发/验证:eager 直调、golden 对拍 |
| GE 交付件 | `../scripts/install_ge.sh` | 仅 CANN/bisheng+cmake | 模型接入:torchair 下沉 / GE 图 / OM |

python 前端(`prefix_attn.torch_ops`)注册 torch op + Meta + converter,并自动
挂载 OPP;eager 走 DSL python host(需 cannbotdsl),下沉走 GE C++(不需)。
`prefix_attn` 包用 PEP 562 懒加载,故**无 cannbotdsl 环境也能 import torch_ops
完成下沉**,只有 eager 直调才需要 DSL(见 `../tests/test_no_dsl_env.py`)。

## 算子原型

```cpp
REG_OP(PrefixAttention)
    .INPUT(query,  TensorType({DT_FLOAT16, DT_BF16}))   // [T,N1,64] TND
    .INPUT(key,    TensorType({DT_FLOAT16, DT_BF16}))   // [T,N2,64]
    .INPUT(value,  TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(actual_seq_lengths, TensorType({DT_INT32}))  // [B+1] 段边界 cumsum([P,L0,...])
    .ATTR(scale, Float, 0.0)                            // <=0 → 1/sqrt(D)
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))  // shape/dtype = query
```

约束与 `prefix_attn/validate.py` 一致(D=64、N1%N2==0、bounds 严格递增等,
Execute 内 host 侧校验,无 D2H 同步)。

## 目录

| 路径 | 内容 |
|---|---|
| `include/prefix_attention_op.h` | REG_OP proto(构图侧/ES 生成共用) |
| `src/prefix_attention_proto.cc` | InferShape/InferDataType + IMPL_OP 注册 |
| `src/prefix_attention_op.cpp` | `EagerExecuteOp::Execute` + `REG_AUTO_MAPPING_OP` |
| `src/dsl_launcher.{h,cpp}` | schema 驱动的 ABI v9 frame 构造 + dlopen 三产物 launch |
| `src/artifacts.h` | 生成:.asc/.so 名 + host ABI 符号 + 核数默认值(dump 脚本产出) |
| `scripts/dump_kernel_src.py` | **唯一需要 cannbotdsl 的环节**:dump fp16/bf16 .asc + aicpu 源码 + 生成 artifacts.h/SOURCE_INFO(kernel改动时才跑) |
| `CMakeLists.txt` | bisheng 编 .asc/aicpu → 三 .so;CXX 编 libcust_opapi.so;ES wrapper 生成 |
| `probe/` | P0 ABI 探针:C++ 独立 launch 与 python host 路径**逐位一致**(已 PASS) |
| `opp/` | OPP 交付包(`ASCEND_CUSTOM_OPP_PATH` 指向此;`prefix_attn.torch_ops` import 时自动挂载) |
| `tests/` | GE 在线 Session 端到端 + torchair 小网络下沉测试 |

## 关键机制:host ABI v9 frame

kernel .so 自描述(`codegen/launch_prefix_macro_*.asc` 尾部):

- `<base>__cannbot_schema_v2()` → `CannBotHostSchemaV2*`:frame_bytes/alignment、
  contract_digest、每 slot 的 offset/pointer_offset/shape_offset/stride_offset
- `<base>__cannbot_invoke_v9(stream, frame, bytes)` → 严格校验 header
  (magic `0x43424639`/abi=9/digest)后 launch

`dsl_launcher.cpp` 完全按 schema 填 frame(不硬编码布局),8 个 slot 依次为
out/query/key/value/scale/bounds/metadata/block_dim。AICPU 产物走扁平 C ABI
`_prefix_attn_metadata_kernel_do(stream, bounds, metadata, len, n1, n2, aic, macro_cost)`。
核数 = min(设备 AIC 数, 36),`PREFIX_ATTN_AIC_CORES` 可收窄,metadata 参数与
macro kernel block_dim 同源保证一致。

**为何 fp16/bf16 分两个 .so**:两特化的设备 kernel 符号同名
(`kernel__PrefixMacroKernel___call___0`),同库会多重定义冲突;host invoke/schema
符号带各自 hash 区分,故各编独立 .so、按 dtype dlopen。

## 构建与验证

```bash
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh   # ASCEND_HOME_PATH

# (仅 kernel 改动时)重新 dump .asc 源码,需 cannbotdsl + NPU
python ge_custom_op/scripts/dump_kernel_src.py

# GE 交付件:bisheng 从签入 .asc 编译 + 安装 OPP(无需 cannbotdsl)
bash scripts/install_ge.sh

# P0 探针:C++ 独立 launch vs python host 路径,逐位比对(需 cannbotdsl 出参考)
bash ge_custom_op/probe/run_probe.sh     # 期望 PROBE_PASS (bit-exact)

# GE 在线 Session 端到端
bash ge_custom_op/tests/run_online.sh    # 期望 GE_SESSION_PASS

# torchair 小网络下沉端到端
bash ge_custom_op/tests/run_torchair.sh  # 期望 TORCHAIR_NET_PASS
```

## PyTorch 中使用

torchair 随 torch_npu 自带(`torch_npu.dynamo.torchair`)。装好两个交付件后:

```python
import torch, torch_npu
import prefix_attn.torch_ops   # 注册 torch.ops.ascendc_ops.prefix_attention
                               # + 自动挂载 OPP + 注册 torchair converter

# eager(需 cannbotdsl;走 DSL python host)
out = torch.ops.ascendc_ops.prefix_attention(q, k, v, bounds, 0.125)

# 图模式(下沉到 GE PrefixAttention C++ 节点,运行时不需 cannbotdsl)
from torch_npu.dynamo import torchair
compiled = torch.compile(model, backend=torchair.get_npu_backend())
out = compiled(x, bounds)
```

`prefix_attn.torch_ops` import 时自动把 `opp/` 挂到 `ASCEND_CUSTOM_OPP_PATH`
(可用 `PREFIX_ATTN_OPP` 覆盖)。下沉验证(test_torchair_net.py)在编译前把
python host 路径替换为 raise——若 graph break 回退 eager 必报错;实测通过且与
eager 逐位一致,证明真实下沉到 GE 节点。msprof 确认每次 Execute 产生两个
task:`AI_CPU _prefix_attn_metadata_kernel` + `MIX_AIC kernel__PrefixMacroKernel`。

## 部署

```bash
export ASCEND_CUSTOM_OPP_PATH="<...>/ge_custom_op/opp:$ASCEND_CUSTOM_OPP_PATH"
```

- 交付件:`opp/op_graph/lib/linux/aarch64/libcust_opapi.so` + 同目录
  `artifacts/`(三个 kernel .so + MANIFEST.json);均为构建产物,不入库
- 板级绑定(NPU_ARCH=dav-3510 / Ascend950PR);换板用 `-DNPU_ARCH=...` 重编,
  或重跑 dump_kernel_src.py
- `ge.Tensor.get_data()` 走文本序列化,fp16 比对有 ~1e-7 噪声;设备侧输出与
  python 路径逐位一致(见 probe)

## 状态与后续

- [x] P0 ABI 探针(bit-exact PASS)
- [x] GE kernel 改为 codegen .asc 源码经 bisheng 同构建(clean 重建无需 cannbotdsl)
- [x] P1 GE 在线 Session 端到端(PASS,msprof 双 kernel 确认)
- [x] P2 PyTorch/TorchAir 前端(小网络下沉 PASS,与 eager 逐位一致)
- [x] python 前端去 cannbotdsl 硬依赖(无 DSL 环境可 import + 下沉)
- [ ] P3 OM 离线下沉(CompilableOp/PortableOp,参考 compilable_add_custom 样例)
