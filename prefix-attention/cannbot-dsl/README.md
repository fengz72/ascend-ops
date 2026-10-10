# prefix_attn (CANNBotDSL 实现)

prefix-in-Q prefill attention 的 CANNBotDSL 实现(K@Q^T 宏流水数据通路):
去重 TND 布局下,共享 prefix 的 Q/KV/输出只存一份、只算一份;各 batch 私有段
attend prefix KV + 自身私有 KV,causal;GQA(N1 % N2 == 0),D=64,
fp16/bf16,fp32 score/PV 累加。

## 安装(两个交付件解耦,按板执行)

```bash
bash scripts/install.sh       # python 前端:pip install -e + eager 冒烟(需 cannbotdsl)
bash scripts/install_ge.sh    # GE 交付件:bisheng 从签入 .asc 编译 + 装 OPP(无需 cannbotdsl)
```

- **python 前端**(开发/验证侧):eager 直调走 DSL host,需 cannbotdsl
- **GE 交付件**(模型接入侧):CMake 调 bisheng 从 `codegen/*.asc` 编 kernel .so,
  构建与运行均不需 cannbotdsl;cannbotdsl 仅在 kernel 改动后重跑
  `ge_custom_op/scripts/dump_kernel_src.py` 再生成 .asc 时用到

装好后**任意目录**的 python 里:

```python
import torch, torch_npu
import prefix_attn.torch_ops        # 注册 torch op + 自动挂载 OPP + torchair converter

out = torch.ops.ascendc_ops.prefix_attention(q, k, v, bounds)          # eager(scale 可选,0=1/sqrt(D))

# 图模式(下沉到 GE PrefixAttention C++ 节点):forward 里用 torch.ops.ascendc_ops.prefix_attention
from torch_npu.dynamo import torchair                    # torch_npu 自带,无需另装
compiled = torch.compile(model, backend=torchair.get_npu_backend())
```

> `prefix_attn` 用 PEP 562 懒加载:无 cannbotdsl 环境也能 `import prefix_attn.torch_ops`
> 完成图下沉,只有 eager 直调才需要 DSL(见 `tests/test_no_dsl_env.py`)。

原始函数接口(不经 torch.ops)同样可用:

```python
from prefix_attn import prefix_attention

# q [T,N1,64], k/v [T,N2,64] (npu, contiguous, fp16/bf16)
# actual_seq_lengths: int32 [B+1] 段边界,b[0]=p>0,末项=T,严格递增
out = prefix_attention(q, k, v, actual_seq_lengths)          # scale 默认 1/sqrt(64)
```

## 目录

| 路径 | 内容 |
|---|---|
| `prefix_attn/attention.py` | macro kernel(cube/vector 数据通路为 K@Q^T 方向;`@host` 入口 + AOT compile 缓存,T/B 动态维) |
| `prefix_attn/metadata.py` | AICPU compact-box metadata planner(每次调用先跑的小 kernel,为各 AIC 划分 work item) |
| `prefix_attn/validate.py` | host 侧输入校验(契约见上;D!=64、N1<N2、非 contiguous 等直接 raise) |
| `prefix_attn/torch_ops.py` | PyTorch 前端:torch.ops.ascendc_ops.prefix_attention 注册(PrivateUse1/Meta)+ OPP 自动挂载 + torchair converter |
| `prefix_attn/__init__.py` | PEP 562 懒加载(接入层与 DSL 实现层按 import 时机解耦) |
| `scripts/install.sh` | python 前端安装(pip install -e + eager 冒烟) |
| `scripts/install_ge.sh` | GE 交付件构建安装(bisheng 编 .asc → OPP,无需 cannbotdsl) |
| `pyproject.toml` | 打包配置(prefix-attention) |
| `codegen/` | **GE 构建源码**:`launch_prefix_macro_{fp16,bf16}.asc`(macro kernel AscendC)+ `aicpu/`(metadata planner 源码);由 `dump_kernel_src.py` 生成、签入,CMake+bisheng 据此编译 |
| `ge_custom_op/` | GE 自定义算子接入(custom_op_v2 形态1):单节点 `PrefixAttention`,Execute 内双 launch;含 dump 脚本、ABI 探针、CMake、OPP 包与端到端测试,详见其 README |
| `tests/` | NPU 功能测试(fp64 golden,4 case)+ 无 cannbotdsl 环境测试 |


## 环境要求(实测基线)

| 项 | 版本 |
|---|---|
| SoC | Ascend950PR_9589(32 AIC / 64 AIV) |
| CANN | 9.2.0-beta.2 |
| cannbotdsl | 0.7.0(commit af8e71f5,build 2026-09-30,cp312 aarch64) |
| torch / torch_npu | 2.10.0 / 2.10.0.post2 |
| Python | 3.12 |

核数:默认用满设备 AIC(上限常量 `AIC_CORE_NUM=36`);可用环境变量
`PREFIX_ATTN_AIC_CORES=N` 收窄(host launch 与 metadata planner 同步生效)。

## 运行测试

```bash
cd cannbot-dsl
python -m pytest tests/ -v        # 无 NPU / 无 torch_npu 时自动 skip
```

容差:fp16 0.003,bf16 0.025(与完整回归套件一致)。

## codegen 源码再生成(仅 kernel 改动时)

`codegen/` 内是 GE 交付件的**构建源码**(不是参考快照):macro kernel 的
fp16/bf16 AscendC(`launch_prefix_macro_{fp16,bf16}.asc`)+ metadata AICPU
planner 源码(`aicpu/`)。CMake 用 bisheng 据此编译出三个 kernel .so。

kernel(`prefix_attn/*.py`)改动后,重新生成(唯一需要 cannbotdsl 的环节,需 NPU):

```bash
cd cannbot-dsl
python ge_custom_op/scripts/dump_kernel_src.py
```

它一次性 dump fp16/bf16 .asc + aicpu 源码(去掉仅 host 仿真用的 `_x86.cpp`),
并生成 `ge_custom_op/src/artifacts.h`(host ABI 符号名/核数)与
`codegen/SOURCE_INFO.json`(源码 sha256、cannbotdsl 版本、device、编译 case)。
之后 `bash scripts/install_ge.sh` 重新编译 GE 交付件即可。

> fp16/bf16 设备 kernel 符号同名,故分两个 .so;host invoke/schema 符号带各自
> hash 区分(记录在 artifacts.h)。aicpu 源码与真实 `compile_aicpu_kernel()`
> 落盘逐字节一致(已验证)。


## 性能(Ascend950PR,msprof Task Duration min,warmup 5 + repeat 50)

| case | 本算子@32AIC | FIA sm0 | FIA sm2/3 |
|---|---:|---:|---:|
| p20/130×10 fp16 | 17.14 us | 21.18 us | 17.29/17.34 us |
| p25/193×10 bf16 | 17.84 us | 20.59 us | 17.94/17.94 us |

FIA = `npu_fused_infer_attention_score`(口径 A 物化 prefix,多算 11–14% 行);
完整数据与方法见源仓 `prefix_attn/perf/dn_vs_fia_20261009/README.md`。
注意:上表仅为 attention kernel 本身;每次调用还有一个独立的 AICPU
metadata kernel(~27 us),端到端墙钟需另计。
