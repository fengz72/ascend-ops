# prefix_attn (CANNBotDSL 实现)

prefix-in-Q prefill attention:去重 TND 布局下,共享 prefix 的 Q/KV/输出只存一份、
只算一份;各 batch 私有段 attend prefix KV + 自身私有 KV,causal;GQA
(N1 % N2 == 0),D=64,fp32 score/PV 累加。

**GE 交付件(`ge/`)只支持 fp16**,不支持 bf16——原因与恢复路径见
`ge/README.md`「为什么只支持 fp16」。python 前端(`prefix_attn/`)是
dtype 泛型的 DSL 内核源码,仅用于开发/验证,不打包交付。

## 安装

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

bash ge/scripts/install.sh     # GE 交付件:bisheng 从签入 .asc 编译,直装 CANN 目录(无需 cannbotdsl)
pip install -e ge --no-deps    # torch binding(prefix_attn_torch):模型直调
```

python 前端(`prefix_attn/`)是开发/验证源码,不打包:`cd cannbot-dsl`
(或 `PYTHONPATH=<此目录>`)后直接 import。

## 使用

```python
# eager 直调(python 前端,需 cannbotdsl;PEP 562 懒加载,import 不触发 DSL)
from prefix_attn import prefix_attention

# q [T,N1,64], k/v [T,N2,64] (npu, contiguous, fp16)
# actual_seq_lengths: int32 [B+1] 段边界,b[0]=p>0,末项=T,严格递增
out = prefix_attention(q, k, v, actual_seq_lengths)   # scale 默认 1/sqrt(64)
```

```python
# torch.ops + 图模式(torch binding,无需 cannbotdsl)
# GE 交付件为 qwen2.5-0.5b 特化:N1=14/N2=2 固定(launch 时校验);T/B 动态
import prefix_attn_torch                              # 注册 + OPP 挂载 + converter

out = torch.ops.ascendc_ops.prefix_attention(q, k, v, bounds, 0.125)

from torch_npu.dynamo import torchair                 # torch_npu 自带
compiled = torch.compile(model, backend=torchair.get_npu_backend())
out = compiled(x, bounds)                             # 下沉到 GE PrefixAttention 节点
```

## kernel 改动时再生成(唯一需要 cannbotdsl 的环节,需 NPU)

`ge/codegen/` 是 GE 交付件的**构建源码**(非参考快照):macro kernel 的 fp16
AscendC + metadata AICPU planner。kernel(`prefix_attn/*.py`)改动后:

```bash
cd cannbot-dsl
python ge/scripts/dump_kernel_src.py    # dump .asc/aicpu + 生成 ge/src/artifacts.h
bash ge/scripts/install.sh              # 重新编译安装
```

> 交付件仅 fp16(原因见 `ge/README.md`);host invoke/schema 符号带特化
> hash 区分(记录在 artifacts.h)。

## 目录

| 路径 | 内容 |
|---|---|
| `prefix_attn/attention.py` | macro kernel(K@Q^T 方向;`@host` 入口 + AOT 缓存,T/B 动态维) |
| `prefix_attn/metadata.py` | AICPU metadata planner(每次调用先跑,为各 AIC 划分 work item) |
| `prefix_attn/validate.py` | host 侧输入校验 |
| `prefix_attn/__init__.py` | PEP 562 懒加载 |
| `ge/` | GE 自定义算子工程(custom_op_v2,自包含):构建/安装/使用见 `ge/README.md` |

## 环境基线

| 项 | 版本 |
|---|---|
| SoC | Ascend950PR_9589(32 AIC / 64 AIV) |
| CANN | 9.2.0-beta.2 |
| cannbotdsl | 0.7.0 |
| torch / torch_npu | 2.10.0 / 2.10.0.post2 |
| Python | 3.12 |

核数:python host 运行时经 torch_npu 查询设备(上限 36);GE 侧固定 16。

## 性能(Ascend950PR,msprof Task Duration min,warmup 5 + repeat 50)

| case | 本算子@32AIC | FIA sm0 | FIA sm2/3 |
|---|---:|---:|---:|
| p20/130×10 fp16 | 17.14 us | 21.18 us | 17.29/17.34 us |

FIA = `npu_fused_infer_attention_score`(口径 A 物化 prefix,多算 11–14% 行)。
上表仅为 attention kernel 本身;每次调用另有独立的 AICPU metadata kernel
(~27 us),端到端墙钟需另计。
