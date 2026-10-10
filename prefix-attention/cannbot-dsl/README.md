# prefix_attn (CANNBotDSL DN 实现)

prefix-in-Q prefill attention 的 CANNBotDSL 实现,仅含 DN(K@Q^T)变体:
去重 TND 布局下,共享 prefix 的 Q/KV/输出只存一份、只算一份;各 batch 私有段
attend prefix KV + 自身私有 KV,causal;GQA(N1 % N2 == 0),D=64,
fp16/bf16,fp32 score/PV 累加。

## 安装(一键,按板执行)

```bash
bash scripts/install.sh
```

依次完成:① 导出 kernel 产物(DN fp16/bf16 AOT .so + AICPU .so,已存在则跳过,
`PREFIX_ATTN_FORCE_EXPORT=1` 强制重导)② 构建/安装 GE 交付件 `libcust_opapi.so`
③ `pip install -e` 本包 ④ eager 冒烟。之后**任意目录**的 python 里:

```python
import torch, torch_npu
import prefix_attn.torch_ops        # 注册 torch op + 自动挂载 OPP + torchair converter

out = torch.ops.prefix_attn.dn(q, k, v, bounds)          # eager(scale 可选,0=1/sqrt(D))

# 图模式(下沉到 GE PrefixAttnDn C++ 节点):forward 里用 torch.ops.prefix_attn.dn
from torch_npu.dynamo import torchair                    # torch_npu 自带,无需另装
compiled = torch.compile(model, backend=torchair.get_npu_backend())
```

原始函数接口(不经 torch.ops)同样可用:

```python
from prefix_attn import prefix_attn_dn

# q [T,N1,64], k/v [T,N2,64] (npu, contiguous, fp16/bf16)
# actual_seq_lengths: int32 [B+1] 段边界,b[0]=p>0,末项=T,严格递增
out = prefix_attn_dn(q, k, v, actual_seq_lengths)          # scale 默认 1/sqrt(64)
```

## 目录

| 路径 | 内容 |
|---|---|
| `prefix_attn/dn.py` | DN kernel(cube/vector 数据通路旋转为 K@Q^T 方向;`@host` 入口 + AOT compile 缓存,T/B 动态维) |
| `prefix_attn/metadata.py` | AICPU compact-box metadata planner(每次调用先跑的小 kernel,为各 AIC 划分 work item) |
| `prefix_attn/validate.py` | host 侧输入校验(契约见上;D!=64、N1<N2、非 contiguous 等直接 raise) |
| `prefix_attn/torch_ops.py` | PyTorch 前端:torch.ops.prefix_attn.dn 注册(PrivateUse1/Meta)+ OPP 自动挂载 + torchair converter |
| `scripts/install.sh` | 一键安装(产物导出 → GE 交付件构建 → pip install -e → 冒烟) |
| `pyproject.toml` | 打包配置(prefix-attn-dn) |
| `codegen/` | codegen 产物快照:`launch_prefix_macro_dn.asc` 为 DN kernel(float16 特化)的 AscendC 代码;`codegen/aicpu/` 为 metadata AICPU kernel 的生成源码(与 dtype 无关),再生成命令见下 |
| `ge_custom_op/` | GE 自定义算子接入(custom_op_v2 形态1):单节点 `PrefixAttnDn`,Execute 内双 launch;含产物导出脚本、ABI 探针、OPP 包与 GE Session 端到端测试,详见其 README |
| `tests/` | 最简 NPU 功能测试:fp64 golden,4 case(单 batch / GQA 多 batch / 两个生产 uniform shape) |

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

## codegen 快照再生成

`codegen/` 内是 DSL 编译 DN kernel 时 dump 的 AscendC 代码(float16 特化):

```bash
cd cannbot-dsl
rm -rf codegen && mkdir codegen
CANNBOTDSL_DUMP_ASCENDC=1 CANNBOTDSL_DUMP_DIR=$PWD/codegen python - <<'EOF'
import torch
from prefix_attn import prefix_attn_dn
bounds = [20] + [20 + 130 * (i + 1) for i in range(10)]
gen = torch.Generator().manual_seed(71000)
q = torch.randn((bounds[-1], 14, 64), generator=gen).half()
k = torch.randn((bounds[-1], 2, 64), generator=gen).half()
v = torch.randn((bounds[-1], 2, 64), generator=gen).half()
prefix_attn_dn(q.npu(), k.npu(), v.npu(),
               torch.tensor(bounds, dtype=torch.int32, device="npu"))
torch.npu.synchronize()
EOF
```

对应源码版本记录在 `codegen/SOURCE_INFO.json`(源仓 sha256)。

metadata AICPU kernel 的生成源码(与 dtype 无关,运行时由
`compile_aicpu_kernel()` 在临时目录即时生成编译):

```bash
python - <<'EOF'
from pathlib import Path
from prefix_attn.metadata import _prefix_attn_metadata_kernel as k
out = Path("codegen/aicpu"); out.mkdir(parents=True, exist_ok=True)
for fname, text in k.generate_sources(launch_mode="interface").items():
    (out / fname).write_text(text)
EOF
```

已验证:`codegen/aicpu/` 与真实编译路径(`compile_aicpu_kernel`)落盘源码
逐字节一致,且可正常编译链接出 `.so`。

## 性能(Ascend950PR,msprof Task Duration min,warmup 5 + repeat 50)

| case | DN@32AIC | FIA sm0 | FIA sm2/3 |
|---|---:|---:|---:|
| p20/130×10 fp16 | 17.14 us | 21.18 us | 17.29/17.34 us |
| p25/193×10 bf16 | 17.84 us | 20.59 us | 17.94/17.94 us |

FIA = `npu_fused_infer_attention_score`(口径 A 物化 prefix,多算 11–14% 行);
完整数据与方法见源仓 `prefix_attn/perf/dn_vs_fia_20261009/README.md`。
注意:上表仅为 DN attention kernel 本身;每次调用还有一个独立的 AICPU
metadata kernel(~27 us),端到端墙钟需另计。
