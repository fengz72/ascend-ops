# cannbot-dsl 动态 Shape 支持需求

| 项 | 内容 |
|---|---|
| 文档类型 | 需求文档（提交上游 cannbot-dsl） |
| 提出方 | MTGR Ragged Segment Attention 算子（A5 移植，`mtgr_ragged_segment_attention`） |
| 验证环境 | Ascend950PR / CANN 9.3.0 / cannbot-dsl **0.8.3** / torch 2.12.0+cpu / torch_npu 2.12.0 |
| 撰写日期 | 2026-11-24 |
| 结论摘要 | DSL 侧 `Dim` 原语已可用且无性能损失，但尚未落到交付产物——每次 `batch` / 序列长度变化仍需重新编译；诉求为五个 ragged 维、`kv_group`、`core_num` 均成为运行时参数（R1–R3） |

---

## 1. 背景与业务影响

本算子是生成式推荐推理中的自定义掩码 Flash Attention，输入是 ragged 变长序列：`total_q`（全部请求的实时 token 总数）、`batch`、`num_blocks`（分页 KV 缓存物理块数）、`max_blocks`（页表容量）、`max_segs`（分段数）在每次请求间都会变化。

A3 基线实现（TileLang）用 `T.symbolic` 把这五个维度全部声明为动态：

```python
batch      = T.symbolic("batch")
total_q    = T.symbolic("total_q")
num_blocks = T.symbolic("num_blocks")
max_blocks = T.symbolic("max_blocks")
max_segs   = T.symbolic("max_segs")
```

当前 cannbot-dsl 版本做不到等价能力，算子只能按具体形状静态编译，代价是：

- 每一种 `(total_q, num_blocks, max_blocks, batch, max_segs)` 组合触发一次独立 JIT 编译，实测单次 **25~27 秒**；
- 编译缓存 `@lru_cache(maxsize=64)` 只保留 64 个形状变体，线上变长序列场景会持续淘汰—重编，产生秒级抖动；
- 交付验收跑 468 条配置时，绝大部分墙钟时间花在重复编译而非计算。

---

## 2. 现状：已实测可用的部分

**请上游注意：`Dim` 原语本身在 JIT 路径已经可用，本需求不是"请实现动态 shape"，而是"请让动态 shape 能落到交付形态与本算子的编译期维度上"。**

`cannbotdsl.Dim`（`cannbotdsl/aot/spec.py:100`，已从包顶层导出）支持 `min` / `max` / `multiple_of` 约束与 `+ - * //` 表达式。我们把 A3 的五个 symbolic 维全部替换为 `Dim`，**只编译一次**，同一 executable 跑通 7 种形状，结果全部对齐 golden：

| case | batch | segs | rules | total_q | num_blocks | max_blocks | 结果 |
|---:|---:|---:|:--|---:|---:|---:|:--|
| 0 | 1 | 4 | `[0,1,2,2]` | 3008 | 1 | 24 | PASS |
| 1 | 2 | 4 | `[0,1,2,2]` | 6040 | 1 | 24 | PASS |
| 3 | 8 | 4 | `[0,1,2,2]` | 32104 | 1 | 38 | PASS |
| 14 | 1 | 19 | `[0,1,2×17]` | 2888 | 1 | 23 | PASS |
| 26 | 1 | 1027 | `[0,1,2×1025]` | 7928 | 1 | 62 | PASS |
| 96 | 8 | 7 | `[0,1,2×5]` | 31651 | 1 | 42 | PASS |
| 464 | 1 | 2 | `[1,2]` | 308 | 1 | 3 | PASS |

动态化**不损失 kernel 性能**（warmup 5 + 采 5 次，`torch_npu.profiler` 取主 kernel 设备时长）：

| case | 静态编译 | 动态编译 | 差异 |
|---:|---:|---:|---:|
| 0 | 68.34 µs | 68.37 µs | +0.04% |
| 3 | 755.73 µs | 755.11 µs | −0.08% |

均在测量噪声内。

---

## 3. 需求条目

### R1（P0）五个 ragged 维作为运行时参数

**现状**：每一次 `batch`、序列长度（`total_q`）变化，kernel 都需要重新编译——单次编译实测 25~27 秒，编译缓存只保留 64 个形状变体，变长序列场景下持续淘汰—重编，产生秒级抖动（同第 1 节）。第 2 节实测的 `Dim` 动态能力目前只覆盖 JIT 路径，尚未落到交付产物。

**诉求**：`batch` / `total_q` / `num_blocks` / `max_blocks` / `max_segs` 五个维（A3 基线用 `T.symbolic` 声明；对应 cannbot-dsl `TensorSpec` 的 `batch` / `total` / `cache_blocks` / `table_cols` / `segments`）作为运行时参数传入：一次编译，产物服务所有形状，输入形状变化不再重新编译。

**示例**: `transpose_bmm_bias_relu.py`
```python
@lru_cache(maxsize=_PROGRAM_CACHE_SIZE)
def _get_dynamic_program(dtype: torch.dtype, transpose_b: bool) -> "ProviderCallable":
    """Compile one shape-dynamic program per dtype and B layout."""
    t = TransposeBmmBiasReluTiling(dtype, transpose_b)
    # A is a strided metadata view of the original contiguous [M, batch, K].
    m = Dim('M', min=1)
    batch = Dim('BATCH', min=1)
    k = Dim('K', min=1)
    n = Dim('N', min=1)
    a_spec = TensorSpec((batch, m, k), t.dtype, stride=(k, batch * k, 1))
    b_spec = TensorSpec((batch, n, k) if transpose_b else (batch, k, n), t.dtype)
    c_spec = TensorSpec(
        (
            _ceil_align(m, t.base_m),
            batch * _ceil_align(n, t.base_n),
        ),
        t.dtype,
    )
    bias_spec = TensorSpec((batch, n), t.dtype)
    return compile_program(TransposeBmmBiasReluKernel(t).run, a_spec, b_spec, c_spec, bias_spec)
```

**验收标准**：一次编译后，同一份产物在第 2 节表格的 7 种形状下均正确执行（至少覆盖 `total_q` = 308 / 3008 / 32104），无需重新编译。

---

### R2（P1）`kv_group` / `group` 支持动态

**现状**：`group` 直接参与 Python 期的 spec 构造，无法动态：

```python
kv_specification = TensorSpec((total, QUERY_HEAD_COUNT // group, head_dimension), bfloat16)
cache_specification = TensorSpec((cache_blocks, CACHE_BLOCK_TOKENS, QUERY_HEAD_COUNT // group, head_dimension), bfloat16)
```

GQA 的 `kv_group ∈ {1,2,4,8}` 是业务必需配置，当前每个取值一份编译产物。

**期望**：支持 `Dim` 表达式作为张量维度（如 `QUERY_HEAD_COUNT // group`，其中 `group` 为动态维），或提供等价的运行时 kv_heads 方案。

**验收标准**：`kv_group` 取 1/2/4/8 时复用同一份编译产物，输出与各自静态编译版本逐位一致。

---

### R3（P1）metadata 按固定上界长度分配，`core_num` 成为纯运行期参数

**场景**：同一份编译产物需要在不同 kernel（不同核数配置）下运行。交付形态是预编译原生库 + OPP 包，编译期无法预知目标部署的核数，`core_num` 不能写死在产物里。

**现状**：数据通路其实已经全部运行期化——metadata 生成 kernel 用运行期 `args.cores` 做槽位排布 stride，并把实际核数写入头部 `HEADER_CORE_COUNT`（`mtgr_ragged_segment_attention_metadata.py:278,377`）；主 kernel 运行期校验 `HEADER_CORE_COUNT == get_block_num()`，并按 `get_block_num()` 计算自己的槽位偏移（`mtgr_ragged_segment_attention.py:1861,1886`）。

唯一的编译期耦合是 metadata 张量的**长度**：spec 构造公式把 `cores` 当常量，

```
METADATA_HEADER_WORDS + batch * (QUERY_HEAD_COUNT // group) * cores * CORE_METADATA_WORDS
```

（`mtgr_ragged_segment_attention.py:2551-2553`，metadata 生成侧 `:524` 同式），host 侧 `validate_metadata` 又要求长度与公式值**精确相等**（`:2406-2412`），且 `cores` 参与编译缓存 key（`:2518`）。因此核数每变一次都要重构 spec 并重编（单次 25~27 秒）。"按上界核数分配缓冲、按实际核数生成内容"的 workaround 语义上成立（内容按实际核数 stride 连续排布，尾部空置即可），但被两端 spec 与 host 校验的精确形状匹配堵死。另外实测平台 `cube_core_num=32 / vector_core_num=64`，而算子接口声明支持 `core_num` 1~32，两者能力不匹配。

**期望**：metadata 张量长度与运行期核数解耦——按上界核数分配**固定长度**（例如上界取 36 核，即 `METADATA_HEADER_WORDS + batch * kv_heads * 36 * CORE_METADATA_WORDS`，覆盖接口声明的 1~32 并留余量），实际核数只体现在运行期的布局 stride 与头部字段上（现有代码已天然支持）；host 侧长度校验相应放宽为对固定上界长度的校验。（用带 `max` 上界的 `Dim` 表达式描述该长度是等价实现，二者取一即可。）

**验收标准**：按固定上界长度编译/导出一次，同一份产物在 `core_num = 8 / 16 / 28 / 32` 下均正确执行，metadata 每次按实际核数生成，无需重编或重构 spec。

---

## 4. 优先级汇总

| 编号 | 需求 | 优先级 | 阻塞点 |
|---|---|:--|---|
| R1 | 五个 ragged 维作为运行时参数 | P0 | 每次 batch / 序列长度变化都需重新编译（25~27 s/次） |
| R2 | `kv_group` 动态 | P1 | GQA 1/2/4/8 组合放大产物数 |
| R3 | metadata 固定上界长度，`core_num` 纯运行期 | P1 | 同一产物需在不同核数 kernel 下运行 |

---

## 5. 复现方式

以下结论均可在 Ascend950PR / CANN 9.3.0 / cannbot-dsl 0.8.3 上复现。算子源码与测试用例来自 A5 交付包（`golden.py`、`testcase.py`、`tilelang_precision_checker.py` 未作修改）。

1. **动态维可用性**（第 2 节表格）：用 `Dim` 替换 `build_mtgr_ragged_segment_attention_executable` 中 `total` / `cache_blocks` / `table_cols` / `batch` / `segments` 对应的 `TensorSpec` 维度，`dsl.compile` 一次，对 case 0/1/3/14/26/96/464 分别生成 metadata 并执行，与 `golden_attention_simulated_kernel` 比对。
2. **性能无损失**（第 2 节表格）：同一 case 分别用静态与动态 spec 编译，各 warmup 5 次后采 5 次，`torch_npu.profiler` 过滤 `kernel_MtgrRaggedSegmentAttention` 且 `Task Type` 以 `KERNEL` 开头的事件取 `dur`。
3. **R2/R3 编译期约束**：见算子源码 `mtgr_ragged_segment_attention.py:2540-2545`（group 参与 spec 构造）、`:2551-2553`（cores 参与 metadata 长度）、`:2406-2412`（host 精确长度校验）、`:2518`（cores 参与编译缓存 key）；运行期通路已按实际核数计算，见 `:1861,1886`（主 kernel `get_block_num()` 校验与寻址）、`mtgr_ragged_segment_attention_metadata.py:278,377`（生成侧 stride 与头部核数）。

> 说明：第 2 节数据与 R1 现状（每形状重编、单次 25~27 秒）均为本机实测。
