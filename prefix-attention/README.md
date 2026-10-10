# PrefixInferAttentionScore / prefix_attn — prefix-in-Q attention 算子

同一 prefix-in-Q prefill attention 语义(去重 TND:共享 prefix 的 Q/KV/输出只存
一份、只算一份;各 batch 私有段 attend prefix KV + 自身 KV,causal;GQA
N1%N2==0,D=64,fp16/bf16)的两种实现,按目录分开维护:

| 目录 | 技术路线 | 目标环境 | 入口 |
|---|---|---|---|
| [`ascendc/`](ascendc/README.md) | AscendC C++ 自定义算子,vendored FIA 模板链(源仓 ops-transformer `origin/9.0.0` @ e88633357),aclnn/torch_binding 交付 | Ascend910_93,CANN 9.0.0,Python 3.11 | `ascendc/build.sh`;测试 `ascendc/tests/` |
| [`cannbot-dsl/`](cannbot-dsl/README.md) | CANNBotDSL(Python DSL)DN(K@Q^T)变体 kernel + AICPU metadata planner,含 codegen 出的 AscendC 代码快照 | Ascend950PR(32 AIC),CANN 9.2.0-beta.2,cannbotdsl 0.7.0,torch_npu 2.10 | `import prefix_attn`;测试 `cannbot-dsl/tests/` |

两者接口形态不同:ascendc 版走 aclnn 算子签名(cumsum 形式
`actualSeqLengths`);cannbot-dsl 版是 python 函数
`prefix_attn_dn(q, k, v, actual_seq_lengths, scale=None)`,`actual_seq_lengths`
为 `[B+1]` 段边界(`b[0]=p`,末项 `T`)。语义细节各见其 README。

性能对比(DN vs `npu_fused_infer_attention_score`,Ascend950PR,口径与数据见
源仓 `prefix_attn/perf/dn_vs_fia_20261009/`):DN 17.1–17.8us(min),FIA 压缩
causal(sparse_mode=2/3)17.3–18.2us,FIA 完整 mask(sparse_mode=0)20.6–21.5us。
