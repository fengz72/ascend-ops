# PrefixAttnDn — GE 自定义算子(custom_op_v2,形态1)

把 cannbot-dsl DN 实现接入 GE 图:单个算子节点 `PrefixAttnDn`,其
`Execute()` 内部完成完整 host 编排(零初始化 metadata workspace → launch
AICPU metadata planner → launch DN macro attention kernel),两次 launch 同
stream 串行,图上不可见第二个节点。kernel 为预编译的 cannbotdsl AOT 产物
(dlopen 加载),GE 侧零 RTC。

参考:CANN GE `custom_op_v2` 自定义算子开发指南(场景 A:动态图在线执行)。

## 算子原型

```cpp
REG_OP(PrefixAttnDn)
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
| `include/prefix_attn_dn_op.h` | REG_OP proto(构图侧/ES 生成共用) |
| `src/prefix_attn_dn_proto.cc` | InferShape/InferDataType + IMPL_OP 注册 |
| `src/prefix_attn_dn_op.cpp` | `EagerExecuteOp::Execute` + `REG_AUTO_MAPPING_OP` |
| `src/dsl_launcher.{h,cpp}` | schema 驱动的 ABI v9 frame 构造 + dlopen 双产物 launch |
| `src/artifacts.h` | 生成:产物文件名/符号名/核数默认值(export 脚本产出) |
| `scripts/export_artifacts.py` | 板上导出 DN fp16/bf16 AOT .so + AICPU .so + MANIFEST |
| `probe/` | P0 ABI 探针:C++ 独立 launch 与 python host 路径**逐位一致**(已 PASS) |
| `opp/` | OPP 交付包(`ASCEND_CUSTOM_OPP_PATH` 指向此) |
| `tests/` | GE 在线 Session 端到端(ES wrapper + GraphBuilder + golden 比对) |

## 关键机制:host ABI v9 frame

DN 产物 .so 自描述(`codegen/launch_prefix_macro_dn.asc` 尾部同款):

- `<base>__cannbot_schema_v2()` → `CannBotHostSchemaV2*`:frame_bytes/alignment、
  contract_digest、每 slot 的 offset/pointer_offset/shape_offset/stride_offset
- `<base>__cannbot_invoke_v9(stream, frame, bytes)` → 严格校验 header
  (magic `0x43424639`/abi=9/digest)后 launch

`dsl_launcher.cpp` 完全按 schema 填 frame(不硬编码布局),8 个 slot 依次为
out/query/key/value/scale/bounds/metadata/block_dim。AICPU 产物走扁平 C ABI
`_prefix_attn_metadata_kernel_do(stream, bounds, metadata, len, n1, n2, aic, dn_cost)`。
核数 = min(导出时设备 AIC 数, 36),`PREFIX_ATTN_AIC_CORES` 可收窄,metadata
参数与 DN block_dim 同源保证一致。

## 构建与验证

```bash
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh   # ASCEND_HOME_PATH

# 0) 导出 kernel 产物(需 NPU + cannbotdsl 0.7.0;产物 .so 不入库,MANIFEST 入库)
python scripts/export_artifacts.py

# 1) P0 探针:C++ 独立 launch vs python host 路径,逐位比对
bash probe/run_probe.sh          # 期望 PROBE_PASS (bit-exact)

# 2) 构建 + OPP 安装 + ES wrapper 生成 + GE 在线 Session 端到端
bash tests/run_online.sh         # 期望 GE_SESSION_PASS
```

msprof 确认图内每次 Execute 产生两个 task:
`AI_CPU _prefix_attn_metadata_kernel` + `MIX_AIC kernel__PrefixMacroKernelDn`(Block=32)。

## 部署

```bash
export ASCEND_CUSTOM_OPP_PATH="<...>/ge_custom_op/opp:$ASCEND_CUSTOM_OPP_PATH"
```

- 交付件:`opp/op_graph/lib/linux/aarch64/libcust_opapi.so`(构建产物,不入库)
  与同目录 `artifacts/`(export 产物,不入库;MANIFEST.json 入库供校验)
- 产物为板级绑定(dav-3510 / Ascend950PR,32 AIC);换板重跑 export 即可
- 注意:`ge.Tensor.get_data()` 走文本序列化,fp16 比对有 ~1e-7 噪声;设备侧
  输出与 python 路径逐位一致(见 probe)

## 状态与后续

- [x] P0 ABI 探针(bit-exact PASS)
- [x] P1 GE 在线 Session 端到端(PASS,msprof 双 kernel 确认)
- [ ] P2 PyTorch/TorchAir 前端(TORCH_LIBRARY + converter;需安装 torchair)
- [ ] P3 OM 离线下沉(CompilableOp/PortableOp,参考 compilable_add_custom 样例)
