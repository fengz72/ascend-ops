# causal-conv1d-qkv — xllm_ops CausalConv1dQkv vendored + GE custom-op 入图

vendored [xLLM-AI/xllm_ops](https://gitcode.com/xLLM-AI/xllm_ops)（main @
`ecc8a32`）的 `xllm_ops/causal_conv1d_qkv` 算子（AscendC，6 文件薄壳）与其硬
依赖 `xllm_ops/causal_conv1d`（21 文件，提供 tiling/planner/validation 与
kernel 实现），外加项目自建的 GE custom-op-v2 图模式包装
（`CausalConv1dQkvGE`）、eager flat 入口与 torch 绑定。算子源码与上游逐字节
一致；`cmake/`、`common/`、`scripts/`、`src/CMakeLists.txt`、`src/build.sh`
为 xllm_ops 构建机制快照（取自本仓库 `mega-gdn-prefill/` 已适配版本，含本包
改动，见文末改动清单）。许可见包内 `LICENSE`/`LICENSES/`：算子源码 CANN OSL
v2.0（Huawei），上游仓库根 Apache-2.0（JD.com），qkv 六文件上游未带license头。

与同仓库 `causal-conv1d/`（fla-npu 谱系 `CausalConv1d`，dense prefill）是
**不同谱系、不同算子类型**：本包是 xllm 谱系的 varlen packed-QKV 融合算子，
只注册 `CausalConv1dQkv`，不引入第二个 `CausalConv1d`（见下）。

## 算子语义

一次调用完成 GDN 前处理链：varlen causal depthwise conv（width=4，历史取自
conv 状态缓存）→ silu → 按 `[q|k|v]` 切分 → q/k 逐 head（head_dim=128）L2
归一化（fp32 累加，eps=1e-6，经 bf16 中转）→ v 直通 → 原布局拼回。

```
y[t, :] = packed_qkv(silu(sum_j weight[j] * x[t-j]))   # 每序列左接 conv_states 历史
```

- `x`：**2D varlen** `(T, dim)`，bf16；`dim = q_dim + k_dim + v_dim`，需为 16 的倍数
- `weight`：`(4, dim)` bf16；packed 路径**不接受 bias**
- `conv_states`：`(num_slots, 3, dim)` bf16，**原地读写**（`initial_state_mode[b]!=0`
  时作为历史读入，调用后总是写回该序列末尾 3 行）
- 元数据（1D int64，host 或 NPU 皆可）：`query_start_loc` `(B+1)`（`[0]==0`、
  单调不减、`[B]==T`）、`cache_indices` `(B)`、`initial_state_mode` `(B)`、
  `num_accepted_tokens` 可选
- `y`：与 `x` 同 shape，dtype 由 `out_dtype` 选 bf16（默认）或 fp16（输入恒 bf16，
  对应 OpDef 的两条 dtype 配置，各自产出 kernel 二进制）
- 属性默认：`activation_mode=2`（SILU_PACKED_QKV）、`pad_slot_id=-1`、`run_mode=0`、
  `q_dim=1024`、`k_dim=1024`、`v_dim=3072`、`head_dim=128`
- 上游 host tiling 硬门禁（不满足即 561002）：`run_mode=0`、2D varlen、width=4、
  无 bias、`q_dim==k_dim`、`head_dim==128`、q/k/v 各为 128 的倍数且和等于 `dim`
- UPDATE/cache 解码路径（`run_mode=1`）被上游 validation 拒绝，不随本包暴露

## 构建 + 安装

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash install.sh          # 构建 .run + 装入 $ASCEND_HOME_PATH/opp + pip 注册 torch 绑定
SOC_VERSION=ascend910b bash install.sh        # SoC 覆盖（默认 ascend910_93）
OPS=causal_conv1d_qkv bash install.sh         # 算子覆盖
SKIP_TORCH_BINDING=1 bash install.sh          # 只装算子，不注册绑定
```

- vendor 目录 `$ASCEND_OPP_PATH/vendors/causal_conv1d_qkv/`；图模式 kernel 编译
  走标准 tbe-custom store，全局 opp 必须有本包
- **只注册 `CausalConv1dQkv`**：宿主已装 fla 谱系 `CausalConv1d`（vendor
  `causal_conv1d`，`load_priority` 首位，aclnn 签名与本谱系完全不同），若本包
  再带一份 xllm 谱系 `CausalConv1d` 的 proto/tiling，activator 前置挂载
  `ASCEND_CUSTOM_OPP_PATH` 时可能劫持该算子。做法：`cmake/custom_build.cmake`
  把 `causal_conv1d` 从 `add_subdirectory` 列表剔除（其 `op_host` 不参与构建），
  共享 tiling 定义经项目自建 `src/causal_conv1d_qkv_tiling_shim.cpp` 编译——
  该 TU 把 `IMPL_OP_OPTILING(CausalConv1d)` 宏替换为空注册后再 include 上游
  未修改的 `causal_conv1d_tiling.cpp`，只取 `CausalConv1dTilingFunc` /
  `TilingParseForCausalConv1d` 定义
- `install.sh` 对 `vendors/config.ini` 的 `load_priority` **只追加、不重排**
  （`causal_conv1d` 必须原位不动）
- torch 绑定随 vendor 包 `op_api/python/` 交付（pip editable 注册，包名
  `causal-conv1d-qkv-torch`），import 即生效，零 shell 导出

## torch 接入

```python
import torch, torch_npu
import causal_conv1d_qkv        # dlopen + 挂 opp + 注册三件套 + torchair converter

y = torch.ops.ascend_ops.causal_conv1d_qkv(
    x, weight, conv_states,               # conv_states 原地更新
    q_dim, k_dim, v_dim,                  # 例：2048, 2048, 6144
    query_start_loc=qsl,                  # 1D int64，[0]==0、[-1]==T
    cache_indices=cache_indices,          # 1D int64
    initial_state_mode=initial_state_mode,# 1D int64，逐序列 0/1
    out_dtype=torch.float16)              # 可选；None => 与 x 同 dtype(bf16)
```

schema（标量刻意排在可选 Tensor 之前，见下）：

```
causal_conv1d_qkv(Tensor x, Tensor weight, Tensor(a!) conv_states,
                  int q_dim, int k_dim, int v_dim, int head_dim=128,
                  int activation_mode=2, int pad_slot_id=-1, int run_mode=0,
                  ScalarType? out_dtype=None, Tensor? query_start_loc=None,
                  Tensor? cache_indices=None, Tensor? initial_state_mode=None,
                  Tensor? num_accepted_tokens=None) -> Tensor
```

- eager：`PrivateUse1` impl → JIT pybind 桥 → `libcust_opapi.so` 内 flat 入口
  `causal_conv1d_qkv_run` → `aclnnCausalConv1dQkvGetWorkspaceSize`（元数据走
  `aclIntArray` 变体：host tiling 需要这些值，`*Tensor` 变体不做 D2H，行为依赖
  nnopbase 版本）。设备侧元数据由入口做流序 D2H + `aclrtSynchronizeStream`
- 图模式：converter 映射到 `torchair.ge.custom_op("CausalConv1dQkvGE")`；GE 侧
  `REG_OP(CausalConv1dQkvGE)` + `IMPL_OP(...).OpExecuteFunc(&CausalConv1dQkvExecute)
  .HostInputs({3,4,5,6})`（aclnn lowering 路由，kernel/tiling 复用 vendored 算子）。
  `HostInputs` 让 GE 把四个 `.ValueDepend` int64 元数据输入在 **HOST** 上生成
  （`aclnn_node_converter.cc: IsHostInput(ir_index) ? kOnHost : kOnDeviceHbm`），
  框架在每次图执行时插入 3 个 `CopyD2H`（per-run 图，不在 Init 图），`Execute`
  直接读 host 值，包装内不再做 D2H/drain；输出先经 `InferShape`/`InferDataType`
  定型再由框架分配，workspace 走 `ctx->MallocWorkspace/FreeWorkspace`。
  `out_dtype` 必须传（GE attr 0=bf16、1=fp16），漏传恒得 bf16
- **图模式依赖 ops-info**：aclnn lowering 从 per-SoC `aic-<soc>-ops-info.json`
  解析 engine；生成器只认 vendored OpDef，故构建期由
  `cmake/inject_causal_conv1d_qkv_ge_info.py` 注入 `CausalConv1dQkvGE` 条目
  （必须含 `aclnnSupport=support_aclnn`，否则整条被忽略；dtype/format 需按输出
  dtype 配置逐项重复）。`install.sh` 对安装后的 JSON 断言该条目，缺失即失败——
  否则只有图模式编译报 EZ3003，eager 仍正常，容易漏到模型侧
- **图模式必须 `dynamic=True`**：`dynamic=False` 时 torchair 把全部输入搬进 GE
  自有静态缓冲，1D int64 元数据缓冲到达时是全零，host tiling 直接报
  `queryStartLoc[last] must equal cuSeqlen`（eager 不受影响）。与
  `causal-conv1d/` 的口径一致
- `conv_states` 的原地更新在图模式成立：GE 优化后该节点吃 `TensorMove` 副本，
  副本经 `Identity` 挂到 `NetOutput`，torch 的 AOT mutation epilogue 写回调用方
  张量（代价：一次全尺寸 device copy + 一个额外图输出）
- **坑位**：`torch.library.Library` 句柄必须模块级存活；torch_npu 先于 torchair
  import；标量排在可选 `Tensor?` 之前是规避 torch 2.12 boxed 调用丢标量
  （见 `causal-conv1d/README.md`）；`REG_OP` 的输入顺序被 converter 与构建期
  注入的 ops-info 条目双向钉死，改一处必须同步另一处

## 精度门禁

```bash
python3 test/test_causal_conv1d_qkv.py [device]    # 前置：bash install.sh
```

门禁：graph vs eager `rel_l2 < 0.001`（实测 0.00000000，逐位一致）、eager vs
CPU fp32 golden `cos > 0.95` 且 `rel_l2 < 0.02`（实测 cos 0.999080–1.000051、
rel_l2 ≤ 0.000011）、**`conv_states` 与 golden 逐位相等**（`torch.equal`，
eager 与 graph 两条腿各自独立 clone 校验）。

覆盖上游全部 6 组 shape（tp1 16/48 heads len 64、tp2 ragged (127,129) 混合
`initial_state_mode`、tp4 len 256、tp8 ragged (63,65)、tp16 len 129、
qwen3.5-2b prefill 16/16 len 22）× 输出 bf16/fp16，共 12 例；`q/k/v_dim`
逐例不同（dim ∈ {640,1280,2560,5120,6144,10240}），可证非标量默认值确实下到
kernel。测试同时断言每例都新追踪了图（`dynamo` frame 计数增长），防止静默
回落 eager 让图门禁空转。图编译用默认 `CompilerConfig()`，与 bench/部署同一路由。

主扫描之外另有四组腿：

- **host 元数据 eager**——activator/README/bench 都推荐这条零拷贝路线，此前
  只有计时无正确性覆盖
- **中间省略回归**——`cache_indices` 缺省而 `initial_state_mode` 在场，钉住
  aclnn 路由下"取输入按 IR 索引、缺位槽不前移邻居"的契约
- **两步解码链**——第一步写回的历史被第二步以 `initial_state_mode=1` 读回，
  eager 与 graph 均与链式 golden 逐位一致（原地状态契约的正面验证）
- **6 条负例**——`run_mode=1`、fp16 的 x、3-D x、`q_dim!=k_dim`、维度非
  `head_dim` 整数倍、`run_mode=0` 带 `num_accepted_tokens`；只断言"必须响亮
  失败"，不绑定错误码/文案（校验由上游 tiling 拥有，包装层不再镜像
  `query_start_loc` 的取值检查）

## 性能（startlux-decision-v1.1-0.8b-instant 真实 shape）

```bash
npu-smi info                                              # 挑 AICore 0%、无进程的卡
python3 test/bench_causal_conv1d_qkv.py --device <chip>   # 结果写 test/bench_result.json
```

模型侧几何：18 个 GDN 层，`conv_dim=6144 = q 2048 + k 2048 + v 2048`，
`head_dim=128`、width=4、无 bias——满足本算子全部门禁，`q_dim==k_dim` 由模型
结构天然成立。口径与仓库既有 `work/fla-npu-bench/bench_gdn_ops.py` 一致
（warmup 32 + 200 次，每次调用一对 npu.Event），Ascend910_9382 / CANN 9.1.0 /
chip 3（`npu-smi` 确认空闲，宿主 loadavg 56–74）。"现链"= 模型当前每层子链
（`causal_conv1d` + 3×slice/reshape + 2×`npu_rms_norm` + flatten，见
`modeling_text.py:142-155`）；"替换链"= 本算子 + 输出侧同样的
3×slice/reshape/flatten（两个 RmsNorm 被折进算子，gamma 一并折入）。
均值 µs/调用：

| case | B,S | fused eager(NPU 元数据) | fused eager(host 元数据) | conv-only eager | 现链 eager | 替换链 eager | 现链 graph | 替换链 graph | graph 启动地板 |
|---|---|---|---|---|---|---|---|---|---|
| team | 1,99 | 151.9 | 75.8 | 83.5 | 315.2 | 265.7 | 465.5 | 451.7 | 242.4 |
| severity | 1,99 | 156.7 | 85.3 | 71.8 | 307.0 | 277.2 | 459.7 | 452.0 | 220.5 |
| urgent | 1,85 | 148.0 | 81.1 | 69.5 | 286.7 | 242.6 | 454.6 | 450.9 | 229.5 |
| b3 | 3,99 | 150.7 | 86.3 | 72.5 | 280.9 | 247.5 | 459.6 | 435.1 | 212.8 |
| image | 1,130 | 133.3 | 76.5 | 70.8 | 284.6 | 224.7 | 477.4 | 440.3 | 219.7 |
| long-seq | 1,1001 | 166.1 | 98.4 | 90.0 | 276.3 | 233.1 | 456.5 | 430.0 | 221.4 |

- **算子本体很便宜**：eager 下 fused（host 元数据）75.8–98.4 µs vs conv-only
  69.5–90.0 µs，即 q/k 逐 head L2 + `[q|k|v]` 重组只多 6–8 µs；而它替掉的
  2× L2 RmsNorm 在模型侧公布值约 49 µs/层（S=1001，`performance.md:977`）
- **eager 全链净赢**：替换链 224.7–277.2 µs vs 现链 276.3–315.2 µs，每层省
  30–60 µs（18 层每请求省 0.54–1.08 ms）
- **图模式（模型实际部署路径）6 例全胜**：替换链比现链快 3.6–37.1 µs/层
  （+0.8…+7.8%，每请求 +65…+669 µs）。收益来自 `HostInputs`：四个 int64 元数据
  由 GE 在 host 上生成、框架以 per-run `CopyD2H` 搬运，`Execute` 内不再
  `aclrtSynchronizeStream`（改写前同一 harness 上图模式是**慢** 23–52 µs/层）
- **元数据驻留是 eager 侧最大可控项**：设备侧元数据需一次批量 pinned D2H +
  一次 drain，实测 66.9–67.7 µs/调用（fused 与 fused_hmeta 之差）；host 侧
  元数据零拷贝
- **口径警告**：本 harness 测的是每次调用的端到端延迟（含派发/启动，graph 臂
  含 213–242 µs 的 torchair 启动地板），不是 msprof 的 in-graph 单节点 task
  duration，故绝对值不可与 `performance.md` 的 22.9 µs/层（in-graph
  CausalConv1d）直接比；只有同 harness 内的差值可用。模型级结论需在导出 AIR
  上做 msprof A/B
- **离散度**：harness 是 host-bound 的。本次最小例复跑 eager 臂 −9.6…+0.3%、
  graph 臂 −3.9…+2.1%；宿主 loadavg 高到 ~200 时**所有**臂（含未改动的现链与
  1 节点地板图）一起抬高，那种数据要整轮丢弃重测

## 目录

```
src/causal_conv1d_qkv/     vendored 算子（6 文件；op_host/CMakeLists.txt 为构建文件，有改动）
src/causal_conv1d/         vendored 依赖算子（21 文件，逐字节一致；不参与算子注册）
src/causal_conv1d_qkv_aclnn.h   项目自建：aclnn 入口本地声明
src/causal_conv1d_qkv_eager.cc  项目自建：eager flat 入口 causal_conv1d_qkv_run
src/causal_conv1d_qkv_ge.cpp    项目自建：CausalConv1dQkvGE GE 包装（aclnn 路由 + HostInputs）
src/causal_conv1d_qkv_common.h  项目自建：两条腿共用的元数据 host 化与 packed 门禁镜像
src/causal_conv1d_qkv_tiling_shim.cpp  项目自建：共享 tiling 定义、屏蔽 CausalConv1d 注册
cmake/inject_causal_conv1d_qkv_ge_info.py  项目自建：注入 CausalConv1dQkvGE 的 ops-info 条目
python/                    torch 绑定（activator + JIT binding + bridge）
cmake/ common/ scripts/ src/CMakeLists.txt src/build.sh   xllm_ops 构建机制快照
test/                      精度门禁 + 性能 bench（bench_causal_conv1d_qkv.py / bench_result.json）
```

## 构建机制改动清单

相对 `mega-gdn-prefill/` 的 xllm_ops 机制快照（该快照相对上游已适配本仓库）：

- `src/CMakeLists.txt`：`VENDOR_NAME` → `causal_conv1d_qkv`
- `cmake/custom_build.cmake`：GE/eager 钩子改指本包三件；`op_api/python` 安装
  规则改为按文件名通配（`*.py *.toml *.cpp *.h`）；cust_opapi 补
  `-fvisibility=hidden`，并追加链接期白名单
  `cmake/cust_opapi_exports.map`（global：`causal_conv1d_qkv_run`、
  `aclnnCausalConv1dQkv*`；local：`*`），flat 入口在
  `src/causal_conv1d_qkv_eager.cc` 显式 `visibility("default")`；`op_add_depend_directory`
  后从 `OP_DEPEND_DIR_LIST` 剔除 `causal_conv1d`（只留 kernel 源/头拷贝，不建
  其 `op_host`）；依赖算子的 kernel 头安装恒追加 `arch35/`（`causal_conv1d.h`
  无条件 include 它，而 `filter_copy_files` 只挑当前 arch）；`.run` 更名
  `causal_conv1d_qkv_linux-*.run`；新增 ALL 目标
  `inject_causal_conv1d_qkv_ge_info_<soc>`（挂在 `generate_ops_info_<soc>` 后），
  把 `CausalConv1dQkvGE` 条目写进生成的 `aic-<soc>-ops-info.json` 与 `custom/`
  暂存副本；`graph ascendcl lowering` 与 GE 头目录的链接守卫改为"两个项目文件
  任一存在即挂"（`_eager.cc` 同样需要 `ascendcl`）
- 新增 `cmake/inject_causal_conv1d_qkv_ge_info.py`：ops-info 幂等后处理器；
  `--check` 供安装期断言复用
- `install.sh`：追加安装后断言——vendor 的 `aic-*-ops-info.json` 必须含
  `CausalConv1dQkvGE` 且 `aclnnSupport=support_aclnn`，否则 fail-loud；安装路径、
  断言与 `config.ini` 统一绑定 `OPP_ROOT=$ASCEND_HOME_PATH/opp`（不跟随可能被
  其它 wheel 的 `.pth` 重定向到未安装树的 `ASCEND_OPP_PATH`）；`shopt -s
  nullglob` 保证零产物时"恰好一个 installer"的断言真的失败，而不是把字面量
  交给 `bash`
- `cmake/third_party/`：补丁源目录 `build/modules/patch/` → `patches/modules/patch/`
  （`abseil-cpp.cmake`、`ascend_protobuf.cmake` 两处 `PATCH_COMMAND` 同步），
  原因是仓库根 `.gitignore` 忽略一切 `build/`
- `src/causal_conv1d_qkv/op_host/CMakeLists.txt`（vendored 构建文件，唯一改动的
  算子侧文件）：保留 `_depends`（kernel 头拷贝需要）；把 shim TU 加进
  `${OPHOST_NAME}_tiling_obj`；把 `causal_conv1d_qkv_proto.cpp` 显式加进
  `${OPHOST_NAME}_infer_obj`（`add_modules_sources` 只 glob `*_infershape*.cpp`，
  上游这个文件名会让 `IMPL_OP_INFERSHAPE` 漏编）
- `src/attention/CMakeLists.txt`：随快照带入（`custom_build.cmake` 无条件
  `add_subdirectory(attention)`）
- 新增 `install.sh`、`.gitignore`、`LICENSE`、`LICENSES/`、`version.cmake`、
  `version.info`
- 算子源码（`.cpp`/`.h`）零修改

## 已知限制

- 图模式仅 `dynamic=True` 可用（见上）；`dynamic=False` 的静态输入搬运把 int64
  元数据填零，属 torchair 侧问题，未在本包内规避
- 元数据为 CPU 张量时 eager 可用，但 `torch.compile` 追踪期报 fake tensor 错
  （PrivateUse1-only 自定义算子收到 CPU 张量）；图模式调用方须传 NPU 张量
- 元数据可按**任意位置**省略：aclnn 路由下 `OpExecuteContext` 的取输入接口是
  IR 索引的，缺省的 optional 槽返回 null 而不会把后续输入前移（实测
  `cache_indices` 缺省、`initial_state_mode` 在场时 graph 与 eager 逐位一致，
  门禁有专门回归腿）。旧 `EagerExecuteOp` 路由的 `GetInputTensor(i)` 是实例
  索引，才有"只能按后缀省略"的限制。注意 `num_accepted_tokens` 在
  `run_mode=0` 下被上游拒绝（`numAcceptedTokens is only supported in runMode=1`）
- 原地状态只在 `conv_states` 作为**图输入**时验证过；若它由更大图内部产出，
  GE 内存规划可能复用/别名该缓冲，模型级集成后需复测
- `causal_conv1d_tiling_key.h` 里的 `ASCENDC_TPL_ARGS_DECL(CausalConv1d, ...)`
  沿用兄弟算子名（上游 qkv kernel 未定义 `CAUSAL_CONV1D_SKIP_TPL_REGISTRATION`）；
  ascend910_93 实测未错联（二进制落在 `causal_conv1d_qkv/CausalConv1dQkv_*.o`），
  但换 SoC（如 ascend950）或运行期 JIT 重编时是潜在雷区
- 门禁已覆盖两步解码链（第一步写回的历史被第二步以 `initial_state_mode=1`
  读回，eager 与 graph 均与链式 golden 逐位一致）；更长的解码循环未断言
- 仅在 ascend910_93 + CANN 9.1.0 验证；OpDef 另声明 ascend910b/ascend950
- 同进程共存多份同名 `libcust_opapi.so`（本包 / `causal-conv1d` / fla wheel）：
  activator 以 `RTLD_LOCAL` dlopen，且本包按官方 cust_opapi 模板
  `-fvisibility=hidden` + 导出白名单收口，vendor 副本的 CANN 内部符号不会被
  提升到全局作用域；flat ABI 符号名仍包内唯一（`causal_conv1d_qkv_run` 只在
  本包里），activator 另以 `$ASCEND_OPP_PATH` 优先定位并校验符号 fail-loud。
  同 SONAME 的 `DT_NEEDED` 仍会解析到先加载者，**桥的 rpath 不构成保护**
- **`aclnnCausalConv1dQkvTensorGetWorkspaceSize` 不可用于设备侧元数据**：
  nnopbase 不为 `ValueDepend` 输入做 D2H，直接把 device 地址当 host 内存读，
  实测先报 tiling 561002、再复现即段错误。两条腿都只用 `aclIntArray` 变体
- eager 入口按线程缓存 aclnn workspace（只增不减；键含 device 与 stream——换流
  或换卡先 drain 再重分配，线程退出时故意不 free，因为 `aclrtFree` 非流序且
  析构时的当前设备不确定），去掉了每次调用的 `aclrtMalloc`/`aclrtFree`/drain；
  常驻内存约 `num_cache_lines*state_len*dim*2`（模型 shape 下 36 KB）
- 图模式下元数据 D2H 由框架承担：`HostInputs` 让 GE 在每次图执行前插入 3 个
  `CopyD2H`（per-run 图，不在 Init 图），包装内不再 drain；成本已含在 graph 数字内
