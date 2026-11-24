# AGENTS.md

本仓库是 Ascend 自定义算子源码集合（vendored / 自建），供上层工程经构建脚本产出
custom-opp 安装包并接入 torch。目录：`mega-gdn-prefill/`（vendored 三算子 +
GE custom-op-v2 包装）、`causal-conv1d/`（vendored fla-npu `CausalConv1d`，
dense prefill）、`causal-conv1d-qkv/`（vendored xllm `CausalConv1dQkv`，
varlen packed-QKV 融合）、`mtgr_attention/`、`prefix-attention/`。

同名算子类型跨谱系冲突：`CausalConv1d` 同时存在于 `causal-conv1d/`（fla 谱系，
`load_priority` 首位）与 `mega-gdn-prefill/`（xllm 旧谱系）；`causal-conv1d-qkv/`
因此只注册 `CausalConv1dQkv`，其 vendored `causal_conv1d` 依赖仅作源码/头文件
使用（tiling 定义经 shim TU 编译，屏蔽 `IMPL_OP_OPTILING(CausalConv1d)`）。
新增算子包沿用此原则：不要往 opp 里再塞同名算子类型。

接入 GE 的完整规范见 [docs/ge-integration-spec.md](docs/ge-integration-spec.md)
（包结构与交付件、GE 接入路线选择、符号面与构建卫生、torch 接入、装载与部署
契约、共存与冲突、测试门禁、版本矩阵）：新增包与整改以该规范为准，本文件
仅保留常用约定。`mtgr_attention/` 为 TileLang DSL 实现（非 custom_op_v2
交付件），规范 §2–§7 对其不适用。

## torch 接入约定（必须遵守）

自定义算子接入 torch 时，**统一挂载到 `torch.ops.ascend_ops` 命名空间**，
调用形式为 `torch.ops.ascend_ops.<op_name>`，例如
`torch.ops.ascend_ops.mega_gdn_prefill`。

- 通过 `torch.library.Library("ascend_ops", "FRAGMENT")` 获取命名空间
  （FRAGMENT 模式可被多个算子模块重复使用，不得用 DEFANDIMPL 独占）。
- 注册内容按现有模式补齐三件套：
  1. `lib.define(SCHEMA)`：schema 字符串 `<op_name>(Tensor x, ...) -> Tensor`；
  2. `lib.impl(<op_name>, <meta_impl>, "Meta")`：返回 `torch.empty_like(...)`
     之类的形状/类型推导，供 dynamo 导出与图编译；
  3. `lib.impl(<op_name>, <eager_impl>, "PrivateUse1")`：NPU eager 实现
     （JIT 桥经 flat ABI 直调 `libcust_opapi.so`；绑定随交付必备，
     缺失即安装/加载失败，fail-loud）。
- 图模式经 torchair converter 接入：
  `torchair.register_fx_node_ge_converter(torch.ops.ascend_ops.<op_name>.default)(_convert)`，
  `_convert` 内用 `torchair.ge.custom_op("<GE op 名>", inputs={...}, outputs=[...])`
  映射到 REG_OP 注册的 GE 节点。
- **坑位**：`torch.library.Library` 对象必须模块级存活（赋给模块级全局变量），
  函数局部变量会被 GC 注销 schema，进而在 dynamo 内引发 ATen 断言/段错误。
- 参照实现：`mega-gdn-prefill/python/mega_gdn_prefill.py`（activator：dlopen
  `libcust_opapi.so` + 注册 torch op + torchair converter，import 即生效）。

## 构建与测试

- 构建并安装（vendored 包，先 `source CANN set_env.sh`）：
  ```bash
  bash mega-gdn-prefill/install.sh   # 一条命令：构建+装入 $ASCEND_HOME_PATH/opp+pip 注册 torch 绑定
  SOC_VERSION=ascend910b bash mega-gdn-prefill/install.sh    # SoC 覆盖
  OPS=mega_gdn_prefill_op bash mega-gdn-prefill/install.sh  # 算子覆盖
  # 开发机备选：pip install -e mega-gdn-prefill/python --no-deps（源码树 editable，
  # 改 python/ 即时生效；与 vendor 安装二选一，后装者生效）
  ```
- 测试（需 CANN 环境 + `bash mega-gdn-prefill/install.sh`（含绑定注册）；
  activator 自发现 `.so` 并自挂 opp，无需手工导出）：
  ```bash
  python3 mega-gdn-prefill/test/test_mega_gdn_prefill.py [device]
  ```
  门禁：graph vs eager `rel_l2 < 0.001`，eager vs CPU golden `cos > 0.95`。
- 其余 vendored 包同构（各自 `install.sh` + `test/`）：
  ```bash
  bash causal-conv1d/install.sh      && python3 causal-conv1d/test/test_causal_conv1d.py
  bash causal-conv1d-qkv/install.sh  && python3 causal-conv1d-qkv/test/test_causal_conv1d_qkv.py
  ```
  `causal-conv1d-qkv` 额外门禁：`conv_states` 与 golden 逐位相等；图模式仅
  `dynamic=True` 可用（详见其 README 已知限制）。

## 其他约定

- 新增注释尽量简洁：讲清楚原因（why）即可，不写过程记录（how / 验证日志 /
  A/B 对照）。
- 算子代码零修改；构建机器文件（cmake/sh/build 脚本）允许修改，改动记入
  README 改动清单，构建产物不落进 `cmake/`/`common/`/`scripts/` 源码树
  （全部进 `build/` 与 `.cache/`）。详见 `mega-gdn-prefill/README.md`。
- GE 包装算子（如 `mega_gdn_prefill_ge.cpp`）是项目自建非算子代码，经
  `cmake/custom_build.cmake` 的 EXISTS 守卫钩子编进 `cust_opapi` 目标，与
  aclnn 实现产出单一 `libcust_opapi.so`。
- 提交信息格式参照 `git log`：`type(scope): subject`（如
  `refactor(mega-gdn-prefill): ...`）。
