# AGENTS.md

本仓库是 Ascend 自定义算子源码集合（vendored / 自建），供上层工程经构建脚本产出
custom-opp 安装包并接入 torch。目录：`mega-gdn-prefill/`（vendored 三算子 +
GE custom-op-v2 包装）、`mtgr_attention/`、`prefix-attention/`。

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
  3. `lib.impl(<op_name>, <eager_impl>, "PrivateUse1")`：NPU eager 实现，
     绑定不可用时注册抛错占位（图模式导出只依赖 Meta）。
- 图模式经 torchair converter 接入：
  `torchair.register_fx_node_ge_converter(torch.ops.ascend_ops.<op_name>.default)(_convert)`，
  `_convert` 内用 `torchair.ge.custom_op("<GE op 名>", inputs={...}, outputs=[...])`
  映射到 REG_OP 注册的 GE 节点。
- **坑位**：`torch.library.Library` 对象必须模块级存活（赋给模块级全局变量），
  函数局部变量会被 GC 注销 schema，进而在 dynamo 内引发 ATen 断言/段错误。
- 参照实现：`mega-gdn-prefill/python/mega_gdn_prefill.py`（activator：dlopen
  `libcust_opapi.so` + 注册 torch op + torchair converter，import 即生效）。

## 构建与测试

- 构建（vendored 包，先 `source CANN set_env.sh`）：
  ```bash
  bash mega-gdn-prefill/build.sh          # 默认 ascend910_93、三算子
  SOC_VERSION=ascend910b bash mega-gdn-prefill/build.sh   # SoC 覆盖
  OPS=mega_gdn_prefill_op bash mega-gdn-prefill/build.sh  # 算子覆盖
  ```
- 测试（需 CANN 环境 + `mega_gdn_prefill` 可从 PYTHONPATH import）：
  ```bash
  python3 mega-gdn-prefill/test/test_mega_gdn_prefill.py [device]
  ```
  门禁：graph vs eager `rel_l2 < 0.001`，eager vs CPU golden `cos > 0.95`。

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
