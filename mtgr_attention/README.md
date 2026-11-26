# MTGR Attention（TileLang-Ascend DSL 交付）

本目录是 TileLang-Ascend DSL 实现与 A5 需求/交付文档（`tilelang-ascend/`、
`A5需求文档.md`），不是 CANN custom_op_v2 交付件：不产出 `libcust_opapi.so`，
没有 vendor 包、GE 包装或 torch 注册，当前无在用消费方。

因此算子包符号收敛与装载规范（hidden/导出白名单、`RTLD_LOCAL`）不适用于本包。
