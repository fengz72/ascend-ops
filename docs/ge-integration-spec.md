# ascend-ops 接入 GE 规范

本规范约束 `third_party/ascend-ops/` 下算子包接入 GE（图引擎）的构建、打包、
注册、装载、测试方式，以及消费方（Python 框架进程与 C++ GE 应用）使用这些
算子库的契约，避免接入方式偏离 Ascend 官方规范引发正确性/稳定性问题。

**条款级别**（RFC 2119 风格）：**必须**/MUST、**禁止**/MUST NOT、
**应**/SHOULD、**可**/MAY。每条标注出处：`[官方: 来源 「引文」]` 为官方文档
条款；`[仓库自定: 依据]` 为官方未覆盖区的本仓约定（依据多为实机实证或上游
先例）。

**来源口径**：官方条款取自 gitcode 源仓 **master** 分支（`cann/ge`、
`cann/asc-devkit`、`cann/docs`、`Ascend/torchair`；hiascend.com 文档中心与
gitcode 路径逐段同源），可能领先于本环境 CANN 9.1.0 随包文档；与 9.1.0 实测
冲突处以随包头文件与实测为准（见 §9）。来源清单见附录 A。

**维护约定**：新增算子包按本规范执行；修改本规范需附官方出处或实证依据；
CANN 升级后按 §9.4 复测并更新 §9.1 实测面与附录 B。

## 1. 范围与术语

**§1.1** 本规范适用于 `third_party/ascend-ops/` 下全部算子包（vendored 与
自建）及其消费方接入代码。非 custom_op_v2 交付件（如纯 TileLang DSL 实现
`mtgr_attention/`）不适用 §2–§7，见附录 B。
`[仓库自定: AGENTS.md 目录职责]`

**§1.2** **交付件（GE 侧）**：包含算子接入逻辑的单一 `.so`。官方定义：
「自定义算子以 .so 交付件形式被 GE 加载，通过 ASCEND_CUSTOM_OPP_PATH 环境
变量指定路径」「一个 .so 包含一个或多个算子的全部接入逻辑」。
`[官方: ge/custom_op_v2/custom_op_development_guide.md §2.3；custom_op_architecture.md §2.2]`

**§1.3** **vendor 包**：按官方 OPP 规范安装到 `vendors/<vendor_name>/` 的
算子包，含 `op_api/`、`op_proto/`、`op_impl/`、`framework/`（按需）、
`bin/set_env.bash`、`version.info`。
`[官方: asc-devkit basic_process.md 「部署后的目录结构」]`

**§1.4** **op_graph 交付树**：GE custom_op_v2 的 OPP 包模式布局
`<custom_opp_root>/op_graph/{include/, lib/<os>/<arch>/*.so}`。
`[官方: custom_op_development_guide.md §2.3/§7.2]`

**§1.5** **activator**：随 vendor 包交付的 Python 模块，import 即完成
「定位 .so → dlopen → 挂载 ASCEND_CUSTOM_OPP_PATH → 注册 torch op 与
torchair converter」；对应官方 torchair「插件化适配」交付件（「可以在任意
py 文件中实现上述交付件，并在模型执行前加载即可」）。
`[仓库自定: 各包 python/<op>.py；官方概念对应 torchair op_plugin_adapt_torchair.md]`

**§1.6** **flat ABI 入口**：交付件导出的 `extern "C"` 平面参数入口
（`<op>_run(...)`），供 JIT torch 桥直链，隔离 torch 头文件依赖。
`[仓库自定: 各包 python/<op>_bridge.cpp]`

**§1.7** **auto-mapping 路线（custom_op_v2）**：`REG_OP` +
`REG_AUTO_MAPPING_OP`，实现类按需继承能力接口（`EagerExecuteOp` /
`ShapeInferOp` / `CompilableOp` / `PortableOp` / `ArgsUpdater`），GE 经
`CustomOpFactory` 回调用户 `Execute`。官方定位：「新版本将控制权交给算子
开发者……适合非标准 kernel（用户全控制）」。
`[官方: 随包头 include/graph/custom_op.h；custom_op_architecture.md §2.5]`

**§1.8** **aclnn lowering 路线（传统机制）**：`REG_OP`/`OpDef` 原型 +
`IMPL_OP` 链式注册（`InferShape`/`Tiling`/`OpExecuteFunc` 等），框架托管
编译/序列化/调度，`Execute` 内直调 `aclnnXxx`。官方定位：「老版本面向标准
引擎流水线设计……两套机制在 GE 中共存」。
`[官方: 随包头 include/register/op_impl_registry.h；custom_op_architecture.md §2.5]`

**§1.9** **converter**：torchair GE 图模式下把 PyTorch FX 节点映射为 GE
算子节点的交付件；「对于自定义算子，需要先实现对应的 Converter，否则会出现
报错」。
`[官方: torchair ascend_ir/api/ge/custom_op.md]`

## 2. 包结构与交付件

**§2.1** vendor 包目录**必须**按官方布局交付：
`op_api/{include, lib/libcust_opapi.so}`、`op_proto/{inc,lib}`、
`op_impl/ai_core/tbe/{config,<vendor>_impl,kernel/<soc>,op_tiling/lib/liboptiling.so}`、
`scripts/`、`version.info`；框架适配场景另加 `framework/`。
`[官方: basic_process.md 「部署后的目录结构」]`

**§2.2** 面向 GE custom_op_v2 扫描的交付，**应**同时提供 op_graph 交付树
（§1.4 布局），或由消费方以软链构造该布局。9.1.0 实测该布局可被扫描加载
（见 §9.1）。
`[官方: custom_op_development_guide.md §7.2；仓库实证: startlux-text prepare.sh 组装 work/opp]`

**§2.3** `ASCEND_CUSTOM_OPP_PATH` 语义**必须**按官方口径使用：「支持配置
多个路径，以英文冒号分隔」；GE 扫描两种模式——「(a) libcust_opapi.so
（新 SO 模式）；(b) op_graph/lib/<os>/<arch>/*.so（OPP 包模式）」；「动态库
编译和自定义算子包编译功能同时使用时，前者生成的动态库优先级更高」。
`[官方: cann/docs env-vars/ASCEND_CUSTOM_OPP_PATH.md；custom_op_architecture.md §4.3。9.1.0 实测差异见 §9.1]`

**§2.4** `.run` 安装包**必须**经 `npu_op_package(<name> TYPE RUN ...)`
产出；安装用 `./custom_opp_*.run [--install-path=<path>]`。
`[官方: asc-devkit cmake_function_reference.md 「npu_op_package」；basic_process.md]`

**§2.5** `vendor_name` **必须**唯一且稳定：「用于形成 vendors/<vendor_name>
安装目录，避免与其他厂商算子包冲突；同一安装路径下 vendor_name 相同的算子包
会安装到同一目录，同名文件会覆盖」。
`[官方: basic_process.md 编译配置表；torchair non_in_place_op_cases.md 同口径]`

**§2.6** 交付件 `libcust_opapi.so` **必须**不设置 `DT_SONAME`（构建时
`NO_SONAME ON`）。官方未规定 SONAME；本条沿用上游 fla-npu #391 约定，避免
动态加载器按通用逻辑名复用已按绝对路径加载的对象，并保证多包同名库各自独立
映射（§7.5(d) 依赖本条）。
`[仓库自定: 官方无规定；依据 fla-npu PR #391 + 本仓三包（c1d/qkv/mega）NO_SONAME 实证]`

**§2.7** 同进程自定义算子 .so 装载**必须**满足官方限额：「.so 文件数量
64 个；单个 .so 大小 800 MB；总加载大小 1000 MB」。
`[官方: custom_op_architecture.md §4.3 加载限制表]`

**§2.8** Host 侧库名**应**保留官方默认：ACLNN→`cust_opapi`、
GRAPH→`cust_opsproto_rt2.0`、TILING→`cust_opmaster_rt2.0`；「修改 Tiling
库或原型库的名称时，CMake 会提示兼容性警告。通常保留默认名称即可」。
`[官方: cmake_function_reference.md 「npu_op_library」]`

## 3. GE 接入路线

**§3.1** 路线选择**应**遵循官方分工：非标准 kernel/用户全控制执行 →
auto-mapping 路线；标准 AICore 流水线（框架托管 tiling/编译/下沉）→ aclnn
lowering 路线。「两者互补而非替代……两套机制在 GE 中共存」。
`[官方: custom_op_architecture.md §2.5]`

**§3.2** auto-mapping 路线**必须**使用随包官方接口头 `graph/custom_op.h`：
能力按需组合继承；执行上下文 API 按官方表使用（`GetInputTensor` /
`MallocOutputTensor` / `MallocWorkSpace` / `GetStream` / `GetOutputTensor` /
`MakeOutputRefInput` / `GetOption`——9.1.0 随包头实存；
`RequestAttachedStream` / `GetContextValue` 为 **master 文档口径**，9.1.0
未提供，迁移时按 §9.2 复测）；「输出内存由 context 管理生命周期，调用者
无需释放」。
`[官方: 随包头 custom_op.h；custom_op_development_guide.md §5.1]`

**§3.3** 注册时机为 **dlopen 触发静态初始化器**：「dlopen("libcust_opapi.so")
→ 执行静态初始化器 → CustomOpCreatorRegister → RegisterCustomOpCreator」。
交付件**不得**假设注册之外的任何入口被框架 dlsym。
`[官方: custom_op_architecture.md §4.2 时序图；custom_op.h 宏展开]`

**§3.4** 同一进程内**禁止**注册同名 op type：「注册一次性 |
RegisterCustomOpCreator 拒绝重复注册 | 同一进程中不可注册同名 op type」。
`[官方: custom_op_architecture.md §7 设计约束表]`

**§3.5** OpType **必须**大驼峰、首字符大写英文字母，「不要使用下划线、
连字符、空格、中文或其他特殊字符」；**禁止**与内置算子重名：「确定 OpType
后，可使用 optype_collector 工具查询 OpType 信息并检测命名冲突」。
`[官方: asc-devkit operator_prototype_definition.md]`

**§3.6** auto-mapping 路线三名**必须**一致：「REG_OP 中的 op type 名、
REG_AUTO_MAPPING_OP 的类名、构图侧使用的 op type 必须三者一致」。
`[官方: custom_op_development_guide.md §2.2]`

**§3.7** 数据/值依赖输入**必须**显式声明：原型侧 `ValueDepend(REQUIRED|
OPTIONAL)`（两条路线共用）。声明侧按路线分：aclnn lowering 路线经
`IMPL_OP(...).InputsDataDependency({...})` / `HostInputs({...})`（9.1.0
随包头 `register/op_impl_registry.h:115/118` 实存）；auto-mapping 路线
（`REG_AUTO_MAPPING_OP`）在 9.1.0 **无**链式声明接口，值依赖的 host 物化
不可用——需要 host 元数据取值的算子**应**选 aclnn lowering 路线（§9.1(d)，
causal-conv1d-qkv 先例）。master 文档描述的 auto-mapping 值依赖机制
（「框架自动为其生成 host 副本（执行图内插入 D2H 拷贝节点）」「动态输入
场景下取值须按 IR 索引（GetRequiredInputTensor），不能用扁平索引」）为
master 口径（§9.2），启用前须实测。「Execute 无需（也不应）自行 D2H 读值」
与「未声明数据依赖的输入无法获取到真实的 tensor 数据地址（获取到的地址为
nullptr）」引文保留。
`[官方(9.1.0 实锚): op_impl_registry.h:115/118；引文 = asc-devkit basic_development_flow.md 「数据依赖」、custom_op_development_guide.md §4.6；auto-mapping 无载体 = 本仓随包头核验]`

**§3.8** aclnn lowering 路线使用 `HostInputs` 时：该接口仅见于随包头声明
（`OpImplRegisterV2 &HostInputs(std::initializer_list<int32_t>)`），未检索到
公开文档条款；其生效依赖构建期注入 ops-info.json 的
`aclnnSupport=support_aclnn` 条目（`aic-<soc>-ops-info.json` 为
`npu_op_code_gen` 生成物，字段级格式官方未公开）。运行实现**必须**带探针与
回退（host 命中失败时退流序 D2H 并告警）。
`[仓库自定: 官方无文档条款；依据 随包头 op_impl_registry.h、本机 vendor ops-info 实读、causal-conv1d-qkv GE 包装实现]`

**§3.9** auto-mapping 算子实现**必须**线程安全且**不得**依赖每节点独立
实例：「同一 op type 的所有图节点共享同一个 BaseCustomOp 实例」「Compile
回调可能被并行调用（CustomGraphOptimizer 使用线程池），实现需保证线程
安全」。
`[官方: custom_op_architecture.md §4.2/§7]`

**§3.10** 能力接口**可**按需组合，GE 以 `dynamic_cast` 检测：「未实现的
接口自动跳过，不影响其他流程」；未实现 `PortableOp` 的算子仍可生成离线 OM，
但「其 kernel bin 等数据不会随 OM 保存和恢复，需自行管理」。
`[官方: custom_op_architecture.md §7；custom_op_development_guide.md §4.1]`

## 4. 符号面与构建卫生

**§4.1** `cust_opapi`、`cust_op_proto`、`cust_optiling` 三个 Host 目标
**必须**施加 `-fvisibility=hidden`（官方 CMake 模板逐目标带此选项）。
`[官方: basic_process.md Host 侧 CMakeLists 模板 「target_compile_options(cust_opapi PRIVATE -fvisibility=hidden)」]`

**§4.2** 经 `--whole-archive` 链入的预编译对象（opbuild 生成的 aclnn 对象、
vendored 静态库）**必须**在链接期以 version script 收敛导出面。官方未提及
version script；本条为事故驱动条款——§4.1 的编译期 hidden 覆盖不到预编译
对象，实测未收敛的安装库导出 72/57/119 个符号（c1d/qkv/mega 整改前）。
`[仓库自定: 依据 fla-npu #310 根因认定 + 本机 nm -D 实证]`

**§4.3** 导出白名单**必须**取最小集：(a) aclnn C 入口（`aclnnXxx`、
`aclnnXxxGetWorkspaceSize` 等——框架/应用经链接或 dlsym 调用）；(b) flat ABI
入口（`<op>_run`，供 JIT 桥直链与 activator fail-loud 校验）。注册类符号
一律不导出（注册靠静态初始化器，§3.3）。整改后参考值：c1d/qkv 各 4 导出、
mega 8 导出、危险泄漏均为 0。
`[仓库自定: 由官方模型推导——§3.3 + basic_process.md 交付件定义 + JIT 桥直链实证]`

**§4.4** **禁止**导出：GE 内部弱符号（`ge::StatusFactory` 及其 guard
variable、`ge::AscendString`/`ge::Operator` 析构）、STL/libstdc++ 内部符号
（旧 ABI `Ss` 串、`_Rb_tree`、`_Sp_counted_*`、`_Vector_base`、
`_Function_base`）、CANN 日志内部符号（`DlogRecordInner` /
`CheckLogLevelInner`）、typeinfo/vtable（`ge::BaseCustomOp` 系）。此类符号
在全局作用域下与 CANN 库同名符号互窜，是退出期堆损坏事故（glibc
`corrupted size vs. prev_size in fastbins`）的直接机制——LD_DEBUG=bindings
实证 40+ 个 CANN 库将上述符号绑到 vendor 库副本。
`[仓库自定: 本机 nm/LD_DEBUG 实证 + fla #310 + Ascend/memcache#2]`

**§4.5** `-Wl,-Bsymbolic` **可**用于让库内引用优先绑定本地定义，但**不得**
以之替代 §4.2/§4.3 的导出面收敛——它不减少动态符号表。
`[仓库自定: 实证——含 -Bsymbolic 的构建仍导出 72 符号并发生互窜事故]`

**§4.6** 交付件**导出面禁止**出现任何 C++ 实现细节符号（含 STL 内部符号，
不论新旧 ABI）；编译期 `_GLIBCXX_USE_CXX11_ABI` 取值**必须**与同进程
torch 一致（本环境 torch 为旧 ABI，跟随取值），与 CANN 库混编的目标跟随
CANN 取值。
`[仓库自定: JIT 桥构建实读 + nm 实证（旧 ABI Ss 符号属 §4.4 禁项）]`

## 5. torch 接入

**§5.1** torch 侧注册**必须**齐三件套：schema 定义
（`torch.library.Library(...).define` 或 `TORCH_LIBRARY`）、Meta 实现、
PrivateUse1 实现。「针对 NPU 设备，需要将算子实现注册到 PrivateUse1 这一
专属的 DispatchKey 上」。
`[官方: asc-devkit pytorch_framework.md；torchair op_plugin_adapt_torchair.md 样例]`

**§5.2** schema **必须**遵循「Tensor 类型输入在前，基本类型输入在后」；
仓库细化：可选 Tensor（如 `Tensor? bias`）视为 Tensor 类，**应**排在标量
属性之前，以保证 torchair 自动转换资格与 converter 入参序稳定（torch 2.12
boxed 调用存在标量丢失坑位）。
`[官方: torchair non_in_place_op_cases.md；后半句仓库自定: causal-conv1d 坑位记录]`

**§5.3** eager 库加载**应**优先官方口径 `torch.ops.load_library`；
activator 以 ctypes/dlopen 装载交付件时**必须**用 `RTLD_LOCAL`（可加
`RTLD_NOW|RTLD_NODELETE`），**禁止** `RTLD_GLOBAL`。
`[官方(load_library): pytorch_framework.md；RTLD 纪律=仓库自定: fla #391（「不支持预先以 RTLD_GLOBAL 发布 custom opapi 符号的用法」）+ 本次事故实证]`

**§5.4** 图模式 converter **必须**用
`@torchair.register_fx_node_ge_converter(torch.ops.<ns>.<op>.default)`
注册，函数体调用 `torchair.ge.custom_op(op_type, ...)`；`op_type`「取值与
算子原型 REG_OP 保持一致」，`inputs`/`outputs`/`attrs`「取值和顺序需与算子
原型 REG_OP 保持一致」；attrs **必须**经 `ge.attr.Xxx` 显式定类型；动态输出
**必须**用 `("名称", 数量)` 元组形式。
`[官方: torchair ascend_ir/api/ge/custom_op.md]`

**§5.5** 依赖 torchair 自动转换（免写 converter）时**必须**满足：Ascend C
算子名为 PyTorch 算子的大驼峰、入参顺序类型一致、且「PyTorch 算子原型定义中
不存在 Scalar 类型入参」；不满足任一即**必须**手写 converter。
`[官方: non_in_place_op_cases.md 「实现 Converter（可选）」]`

**§5.6** Meta 推导**必须**在 `torch.compile` 执行前完成注册。
`[官方: non_in_place_op_cases.md 「实现 Meta 推导函数」NOTE]`

**§5.7** 走 torchair max-autotune（GE/Ascend IR）模式时，算子 NPU 实现
**必须**以「Ascend C 算子工程化方式」开发并生成 `aclnnXxx`：「以 Kernel
直调方式开发的 NPU 实现不会生成 Ascend IR 注册逻辑……无法完成 PyTorch 算子
到 Ascend IR 的转换」；npugraph_ex（aclgraph）模式则「只需要保证 Torch
算子能在 Eager 模式下正常工作」。
`[官方: non_in_place_op_cases.md NOTE；torchair overview.md]`

**§5.8** `torch.library.Library` 句柄**必须**模块级存活（赋模块全局）；
函数局部句柄会被 GC 注销 schema，进而在 dynamo 内引发断言/段错误。
`[仓库自定: activator 实证 + AGENTS.md 坑位记录]`

**§5.9** import 顺序**必须** torch_npu 先于 torchair：torch_npu 负责在
`sys.modules` 注册顶层 torchair 别名（其捆绑副本）。
`[仓库自定: activator docstring 实证]`

**§5.10** 图模式可用性**必须**按包实测标注 dynamic 约束（如
causal-conv1d-qkv「图模式仅 dynamic=True 可用」），**禁止**跨包外推。
`[仓库自定: qkv 包门禁与 README 已知限制]`

**§5.11** activator 修改 `os.environ`（挂载 ASCEND_CUSTOM_OPP_PATH）
**必须**在任何 GE 初始化/算子编译发生前完成，且**不得**与其他线程的环境变量
读写并发。官方事故口径：「如果同时存在读、写操作，例如 getenv、putenv，则
可能导致环境变量访问冲突，进而导致程序异常」，处置为「删除代码中动态设置
环境变量的逻辑，可在执行程序前设置环境变量」。
`[官方(风险依据): cann/docs troubleshooting/env_var_conflict_app_terminated.md；import 期挂载=仓库自定]`

## 6. 装载与部署契约（消费方）

本章约束算子库的调用方（Python 框架进程、C++ GE 应用），不涉及任何具体
模型流程。

**§6.1** Python/框架进程装载 GE 交付件的**唯一文档化路径**：把官方
`set_env.bash` 指定的条目（其同时挂载包根与 `<pkg>/op_api/lib` 两段）或
op_graph 交付树挂到 `ASCEND_CUSTOM_OPP_PATH`，由框架/GE 扫描加载；「配置
该路径后，算子入图场景下，GE 框架在图编译和执行时根据该环境变量搜索算子
二进制并使用」。
`[官方: env-vars/ASCEND_CUSTOM_OPP_PATH.md；basic_process.md set_env.bash 条款]`

9.1.0 口径警告：vendor **包根目录作为条目实测无效**（§9.1(b)），官方
`set_env.bash` 的可用性来自其同时挂载的 `<pkg>/op_api/lib` 段（扫描模式
(a)，本仓未实测，§9.1(c)）；本仓已验证生效的条目是 op_graph 交付树
（§9.1(a)）。消费方**不得**依据「挂了包根但图编译报缺算子/段错误」推断
扫描机制本身无效——该现象是条目布局不匹配的特征（§9.1(b)），应按 §6.9
预检定位。

**§6.2** activator 需要主动 dlopen 交付件时（GE 扫描不可用或时序不满足），
**必须** `RTLD_LOCAL|RTLD_NOW|RTLD_NODELETE`，并保留 fail-loud 符号校验；
**禁止** `RTLD_GLOBAL`（同 §5.3）。
`[仓库自定: fla #391 + 本次事故（全局暴露 → 退出期 fastbins 损坏，4/4 确定性复现）]`

**§6.3** C++ GE 应用（API v1/v2）装载自定义算子库的**唯一文档化路径**同
§6.1：「export ASCEND_CUSTOM_OPP_PATH=... # 运行你的构图/执行程序」，GE 于
进程启动/初始化期扫描并 `mmDlopen(RTLD_NOW)`。**无** session 选项、GE 配置
项或显式装载 API。
`[官方: custom_op_development_guide.md §3 Step5；custom_op_architecture.md §4.3/§5.1；随包头 ge_api_v2.h 全文核验无装载接口]`

**§6.4** 应用侧自行 dlopen 交付件属**官方未覆盖区**；若目标 CANN 版本扫描
机制不可用（§9.1）而必须自行装载，**必须**：(a) flag 取
`RTLD_LOCAL|RTLD_NOW|RTLD_NODELETE`；(b) 在 `GEInitialize/GEInitializeV2`
之前完成；(c) 装载后校验关键入口符号。
`[仓库自定: 架构文档 §5.1 加载时序推导 + 本仓实证]`

**§6.5** **禁止**以 `LD_PRELOAD` 装载 vendor 算子库。理由：LD_PRELOAD 等价
于全局符号域发布，官方文档从未将其列为算子装载方式（唯一出处是 ASAN 调试
指南）；生态明确不推荐（memcache#2「Preloading ... globally with LD_PRELOAD
leads to heap corruption on shutdown」）；上游 fla #391 明确不支持；本仓
实测退出期堆损坏 4/4 确定性复现。
`[仓库自定(禁令): memcache#2 + fla #391 + 本次事故记录]`

**§6.6** 非 custom_op_v2 交付件（纯 `REG_OP` 原型库，如图兼容原型库）的
装载官方无规定（GE 扫描模式仅针对 `libcust_opapi.so`/op_graph 树）；此类库
**应**经 op_graph 交付树由官方扫描装载（异名 `.so` 在模式 (b) 通配范围内，
9.1.0 实测可用），或比照 §6.4 以 `RTLD_LOCAL` 显式 dlopen；**禁止**
LD_PRELOAD（§6.5 同源风险）。
`[仓库自定: 官方空白；依据 op_graph 树含原型库的实测（startlux-text 执行侧）]`

**§6.7** 应用经系统动态链接器使用 aclnn 动态库时，**必须**保证 `op_api/lib`
在搜索路径：「需要将 ${ASCEND_OPP_PATH}/vendors/<vendor_name>/op_api/lib/
加入 LD_LIBRARY_PATH，或使用应用自身的动态库搜索路径配置」。
`[官方: basic_process.md 表 1 及 NOTE]`

**§6.8** GE API v2 应用：`ge_api_v2.h` Migration Guide **无**自定义算子
装载条款；资源释放顺序**应**沿用官方 v1 样例模式：释放实例资源 → 销毁
Session → `aclrtResetDevice` → `GEFinalizeV2` → `aclFinalize`。
`[官方(随包): ge_api_v2.h 头注；v1 顺序 = gitee ascend/samples graph_run + ACL 样例]`

**§6.9** 消费方在 GE 初始化前**必须**对 `ASCEND_CUSTOM_OPP_PATH` 的每个
自定义算子条目做布局预检并 fail-loud：条目本身直接含 `libcust_opapi.so`
（扫描模式 (a)）或条目下存在 `op_graph/lib/<os>/<arch>/*.so`（模式 (b)）
方为有效；两者皆不满足（如 vendor 包根）时**必须**拒绝启动并按 §9.1(b)
口径提示，**不得**将执行交给 GE。依据：9.1.0 上缺库/错布局的失败形态是
`LoadGraph` 报 `Do not find op impl functions` 或直接 SIGSEGV（编译期无
显式报错），消费方无法事后归因。预检通过后**应**再以「图内每个自定义
op type 均有注册」的首次编译结果断言收口。
`[仓库自定: §9.1(b) 实测 + GE API v2 无装载/查询接口（§6.3 核验）]`

**§6.10** 同一算子库在同一交付流程的构图（导出）进程与执行进程中**应**
使用同一装载契约（同一条目的官方扫描装载，或同为 §6.4 显式 dlopen）；
确需分轨时（如附录 B 差距项 2 的豁免）**必须**在附录 B 登记分轨双方、
差异原因与合轨条件。**禁止**无登记的双轨漂移（例如导出侧 LD_PRELOAD 而
执行侧扫描）——两轨对注册时序与符号作用域的依赖不同，漂移会把「导出成功」
错误地外推为「执行装载也正确」。
`[仓库自定: 附录 B 差距项 2 现状抽象 + torchair 注册检查与 GE 扫描两套时序]`

## 7. 共存与冲突

**§7.1** 同一进程**禁止**出现同名 op type 的重复注册（框架拒绝：
`RegisterCustomOpCreator` 重复注册返回 `GRAPH_FAILED`）；新增算子包
**必须**先核查 op type 不与既有包/内置算子冲突（§3.4/§3.5）。
`[官方: custom_op_architecture.md §7；operator_prototype_definition.md]`

**§7.2** 多包含同名 OpType 时，生效版本由优先级决定：「多个算子包共存时，
如果包含相同 OpType 的算子，优先级高的算子包中的版本生效」；默认安装场景经
`opp/vendors/config.ini` 的 `load_priority`（「vendor 按从左到右的顺序确定
优先级，最左侧的 vendor 优先级最高」），指定目录安装场景经 set_env.bash
执行顺序（「脚本执行顺序越靠后，优先级越高」）。
`[官方: basic_process.md 「配置自定义算子优先级」]`

**§7.3** `load_priority` 有效性规则**必须**遵守：「配置项名称必须为
load_priority，不可修改」「等号右侧至少包含一个 vendor 名称，不能为空」
「vendor 名称应与 opp/vendors 目录下已安装算子包的目录名一致」「使用英文
逗号分隔」。
`[官方: basic_process.md 同节]`

**§7.4** 本仓安装脚本对 `load_priority` **必须**只追加、不重排既有顺序
（官方安装脚本语义是「将当前 vendor 调整到首位」，本仓刻意偏离）；偏离
理由：跨谱系同名算子类型（`CausalConv1d` 同时存在于 fla 系与 xllm 旧谱系
包）依赖固定优先级顺序保证生效者确定。本仓 `install.sh` **必须**在写入
config.ini 前后自动校验：既有 vendor 相对顺序保持不变、当前包按追加语义
落位；校验失败即回滚安装并报错。跨谱系同名算子的生效确定性**必须**由该
机器化校验保证，不得依赖人工复查。
`[仓库自定: 官方语义引文见 basic_process.md；偏离依据 = AGENTS.md 跨谱系红线 + 本机 config.ini 实况]`

**§7.5** 多包同名 `libcust_opapi.so` 共存：官方无文件级规定（仅 vendor_name
目录隔离 + 同名覆盖警告，§2.5）。仓库**必须**：(a) flat ABI 入口符号包内
唯一且带算子名前缀；(b) activator 定位库后 fail-loud 校验该符号；(c) 定位
顺序固定为 `$ASCEND_OPP_PATH` 先于 `$ASCEND_CUSTOM_OPP_PATH`，防他包同名库
遮蔽；(d) 各包满足 §2.6（无 SONAME）与 §4（符号收敛），使共载不依赖全局
作用域。跨谱系同名 `aclnnXxx` 导出（如 mega-gdn-prefill 与 causal-conv1d
均导出 `aclnnCausalConv1d*`，nm 实证各 3 个）**应**在包 README 已知限制中
登记；外部直调方**必须**走 (c) 的定位序或经 flat ABI 入口，**禁止**依赖
全局符号解析。
`[仓库自定: activator fail-loud 设计 + vendored README 已知限制 + fla #310 教训 + 2026-11-24 nm 审计]`

## 8. 测试与门禁

**§8.1** 每个算子包的排障附录**应**引用官方定位文档：单算子 coredump 分析、
进程中断定位思路（plog 三级排查）、ASAN 注入指南。
`[官方: cann/docs troubleshooting/{single_operator_coredump_analysis, process_interrupt_analysis, third_party_asan_memory_error}.md]`

**§8.2** 退出清洁度**必须**纳入包级门禁：业务完成后进程正常退出（exit 0、
无 glibc abort/coredump）。发现退出期堆损坏时，允许以「显式 teardown 后
`_exit(0)`」作为已记录规避，但**必须**在包 README 已知限制中登记并附根因
追踪项。门禁证据**必须**至少包含一次**主线程执行形态**（单实例 infer 顺序
执行，或等价的全部 GE/ACL 宿主分配发生在主线程的流程）；多线程/多实例
bench 的干净退出**不单独构成**门禁通过依据——glibc 每线程 arena 使执行期
堆损坏可被非主 arena 隔离掩蔽（事故实证：同一损坏 bench 2/2 干净、infer
4/4 abort）。疑似漏检时**应**以 `GLIBC_TUNABLES=glibc.malloc.arena_max=1`
复测单 arena 形态。
`[仓库自定: 本次事故 + Mooncake #4504 先例 + bench/infer 不对称实证]`

**§8.3** 功能门禁集（官方无规定，仓库统一要求）：(a) graph-vs-eager
`rel_l2 < 0.001`；(b) eager-vs-CPU-golden `cos > 0.95`；(c) 负例/边界用例
（不支持的 shape/dtype 必须报官方错误码而非静默错值）；(d) 状态类算子加
逐位相等门禁（如 conv_states bitwise）。
`[仓库自定: AGENTS.md 门禁条目 + 各包现行门禁]`

**§8.4** 性能结论**必须**来自消费方级 A/B（同图、同 case、同芯片、同计时
边界、同时段），**禁止**以单算子 harness 数据外推端到端收益。
`[仓库自定: qkv harness 外推被模型级实测证伪（harness 判定快 2%，模型级实测慢 5.9%）+ MegaGdnPrefill 单算子占优/端到端净负先例]`

**§8.5** 门禁执行环境**必须**在记录中固定：CANN 版本、SoC、torch/torch_npu
版本、逻辑芯片号；bench 前确认卡空闲，宿主高负载时段的数据整轮丢弃重测。
`[仓库自定: 本仓 provenance 口径 + qkv bench 离散度记录]`

## 9. 版本与环境矩阵

**§9.1** CANN 9.1.0（V100R001C11SPC001B243 / ascend910_93）实测面，
**必须**随版本升级复测：
(a) op_graph 交付树扫描**可用**——GE 扫描 `ASCEND_CUSTOM_OPP_PATH` 条目下
`op_graph/lib/<os>/<arch>/*.so`（含异名原型库）并自行 dlopen，装载作用域不
触发符号互窜（LD_PRELOAD 下必崩的配置在扫描装载下 3/3 干净退出；实测对象
为已按 §4 收敛的安装库，未收敛库在扫描装载下的行为不在本结论内）；
(b) vendor 包根目录作为 `ASCEND_CUSTOM_OPP_PATH` 条目**无效**（其
`op_api/lib` 子布局不匹配扫描模式 (a)/(b)），缺库时 `LoadGraph` 报
`Do not find op impl functions`，仅原型库在场时甚至直接段错误；
(c) 模式 (a)（条目目录直接含 `libcust_opapi.so`，即挂 `<pkg>/op_api/lib`）
在 9.1.0 **未实测**，启用前须按 (a) 同法验证；
(d) `IMPL_OP + OpExecuteFunc/HostInputs` 是 9.1.0 上框架 host 物化
`ValueDepend` 元数据输入的唯一机制，且依赖 ops-info `aclnnSupport` 条目。
`[仓库自定(实测记录): startlux-text 交付流程与实验日志]`

**§9.2** 文档版本差处理原则：gitcode master 文档可能领先于目标 CANN 版本的
实际行为；引用官方条款时**必须**标注「master 口径」，与实测冲突处**必须**
以随包头文件 + 本仓实测为准，并登记复测条件。
`[仓库自定: master 架构文档声明扫描模式 (a)+(b) vs 9.1.0 实测差异]`

**§9.3** 框架配套**应**以官方 release notes 配套表为准（如 torch_npu
v2.12.0-26.1.0 ↔ PyTorch 2.12.0 ↔ CANN 9.1.0 ↔ Python 3.10~3.13）；
「TorchNPU：注意与 CANN 等软件配套关系」。
`[官方: torchair overview.md 环境准备；hiascend release notes]`

**§9.4** CANN 升级或换 SoC **必须**重跑包级全量门禁（§8.2/§8.3），且
**不得**假设旧版本实测结论（含 §9.1 各项）继续成立。
`[仓库自定: c1d README 升级条款 + 上游 fla #1326 跨版本 API 漂移实例]`

## 附录 A：官方来源清单

| 来源 | 内容 |
|---|---|
| gitcode `cann/ge` docs/zh/user_guides/custom_op/custom_op_v2/{custom_op_development_guide, custom_op_architecture, custom_op_rfc}.md | custom_op_v2 交付件、SO 加载机制、注册时序、设计约束 |
| gitcode `cann/asc-devkit` .../aclnn_operator_development/compilation_and_deployment/basic_process.md | 官方 CMake 模板（-fvisibility=hidden）、vendor 包目录、load_priority、set_env.bash |
| gitcode `cann/asc-devkit` .../appendix/cmake_function_reference.md | npu_op_library / npu_op_package |
| gitcode `cann/asc-devkit` .../ai_framework_adaptation/pytorch_framework.md（=hiascend《PyTorch 框架》同源） | torch 三件套、torch.ops.load_library |
| gitcode `cann/asc-devkit` .../operator_graph_development/basic_development_flow.md、.../design_and_implementation/operator_prototype_definition.md | ValueDepend 数据依赖、OpType 命名 |
| gitcode `Ascend/torchair` docs/zh/custom_op_graph/{overview, op_plugin_adapt_torchair, non_in_place_op_cases}.md、docs/zh/ascend_ir/api/ge/custom_op.md | converter、Meta、自动转换条件、schema 惯例 |
| gitcode `cann/docs` docs/zh/env-vars/ASCEND_CUSTOM_OPP_PATH.md、docs/zh/troubleshooting/{single_operator_coredump_analysis, process_interrupt_analysis, third_party_asan_memory_error, env_var_conflict_app_terminated}.md | 环境变量语义、排障 |
| 随包头 `/usr/local/Ascend/ascend-toolkit/latest/include/graph/custom_op.h`、`include/register/op_impl_registry.h`、`include/ge/ge_api_v2.h` | auto-mapping 接口、IMPL_OP/HostInputs/InputsDataDependency、API v2 迁移 |
| hiascend release notes（Pytorch 2610 配套表，经本仓浏览记录转引） | torch_npu ↔ PyTorch ↔ CANN ↔ Python 配套关系（§9.3） |
| github `flashserve/flash-linear-attention-npu` #310 / PR #391、`Ascend/memcache` #2、`kvcache-ai/Mooncake` #4504 | RTLD_LOCAL 纪律、符号互窜先例、退出期损坏先例（非官方，生态佐证） |

## 附录 B：现状符合度与差距清单

符合度（2026-11-24 审计，nm 为安装态实测）：

| 包 | 路线 | hidden | version script | NO_SONAME | activator | 导出/危险泄漏 | 状态 |
|---|---|---|---|---|---|---|---|
| causal-conv1d | auto-mapping v2 | ✓ | ✓（4 导出） | ✓ | RTLD_LOCAL | 4 / 0 | 符合 |
| causal-conv1d-qkv | aclnn lowering（IMPL_OP+HostInputs+ops-info 注入） | ✓ | ✓（4 导出） | ✓ | RTLD_LOCAL | 4 / 0 | 符合 |
| mega-gdn-prefill | auto-mapping v2 | ✓ | ✓（8 导出） | ✓ | RTLD_LOCAL | 8 / 0 | 符合（2026-11-24 整改） |
| mtgr_attention | 非交付件（TileLang DSL） | N/A | N/A | N/A | 无 | 未构建 | 不适用（见其 README） |
| prefix-attention | auto-mapping v2 ×2 套构建 | ✗/部分 | ✗ | ✗ | pybind（无 ctypes） | 未构建，无法 nm | **差距，见下** |

差距与待办：

1. **prefix-attention**（在用消费方走 `third_party/custom_development_code/
   custom_op/prefix-attention/` 的分叉拷贝——11 文件差异，非本目录快照）：
   先拍板 canonical 拷贝归属，再整改——ascendc 腿：`op_host/CMakeLists.txt`
   直编/链接分支补 hidden、`-Bsymbolic`、`NO_SONAME`、version script
   （白名单 `aclnnPrefixInferAttentionScore*`，无 flat 入口），
   `torch_binding` 的 `dlopen(RTLD_LAZY)` 收紧为 LOCAL + fail-loud 校验；
   cannbot-dsl/ge 腿：`CMakeLists.txt` cust_opapi 补 hidden/`NO_SONAME`/
   version script（白名单 `prefix_attn_run_v2`），`src/prefix_attn_eager.cc`
   flat 入口补 visibility("default")，`prefix_attn_torch` 预载收紧。
   cannbot-dsl/ge 腿构建门禁 CANN ≥9.3，本机 9.1.0 无法验证，整改须在
   9.3 环境执行并回归 qwen2.5-0.5b。
2. **导出侧原型库装载（过渡豁免，附整改条件）**：Python 导出进程对纯原型库
   暂用 LD_PRELOAD（§6.5 唯一豁免对象，仅限
   `libchunk_gated_delta_rule_compat_proto.so`、仅限导出子进程，§6.10 登记
   于此）。豁免有效性**以导出面先按 §4.2/§4.3 收敛为前提**——当前导出面
   26 符号含 §4.4 类危险符号（`ge::AscendString`/`ge::Operator` 析构、
   `std::_Function_base`、`_Sp_counted_*` 等，nm 2026-11-24 实测），**该库
   必须先整改为 hidden + version script（纯 REG_OP 注册库收敛后可零导出），
   否则豁免失效**。装载方式迁移目标态为 op_graph 树挂载（执行侧已将该库
   软链进 op_graph 树并实测生效；导出进程挂同一 `ASCEND_CUSTOM_OPP_PATH`
   即可验证，若 GE 扫描时序晚于 torchair 注册检查则登记时序障碍后维持
   豁免）。豁免的移除条件与验证记录登记于本项。
3. **扫描模式 (a) 验证（短期必办）**：官方 `set_env.bash` 本身即「包根 +
   `<pkg>/op_api/lib`（模式 (a) 条目）」挂载样例（实读），验证可直接
   source 官方脚本对单一 vendor 跑一轮 infer 完成；通过后可省去消费方
   组装 op_graph 树，并同步消除 §6.1 的口径含混。
4. **CANN ≥9.2/9.3 复测**：升级后重跑 §9.1 四项 + 各包门禁（§9.4）。
