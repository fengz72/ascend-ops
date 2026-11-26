# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""文件说明：对外接口层：导出attention接口、预编译入口和配置；计算及launch行为位于host层。
模块层级：接口 → Host调度 → 独立metadata预处理及一次attention主kernel。
生命周期：跨流水的缓冲复用依赖真实生产消费关系，不能跳过对应同步。

算子流程图（括号内为代码模块的职责；AICPU预处理独立，attention主kernel一次launch）：

mtgr_ragged_segment_attention
  │  创建切分配置、检查输入，调用Host；返回调用方的output_snd。
  ▼
MtgrRaggedSegmentAttentionTiling.validate_inputs
  │  检查Q/K/V、分页缓存、分段规则、请求范围和核数等属性。
  ▼
launch_attention
  ├─ metadata未提供 → MtgrMetadataHost._prepare_metadata
  │                      └─ MtgrMetadataKernel._mtgr_metadata_kernel（独立AICPU任务，按L2工作集划分section）
  │                           ├─ _mtgr_task_cost：统计各Q任务实际活动KV块工作量。
  │                           └─ _mtgr_section：按活动KV权重计算每个核的连续Q任务范围。
  └─ metadata已提供 → 复用调用方计划。
  │
  ├─ MtgrRaggedSegmentAttentionTiling.validate_metadata（检查计划类型、设备和长度）
  └─ build_mtgr_ragged_segment_attention_executable（按规格编译或复用可执行对象）
         └─ MtgrRaggedSegmentAttentionLaunch.launch（一次attention主kernel发射）
              ▼
MtgrRaggedSegmentAttention.__call__（以下均为同一主kernel内的Cube/SIMD流水）
  │  读取metadata：定位当前section和本核负责的完整Q任务。
  ▼
[每个Q任务]
  ├─ MtgrRaggedSegmentAttentionVector.prepare_rows
  │    └─ cache_row_span：缓存本AIV各Q行的连续可见前段上界和对角位置。
  ├─ MtgrRaggedSegmentAttention.cube_visible_bounds：计算完整128行Q的可见KV块并集。
  └─ MtgrRaggedSegmentAttentionCube.copy_q：Q从GM整块搬入L1，同一Q任务跨KV块复用。
       ▼
[按逻辑KV顺序扫描；__call__筛选可见块]
  ├─ 整个Cube块全遮蔽 → 跳过该块，不搬运K/V，不执行QK/PV，不推进有效任务节拍。
  └─ 保留块 → 写入延迟任务记录（请求、Q/KV位置、状态槽和最后有效终点）。
       │
       ├─ C1：MtgrRaggedSegmentAttentionCube.compute_query_key
       │       ├─ copy_kv：按页表/实时序列选择K，GM→L1；Q/K从L1装入L0A/L0B。
       │       └─ Matmul(Q, Kᵀ)：Cube计算分数，FIXPIPE分发给两个AIV的UB。
       │
       ├─ V1：MtgrRaggedSegmentAttentionVector.softmax_and_publish_probability
       │       ├─ 全遮蔽AIV半块：zero_probability_tile（生成P=0，维护空块状态和必要消费同步）。
       │       ├─ 密集块：softmax_dense_batched（raw max、scale外提、64行状态批处理）。
       │       ├─ 普通掩码块：softmax_tile（逐元素mask、online max/exp/sum）。
       │       └─ 直接打包P_NZ，UB→L1发布P，同时保留alpha/den状态。
       │
       └─ 延迟两个有效节拍消费对应任务：
               ├─ C2：MtgrRaggedSegmentAttentionCube.compute_probability_value
               │       ├─ copy_kv：V从GM搬入L1；P/V装入L0A/L0B。
               │       └─ Matmul(P, V)：Cube计算PV，FIXPIPE分发结果到AIV的UB。
               └─ V2：MtgrRaggedSegmentAttentionVector.update_and_store_output
                       ├─ consume_masked_update：非首非末的空半块只保留真实消费依赖。
                       └─ update_tile：用alpha更新输出累加器；最后有效块除以den，
                                      转BF16并将有效Q行写回GM中的output_snd。
  ▼
__call__排空section末尾的延迟任务并完成必要同步，然后处理下一section或结束。

说明：上图表示数据依赖和职责，并非严格串行执行顺序或按时间比例绘制的时间线；
      环形buffer与延迟任务记录使不同有效块的Cube、Vector和搬运阶段能够交错执行。
"""


import cannbotdsl as dsl
from cannbotdsl import Array, MemLoc, PIPE, Tensor, dtypes, jit, kernel, make_layout, vf
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.ops import reg as vector_registers
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import cube_sync_pipe

if __package__:
    from .mtgr_ragged_segment_attention_common import (
        AIVS_PER_CUBE,
        BF16_BITS,
        CACHE_BLOCK_TOKENS,
        CORE_BEGIN_QUERY_TASK,
        CORE_END_QUERY_TASK,
        CORE_END_WORKGROUP,
        CORE_METADATA_WORDS,
        DEFAULT_KV_TILE_COLUMNS,
        FLOAT32_BITS,
        SMALL_HEAD_DIMENSION,
        FLOAT32_MAX_VALUE,
        HEADER_CORE_COUNT,
        HEADER_GQA_GROUP,
        HEADER_HEAD_DIMENSION,
        HEADER_KV_COLUMNS,
        HEADER_KV_HEADS,
        HEADER_MAGIC,
        HEADER_QUERY_ROWS,
        HEADER_REQUEST_COUNT,
        HEADER_SECTION_COUNT,
        MASKED_SCORE_FLOOR,
        MASK_CAUSAL,
        MASK_PREFIX_AND_SELF,
        MASK_SEGMENT_FULL,
        METADATA_HEADER_WORDS,
        METADATA_MAGIC,
        NO_DIAGONAL_TOKEN,
        NZ_BLOCK_EDGE,
        NZ_BLOCK_ELEMENTS,
        ONLINE_STATE_SLOTS,
        ROW_RULE_PARAMETER_PLANES,
        VECTOR_QUERY_ROWS,
        VECTOR_BF16_LANES,
        VECTOR_FLOAT32_LANES,
    )
else:
    from mtgr_ragged_segment_attention_common import (
        AIVS_PER_CUBE,
        BF16_BITS,
        CACHE_BLOCK_TOKENS,
        CORE_BEGIN_QUERY_TASK,
        CORE_END_QUERY_TASK,
        CORE_END_WORKGROUP,
        CORE_METADATA_WORDS,
        DEFAULT_KV_TILE_COLUMNS,
        FLOAT32_BITS,
        SMALL_HEAD_DIMENSION,
        FLOAT32_MAX_VALUE,
        HEADER_CORE_COUNT,
        HEADER_GQA_GROUP,
        HEADER_HEAD_DIMENSION,
        HEADER_KV_COLUMNS,
        HEADER_KV_HEADS,
        HEADER_MAGIC,
        HEADER_QUERY_ROWS,
        HEADER_REQUEST_COUNT,
        HEADER_SECTION_COUNT,
        MASKED_SCORE_FLOOR,
        MASK_CAUSAL,
        MASK_PREFIX_AND_SELF,
        MASK_SEGMENT_FULL,
        METADATA_HEADER_WORDS,
        METADATA_MAGIC,
        NO_DIAGONAL_TOKEN,
        NZ_BLOCK_EDGE,
        NZ_BLOCK_ELEMENTS,
        ONLINE_STATE_SLOTS,
        ROW_RULE_PARAMETER_PLANES,
        VECTOR_QUERY_ROWS,
        VECTOR_BF16_LANES,
        VECTOR_FLOAT32_LANES,
    )


@jit
def row3(
    source_tensor: Tensor,
    tile_row_index: int,
    attention_head_index: int,
):
    """按行和注意力头选出三维张量的二维行视图，不复制数据。

    Args:
        source_tensor: 用于选取行视图的源张量。
        tile_row_index: 当前AIV块内的行号。
        attention_head_index: 源张量中当前注意力头的索引。

    Returns:
        Tensor: 单行二维视图。
    """
    tensor_feature_axis = 2  # 本函数使用的tensor_feature_axis编译期配置。
    return source_tensor[tile_row_index, attention_head_index, None].reshape(
        (1, source_tensor.shape[tensor_feature_axis])
    )


@jit
def row4(
    source_tensor: Tensor,
    physical_cache_page_index: int,
    tile_row_index: int,
    attention_head_index: int,
):
    """按物理页、页内行和注意力头选出缓存行视图，不复制数据。

    Args:
        source_tensor: 用于选取行视图的源张量。
        physical_cache_page_index: 缓存中的物理页索引。
        tile_row_index: 当前AIV块内的行号。
        attention_head_index: 源张量中当前注意力头的索引。

    Returns:
        Tensor: 缓存单行二维视图。
    """
    tensor_feature_axis = 3  # 本函数使用的tensor_feature_axis编译期配置。
    return source_tensor[physical_cache_page_index, tile_row_index, attention_head_index, None].reshape(
        (1, source_tensor.shape[tensor_feature_axis])
    )


@jit
def onchip_row(
    source_tensor: Tensor,
    tile_row_index: int,
):
    """从片上二维张量选择一行，供安全尾块搬运使用。

    Args:
        source_tensor: 用于选取行视图的源张量。
        tile_row_index: 当前AIV块内的行号。

    Returns:
        Tensor: 片上单行视图。
    """
    return dsl.tile_slice(source_tensor, (1, source_tensor.shape[1]), (tile_row_index, 0))


class _MtgrPipelineResources:
    """管理同一 attention 主 kernel 内的片上缓冲和共享状态。

    __init__ 按固定顺序分配 L1、L0A、L0B、L0C 和 UB 资源，并建立 K/V 等共享别名。
    Cube 与 Vector 阶段通过传入同一资源对象复用缓冲，槽位读写依赖由各阶段维护。

    Attributes:
        query_tile_rows: 完整 Cube 查询块的行数。
        key_tile_columns: 每个键值块的列数。
        head_dimension: 每个注意力头的维度。
        vector_query_rows: 单个AIV负责的查询行数。
        vector_subblock_index: 本AIV子核编号。
        query_tensor: 保存查询块的 L1 环形缓冲池。
        key_tensor: 保存键值块的 L1 环形缓冲池，与 value_tensor 共享。
        value_tensor: L1 值缓冲，与 key_tensor 共享环形池。
        left_operand_slots: L0A左操作数环形池。
        qk_right_operand_slots: QK矩阵乘使用的L0B右操作数池。
        matrix_result_slots: L0C矩阵结果环形池。
        scores: QK分数池或分数视图。
        probability_l1: 两个AIV共享的L1概率环形池。
        alpha: 在线最大值变化的缩放因子池。
        maximum: 在线最大值存储。
        normalization_sum: 在线softmax分母。
        row_params: 每行前段及对角参数。
        softmax_scratch: 批量状态更新的临时存储。
        output_accumulator: 在线输出累加器。
        out: 最终输出暂存或设备输出视图。
        pv_right_operand_slots: PV矩阵乘使用的L0B右操作数池。
        weighted_values: 独立PV结果环形池；D128的行跨度包含32字节padding。
        probability_nz: NZ概率环形池。
    """
    def __init__(
        self,
        query_tile_rows: int,
        key_tile_columns: int,
        head_dimension: int,
    ):
        """分配片上缓冲并初始化共享流水状态。

        Args:
            query_tile_rows: 完整 Cube 查询块的行数。
            key_tile_columns: 每个键值块的列数。
            head_dimension: 每个注意力头的维度。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        weighted_value_row_padding_elements = 8  # D128的PV结果每行增加32字节跨度。
        weighted_value_slot_count = 1  # PV结果使用独立单槽，生命周期由RingBuffer管理。
        probability_nz_padding_elements = 16  # 每个NZ列组增加32字节，BF16为16个元素。
        softmax_scratch_planes = 3  # 保留三平面物理分配；dense仅使用max与配对sum两个平面。
        query_tensor_slot_count = 2  # 本函数使用的query_tensor_slot_count编译期配置。
        key_tensor_slot_count = 4  # 本函数使用的key_tensor_slot_count编译期配置。
        left_operand_slot_count = 2  # 本函数使用的left_operand_slot_count编译期配置。
        qk_right_operand_slot_count = 2  # 本函数使用的qk_right_operand_slot_count编译期配置。
        matrix_result_slot_count = 4  # 本函数使用的matrix_result_slot_count编译期配置。
        probability_l1_slot_count = 3  # 本函数使用的probability_l1_slot_count编译期配置。
        alpha_slot_count = 3  # 本函数使用的alpha_slot_count编译期配置。
        pv_right_operand_slot_count = 2  # 本函数使用的pv_right_operand_slot_count编译期配置。
        small_head_score_slots = 5  # 本函数使用的small_head_score_slots编译期配置。
        default_score_slot_count = 3 if head_dimension == key_tile_columns else 4  # D128分数池使用三槽，其余头维度保持四槽。
        probability_nz_slot_count = 4  # 本函数使用的probability_nz_slot_count编译期配置。
        self.query_tile_rows, self.key_tile_columns, self.head_dimension = (
            query_tile_rows,
            key_tile_columns,
            head_dimension,
        )  # 查询分块行数；键值分块列数；每个注意力头的维度。
        self.vector_query_rows = query_tile_rows // AIVS_PER_CUBE  # 单个AIV负责的查询行数。
        self.vector_subblock_index = get_subblock_id()  # 本AIV子核编号。
        self.query_tensor = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((query_tile_rows, head_dimension)), depth=query_tensor_slot_count
        )  # 查询张量；分配环形缓冲池，槽位由depth指定并由读写游标管理。
        self.key_tensor = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((key_tile_columns, head_dimension)), depth=key_tensor_slot_count
        )  # 键张量；分配环形缓冲池，槽位由depth指定并由读写游标管理。
        self.value_tensor = self.key_tensor  # 值张量。
        self.left_operand_slots = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((query_tile_rows, max(key_tile_columns, head_dimension))),
            depth=left_operand_slot_count,
        )  # L0A左操作数环形池；分配环形缓冲池，槽位由depth指定并由读写游标管理。
        self.qk_right_operand_slots = RingBuffer(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout((key_tile_columns, head_dimension)),
            depth=qk_right_operand_slot_count,
        )  # QK矩阵乘使用的L0B右操作数池；分配环形缓冲池，槽位由depth指定并由读写游标管理。
        if key_tile_columns == head_dimension:
            self.pv_right_operand_slots = self.qk_right_operand_slots  # PV矩阵乘使用的L0B右操作数池。
        else:
            self.pv_right_operand_slots = RingBuffer(
                MemLoc.L0B,
                dtypes.bfloat16,
                make_layout((head_dimension, key_tile_columns)),
                depth=pv_right_operand_slot_count,
            )  # PV矩阵乘使用的L0B右操作数池；分配环形缓冲池，槽位由depth指定并由读写游标管理。
        self.matrix_result_slots = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((query_tile_rows, max(key_tile_columns, head_dimension))),
            depth=matrix_result_slot_count,
        )  # L0C矩阵结果环形池；分配环形缓冲池，槽位由depth指定并由读写游标管理。
        self.scores = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.vector_query_rows, key_tile_columns)),
            depth=small_head_score_slots if head_dimension == SMALL_HEAD_DIMENSION else default_score_slot_count,
        )  # QK分数池或分数视图；分配环形缓冲池，槽位由depth指定并由读写游标管理。
        if head_dimension == key_tile_columns:
            self.weighted_values = RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout(
                    (self.vector_query_rows, head_dimension),
                    stride=(head_dimension + weighted_value_row_padding_elements, 1),
                ),
                depth=weighted_value_slot_count,
            )  # D128的独立PV单槽，不再与QK分数池交替复用。
        else:
            self.weighted_values = RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((self.vector_query_rows, head_dimension)),
                depth=weighted_value_slot_count,
            )  # PV结果池或结果视图；分配环形缓冲池，槽位由depth指定并由读写游标管理。

        if head_dimension == SMALL_HEAD_DIMENSION:
            self.probability_nz = RingBuffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((self.vector_query_rows, key_tile_columns)),
                depth=probability_nz_slot_count,
                data_format="nz",
            )  # NZ概率环形池；分配环形缓冲池，槽位由depth指定并由读写游标管理。
        else:
            self.probability_nz = RingBuffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((self.vector_query_rows, key_tile_columns)),
                depth=probability_nz_slot_count,
                physical_layout=make_layout(
                    (
                        key_tile_columns // NZ_BLOCK_EDGE,
                        self.vector_query_rows // NZ_BLOCK_EDGE,
                        NZ_BLOCK_EDGE,
                        NZ_BLOCK_EDGE,
                    ),
                    stride=(
                        (self.vector_query_rows // NZ_BLOCK_EDGE) * NZ_BLOCK_ELEMENTS + probability_nz_padding_elements,
                        NZ_BLOCK_ELEMENTS,
                        NZ_BLOCK_EDGE,
                        1,
                    ),
                ),
                data_format="nz",
            )  # NZ概率环形池；分配环形缓冲池，槽位由depth指定并由读写游标管理。
        self.probability_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((query_tile_rows, key_tile_columns)),
            depth=probability_l1_slot_count,
        )  # 两个AIV共享的L1概率环形池；分配环形缓冲池，槽位由depth指定并由读写游标管理。
        self.alpha = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, self.vector_query_rows)), depth=alpha_slot_count
        )  # 在线最大值变化的缩放因子池；分配环形缓冲池，槽位由depth指定并由读写游标管理。
        self.maximum = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, ONLINE_STATE_SLOTS * self.vector_query_rows))
        )  # 在线最大值存储；分配本核片上存储，供后续流水复用。
        self.normalization_sum = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, ONLINE_STATE_SLOTS * self.vector_query_rows))
        )  # 在线softmax分母；分配本核片上存储，供后续流水复用。
        self.row_params = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((1, ROW_RULE_PARAMETER_PLANES * self.vector_query_rows))
        )  # 每行前段及对角参数；分配本核片上存储，供后续流水复用。
        self.softmax_scratch = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, softmax_scratch_planes * self.vector_query_rows))
        )  # 批量状态更新的临时存储；分配本核片上存储，供后续流水复用。
        self.output_accumulator = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.vector_query_rows, head_dimension))
        )  # 在线输出累加器；分配本核片上存储，供后续流水复用。
        self.out = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((self.vector_query_rows, head_dimension))
        )  # 最终输出暂存或设备输出视图；分配本核片上存储，供后续流水复用。


class MtgrRaggedSegmentAttentionCube:
    """组织 QK 和 PV 两阶段的 Cube 计算及矩阵搬运。

    copy_q 和 copy_kv 负责 GM 到 L1 的查询、键值搬运；compute_query_key 和
    compute_probability_value 组织 L1 到 L0 的操作数装载、矩阵乘及结果分发。
    本类不保存实例属性，片上资源由各方法接收的 state 参数统一管理。
    """
    @jit
    def copy_q(
        self,
        destination: Tensor,
        query_tensor: Tensor,
        query_gm_token_begin: int,
        attention_head_index: int,
        query_tile_row_begin: int,
        live_sequence_length: int,
    ):
        """先用合法行填充尾部，再以带步长的整块视图将当前头的有效查询行搬入L1。

        Args:
            destination: 当前搬运目标视图。
            query_tensor: 查询张量。
            query_gm_token_begin: 请求查询在 GM 张量中的起始 token 偏移。
            attention_head_index: 源张量中当前注意力头的索引。
            query_tile_row_begin: 查询块在当前请求实时序列中的起始行偏移。
            live_sequence_length: 当前请求的实时序列长度。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        live_head_axis = 1  # 本函数使用的live_head_axis编译期配置。
        live_token_axis = 0  # 本函数使用的live_token_axis编译期配置。
        live_feature_axis = 2  # 本函数使用的live_feature_axis编译期配置。
        rows = min(destination.shape[0], live_sequence_length - query_tile_row_begin)  # 当前块中的有效行数。

        for tile_row_index in range(rows, destination.shape[0]):  # 当前AIV块内的行号；按给定区间依次遍历当前处理项。
            mem_copy(
                onchip_row(destination, tile_row_index), row3(query_tensor, query_gm_token_begin, attention_head_index)
            )
        source = query_tensor.permute(live_head_axis, live_token_axis, live_feature_axis)[
            attention_head_index,
            query_gm_token_begin + query_tile_row_begin : query_gm_token_begin + query_tile_row_begin + rows,
            None,
        ]  # 当前搬运来源视图。
        source = dsl.tile_slice(source, destination.shape, (0, 0))  # 当前搬运来源视图。
        mem_copy(destination, source)

    @jit
    def copy_kv(
        self,
        destination: Tensor,
        live_kv_tensor: Tensor,
        paged_kv_cache: Tensor,
        kv_page_table: Tensor,
        request_index: int,
        live_kv_gm_token_begin: int,
        attention_head_index: int,
        kv_logical_token_begin: int,
        cached_prefix_tokens: int,
        live_sequence_length: int,
    ):
        """依据逻辑列位置选择分页缓存或当前请求的实时键值，尾部使用合法行填充。

        Args:
            destination: 当前搬运目标视图。
            live_kv_tensor: 请求实时部分的键或值张量。
            paged_kv_cache: 缓存前缀使用的分页键或值张量。
            kv_page_table: 逻辑页到物理页的映射。
            request_index: 当前请求在批次中的索引。
            live_kv_gm_token_begin: 请求实时 K/V 在 GM 张量中的起始 token 偏移。
            attention_head_index: 源张量中当前注意力头的索引。
            kv_logical_token_begin: 键值块在包含缓存前缀的逻辑序列中的起始 token 偏移。
            cached_prefix_tokens: 当前请求的命中前缀长度。
            live_sequence_length: 当前请求的实时序列长度。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        live_head_axis = 1  # 本函数使用的live_head_axis编译期配置。
        live_token_axis = 0  # 本函数使用的live_token_axis编译期配置。
        live_feature_axis = 2  # 本函数使用的live_feature_axis编译期配置。
        cache_page_axis = 0  # 本函数使用的cache_page_axis编译期配置。
        cache_head_axis = 2  # 本函数使用的cache_head_axis编译期配置。
        cache_token_axis = 1  # 本函数使用的cache_token_axis编译期配置。
        cache_feature_axis = 3  # 本函数使用的cache_feature_axis编译期配置。
        rows = min(
            destination.shape[0], cached_prefix_tokens + live_sequence_length - kv_logical_token_begin
        )  # 当前块中的有效行数。
        if rows < destination.shape[0]:
            if cached_prefix_tokens > 0:
                first_page = kv_page_table[request_index, 0]  # 用于尾行安全填充的首个物理缓存页。
                if first_page >= 0 and first_page < paged_kv_cache.shape[0]:
                    for tile_row_index in range(
                        rows, destination.shape[0]
                    ):  # 当前AIV块内的行号；按给定区间依次遍历当前处理项。
                        mem_copy(
                            onchip_row(destination, tile_row_index),
                            row4(paged_kv_cache, first_page, 0, attention_head_index),
                        )
            else:
                for tile_row_index in range(
                    rows, destination.shape[0]
                ):  # 当前AIV块内的行号；按给定区间依次遍历当前处理项。
                    mem_copy(
                        onchip_row(destination, tile_row_index),
                        row3(live_kv_tensor, live_kv_gm_token_begin, attention_head_index),
                    )

        if kv_logical_token_begin < cached_prefix_tokens:
            physical = kv_page_table[
                request_index, kv_logical_token_begin // CACHE_BLOCK_TOKENS
            ]  # 当前逻辑缓存页对应的物理页号。
            if physical >= 0 and physical < paged_kv_cache.shape[0]:
                source = paged_kv_cache.permute(cache_page_axis, cache_head_axis, cache_token_axis, cache_feature_axis)[
                    physical,
                    attention_head_index,
                    kv_logical_token_begin % CACHE_BLOCK_TOKENS : kv_logical_token_begin % CACHE_BLOCK_TOKENS + rows,
                    None,
                ]  # 当前搬运来源视图。
                source = dsl.tile_slice(source, destination.shape, (0, 0))  # 当前搬运来源视图。
                mem_copy(destination, source)
        else:
            live_kv_gm_copy_begin = (
                live_kv_gm_token_begin + kv_logical_token_begin - cached_prefix_tokens
            )  # 首个有效块标记。
            source = live_kv_tensor.permute(live_head_axis, live_token_axis, live_feature_axis)[
                attention_head_index, live_kv_gm_copy_begin : live_kv_gm_copy_begin + rows, None
            ]  # 当前搬运来源视图。
            source = dsl.tile_slice(source, destination.shape, (0, 0))  # 当前搬运来源视图。
            mem_copy(destination, source)

    @jit
    def compute_query_key(
        self,
        state: _MtgrPipelineResources,
        query_tile_write: Tensor,
        key: Tensor,
        paged_kv_cache: Tensor,
        kv_page_table: Tensor,
        batch_index: int,
        sequence_gm_token_begin: int,
        kv_head_index: int,
        key_column_start: int,
        cached_prefix_tokens: int,
        live_sequence_length: int,
    ):
        """搬入当前键块，将查询和键装入左右矩阵操作数，执行QK矩阵乘并分发分数。

        Args:
            state: 共享片上资源状态。
            query_tile_write: 已取得的 L1 查询缓冲槽，供 QK 矩阵乘读取。
            key: GM 中的实时键张量。
            paged_kv_cache: 缓存前缀使用的分页键或值张量。
            kv_page_table: 逻辑页到物理页的映射。
            batch_index: 请求编号。
            sequence_gm_token_begin: 当前请求的实时token起点。
            kv_head_index: 键值头编号。
            key_column_start: 逻辑键值块起点。
            cached_prefix_tokens: 当前请求的命中前缀长度。
            live_sequence_length: 当前请求的实时序列长度。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        key_tile_write = state.key_tensor.next_write()  # 当前键块的L1待写槽；取得受生产消费依赖保护的待写槽。
        self.copy_kv(
            key_tile_write,
            key,
            paged_kv_cache,
            kv_page_table,
            batch_index,
            sequence_gm_token_begin,
            kv_head_index,
            key_column_start,
            cached_prefix_tokens,
            live_sequence_length,
        )
        left_operand_write = (
            state.left_operand_slots.next_write()
        )  # 当前矩阵乘的L0A待写槽；取得受生产消费依赖保护的待写槽。
        right_operand_write = (
            state.qk_right_operand_slots.next_write()
        )  # 当前矩阵乘的L0B待写槽；取得受生产消费依赖保护的待写槽。
        mem_copy(left_operand_write[: state.query_tile_rows, : state.head_dimension], query_tile_write)
        mem_copy(right_operand_write, state.key_tensor.next_read())
        matrix_result_write = (
            state.matrix_result_slots.next_write()
        )  # 当前矩阵乘的L0C结果槽；取得受生产消费依赖保护的待写槽。
        matmul(
            matrix_result_write[: state.query_tile_rows, : state.key_tile_columns],
            state.left_operand_slots.next_read()[: state.query_tile_rows, : state.head_dimension],
            state.qk_right_operand_slots.next_read(),
            init=True,
        )
        mem_copy(
            state.scores.next_write(),
            state.matrix_result_slots.next_read()[: state.query_tile_rows, : state.key_tile_columns].partition(dim=0),
        )

    @jit
    def compute_probability_value(
        self,
        state: _MtgrPipelineResources,
        value: Tensor,
        paged_kv_cache: Tensor,
        kv_page_table: Tensor,
        batch_index: int,
        sequence_gm_token_begin: int,
        kv_head_index: int,
        key_column_start: int,
        cached_prefix_tokens: int,
        live_sequence_length: int,
    ):
        """搬入当前值块，以已发布概率执行PV矩阵乘，并分发加权值结果。

        Args:
            state: 共享片上资源状态。
            value: GM 中的实时值张量。
            paged_kv_cache: 缓存前缀使用的分页键或值张量。
            kv_page_table: 逻辑页到物理页的映射。
            batch_index: 请求编号。
            sequence_gm_token_begin: 当前请求的实时token起点。
            kv_head_index: 键值头编号。
            key_column_start: 逻辑键值块起点。
            cached_prefix_tokens: 当前请求的命中前缀长度。
            live_sequence_length: 当前请求的实时序列长度。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        value_tile_write = state.value_tensor.next_write()  # 当前值块的L1待写槽；取得受生产消费依赖保护的待写槽。
        self.copy_kv(
            value_tile_write,
            value,
            paged_kv_cache,
            kv_page_table,
            batch_index,
            sequence_gm_token_begin,
            kv_head_index,
            key_column_start,
            cached_prefix_tokens,
            live_sequence_length,
        )
        left_operand_write = (
            state.left_operand_slots.next_write()
        )  # 当前矩阵乘的L0A待写槽；取得受生产消费依赖保护的待写槽。
        right_operand_write = (
            state.pv_right_operand_slots.next_write()
        )  # 当前矩阵乘的L0B待写槽；取得受生产消费依赖保护的待写槽。
        mem_copy(
            left_operand_write[: state.query_tile_rows, : state.key_tile_columns], state.probability_l1.next_read()
        )
        mem_copy(right_operand_write, state.value_tensor.next_read().permute(1, 0))
        matrix_result_write = (
            state.matrix_result_slots.next_write()
        )  # 当前矩阵乘的L0C结果槽；取得受生产消费依赖保护的待写槽。
        matmul(
            matrix_result_write[: state.query_tile_rows, : state.head_dimension],
            state.left_operand_slots.next_read()[: state.query_tile_rows, : state.key_tile_columns],
            state.pv_right_operand_slots.next_read(),
            init=True,
        )
        mem_copy(
            state.weighted_values.next_write(),
            state.matrix_result_slots.next_read()[: state.query_tile_rows, : state.head_dimension].partition(dim=0),
        )


class MtgrRaggedSegmentAttentionVector:
    """组织 SIMD 掩码、online softmax 和输出更新。

    prepare_rows 缓存行级可见范围；softmax_and_publish_probability 选择密集、掩码或
    全遮蔽路径并发布概率；update_and_store_output 消费 PV，更新累加器并在末块写回。
    softmax_diagonal_batched 批量更新仅对角可见块；softmax_dense_batched 处理密集块，
    softmax_tile 处理其余掩码块。片上资源通过 state 或缓冲参数传入。
    """
    @jit
    def cache_row_span(
        self,
        params: Tensor,
        query_logical_begin: int,
        segment_kv_begin: int,
        segment_kv_end: int,
        segment_mask_rule: int,
        is_first_segment_intersection: int,
    ):
        """以向量通道表示查询行，根据段交集批量写入可见前段上界和对角位置。

        Args:
            params: 每行连续前段上界与对角位置的INT32存储。
            query_logical_begin: 当前查询块的逻辑起始位置。
            segment_kv_begin: 当前段的逻辑起点。
            segment_kv_end: 当前区间的结束位置。
            segment_mask_rule: 当前段的掩码规则编号。
            is_first_segment_intersection: 首个有效块标记。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        with vf(mode="simd"):
            full, _ = vector_registers.update_mask(
                VECTOR_FLOAT32_LANES, elem_bits=FLOAT32_BITS
            )  # 全部有效向量通道的执行掩码；调用方无需使用的返回项。
            positions = vector_registers.varange(
                dtypes.int32(query_logical_begin), dtypes.int32
            )  # 64个查询行的逻辑位置向量。
            active = vector_registers.mask_and(
                vector_registers.vges(positions, dtypes.int32(segment_kv_begin), mask=full),
                vector_registers.vlts(positions, dtypes.int32(segment_kv_end), mask=full),
                exec_mask=full,
            )  # 当前段覆盖的查询行通道掩码。
            old_cutoff = vector_registers.vdups(0, dtypes.int32, mask=full)  # 此前已写入的各行连续前段边界。
            old_diagonal = vector_registers.vdups(
                NO_DIAGONAL_TOKEN, dtypes.int32, mask=full
            )  # 此前已写入的各行对角位置。
            if is_first_segment_intersection == 0:
                old_cutoff = vector_registers.vload(
                    params, 0
                )  # 此前已写入的各行连续前段边界；从片上存储读取当前计算需要的数据。
                old_diagonal = vector_registers.vload(
                    params, VECTOR_QUERY_ROWS
                )  # 此前已写入的各行对角位置；从片上存储读取当前计算需要的数据。
            cutoff = vector_registers.vdups(
                dtypes.int32(segment_kv_begin), dtypes.int32, mask=full
            )  # 当前行连续可见前段的结束位置。
            diagonal = vector_registers.vdups(NO_DIAGONAL_TOKEN, dtypes.int32, mask=full)  # 当前行单独可见的对角位置。
            if const_expr(segment_mask_rule == MASK_CAUSAL):
                cutoff = vector_registers.vadds(positions, 1, mask=full)  # 当前行连续可见前段的结束位置。
            elif const_expr(segment_mask_rule == MASK_SEGMENT_FULL):
                cutoff = vector_registers.vdups(
                    dtypes.int32(segment_kv_end), dtypes.int32, mask=full
                )  # 当前行连续可见前段的结束位置。
            else:
                diagonal = positions  # 当前行单独可见的对角位置。
            vector_registers.vstore(params, 0, vector_registers.vselect(cutoff, old_cutoff, cond_mask=active), full)
            vector_registers.vstore(
                params, VECTOR_QUERY_ROWS, vector_registers.vselect(diagonal, old_diagonal, cond_mask=active), full
            )

    @jit
    def softmax_diagonal_batched(
        self,
        scores: Tensor,
        prob: Tensor,
        maximum: Tensor,
        denominator: Tensor,
        alpha: Tensor,
        params: Tensor,
        key_column_start: int,
        query_task_sequence: int,
        scale: float,
        is_first_kv_block: bool,
    ):
        """批量处理每行至多一个可见对角元素的键块。

        调用方保证键块不含任何连续前段位置、半块非全遮蔽且scale正而有限。一次gather读取各行对角分数，
        并行更新行最大值、分母与alpha；随后按真实NZ列组跨度清零P，仅scatter有效对角概率。
        空行沿用历史状态；零填充和scatter之间保留写后写屏障。

        Args:
            scores: 当前AIV的64行、128列FP32分数。
            prob: 待发布的NZ布局BF16概率。
            maximum: 三组在线行最大值。
            denominator: 三组在线行分母。
            alpha: 当前块的64行历史缩放因子。
            params: 行连续前段与对角位置参数。
            key_column_start: 当前128列键块的逻辑起点。
            query_task_sequence: 查询任务序号。
            scale: 分数缩放因子。
            is_first_kv_block: 首键块的编译期标记。

        Returns:
            None: 原位更新在线状态和概率缓冲区。
        """
        score_row_shift = 7  # 每行128个FP32元素的地址移位。
        nz_row_shift = 4  # NZ内块宽16元素的地址移位。
        nz_group_stride = prob.physical_layout.stride[0]  # 包括列组padding的真实物理跨度。
        nz_column_mask = 15  # NZ列内偏移的低四位。
        clear_unroll = 4  # 连续零填充循环的展开提示。
        with vf(mode="simd"):
            full, _ = vector_registers.update_mask(VECTOR_FLOAT32_LANES, elem_bits=FLOAT32_BITS)  # 64行FP32在线状态的执行掩码。
            packed_full, _ = vector_registers.update_mask(VECTOR_BF16_LANES, elem_bits=BF16_BITS)  # 完整128列BF16清零掩码。
            packed_rows, _ = vector_registers.update_mask(VECTOR_QUERY_ROWS, elem_bits=BF16_BITS)  # 打包后仅前64个BF16槽参与scatter。
            zero = vector_registers.vdups(0.0, dtypes.float32, mask=full)  # 空概率与无历史状态的FP32零向量。
            one = vector_registers.vdups(1.0, dtypes.float32, mask=full)  # 空行保持历史时使用的单位缩放向量。
            zero_index = vector_registers.vdups(0, dtypes.uint32, mask=full)  # 无效行的安全列索引及布尔假值。
            one_index = vector_registers.vdups(1, dtypes.uint32, mask=full)  # 布尔真值，供跨位宽打包有效掩码使用。
            row_index = vector_registers.varange(0, dtypes.uint32)  # 当前AIV的64个行号，每个寄存器通道对应一行。
            diagonal = vector_registers.vload(params, VECTOR_QUERY_ROWS)  # 各行逻辑对角位置，空行使用负哨兵。
            nonempty = vector_registers.mask_and(
                vector_registers.vges(diagonal, dtypes.int32(key_column_start), mask=full),
                vector_registers.vles(
                    diagonal, dtypes.int32(key_column_start + DEFAULT_KV_TILE_COLUMNS - 1), mask=full
                ),
                exec_mask=full,
            )
            column_index = vector_registers.vreinterpret(
                vector_registers.vadds(diagonal, -dtypes.int32(key_column_start), mask=full), dtypes.uint32
            )
            column_index = vector_registers.vselect(column_index, zero_index, cond_mask=nonempty)  # 无效行钳到安全列，避免gather索引越界。
            score_index = vector_registers.vadd(
                vector_registers.vshl(row_index, score_row_shift, mask=full), column_index, mask=full
            )
            selected_score = vector_registers.vmuls(
                vector_registers.vgather(scores, score_index, mask=full), scale, mask=full
            )  # 64行各自唯一可见位置的缩放分数。
            block_maximum = vector_registers.vmax(
                vector_registers.vdups(MASKED_SCORE_FLOOR, dtypes.float32, mask=full), selected_score, mask=full
            )
            state_base = query_task_sequence % ONLINE_STATE_SLOTS * VECTOR_QUERY_ROWS  # 当前任务的在线状态槽起点。
            previous_maximum = zero  # 首键块使用零历史最大值。
            previous_denominator = zero  # 首键块使用零历史分母。
            updated_maximum = block_maximum  # 首键块以当前唯一分数的最大值为初值。
            factor = zero  # 首个非空块不保留旧累加器。
            if const_expr(not is_first_kv_block):
                previous_maximum = vector_registers.vload(maximum, state_base)  # 一次载入64行历史最大值。
                previous_denominator = vector_registers.vload(denominator, state_base)  # 一次载入64行历史分母。
                history = vector_registers.vgts(previous_denominator, 0.0, mask=full)  # 各行已有有效历史的标记。
                updated_maximum = vector_registers.vselect(
                    vector_registers.vmax(previous_maximum, block_maximum, mask=full), block_maximum,
                    cond_mask=history,
                )
                factor = vector_registers.vselect(
                    vector_registers.vexp_sub(previous_maximum, updated_maximum, mask=full), zero, cond_mask=history
                )
            updated_maximum = vector_registers.vselect(updated_maximum, previous_maximum, cond_mask=nonempty)  # 无可见对角位置的行保留历史最大值。
            factor = vector_registers.vselect(factor, one, cond_mask=nonempty)  # 空行的历史缩放固定为一。
            probability = vector_registers.vselect(
                vector_registers.vexp_sub(selected_score, updated_maximum, mask=full), zero, cond_mask=nonempty
            )
            total = vector_registers.vadd(
                vector_registers.vmul(previous_denominator, factor, mask=full), probability, mask=full
            )
            vector_registers.vstore(maximum, state_base, updated_maximum, full)
            vector_registers.vstore(denominator, state_base, total, full)
            vector_registers.vstore(alpha, 0, factor, full)
            packed_zero = vector_registers.vdups(0.0, dtypes.bfloat16, mask=packed_full)  # 用于清空128列概率的BF16零向量。
            for probability_row in dsl_range(0, prob.shape[0], 1, unroll=clear_unroll):
                vector_registers.vstore_strided(
                    prob, probability_row * NZ_BLOCK_EDGE, packed_zero, packed_full,
                    block_stride=nz_group_stride // NZ_BLOCK_EDGE, repeat_stride=0,
                )
            nz_index = vector_registers.vadd(
                vector_registers.vadd(
                    vector_registers.vmuls(
                        vector_registers.vshr(column_index, nz_row_shift, mask=full), nz_group_stride, mask=full
                    ),
                    vector_registers.vshl(row_index, nz_row_shift, mask=full), mask=full,
                ),
                vector_registers.vbitwise_and(
                    column_index, vector_registers.vdups(nz_column_mask, dtypes.uint32, mask=full), mask=full
                ),
                mask=full,
            )
            packed_index = vector_registers.vcast(nz_index, dtypes.uint16, mask=full)  # 物理NZ偏移最大8303，可精确转换为uint16。
            packed_index, _ = vector_registers.vdeinterleave(packed_index, packed_index)  # 将64个有效索引收紧至前64个16位槽。
            packed_probability = vector_registers.vcast(probability, dtypes.bfloat16, mask=full)  # 保持原RN规则转换64行唯一概率。
            packed_probability, _ = vector_registers.vdeinterleave(packed_probability, packed_probability)  # 概率与索引按相同顺序紧凑排列。
            packed_valid = vector_registers.vcast(
                vector_registers.vselect(one_index, zero_index, cond_mask=nonempty), dtypes.uint16, mask=full
            )
            packed_valid, _ = vector_registers.vdeinterleave(packed_valid, packed_valid)  # 将有效行标记压紧到BF16粒度。
            scatter_mask = vector_registers.vgts(packed_valid, 0, mask=packed_rows)  # 仅真实可见行写P，避免空行覆盖有效位置。
            vector_registers.vmem_bar("vst_vst")  # 保证完整零填充先于对角概率覆盖。
            vector_registers.vscatter(prob, packed_probability, packed_index, mask=scatter_mask)

    @jit
    def softmax_tile(
        self,
        scores: Tensor,
        prob: Tensor,
        maximum: Tensor,
        denominator: Tensor,
        alpha: Tensor,
        params: Tensor,
        key_column_start: int,
        query_task_sequence: int,
        scale: float,
        dense: bool,
        is_first_kv_block: bool,
        has_diagonal: bool,
        uniform_prefix_mask: bool = False,
        specialize_prefix_halves: bool = False,
        prefix_within_low_half: bool = False,
    ):
        """处理非密集块的掩码及在线状态，保留空历史、空块语义，配对归约分母并直接发布NZ概率。

        Args:
            scores: QK分数池或分数视图。
            prob: 待发布的BF16概率视图。
            maximum: 在线最大值存储。
            denominator: 在线分母存储。
            alpha: 在线最大值变化的缩放因子池。
            params: 每行连续前段上界与对角位置的INT32存储。
            key_column_start: 逻辑键值块起点。
            query_task_sequence: 当前查询任务序号。
            scale: 分数缩放因子。
            dense: 整块连续可见标记。
            is_first_kv_block: 首个有效块标记。
            has_diagonal: 当前键块可能包含可见对角位置的编译期标记。
            uniform_prefix_mask: 有效64行共享非空前段掩码、且当前块无可见对角的编译期标记。
            specialize_prefix_halves: 根据前段边界特化两个64列半块的编译期标记。
            prefix_within_low_half: 前段终点位于前64列以内，后64列完全不可见的编译期标记。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        query_row_unroll = 4  # 本函数使用的query_row_unroll编译期配置。
        last_kv_column_offset = 127  # 本函数使用的last_kv_column_offset编译期配置。
        with vf(mode="simd"):
            full, _ = vector_registers.update_mask(
                VECTOR_FLOAT32_LANES, elem_bits=FLOAT32_BITS
            )  # 全部有效向量通道的执行掩码；调用方无需使用的返回项。
            packed_mask, _ = vector_registers.update_mask(
                VECTOR_BF16_LANES, elem_bits=BF16_BITS
            )  # BF16打包存储的有效通道掩码；调用方无需使用的返回项。
            zero = vector_registers.vdups(0.0, dtypes.float32, mask=full)  # 零值向量。
            one = vector_registers.vdups(1.0, dtypes.float32, mask=full)  # 单位值向量。
            negative_sentinel = vector_registers.vdups(
                MASKED_SCORE_FLOOR, dtypes.float32, mask=full
            )  # 被遮蔽分数使用的有限负哨兵。
            keys0 = vector_registers.varange(
                dtypes.int32(key_column_start), dtypes.int32
            )  # 当前键值块前64列的逻辑位置向量。
            keys1 = vector_registers.varange(
                dtypes.int32(key_column_start + VECTOR_FLOAT32_LANES), dtypes.int32
            )  # 当前键值块后64列的逻辑位置向量。
            state_base = (
                query_task_sequence % ONLINE_STATE_SLOTS * VECTOR_QUERY_ROWS
            )  # 当前查询任务对应的在线状态槽起点。
            shared_visible0 = full  # 特化路径在行循环外生成前64列共享掩码。
            shared_visible1 = full  # 特化路径在行循环外生成后64列共享掩码。
            packed_high_zero = vector_registers.vdups(0.0, dtypes.bfloat16, mask=packed_mask)  # 全遮蔽后半块的打包零值。
            if const_expr(uniform_prefix_mask):
                shared_cutoff = vector_registers.vload_broadcast(params, 0)
                shared_visible0 = vector_registers.vlt(keys0, shared_cutoff, mask=full)
                shared_visible1 = vector_registers.vlt(keys1, shared_cutoff, mask=full)
            for tile_row_index in dsl_range(
                0, scores.shape[0], 1, unroll=query_row_unroll
            ):  # 当前AIV块内的行号；按给定区间依次遍历当前处理项。
                visible0 = full  # 前64列可见掩码。
                visible1 = full  # 后64列可见掩码。
                nonempty = full  # 当前行与键值块有可见交集的标记。
                score_low_half = vector_registers.vmuls(
                    vector_registers.vload(scores, tile_row_index * DEFAULT_KV_TILE_COLUMNS), scale, mask=full
                )  # 当前行前64列分数。
                if const_expr(not specialize_prefix_halves or not prefix_within_low_half):
                    score_high_half = vector_registers.vmuls(
                        vector_registers.vload(scores, tile_row_index * DEFAULT_KV_TILE_COLUMNS + VECTOR_FLOAT32_LANES),
                        scale,
                        mask=full,
                    )  # 当前行后64列分数。
                else:
                    score_high_half = negative_sentinel
                if const_expr(not dense):
                    if const_expr(uniform_prefix_mask):
                        visible0 = shared_visible0
                        visible1 = shared_visible1
                    else:
                        cutoff = vector_registers.vload_broadcast(
                            params, tile_row_index
                        )  # 当前行连续可见前段的结束位置；从片上存储读取当前计算需要的数据。
                        visible0 = vector_registers.vlt(keys0, cutoff, mask=full)
                        visible1 = vector_registers.vlt(keys1, cutoff, mask=full)
                        nonempty = vector_registers.vgts(cutoff, dtypes.int32(key_column_start), mask=full)
                        if const_expr(has_diagonal):
                            diagonal = vector_registers.vload_broadcast(
                                params, VECTOR_QUERY_ROWS + tile_row_index
                            )  # 当前行单独可见的对角位置；仅对角范围与键块相交时读取。
                            visible0 = vector_registers.mask_or(
                                visible0, vector_registers.veq(keys0, diagonal, mask=full), exec_mask=full
                            )
                            visible1 = vector_registers.mask_or(
                                visible1, vector_registers.veq(keys1, diagonal, mask=full), exec_mask=full
                            )
                            nonempty = vector_registers.mask_or(
                                nonempty,
                                vector_registers.mask_and(
                                    vector_registers.vges(diagonal, dtypes.int32(key_column_start), mask=full),
                                    vector_registers.vles(
                                        diagonal, dtypes.int32(key_column_start + last_kv_column_offset), mask=full
                                    ),
                                    exec_mask=full,
                                ),
                                exec_mask=full,
                            )
                    if const_expr(not specialize_prefix_halves or prefix_within_low_half):
                        score_low_half = vector_registers.vselect(
                            score_low_half, negative_sentinel, cond_mask=visible0
                        )  # 当前行前64列分数。
                    if const_expr(not specialize_prefix_halves or not prefix_within_low_half):
                        score_high_half = vector_registers.vselect(
                            score_high_half, negative_sentinel, cond_mask=visible1
                        )  # 当前行后64列分数。

                pair_max = vector_registers.vmax(
                    score_low_half, score_high_half, mask=full
                )  # 两个64列片段逐通道合并的最大值。
                physical_cache_page_index = vector_registers.vmax(
                    negative_sentinel, vector_registers.vreduce_max(pair_max, mask=full), mask=full
                )  # 当前行的块内最大值。

                physical_cache_page_index = vector_registers.vdup(
                    physical_cache_page_index, mask=full
                )  # 当前行的块内最大值。
                state_offset = state_base + tile_row_index  # 当前任务在线状态偏移。
                previous_maximum = zero  # 上一有效块的在线最大值。
                normalization_sum = zero  # 在线softmax分母。
                if const_expr(not is_first_kv_block):
                    previous_maximum = vector_registers.vload_broadcast(
                        maximum, state_offset
                    )  # 上一有效块的在线最大值；从片上存储读取当前计算需要的数据。
                    normalization_sum = vector_registers.vload_broadcast(
                        denominator, state_offset
                    )  # 在线softmax分母；从片上存储读取当前计算需要的数据。
                updated = physical_cache_page_index  # 当前块合并后的最大值。
                factor = zero  # 本行或本批次在线缩放因子。
                if const_expr(not is_first_kv_block):
                    if const_expr(dense):
                        updated = vector_registers.vmax(
                            previous_maximum, physical_cache_page_index, mask=full
                        )  # 当前块合并后的最大值。
                        factor = vector_registers.vexp_sub(
                            previous_maximum, updated, mask=full
                        )  # 本行或本批次在线缩放因子。
                    else:
                        history = vector_registers.vgts(
                            normalization_sum, 0.0, mask=full
                        )  # 当前行已有有效历史分母的标记。
                        updated = vector_registers.vselect(
                            vector_registers.vmax(previous_maximum, physical_cache_page_index, mask=full),
                            physical_cache_page_index,
                            cond_mask=history,
                        )  # 当前块合并后的最大值。
                        factor = vector_registers.vselect(
                            vector_registers.vexp_sub(previous_maximum, updated, mask=full), zero, cond_mask=history
                        )  # 本行或本批次在线缩放因子。
                if const_expr(dense):
                    probability_low_half = vector_registers.vexp_sub(
                        score_low_half, updated, mask=full
                    )  # 当前行前64列的未归一化概率。
                    probability_high_half = vector_registers.vexp_sub(
                        score_high_half, updated, mask=full
                    )  # 当前行后64列的未归一化概率。
                else:
                    if const_expr(not uniform_prefix_mask):
                        updated = vector_registers.vselect(
                            updated, previous_maximum, cond_mask=nonempty
                        )  # 当前块合并后的最大值。
                        factor = vector_registers.vselect(factor, one, cond_mask=nonempty)  # 本行或本批次在线缩放因子。
                    probability_low_half = vector_registers.vexp_sub(
                        score_low_half, updated, mask=full
                    )  # 当前行前64列的未归一化概率。
                    if const_expr(not specialize_prefix_halves or not prefix_within_low_half):
                        probability_high_half = vector_registers.vexp_sub(
                            score_high_half, updated, mask=full
                        )  # 当前行后64列的未归一化概率。
                    else:
                        probability_high_half = zero
                    if const_expr(not specialize_prefix_halves or prefix_within_low_half):
                        probability_low_half = vector_registers.vselect(
                            probability_low_half, zero, cond_mask=visible0
                        )  # 当前行前64列的未归一化概率。
                    if const_expr(not specialize_prefix_halves or not prefix_within_low_half):
                        probability_high_half = vector_registers.vselect(
                            probability_high_half, zero, cond_mask=visible1
                        )  # 当前行后64列的未归一化概率。
                total = vector_registers.vmul(
                    normalization_sum, factor, mask=full
                )  # 由函数上下文决定的分母累计值或总token数。
                combined_probability = vector_registers.vadd(probability_low_half, probability_high_half, mask=full)
                total = vector_registers.vadd(
                    total, vector_registers.vreduce_sum(combined_probability, mask=full), mask=full
                )  # 两半概率先配对，再进行一次横向求和。
                vector_registers.vstore_first(maximum, state_offset, updated)
                vector_registers.vstore_first(denominator, state_offset, total)
                vector_registers.vstore_first(alpha, tile_row_index, factor)

                packed_probability_low = vector_registers.vcast(
                    probability_low_half, dtypes.bfloat16, mask=full
                )  # 前64列概率转换并压紧后的BF16数据。
                if const_expr(not specialize_prefix_halves or not prefix_within_low_half):
                    packed_probability_high = vector_registers.vcast(
                        probability_high_half, dtypes.bfloat16, mask=full
                    )  # 后64列概率转换并压紧后的BF16数据。
                else:
                    packed_probability_high = packed_high_zero
                packed_probability_row, _ = vector_registers.vdeinterleave(
                    packed_probability_low, packed_probability_high
                )  # 合并两半BF16有效槽，生成完整128列概率。
                stride = prob.physical_layout.stride[0]  # NZ列组之间的物理间距，单位为BF16元素。
                vector_registers.vstore_strided(
                    prob,
                    tile_row_index * NZ_BLOCK_EDGE,
                    packed_probability_row,
                    packed_mask,
                    block_stride=stride // NZ_BLOCK_EDGE,
                    repeat_stride=0,
                )

    @jit
    def softmax_dense_batched(
        self,
        scores: Tensor,
        prob: Tensor,
        maximum: Tensor,
        denominator: Tensor,
        alpha: Tensor,
        scratch: Tensor,
        query_task_sequence: int,
        scale: float,
        is_first_kv_block: bool,
        hoist: bool,
    ):
        """先收集各行最大值，批量更新在线状态，再逐行生成概率及两半配对归约和，最后批量更新分母。

        Args:
            scores: QK分数池或分数视图。
            prob: 待发布的BF16概率视图。
            maximum: 在线最大值存储。
            denominator: 在线分母存储。
            alpha: 在线最大值变化的缩放因子池。
            scratch: UB 临时缓冲，保存块最大值和两半概率配对后的行归约和。
            query_task_sequence: 当前查询任务序号。
            scale: 分数缩放因子。
            is_first_kv_block: 首个有效块标记。
            hoist: 首遍最大值扫描采用scale外提标记。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        scratch_sum_low_begin = 64  # 沿用原low平面存储配对归约和，偏移以FP32元素计。
        maximum_row_unroll = 2  # 最大值扫描独立使用较小的循环展开提示，减少地址控制开销。
        probability_row_unroll = 4  # 概率计算与NZ打包保留原四行展开提示。
        with vf(mode="simd"):
            full, _ = vector_registers.update_mask(
                VECTOR_FLOAT32_LANES, elem_bits=FLOAT32_BITS
            )  # 全部有效向量通道的执行掩码；调用方无需使用的返回项。
            packed_mask, _ = vector_registers.update_mask(
                VECTOR_BF16_LANES, elem_bits=BF16_BITS
            )  # BF16打包存储的有效通道掩码；调用方无需使用的返回项。
            negative_sentinel = vector_registers.vdups(
                MASKED_SCORE_FLOOR, dtypes.float32, mask=full
            )  # 被遮蔽分数使用的有限负哨兵。
            zero = vector_registers.vdups(0.0, dtypes.float32, mask=full)  # 零值向量。
            state_base = (
                query_task_sequence % ONLINE_STATE_SLOTS * VECTOR_QUERY_ROWS
            )  # 当前查询任务对应的在线状态槽起点。
            for tile_row_index in dsl_range(
                0, scores.shape[0], 1, unroll=maximum_row_unroll
            ):  # 当前AIV块内的行号；按给定区间依次遍历当前处理项。
                score_low_half = vector_registers.vload(
                    scores, tile_row_index * DEFAULT_KV_TILE_COLUMNS
                )  # 当前行前64列分数；从片上存储读取当前计算需要的数据。
                score_high_half = vector_registers.vload(
                    scores, tile_row_index * DEFAULT_KV_TILE_COLUMNS + VECTOR_FLOAT32_LANES
                )  # 当前行后64列分数；从片上存储读取当前计算需要的数据。
                if const_expr(not hoist):
                    score_low_half = vector_registers.vmuls(score_low_half, scale, mask=full)  # 当前行前64列分数。
                    score_high_half = vector_registers.vmuls(score_high_half, scale, mask=full)  # 当前行后64列分数。
                physical_cache_page_index = vector_registers.vreduce_max(
                    vector_registers.vmax(score_low_half, score_high_half, mask=full), mask=full
                )  # 当前行的块内最大值。
                if const_expr(not hoist):
                    physical_cache_page_index = vector_registers.vmax(
                        negative_sentinel, physical_cache_page_index, mask=full
                    )  # 当前行的块内最大值。
                vector_registers.vstore_first(scratch, tile_row_index, physical_cache_page_index)
            vector_registers.vmem_bar("vst_vld")
            blocks = vector_registers.vload(scratch, 0)  # 64行块内最大值构成的向量；从片上存储读取当前计算需要的数据。
            if const_expr(hoist):
                blocks = vector_registers.vmax(
                    negative_sentinel, vector_registers.vmuls(blocks, scale, mask=full), mask=full
                )  # 64行块内最大值构成的向量。
            old_den = zero  # 此前有效块累计的归一化分母。
            updated = blocks  # 当前块合并后的最大值。
            factor = zero  # 本行或本批次在线缩放因子。
            if const_expr(not is_first_kv_block):
                previous_maximum = vector_registers.vload(
                    maximum, state_base
                )  # 上一有效块的在线最大值；从片上存储读取当前计算需要的数据。
                old_den = vector_registers.vload(
                    denominator, state_base
                )  # 此前有效块累计的归一化分母；从片上存储读取当前计算需要的数据。
                updated = vector_registers.vmax(previous_maximum, blocks, mask=full)  # 当前块合并后的最大值。
                factor = vector_registers.vexp_sub(previous_maximum, updated, mask=full)  # 本行或本批次在线缩放因子。
            vector_registers.vstore(maximum, state_base, updated, full)
            vector_registers.vstore(alpha, 0, factor, full)
            vector_registers.vmem_bar("vst_vld")
            for tile_row_index in dsl_range(
                0, scores.shape[0], 1, unroll=probability_row_unroll
            ):  # 当前AIV块内的行号；按给定区间依次遍历当前处理项。
                score_low_half = vector_registers.vmuls(
                    vector_registers.vload(scores, tile_row_index * DEFAULT_KV_TILE_COLUMNS), scale, mask=full
                )  # 当前行前64列分数。
                score_high_half = vector_registers.vmuls(
                    vector_registers.vload(scores, tile_row_index * DEFAULT_KV_TILE_COLUMNS + VECTOR_FLOAT32_LANES),
                    scale,
                    mask=full,
                )  # 当前行后64列分数。
                row_max = vector_registers.vload_broadcast(
                    maximum, state_base + tile_row_index
                )  # 当前行广播后的块内最大值；从片上存储读取当前计算需要的数据。
                probability_low_half = vector_registers.vexp_sub(
                    score_low_half, row_max, mask=full
                )  # 当前行前64列的未归一化概率。
                probability_high_half = vector_registers.vexp_sub(
                    score_high_half, row_max, mask=full
                )  # 当前行后64列的未归一化概率。
                packed_probability_low = vector_registers.vcast(
                    probability_low_half, dtypes.bfloat16, mask=full
                )  # 前64列概率转换并压紧后的BF16数据。
                packed_probability_high = vector_registers.vcast(
                    probability_high_half, dtypes.bfloat16, mask=full
                )  # 后64列概率转换并压紧后的BF16数据。
                packed_probability_row, _ = vector_registers.vdeinterleave(
                    packed_probability_low, packed_probability_high
                )  # 提取两半BF16有效槽，直接合成128列连续概率。
                stride = prob.physical_layout.stride[0]  # NZ列组之间的物理间距，单位为BF16元素。
                vector_registers.vstore_strided(
                    prob,
                    tile_row_index * NZ_BLOCK_EDGE,
                    packed_probability_row,
                    packed_mask,
                    block_stride=stride // NZ_BLOCK_EDGE,
                    repeat_stride=0,
                )
                combined_probability = vector_registers.vadd(probability_low_half, probability_high_half, mask=full)
                vector_registers.vstore_first(
                    scratch, scratch_sum_low_begin + tile_row_index,
                    vector_registers.vreduce_sum(combined_probability, mask=full),
                )
            vector_registers.vmem_bar("vst_vld")
            total = vector_registers.vmul(old_den, factor, mask=full)  # 由函数上下文决定的分母累计值或总token数。
            total = vector_registers.vadd(
                total, vector_registers.vload(scratch, scratch_sum_low_begin), mask=full
            )  # 由函数上下文决定的分母累计值或总token数。
            vector_registers.vstore(denominator, state_base, total, full)

    @jit
    def zero_probability_tile(
        self,
        scores: Tensor,
        prob: Tensor,
        maximum: Tensor,
        denominator: Tensor,
        alpha: Tensor,
        is_first_kv_block: bool,
        query_task_sequence: int,
    ):
        """消费真实分数并写零概率和单位缩放，保持流水依赖及空块数学语义。

        Args:
            scores: QK分数池或分数视图。
            prob: 待发布的BF16概率视图。
            maximum: 在线最大值存储。
            denominator: 在线分母存储。
            alpha: 在线最大值变化的缩放因子池。
            is_first_kv_block: 首个KV块的编译期标记。
            query_task_sequence: 当前查询任务序号。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        query_row_unroll = 4  # 本函数使用的query_row_unroll编译期配置。
        probability_group_stride = prob.physical_layout.stride[0] // NZ_BLOCK_EDGE  # NZ列组跨度，单位为32字节块。
        with vf(mode="simd"):
            full, _ = vector_registers.update_mask(
                VECTOR_FLOAT32_LANES, elem_bits=FLOAT32_BITS
            )  # 全部有效向量通道的执行掩码；调用方无需使用的返回项。
            packed, _ = vector_registers.update_mask(
                VECTOR_BF16_LANES, elem_bits=BF16_BITS
            )  # 转换为BF16并准备输出的向量片段；调用方无需使用的返回项。
            zero = vector_registers.vdups(0.0, dtypes.float32, mask=full)  # 零值向量。
            vector_registers.vstore(alpha, 0, vector_registers.vdups(1.0, dtypes.float32, mask=full), full)
            if const_expr(is_first_kv_block):
                vector_registers.vstore(
                    maximum, query_task_sequence % ONLINE_STATE_SLOTS * VECTOR_QUERY_ROWS, zero, full
                )
                vector_registers.vstore(
                    denominator, query_task_sequence % ONLINE_STATE_SLOTS * VECTOR_QUERY_ROWS, zero, full
                )

            seed = vector_registers.vreinterpret(
                vector_registers.vload(scores, 0), dtypes.uint32
            )  # 用于构造精确零输出的依赖来源。
            bits = vector_registers.vbitwise_and(
                seed, vector_registers.vdups(0, dtypes.uint32, mask=full), mask=full
            )  # 真实输入数据的整数位模式。
            zeros = vector_registers.vreinterpret_lanes(bits, dtypes.bfloat16)  # 用于产生精确零值的向量。
            for probability_row in dsl_range(
                0, prob.shape[0], 1, unroll=query_row_unroll
            ):  # 逐行覆盖全部128个逻辑概率元素，跨越NZ列组之间的padding。
                vector_registers.vstore_strided(
                    prob, probability_row * NZ_BLOCK_EDGE, zeros, packed,
                    block_stride=probability_group_stride, repeat_stride=0,
                )

    @jit
    def consume_masked_update(
        self,
        output_accumulator: Tensor,
        value: Tensor,
        alpha: Tensor,
    ):
        """消费PV和缩放因子，以整数位运算保持累加器原位模式，避免无效浮点乘加。

        Args:
            output_accumulator: 在线输出累加器。
            value: 值张量或当前 PV 结果视图；具体布局由所在计算阶段决定。
            alpha: 在线最大值变化的缩放因子池。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        with vf(mode="simd"):
            full, _ = vector_registers.update_mask(
                VECTOR_FLOAT32_LANES, elem_bits=FLOAT32_BITS
            )  # 全部有效向量通道的执行掩码；调用方无需使用的返回项。
            zero = vector_registers.vdups(0, dtypes.uint32, mask=full)  # 零值向量。
            pv_bits = vector_registers.vreinterpret(
                vector_registers.vload(value, 0), dtypes.uint32
            )  # PV消费者数据的整数位模式。
            alpha_bits = vector_registers.vreinterpret(
                vector_registers.vload(alpha, 0), dtypes.uint32
            )  # 缩放因子的整数位模式。
            cleared = vector_registers.vbitwise_or(
                vector_registers.vbitwise_and(pv_bits, zero, mask=full),
                vector_registers.vbitwise_and(alpha_bits, zero, mask=full),
                mask=full,
            )  # 真实消费者数据按位清零后的依赖值。
            previous_maximum = vector_registers.vreinterpret(
                vector_registers.vload(output_accumulator, 0), dtypes.uint32
            )  # 原累加器的FP32整数位模式。
            same = vector_registers.vbitwise_or(previous_maximum, cleared, mask=full)  # 保持原累加器位模式的结果。
            vector_registers.vstore(output_accumulator, 0, vector_registers.vreinterpret(same, dtypes.float32), full)

    @jit
    def update_tile(
        self,
        output_accumulator: Tensor,
        value: Tensor,
        alpha: Tensor,
        denominator: Tensor,
        out: Tensor,
        query_task_sequence: int,
        is_first_kv_block: bool,
        is_last_kv_block: bool,
    ):
        """按首块与末块的编译期分支更新输出累加器，末块执行安全归一化和BF16转换。

        Args:
            output_accumulator: 在线输出累加器。
            value: 值张量或当前 PV 结果视图；具体布局由所在计算阶段决定。
            alpha: 在线最大值变化的缩放因子池。
            denominator: 在线分母存储。
            out: 最终输出暂存或设备输出视图。
            query_task_sequence: 当前查询任务序号。
            is_first_kv_block: 首个有效块标记。
            is_last_kv_block: 末个有效块标记。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        query_row_unroll = 4  # 本函数使用的query_row_unroll编译期配置。
        output_column_unroll = 2  # 本函数使用的output_column_unroll编译期配置。
        with vf(mode="simd"):
            full, _ = vector_registers.update_mask(
                min(output_accumulator.shape[1], VECTOR_FLOAT32_LANES), elem_bits=FLOAT32_BITS
            )  # 全部有效向量通道的执行掩码；调用方无需使用的返回项。
            if const_expr(is_last_kv_block):
                one = vector_registers.vdups(1.0, dtypes.float32, mask=full)  # 单位值向量。
                zero = vector_registers.vdups(0.0, dtypes.float32, mask=full)  # 零值向量。
                state_base = (
                    query_task_sequence % ONLINE_STATE_SLOTS * VECTOR_QUERY_ROWS
                )  # 当前查询任务对应的在线状态槽起点。
            for tile_row_index in dsl_range(
                0, output_accumulator.shape[0], 1, unroll=query_row_unroll
            ):  # 当前AIV块内的行号；按给定区间依次遍历当前处理项。
                if const_expr(not is_first_kv_block):
                    factor = vector_registers.vload_broadcast(
                        alpha, tile_row_index
                    )  # 本行或本批次在线缩放因子；从片上存储读取当前计算需要的数据。
                if const_expr(is_last_kv_block):
                    normalization_sum = vector_registers.vload_broadcast(
                        denominator, state_base + tile_row_index
                    )  # 在线softmax分母；从片上存储读取当前计算需要的数据。
                    has_denominator = vector_registers.vgts(
                        normalization_sum, 0.0, mask=full
                    )  # 归一化分母大于零的通道掩码。
                    safe = vector_registers.vselect(
                        normalization_sum, one, cond_mask=has_denominator
                    )  # 用单位值替换零分母后的安全除数。
                for column_index in dsl_range(
                    0, output_accumulator.shape[1], VECTOR_FLOAT32_LANES, unroll=output_column_unroll
                ):  # 当前输出列片段的起点；按给定区间依次遍历当前处理项。
                    mask, _ = vector_registers.update_mask(
                        output_accumulator.shape[1] - column_index, elem_bits=FLOAT32_BITS
                    )  # 当前向量片段有效通道掩码；调用方无需使用的返回项。
                    offset = tile_row_index * output_accumulator.shape[1] + column_index  # 当前行列对应的线性存储偏移。
                    current_output = vector_registers.vload(
                        value, tile_row_index * value.physical_layout.stride[0] + column_index
                    )  # 当前输出片段的FP32中间值；从片上存储读取当前计算需要的数据。
                    if const_expr(not is_first_kv_block):
                        current_output = vector_registers.vmadd(
                            vector_registers.vload(output_accumulator, offset), factor, current_output, mask=mask
                        )  # 当前输出片段的FP32中间值。
                    if const_expr(is_last_kv_block):
                        current_output = vector_registers.vdiv(
                            current_output, safe, mask=mask
                        )  # 当前输出片段的FP32中间值。
                        current_output = vector_registers.vselect(
                            current_output, zero, cond_mask=has_denominator
                        )  # 当前输出片段的FP32中间值。
                        packed = vector_registers.vcast(
                            current_output, dtypes.bfloat16, mask=mask
                        )  # 转换为BF16并准备输出的向量片段。
                        vector_registers.vstore_pack(
                            out, offset, packed, mask, pack_mode=vector_registers.PackMode.B32_TO_B16
                        )
                    else:
                        vector_registers.vstore(output_accumulator, offset, current_output, mask)

    @jit
    def prepare_rows(
        self,
        state: _MtgrPipelineResources,
        offsets: Tensor,
        rules: Tensor,
        batch_index: int,
        query_row_start: int,
        cached_prefix_tokens: int,
        live_sequence_length: int,
    ):
        """仅在查询任务开始时计算每行掩码规则及半块可见范围，供所有键值块复用。

        Args:
            state: 共享片上资源状态。
            offsets: 各请求分段偏移。
            rules: 各段掩码规则。
            batch_index: 请求编号。
            query_row_start: 查询块起点。
            cached_prefix_tokens: 当前请求的命中前缀长度。
            live_sequence_length: 当前请求的实时序列长度。

        Returns:
            tuple[int, int, int, int]: 本AIV的前段及对角范围。
        """
        binary_search_divisor = 2  # 二分区间折半因子。
        query_logical_begin = (
            cached_prefix_tokens + query_row_start + state.vector_subblock_index * state.vector_query_rows
        )  # 当前查询块的逻辑起始位置。
        query_logical_end = min(
            query_logical_begin + state.vector_query_rows, cached_prefix_tokens + live_sequence_length
        )  # 当前查询块的逻辑结束位置。
        is_first_segment_intersection = 1  # 首个有效块标记。
        cutoff_min = cached_prefix_tokens + live_sequence_length  # 本AIV所有有效行的连续前段最小结束位置。
        cutoff_max = 0  # 本AIV可见连续前段的最大结束位置。
        diagonal_min = cached_prefix_tokens + live_sequence_length  # 当前AIV半块最小对角位置。
        diagonal_max = dtypes.int64(NO_DIAGONAL_TOKEN)  # 当前AIV半块最大对角位置。
        search_lower = 0  # 段二分查找的下界。
        search_upper = rules.shape[0]  # 段二分查找的上界。
        while search_lower < search_upper:
            search_middle = (search_lower + search_upper) // binary_search_divisor  # 段二分查找的中点。
            if query_logical_begin >= offsets[batch_index, search_middle + 1]:
                search_lower = search_middle + 1  # 段二分查找的下界。
            else:
                search_upper = search_middle  # 段二分查找的上界。
        while search_lower < rules.shape[0] and offsets[batch_index, search_lower] < query_logical_end:
            segment_kv_begin = offsets[batch_index, search_lower]  # 当前段的逻辑起点。
            segment_kv_end = offsets[batch_index, search_lower + 1]  # 当前区间的结束位置。
            segment_mask_rule = rules[search_lower]  # 当前段的掩码规则编号。
            low = dtypes.int64(segment_kv_begin)  # 当前范围下界的临时值。
            high = dtypes.int64(min(query_logical_end, segment_kv_end))  # 当前段对连续前段上界的贡献。
            if segment_mask_rule == MASK_CAUSAL:
                low = max(query_logical_begin, segment_kv_begin) + 1  # 当前范围下界的临时值。
                self.cache_row_span(
                    state.row_params,
                    query_logical_begin,
                    segment_kv_begin,
                    segment_kv_end,
                    MASK_CAUSAL,
                    is_first_segment_intersection,
                )
            elif segment_mask_rule == MASK_SEGMENT_FULL:
                low = dtypes.int64(segment_kv_end)  # 当前范围下界的临时值。
                high = dtypes.int64(segment_kv_end)  # 当前段对连续前段上界的贡献。
                self.cache_row_span(
                    state.row_params,
                    query_logical_begin,
                    segment_kv_begin,
                    segment_kv_end,
                    MASK_SEGMENT_FULL,
                    is_first_segment_intersection,
                )
            else:
                high = dtypes.int64(segment_kv_begin)  # 当前段对连续前段上界的贡献。
                diagonal_min = min(
                    diagonal_min, max(query_logical_begin, segment_kv_begin)
                )  # 当前AIV半块最小对角位置。
                diagonal_max = max(
                    diagonal_max, min(query_logical_end, segment_kv_end) - 1
                )  # 当前AIV半块最大对角位置。
                self.cache_row_span(
                    state.row_params,
                    query_logical_begin,
                    segment_kv_begin,
                    segment_kv_end,
                    MASK_PREFIX_AND_SELF,
                    is_first_segment_intersection,
                )
            cutoff_min = min(cutoff_min, low)  # 本AIV所有有效行的连续前段最小结束位置。
            cutoff_max = max(cutoff_max, high)  # 本AIV可见连续前段的最大结束位置。
            is_first_segment_intersection = 0  # 首个有效块标记。
            search_lower += 1  # 段二分查找的下界。
        if is_first_segment_intersection != 0:
            self.cache_row_span(
                state.row_params, query_logical_begin, query_logical_begin, query_logical_begin, MASK_PREFIX_AND_SELF, 1
            )
            cutoff_min = 0  # 本AIV所有有效行的连续前段最小结束位置。
        return cutoff_min, cutoff_max, diagonal_min, diagonal_max

    @jit
    def softmax_and_publish_probability(
        self,
        state: _MtgrPipelineResources,
        key_column_start: int,
        query_task_sequence: int,
        scale: float,
        cutoff_min: int,
        cutoff_max: int,
        diagonal_min: int,
        diagonal_max: int,
        masked: int,
        full_query_rows: bool = False,
    ):
        """区分全遮蔽、密集和普通掩码路径，生成概率、缩放和分母状态，并将概率搬入L1。

        Args:
            state: 共享片上资源状态。
            key_column_start: 逻辑键值块起点。
            query_task_sequence: 当前查询任务序号。
            scale: 分数缩放因子。
            cutoff_min: 本AIV所有有效行的连续前段最小结束位置。
            cutoff_max: 本AIV所有行连续前段结束位置的最大值。
            diagonal_min: 本AIV有效对角位置的最小值。
            diagonal_max: 本AIV有效对角位置的最大值。
            masked: 当前AIV半块全遮蔽标记。
            full_query_rows: 当前AIV的64个查询行均有效，用于保护共享前段掩码特化。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        scores = state.scores.next_read()  # QK分数池或分数视图；取得已就绪的待消费槽。
        prob = state.probability_nz.next_write()  # 待发布的BF16概率视图；取得受生产消费依赖保护的待写槽。
        factors = state.alpha.next_write()  # 当前键值块的在线缩放因子写入或读取视图；取得受生产消费依赖保护的待写槽。
        if masked != 0:
            if key_column_start == 0:
                self.zero_probability_tile(
                    scores, prob, state.maximum, state.normalization_sum, factors, True, query_task_sequence
                )
            else:
                self.zero_probability_tile(
                    scores, prob, state.maximum, state.normalization_sum, factors, False, query_task_sequence
                )
        elif key_column_start + state.key_tile_columns <= cutoff_min:
            if key_column_start == 0:
                if scale > 0.0 and scale <= FLOAT32_MAX_VALUE:
                    self.softmax_dense_batched(
                        scores,
                        prob,
                        state.maximum,
                        state.normalization_sum,
                        factors,
                        state.softmax_scratch,
                        query_task_sequence,
                        scale,
                        True,
                        True,
                    )
                else:
                    self.softmax_dense_batched(
                        scores,
                        prob,
                        state.maximum,
                        state.normalization_sum,
                        factors,
                        state.softmax_scratch,
                        query_task_sequence,
                        scale,
                        True,
                        False,
                    )
            else:
                if scale > 0.0 and scale <= FLOAT32_MAX_VALUE:
                    self.softmax_dense_batched(
                        scores,
                        prob,
                        state.maximum,
                        state.normalization_sum,
                        factors,
                        state.softmax_scratch,
                        query_task_sequence,
                        scale,
                        False,
                        True,
                    )
                else:
                    self.softmax_dense_batched(
                        scores,
                        prob,
                        state.maximum,
                        state.normalization_sum,
                        factors,
                        state.softmax_scratch,
                        query_task_sequence,
                        scale,
                        False,
                        False,
                    )
        elif key_column_start >= cutoff_max and scale > 0.0 and scale <= FLOAT32_MAX_VALUE:
            if key_column_start == 0:
                self.softmax_diagonal_batched(
                    scores, prob, state.maximum, state.normalization_sum, factors, state.row_params,
                    key_column_start, query_task_sequence, scale, True,
                )
            else:
                self.softmax_diagonal_batched(
                    scores, prob, state.maximum, state.normalization_sum, factors, state.row_params,
                    key_column_start, query_task_sequence, scale, False,
                )
        else:
            if key_column_start + state.key_tile_columns <= diagonal_min or key_column_start > diagonal_max:
                if full_query_rows and cutoff_min == cutoff_max and key_column_start < cutoff_min:
                    if cutoff_max - key_column_start <= VECTOR_FLOAT32_LANES:
                        if key_column_start == 0:
                            self.softmax_tile(
                                scores,
                                prob,
                                state.maximum,
                                state.normalization_sum,
                                factors,
                                state.row_params,
                                key_column_start,
                                query_task_sequence,
                                scale,
                                False,
                                True,
                                False,
                                True,
                                True,
                                True,
                            )
                        else:
                            self.softmax_tile(
                                scores,
                                prob,
                                state.maximum,
                                state.normalization_sum,
                                factors,
                                state.row_params,
                                key_column_start,
                                query_task_sequence,
                                scale,
                                False,
                                False,
                                False,
                                True,
                                True,
                                True,
                            )
                    else:
                        if key_column_start == 0:
                            self.softmax_tile(
                                scores,
                                prob,
                                state.maximum,
                                state.normalization_sum,
                                factors,
                                state.row_params,
                                key_column_start,
                                query_task_sequence,
                                scale,
                                False,
                                True,
                                False,
                                True,
                                True,
                                False,
                            )
                        else:
                            self.softmax_tile(
                                scores,
                                prob,
                                state.maximum,
                                state.normalization_sum,
                                factors,
                                state.row_params,
                                key_column_start,
                                query_task_sequence,
                                scale,
                                False,
                                False,
                                False,
                                True,
                                True,
                                False,
                            )
                elif key_column_start == 0:
                    self.softmax_tile(
                        scores,
                        prob,
                        state.maximum,
                        state.normalization_sum,
                        factors,
                        state.row_params,
                        key_column_start,
                        query_task_sequence,
                        scale,
                        False,
                        True,
                        False,
                    )
                else:
                    self.softmax_tile(
                        scores,
                        prob,
                        state.maximum,
                        state.normalization_sum,
                        factors,
                        state.row_params,
                        key_column_start,
                        query_task_sequence,
                        scale,
                        False,
                        False,
                        False,
                    )
            else:
                if key_column_start == 0:
                    self.softmax_tile(
                        scores,
                        prob,
                        state.maximum,
                        state.normalization_sum,
                        factors,
                        state.row_params,
                        key_column_start,
                        query_task_sequence,
                        scale,
                        False,
                        True,
                        True,
                    )
                else:
                    self.softmax_tile(
                        scores,
                        prob,
                        state.maximum,
                        state.normalization_sum,
                        factors,
                        state.row_params,
                        key_column_start,
                        query_task_sequence,
                        scale,
                        False,
                        False,
                        True,
                    )
        mem_copy(
            state.probability_l1.next_write().partition(dim=0, part_id=state.vector_subblock_index),
            state.probability_nz.next_read(),
        )

    @jit
    def update_and_store_output(
        self,
        state: _MtgrPipelineResources,
        output: Tensor,
        sequence_gm_token_begin: int,
        query_head_index: int,
        query_row_start: int,
        key_column_start: int,
        live_sequence_length: int,
        logical_kv_length: int,
        query_task_sequence: int,
        masked: int,
    ):
        """匹配延迟任务的PV与状态，旁路可省略的中间空块，末块按有效行数写回输出。

        Args:
            state: 共享片上资源状态。
            output: 设备输出张量。
            sequence_gm_token_begin: 当前请求的实时token起点。
            query_head_index: 查询头编号。
            query_row_start: 查询块起点。
            key_column_start: 逻辑键值块起点。
            live_sequence_length: 当前请求的实时序列长度。
            logical_kv_length: 含前缀的逻辑序列长度。
            query_task_sequence: 当前查询任务序号。
            masked: 当前AIV半块全遮蔽标记。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        live_head_axis = 1  # 本函数使用的live_head_axis编译期配置。
        live_token_axis = 0  # 本函数使用的live_token_axis编译期配置。
        live_feature_axis = 2  # 本函数使用的live_feature_axis编译期配置。
        value = state.weighted_values.next_read()  # 当前值张量或向量片段，具体由函数上下文确定；取得已就绪的待消费槽。
        factors = state.alpha.next_read()  # 当前键值块的在线缩放因子写入或读取视图；取得已就绪的待消费槽。
        is_first_kv_block = 0  # 首个有效块标记。
        is_last_kv_block = 0  # 末个有效块标记。
        if key_column_start == 0:
            is_first_kv_block = 1  # 首个有效块标记。
        if key_column_start + state.key_tile_columns >= logical_kv_length:
            is_last_kv_block = 1  # 末个有效块标记。
        if is_first_kv_block == 0 and is_last_kv_block == 0 and masked != 0:
            self.consume_masked_update(state.output_accumulator, value, factors)
        elif is_last_kv_block != 0:
            if is_first_kv_block != 0:
                self.update_tile(
                    state.output_accumulator,
                    value,
                    factors,
                    state.normalization_sum,
                    state.out,
                    query_task_sequence,
                    True,
                    True,
                )
            else:
                self.update_tile(
                    state.output_accumulator,
                    value,
                    factors,
                    state.normalization_sum,
                    state.out,
                    query_task_sequence,
                    False,
                    True,
                )
        elif is_first_kv_block != 0:
            self.update_tile(
                state.output_accumulator,
                value,
                factors,
                state.normalization_sum,
                state.out,
                query_task_sequence,
                True,
                False,
            )
        else:
            self.update_tile(
                state.output_accumulator,
                value,
                factors,
                state.normalization_sum,
                state.out,
                query_task_sequence,
                False,
                False,
            )
        if is_last_kv_block != 0:
            query_position = (
                query_row_start + state.vector_subblock_index * state.vector_query_rows
            )  # 当前查询行在请求内的位置。
            rows = min(state.vector_query_rows, live_sequence_length - query_position)  # 当前块中的有效行数。
            if rows == state.vector_query_rows:
                destination = output.permute(live_head_axis, live_token_axis, live_feature_axis)[
                    query_head_index,
                    sequence_gm_token_begin + query_position : sequence_gm_token_begin + query_position + rows,
                    None,
                ]  # 当前搬运目标视图。
                destination = dsl.tile_slice(destination, state.out.shape, (0, 0))  # 当前搬运目标视图。
                mem_copy(destination, state.out)
            else:
                for tile_row_index in range(max(rows, 0)):  # 当前AIV块内的行号；按给定区间依次遍历当前处理项。
                    mem_copy(
                        row3(output, sequence_gm_token_begin + query_position + tile_row_index, query_head_index),
                        onchip_row(state.out, tile_row_index),
                    )


@kernel
class MtgrRaggedSegmentAttention:
    """串联单次 attention 主 kernel 中的 Cube、Vector 和搬运流水。

    __call__ 读取 metadata 并遍历 section 和核内查询任务，调用 cube_visible_bounds 筛选
    可见键值块，再交错执行 QK、softmax、PV 和输出更新，结束时排空延迟任务。

    Attributes:
        query_tile_rows: 完整 Cube 查询块的行数。
        key_tile_columns: 每个键值块的列数。
    """
    def __init__(
        self,
        query_tile_rows: int,
        key_tile_columns: int,
    ):
        """保存查询和键值分块规格。

        Args:
            query_tile_rows: 完整 Cube 查询块的行数。
            key_tile_columns: 每个键值块的列数。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        self.query_tile_rows, self.key_tile_columns = (
            query_tile_rows,
            key_tile_columns,
        )  # 查询分块行数；键值分块列数。

    def __call__(
        self,
        query_tensor: Tensor,
        key_tensor: Tensor,
        value_tensor: Tensor,
        key_cache_tensor: Tensor,
        value_cache_tensor: Tensor,
        kv_page_table: Tensor,
        offsets: Tensor,
        rules: Tensor,
        starts: Tensor,
        prefixes: Tensor,
        metadata: Tensor,
        out: Tensor,
        scale: float,
    ):
        """校验metadata头，逐section读取本核任务范围，裁剪无效键值块并串联Cube与Vector流水，最后排空任务。

        Args:
            query_tensor: 查询张量。
            key_tensor: 键张量。
            value_tensor: 值张量。
            key_cache_tensor: 分页键缓存张量。
            value_cache_tensor: 分页值缓存张量。
            kv_page_table: 逻辑页到物理页的映射。
            offsets: 各请求分段偏移。
            rules: 各段掩码规则。
            starts: 各请求实时token起点。
            prefixes: 各请求命中前缀长度。
            metadata: 设备侧 INT32 分核计划；允许为 None 的入口会先生成计划。
            out: 最终输出暂存或设备输出视图。
            scale: 分数缩放因子。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        tensor_feature_axis = 2  # 本函数使用的tensor_feature_axis编译期配置。
        packet_gm_begin_field = 1  # 本函数使用的packet_gm_begin_field编译期配置。
        packet_kv_column_field = 4  # 本函数使用的packet_kv_column_field编译期配置。
        packet_prefix_field = 5  # 本函数使用的packet_prefix_field编译期配置。
        packet_live_length_field = 6  # 本函数使用的packet_live_length_field编译期配置。
        packet_query_head_field = 2  # 本函数使用的packet_query_head_field编译期配置。
        packet_query_row_field = 3  # 本函数使用的packet_query_row_field编译期配置。
        packet_active_stop_field = 7  # 本函数使用的packet_active_stop_field编译期配置。
        packet_query_sequence_field = 8  # 本函数使用的packet_query_sequence_field编译期配置。
        packet_masked_field = 9  # 本函数使用的packet_masked_field编译期配置。
        task_packet_storage_words = 40  # 延迟任务队列的总字段容量。
        query_l1_reuse_interval = 2  # 双槽Q缓冲允许相邻两项查询任务交错。
        pipeline_delay_ticks = 2  # PV/输出阶段相对QK的延迟节拍。
        task_packet_words = 10  # 每条延迟任务记录的字段数。
        task_packet_slots = 4  # 延迟任务队列槽数。
        pipeline_resources = _MtgrPipelineResources(
            self.query_tile_rows, self.key_tile_columns, query_tensor.shape[tensor_feature_axis]
        )  # 共享的片上资源对象。
        cube = MtgrRaggedSegmentAttentionCube()  # Cube行为管理对象。
        vector = MtgrRaggedSegmentAttentionVector()  # Vector行为管理对象。
        task_packets = Array(dtypes.int64, task_packet_storage_words)  # 延迟任务参数队列。
        group = query_tensor.shape[1] // key_tensor.shape[1]  # 每个KV头对应的Q头数。
        header_valid = (
            metadata[HEADER_MAGIC] == METADATA_MAGIC and metadata[HEADER_CORE_COUNT] == get_block_num()
        )  # metadata头与当前kernel配置一致的校验标记。
        header_valid = (
            header_valid and metadata[HEADER_GQA_GROUP] == group and metadata[HEADER_QUERY_ROWS] == self.query_tile_rows
        )  # metadata头与当前kernel配置一致的校验标记。
        header_valid = (
            header_valid
            and metadata[HEADER_KV_COLUMNS] == self.key_tile_columns
            and metadata[HEADER_KV_HEADS] == key_tensor.shape[1]
        )  # metadata头与当前kernel配置一致的校验标记。
        header_valid = (
            header_valid
            and metadata[HEADER_REQUEST_COUNT] == prefixes.shape[0]
            and metadata[HEADER_HEAD_DIMENSION] == query_tensor.shape[tensor_feature_axis]
        )  # metadata头与当前kernel配置一致的校验标记。
        sections = dtypes.int64(metadata[HEADER_SECTION_COUNT])  # metadata中的L2分区数量。
        header_valid = (
            header_valid and sections > 0 and sections <= prefixes.shape[0] * key_tensor.shape[1]
        )  # metadata头与当前kernel配置一致的校验标记。
        if header_valid:
            for section in range(sections):  # L2工作集分区编号；按给定区间依次遍历当前处理项。
                tick = 0  # 流水发射节拍。
                issued = 0  # 已经发射的有效块数。
                query_task_sequence = 0  # 当前查询任务序号。
                core_metadata_word_begin = (
                    METADATA_HEADER_WORDS + (section * get_block_num() + get_block_idx()) * CORE_METADATA_WORDS
                )  # 当前记录或块的起始偏移，单位由所在存储视图确定。
                first_batch_kv_head = dtypes.int64(metadata[core_metadata_word_begin])  # 当前核起始请求与KV头联合编号。
                first_task_index = dtypes.int64(
                    metadata[core_metadata_word_begin + CORE_BEGIN_QUERY_TASK]
                )  # 当前核起始查询任务编号。
                last_batch_kv_head = dtypes.int64(
                    metadata[core_metadata_word_begin + CORE_END_WORKGROUP]
                )  # 当前核结束请求与KV头联合编号。
                last_task_index = dtypes.int64(
                    metadata[core_metadata_word_begin + CORE_END_QUERY_TASK]
                )  # 当前核结束查询任务编号。
                cached_batch_index = dtypes.int64(-1)  # 当前section内已缓存的请求编号。
                sequence_gm_token_begin = dtypes.int64(0)  # 缓存请求起点，batch切换时更新。
                live_sequence_length = dtypes.int64(0)  # 缓存实时序列长度。
                cached_prefix_tokens = dtypes.int64(0)  # 缓存前缀长度。
                logical_kv_length = dtypes.int64(0)  # 缓存含前缀的序列长度。
                cached_sequence_end = dtypes.int64(0)  # 缓存分段offsets的逻辑末端。
                tiles = dtypes.int64(0)  # 缓存查询块数。
                for batch_kv_head_index in range(
                    first_batch_kv_head, last_batch_kv_head + 1
                ):  # batch_kv_head_index，用于本函数的循环或分块定位；按给定区间依次遍历当前处理项。
                    if batch_kv_head_index >= 0 and batch_kv_head_index < prefixes.shape[0] * key_tensor.shape[1]:
                        batch_index = batch_kv_head_index // key_tensor.shape[1]  # 请求编号。
                        kv_head_index = batch_kv_head_index % key_tensor.shape[1]  # 键值头编号。
                        if batch_index != cached_batch_index:
                            sequence_gm_token_begin = dtypes.int64(starts[batch_index])  # 当前请求的实时token起点。
                            live_sequence_length = (
                                dtypes.int64(starts[batch_index + 1]) - sequence_gm_token_begin
                            )  # 当前请求的实时序列长度。
                            cached_prefix_tokens = dtypes.int64(prefixes[batch_index])  # 当前请求的命中前缀长度。
                            logical_kv_length = live_sequence_length + cached_prefix_tokens  # 含前缀的逻辑序列长度。
                            tiles = (
                                live_sequence_length + self.query_tile_rows - 1
                            ) // self.query_tile_rows  # 当前请求的查询块数。
                            cached_sequence_end = dtypes.int64(offsets[batch_index, rules.shape[0]])
                            cached_batch_index = batch_index
                        core_query_task_begin = 0  # 首个有效块标记。
                        core_query_task_end = group * tiles  # 末个有效块标记。
                        if batch_kv_head_index == first_batch_kv_head:
                            core_query_task_begin = first_task_index  # 首个有效块标记。
                        if batch_kv_head_index == last_batch_kv_head:
                            core_query_task_end = last_task_index  # 末个有效块标记。
                        valid = (
                            sequence_gm_token_begin >= 0
                            and live_sequence_length > 0
                            and sequence_gm_token_begin + live_sequence_length <= query_tensor.shape[0]
                            and cached_prefix_tokens >= 0
                        )  # 请求范围和调度边界合法标记。
                        valid = (
                            valid
                            and cached_prefix_tokens % CACHE_BLOCK_TOKENS == 0
                            and cached_prefix_tokens // CACHE_BLOCK_TOKENS <= kv_page_table.shape[1]
                        )  # 请求范围和调度边界合法标记。
                        valid = (
                            valid and logical_kv_length == cached_sequence_end
                        )  # 请求范围和调度边界合法标记。
                        valid = (
                            valid
                            and core_query_task_begin >= 0
                            and core_query_task_end >= core_query_task_begin
                            and core_query_task_end <= group * tiles
                        )  # 请求范围和调度边界合法标记。
                        if valid:
                            for task in range(
                                core_query_task_begin, core_query_task_end
                            ):  # 当前工作组内的查询任务编号；按给定区间依次遍历当前处理项。
                                query_head_index = kv_head_index * group + task // tiles  # 查询头编号。
                                query_row_start = task % tiles * self.query_tile_rows  # 查询块起点。
                                cutoff_min, cutoff_max, diagonal_min, diagonal_max = vector.prepare_rows(
                                    pipeline_resources,
                                    offsets,
                                    rules,
                                    batch_index,
                                    query_row_start,
                                    cached_prefix_tokens,
                                    live_sequence_length,
                                )  # 本AIV所有有效行的连续前段最小结束位置；本AIV可见连续前段的最大结束位置；当前AIV半块最小对角位置；当前AIV半块最大对角位置。
                                full_query_rows = (
                                    live_sequence_length - query_row_start
                                    >= (pipeline_resources.vector_subblock_index + 1) * pipeline_resources.vector_query_rows
                                )  # 每个查询任务仅计算一次有效行标记，尾块保留原逐行掩码路径。
                                full_cutoff, full_diag_min, full_diag_max = self.cube_visible_bounds(
                                    offsets,
                                    rules,
                                    batch_index,
                                    cached_prefix_tokens + query_row_start,
                                    min(
                                        cached_prefix_tokens + query_row_start + self.query_tile_rows, logical_kv_length
                                    ),
                                    logical_kv_length,
                                )  # 整个128行Cube块的连续可见前段上界；整个128行Cube块的对角位置下界；整个128行Cube块的对角位置上界。

                                active_stop = max(
                                    ((full_cutoff + self.key_tile_columns - 1) // self.key_tile_columns)
                                    * self.key_tile_columns,
                                    ((full_diag_max // self.key_tile_columns) + 1) * self.key_tile_columns,
                                )  # 最后有效键值块的结束位置。
                                if query_task_sequence >= query_l1_reuse_interval:
                                    cube_sync_pipe(PIPE.MTE1)
                                query_tile_write = (
                                    pipeline_resources.query_tensor.next_write()
                                )  # 当前查询块的L1待写槽；取得受生产消费依赖保护的待写槽。
                                cube.copy_q(
                                    query_tile_write,
                                    query_tensor,
                                    sequence_gm_token_begin,
                                    query_head_index,
                                    query_row_start,
                                    live_sequence_length,
                                )
                                cube_sync_pipe(PIPE.MTE2)
                                query_tile_read = (
                                    pipeline_resources.query_tensor.next_read()
                                )  # 已经搬入L1的查询块读取槽；取得已就绪的待消费槽。
                                for key_column_start in range(
                                    0, active_stop, self.key_tile_columns
                                ):  # 逻辑键值块起点；按给定区间依次遍历当前处理项。
                                    if key_column_start < full_cutoff or (
                                        key_column_start + self.key_tile_columns > full_diag_min
                                        and key_column_start <= full_diag_max
                                    ):
                                        masked = 0  # 当前AIV半块全遮蔽标记。
                                        if key_column_start >= cutoff_max and (
                                            key_column_start + self.key_tile_columns <= diagonal_min
                                            or key_column_start > diagonal_max
                                        ):
                                            masked = 1  # 当前AIV半块全遮蔽标记。
                                        slot = (
                                            tick % task_packet_slots
                                        ) * task_packet_words  # 当前四槽任务队列的写入偏移。
                                        task_packets[slot] = batch_index  # 延迟流水任务记录中的请求编号。
                                        task_packets[slot + packet_gm_begin_field] = (
                                            sequence_gm_token_begin  # 延迟流水任务记录中的请求实时token起点。
                                        )
                                        task_packets[slot + packet_query_head_field] = (
                                            query_head_index  # 延迟流水任务记录中的查询头编号。
                                        )
                                        task_packets[slot + packet_query_row_field] = (
                                            query_row_start  # 延迟流水任务记录中的查询块起点。
                                        )
                                        task_packets[slot + packet_kv_column_field] = key_column_start  # 键块起点。
                                        task_packets[slot + packet_prefix_field] = (
                                            cached_prefix_tokens  # 延迟流水任务记录中的命中前缀长度。
                                        )
                                        task_packets[slot + packet_live_length_field] = (
                                            live_sequence_length  # 延迟流水任务记录中的实时序列长度。
                                        )
                                        task_packets[slot + packet_active_stop_field] = (
                                            active_stop  # 延迟流水任务记录中的最后有效块终点。
                                        )
                                        task_packets[slot + packet_query_sequence_field] = (
                                            query_task_sequence  # 查询任务序号。
                                        )
                                        task_packets[slot + packet_masked_field] = (
                                            masked  # 延迟流水任务记录中的半块全遮蔽标记。
                                        )
                                        cube.compute_query_key(
                                            pipeline_resources,
                                            query_tile_read,
                                            key_tensor,
                                            key_cache_tensor,
                                            kv_page_table,
                                            batch_index,
                                            sequence_gm_token_begin,
                                            kv_head_index,
                                            key_column_start,
                                            cached_prefix_tokens,
                                            live_sequence_length,
                                        )
                                        vector.softmax_and_publish_probability(
                                            pipeline_resources,
                                            key_column_start,
                                            query_task_sequence,
                                            scale,
                                            cutoff_min,
                                            cutoff_max,
                                            diagonal_min,
                                            diagonal_max,
                                            masked,
                                            full_query_rows,
                                        )
                                        issued += 1  # 已经发射的有效块数。
                                        if tick >= pipeline_delay_ticks:
                                            read_packet_offset = (
                                                (tick - pipeline_delay_ticks) % task_packet_slots
                                            ) * task_packet_words  # 延迟两个发射节拍的任务记录偏移。
                                            cube.compute_probability_value(
                                                pipeline_resources,
                                                value_tensor,
                                                value_cache_tensor,
                                                kv_page_table,
                                                task_packets[read_packet_offset],
                                                task_packets[read_packet_offset + packet_gm_begin_field],
                                                task_packets[read_packet_offset + packet_query_head_field] // group,
                                                task_packets[read_packet_offset + packet_kv_column_field],
                                                task_packets[read_packet_offset + packet_prefix_field],
                                                task_packets[read_packet_offset + packet_live_length_field],
                                            )
                                            vector.update_and_store_output(
                                                pipeline_resources,
                                                out,
                                                task_packets[read_packet_offset + packet_gm_begin_field],
                                                task_packets[read_packet_offset + packet_query_head_field],
                                                task_packets[read_packet_offset + packet_query_row_field],
                                                task_packets[read_packet_offset + packet_kv_column_field],
                                                task_packets[read_packet_offset + packet_live_length_field],
                                                task_packets[read_packet_offset + packet_active_stop_field],
                                                task_packets[read_packet_offset + packet_query_sequence_field],
                                                task_packets[read_packet_offset + packet_masked_field],
                                            )
                                        tick += 1  # 流水发射节拍。
                                query_task_sequence += 1  # 当前查询任务序号。
                for drain in range(pipeline_delay_ticks):  # 流水排空阶段的迭代号；按给定区间依次遍历当前处理项。
                    if tick >= pipeline_delay_ticks and tick - pipeline_delay_ticks < issued:
                        read_packet_offset = (
                            (tick - pipeline_delay_ticks) % task_packet_slots
                        ) * task_packet_words  # 延迟两个发射节拍的任务记录偏移。
                        cube.compute_probability_value(
                            pipeline_resources,
                            value_tensor,
                            value_cache_tensor,
                            kv_page_table,
                            task_packets[read_packet_offset],
                            task_packets[read_packet_offset + packet_gm_begin_field],
                            task_packets[read_packet_offset + packet_query_head_field] // group,
                            task_packets[read_packet_offset + packet_kv_column_field],
                            task_packets[read_packet_offset + packet_prefix_field],
                            task_packets[read_packet_offset + packet_live_length_field],
                        )
                        vector.update_and_store_output(
                            pipeline_resources,
                            out,
                            task_packets[read_packet_offset + packet_gm_begin_field],
                            task_packets[read_packet_offset + packet_query_head_field],
                            task_packets[read_packet_offset + packet_query_row_field],
                            task_packets[read_packet_offset + packet_kv_column_field],
                            task_packets[read_packet_offset + packet_live_length_field],
                            task_packets[read_packet_offset + packet_active_stop_field],
                            task_packets[read_packet_offset + packet_query_sequence_field],
                            task_packets[read_packet_offset + packet_masked_field],
                        )
                    tick += 1  # 流水发射节拍。
                cube_sync_pipe(PIPE.MTE1)

    @jit
    def cube_visible_bounds(
        self,
        offsets: Tensor,
        rules: Tensor,
        batch_index: int,
        query_logical_begin: int,
        query_logical_end: int,
        logical_kv_length: int,
    ):
        """扫描与整个查询块相交的段，合并连续可见前段和对角范围，供整块矩阵乘裁剪。

        Args:
            offsets: 各请求分段偏移。
            rules: 各段掩码规则。
            batch_index: 请求编号。
            query_logical_begin: 当前查询块的逻辑起始位置。
            query_logical_end: 当前查询块的逻辑结束位置。
            logical_kv_length: 含前缀的逻辑序列长度。

        Returns:
            tuple[int, int, int]: 连续前段上界、对角最小位置和最大位置。
        """
        binary_search_divisor = 2  # 二分区间折半因子。
        cutoff = dtypes.int64(0)  # 当前行连续可见前段的结束位置。
        diag_min = dtypes.int64(logical_kv_length)  # 整个Cube查询块的最小可见对角位置。
        diag_max = dtypes.int64(NO_DIAGONAL_TOKEN)  # 整个Cube查询块的最大可见对角位置。
        search_lower = 0  # 段二分查找的下界。
        search_upper = rules.shape[0]  # 段二分查找的上界。
        while search_lower < search_upper:
            search_middle = (search_lower + search_upper) // binary_search_divisor  # 段二分查找的中点。
            if query_logical_begin >= offsets[batch_index, search_middle + 1]:
                search_lower = search_middle + 1  # 段二分查找的下界。
            else:
                search_upper = search_middle  # 段二分查找的上界。
        while search_lower < rules.shape[0] and offsets[batch_index, search_lower] < query_logical_end:
            segment_kv_begin = offsets[batch_index, search_lower]  # 当前段的逻辑起点。
            segment_kv_end = offsets[batch_index, search_lower + 1]  # 当前区间的结束位置。
            segment_mask_rule = rules[search_lower]  # 当前段的掩码规则编号。
            high = dtypes.int64(min(query_logical_end, segment_kv_end))  # 当前段对连续前段上界的贡献。
            if segment_mask_rule == MASK_SEGMENT_FULL:
                high = dtypes.int64(segment_kv_end)  # 当前段对连续前段上界的贡献。
            elif segment_mask_rule == MASK_PREFIX_AND_SELF:
                high = dtypes.int64(segment_kv_begin)  # 当前段对连续前段上界的贡献。
                diag_min = min(
                    diag_min, max(query_logical_begin, segment_kv_begin)
                )  # 整个Cube查询块的最小可见对角位置。
                diag_max = max(
                    diag_max, min(query_logical_end, segment_kv_end) - 1
                )  # 整个Cube查询块的最大可见对角位置。
            cutoff = max(cutoff, high)  # 当前行连续可见前段的结束位置。
            search_lower += 1  # 段二分查找的下界。
        return cutoff, diag_min, diag_max


from typing import Any
from dataclasses import dataclass
import torch
from cannbotdsl import get_platform_info
import cannbotdsl as dsl
from functools import lru_cache
from cannbotdsl import Dim, Tensor, TensorSpec, dtypes, host
from cannbotdsl.types.dtypes import bfloat16

if __package__:
    pass
    from .mtgr_ragged_segment_attention_metadata import MtgrMetadataHost
else:
    pass
    from mtgr_ragged_segment_attention_metadata import MtgrMetadataHost

if __package__:
    from .mtgr_ragged_segment_attention_common import (
        ATTENTION_COMPILE_CACHE_SIZE,
        CACHE_BLOCK_TOKENS,
        CORE_METADATA_WORDS,
        METADATA_CORE_CAPACITY,
        DEFAULT_COMPILE_CORE_COUNT,
        DEFAULT_CORE_COUNT,
        DEFAULT_KV_TILE_COLUMNS,
        DEFAULT_QUERY_TILE_ROWS,
        MAX_REQUESTS,
        MAX_CUBE_CORES,
        METADATA_HEADER_WORDS,
        MIN_SEGMENTS,
        QUERY_HEAD_COUNT,
        SND_TENSOR_RANK,
        SUPPORTED_HEAD_DIMENSIONS,
        SUPPORTED_GQA_GROUPS,
        SUPPORTED_KV_HEADS,
        SUPPORTED_MATCH_MODES,
        SUPPORTED_QUERY_TILE_ROWS,
        SUPPORTED_RULE1_INDICES,
    )
else:
    from mtgr_ragged_segment_attention_common import (
        ATTENTION_COMPILE_CACHE_SIZE,
        CACHE_BLOCK_TOKENS,
        CORE_METADATA_WORDS,
        METADATA_CORE_CAPACITY,
        DEFAULT_COMPILE_CORE_COUNT,
        DEFAULT_CORE_COUNT,
        DEFAULT_KV_TILE_COLUMNS,
        DEFAULT_QUERY_TILE_ROWS,
        MAX_REQUESTS,
        MAX_CUBE_CORES,
        METADATA_HEADER_WORDS,
        MIN_SEGMENTS,
        QUERY_HEAD_COUNT,
        SND_TENSOR_RANK,
        SUPPORTED_HEAD_DIMENSIONS,
        SUPPORTED_GQA_GROUPS,
        SUPPORTED_KV_HEADS,
        SUPPORTED_MATCH_MODES,
        SUPPORTED_QUERY_TILE_ROWS,
        SUPPORTED_RULE1_INDICES,
    )


@dataclass(frozen=True)
class MtgrRaggedSegmentAttentionTiling:
    """管理 attention 分块配置和 Host 侧输入校验。

    validate_inputs 检查接口规格并推导 GQA 分组及请求规模；validate_metadata 检查设备计划布局。

    Attributes:
        query_tile_rows: 完整 Cube 查询块的行数。
        key_tile_columns: 每个键值块的列数。
    """
    query_tile_rows: int = DEFAULT_QUERY_TILE_ROWS
    key_tile_columns: int = DEFAULT_KV_TILE_COLUMNS

    def validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        segment_offsets_i32: torch.Tensor,
        segment_rules_i32: torch.Tensor,
        q_seq_starts_i32: torch.Tensor,
        matched_prefix_lens_i32: torch.Tensor,
        match_mode: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table_i32: torch.Tensor,
        block_size: int,
        max_request_len: int,
        output_snd: torch.Tensor,
        block_M: int = DEFAULT_QUERY_TILE_ROWS,
        core_num: int = DEFAULT_CORE_COUNT,
        kv_group: int = 1,
        rule1_seg_idx: int = 1,
    ):
        """校验接口张量、分页缓存及调度属性。

        只检查设备、形状、类型、布局和属性范围，不读取设备张量的元素值。

        Args:
            query: BF16 查询张量，布局为 [total_q, 8, D]。
            key: BF16 实时键张量，布局为 [total_q, kv_heads, D]。
            value: BF16 实时值张量，形状与 key 一致。
            segment_offsets_i32: INT32 分段边界张量，形状为 [batch, segments + 1]。
            segment_rules_i32: INT32 分段掩码规则张量，每段一个规则。
            q_seq_starts_i32: 各请求在实时 Q/K/V 中的起止偏移，共 batch + 1 项。
            matched_prefix_lens_i32: 各请求命中的分页缓存前缀长度。
            match_mode: 匹配模式属性；当前入口校验其取值属于 0、1、2。
            key_cache: BF16 分页键缓存，布局为 [cache_blocks, 128, kv_heads, D]。
            value_cache: BF16 分页值缓存，形状与 key_cache 一致。
            block_table_i32: INT32 页表，按请求和逻辑页号索引物理页。
            block_size: 缓存块 token 数，当前要求为 128。
            max_request_len: 请求长度上界属性，当前入口要求其为正值。
            output_snd: 调用方分配的 BF16 输出张量，形状与 query 一致，计算时原地更新。
            block_M: 兼容接口的查询分块属性，校验支持 64 或 128；实际分块取自 tiling。
            core_num: attention 主 kernel 使用的 Cube 核数。
            kv_group: 每个 KV 头对应的查询头数；取 1 时也允许根据输入头数推导。
            rule1_seg_idx: 兼容接口的规则 1 段索引属性，当前入口校验其取值为 0 或 1。

        Returns:
            tuple[int, int, int]: 依次为推导的 GQA 分组倍率、请求数和分段数。

        Raises:
            ValueError: 输入设备、形状、类型、连续性、分段规格或调度属性不满足当前支持范围。
        """
        tensor_feature_axis = 2  # 本函数使用的tensor_feature_axis编译期配置。
        tensor_matrix_rank = 2  # 本函数使用的tensor_matrix_rank编译期配置。
        if query.device.type not in ("npu", "privateuseone"):
            raise ValueError("Query must reside on an NPU device; CPU tensors are not supported.")
        if block_size != CACHE_BLOCK_TOKENS or block_M not in SUPPORTED_QUERY_TILE_ROWS or max_request_len <= 0:
            raise ValueError("Block size must be 128, block_M must be 64 or 128, and max_request_len must be positive.")
        if (
            query.ndim != SND_TENSOR_RANK
            or query.shape[1] != QUERY_HEAD_COUNT
            or query.shape[tensor_feature_axis] not in SUPPORTED_HEAD_DIMENSIONS
        ):
            raise ValueError("Query must have shape [total_q, 8, D], with D equal to 32, 64, or 128.")
        if key.ndim != SND_TENSOR_RANK or key.shape[1] not in SUPPORTED_KV_HEADS:
            raise ValueError("Key must be a rank-three tensor with 1, 2, 4, or 8 KV heads.")
        group = QUERY_HEAD_COUNT // key.shape[1]  # 每个KV头对应的Q头数。
        if (
            kv_group not in (1, group)
            or key.shape != value.shape
            or key.shape[0] != query.shape[0]
            or key.shape[tensor_feature_axis] != query.shape[tensor_feature_axis]
        ):
            raise ValueError(
                "Live key and value shapes must match the query token count and head dimension, and kv_group must match the inferred GQA group."
            )
        if output_snd.shape != query.shape or key_cache.shape != value_cache.shape:
            raise ValueError("Output shape must match the query shape, and key/value cache shapes must match.")
        if tuple(key_cache.shape[1:]) != (CACHE_BLOCK_TOKENS, key.shape[1], query.shape[tensor_feature_axis]):
            raise ValueError("Paged KV cache must have trailing shape [128, kv_heads, query_head_dimension].")
        bfloat16_tensors = (
            query,
            key,
            value,
            key_cache,
            value_cache,
            output_snd,
        )  # 需要检查类型和布局的BF16输入输出集合。
        integer_metadata_tensors = (
            segment_offsets_i32,
            segment_rules_i32,
            q_seq_starts_i32,
            matched_prefix_lens_i32,
            block_table_i32,
        )  # 需要检查的INT32元数据张量集合。
        if any(
            tensor.dtype != torch.bfloat16 or tensor.device != query.device or not tensor.is_contiguous()
            for tensor in bfloat16_tensors
        ):
            raise ValueError(
                "Query, key, value, caches, and output must be contiguous BF16 tensors on the query device."
            )
        if any(
            tensor.dtype != torch.int32 or tensor.device != query.device or not tensor.is_contiguous()
            for tensor in integer_metadata_tensors
        ):
            raise ValueError(
                "Segment offsets, rules, sequence starts, prefix lengths, and page table must be contiguous INT32 tensors on the query device."
            )
        batch = matched_prefix_lens_i32.numel()  # 请求数量。
        segments = segment_rules_i32.numel()  # 分段数量。
        if (
            not 1 <= batch <= MAX_REQUESTS
            or segments < MIN_SEGMENTS
            or segment_offsets_i32.shape != (batch, segments + 1)
        ):
            raise ValueError(
                "Segment offsets must have shape [batch, segments + 1], with 1 to 8 requests and at least two segments."
            )
        if q_seq_starts_i32.shape != (batch + 1,) or matched_prefix_lens_i32.shape != (batch,):
            raise ValueError("Sequence starts must have batch + 1 entries, and prefix lengths must have batch entries.")
        if block_table_i32.ndim != tensor_matrix_rank or block_table_i32.shape[0] != batch:
            raise ValueError("KV page table must be rank two with one row per request.")
        if rule1_seg_idx not in SUPPORTED_RULE1_INDICES or match_mode not in SUPPORTED_MATCH_MODES:
            raise ValueError("Rule-1 segment index must be 0 or 1, and match_mode must be 0, 1, or 2.")
        available = int(get_platform_info(stream=torch.npu.current_stream()).cube_core_num)  # 平台可用的Cube核数。
        if not isinstance(core_num, int) or not 1 <= core_num <= available:
            raise ValueError("Core count must be an integer between 1 and the available Cube core count.")
        return group, batch, segments

    def validate_metadata(
        self,
        metadata: torch.Tensor | None,
        query: torch.Tensor,
        batch: int,
        kv_heads: int,
        core_num: int,
    ):
        """校验已有设备计划的规格。

        根据请求数、KV 头数及核数推导计划长度，并检查类型、设备和连续性。

        Args:
            metadata: 设备侧 INT32 分核计划；允许为 None 的入口会先生成计划。
            query: BF16 查询张量，布局为 [total_q, 8, D]。
            batch: 请求数量。
            kv_heads: KV 头数，用于推导计划区大小。
            core_num: attention 主 kernel 使用的 Cube 核数。

        Returns:
            None: 校验通过后直接返回，不修改设备计划。

        Raises:
            ValueError: 计划的类型、设备、长度或连续性与预期规格不一致。
        """
        expected_size = (
            METADATA_HEADER_WORDS + batch * kv_heads * METADATA_CORE_CAPACITY * CORE_METADATA_WORDS
        )  # metadata计划区预期INT32元素数。
        if (
            metadata.dtype != torch.int32
            or metadata.device != query.device
            or metadata.shape != (expected_size,)
            or not metadata.is_contiguous()
        ):
            raise ValueError(
                "Metadata must be a contiguous INT32 tensor on the query device with the expected scheduling-plan length."
            )


class MtgrRaggedSegmentAttentionLaunch:
    """管理 attention 主 kernel 的 Host 发射入口。

    保存分块规格，由 launch 构造 kernel 对象并在指定核数下发射一次主任务。

    Attributes:
        query_tile_rows: 完整 Cube 查询块的行数。
        key_tile_columns: 每个键值块的列数。
    """
    def __init__(
        self,
        query_tile_rows: int,
        key_tile_columns: int,
    ):
        """保存查询和键值分块规格。

        Args:
            query_tile_rows: 完整 Cube 查询块的行数。
            key_tile_columns: 每个键值块的列数。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        self.query_tile_rows, self.key_tile_columns = (
            query_tile_rows,
            key_tile_columns,
        )  # 查询分块行数；键值分块列数。

    @host
    def launch(
        self,
        query_tensor: Tensor,
        key_tensor: Tensor,
        value_tensor: Tensor,
        key_cache_tensor: Tensor,
        value_cache_tensor: Tensor,
        kv_page_table: Tensor,
        offsets: Tensor,
        rules: Tensor,
        starts: Tensor,
        prefixes: Tensor,
        metadata: Tensor,
        out: Tensor,
        scale: float,
        cores: int,
    ):
        """按配置构造主kernel，在指定核数下发起一次attention主任务。

        Args:
            query_tensor: 查询张量。
            key_tensor: 键张量。
            value_tensor: 值张量。
            key_cache_tensor: 分页键缓存张量。
            value_cache_tensor: 分页值缓存张量。
            kv_page_table: 逻辑页到物理页的映射。
            offsets: 各请求分段偏移。
            rules: 各段掩码规则。
            starts: 各请求实时token起点。
            prefixes: 各请求命中前缀长度。
            metadata: 设备侧 INT32 分核计划；允许为 None 的入口会先生成计划。
            out: 最终输出暂存或设备输出视图。
            scale: 分数缩放因子。
            cores: attention 主 kernel 使用的 Cube 核数。

        Returns:
            None: 通过对象属性或传入缓冲区保存结果。
        """
        attention_kernel = MtgrRaggedSegmentAttention(
            self.query_tile_rows, self.key_tile_columns
        )  # 当前分块配置的attention主kernel实例。
        attention_kernel[cores](
            query_tensor,
            key_tensor,
            value_tensor,
            key_cache_tensor,
            value_cache_tensor,
            kv_page_table,
            offsets,
            rules,
            starts,
            prefixes,
            metadata,
            out,
            scale,
        )


def build_mtgr_ragged_segment_attention_executable(
    total: int,
    head_dimension: int,
    group: int,
    batch: int,
    segments: int,
    cache_blocks: int,
    table_cols: int,
    query_tile_rows: int = DEFAULT_QUERY_TILE_ROWS,
    key_tile_columns: int = DEFAULT_KV_TILE_COLUMNS,
    cores: int = DEFAULT_COMPILE_CORE_COUNT,
):
    """兼容原编译入口，返回支持运行时形状的可执行对象。

    Args:
        total: 兼容参数，实时token总数不参与缓存区分。
        head_dimension: 静态头维度，支持32、64、128。
        group: 兼容GQA参数，仅检查取值，不按GQA生成编译变体。
        batch: 兼容参数，请求数量在运行时读取。
        segments: 兼容参数，段数量在运行时读取。
        cache_blocks: 兼容参数，缓存物理块数在运行时读取。
        table_cols: 兼容参数，页表列数在运行时读取。
        query_tile_rows: 编译期查询分块行数。
        key_tile_columns: 编译期键值分块列数。
        cores: 兼容核数参数，仅检查取值，实际发射核数由调用时提供。

    Returns:
        可跨ragged形状、GQA和核数复用的DSL可执行对象。

    Raises:
        ValueError: GQA或核数不在接口支持范围内。
    """
    if group not in SUPPORTED_GQA_GROUPS or not 1 <= cores <= MAX_CUBE_CORES:
        raise ValueError("GQA group or core count is outside the supported range.")
    return _compile_mtgr_dynamic_executable(head_dimension, query_tile_rows, key_tile_columns)


@lru_cache(maxsize=ATTENTION_COMPILE_CACHE_SIZE)
def _compile_mtgr_dynamic_executable(
    head_dimension: int,
    query_tile_rows: int,
    key_tile_columns: int,
):
    """每种头维度及分块只编译一次，其他轴和核数在运行时提供。

    Args:
        head_dimension: 编译期头维度。
        query_tile_rows: 编译期查询分块行数。
        key_tile_columns: 编译期键值分块列数。

    Returns:
        具有动态TensorSpec的已编译DSL可执行对象。
    """
    total = Dim("total_q", min=1)
    batch = Dim("batch", min=1, max=MAX_REQUESTS)
    segments = Dim("segments", min=MIN_SEGMENTS)
    cache_blocks = Dim("cache_blocks", min=1)
    table_cols = Dim("table_cols", min=1)
    kv_heads = Dim("kv_heads", min=1, max=QUERY_HEAD_COUNT)
    metadata_words = Dim("metadata_words", min=METADATA_HEADER_WORDS + METADATA_CORE_CAPACITY * CORE_METADATA_WORDS)
    query_specification = TensorSpec(
        (total, QUERY_HEAD_COUNT, head_dimension), bfloat16
    )  # 查询张量的编译规格，token轴为动态维。
    kv_specification = TensorSpec(
        (total, kv_heads, head_dimension), bfloat16
    )  # 实时KV张量的编译规格，token轴和KV头轴均为动态维。
    cache_specification = TensorSpec(
        (cache_blocks, CACHE_BLOCK_TOKENS, kv_heads, head_dimension), bfloat16
    )  # 分页KV缓存的形状和类型规格；构造输入的静态形状及类型规格。
    page_table_specification = TensorSpec((batch, table_cols), dtypes.int32)  # 分页KV缓存的物理页索引规格。
    segment_offsets_specification = TensorSpec((batch, segments + 1), dtypes.int32)  # 每个请求的分段边界规格。
    segment_rules_specification = TensorSpec((segments,), dtypes.int32)  # 各段掩码规则规格。
    sequence_starts_specification = TensorSpec((batch + 1,), dtypes.int32)  # 实时Q/K/V的请求起止索引规格。
    prefix_lengths_specification = TensorSpec((batch,), dtypes.int32)  # 每个请求的缓存前缀长度规格。
    metadata_specification = TensorSpec((metadata_words,), dtypes.int32)
    return dsl.compile(
        MtgrRaggedSegmentAttentionLaunch(query_tile_rows, key_tile_columns).launch,
        query_specification,
        kv_specification,
        kv_specification,
        cache_specification,
        cache_specification,
        page_table_specification,
        segment_offsets_specification,
        segment_rules_specification,
        sequence_starts_specification,
        prefix_lengths_specification,
        metadata_specification,
        query_specification,
        dtypes.float32,
        dtypes.int64,
    )


def precompile(
    config: dict[str, Any],
    data: dict[str, Any],
    core_num: int = DEFAULT_CORE_COUNT,
):
    """预编译独立metadata程序和当前输入规格的attention程序，不运行设备计算。

    Args:
        config: 用例配置字典，需包含 D、seg_lengths、rules，可选 kv_group。
        data: 输入数据字典，需包含 query_snd、key_cache、block_table_tensor。
        core_num: attention 主 kernel 使用的 Cube 核数。

    Returns:
        当前输入规格对应的 attention 可执行对象；同时预编译并缓存 metadata 程序。

    Raises:
        KeyError: config 或 data 缺少预编译所需的字段。
    """
    tiling = MtgrRaggedSegmentAttentionTiling()
    MtgrMetadataHost.compiled_metadata()
    return build_mtgr_ragged_segment_attention_executable(
        data["query_snd"].shape[0],
        config["D"],
        config.get("kv_group", 1),
        len(config["seg_lengths"]),
        len(config["rules"]),
        data["key_cache"].shape[0],
        data["block_table_tensor"].shape[1],
        tiling.query_tile_rows,
        tiling.key_tile_columns,
        core_num,
    )


def launch_attention(
    tiling: MtgrRaggedSegmentAttentionTiling,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    segment_offsets_i32: torch.Tensor,
    segment_rules_i32: torch.Tensor,
    q_seq_starts_i32: torch.Tensor,
    matched_prefix_lens_i32: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_i32: torch.Tensor,
    output_snd: torch.Tensor,
    metadata: torch.Tensor | None,
    sm_scale: float,
    l2_budget: int,
    core_num: int,
    group: int,
    batch: int,
    segments: int,
) -> torch.Tensor:
    """准备设备计划并发射一次 attention 主 kernel。

    metadata 为 None 时先生成计划；随后校验计划布局，复用编译缓存，并原地写入输出。

    Args:
        tiling: 分块配置及输入、metadata 校验方法的管理对象。
        query: BF16 查询张量，布局为 [total_q, 8, D]。
        key: BF16 实时键张量，布局为 [total_q, kv_heads, D]。
        value: BF16 实时值张量，形状与 key 一致。
        segment_offsets_i32: INT32 分段边界张量，形状为 [batch, segments + 1]。
        segment_rules_i32: INT32 分段掩码规则张量，每段一个规则。
        q_seq_starts_i32: 各请求在实时 Q/K/V 中的起止偏移，共 batch + 1 项。
        matched_prefix_lens_i32: 各请求命中的分页缓存前缀长度。
        key_cache: BF16 分页键缓存，布局为 [cache_blocks, 128, kv_heads, D]。
        value_cache: BF16 分页值缓存，形状与 key_cache 一致。
        block_table_i32: INT32 页表，按请求和逻辑页号索引物理页。
        output_snd: 调用方分配的 BF16 输出张量，形状与 query 一致，计算时原地更新。
        metadata: 设备侧 INT32 分核计划；允许为 None 的入口会先生成计划。
        sm_scale: QK 分数的 softmax 缩放系数。
        l2_budget: metadata 划分 section 使用的 L2 工作集预算，单位为字节。
        core_num: attention 主 kernel 使用的 Cube 核数。
        group: 每个 KV 头对应的查询头数。
        batch: 请求数量。
        segments: 每个请求的分段数量。

    Returns:
        torch.Tensor: 原地更新后的 output_snd，与传入对象相同。

    Raises:
        ValueError: metadata 的生成入参或已有计划规格不合法。
    """
    tensor_feature_axis = 2  # 本函数使用的tensor_feature_axis编译期配置。
    if metadata is None:
        metadata = MtgrMetadataHost._prepare_metadata(
            q_seq_starts_i32,
            matched_prefix_lens_i32,
            segment_offsets_i32,
            total=query.shape[0],
            head_dimension=query.shape[tensor_feature_axis],
            group=group,
            cores=core_num,
            table_cols=block_table_i32.shape[1],
            query_tile_rows=tiling.query_tile_rows,
            key_tile_columns=tiling.key_tile_columns,
            l2_budget=l2_budget,
            rules=segment_rules_i32,
        )  # 设备侧分核计划。
    tiling.validate_metadata(metadata, query, batch, key.shape[1], core_num)
    mtgr_ragged_segment_attention_executable = build_mtgr_ragged_segment_attention_executable(
        query.shape[0],
        query.shape[tensor_feature_axis],
        group,
        batch,
        segments,
        key_cache.shape[0],
        block_table_i32.shape[1],
        tiling.query_tile_rows,
        tiling.key_tile_columns,
        core_num,
    )  # 已经编译的attention程序对象。
    mtgr_ragged_segment_attention_executable(
        query,
        key,
        value,
        key_cache,
        value_cache,
        block_table_i32,
        segment_offsets_i32,
        segment_rules_i32,
        q_seq_starts_i32,
        matched_prefix_lens_i32,
        metadata,
        output_snd,
        float(sm_scale),
        core_num,
    )
    return output_snd


import torch

if __package__:
    pass
    from .mtgr_ragged_segment_attention_metadata import DEFAULT_L2_BUDGET
else:
    pass
    from mtgr_ragged_segment_attention_metadata import DEFAULT_L2_BUDGET

if __package__:
    from .mtgr_ragged_segment_attention_common import (
        DEFAULT_CORE_COUNT,
        DEFAULT_CROSS_INTERVAL,
        DEFAULT_PIPELINE_STAGES,
        DEFAULT_QUERY_TILE_ROWS,
    )
else:
    from mtgr_ragged_segment_attention_common import (
        DEFAULT_CORE_COUNT,
        DEFAULT_CROSS_INTERVAL,
        DEFAULT_PIPELINE_STAGES,
        DEFAULT_QUERY_TILE_ROWS,
    )


__all__ = ("mtgr_ragged_segment_attention", "precompile", "MtgrRaggedSegmentAttentionTiling")


def mtgr_ragged_segment_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    segment_offsets_i32: torch.Tensor,
    segment_rules_i32: torch.Tensor,
    q_seq_starts_i32: torch.Tensor,
    matched_prefix_lens_i32: torch.Tensor,
    match_mode: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_i32: torch.Tensor,
    block_size: int,
    max_request_len: int,
    sm_scale: float,
    output_snd: torch.Tensor,
    block_M: int = DEFAULT_QUERY_TILE_ROWS,
    core_num: int = DEFAULT_CORE_COUNT,
    num_stages: int = DEFAULT_PIPELINE_STAGES,
    kv_group: int = 1,
    cross_interval: int = DEFAULT_CROSS_INTERVAL,
    rule1_seg_idx: int = 1,
    *,
    metadata: torch.Tensor | None = None,
    l2_budget: int = DEFAULT_L2_BUDGET,
):
    """计算分段掩码和分页前缀缓存下的 ragged attention。

    构造切分配置并校验输入，按需生成独立 metadata，再编译并发射一次 attention 主 kernel。

    Args:
        query: BF16 查询张量，布局为 [total_q, 8, D]。
        key: BF16 实时键张量，布局为 [total_q, kv_heads, D]。
        value: BF16 实时值张量，形状与 key 一致。
        segment_offsets_i32: INT32 分段边界张量，形状为 [batch, segments + 1]。
        segment_rules_i32: INT32 分段掩码规则张量，每段一个规则。
        q_seq_starts_i32: 各请求在实时 Q/K/V 中的起止偏移，共 batch + 1 项。
        matched_prefix_lens_i32: 各请求命中的分页缓存前缀长度。
        match_mode: 匹配模式属性；当前入口校验其取值属于 0、1、2。
        key_cache: BF16 分页键缓存，布局为 [cache_blocks, 128, kv_heads, D]。
        value_cache: BF16 分页值缓存，形状与 key_cache 一致。
        block_table_i32: INT32 页表，按请求和逻辑页号索引物理页。
        block_size: 缓存块 token 数，当前要求为 128。
        max_request_len: 请求长度上界属性，当前入口要求其为正值。
        sm_scale: QK 分数的 softmax 缩放系数。
        output_snd: 调用方分配的 BF16 输出张量，形状与 query 一致，计算时原地更新。
        block_M: 兼容接口的查询分块属性，校验支持 64 或 128；实际分块取自 tiling。
        core_num: attention 主 kernel 使用的 Cube 核数。
        num_stages: 兼容接口的流水级数属性，当前实现不以此参数改变固定缓冲配置。
        kv_group: 每个 KV 头对应的查询头数；取 1 时也允许根据输入头数推导。
        cross_interval: 兼容接口的交错间隔属性，当前实现不以此参数改变固定调度节拍。
        rule1_seg_idx: 兼容接口的规则 1 段索引属性，当前入口校验其取值为 0 或 1。
        metadata: 设备侧 INT32 分核计划；允许为 None 的入口会先生成计划。
        l2_budget: metadata 划分 section 使用的 L2 工作集预算，单位为字节。

    Returns:
        torch.Tensor: 原地更新后的 output_snd，与传入对象相同。

    Raises:
        ValueError: 输入校验失败，或 metadata 的输入、调度属性及计划规格不合法。
    """
    tiling = MtgrRaggedSegmentAttentionTiling()
    group, batch, segments = tiling.validate_inputs(
        query,
        key,
        value,
        segment_offsets_i32,
        segment_rules_i32,
        q_seq_starts_i32,
        matched_prefix_lens_i32,
        match_mode,
        key_cache,
        value_cache,
        block_table_i32,
        block_size,
        max_request_len,
        output_snd,
        block_M,
        core_num,
        kv_group,
        rule1_seg_idx,
    )
    return launch_attention(
        tiling,
        query,
        key,
        value,
        segment_offsets_i32,
        segment_rules_i32,
        q_seq_starts_i32,
        matched_prefix_lens_i32,
        key_cache,
        value_cache,
        block_table_i32,
        output_snd,
        metadata,
        sm_scale,
        l2_budget,
        core_num,
        group,
        batch,
        segments,
    )
