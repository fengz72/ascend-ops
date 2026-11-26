# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""文件说明：Metadata层：参数结构描述GM接口，辅助函数计算块权重及边界，设备入口类生成分区计划，Host类管理检查、编译和独立AICPU发射。
模块层级：接口 → Host调度 → 独立metadata预处理及一次attention主kernel。
生命周期：跨流水的缓冲复用依赖真实生产消费关系，不能跳过对应同步。
"""

from functools import lru_cache
import ast
import tempfile
import inspect
import textwrap

import torch
from cannbotdsl.aicpu import I32, I64, U32, current_raw_stream, gm_in, gm_out
from cannbotdsl.aicpu import aicpu_kernel as aicpu_kernel  # 临时入口源码通过编译命名空间间接引用此装饰器。
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel

if __package__:
    from .mtgr_ragged_segment_attention_common import (
        CACHE_BLOCK_TOKENS,
        CORE_BEGIN_QUERY_TASK,
        CORE_END_QUERY_TASK,
        CORE_END_WORKGROUP,
        CORE_METADATA_WORDS,
        METADATA_CORE_CAPACITY,
        CORE_PREVIOUS_END_TASK_DELTA,
        CORE_PREVIOUS_END_WORKGROUP_DELTA,
        CORE_SECTION_BEGIN,
        CORE_SECTION_END,
        DEFAULT_KV_TILE_COLUMNS,
        DEFAULT_L2_BUDGET,
        HEADER_CORE_COUNT,
        HEADER_GQA_GROUP,
        HEADER_HEAD_DIMENSION,
        HEADER_KV_COLUMNS,
        HEADER_KV_HEADS,
        HEADER_L2_BUDGET,
        HEADER_MAGIC,
        HEADER_QUERY_ROWS,
        HEADER_REQUEST_COUNT,
        HEADER_SECTION_COUNT,
        MASK_PREFIX_AND_SELF,
        MASK_SEGMENT_FULL,
        MAX_CUBE_CORES,
        MAX_REQUESTS,
        METADATA_HEADER_WORDS,
        METADATA_MAGIC,
        MIN_SEGMENT_OFFSET_WORDS,
        NO_DIAGONAL_TOKEN,
        QUERY_HEAD_COUNT,
        SUPPORTED_GQA_GROUPS,
        SUPPORTED_HEAD_DIMENSIONS,
        SUPPORTED_QUERY_TILE_ROWS,
    )
else:
    from mtgr_ragged_segment_attention_common import (
        CACHE_BLOCK_TOKENS,
        CORE_BEGIN_QUERY_TASK,
        CORE_END_QUERY_TASK,
        CORE_END_WORKGROUP,
        CORE_METADATA_WORDS,
        METADATA_CORE_CAPACITY,
        CORE_PREVIOUS_END_TASK_DELTA,
        CORE_PREVIOUS_END_WORKGROUP_DELTA,
        CORE_SECTION_BEGIN,
        CORE_SECTION_END,
        DEFAULT_KV_TILE_COLUMNS,
        DEFAULT_L2_BUDGET,
        HEADER_CORE_COUNT,
        HEADER_GQA_GROUP,
        HEADER_HEAD_DIMENSION,
        HEADER_KV_COLUMNS,
        HEADER_KV_HEADS,
        HEADER_L2_BUDGET,
        HEADER_MAGIC,
        HEADER_QUERY_ROWS,
        HEADER_REQUEST_COUNT,
        HEADER_SECTION_COUNT,
        MASK_PREFIX_AND_SELF,
        MASK_SEGMENT_FULL,
        MAX_CUBE_CORES,
        MAX_REQUESTS,
        METADATA_HEADER_WORDS,
        METADATA_MAGIC,
        MIN_SEGMENT_OFFSET_WORDS,
        NO_DIAGONAL_TOKEN,
        QUERY_HEAD_COUNT,
        SUPPORTED_GQA_GROUPS,
        SUPPORTED_HEAD_DIMENSIONS,
        SUPPORTED_QUERY_TILE_ROWS,
    )


class MtgrMetadataArguments:
    """声明独立 AICPU metadata 程序的设备接口字段。

    字段按注解描述 GM 地址与标量属性，供 metadata 辅助函数和 kernel 共享；本类不包含计算入口。

    Attributes:
        starts: 各请求实时token起点。
        prefixes: 各请求命中前缀长度。
        offsets: 各请求分段偏移。
        rules: 各段掩码规则。
        sparse_weights: 使用活动块权重的开关。
        metadata: 设备侧分核计划。
        batch: 请求数量。
        segments: 分段数量。
        total: 所有请求实时查询 token 的总数。
        table_cols: 每个请求的页表容量。
        group: 每个KV头对应的Q头数。
        head_dimension: 每个注意力头的维度。
        query_tile_rows: 完整 Cube 查询块的行数。
        key_tile_columns: 每个键值块的列数。
        cores: attention主kernel使用的核数。
        l2_budget: 调度使用的L2工作集预算。
    """
    starts: gm_in(I32)  # 各请求实时token起点；声明设备接口字段及其数据类型。
    prefixes: gm_in(I32)  # 各请求命中前缀长度；声明设备接口字段及其数据类型。
    offsets: gm_in(I32)  # 各请求分段偏移；声明设备接口字段及其数据类型。
    rules: gm_in(I32)  # 各段掩码规则；声明设备接口字段及其数据类型。
    sparse_weights: U32  # 使用活动块权重的开关；声明设备接口字段及其数据类型。
    metadata: gm_out(U32)  # 设备侧分核计划；声明设备接口字段及其数据类型。
    batch: U32  # 请求数量；声明设备接口字段及其数据类型。
    segments: U32  # 分段数量；声明设备接口字段及其数据类型。
    total: U32  # 由函数上下文决定的分母累计值或总token数；声明设备接口字段及其数据类型。
    table_cols: U32  # 每个请求的页表容量；声明设备接口字段及其数据类型。
    group: U32  # 每个KV头对应的Q头数；声明设备接口字段及其数据类型。
    head_dimension: U32  # 每个注意力头的维度；声明设备接口字段及其数据类型。
    query_tile_rows: U32  # 查询分块行数；声明设备接口字段及其数据类型。
    key_tile_columns: U32  # 键值分块列数；声明设备接口字段及其数据类型。
    cores: U32  # attention主kernel使用的核数；声明设备接口字段及其数据类型。
    l2_budget: I64  # 调度使用的L2工作集预算；声明设备接口字段及其数据类型。


def _mtgr_mtiles(
    args: MtgrMetadataArguments,
    batch_index: int,
):
    """检查请求起点、长度、前缀及段末边界，返回该请求有效查询块数。

    Args:
        args: metadata设备接口参数。
        batch_index: 请求编号。

    Returns:
        int: 有效查询块数。
    """
    sequence_gm_token_begin: I64 = args.starts[batch_index]  # 当前请求的实时token起点。
    live_sequence_length: I64 = args.starts[batch_index + 1] - sequence_gm_token_begin  # 当前请求的实时序列长度。
    cached_prefix_tokens: I64 = args.prefixes[batch_index]  # 当前请求的命中前缀长度。
    valid = (
        sequence_gm_token_begin >= 0
        and live_sequence_length > 0
        and sequence_gm_token_begin + live_sequence_length <= args.total
        and cached_prefix_tokens >= 0
    )  # 请求范围和调度边界合法标记。
    valid = (
        valid
        and cached_prefix_tokens % CACHE_BLOCK_TOKENS == 0
        and cached_prefix_tokens // CACHE_BLOCK_TOKENS <= args.table_cols
    )  # 请求范围和调度边界合法标记。
    valid = (
        valid
        and cached_prefix_tokens + live_sequence_length
        == args.offsets[batch_index * (args.segments + 1) + args.segments]
    )  # 请求范围和调度边界合法标记。
    if not valid:
        return 0
    return (live_sequence_length + args.query_tile_rows - 1) // args.query_tile_rows


def _mtgr_ktiles(
    args: MtgrMetadataArguments,
    batch_index: int,
):
    """按请求加前缀的逻辑长度计算键值块数。

    Args:
        args: metadata设备接口参数。
        batch_index: 请求编号。

    Returns:
        int: 逻辑键值块数。
    """
    logical_kv_length: I64 = (
        args.starts[batch_index + 1] - args.starts[batch_index] + args.prefixes[batch_index]
    )  # 含前缀的逻辑序列长度。
    return (logical_kv_length + args.key_tile_columns - 1) // args.key_tile_columns


def _mtgr_task_cost(
    args: MtgrMetadataArguments,
    batch_index: int,
    tile: int,
):
    """按段掩码计算当前查询任务需处理的活动键值块数，用作分核权重。

    Args:
        args: metadata设备接口参数。
        batch_index: 请求编号。
        tile: 当前请求中的查询块索引。

    Returns:
        int: 活动块权重。
    """
    binary_search_divisor = 2  # 二分区间折半因子。
    if args.sparse_weights == 0:
        return _mtgr_ktiles(args, batch_index)
    query_logical_begin: I64 = args.prefixes[batch_index] + tile * args.query_tile_rows  # 当前查询块的逻辑起始位置。
    query_logical_end: I64 = min(
        query_logical_begin + args.query_tile_rows,
        args.prefixes[batch_index] + args.starts[batch_index + 1] - args.starts[batch_index],
    )  # 当前查询块的逻辑结束位置。
    cutoff: I64 = 0  # 当前行连续可见前段的结束位置。
    diag: I64 = NO_DIAGONAL_TOKEN  # 当前行的单独可见对角位置。
    search_lower = 0  # 段二分查找的下界。
    search_upper = args.segments  # 段二分查找的上界。
    request_segment_word_begin = batch_index * (args.segments + 1)  # 当前记录或块的起始偏移，单位由所在存储视图确定。
    while search_lower < search_upper:
        search_middle = (search_lower + search_upper) // binary_search_divisor  # 段二分查找的中点。
        if query_logical_begin >= args.offsets[request_segment_word_begin + search_middle + 1]:
            search_lower = search_middle + 1  # 段二分查找的下界。
        else:
            search_upper = search_middle  # 段二分查找的上界。
    while search_lower < args.segments and args.offsets[request_segment_word_begin + search_lower] < query_logical_end:
        segment_kv_begin: I64 = args.offsets[request_segment_word_begin + search_lower]  # 当前段的逻辑起点。
        segment_kv_end: I64 = args.offsets[request_segment_word_begin + search_lower + 1]  # 当前区间的结束位置。
        segment_mask_rule = args.rules[search_lower]  # 当前段的掩码规则编号。
        high: I64 = min(query_logical_end, segment_kv_end)  # 当前段对连续前段上界的贡献。
        if segment_mask_rule == MASK_SEGMENT_FULL:
            high = segment_kv_end  # 当前段对连续前段上界的贡献。
        elif segment_mask_rule == MASK_PREFIX_AND_SELF:
            high = segment_kv_begin  # 当前段对连续前段上界的贡献。
            diag = max(diag, min(query_logical_end, segment_kv_end) - 1)  # 当前行的单独可见对角位置。
        cutoff = max(cutoff, high)  # 当前行连续可见前段的结束位置。
        search_lower += 1  # 段二分查找的下界。
    active: I64 = (cutoff + args.key_tile_columns - 1) // args.key_tile_columns  # 当前段覆盖的查询行通道掩码。
    if diag >= 0 and diag // args.key_tile_columns >= active:
        active += 1  # 当前段覆盖的查询行通道掩码。
    return active


def _mtgr_section(
    args: MtgrMetadataArguments,
    section: int,
    section_workgroup_begin: int,
    section_workgroup_end: int,
):
    """累计section总工作量，以单调游标扫描一次任务序列，发出各核半开区间边界。

    Args:
        args: metadata设备接口参数。
        section: 当前 L2 工作集 section 的索引。
        section_workgroup_begin: 当前 section 的起始工作组索引，包含该位置。
        section_workgroup_end: 当前 section 的结束工作组索引，不包含该位置。

    Returns:
        None: 通过对象属性或传入缓冲区保存结果。
    """
    kv_heads = QUERY_HEAD_COUNT // args.group  # KV头的数量。
    work: I64 = 0  # 当前section总工作量。
    for batch_kv_head_index in range(
        section_workgroup_begin, section_workgroup_end
    ):  # batch_kv_head_index，用于本函数的循环或分块定位；按给定区间依次遍历当前处理项。
        batch_index = batch_kv_head_index // kv_heads  # 请求编号。
        for tile in range(
            _mtgr_mtiles(args, batch_index)
        ):  # tile，用于本函数的中间状态或数据视图；按给定区间依次遍历当前处理项。
            work += args.group * _mtgr_task_cost(args, batch_index, tile)  # 当前section总工作量。

    for core in range(args.cores):  # core，用于本函数的中间状态或数据视图；按给定区间依次遍历当前处理项。
        core_metadata_word_begin = (
            METADATA_HEADER_WORDS + (section * args.cores + core) * CORE_METADATA_WORDS
        )  # 当前记录或块的起始偏移，单位由所在存储视图确定。
        args.metadata[core_metadata_word_begin] = section_workgroup_end  # 当前核分配记录中的起始联合工作组编号。
        args.metadata[core_metadata_word_begin + CORE_BEGIN_QUERY_TASK] = 0  # 当前核分配记录中的起始查询任务编号。
        args.metadata[core_metadata_word_begin + CORE_END_WORKGROUP] = (
            section_workgroup_end  # 当前核分配记录中的结束联合工作组编号。
        )
        args.metadata[core_metadata_word_begin + CORE_END_QUERY_TASK] = 0  # 当前核分配记录中的结束查询任务编号。
        args.metadata[core_metadata_word_begin + CORE_SECTION_BEGIN] = (
            section_workgroup_begin  # 当前核分配记录中的section起始工作组。
        )
        args.metadata[core_metadata_word_begin + CORE_SECTION_END] = (
            section_workgroup_end  # 当前核分配记录中的section结束工作组。
        )
    accumulated: I64 = 0  # 已遍历任务的累计工作量。
    boundary = 0  # 待写入的核边界编号。
    for batch_kv_head_index in range(
        section_workgroup_begin, section_workgroup_end
    ):  # batch_kv_head_index，用于本函数的循环或分块定位；按给定区间依次遍历当前处理项。
        batch_index = batch_kv_head_index // kv_heads  # 请求编号。
        tiles: I64 = _mtgr_mtiles(args, batch_index)  # 当前请求的查询块数。
        for task in range(args.group * tiles):  # 当前工作组内的查询任务编号；按给定区间依次遍历当前处理项。
            cost: I64 = _mtgr_task_cost(args, batch_index, task % tiles)  # 当前查询任务的有效工作量。
            task_work_end: I64 = accumulated + cost  # 当前区间的结束位置。
            while boundary < args.cores and work * boundary // args.cores < task_work_end:
                core_metadata_word_begin = (
                    METADATA_HEADER_WORDS + (section * args.cores + boundary) * CORE_METADATA_WORDS
                )  # 当前记录或块的起始偏移，单位由所在存储视图确定。
                args.metadata[core_metadata_word_begin] = batch_kv_head_index  # 当前核分配记录中的起始联合工作组编号。
                args.metadata[core_metadata_word_begin + CORE_BEGIN_QUERY_TASK] = (
                    task  # 当前核分配记录中的起始查询任务编号。
                )
                if boundary > 0:
                    args.metadata[core_metadata_word_begin - CORE_PREVIOUS_END_WORKGROUP_DELTA] = (
                        batch_kv_head_index  # 当前核分配记录中的上一核结束工作组。
                    )
                    args.metadata[core_metadata_word_begin - CORE_PREVIOUS_END_TASK_DELTA] = (
                        task  # 当前核分配记录中的上一核结束查询任务。
                    )
                boundary += 1  # 待写入的核边界编号。
            accumulated = task_work_end  # 已遍历任务的累计工作量。


class MtgrMetadataKernel:
    """组织独立 AICPU metadata 计算入口。

    _mtgr_metadata_kernel 清零计划区，按 L2 预算划分 section，结合活动块工作量计算核内范围，
    最后写入计划头。本类无实例属性，全部运行参数由 MtgrMetadataArguments 传入。
    """
    @staticmethod
    def _mtgr_metadata_kernel(
        args: MtgrMetadataArguments,
    ):
        """清零计划区，按L2预算划分section，计算各核范围并写入metadata头。

        Args:
            args: metadata设备接口参数。

        Returns:
            int: 成功时返回0。
        """
        kv_tensor_count = 2  # 逻辑缓存中需要读取键和值两份张量。
        bf16_element_bytes = 2  # 本函数使用的bf16_element_bytes编译期配置。
        kv_heads = QUERY_HEAD_COUNT // args.group  # KV头的数量。
        for element_index in range(
            METADATA_HEADER_WORDS + args.batch * kv_heads * args.cores * CORE_METADATA_WORDS
        ):  # element_index，用于本函数的循环或分块定位；按给定区间依次遍历当前处理项。
            args.metadata[element_index] = 0  # 初始化的metadata元素。
        section_workgroup_begin = 0  # 段二分查找的下界。
        section = 0  # L2工作集分区编号。
        used: I64 = 0  # 当前section累计工作集字节数。
        for batch_kv_head_index in range(
            args.batch * kv_heads
        ):  # batch_kv_head_index，用于本函数的循环或分块定位；按给定区间依次遍历当前处理项。
            batch_index = batch_kv_head_index // kv_heads  # 请求编号。
            live_sequence_length: I64 = (
                args.starts[batch_index + 1] - args.starts[batch_index]
            )  # 当前请求的实时序列长度。
            logical_kv_length: I64 = live_sequence_length + args.prefixes[batch_index]  # 含前缀的逻辑序列长度。
            footprint: I64 = 0  # 当前工作组的估算工作集字节数。
            if _mtgr_mtiles(args, batch_index) > 0:
                footprint = (
                    bf16_element_bytes
                    * args.head_dimension
                    * (args.group * live_sequence_length + kv_tensor_count * logical_kv_length)
                )  # 当前工作组的估算工作集字节数。
            if (
                args.l2_budget > 0
                and batch_kv_head_index > section_workgroup_begin
                and used + footprint > args.l2_budget
            ):
                _mtgr_section(args, section, section_workgroup_begin, batch_kv_head_index)
                section += 1  # L2工作集分区编号。
                section_workgroup_begin = batch_kv_head_index  # 段二分查找的下界。
                used = 0  # 当前section累计工作集字节数。
            used += footprint  # 当前section累计工作集字节数。
        _mtgr_section(args, section, section_workgroup_begin, args.batch * kv_heads)
        args.metadata[HEADER_MAGIC] = METADATA_MAGIC  # metadata头中的格式识别标记。
        args.metadata[HEADER_SECTION_COUNT] = section + 1  # metadata头中的section数量。
        args.metadata[HEADER_CORE_COUNT] = args.cores  # metadata头中的主kernel核数。
        args.metadata[HEADER_GQA_GROUP] = args.group  # metadata头中的GQA分组倍率。
        args.metadata[HEADER_QUERY_ROWS] = args.query_tile_rows  # metadata头中的查询块行数。
        args.metadata[HEADER_KV_COLUMNS] = args.key_tile_columns  # metadata头中的键值块列数。
        args.metadata[HEADER_KV_HEADS] = kv_heads  # metadata头中的KV头数。
        args.metadata[HEADER_REQUEST_COUNT] = args.batch  # metadata头中的请求数。
        args.metadata[HEADER_HEAD_DIMENSION] = args.head_dimension  # metadata头中的头维度。
        args.metadata[HEADER_L2_BUDGET] = args.l2_budget  # metadata头中的L2预算。
        return 0


class MtgrMetadataHost:
    """管理 metadata 程序的编译缓存和 Host 调用。

    compiled_metadata 生成独立编译源码并缓存临时目录与 AICPU 程序；_prepare_metadata 校验输入、
    分配计划并将任务加入当前流；prepare_metadata 保留兼容关键字入口。
    本类无实例属性，编译资源由静态方法的缓存保持生命周期。
    """
    @staticmethod
    @lru_cache(maxsize=1)
    def compiled_metadata():
        """将类内入口源码规范化为编译器要求的顶层形式，编译并缓存独立AICPU程序及其临时目录。

        Returns:
            tuple: 临时目录对象和已编译AICPU程序。
        """
        directory = tempfile.TemporaryDirectory(prefix="mtgr_metadata_")  # metadata编译临时目录的生命周期对象。
        source_functions = (
            _mtgr_mtiles,
            _mtgr_ktiles,
            _mtgr_task_cost,
            _mtgr_section,
            MtgrMetadataKernel._mtgr_metadata_kernel,
        )  # AICPU入口及其辅助函数，统一生成编译副本。
        source_fragments = []  # 保留仓内文档字符串，仅从AICPU解释器读取的副本移除。
        for source_function in source_functions:
            source_tree = ast.parse(textwrap.dedent(inspect.getsource(source_function)))
            function_node = source_tree.body[0]
            if ast.get_docstring(function_node) is not None:
                function_node.body.pop(0)
            function_node.decorator_list = []
            if source_function is MtgrMetadataKernel._mtgr_metadata_kernel:
                function_node.decorator_list = [ast.Name(id="aicpu_kernel", ctx=ast.Load())]
            source_fragments.append(ast.unparse(ast.fix_missing_locations(source_tree)))
        entry_source = "\n\n".join(source_fragments) + "\n"  # 不包含运行时文档常量的临时设备源码。
        from pathlib import Path

        entry_path = Path(directory.name) / "mtgr_metadata_entry.py"  # 编译器读取的临时源码路径。
        entry_path.write_text(entry_source)
        namespace = dict(globals())  # 设备入口及辅助函数的编译命名空间。
        exec(compile(entry_source, str(entry_path), "exec"), namespace)
        compiled = compile_aicpu_kernel(
            namespace["_mtgr_metadata_kernel"], workdir=directory.name, launch_mode="interface"
        )  # 已编译AICPU任务。
        return directory, compiled

    @staticmethod
    def _prepare_metadata(
        starts: torch.Tensor,
        prefixes: torch.Tensor,
        offsets: torch.Tensor,
        *,
        total: int,
        head_dimension: int,
        group: int,
        cores: int,
        table_cols: int,
        query_tile_rows: int = CACHE_BLOCK_TOKENS,
        key_tile_columns: int = CACHE_BLOCK_TOKENS,
        l2_budget: int = DEFAULT_L2_BUDGET,
        rules: torch.Tensor | None = None,
    ):
        """检查调度入参与设备，将独立AICPU任务加入当前流，记录张量流生命周期并返回设备侧计划。

        Args:
            starts: 各请求实时token起点。
            prefixes: 各请求命中前缀长度。
            offsets: 各请求分段偏移。
            total: 所有请求实时查询 token 的总数。
            head_dimension: 每个注意力头的维度。
            group: 每个 KV 头对应的查询头数。
            cores: attention 主 kernel 使用的 Cube 核数。
            table_cols: 每个请求的页表容量。
            query_tile_rows: 完整 Cube 查询块的行数。
            key_tile_columns: 每个键值块的列数。
            l2_budget: metadata 划分 section 使用的 L2 工作集预算，单位为字节。
            rules: 各段掩码规则。

        Returns:
            torch.Tensor: 设备侧INT32调度计划。

        Raises:
            ValueError: 输入设备、请求或分段规格、GQA 分组、头维度、核数、分块或 L2 预算不合法。
        """
        tensor_matrix_rank = 2  # 本函数使用的tensor_matrix_rank编译期配置。
        metadata_int32_exclusive_limit = 2**31  # INT32输入元素数与预算的独占上界。
        if starts.device.type not in ("npu", "privateuseone"):
            raise ValueError("Metadata inputs must reside on an NPU device; CPU tensors are not supported.")
        batch = prefixes.numel()  # 请求数量。
        if (
            not 1 <= batch <= MAX_REQUESTS
            or starts.shape != (batch + 1,)
            or prefixes.shape != (batch,)
            or offsets.ndim != tensor_matrix_rank
            or offsets.shape[0] != batch
            or offsets.shape[1] < MIN_SEGMENT_OFFSET_WORDS
        ):
            raise ValueError(
                "Metadata inputs must describe 1 to 8 requests, batch + 1 sequence starts, batch prefix lengths, and at least two segments."
            )
        if (
            group not in SUPPORTED_GQA_GROUPS
            or head_dimension not in SUPPORTED_HEAD_DIMENSIONS
            or not 1 <= cores <= MAX_CUBE_CORES
        ):
            raise ValueError(
                "Metadata requires GQA group 1, 2, 4, or 8, head dimension 32, 64, or 128, and 1 to 32 Cube cores."
            )
        if (
            query_tile_rows not in SUPPORTED_QUERY_TILE_ROWS
            or key_tile_columns != DEFAULT_KV_TILE_COLUMNS
            or not 0 <= l2_budget < metadata_int32_exclusive_limit
            or not 0 < total < metadata_int32_exclusive_limit
        ):
            raise ValueError(
                "Metadata requires query tile 64 or 128, KV tile 128, a nonnegative L2 budget below 2**31, and a positive query token count below 2**31."
            )
        if any(
            weighted_values.dtype != torch.int32
            or weighted_values.device != starts.device
            or not weighted_values.is_contiguous()
            for weighted_values in (starts, prefixes, offsets)
        ):
            raise ValueError(
                "Sequence starts, prefix lengths, and segment offsets must be contiguous INT32 tensors on the same NPU device."
            )
        if rules is not None and (
            rules.shape != (offsets.shape[1] - 1,)
            or rules.dtype != torch.int32
            or rules.device != starts.device
            or not rules.is_contiguous()
        ):
            raise ValueError(
                "Segment rules must be contiguous INT32 on the metadata device with one entry per segment."
            )
        rule_input = starts if rules is None else rules  # 传入AICPU的规则张量或占位张量。
        size = (
            METADATA_HEADER_WORDS + batch * (QUERY_HEAD_COUNT // group) * METADATA_CORE_CAPACITY * CORE_METADATA_WORDS
        )  # metadata输出分配的INT32元素数。
        output = torch.empty(size, dtype=torch.int32, device=starts.device)  # 设备输出张量。
        _, compiled = MtgrMetadataHost.compiled_metadata()  # 调用方无需使用的返回项；已编译AICPU任务。
        device_id = starts.device.index  # 当前输入所在NPU编号。
        compiled.launch(
            current_raw_stream(device_id),
            starts=starts.data_ptr(),
            prefixes=prefixes.data_ptr(),
            offsets=offsets.data_ptr(),
            metadata=output.data_ptr(),
            rules=rule_input.data_ptr(),
            sparse_weights=0 if rules is None else 1,
            batch=batch,
            segments=offsets.shape[1] - 1,
            total=total,
            table_cols=table_cols,
            group=group,
            head_dimension=head_dimension,
            query_tile_rows=query_tile_rows,
            key_tile_columns=key_tile_columns,
            cores=cores,
            l2_budget=l2_budget,
        )
        stream = torch.npu.current_stream(device_id)  # 当前设备执行流。
        for tensor in (
            starts,
            prefixes,
            offsets,
            rule_input,
            output,
        ):  # 当前需要检查或记录流生命周期的张量；按给定区间依次遍历当前处理项。
            tensor.record_stream(stream)
        return output

    @staticmethod
    def prepare_metadata(
        starts: torch.Tensor,
        prefixes: torch.Tensor,
        offsets: torch.Tensor,
        *,
        total: int,
        d: int,
        group: int,
        cores: int,
        table_cols: int,
        m: int = CACHE_BLOCK_TOKENS,
        n: int = CACHE_BLOCK_TOKENS,
        l2_budget: int = DEFAULT_L2_BUDGET,
        rules: torch.Tensor | None = None,
    ):
        """检查调度入参与设备，将独立AICPU任务加入当前流，记录张量流生命周期并返回设备侧计划。

        Args:
            starts: 各请求实时token起点。
            prefixes: 各请求命中前缀长度。
            offsets: 各请求分段偏移。
            total: 所有请求实时查询 token 的总数。
            d: 兼容关键字参数，内部作为 head_dimension 使用。
            group: 每个 KV 头对应的查询头数。
            cores: attention 主 kernel 使用的 Cube 核数。
            table_cols: 每个请求的页表容量。
            m: 兼容关键字参数，内部作为 query_tile_rows 使用。
            n: 兼容关键字参数，内部作为 key_tile_columns 使用。
            l2_budget: metadata 划分 section 使用的 L2 工作集预算，单位为字节。
            rules: 各段掩码规则。

        Returns:
            torch.Tensor: 设备侧INT32调度计划。

        Raises:
            ValueError: 传给 _prepare_metadata 的输入或调度属性不满足其校验要求。
        """
        return MtgrMetadataHost._prepare_metadata(
            starts,
            prefixes,
            offsets,
            total=total,
            head_dimension=d,
            group=group,
            cores=cores,
            table_cols=table_cols,
            query_tile_rows=m,
            key_tile_columns=n,
            l2_budget=l2_budget,
            rules=rules,
        )
