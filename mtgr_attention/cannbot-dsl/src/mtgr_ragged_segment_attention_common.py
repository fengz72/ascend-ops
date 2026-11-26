# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""文件说明：多个Host/设备函数共享的硬件规格、分段规则、metadata ABI与接口默认值。
函数专用的槽数、展开因子和任务队列字段在各自函数内声明；0/1的数学单位和结构性步进保留。
"""

QUERY_HEAD_COUNT = 8  # 接口固定的查询头数量。
MAX_REQUESTS = 8  # 接口允许的最大请求数。
METADATA_CORE_CAPACITY = 36  # 固定计划容量，与实际发射核数解耦。
MAX_CUBE_CORES = 32  # 支持的最大Cube核数。
CACHE_BLOCK_TOKENS = 128  # 分页缓存每个物理页的token数量。
DEFAULT_QUERY_TILE_ROWS = 128  # 接口默认值或编译期规格；在函数定义/编译前保持固定。
DEFAULT_KV_TILE_COLUMNS = 128  # 接口默认值或编译期规格；在函数定义/编译前保持固定。
DEFAULT_CORE_COUNT = 32  # 接口默认值或编译期规格；在函数定义/编译前保持固定。
DEFAULT_COMPILE_CORE_COUNT = 24  # 旧辅助入口的兼容默认值；动态编译缓存不使用此值。
DEFAULT_PIPELINE_STAGES = 14  # 接口默认值或编译期规格；在函数定义/编译前保持固定。
DEFAULT_CROSS_INTERVAL = 2  # 接口默认值或编译期规格；在函数定义/编译前保持固定。
FLOAT32_BITS = 32  # FP32元素的位宽。
BF16_BITS = 16  # BF16元素的位宽。
VECTOR_FLOAT32_LANES = 64  # 单条FP32向量指令的有效通道数。
NZ_BLOCK_EDGE = 16  # NZ分形内边长。
NZ_BLOCK_ELEMENTS = 256  # 一个16×16 NZ分形的元素数。
AIVS_PER_CUBE = 2  # 每个Cube核对应的Vector子核数。
ONLINE_STATE_SLOTS = 3  # 跨查询任务的在线softmax状态组数。
METADATA_HEADER_WORDS = 16  # metadata头的INT32字段数。
CORE_METADATA_WORDS = 16  # 每核metadata记录的INT32字段数。
METADATA_MAGIC = 1297368914  # metadata ABI识别标记。
MASK_CAUSAL = 0  # 分段mask规则编号。
MASK_SEGMENT_FULL = 1  # 分段mask规则编号。
MASK_PREFIX_AND_SELF = 2  # 分段mask规则编号。
NO_DIAGONAL_TOKEN = -1  # 不存在单独可见对角token时的哨兵。
MASKED_SCORE_FLOOR = -1e30  # 遮蔽分数的有限负哨兵。
FLOAT32_MAX_VALUE = 3.4028234663852886e38  # 用于缩放路径选择的FP32最大有限值。
ATTENTION_COMPILE_CACHE_SIZE = 64  # 接口默认值或编译期规格；在函数定义/编译前保持固定。
MIN_SEGMENTS = 2  # 接口默认值或编译期规格；在函数定义/编译前保持固定。
MIN_SEGMENT_OFFSET_WORDS = 3  # 接口默认值或编译期规格；在函数定义/编译前保持固定。
SND_TENSOR_RANK = 3  # 接口默认值或编译期规格；在函数定义/编译前保持固定。
INT32_VALUE_BITS = 31  # 接口默认值或编译期规格；在函数定义/编译前保持固定。
DEFAULT_L2_BUDGET = 100663296  # 默认L2调度工作集预算，单位字节。
HEADER_MAGIC = 0  # metadata头字段的字偏移。
HEADER_SECTION_COUNT = 1  # metadata头字段的字偏移。
HEADER_CORE_COUNT = 2  # metadata头字段的字偏移。
HEADER_GQA_GROUP = 3  # metadata头字段的字偏移。
HEADER_QUERY_ROWS = 4  # metadata头字段的字偏移。
HEADER_KV_COLUMNS = 5  # metadata头字段的字偏移。
HEADER_KV_HEADS = 6  # metadata头字段的字偏移。
HEADER_REQUEST_COUNT = 7  # metadata头字段的字偏移。
HEADER_HEAD_DIMENSION = 8  # metadata头字段的字偏移。
HEADER_L2_BUDGET = 9  # metadata头字段的字偏移。
CORE_BEGIN_WORKGROUP = 0  # 核范围记录的字偏移或相邻记录距离。
CORE_BEGIN_QUERY_TASK = 1  # 核范围记录的字偏移或相邻记录距离。
CORE_END_WORKGROUP = 2  # 核范围记录的字偏移或相邻记录距离。
CORE_END_QUERY_TASK = 3  # 核范围记录的字偏移或相邻记录距离。
CORE_SECTION_BEGIN = 4  # 核范围记录的字偏移或相邻记录距离。
CORE_SECTION_END = 5  # 核范围记录的字偏移或相邻记录距离。
CORE_PREVIOUS_END_WORKGROUP_DELTA = 14  # 核范围记录的字偏移或相邻记录距离。
CORE_PREVIOUS_END_TASK_DELTA = 13  # 核范围记录的字偏移或相邻记录距离。
SUPPORTED_HEAD_DIMENSIONS = (32, 64, 128)  # Host与设备共同使用的合法属性集合。
SUPPORTED_QUERY_TILE_ROWS = (64, 128)  # Host与设备共同使用的合法属性集合。
SUPPORTED_GQA_GROUPS = (1, 2, 4, 8)  # Host与设备共同使用的合法属性集合。
SUPPORTED_KV_HEADS = (1, 2, 4, 8)  # Host与设备共同使用的合法属性集合。
SUPPORTED_MATCH_MODES = (0, 1, 2)  # Host与设备共同使用的合法属性集合。
SUPPORTED_RULE1_INDICES = (0, 1)  # Host与设备共同使用的合法属性集合。

SMALL_HEAD_DIMENSION = 32  # 接口默认值或编译期规格；在函数定义/编译前保持固定。

ROW_RULE_PARAMETER_PLANES = 2  # 每行掩码缓存包含连续前段上界和对角位置两组参数。

VECTOR_QUERY_ROWS = 64  # M128由两个AIV分片，每个AIV的行状态跨度为64。
VECTOR_BF16_LANES = 128  # 一个完整BF16向量寄存器的元素数。
