# Transpose + BMM + Transpose + ReLU 融合算子需求文档

## 1. 背景与目标

网络中存在如下固定子图结构(典型 dtype 为 fp16):

```
Transpose1 → BMM(+bias) → Transpose2 → ReLU
```

- **Transpose1**: `(m, batch, k) → (batch, m, k)`,perm = `[1, 0, 2]`
- **BMM**: `(batch, m, k) × (batch, k, n) + (batch, 1, n) → (batch, m, n)`,带 bias 的批量矩阵乘,bias 沿 m 维广播
- **Transpose2**: `(batch, m, n) → (m, batch, n)`,perm = `[1, 0, 2]`
- **ReLU**: `(m, batch, n) → (m, batch, n)`

当前 4 个算子分立执行,两个 Transpose 为纯搬数、ReLU 为纯逐元素操作,额外引入 ~25 us 访存开销与 3 次 kernel 启动/调度开销。

**目标**:实现单个融合算子(暂名 `TransposeBmmBiasRelu`),一次 kernel 完成全部计算,消除 Transpose/ReLU 的独立访存与调度开销。

## 2. 算子说明

### 计算公式

```
out[i, b, j] = ReLU( Σ_c A[i, b, c] · B[b, c, j] + bias[b, j] )
```

等价的分 batch 视角:对每个 `b ∈ [0, batch)`:

```
Out[:, b, :] = ReLU( A[:, b, :] @ B[b] + bias[b] )
```

其中 `A[:, b, :]` 为 `(m, k)` 子矩阵(行 stride = batch·k), `B[b]` 为 `(k, n)`, `bias[b]` 为 `(1, n)` 沿 m 广播。

> `transpose_b = true` 时,B 为转置 layout `[batch, n, k]`:`B[b]` 为 `(n, k)`,公式中 `B[b, c, j]` 对应存储位置 `B[b, j, c]`,即计算 `A[:, b, :] @ B[b]ᵀ`,其余语义不变。

### 融合要点

- Transpose1 / Transpose2 均为纯 layout 变换,不产生计算:融合算子直接按 `(m, batch, k)` layout 读 A、按 `(m, batch, n)` layout 写 out,矩阵乘按 batch 维以 stride 方式取子矩阵;
- bias 加法与 ReLU 作为矩阵乘 epilogue 融合,不落中间结果。

### 输入

| 参数     | dtype       | shape           | format | 动态维         | 说明                        |
| ------ | ----------- | --------------- | ------ | ----------- | ------------------------- |
| `A`    | FP16 / BF16 | `[m, batch, k]` | ND     | m, batch, k | 激活输入(Transpose1 已融合,无需转置) |
| `B`    | FP16 / BF16 | `[batch, k, n]` 或 `[batch, n, k]` (转置) | ND     | batch, k, n | 批量权重,需支持两种 layout        |
| `bias` | FP16 / BF16 | `[batch, 1, n]` | ND     | batch, n    | 偏置,沿 m 维广播                |
| `transpose_b` | BOOL | 标量(attr) | -      | 无(编译期常量) | B 是否为转置 layout `[batch, n, k]`,默认 false |

> batch、m、k、n 均为运行时动态,**要求全部泛化**(动态 shape)。本组 profiling:m 在 1 ~ 85 变化,batch 仅出现 8,k、n 固定 2048。

### 输出

| dtype | shape | format |
|-------|-------|--------|
| FP16 / BF16(同输入) | `[m, batch, n]` | ND(Transpose2 + ReLU 已融合) |

## 3. 现状与性能(profiling 数据)

### 典型 case 分算子耗时(fp16,batch=8,k=n=2048,m=85)

| 算子         | 输入 → 输出                                          | 耗时 (us)   |
| ---------- | ------------------------------------------------ | --------- |
| Transpose1 | `"85,8,2048;3" → "8,85,2048"`                    | 12.5      |
| BMM        | `"8,85,2048;8,2048,2048;8,1,2048" → "8,85,2048"` | 61        |
| Transpose2 | `"8,85,2048;3" → "85,8,2048"`                    | 9.6       |
| ReLU       | `"85,8,2048" → "85,8,2048"`                      | 3.5       |
| **合计**     |                                                  | **~86.6** |

### 端到端基线(分立算子,fp16,batch=8,k=n=2048)

| m | 耗时 (us) |
|---|-----------|
| 1 | 70 |
| 32 | 73 |
| 41 | 79 |
| 50 | 78 |
| 85 | 86 |

## 4. 需求与验收标准

- **优先级**:典型 case(batch=8,k=n=2048,m ∈ {1, 32, 41, 50, 85})为 **P0**(性能优先);其余泛化规格为 **P1**(功能正确优先);
- **dtype**:FP16 / BF16;
- **功能**:单 kernel 实现融合语义,B 支持 `[batch, k, n]` / `[batch, n, k]` 两种 layout,以编译期属性 `transpose_b`(BOOL,默认 false)区分;P0 全 m 规格 + 泛化规格回归通过;profiling 中该子图仅出现 1 个融合 kernel,无独立 Transpose / ReLU;
- **性能**: 融合算子耗时 ≤ BMM 单算子 + 0.2 * `Transpose1 + Transpose2` (m=85 时 ≤ 66 us)
- **精度**:与分立算子基线(Transpose + BatchMatMul + BiasAdd + ReLU)对齐,满足 fp16 / bf16 精度验收标准。
