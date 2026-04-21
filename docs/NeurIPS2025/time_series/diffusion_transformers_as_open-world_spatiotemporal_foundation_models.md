---
title: >-
  [论文解读] Diffusion Transformers as Open-World Spatiotemporal Foundation Models
description: >-
  [NEURIPS2025][时间序列][Transformer] 提出 UrbanDiT，首个基于 Diffusion Transformer 的开放世界城市时空基础模型，通过统一的 prompt learning 框架整合异构数据类型（grid/graph）和多种任务（预测/插值/外推/填补），在多城市多场景下实现 SOTA 性能并展现强大的 zero-shot 泛化能力。
tags:
  - NEURIPS2025
  - 时间序列
  - Transformer
  - Spatiotemporal Foundation Model
  - Urban Computing
  - 提示学习
  - Zero-shot
---

# Diffusion Transformers as Open-World Spatiotemporal Foundation Models

**会议**: NEURIPS2025  
**arXiv**: [2411.12164](https://arxiv.org/abs/2411.12164)  
**代码**: [tsinghua-fib-lab/UrbanDiT](https://github.com/tsinghua-fib-lab/UrbanDiT)  
**领域**: time_series  
**关键词**: Diffusion Transformer, Spatiotemporal Foundation Model, Urban Computing, Prompt Learning, Zero-shot

## 一句话总结
提出 UrbanDiT，首个基于 Diffusion Transformer 的开放世界城市时空基础模型，通过统一的 prompt learning 框架整合异构数据类型（grid/graph）和多种任务（预测/插值/外推/填补），在多城市多场景下实现 SOTA 性能并展现强大的 zero-shot 泛化能力。

## 背景与动机
城市环境中的时空动态来源于多样化的人类活动，体现为不同类型的数据：grid-based 数据（如人群流量）和 graph-based 数据（如道路网络交通速度）。现有的城市时空模型存在明显局限：

- **UniST**：仅支持 grid-based 数据，只做预测任务
- **UrbanGPT**：基于 LLM，仅支持 grid-based 数据，任务单一
- **GPD**：仅处理 graph-based 交通数据，不支持多数据源和 zero-shot
- **CityGPT**：基于 LLM 处理语言形式的城市数据，不支持多数据源和 zero-shot

核心痛点在于：**缺乏一个能同时处理异构数据类型、支持多种时空任务、并在开放世界场景下泛化的统一基础模型**。Diffusion Transformer（如 Sora）结合了扩散模型的生成能力和 Transformer 的可扩展性，为解决这一问题提供了理想的骨架。

## 核心问题
能否构建一个类似 NLP/CV 领域的基础模型，学习通用时空模式，成为城市时空应用的通用模型？具体需解决三个子问题：

1. 如何统一 grid-based 和 graph-based 异构数据格式？
2. 如何用单一模型支持预测、插值、外推、填补等多种任务？
3. 如何在未见过的城市/数据集上实现 zero-shot 泛化？

## 方法详解

### 数据统一：异构时空数据转序列
将三维结构（2D 空间 + 1D 时间）的数据统一为一维序列格式：

- **时间维度**：采用 temporal patching（类似 PatchTST），将时间序列切分为 patch
- **Grid-based 数据**：使用 2D 空间 patching（类似 ViT），将 $H \times W \times T$ 重组为序列 $L = \frac{H \times W \times T}{p_s \times p_s \times p_t}$
- **Graph-based 数据**：使用 GCN 处理每个节点，再与时间维度整合为一维序列

### 任务统一：基于 Mask 的多任务框架
将所有任务统一为「重建被 mask 部分」的范式，通过不同 mask 策略实现：

| 任务 | Mask 策略 |
|------|-----------|
| Forward Prediction | mask 未来时间步 |
| Backward Prediction | mask 过去时间步 |
| Temporal Interpolation | mask 序列中特定时间点 |
| Spatial Extrapolation | mask 观测区域外的空间位置 |
| Spatio-Temporal Imputation | 随机 mask 时空位置 |

输入表示为：$X^t = X^t \ast (1-M) + X^0 \ast M$，其中 $M$ 控制不同任务的条件信息。

### Spatio-Temporal Transformer Block
模型由多个时空 Transformer block 堆叠而成，每个 block 包含独立的 temporal attention 和 spatial attention。分离设计避免了联合注意力的二次复杂度问题。

### Unified Prompt Learning（核心创新）

**Data-Driven Prompt**：使用三组 memory pool（key-value store），分别捕获：

- **时域模式**：$(K_t, V_t)$，基于输入的时域特征检索
- **频域模式**：$(K_f, V_f)$，基于输入的频域特征检索
- **空间模式**：$(K_s, V_s)$，基于输入的空间特征检索

通过 softmax 注意力从 memory pool 中检索最相关的 prompt，类似 RAG 的思想，帮助模型区分不同数据集的分布差异。

**Task-Specific Prompt**：从 mask map 生成任务特定的 prompt $P_m = \text{Attention}(\text{Flatten}(M))$，让模型感知当前执行的任务类型。

所有 prompt 与输入序列拼接后送入 Transformer，利用 Transformer 处理变长序列的能力实现灵活输入。

### 训练策略
- 每次迭代随机选择一个数据集和一个任务进行梯度下降
- 采用 InstaFlow 的 rectified flow 训练方法（ODE 框架，直线化噪声-数据轨迹），提升生成效率
- 训练时 diffusion steps 设为 500，推理时仅需 20 步（25 倍加速）

## 实验关键数据

### 数据集
涵盖 5 个城市（纽约、北京、上海、南京等）的多领域数据集：出租车需求、蜂窝网络流量、人群流动、交通、动态人口。数据集按 6:2:2 时间划分。

### Forward Prediction（Grid-based，表 2）

| 模型 | TaxiBJ MAE | FlowSH MAE | TaxiNYC MAE | CrowdNJ MAE | PopBJ MAE |
|------|-----------|------------|------------|------------|----------|
| UniST | 14.04 | 9.10 | 5.85 | 0.119 | 0.106 |
| CSDI | 14.76 | 8.77 | 5.05 | 0.094 | 0.078 |
| **UrbanDiT** | **12.61** | **5.61** | **5.58** | **0.092** | **0.077** |

整体相对提升 **11.3%**，反向预测相对提升 **30.4%**。

### Spatial Extrapolation（表 3，50% 空间位置被 mask）
UrbanDiT 大幅超越所有基线，例如 TaxiBJ 上 MAE 从 CSDI 的 36.66 降至 **8.10**。

### Zero-shot 性能
在 PopSH 数据集上，UrbanDiT 的 zero-shot 性能超越了几乎所有使用训练数据的基线模型，验证了开放世界泛化能力。

### 消融实验
去除任一 prompt 均显著降低性能；去除所有 prompt 后性能最差；**频域 prompt 影响最大**。

### 可扩展性
UrbanDiT-L 在数据量从 0.8 增加到 1.0 时展现了最陡的性能提升斜率（0.011 vs 0.0015/0.0019），表明大模型在更多数据下仍有提升空间。

## 亮点
1. **真正的统一模型**：首次同时支持 grid/graph 数据 + 5 种时空任务 + 多城市多领域数据
2. **Prompt Learning 设计精巧**：数据驱动 prompt（三域 memory pool）+ 任务特定 prompt 的组合，灵活且有效
3. **强大的 zero-shot 能力**：无需任何目标数据训练即超越大多数有监督基线
4. **高效推理**：rectified flow + 20 步推理实现 25 倍加速，兼顾质量与效率
5. **可扩展性良好**：大模型随数据增长持续受益，符合 foundation model 的 scaling law

## 局限与展望
1. **数据覆盖有限**：目前仅关注人类活动数据（出行、交通），未覆盖环境变量（空气污染、气候指标、微气候动态）
2. **计算开销**：Diffusion Transformer 的训练和推理成本仍较高，尤其在大规模城市数据上
3. **Prompt 设计依赖先验**：三组 memory pool 的结构（时域/频域/空间）是人工设计的，可以探索自动化 prompt 发现
4. **空间外推的上限**：50% mask 率下的空间外推虽然大幅领先但极端稀疏场景未验证
5. **仅支持数值型数据**：未考虑文本、图像等多模态城市数据

## 与相关工作的对比

| 维度 | UniST | UrbanGPT | GPD | UrbanDiT |
|------|-------|----------|-----|----------|
| 模型初始化 | 从头训练 | LLM | 从头训练 | 从头训练 |
| 数据类型 | Grid | Grid | Graph | Grid + Graph |
| 多数据源 | ✓ | ✓ | ✗ | ✓ |
| 任务灵活性 | ✗ | ✗ | ✗ | ✓ |
| Zero-shot | ✓ | ✓ | ✗ | ✓ |

UrbanDiT 是唯一在所有五个维度上都满足的模型。与 CSDI（第二名基线）相比，UrbanDiT 不仅在 CSDI 擅长的扩散生成任务上全面超越，还支持 CSDI 无法处理的多数据类型和多任务。

## 启发与关联
- **Prompt Learning 的通用价值**：数据驱动 prompt 通过 memory pool 检索的方式，可迁移到其他需要统一异构数据的基础模型场景
- **Mask-based 任务统一**：将预测/插值/外推/填补统一为 mask 重建的思路，类似 MAE 在视觉中的做法，值得在更多领域推广
- **Rectified Flow 加速**：ODE 直线化轨迹的训练策略可显著降低扩散模型推理步数，适合计算受限的实际部署
- **城市计算 × Foundation Model**：随着城市数据的持续积累，领域专用基础模型的价值将越来越大

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次将 DiT 扩展到城市时空统一建模，prompt learning 框架设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ — 5 个城市多领域数据集、5 种任务、few-shot/zero-shot、消融、可扩展性实验全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，方法阐述完整
- 价值: ⭐⭐⭐⭐ — 为城市时空基础模型树立了新 benchmark，开源代码和数据集有利于后续研究

<!-- RELATED:START -->

## 相关论文

- [Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)
- [Relational Feature Caching for Accelerating Diffusion Transformers](../../ICLR2026/time_series/relational_feature_caching_for_accelerating_diffusion_transformers.md)
- [How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)

<!-- RELATED:END -->
