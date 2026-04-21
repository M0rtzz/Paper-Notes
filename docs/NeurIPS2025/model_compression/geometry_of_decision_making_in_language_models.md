---
title: >-
  [论文解读] Geometry of Decision Making in Language Models
description: >-
  [NeurIPS 2025][模型压缩][内在维度] 通过在 28 个开源 Transformer 模型上大规模测量各层隐藏表示的**内在维度（Intrinsic Dimension, ID）**，揭示了一致的"低-高-低"维度变化模式：早期层在低维流形上操作，中间层扩展空间，后期层再压缩至与决策相关的低维表示。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 内在维度
  - 隐藏表示几何
  - 决策动态
  - 多选问答
  - Transformer
---

# Geometry of Decision Making in Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.20315](https://arxiv.org/abs/2511.20315)  
**代码**: 无  
**领域**: LLM 可解释性 / 表示学习  
**关键词**: 内在维度, 隐藏表示几何, 决策动态, 多选问答, Transformer

## 一句话总结

通过在 28 个开源 Transformer 模型上大规模测量各层隐藏表示的**内在维度（Intrinsic Dimension, ID）**，揭示了一致的"低-高-低"维度变化模式：早期层在低维流形上操作，中间层扩展空间，后期层再压缩至与决策相关的低维表示。

## 研究背景与动机

### 核心问题

大语言模型（LLM）在多种任务上展现出强大的泛化能力，但其内部的决策过程——即模型如何从输入一步步得出预测——仍然不透明。已有工作从注意力机制、probe 分析等角度研究模型内部机制，但对隐藏表示的**几何结构**研究较少。

### 内在维度（Intrinsic Dimension）

内在维度是一种度量高维数据点集所在流形真实维度的统计量。直觉上，即使隐藏层有 $d = 4096$ 维，这些表示向量可能实际集中在一个远低于 $d$ 维的子流形上。ID 可以揭示模型各层对信息压缩和扩展的程度。

### 为何选择 MCQA 设置

多选问答（MCQA）任务提供了明确的决策结构：模型需要在有限个选项中选出正确答案。这使得研究者可以：
- 量化每层对最终决策的贡献（逐层准确率）
- 将 ID 变化与决策质量关联
- 控制实验变量，避免开放生成任务的不确定性

## 方法详解

### 整体框架

实验设计如下：
1. 选取 28 个开源 Transformer 模型（不同架构、不同参数量）
2. 在 MCQA 任务上输入测试数据
3. 提取每层的隐藏表示
4. 使用多种 ID 估计器计算每层的内在维度
5. 同时计算每层的 MCQA 准确率（通过在该层输出上直接做分类）
6. 分析 ID 与逐层性能的关系

### 关键设计

#### ID 估计方法

作者使用多种 ID 估计器以确保结论的鲁棒性：

| 估计器 | 类型 | 原理 |
|:---|:---|:---|
| TwoNN | 局部方法 | 基于最近邻距离比 |
| MLE (Levina-Bickel) | 局部方法 | 最大似然估计 |
| PCA (explained variance) | 全局方法 | 解释方差比例 |
| 其他拓扑方法 | 混合 | 基于持久同调等 |

使用多种估计器的好处是避免单一方法的偏差，增强结论可信度。

#### 逐层性能量化

对于每一层 $l$，将隐藏表示 $h^{(l)}$ 直接用于预测：
- 计算各选项的表示相似度或使用线性探针（linear probe）
- 得到该层的 MCQA 准确率 $\text{Acc}^{(l)}$
- 建立 $\text{ID}^{(l)}$ 和 $\text{Acc}^{(l)}$ 的对应关系

### 训练策略

本文不涉及训练新模型。所有分析基于预训练的现有模型，属于**分析性研究**。

## 实验关键数据

### 主实验：ID 变化模式

在所有 28 个模型上，一致地观察到以下三阶段模式：

| 层区间 | ID 行为 | 解释 |
|:---|:---|:---|
| 早期层（0-20%深度） | 低 ID | 输入嵌入在低维流形上，初始编码紧凑 |
| 中间层（20-70%深度） | ID 上升至峰值 | 空间扩展，模型探索丰富表示 |
| 后期层（70-100%深度） | ID 再次下降 | 压缩至与决策相关的低维结构 |

| 模型类别 | 典型模型 | 早期 ID | 峰值 ID | 最终 ID |
|:---|:---|:---|:---|:---|
| 小模型 (~1B) | Pythia-1B, GPT-Neo-1.3B | ~10-20 | ~40-60 | ~15-25 |
| 中模型 (~7B) | LLaMA-2-7B, Mistral-7B | ~15-30 | ~80-120 | ~20-40 |
| 大模型 (~13B+) | LLaMA-2-13B, Falcon-40B | ~20-40 | ~100-150 | ~30-50 |

### 消融实验

#### ID 与逐层性能的关系

| 层区间 | 平均 MCQA 准确率 | ID 趋势 | 关系 |
|:---|:---|:---|:---|
| 早期层 | 接近随机（~25%） | 低 ID | 信息尚未整合 |
| 中间偏后层 | 快速上升 | ID 开始下降 | 决策开始形成 |
| 最后几层 | 最高 | 低 ID | 决策已压缩至低维 |

关键发现：**ID 下降与准确率上升高度相关**，即模型在做出决策时恰好将表示压缩至低维流形。

#### 不同估计器的一致性

| 估计器对 | 秩相关系数 (Spearman) |
|:---|:---|
| TwoNN vs MLE | > 0.95 |
| TwoNN vs PCA | > 0.90 |
| MLE vs PCA | > 0.88 |

不同估计器给出高度一致的 ID 趋势，验证了结论的鲁棒性。

### 关键发现

1. **通用的"低-高-低" ID 模式**：在所有 28 个模型和多种 ID 估计器下都一致出现，这是一个架构和规模无关的特性。

2. **ID 压缩与决策形成同步**：最后几层 ID 急剧下降时，对应着 MCQA 准确率的快速提升，表明模型在后期层将表示投影到与任务决策对齐的结构化低维流形。

3. **模型规模的影响**：更大的模型倾向于有更高的峰值 ID，说明它们在中间层具有更丰富的表示空间，但最终仍压缩到相对低维的决策流形。

## 亮点与洞察

- **几何视角的新颖性**：不同于探针或注意力分析，ID 分析提供了一种更本质的、与具体任务无关的几何度量
- **28 模型的大规模验证**：覆盖了 Pythia, LLaMA, Mistral, Falcon, GPT-Neo 等多种架构和规模
- **对"表示学习即维度选择"的支持**：结果暗示 LLM 的学习过程可以理解为在高维空间中找到正确的低维流形
- **对层剪枝/早停的启示**：如果后期层主要在做维度压缩，那么可能有更高效的方式实现这一步

## 局限与展望

- 仅在 MCQA 场景下验证，开放生成任务中的 ID 模式是否一致尚未验证
- ID 估计在有限样本下存在统计噪声，尤其对极高维表示
- 未探究微调（Fine-tuning）或 RLHF 如何改变 ID 模式
- 未与 probing 准确率或信息瓶颈理论做定量对比
- 未涉及因果分析（ID 变化是决策的原因还是结果）

## 相关工作与启发

- **Ansuini et al. (2019)**：首次在深度网络中系统研究 ID 变化模式
- **Cai et al. (2023)**：在 Vision Transformer 中分析 ID
- **信息瓶颈理论 (Shwartz-Ziv & Tishby, 2017)**：认为深度学习分为"拟合"和"压缩"两个阶段，本文的 ID 模式与此一致
- **机制可解释性**：Elhage et al., Olsson et al. 等从电路角度分析 Transformer
- 本文从几何角度补充了对 LLM 内部工作机制的理解

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 大规模 ID 分析在 LLM 中是新方向
- **技术深度**: ⭐⭐⭐ — 实验设计扎实但方法本身较直接
- **实用性**: ⭐⭐⭐ — 分析性工作，对模型压缩和可解释性有启发
- **清晰度**: ⭐⭐⭐⭐ — 结论直观清晰
- **综合评分**: 7.5/10

<!-- RELATED:START -->

## 相关论文

- [Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making](online_mixture_of_experts_no-regret_learning_for_optimal_collective_decision-mak.md)
- [Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)
- [Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)
- [The Structure of Relation Decoding Linear Operators in Large Language Models](the_structure_of_relation_decoding_linear_operators_in_large_language_models.md)

<!-- RELATED:END -->
