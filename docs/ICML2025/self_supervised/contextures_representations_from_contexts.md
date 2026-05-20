---
title: >-
  [论文解读] Contextures: Representations from Contexts
description: >-
  [ICML 2025][自监督学习][表示学习理论] 建立 contexture 理论，统一证明监督学习、自监督学习和流形学习等多种表示学习范式都可被理解为学习上下文变量诱导的期望算子的 top-$d$ 奇异函数，并揭示模型规模增大的边际递减效应以及提出上下文质量评估指标。
tags:
  - "ICML 2025"
  - "自监督学习"
  - "表示学习理论"
  - "contexture"
  - "奇异函数"
  - "期望算子"
  - "上下文变量"
  - "缩放定律"
---

# Contextures: Representations from Contexts

**会议**: ICML 2025

**arXiv**: [2505.01557](https://arxiv.org/abs/2505.01557)

**作者**: Runtian Zhai, Kai Yang, Che-Ping Tsai, Burak Varici, Zico Kolter, Pradeep Ravikumar (CMU)

**领域**: 自监督学习

**关键词**: 表示学习理论, contexture, 奇异函数, 期望算子, 上下文变量, 缩放定律

## 一句话总结

建立 contexture 理论，统一证明监督学习、自监督学习和流形学习等多种表示学习范式都可被理解为学习上下文变量诱导的期望算子的 top-$d$ 奇异函数，并揭示模型规模增大的边际递减效应以及提出上下文质量评估指标。

## 研究背景与动机

基础模型 (foundation models) 在实践中取得了巨大成功，但我们对这些模型所学习的**表示 (representations)** 缺乏系统性理解。核心问题包括：

**什么表示被学习了？** 不同训练范式（监督、自监督、流形学习）学到的表示有何共同本质？

**为什么有用？** 学到的表示为什么能迁移到各种下游任务？

**缩放的边际？** 模型规模增大是否总能提升性能？何时会出现边际递减？

现有理论通常只能分析特定范式（如对比学习的谱理论），缺乏跨范式的统一视角。早期工作（如 HaoChen et al., 2021; Zhai et al., 2024）将自监督学习与增广图的谱联系起来，但未扩展到监督学习和流形学习。

本文目标：建立一个足够通用的理论框架，统一解释各种表示学习方法的本质，并从中推导出实用洞察。

## 方法详解

### 核心概念：Contexture

**上下文变量 (Context Variable)**: 给定输入 $X$，定义一个上下文变量 $A$，表征与 $X$ 相关的信息：
- 监督学习中：$A = Y$（标签）
- 自监督学习中：$A = X'$（数据增强后的视图）
- 流形学习中：$A$ 为 $X$ 的 $K$ 近邻

**期望算子 (Expectation Operator)**: 由 $(X, A)$ 的联合分布 $P(X, A)$ 诱导的算子 $T_{P^+}$：

$$T_{P^+} f(a) = \mathbb{E}[f(X) | A = a]$$

**Contexture 定义**: 表示 $\Phi: \mathcal{X} \to \mathbb{R}^d$ 是如果它逼近期望算子 $T_{P^+}$ 的 top-$d$ 奇异函数 $\{\mu_1, \mu_2, \ldots, \mu_d\}$：

$$T_{P^+}^* T_{P^+} \mu_i = s_i^2 \mu_i$$

其中 $s_1 \geq s_2 \geq \cdots$ 为奇异值，捕获输入与上下文之间的关联强度。

### 统一性证明

论文证明以下方法都在学习 contexture：

| 学习范式 | 上下文变量 $A$ | 对应方法 |
|---------|-------------|---------|
| 监督学习 | 类别标签 $Y$ | 分类损失 → top 奇异函数 |
| 自监督（对比） | 增强视图 $X'$ | SimCLR, 谱对比损失 |
| 自监督（非对比） | 增强视图 $X'$ | VICReg, Barlow Twins |
| 自监督（掩码） | 可见 patch | MAE |
| 流形学习 | $K$ 近邻 | 拉普拉斯特征映射, PCA |

关键统一定理：这些方法的目标函数在最优解处等价于提取 $T_{P^+}$ 的 top-$d$ 奇异函数。

### 最优性定理

**定理（下游最优性）**: 如果下游任务 $(X, Y^*)$ 与上下文 $(X, A)$ 是**兼容的**（compatible），则学习了 contexture 的 $d$ 维表示 $\Phi$ 在该任务上是最优的。

兼容性条件：任务标签 $Y^*$ 可以由上下文 $A$ 的信息表达，即 $\mathbb{E}[Y^* | X]$ 落在 top-$d$ 奇异函数的张成空间中。

### 缩放定律的理论解释

**核心推论**: 一旦模型容量足以逼近 top-$d$ 奇异函数，进一步扩大模型规模带来的收益**边际递减**。

$$\text{模型增大} \to \Phi \text{ 逼近 top-}d\text{ 奇异函数} \to \text{性能饱和}$$

因此，**进一步提升需要更好的上下文**（如更好的数据增强策略），而非更大的模型。

### 上下文质量评估指标

提出**任务无关指标** $\tau_d$，仅基于期望算子的奇异值谱即可预测下游性能：

$$\tau_d = \sum_{i > d} s_i^2$$

直觉：$\tau_d$ 度量被 $d$ 维表示"遗漏"的信息量。$\tau_d$ 越小，context 越有利于 $d$ 维编码器的学习。

**谱估计方法**: 利用 Kernel PCA 或 NeuralEF/NeuralSVD 等方法高效估计奇异值谱，无需实际训练编码器。

## 实验关键数据

### Context-任务兼容性验证（OpenML 数据集）

| 数据集 | 样本数 | 特征 | KNN 上下文 $\tau$ 与真实误差相关性 |
|--------|-------|------|--------------------------------|
| cpu_act | 8,192 | 21 | 高相关 |
| pol | 15,000 | 26 | 高相关 |
| elevators | 16,599 | 16 | 高相关 |
| wine_quality | 6,497 | 11 | 高相关（失败案例）|
| yprop_4_1 | 8,885 | 42 | 中等相关 |

### $\tau_d$ 指标预测能力（Table 1，11 个数据集汇总）

对大部分数据集（11 个中的 9 个），$\tau_d$ 指标与编码器下游线性探测误差呈强正相关，表明该指标可有效预测上下文质量。

失败案例分析：
- **Case 1**: 上下文关联过强/过弱时，$\tau$ 可能误判
- **Case 2**: 指标不能跨上下文类型比较（如增强 vs. 近邻）

### 大规模数据验证（Rebuttal 补充）

| 数据集 | 编码器 | 上下文 | 结果 |
|--------|-------|--------|------|
| MNIST | LeNet | 随机裁剪（不同比例） | $\tau_d$ 与误差强相关 |
| CIFAR-10 | ResNet-18 | SimCLR 增强 | $\tau_d$ 精确追踪实际误差 |

### 缩放实验（不同宽度/深度模型）

使用 $d = 128$ 维表示，CCA 对齐度一般 $> 0.8$、最高达 $\approx 0.9$，验证了较大模型确实更接近 top 奇异函数。但模型过大时，优化变难反而降低对齐度。

## 亮点与洞察

1. **优雅的统一性**: 用一个数学对象（期望算子的奇异函数）统一了监督、自监督和流形学习，理论上非常漂亮
2. **Scaling 的理论界限**: 给出了"模型够大就够了"的理论依据，暗示未来应投资于更好的上下文设计（数据增强、标注策略）而非盲目扩大模型
3. **任务无关指标**: $\tau_d$ 仅依赖谱信息，无需下游标签即可评估上下文质量，对超参数选择（如裁剪比例、掩码比例）有实际指导意义
4. **解释 Platonic Representation Hypothesis**: 可以解释为什么 SSL 和 CLIP 学到高度对齐的表示——因为它们的上下文本质上共享 top 奇异函数

## 局限性

1. **实验规模偏小**: 主实验仅在 OpenML 小数据集（最大 ~28K 样本）上验证，缺乏 ImageNet 等标准大规模验证
2. **仅限线性下游**: 理论分析假设下游任务使用线性探测，非线性 fine-tuning 的情况未覆盖
3. **依赖优化假设**: 理论分析聚焦目标函数的极小值点，但实际训练中模型在"稳定性边缘"振荡，理论与实践的差距未解决
4. **未分析架构归纳偏置**: 如 CNN 的平移不变性对上下文的影响未纳入理论框架
5. **分布漂移**: 假设上游和下游数据同分布，实际中的分布偏移效应未考虑

## 相关工作与启发

- **HaoChen et al. (2021)**: 将对比学习与增广图谱联系，本文将此推广到所有 SSL 方法和更广泛的学习范式
- **Zhai et al. (2024)**: 将谱理论从对比学习扩展到所有 SSL，本文进一步扩展到所有表示学习
- **Balestriero & LeCun (2022)**: 证明对比和非对比 SSL 恢复谱嵌入方法的 top 特征函数
- **Huh et al. (2024) Platonic Representation Hypothesis**: 发现不同模型学到高度对齐的表示，contexture 理论提供了一种解释
- 本框架可能启发：(a) 基于谱分析的增强策略自动搜索；(b) 训练前预测模型最优大小；(c) 不同预训练方法的统一评估标准

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 5 | 统一多范式的理论框架，概念优雅 |
| 理论深度 | 5 | 严格的数学推导，覆盖广泛 |
| 实验充分性 | 2 | 数据集过小，缺乏标准 benchmark 验证 |
| 写作质量 | 4 | 理论表述清晰，但内容覆盖面过广 |
| 实用价值 | 3 | $\tau_d$ 指标有潜力但实用性待大规模验证 |
| **综合** | **3.8** | 理论优美但实验支撑不足的理论论文 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Contrastive Representations for Temporal Reasoning](../../NeurIPS2025/self_supervised/contrastive_representations_for_temporal_reasoning.md)
- [\[ICML 2026\] Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](../../ICML2026/self_supervised/provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)
- [\[CVPR 2026\] Suppressing Non-Semantic Noise in Masked Image Modeling Representations](../../CVPR2026/self_supervised/suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)
- [\[ICML 2025\] Test-Time Training Provably Improves Transformers as In-Context Learners](test-time_training_provably_improves_transformers_as_in-context_learners.md)
- [\[ICML 2025\] ReSA: Clustering Properties of Self-Supervised Learning](clustering_properties_of_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
