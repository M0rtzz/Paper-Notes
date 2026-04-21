---
title: >-
  [论文解读] Gene Incremental Learning for Single-Cell Transcriptomics
description: >-
  [AAAI 2026][基因增量学习] 本文提出了基因增量学习（GIL）框架，利用单细胞转录组学数据的无序性特点，将类增量学习（CIL）的范式扩展到 token（基因）维度，设计了基因回放和基因蒸馏两种基线方法，并建立了包含基因级回归和基因级分类两种评估方式的完整基准。
tags:
  - AAAI 2026
  - 基因增量学习
  - 单细胞转录组学
  - 灾难性遗忘
  - Token增量学习
  - benchmark
---

# Gene Incremental Learning for Single-Cell Transcriptomics

**会议**: AAAI 2026  
**arXiv**: [2511.13762](https://arxiv.org/abs/2511.13762)  
**代码**: [GitHub](https://github.com/simpleshinobu/scbenchmark)  
**领域**: 生物信息学 / 增量学习  
**关键词**: 基因增量学习, 单细胞转录组学, 灾难性遗忘, Token增量学习, benchmark

## 一句话总结

本文提出了基因增量学习（GIL）框架，利用单细胞转录组学数据的无序性特点，将类增量学习（CIL）的范式扩展到 token（基因）维度，设计了基因回放和基因蒸馏两种基线方法，并建立了包含基因级回归和基因级分类两种评估方式的完整基准。

## 研究背景与动机

类增量学习（CIL）在计算机视觉领域已被广泛研究，其核心问题是模型在持续学习新类别时如何避免灾难性遗忘。然而，token 作为很多领域（NLP、生物信息学）中的基本元素，同样具有持续增长的特性（如新词不断被发明、新基因不断被发现），但 token 维度的增量学习一直被忽视。

这一研究空白主要源于语言数据的整体性（holistic nature）——在自然语言中，如果将不同词划分到不同阶段（如某阶段不出现「learning」这个词），要么需要排除含有该词的所有文本（大幅减少数据量），要么需要从文本中删除该词（改变原始语义），两种方案都不可行。

幸运的是，单细胞转录组学数据没有这个问题。在转录组学中，基因就是 token，每个样本由一组基因表达值组成（类似句子），且基因之间没有相对顺序，可以自由划分和重排。这使得设计基因增量学习框架成为可能。

核心 idea：利用转录组学数据的无序性来绕过语言数据的整体性障碍，首次建立 token 增量学习框架，并通过基因回放和基因蒸馏验证该框架中同样存在遗忘问题。

## 方法详解

### 整体框架

GIL 的 pipeline 如下：将全部基因分为基础基因集 $B$（每阶段都出现）和各阶段特有基因集 $T^{s_1}, T^{s_2}, \ldots, T^{s_n}$（不同阶段互不相交）。数据集同样独立划分为 $\mathcal{D}^{s_1}, \ldots, \mathcal{D}^{s_n}$。在阶段 $k$，模型只能看到包含基础基因和 $T^{s_k}$ 的样本，然后在所有已见基因上进行评估。

基因学习通过掩码值预测（masked value prediction）实现：随机遮掩部分基因的表达值，让模型预测被遮掩的值，训练损失为 $\mathcal{L}_{\text{tran}}(\mathcal{D}, \phi) = \frac{1}{N}\sum_{i=1}^{N}\sum_j \|v_{ij} - \hat{v}_{ij}\|^2$。

### 关键设计

1. **基础基因机制（Base Gene Mechanism）**:

    - 功能：保留一部分基因作为基础基因 $B$，在每个增量阶段都出现
    - 核心思路：基因不像类别那样能单独表达样本含义，单个基因只是样本的一个组成部分。基础基因确保每个阶段的样本在转录组学语境下仍然有意义
    - 设计动机：与 CIL 中类和样本直接对应不同，GIL 中基因和样本不对齐——每个样本包含所有基因。基础基因解决了"基因不足导致无意义样本"的风险

2. **基因回放（Gene Replay）**:

    - 功能：在当前阶段训练时保留前序阶段的部分样本一起训练
    - 核心思路：$\mathcal{L}_{\text{dr},s_k} = \mathcal{L}_{\text{tran}}(\mathcal{D}^{s_k}, \phi) + \sum_{i=1}^{k-1}\mathcal{L}_{\text{tran}}(\mathcal{D}_{\text{dr}}^{s_i}, \phi)$，其中 $\mathcal{D}_{\text{dr}}^{s_i} \subset \mathcal{D}^{s_i}$ 是前序阶段的数据子集
    - 设计动机：直接借鉴 CIL 中数据回放策略，通过保留旧样本来防止遗忘。回放样本越多，性能越接近 Oracle

3. **基因蒸馏（Gene Distillation）**:

    - 功能：利用上一阶段的最优模型 $\phi_{s_{k-1}}^*$ 对当前模型进行知识蒸馏
    - 核心思路：$\mathcal{L}_{\text{fd},s_k} = \frac{1}{N_k}\sum_{i=1}^{N_k}(\sum_j \|v_{ij} - \hat{v}_{ij}\|^2 + \lambda\|\hat{\bm{v}}_i - \hat{\bm{v}}_{i,s_{k-1}}^*\|^2)$，即在掩码预测损失基础上加入对旧模型输出的模仿损失
    - 设计动机：从 CIL 的知识蒸馏方法适配而来，假设旧模型能用基础基因来表征当前样本。注意当前阶段的新基因从蒸馏项中移除，因为旧模型不具备预测未见基因的能力

4. **特征提取过程**:

    - 基因嵌入：$\bm{e} = \mathbf{E}_\phi(\bm{x}) + \tilde{\bm{v}}\mathbf{L}_{1,\phi}$
    - Transformer 编码：$\bm{e}' = \mathbf{M}_\phi(\bm{e})$
    - 值预测：$\hat{\bm{v}} = \bm{e}'\mathbf{L}_{2,\phi}$
    - 基因嵌入层将 token ID映射到向量，同时线性层将表达值编码到嵌入空间，两者相加后送入 Transformer

### 评估方法

1. **基因级回归评估**：直接使用掩码预测损失 $\mathcal{L}_{\text{regress},s_k} = \mathbb{E}[\sum_k \|v_{ik} - \hat{v}_{ik}^*\|^2]$，评估特定阶段基因的预测准确率
2. **基因级分类评估**：为每阶段选择对特定下游分类任务至关重要的基因，通过下游分类准确率间接评估基因的学习和遗忘

## 实验关键数据

### 主实验

使用 scGPT 风格的 Transformer（6层,8头,隐藏维度256），在 906,890 训练样本上训练。

**2阶段基因级回归（Norman-Lupus设定）**：

| 方法 | 阶段 | Norman | Lupus | Δ（遗忘） |
|------|------|--------|-------|-----------|
| Baseline | 1 | 0.172 | - | - |
| Baseline | 2 | 0.424 | 0.134 | 0.253 |
| Replay (1000) | 2 | 0.215 | 0.134 | 0.043 |
| Distill (λ=5) | 2 | 0.365 | 0.139 | 0.193 |
| Oracle | - | 0.173 | 0.136 | - |

**2阶段基因级分类（下游准确率%）**：

| 方法 | 阶段 | Norman | Lupus | Δ（遗忘） |
|------|------|--------|-------|-----------|
| Baseline | 1 | 37.73 | 67.31 | - |
| Baseline | 2 | 35.59 | 75.39 | -2.14 |
| Replay | 2 | 36.45 | 75.00 | -1.29 |
| Distill | 2 | 34.16 | 72.94 | -3.74 |
| Oracle | - | 38.11 | 75.42 | - |

### 消融实验

**基因回放和基因蒸馏超参消融（Norman-Lupus 回归）**：

| 方法 | 超参 | Norman | Lupus | Δ |
|------|------|--------|-------|---|
| Replay | 50 样本 | 0.293 | 0.136 | 0.121 |
| Replay | 100 样本 | 0.263 | 0.138 | 0.091 |
| Replay | 1000 样本 | 0.215 | 0.134 | 0.043 |
| Replay | 10000 样本 | 0.190 | 0.133 | 0.018 |
| Distill | λ=0.5 | 0.420 | 0.134 | 0.248 |
| Distill | λ=5.0 | 0.365 | 0.139 | 0.193 |
| Distill | λ=10.0 | 0.326 | 0.143 | 0.154 |

### 关键发现
- Baseline 在 2 阶段设定中回归 Δ 平均为 0.279、分类 Δ 平均为 -1.816%，验证了基因遗忘确实存在
- 基因回放随回放样本增加效果持续提升，最好 Δ 为 0.018（10000 样本，接近 Oracle）
- 基因蒸馏在回归评估上有效（Δ 从 0.253 降到 0.193），但在分类评估上反而更差（Δ 从 -1.816% 恶化到 -2.473%），说明蒸馏虽然平均防止了遗忘，但可能降级了基因特征的质量
- 3 阶段设定中遗忘现象持续存在且可累积
- 两种评估方式的一致性：回归和分类评估都随阶段增加而性能下降，但分类评估的下降不如 CIL 中那么显著

## 亮点与洞察
- 首次提出 token 维度增量学习的概念，巧妙利用转录组学数据无序性来规避语言数据的不可分性
- 基础基因机制是一个简洁但关键的设计，保证了每阶段样本的语义完整性
- 两种评估方式互补：回归直接测量遗忘程度，分类从下游任务角度验证基因记忆质量
- 基因蒸馏在回归好但分类差的发现很有趣，揭示了 token 遗忘的特殊性

## 局限与展望
- 发现的遗忘现象相比 CIL 较弱（分类下降仅 1-3%），这是因为单个基因对样本整体的影响有限
- 没有设计 GIL 专用的新方法，只是适配已有的 CIL 方法，缺乏创新性
- 可创建的设定数量有限——不同下游数据集的关键基因有大量重叠，难以构造大量不冲突的设定
- 阶段数量也受限（仅测试了 2-3 阶段），未验证更长序列的增量场景
- 基础基因的选择策略对框架影响可能很大，但未做详细分析
- 输入长度限制为 512 基因，基因是随机选择的，引入了评估的随机性

## 相关工作与启发
- CIL 中数据回放和知识蒸馏的经典框架可自然适配到 token 维度，为更广阔的增量学习场景提供参考
- scGPT、scBERT 等预训练模型的掩码预测框架为基因学习提供了成熟的训练范式
- 本文的贡献更在于提出问题和建立基准，而非解决问题——这为后续研究开辟了新方向
- 启发：语言数据虽然不适合 token 增量学习，但代码（token 有独立语义）等领域可能同样适用

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning](../../ECCV2024/llm_evaluation/versatile_incremental_learning_towards_class_and_domain-agnostic_incremental_lea.md)
- [scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration](../../NeurIPS2025/llm_evaluation/scmrdr_a_scalable_and_flexible_framework_for_unpaired_single-cell_multi-omics_da.md)
- [Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](../../CVPR2026/llm_evaluation/temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)
- [Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning](../../CVPR2025/llm_evaluation/dual_consolidation_for_pre-trained_model-based_domain-incremental_learning.md)
- [Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)

<!-- RELATED:END -->
