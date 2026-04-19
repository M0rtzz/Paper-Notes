---
title: >-
  [论文解读] Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator
description: >-
  [ICML 2025][Fisher Information Matrix] 本文系统分析了 Adam 优化器的平方梯度累积器（Squisher）与 Fisher 信息矩阵对角线之间的理论联系，证明 Squisher 可以作为 Fisher 对角线的免费近似，在模型合并、持续学习、稀疏化等五大应用中表现与 Fisher 相当。
tags:
  - ICML 2025
  - Fisher Information Matrix
  - Adam Optimizer
  - Squared Gradient Accumulator
  - Model Merging
  - Parameter Sensitivity
---

# Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator

**会议**: ICML 2025  
**arXiv**: [2507.18807](https://arxiv.org/abs/2507.18807)  
**代码**: 有  
**领域**: 优化 / 模型合并  
**关键词**: Fisher Information Matrix, Adam Optimizer, Squared Gradient Accumulator, Model Merging, Parameter Sensitivity

## 一句话总结
本文系统分析了 Adam 优化器的平方梯度累积器（Squisher）与 Fisher 信息矩阵对角线之间的理论联系，证明 Squisher 可以作为 Fisher 对角线的免费近似，在模型合并、持续学习、稀疏化等五大应用中表现与 Fisher 相当。

## 研究背景与动机
**领域现状**：Fisher 信息矩阵对角线广泛用于衡量参数敏感性，但计算需要逐样本梯度平方再求和，代价不低。

**现有痛点**：Fisher 对角线需要额外计算，需要访问训练数据，且高效实现非平凡，阻碍了 Fisher 方法（如 Fisher Merging）的广泛采用。

**核心矛盾**：Adam 优化器已经在训练过程中维护了一个平方梯度的指数移动平均，这与 Fisher 对角线在表面上相似但细节不同。

**切入角度**：严格分析两者的关系，明确差异并量化影响。

**核心idea**：Adam 的 $v^{(t)}$ 可直接"回收"作为 Fisher 对角线的近似（命名为 Squisher），无需额外计算。

## 方法详解

### 整体框架
理论分析 Fisher 对角线与平方梯度累积器的精确关系 → 在六个应用场景中实验验证 Squisher 是否可替代 Fisher。

### 关键设计

1. **Standard vs Joint Fisher**:

    - 标准 Fisher: $\text{diag}(F_{\text{std}}^{\text{emp}}) = \sum_n g_n^2$（先平方再求和）
    - Joint Fisher: $\text{diag}(F_{\text{joint}}^{\text{emp}}) = (\sum_n g_n)^2$（先求和再平方）
    - Squisher 更接近 Joint Fisher 而非 Standard Fisher

2. **Squisher 的三个近似源**:

    - 先聚合后平方 vs 先平方后聚合（$\sum g_n^2$ vs $(\sum g_n)^2$），引入 batch 级别的差异
    - EMA 带来的时间加权，非均匀加权历史梯度
    - 使用 mean reduction 损失时的 $1/N$ 缩放因子
    - 关键观察：许多应用对 Fisher 的缩放不变（如 Fisher Merging），因此缩放差异无影响

3. **实验验证**:

    - 覆盖五大应用：模型合并、持续学习、稀疏训练、任务相似度、模型稀疏化、稀疏微调
    - 每个应用都与使用真实 Fisher 和无 Fisher 基线对比

## 实验关键数据

### 主实验
| 应用 | Fisher | Squisher | 无Fisher基线 |
|------|--------|----------|-------------|
| Fisher Merging (8任务) | 73.2% | 73.2% | 70.8% (均值合并) |
| 持续学习 (EWC) | 82.1% | 81.8% | 79.5% (无正则) |
| 稀疏训练 | 88.3% | 88.1% | 86.7% (随机) |

### 消融实验
| 配置 | 模型合并性能 | 说明 |
|------|-------------|------|
| Standard Fisher | 73.2% | 标准计算 |
| Squisher (直接回收) | 73.2% | 零额外成本 |
| 均匀权重合并 | 70.8% | 完全忽略参数重要性 |

### 关键发现
- Squisher 在所有六个设置中均与 Fisher 表现相当，且始终优于无 Fisher 基线
- Fisher Merging 的公式对 Fisher 缩放不变，因此 Squisher 的缩放差异完全无影响
- EMA 系数 $\beta_2=0.999$ 使 Squisher 包含约 10000 步的梯度历史

## 亮点与洞察
- 实用价值极高：将 Fisher 方法的计算成本降为零（直接从 Adam 状态中读取）
- 理论分析严谨：通过 Joint Fisher 建立了 Squisher 与 Fisher 的精确数学联系
- Lin et al. (2024) 证明 Joint Fisher = Standard Fisher，为本文提供理论基础
- 可能促进 Fisher Merging 等方法被更广泛采用（如 mergekit 可直接集成）

## 局限与展望
- Squisher 是 empirical Fisher 的近似，而非 standard Fisher
- EMA 引入时间偏差，对训练过程中参数变化大的场景可能有影响
- 仅适用于 Adam 系列优化器

## 评分
- 新颖性: ⭐⭐⭐⭐ 联系虽直觉上自然但严格分析有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 六大应用全面验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论清晰，图表精美
- 价值: ⭐⭐⭐⭐⭐ 极强的实用价值

<!-- RELATED:START -->

## 相关论文

- [FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information](../../ECCV2024/others/fisherrf_active_view_selection_and_mapping_with_radiance_fields_using_fisher_inf.md)
- [Residual Matrix Transformers: Scaling the Size of the Residual Stream](residual_matrix_transformers_scaling_the_size_of_the_residual_stream.md)
- [WGFormer: An SE(3)-Transformer Driven by Wasserstein Gradient Flows for Molecular Generation](wgformer_an_se3-transformer_driven_by_wasserstein_gradient_flows_for_molecular_g.md)
- [The Structural Complexity of Matrix-Vector Multiplication](../../NeurIPS2025/others/the_structural_complexity_of_matrix-vector_multiplication.md)
- [Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition](provably_efficient_algorithm_for_best_scoring_rule_identification_in_online_prin.md)

<!-- RELATED:END -->
