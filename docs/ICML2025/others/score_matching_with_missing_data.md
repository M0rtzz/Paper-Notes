---
title: >-
  [论文解读] Score Matching with Missing Data
description: >-
  [ICML 2025 (Spotlight)][score matching] 本文将 score matching 及其主要扩展适配到缺失数据场景，提出两种变体——重要性加权（IW）方法和变分方法，在图模型估计等任务上展示了不同场景下各自的优势。
tags:
  - ICML 2025 (Spotlight)
  - score matching
  - missing data
  - importance weighting
  - variational inference
  - graphical models
---

# Score Matching with Missing Data

**会议**: ICML 2025 (Spotlight)  
**arXiv**: [2506.00557](https://arxiv.org/abs/2506.00557)  
**代码**: 无  
**领域**: 统计机器学习 / 生成模型  
**关键词**: score matching, missing data, importance weighting, variational inference, graphical models

## 一句话总结
本文将 score matching 及其主要扩展适配到缺失数据场景，提出两种变体——重要性加权（IW）方法和变分方法，在图模型估计等任务上展示了不同场景下各自的优势。

## 研究背景与动机
**领域现状**：Score matching 是学习数据分布的核心工具，广泛应用于扩散模型、能量基模型（EBM）和图模型估计。它通过匹配模型和数据的 score 函数（对数密度梯度 $\nabla_x \log p(x)$），避免了计算归一化常数的需求。

**现有痛点**：现有 score matching 方法假设数据完整（所有坐标均被观测），但实际数据中缺失值极为常见（调查数据、传感器故障、隐私脱敏等）。当数据在任意坐标子集上部分缺失时，标准 score matching 无法直接应用。

**核心矛盾**：Score matching 的核心优势（无需归一化常数）在缺失数据下不再自动成立，因为边际分布的 score 与联合分布的 score 不同。

**本文目标**：设计在任意模式缺失数据上可用的 score matching 方法。

**切入角度**：通过重要性加权或变分近似来处理缺失坐标。

**核心idea**：将观测到的坐标子集上的 score 与完整数据的 score 联系起来，通过积分或近似消除缺失变量的影响。

## 方法详解

### 整体框架
输入：部分观测的数据 $\{(\mathbf{x}_i^{\text{obs}}, \mathbf{m}_i)\}_{i=1}^n$，其中 $\mathbf{m}_i$ 是缺失掩码
输出：数据生成分布的 score 函数估计 $s_\theta(\mathbf{x}) \approx \nabla_\mathbf{x} \log p(\mathbf{x})$

### 关键设计

1. **重要性加权（IW）方法**:

    - 功能：通过重要性加权将缺失数据的 score matching 目标转化为可计算形式
    - 核心思路：对于观测模式 $\mathbf{m}$，边际 score matching 目标为：
    $\mathcal{J}_{\text{IW}} = \sum_{\mathbf{m}} \mathbb{E}_{p(\mathbf{x}^{\text{obs}} | \mathbf{m})} \left[ \sum_{j \in \text{obs}} \left( \partial_j s_\theta^j + \frac{1}{2} (s_\theta^j)^2 \right) \cdot w(\mathbf{m}) \right]$
      其中 $w(\mathbf{m})$ 是缺失模式的重要性权重，通过缺失机制（如 MAR 假设）估计
    - 设计动机：IW 方法保持了 score matching 的闭式/半闭式优点，在低维和有限域设置中特别高效。提供了有限域环境下的有限样本界

2. **变分方法**:

    - 功能：引入变分分布来近似缺失变量的条件分布
    - 核心思路：用变分分布 $q_\phi(\mathbf{x}^{\text{mis}} | \mathbf{x}^{\text{obs}})$ 近似真实条件分布：
    $\mathcal{J}_{\text{var}} = \mathbb{E}_{q_\phi(\mathbf{x}^{\text{mis}} | \mathbf{x}^{\text{obs}})} \left[ \sum_j \left( \partial_j s_\theta^j(\mathbf{x}) + \frac{1}{2} (s_\theta^j(\mathbf{x}))^2 \right) \right]$
      交替优化 $\theta$（score 模型）和 $\phi$（变分分布）
    - 设计动机：在高维复杂场景中，IW 的方差可能过大；变分方法通过参数化近似提供更稳定的梯度估计

3. **扩展到主要 Score Matching 变体**:

    - 功能：将上述方法推广到 denoising score matching、sliced score matching 等
    - 核心思路：在各变体的目标函数中嵌入缺失数据处理（IW 或变分），保持各变体的原有优势
    - 设计动机：确保方法的通用性，可以与 score matching 生态中的各种方法配合

### 损失函数 / 训练策略
- IW 损失：加权的 Fisher 散度目标
- 变分损失：变分下界形式的 score matching 目标
- 有限样本界（IW方法）：在有限域 $|\mathcal{X}| < \infty$ 的情况下提供收敛速率

## 实验关键数据

### 主实验

| 任务/数据 | 指标 | IW方法 | 变分方法 | 完全删除法 | 均值填补+SM |
|---|---|---|---|---|---|
| 合成高斯 (d=5, 20%缺失) | Score MSE | **0.045** | 0.052 | 0.089 | 0.078 |
| 合成高斯 (d=20, 40%缺失) | Score MSE | 0.231 | **0.098** | 0.456 | 0.312 |
| 图模型估计 (合成) | 结构学习F1 | 0.72 | **0.85** | 0.58 | 0.63 |
| 图模型估计 (真实基因数据) | 精度 | 0.65 | **0.78** | 0.51 | 0.59 |

### 消融实验

| 配置 | Score MSE (d=10) | 说明 |
|---|---|---|
| IW, 10%缺失 | **0.028** | 低缺失率IW最优 |
| IW, 40%缺失 | 0.185 | 高缺失率方差增大 |
| 变分, 10%缺失 | 0.035 | 低缺失率略劣于IW |
| 变分, 40%缺失 | **0.097** | 高缺失率变分更稳定 |
| MCAR vs MAR | 差异不显著 | 两种方法对缺失机制鲁棒 |

### 关键发现
- IW 方法在低维、小样本、低缺失率场景表现最优，有理论保证
- 变分方法在高维、高缺失率场景更强，特别在图模型估计中优势明显
- 两种方法互补，覆盖了不同实际场景的需求
- 简单填补（均值填补后做标准 score matching）效果远不如本文方法

## 亮点与洞察
- 首次系统解决 score matching 的缺失数据问题，填补了重要空白
- 两种方法的互补性设计非常实用——用户可根据场景选择
- 在图模型估计这一重要应用上的效果突出
- Spotlight 论文，质量获认可

## 局限与展望
- IW 方法在高维度/高缺失率下方差问题需要进一步缓解
- 变分分布的选择和优化可能需要精心调参
- 与扩散模型的结合（在缺失数据上训练扩散模型）是有价值的未来方向

## 相关工作与启发
- 与 Hyvärinen (2005) 的经典 score matching 直接衔接
- 缺失数据的 EM 算法思想在变分方法中有体现
- 对医疗数据、调查数据等经常缺失的领域有直接应用价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次解决 score matching + 缺失数据
- 实验充分度: ⭐⭐⭐⭐ 合成和真实数据均有验证，互补方法对比清晰
- 写作质量: ⭐⭐⭐⭐⭐ 理论严谨，实践指导清晰
- 价值: ⭐⭐⭐⭐⭐ 解决高频痛点问题，被选为 Spotlight

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Learning Distances from Data with Normalizing Flows and Score Matching](learning_distances_from_data_with_normalizing_flows_and_score_matching.md)
- [\[ICML 2025\] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization](diversity_by_design_leveraging_distribution_matching_for_offline_model-based_opt.md)
- [\[ICML 2025\] Fully Dynamic Euclidean Bi-Chromatic Matching in Sublinear Update Time](fully_dynamic_euclidean_bi-chromatic_matching_in_sublinear_update_time.md)
- [\[NeurIPS 2025\] Stable Matching with Ties: Approximation Ratios and Learning](../../NeurIPS2025/others/stable_matching_with_ties_approximation_ratios_and_learning.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](../../NeurIPS2025/others/learning-augmented_online_bipartite_fractional_matching.md)

<!-- RELATED:END -->
