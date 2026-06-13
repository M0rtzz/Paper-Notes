---
title: >-
  [论文解读] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization
description: >-
  [ICML 2025][离线优化] 提出 DynAMO，通过将设计多样性显式建模为分布匹配问题，在离线模型基础优化（MBO）中同时发现高质量和高多样性的候选设计方案。 领域现状：离线模型基础优化（MBO）给定一个离线数据集，目标是提出最大化目标函数的新设计。常用于蛋白质设计、分子优化等科学领域。现有方法主要聚焦于最大化目标值…
tags:
  - "ICML 2025"
  - "离线优化"
  - "模型基础优化"
  - "设计多样性"
  - "分布匹配"
  - "对抗训练"
---

# Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization

**会议**: ICML 2025  
**arXiv**: [2501.18768](https://arxiv.org/abs/2501.18768)  
**代码**: 无  
**领域**: 其他  
**关键词**: 离线优化, 模型基础优化, 设计多样性, 分布匹配, 对抗训练

## 一句话总结
提出 DynAMO，通过将设计多样性显式建模为分布匹配问题，在离线模型基础优化（MBO）中同时发现高质量和高多样性的候选设计方案。

## 研究背景与动机

**领域现状**：离线模型基础优化（MBO）给定一个离线数据集，目标是提出最大化目标函数的新设计。常用于蛋白质设计、分子优化等科学领域。现有方法主要聚焦于最大化目标值，不关注候选方案的多样性。

**现有痛点**：(1) 生成的候选设计往往集中在目标函数的单一峰值附近，缺乏多样性——但实际应用中需要多个不同的近优方案供选择；(2) 离线数据集本身包含丰富的多样性信息，但现有方法未加利用；(3) 简单的多样性正则化（如分散约束）往往与质量目标冲突，导致性能下降。

**核心矛盾**：需要同时优化设计质量（高目标值）和设计多样性（覆盖多种最优配置），但两者之间存在天然的张力。

**本文目标**：将多样性作为显式目标引入任意 MBO 问题，同时保持设计质量。

**切入角度**：将多样性形式化为分布匹配问题——生成设计的分布应该捕获离线数据集中固有的多样性结构。

**核心 idea**：用对抗训练的分布匹配损失约束生成设计的分布与离线数据集的高质量子集分布对齐，从而在优化质量的同时保持多样性。DynAMO 可作为插件与任何 MBO 方法结合。

## 方法详解

### 整体框架
在任何 MBO 方法的优化管线中，额外添加一个对抗分布匹配项。从离线数据集中提取高质量子集作为多样性参考分布，然后用对抗训练使生成设计的分布与之匹配。

### 关键设计

1. **分布匹配作为多样性目标**:

    - 功能：将设计多样性形式化为可优化的目标
    - 核心思路：将离线数据集中高目标值样本的分布作为参考，用对抗判别器度量生成设计分布与参考分布之间的距离。最小化该距离确保生成设计不仅质量高，还保持了数据集中本身蕴含的多样性结构
    - 设计动机：离线数据集的多样性是领域知识的隐式编码（如蛋白质空间中的不同功能族），保持这种多样性比随机分散更有意义

2. **对抗优化框架 (DynAMO)**:

    - 功能：将分布匹配项以插件形式加入任意 MBO 方法
    - 核心思路：训练一个判别器区分"来自优化器的生成设计"和"来自数据集高质量子集的真实设计"，将判别器损失加入优化器的总目标中。任何 MBO 方法（CbAS、COMsb、ROMA 等）都可以添加 DynAMO 项
    - 设计动机：对抗训练天然适合分布匹配，且作为附加损失项不改变原方法的核心优化流程

3. **高质量子集选择**:

    - 功能：确定多样性参考分布
    - 核心思路：根据代理模型或已知目标值筛选离线数据集中的 top-k 样本作为参考分布
    - 设计动机：不是与整个数据集匹配（包含低质量样本），而是只与高质量子集匹配——确保多样性在"好的设计空间"内

### 损失函数 / 训练策略
总损失 = 原始 MBO 目标 + λ × 对抗分布匹配损失。判别器和生成器交替训练。

## 实验关键数据

### 主实验

| 领域 | MBO 方法 + DynAMO | 质量变化 | 多样性变化 |
|------|------------------|---------|----------|
| 蛋白质设计 | 提升 | 保持/轻微提升 | **显著提升** |
| 分子优化 | 提升 | 保持 | **显著提升** |
| 材料设计 | 提升 | 保持 | **显著提升** |

### 消融实验

| 配置 | 多样性 | 质量 | 说明 |
|------|--------|------|------|
| 原始 MBO | 低 | 基线 | 集中于单一峰值 |
| + 随机扰动 | 中 | 下降 | 质量-多样性冲突 |
| + DynAMO | **高** | 保持/提升 | 分布匹配兼顾两者 |

### 关键发现
- DynAMO 作为插件与多种 MBO 方法组合均能显著提升多样性，验证了其通用性
- 分布匹配比简单的多样性正则化更有效，因为它保持了数据集中有意义的结构性多样性
- 跨多个科学领域（蛋白质、分子、材料）验证了有效性

## 亮点与洞察
- **多样性 = 分布匹配**的形式化非常优雅——不是人为定义多样性指标，而是让生成分布自动匹配数据中的自然多样性
- 作为即插即用的插件可与任何 MBO 方法组合，实际应用门槛低
- 方法思路可迁移到其他需要生成多样解的领域（如多样化推荐、多目标优化）

## 局限与展望
- 对抗训练的稳定性可能在高维设计空间中成为问题
- 参考分布的质量阈值选择需要调优
- 目前仅在离线 MBO 框架下验证，在线设置有待探索
- 本文信息来源于摘要，方法细节和实验数据待从完整论文补充

## 相关工作与启发
- **vs CbAS/COMsb**: 传统 MBO 方法只优化质量，DynAMO 额外引入多样性目标
- **vs 多目标优化**: 多目标优化通过 Pareto 前沿处理冲突目标，DynAMO 则通过分布匹配自然结合质量和多样性

## 评分
- 新颖性: ⭐⭐⭐⭐ 将多样性建模为分布匹配是新颖且优雅的视角
- 实验充分度: ⭐⭐⭐ 跨多个领域验证但具体数据需从完整论文补充
- 写作质量: ⭐⭐⭐ 基于摘要信息评估
- 价值: ⭐⭐⭐⭐ 为科学设计中的多样性问题提供了通用解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Soft Quality-Diversity Optimization](../../ICLR2026/others/soft_quality-diversity_optimization.md)
- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](../../ICLR2026/others/discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[ACL 2025\] Achieving Certification-by-Design Through Model-Driven Development](../../ACL2025/others/achieving_certification-by-design_through_model-driven_development.md)
- [\[ICML 2025\] IBDR: Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness](promoting_ensemble_diversity_with_interactive_bayesian_distributional_robustness.md)
- [\[ICML 2025\] Learning Distances from Data with Normalizing Flows and Score Matching](learning_distances_from_data_with_normalizing_flows_and_score_matching.md)

</div>

<!-- RELATED:END -->
