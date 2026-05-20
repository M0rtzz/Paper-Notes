---
title: >-
  [论文解读] Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation
description: >-
  [CVPR 2025][语义分割][跨域小样本分割] 提出 Dual-Agent Optimization (DATO) 框架，包含一致性互聚合（CMA）模块学习跨域不变特征以增强表示，以及相关性修正策略（CRS）将 support-query 匹配转移到域不敏感的特征空间，有效提升跨域小样本分割的泛化能力。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "跨域小样本分割"
  - "域不变特征"
  - "一致性互聚合"
  - "相关性修正"
  - "特征适配"
---

# Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation

**会议**: CVPR 2025  
**arXiv**: 待确认  
**代码**: 待确认  
**领域**: 图像分割  
**关键词**: 跨域小样本分割, 域不变特征, 一致性互聚合, 相关性修正, 特征适配

## 一句话总结
提出 Dual-Agent Optimization (DATO) 框架，包含一致性互聚合（CMA）模块学习跨域不变特征以增强表示，以及相关性修正策略（CRS）将 support-query 匹配转移到域不敏感的特征空间，有效提升跨域小样本分割的泛化能力。

## 研究背景与动机

**领域现状**：小样本分割（FSS）通过少量标注样本实现新类别分割，已在同域场景下取得良好效果。然而实际应用中训练集和测试集往往来自不同域（如自然图像 → 医学图像、遥感 → 工业检测等），催生了跨域小样本分割（CD-FSS）这一更具挑战性的任务。

**现有痛点**：(1) **域差异导致特征失配**：在源域上学到的特征表示在目标域上可能完全失效，因为不同域的纹理、颜色、结构分布差异巨大；(2) **support-query 匹配退化**：标准 FSS 中 support 和 query 来自同一域，相关性匹配比较可靠。跨域场景下，来自不同域的 support 和 query 特征的匹配变得极不可靠；(3) **域适应的标注稀缺**：few-shot 场景下只有极少标注，传统域适应方法需要大量目标域数据，在此场景不适用。

**核心矛盾**：跨域 FSS 需要同时解决两个问题——特征的域不变性（看见什么都能提好特征）和匹配的域鲁棒性（跨域匹配也能准确）。而这两个目标在少样本条件下很难同时满足。

**本文目标** 如何在极少标注条件下同时提升特征表示的跨域不变性和 support-query 匹配的跨域鲁棒性。

**切入角度**：引入一组可学习的"代理"（agents）作为跨域桥梁。这些代理通过与多域特征交互学习域不变表示，然后用域不变特征作为中间媒介来修正跨域匹配过程。

**核心 idea**：用可学习代理聚合跨域不变特征增强原始表示，再将域不变特征作为"桥梁"把跨域匹配转换为域内匹配，双管齐下提升跨域 FSS。

## 方法详解

### 整体框架
DATO 建立在标准 FSS pipeline 之上（backbone 特征提取 → support-query 匹配 → 分割预测）。在此基础上引入两个核心模块：CMA 处理特征层面的域适应，CRS 处理匹配层面的域修正。两个模块协同工作，分别从特征表示和匹配过程两个维度缓解跨域差异。

### 关键设计

1. **一致性互聚合（Consistent Mutual Aggregation, CMA）**

    - 功能：学习域不变特征并用其增强各域的原始特征表示
    - 核心思路：维护一组可学习的代理向量（agents），通过交叉注意力机制（cross-attention）与来自不同域的特征交互。代理首先聚合多域特征中的共性信息（域不变成分），然后将聚合后的域不变特征反馈增强各域的原始表示。"一致性"约束确保代理对不同域输入学到的表示保持一致，避免代理退化为域特定的
    - 设计动机：传统特征增强方法（如 SE、CBAM）只在单域内操作，无法显式建模跨域共性。代理机制提供了一个显式的跨域信息交换通道，使得模型能主动提取和利用域不变信息

2. **相关性修正策略（Correlation Rectification Strategy, CRS）**

    - 功能：将直接的跨域 support-query 匹配转换为在域不变特征空间中的匹配
    - 核心思路：不直接计算 support 和 query 的相关性（因域差异大导致不可靠），而是将两者分别与代理聚合的域不变特征计算相关性，在域不变特征空间中完成匹配。域不变特征作为中间"翻译器"，将跨域匹配转化为两次域内匹配（support→域不变、域不变→query），大幅降低匹配的域敏感性
    - 设计动机：直觉类比——两个不同语言的人（support 和 query）通过共同的翻译（域不变特征）交流，比直接沟通更可靠

3. **双代理协同优化**

    - 功能：CMA 和 CRS 共享同一组代理，形成统一优化
    - 核心思路：CMA 负责"代理学好域不变特征"，CRS 负责"用好域不变特征做匹配"。两个模块的梯度同时流回代理，使代理既学到泛化性好的域不变表示，又学到对匹配最有用的特征维度
    - 设计动机：避免两个模块各自为政——如果分离训练，域不变特征可能对匹配无用，匹配空间可能不够域不变

## 实验关键数据

### 主实验（CD-FSS Benchmark, 1-shot）

| 方法 | Deepglobe | ISIC | Chest X-ray | FSS-1000 | 平均 |
|------|-----------|------|-------------|----------|------|
| PATNet | 37.89 | 33.43 | 66.61 | 78.59 | 54.13 |
| RestNet | 40.39 | 40.30 | 72.47 | 79.16 | 58.08 |
| PINet | 41.07 | 36.67 | 73.36 | 81.60 | 58.18 |
| **DATO (Ours)** | **~44** | **~42** | **~76** | **~83** | **~61** |

### 消融实验

| 配置 | 平均 mIoU |
|------|----------|
| Baseline (vanilla FSS) | ~53 |
| + CMA | ~57 |
| + CRS | ~58 |
| + CMA + CRS (DATO) | **~61** |

### 关键发现
- CMA 和 CRS 各自带来约 4-5 个点的提升，结合后有额外增益，说明两个模块互补
- 在域差异最大的场景（如自然图像→医学图像）中提升最显著，验证了方法对域差异的针对性
- 代理数量存在最优值——过少则不足以捕获域不变特征的多样性，过多则引入冗余
- CRS 的修正效果可以通过可视化相关性图直观观察——修正后的匹配更聚焦于目标区域

## 亮点与洞察
- **将域不变特征同时用于特征增强和匹配修正**的双重利用非常高效，一组代理解决两个问题
- **CRS 的"翻译器"思路**很有启发性——与其硬拉两个域的特征到同一空间，不如通过中间媒介间接匹配
- 框架设计干净，CMA 和 CRS 可以轻松插入任何现有 FSS 方法，具有即插即用的实用性
- 代理（agents）的学习不需要额外的域标签，仅通过 FSS 的分割损失即可隐式学到域不变性

## 局限与展望
- 代理数量和维度是超参数，可能需要针对不同域对进行调整
- "域不变"特征的质量高度依赖训练时见到的域多样性——若训练域组合过于单一，代理可能学不到真正通用的不变特征
- 未探讨在高 shot（5-shot、10-shot）场景下的表现，更多 support 样本可能减少 CRS 的必要性
- 计算开销分析缺失——代理的交叉注意力虽然轻量，但在推理时仍增加了额外计算
- 与近期基于 foundation model（如 SAM）的分割方法对比不足

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](the_devil_is_in_low-level_features_for_cross-domain_few-shot_segmentation.md)
- [\[ICML 2025\] Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation](../../ICML2025/segmentation/self-disentanglement_and_re-composition_for_cross-domain_few-shot_segmentation.md)
- [\[ICML 2025\] Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation](../../ICML2025/segmentation/adapter_naturally_serves_as_decoupler_for_cross-domain_few-shot_semantic_segment.md)
- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](../../AAAI2026/segmentation/bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](../../NeurIPS2025/segmentation/sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)

</div>

<!-- RELATED:END -->
