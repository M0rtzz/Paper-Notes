---
title: >-
  [论文解读] Quiet Feature Learning in Algorithmic Tasks
description: >-
  [AAAI 2026][相变] 在 10 个算法任务（18,544 次训练运行，$10^9$-$10^{16}$ FLOPs）上发现，Transformer 的损失平台期并非学习停滞——模型在此期间悄悄学习了"安静特征"（中间算法子程序），这些特征不直接降低输出损失但对最终性能因果必要（消融后准确率下降 41-75%）。这挑战了用损失曲线判断训练进展的常规做法。
tags:
  - AAAI 2026
  - 相变
  - 隐式特征
  - 算法任务
  - 损失平台
  - Grokking
---

# Quiet Feature Learning in Algorithmic Tasks

**会议**: AAAI 2026  
**arXiv**: [2505.03997](https://arxiv.org/abs/2505.03997)  
**代码**: https://github.com/prudhvirajn/quiet-feature-learning-in-algorithmic-tasks  
**领域**: 深度学习理论 / 涌现  
**关键词**: 相变, 隐式特征, 算法任务, 损失平台, Grokking

## 一句话总结
在 10 个算法任务（18,544 次训练运行，$10^9$-$10^{16}$ FLOPs）上发现，Transformer 的损失平台期并非学习停滞——模型在此期间悄悄学习了"安静特征"（中间算法子程序），这些特征不直接降低输出损失但对最终性能因果必要（消融后准确率下降 41-75%）。这挑战了用损失曲线判断训练进展的常规做法。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：LLM 训练中观察到损失曲线的相变（突然下降）和长时间平台期。缩放定律假设损失平滑下降，但算法任务上的相变违背了这一假设。

**现有痛点**：

### 核心矛盾

**核心矛盾**：相变前的平台期被认为是"浪费计算"——但真的什么都没学吗？

### 现有痛点

**现有痛点**：缺乏工具探测损失平台期间模型内部发生了什么

### 解决思路

**解决思路**："涌现能力"（Emergent Abilities）的讨论缺乏内部机制的证据

**核心矛盾**：损失不变 ≠ 学习不变——模型可能在积累不直接反映在损失上的中间表示。

**本文目标** 用线性探针和消融实验证明平台期有实质性的内部学习。

**切入角度**：在 10 个算法任务的残差流上训练线性探针，检测"安静特征"——那些在损失下降前就已学会的中间计算结果。

**核心 idea**：损失平台 ≠ 学习停滞——模型在后台并行积累子程序，只有所有子程序就位时损失才突降。

## 方法详解

### 整体框架
10 个算法任务（加法/乘法/排序/搜索等）× Transformer++ 架构 × 多种模型规模 → 训练过程中在每个层/位置的残差流上训练线性探针 → 检测中间计算特征的出现时间 → 消融验证因果必要性。

### 关键设计

1. **安静特征探测**:

    - 功能：在损失平台期检测模型是否已学会中间算法步骤
    - 核心思路：对加法任务，探测进位位（carry bits）是否已被编码在残差流中——即使模型还无法正确输出最终答案
    - 发现：进位位在损失下降前就被编码（探针准确率 >90%），但此时输出损失仍在平台

2. **因果消融**:

    - 功能：证明安静特征对最终性能因果必要
    - 核心思路：在模型已学会后，消融（置零）编码安静特征的表示维度，观察性能下降
    - 结果：进位位消融导致准确率下降 41.2%-75.1%（p<0.001），证明这些特征不是冗余的

3. **跨任务普适性**:

    - 功能：验证安静特征不是加法任务的特例
    - 10 个任务都观察到类似模式：损失平台期有特征学习 → 相变后利用这些特征

### 损失函数 / 训练策略
- 标准交叉熵损失
- AdamW + 线性预热 + 余弦退火
- 总计 18,544 次训练运行

## 实验关键数据

### 主实验

| 安静特征 | 探针准确率（平台期）| 消融后准确率下降 |
|---------|-----------------|---------------|
| 加法进位位 | >90% | -41.2% ~ -75.1% |
| 排序中间比较 | >85% | 显著 |
| 搜索中间指针 | >80% | 显著 |

### 消融：相变时间线

| 阶段 | 输出损失 | 安静特征探针 | 说明 |
|------|---------|------------|------|
| 平台期早期 | 高且平 | 随机 | 尚未学习 |
| 平台期后期 | 高且平 | **>90%** | 安静学习 |
| 相变后 | 突然下降 | >95% | 特征被利用 |

### 关键发现
- **安静特征出现在损失下降之前**——模型在平台期积累子程序
- **因果必要**：不是偶然相关，消融后性能崩溃
- **所有 10 个任务都有此模式**：普适性强
- **挑战缩放定律**：幂律损失曲线无法预测相变时间

## 亮点与洞察
- **"损失平台 ≠ 学习停滞"**挑战了用损失曲线做决策（如早停/计算预算分配）的常规做法
- **对"涌现能力"的机制解释**：能力不是突然出现的，而是子程序在后台并行积累到临界点
- 对实践的直接影响：不应该在损失平台期停止训练——模型可能正在做重要的内部学习

## 局限与展望
- 仅在算法任务上验证，自然语言任务是否有类似安静特征未知
- 线性探针可能漏掉非线性编码的特征
- 仅在小模型（<100M）上测试

## 相关工作与启发
- **vs Grokking (Power et al.)**：Grokking 是训练后泛化突然出现。本文分析了 Grokking 前的内部机制
- **vs Emergent Abilities (Wei et al.)**：涌现能力的宏观观察。本文提供微观机制证据
- **vs Lottery Ticket Hypothesis**：彩票假设关注稀疑网络的存在性，本文关注特征在训练过程中的动态变化轨迹，视角互补
- 对训练监控有指导：应监控内部表示变化而非仅看损失

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "安静特征"概念新颖且深刻，揭示了训练前期特征学习的隐藏动态
- 实验充分度: ⭐⭐⭐⭐ 10 任务 × 18K 运行 × 因果消融，统计设计严谨
- 写作质量: ⭐⭐⭐⭐⭐ 发现令人兴奋，从观察到机制解释的呈现逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 对理解深度学习训练动态有重要理论贡献，挑战了 grokking 的传统解释

## 补充说明
- 该工作的方法论和实验设计对相关领域有参考价值
- 后续工作可在更多场景和更大规模上验证方法的泛化性和可扩展性
- 与近期相关工作的结合（如与 RL/MCTS/多模态方法的交叉）有潜在研究价值

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Unsupervised Feature Selection Through Group Discovery](unsupervised_feature_selection_through_group_discovery.md)
- [\[CVPR 2025\] Open Ad-Hoc Categorization with Contextualized Feature Learning](../../CVPR2025/interpretability/open_ad-hoc_categorization_with_contextualized_feature_learning.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](data_whitening_improves_sparse_autoencoder_learning.md)

<!-- RELATED:END -->
