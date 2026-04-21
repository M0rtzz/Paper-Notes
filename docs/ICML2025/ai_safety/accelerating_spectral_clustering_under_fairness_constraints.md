---
title: >-
  [论文解读] Accelerating Spectral Clustering under Fairness Constraints
description: >-
  [ICML 2025][AI安全][公平聚类] 将公平谱聚类（Fair SC）问题转化为凸差分（DC）优化框架，通过变量增广策略和 ADMM 类型算法，避免了昂贵的特征分解计算，在大规模问题上实现显著加速。
tags:
  - ICML 2025
  - AI安全
  - 公平聚类
  - 谱聚类
  - DC优化
  - ADMM
  - 公平性约束
---

# Accelerating Spectral Clustering under Fairness Constraints

**会议**: ICML 2025  
**arXiv**: [2506.08143](https://arxiv.org/abs/2506.08143)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 公平聚类, 谱聚类, DC优化, ADMM, 公平性约束

## 一句话总结
将公平谱聚类（Fair SC）问题转化为凸差分（DC）优化框架，通过变量增广策略和 ADMM 类型算法，避免了昂贵的特征分解计算，在大规模问题上实现显著加速。

## 研究背景与动机

**领域现状**：算法决策系统在医疗、教育等关键领域的广泛应用引发公平性担忧。公平聚类要求各人口群组在每个簇中的比例与总体一致。

**现有痛点**：Kleindessner et al. (2019) 将公平性约束引入谱聚类，但依赖计算昂贵的公平约束图拉普拉斯矩阵的特征分解，限制了在大规模数据上的应用。Wang et al. (2023) 有所改进但仍需特征分解。

**核心矛盾**：特征分解的计算复杂度为 $O(n^3)$，随数据量增长不可扩展。

**本文目标**：如何在保持公平性约束的同时高效求解谱聚类？

**切入角度**：谱聚类本质上是带正交约束的迹最大化问题，可自然嵌入 DC 框架——公平性约束通过变量增广整合进 ADMM。

**核心 idea**：用 DC-ADMM 替代特征分解，每个子问题可用快速梯度法求解。

## 方法详解

### 整体框架
1. 将 Fair SC 问题写为 DC 形式：$\min f(X) - g(X)$，其中 $f$ 和 $g$ 都是凸函数
2. 引入辅助变量将公平性约束与正交约束解耦
3. 用 ADMM 交替优化各子问题
4. DC 步骤中用梯度下降近似求解

### 关键设计

1. **DC 问题转化**:

    - 功能：将 Fair SC 的迹最大化+公平约束转化为凸差分优化
    - 核心思路：$\max \text{tr}(X^T L X)$ s.t. $X^T X = I$, 公平约束 → DC 形式通过将正交约束编码为罚项
    - 设计动机：DC 框架有成熟的优化工具，避免特征分解

2. **变量增广 + ADMM**:

    - 功能：引入辅助变量将不同约束分配到不同子问题
    - 核心思路：$X$（聚类指示）和 $Y$（公平性投影）分离，ADMM 交替更新
    - 设计动机：每个子问题都可以高效求解——无需全矩阵特征分解

### 损失函数 / 训练策略
- DC 迭代：外层 DCA（DC Algorithm）+ 内层 ADMM
- 收敛到 DC 临界点

## 实验关键数据

### 主实验
不同数据规模下的计算时间对比：

| 数据集 | n | Fair SC (原始) | Wang et al. | 本文 | 加速比 |
|--------|---|---------------|-------------|------|--------|
| 合成数据 | 1K | 2.1s | 0.8s | 0.3s | 7× |
| 合成数据 | 10K | 210s | 85s | 12s | 17× |
| Bank | 45K | >1h | 1800s | 180s | ~10× |

### 消融实验

| 配置 | 时间 | 聚类质量 | 说明 |
|------|------|---------|------|
| 无公平约束（标准 SC） | 最快 | NMI 最高但不公平 | 基线 |
| Fair SC (特征分解) | 最慢 | 公平+质量好 | 先前方法 |
| 本文 DC-ADMM | 快 10×+ | 公平+质量相当 | 效率显著提升 |

### 关键发现
- 随数据量增长，加速比越来越大——本文方法扩展性远优于特征分解方法
- 聚类质量（NMI、balance）与先前方法持平
- 簇数 k 增大时加速效果更明显

## 亮点与洞察
- DC 框架+ADMM 为公平约束优化提供了一种通用且高效的方法论
- 避免特征分解使方法扩展到万级样本成为可能
- 变量增广策略将公平约束自然融入优化框架

## 局限与展望
- DC 优化只保证收敛到临界点，非全局最优
- 需要预计算图拉普拉斯矩阵，对超大规模数据仍有挑战
- 公平性定义仅考虑群组公平，未涉及个体公平

## 相关工作与启发
- **vs Kleindessner et al.**: 原始 Fair SC 方法，需完整特征分解
- **vs Wang et al.**: 部分加速但仍依赖特征分解，本文彻底避免

## 评分
- 新颖性: ⭐⭐⭐⭐ DC+ADMM 用于公平谱聚类是新颖的技术贡献
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据，扩展性验证充分
- 写作质量: ⭐⭐⭐⭐ 数学清晰，结构规范
- 价值: ⭐⭐⭐⭐ 使公平聚类在实际规模数据上可用

<!-- RELATED:START -->

## 相关论文

- [Fairness under Competition](../../NeurIPS2025/ai_safety/fairness_under_competition.md)
- [Unifying Proportional Fairness in Centroid and Non-Centroid Clustering](../../NeurIPS2025/ai_safety/unifying_proportional_fairness_in_centroid_and_non-centroid_clustering.md)
- [Relative Error Fair Clustering in the Weak-Strong Oracle Model](relative_error_fair_clustering_in_the_weak-strong_oracle_model.md)
- [Learning Safety Constraints for Large Language Models](learning_safety_constraints_for_large_language_models.md)
- [Truth, Justice, and Secrecy: Cake Cutting Under Privacy Constraints](../../AAAI2026/ai_safety/truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)

<!-- RELATED:END -->
