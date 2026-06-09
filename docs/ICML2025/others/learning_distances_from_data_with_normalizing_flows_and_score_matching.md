---
title: >-
  [论文解读] Learning Distances from Data with Normalizing Flows and Score Matching
description: >-
  [ICML 2025][Fermat Distance] 本文提出利用 normalizing flows 和 score matching 学习密度函数与得分函数，从而高效计算基于密度的 Fermat 距离，解决了传统图方法在高维空间中收敛慢、路径粗糙的问题。
tags:
  - "ICML 2025"
  - "Fermat Distance"
  - "Normalizing Flows"
  - "Score Matching"
  - "Geodesic"
  - "Density-Based Distance"
---

# Learning Distances from Data with Normalizing Flows and Score Matching

**会议**: ICML 2025  
**arXiv**: [2407.09297](https://arxiv.org/abs/2407.09297)  
**代码**: [GitHub](https://github.com/vislearn/Fermat-Distance)  
**领域**: 度量学习 / 黎曼几何 / 密度估计  
**关键词**: Fermat Distance, Normalizing Flows, Score Matching, Geodesic, Density-Based Distance  

## 一句话总结

本文提出利用 normalizing flows 和 score matching 学习密度函数与得分函数，从而高效计算基于密度的 Fermat 距离，解决了传统图方法在高维空间中收敛慢、路径粗糙的问题。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：度量学习是机器学习中的基础任务。传统方法将数据映射到欧氏空间后使用欧氏距离，但这受限于欧氏空间的几何约束。更灵活的方法是在数据空间定义黎曼度量并求解测地距离。

**Fermat 距离**是一类基于密度的距离（DBD），核心思想类似光学中的费马最小时间原理：定义共形度量张量 $g_x(u,v) = \langle u,v\rangle / p(x)^{2\beta}$，使得低密度区域被"拉长"，高密度区域被"压缩"。测地线自然沿数据流形的高密度区域行进。

现有基于最近邻图的 Fermat 距离估计器存在两个关键问题：

**收敛极其缓慢**：因为密度估计精度差，尽管有理论收敛保证，实际表现很差

**高维路径粗糙**：图方法在高维空间中因维度灾难，路径不够平滑

## 方法详解

### 1. 维度自适应 Fermat 距离

在标准高斯分布中，若 $\beta=1$，高维数据点远离原点（距离均值 $\sqrt{D}$），导致测地线行为极端弯曲。作者提出令 $\beta = 1/D$（与维度成反比），使不同维度下的测地行为一致且数值稳定。

### 2. Normalizing Flows 密度加权图

传统方法用局部邻居间欧氏距离的幂次近似密度，精度很差。本文用 normalizing flows 学习精确密度，然后在 $k$-NN 图上用学到的密度对边权重积分近似：

$$\text{dist}(x_1, x_2) \approx \frac{\|x_1 - x_2\|}{S} \sum_{i=1}^{S} \frac{1}{p_\theta(y_{i-1/2})^\beta}$$

为数值稳定性，所有计算在 log 空间完成。

### 3. Score Matching 松弛法

重参数化后的测地方程（常欧氏速度形式）为：

$$\ddot{\varphi} - \beta(s(\varphi)\cdot\dot{\varphi})\dot{\varphi} + \beta s(\varphi)\|\dot{\varphi}\|^2 = 0$$

其中 $s(x) = \nabla_x \log p(x)$ 为 score function。松弛算法迭代更新路径中间点使其满足测地方程。实验表明用 score matching 直接学 score 比用 flow 求导效果更好，揭示了学 log 密度 vs 学其梯度之间的权衡。

### 4. Ground Truth 计算

作者开发了数值稳定的松弛方法（Algorithm 1），以常欧氏速度重参数化路径，能在已知密度分布（如高斯混合）中计算精确测地距离和测地线，作为评估基准。

## 实验

### 主实验：2D 数据集收敛分析

| 方法 | 收敛表现 |
|------|---------|
| 幂加权最短路径（PWSPD） | 极慢收敛，LPR 远高于 0 |
| Flow 密度加权图 | 与 ground truth 密度几乎相同收敛率 |
| Ground truth 密度加权图 | 最快（参考线） |

在 5 个 2D 分布上测试 1000 条随机路径，证明密度估计精度是传统方法的瓶颈。

### 高维扩展（最高 25 维）

| 方法 | 高维表现 |
|------|---------|
| 所有图方法 | 随维度增加急剧退化 |
| Score 松弛法 | 全维度保持良好表现 |

### 消融

- **Flow score vs Score matching**：直接用 flow 导数噪声过大，score matching 更精确
- **维度缩放**：$\beta = 1/D$ 显著改善数值稳定性

## 亮点与洞察

- 首次在 Fermat 距离领域提供了 ground truth 对比评估
- 揭示了传统方法理论保证与实际性能巨大差距的根源：密度估计精度不足
- 维度自适应缩放 $\beta=1/D$ 使方法可推广到高维
- Score 松弛法有效克服维度灾难

## 局限与展望

- 高维实验仅在标准高斯分布上测试，未验证真实高维数据集
- Score matching 和 normalizing flows 本身需要训练，增加前期计算成本
- 对于非常复杂的流形结构，score 模型表达能力可能不足

## 评分

⭐⭐⭐⭐ — 理论扎实、方法优雅，首次建立了 Fermat 距离精确基准并提出实用改进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Score Matching with Missing Data](score_matching_with_missing_data.md)
- [\[ICML 2025\] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization](diversity_by_design_leveraging_distribution_matching_for_offline_model-based_opt.md)
- [\[ICML 2025\] Fully Dynamic Euclidean Bi-Chromatic Matching in Sublinear Update Time](fully_dynamic_euclidean_bi-chromatic_matching_in_sublinear_update_time.md)
- [\[ICML 2025\] FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching](fedtail_federated_long-tailed_domain_generalization_with_sharpness-guided_gradie.md)
- [\[ICML 2025\] Curvature Enhanced Data Augmentation for Regression](curvature_enhanced_data_augmentation_for_regression.md)

</div>

<!-- RELATED:END -->
