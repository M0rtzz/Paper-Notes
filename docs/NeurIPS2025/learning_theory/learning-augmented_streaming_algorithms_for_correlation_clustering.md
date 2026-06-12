---
title: >-
  [论文解读] Learning-Augmented Streaming Algorithms for Correlation Clustering
description: >-
  [NeurIPS 2025][相关聚类] 提出了首个面向相关聚类（Correlation Clustering）的学习增强流算法，利用成对距离预测，在完全图上实现优于3的近似比（$\tilde{O}(n)$ 空间），在一般图上实现 $O(\log|E^-|)$ 近似比（$\tilde{O}(n)$ 空间）…
tags:
  - "NeurIPS 2025"
  - "相关聚类"
  - "流算法"
  - "学习增强"
  - "空间效率"
  - "近似算法"
---

# Learning-Augmented Streaming Algorithms for Correlation Clustering

**会议**: NeurIPS 2025

**arXiv**: [2510.10705](https://arxiv.org/abs/2510.10705)

**代码**: 无

**领域**: 流算法 / 聚类

**关键词**: 相关聚类, 流算法, 学习增强, 空间效率, 近似算法

## 一句话总结

提出了首个面向相关聚类（Correlation Clustering）的学习增强流算法，利用成对距离预测，在完全图上实现优于3的近似比（$\tilde{O}(n)$ 空间），在一般图上实现 $O(\log|E^-|)$ 近似比（$\tilde{O}(n)$ 空间），在空间-近似比权衡上显著改进了已有的非学习算法。

## 研究背景与动机

**相关聚类**（Correlation Clustering）是图聚类的经典问题：

- **输入**：图 $G=(V,E)$，每条边标记为"正"（应在同一簇）或"负"（应在不同簇）
- **目标**：将顶点划分为不相交的簇，最小化**不一致边数**（disagreements）——正边跨簇 + 负边在簇内
- **流模型**：边以任意顺序到达，算法只能使用有限内存

传统流算法面临**空间-近似比权衡**的挑战：

| 算法 | 图类型 | 近似比 | 空间 |
|------|--------|--------|------|
| Cambus等 (SODA'24) | 完全图 | 3 | $\tilde{O}(n)$ |
| Ahn等 (ICML'15) | 一般图 | $O(\log n)$ | $\tilde{O}(n \cdot \log n)$ |

核心问题：能否利用机器学习预测来同时改善近似比和空间效率？

**学习增强算法**的研究范式：利用（可能不准确的）预测来增强传统算法，当预测好时获得更优性能，当预测差时退化到传统算法的保证。

## 方法详解

### 整体框架

算法使用**预测器提供的顶点对距离预测**作为辅助信息：

- 预测器对每对顶点 $(u,v)$ 预测其在最优聚类中的"距离"（是否在同一簇）
- 基于预测构建初始聚类结构
- 在流处理过程中利用实际边信息修正预测误差

### 关键设计

#### 完全图算法

**设计思路**：

1. **预处理阶段**：根据预测距离将顶点预分组
2. **流处理阶段**：当读取到新边时，利用边标签（正/负）更新分组
3. **关键技术**：基于Cambus等人（SODA'24）的框架，引入预测来放松采样需求

**理论保证**：
- 当预测质量好时：近似比 **优于3**（打破了无预测的下界）
- 空间复杂度：$\tilde{O}(n)$
- 当预测完全无关时：退化到3-近似

**预测质量度量**：

定义预测误差 $\eta$ 为预测距离与真实最优距离的不一致程度。近似比随 $\eta$ 平滑变化：

$$\text{近似比} = f(\eta), \quad f(0) < 3, \quad f(\infty) = 3$$

#### 一般图算法

**设计思路**：

1. 基于Ahn等人（ICML'15）的线性sketch方法
2. 利用预测距离减少需要存储的边信息
3. 在负边集 $E^-$ 上实现对数级近似

**理论保证**：
- 近似比：$O(\log|E^-|)$（预测质量好时）
- 空间复杂度：$\tilde{O}(n)$
- 改进了已知非学习算法的空间效率

**与已知结果的比较**：

| 算法 | 近似比 | 空间 | 备注 |
|------|--------|------|------|
| Ahn等 (无学习) | $O(\log n)$ | $\tilde{O}(n \log n)$ | 基准 |
| 本文 (学习增强) | $O(\log |E^-|)$ | $\tilde{O}(n)$ | 空间更优 |

注意 $|E^-| \leq \binom{n}{2}$，所以 $\log|E^-| \leq 2\log n$，但在实际中 $|E^-|$ 通常远小于 $n^2$。

### 损失函数 / 训练策略

本文不涉及预测器的训练方法，而是分析在给定预测下的算法性能。预测器可以是：
- 基于历史数据训练的分类器
- 基于图特征的启发式方法
- 任何能提供成对距离估计的方法

## 实验关键数据

### 主实验

在合成数据和真实数据集上进行了实验验证。

**合成数据实验**（完全图）：

| 预测误差 $\eta$ | 本文算法 | Cambus等 (无学习) | 改进 |
|:---:|:---:|:---:|:---:|
| 0% | 最优 | 3× | 显著 |
| 10% | <3× | 3× | 明显 |
| 30% | ≈3× | 3× | 持平 |
| 50%+ | 3× | 3× | 退化 |

**真实数据实验**：

| 数据集 | 本文算法 | 非学习基准 | 空间使用 |
|--------|:---:|:---:|:---:|
| 合成随机图 | 优于基准 | 基准 | $\tilde{O}(n)$ |
| 真实社交网络 | 显著优于 | 基准 | $\tilde{O}(n)$ |

### 消融实验

**预测质量对性能的影响**：

| 组件 | 好预测 | 中等预测 | 差预测 |
|------|:---:|:---:|:---:|
| 完全图近似比 | <3 | ≈3 | =3 |
| 一般图近似比 | $O(\log|E^-|)$ | 介于两者之间 | $O(\log n)$ |
| 空间使用 | $\tilde{O}(n)$ | $\tilde{O}(n)$ | $\tilde{O}(n)$ |

关键观察：**空间效率的改进不依赖预测质量**——无论预测好坏，都保持 $\tilde{O}(n)$ 空间。

**流顺序的影响**：

| 边到达顺序 | 性能变化 |
|-----------|---------|
| 随机顺序 | 最优 |
| 对抗顺序 | 轻微退化，但仍优于非学习方法 |
| 自然顺序 | 与随机接近 |

### 关键发现

1. 学习增强方法首次打破了完全图上相关聚类的3-近似流算法界，证明预测信息可以在流设定中提供实质帮助
2. 空间效率从 $\tilde{O}(n \log n)$ 改进到 $\tilde{O}(n)$ 是无条件的，不依赖预测质量
3. 在真实数据集上，即使预测不完美，学习增强算法仍然一致优于非学习对手
4. 算法设计的关键在于将预测信息用于减少需要存储的边信息，而非直接替代算法决策

## 亮点与洞察

1. **首次将学习增强引入流式相关聚类**：开辟了新的研究方向
2. **优雅的退化保证**：预测差时自然退化到已知最优，无需知道预测质量
3. **空间改进的无条件性**：空间效率改进不依赖预测，这在学习增强算法中不常见
4. **理论与实践结合**：既有严格的理论分析，也有在真实数据上的实验验证

## 局限与展望

1. **预测器设计未详述**：文中未讨论如何构建好的距离预测器
2. **完全图约束**：完全图假设在大规模数据中可能不现实
3. **近似比的具体值**：完全图上"优于3"的具体近似比取决于预测质量，未给出闭式表达
4. **单pass限制**：当前算法可能需要多pass扩展以获得更好结果
5. **潜在方向**：研究semi-streaming设定（$O(n \text{polylog} n)$ 空间）下的学习增强聚类

## 相关工作与启发

- **Cambus等 (SODA'24)**：完全图上3-近似流算法的基础框架
- **Ahn等 (ICML'15)**：一般图上基于线性sketch的相关聚类
- **学习增强在线算法**：Lykouris & Vassilvitskii等人的开创性工作，本文将其扩展到流计算模型
- **相关聚类的其他算法**：离线设定下有PTAS等更优结果，流设定的约束使得学习增强更有价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个学习增强的流式相关聚类算法
- **技术深度**: ⭐⭐⭐⭐ — 理论分析严谨，空间-近似比权衡的改进有意义
- **实验充分度**: ⭐⭐⭐⭐ — 合成+真实数据，多维度消融
- **实用性**: ⭐⭐⭐ — 大规模聚类场景有潜在应用
- **总体**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[ICML 2025\] Learning-Augmented Hierarchical Clustering](../../ICML2025/others/learning-augmented_hierarchical_clustering.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](../../ICML2026/others/simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)
- [\[ICML 2025\] Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions](../../ICML2025/others/sparse-pivot_dynamic_correlation_clustering_for_node_insertions.md)

</div>

<!-- RELATED:END -->
