---
title: >-
  [论文解读] SpEx: A Spectral Approach to Explainable Clustering
description: >-
  [NeurIPS 2025][可解释聚类] 提出SpEx，基于谱图划分的通用可解释聚类方法，可将任意参考聚类（无需质心）通过坐标切割决策树"圆化"为可解释聚类，或直接在kNN图上进行无参考聚类。
tags:
  - NeurIPS 2025
  - 可解释聚类
  - 谱图划分
  - 坐标切割
  - 决策树
  - Cheeger不等式
---

# SpEx: A Spectral Approach to Explainable Clustering

**会议**: NeurIPS 2025  
**arXiv**: [2511.00885](https://arxiv.org/abs/2511.00885)  
**代码**: [GitHub](https://github.com/talargv/SpEx)  
**领域**: 可解释聚类, 谱方法, 图划分  
**关键词**: 可解释聚类, 谱图划分, 坐标切割, 决策树, Cheeger不等式

## 一句话总结
提出SpEx，基于谱图划分的通用可解释聚类方法，可将任意参考聚类（无需质心）通过坐标切割决策树"圆化"为可解释聚类，或直接在kNN图上进行无参考聚类。

## 研究背景与动机
- 可解释聚类(Moshkovitz 2020)：用二叉决策树描述聚类，每个内部节点对应单坐标阈值
- 现有方法局限：
    - **IMM/EMN**：需要参考聚类具有质心（k-means），无法处理核k-means/谱聚类
    - **Kernel IMM**：仅支持特定核函数，复杂度高，不可扩展
    - **CART**：非为聚类设计，最大化Gini不纯度降低，可能选择与聚类不相容的切割
- 迄今没有真正通用的可解释聚类方法——不受参考聚类类型限制，在真实数据上鲁棒

## 方法详解

### 整体框架
利用Cheeger不等式的推广将坐标切割与图的切割电导联系起来

### 关键设计

#### 核心定理(Theorem 2.2)
对$\mathbb{R}^d$中的点集$X$和图$G(X,E,w)$，存在一个坐标切割$S_{j,\tau}$使得：
$$\Psi_G(S_{j,\tau}(X)) \leq \sqrt{\frac{\mathbb{E}_{adj}\|x-y\|_2^2}{\mathbb{E}_{all}\|x-y\|_2^2}}$$
- 分子：图中相邻对的期望平方距离
- 分母：所有对的期望平方距离
- 当图捕获了数据的有意义几何结构时（邻居比随机对更近），电导上界更小

#### SpEx-Clique（参考聚类模式）
- 给定**任意**参考聚类，构建**团图**(clique graph)：同簇点之间连边
- 迭代选择电导最小的坐标切割构建决策树
- 无需质心，无需特定核函数

#### SpEx-kNN（无参考模式）
- 直接在数据集上构建kNN图($k=20$)
- 无需任何参考聚类即可产生可解释聚类

#### 迭代树构建(Algorithm 1)
- 优先队列维护所有叶节点
- 每次选择切割后电导降低最大的叶节点进行分裂
- 目标：最小化多路归一化切割目标$\bar{\Psi}_G(T) = \sum_{v \in \mathcal{L}(T)}\psi_G(X_v)$
- 叶节点数可自由选择（不受参考聚类簇数限制）

### 统一理论框架：非均匀稀疏切割
通过Trevisan的非均匀稀疏切割框架，统一理解现有方法：
- **IMM** → $G$=星图，$H$=直径质心对单边图
- **EMN** → $G$=星图，$H$=质心间完全图
- **CART** → $H$=独立集图，$G$=$H$-度加权完全图

**CART失败原因的图论解释**：独立集图激励分离不同簇的点对，但**不惩罚分离同簇点**——导致贪心选择子最优切割。

### 计算效率
- 每层时间：$O(dn(\log n + S))$
- SpEx-Clique：$S=O(1)$（维护两个簇直方图即可更新切割分数）
- SpEx-kNN：$S=O(\kappa)$，$\kappa$为邻居数
- 总时间：$O(kdn\log n)$（SpEx-Clique）

## 实验关键数据

### 主实验（8个真实数据集，ARI/AMI评估）

| 数据集 | n | d | k | SpEx-Clique | SpEx-kNN | EMN | CART (k-means) | CART (谱) |
|-------|---|---|---|-------------|----------|-----|--------------|----------|
| CIFAR-10 | 50K | 512 | 10 | **最佳/最佳** | 中 | 中 | 差 | 差 |
| MNIST | 60K | 512 | 10 | **最佳** | 中 | 中 | 差 | 中 |
| 20News | 11K | 768 | 20 | **最佳** | 中 | 中 | 差 | 差 |
| Beans | 13K | 16 | 7 | 中 | **最佳** | 中 | 中 | 中 |
| Ecoli | 336 | 7 | 8 | 中 | **最佳** | 中 | 中 | 中 |
| Iris | 150 | 4 | 3 | 中 | **最佳** | 中 | 中 | 良 |

### 方法稳定性分析

| 特性 | SpEx-Clique | SpEx-kNN | EMN | CART | Kernel IMM |
|-----|------------|---------|-----|------|-----------|
| 需要质心 | ✗ | ✗ | ✓ | ✗ | ✗ |
| 需要参考聚类 | ✓ | ✗ | ✓ | ✓ | ✓ |
| 大数据扩展性 | ✓ | ✓ | ✓ | ✓ | ✗ |
| 一致性高表现 | ✓ | 低维佳 | 中 | 小数据佳 | 不确定 |

### 关键发现
- SpEx-Clique是最一致的高性能算法——在大多数情况下超越每个基线
- SpEx-kNN在低维数据集(≤16维)上表现最佳
- CART在小数据集上OK，但大数据集上表现差——验证了toy example失败模式在真实数据上确实存在
- Kernel IMM在大数据集上无法运行（可扩展性差）
- SpEx的叶节点数量可灵活设置——增加叶节点降低可解释性代价但降低可解释性

## 亮点与洞察
1. **真正通用**：首个可组合任意参考聚类的可解释聚类方法
2. **统一理论视角**：非均匀稀疏切割框架统一理解IMM、EMN、CART
3. **CART失败的深层原因**：独立集图不惩罚分离同簇点→贪心选择劣质切割
4. **无参考模式**：SpEx-kNN可跳过参考聚类直接从数据产出可解释聚类
5. **高效实现**：SpEx-Clique仅需O(1)时间更新切割分数

## 局限与展望
- 仅考虑单坐标(axis-aligned)切割——更复杂但可能更强大的斜切割未探索
- SpEx-kNN在高维数据上表现不如SpEx-Clique
- 尚未分析可解释性代价的理论界
- k近邻图的$k$值选择需用户指定
- 可扩展到在线/流式场景

## 相关工作与启发
- Moshkovitz等人的可解释聚类定义和IMM算法(2020)引发大量理论工作
- EMN(2022)通过比率优化改进IMM
- Kernel IMM(2024)扩展到核k-means但可扩展性有限
- 本文首次将谱图划分理论引入可解释聚类

## 评分
- 新颖性：⭐⭐⭐⭐⭐ （谱方法+统一框架的优雅结合）
- 技术深度：⭐⭐⭐⭐⭐ （理论贡献+算法设计+统一分析）
- 实验充分性：⭐⭐⭐⭐ （8个数据集，多种基线）
- 写作质量：⭐⭐⭐⭐⭐ （逻辑清晰，理论直觉与严格证明平衡好）

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Dynamic Algorithm for Explainable k-medians Clustering under lp Norm](dynamic_algorithm_for_explainable_k-medians_clustering_under_lp_norm.md)
- [\[ACL 2026\] A Structured Clustering Approach for Inducing Media Narratives](../../ACL2026/interpretability/a_structured_clustering_approach_for_inducing_media_narratives.md)
- [\[NeurIPS 2025\] Additive Models Explained: A Computational Complexity Approach](additive_models_explained_a_computational_complexity_approach.md)
- [\[NeurIPS 2025\] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)
- [\[ICML 2025\] A Reasoning-Based Approach to Cryptic Crossword Clue Solving](../../ICML2025/interpretability/a_reasoning-based_approach_to_cryptic_crossword_clue_solving.md)

<!-- RELATED:END -->
