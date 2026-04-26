---
title: >-
  [论文解读] Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions
description: >-
  [ICML2025][Correlation Clustering] 提出 Sparse-Pivot 算法，在节点动态插入的 Correlation Clustering 问题中以摊销 $O_\varepsilon(\log^{O(1)} n)$ 的数据库操作实现 $(20+\varepsilon)$-近似，大幅改善了 Cohen-Addad et al. (ICML 2024) 的近似因子，并在实验中全面优于基线。
tags:
  - ICML2025
  - Correlation Clustering
  - 动态图算法
  - 节点插入
  - 次线性更新时间
  - 近似算法
---

# Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions

**会议**: ICML2025  
**arXiv**: [2507.01830](https://arxiv.org/abs/2507.01830)  
**代码**: 未公开  
**领域**: 其他/图聚类  
**关键词**: Correlation Clustering, 动态图算法, 节点插入, 次线性更新时间, 近似算法

## 一句话总结

提出 Sparse-Pivot 算法，在节点动态插入的 Correlation Clustering 问题中以摊销 $O_\varepsilon(\log^{O(1)} n)$ 的数据库操作实现 $(20+\varepsilon)$-近似，大幅改善了 Cohen-Addad et al. (ICML 2024) 的近似因子，并在实验中全面优于基线。

## 研究背景与动机

**Correlation Clustering** 是经典的图划分问题：给定完全图中节点对的"相似/不相似"标签，目标是找到一个聚类使"不一致"边数（同簇不相似 + 跨簇相似）最小化。该问题由 Bansal et al. (2004) 提出，NP-hard，当前最佳静态近似比为 1.437 (Cao et al., 2024)。

**动态场景需求**：在线商店每日上架新商品并需要动态更新商品聚类——逐个插入节点后高效更新聚类，而非重新聚类整个数据集。Cohen-Addad et al. (ICML 2024) 首次提出节点插入模型下的次线性更新时间算法，但其近似因子是一个极大的常数（论文未给出具体值）。

**本文动机**：在保持 $\mathrm{polylog}(n)$ 更新时间的同时，将近似因子显式降低到 $20+\varepsilon$，并通过实验验证实际性能优势。

## 方法详解

### 整体框架

算法基于 Behnezhad et al. (2023) 的 **5-近似 Pivot 变体**。静态版本中：
1. 为所有节点随机排序 $\pi$
2. 每个节点 $u$ 选取邻域中排名最小的邻居作为 pivot：$p(u) = \arg\min_{w \in N[u]} \pi(w)$
3. 若 $p(u) = u$ 则 $u$ 为 pivot 并建簇；若 $p(u)$ 是 pivot 则加入其簇；否则为单例

直接动态化的问题是每次插入需遍历所有邻居（$\Theta(n)$），本文通过稀疏采样解决。

### 三大关键设计

**设计1：低排名节点的精确搜索**

对 $\pi(u) \leq L/d(u)$（$L = O(\log n)$）的节点，精确扫描全部邻居。若 $u$ 是 pivot 则运行 Explore 过程更新邻居的 pivot 归属。关键观察：Reference Clustering 中的所有 pivot 以高概率满足此条件：

$$\Pr\left[\pi(u) > \frac{L}{d(u)}\right] \leq \left(1 - \frac{L}{d(u)}\right)^{d(u)} \leq \mathrm{poly}(1/n)$$

因此所有 pivot 都能被正确检测。

**设计2：高排名节点的随机采样**

对 $\pi(u) > L/d(u)$ 的节点，采样 $O(\log n)$ 个邻居，检查它们的 pivot，选排名最小的作为 $u$ 的 pivot。对于"好节点"（good node），其簇内大部分节点已正确聚类，采样能以 $1 - 1/n^{100}$ 的概率命中正确 pivot。

**设计3：Break-cluster 去噪**

在暂定聚类中，通过阈值 $t$ 将簇内度数低于 $t$ 的节点移出为单例。算法尝试 $O(\log n)$ 个候选阈值（$1, (1+\epsilon), \ldots, (1+\epsilon)^{\lceil\log n\rceil+1}$），通过采样估计每个阈值的代价，选择最优。每隔 $\epsilon \cdot |B_v|$ 次插入触发一次。

### 节点分类体系

论文定义了精细的节点分类：
- **Light 节点**：簇内度 $\leq |C|/3$（簇内非边太多）
- **Heavy 节点**：总度 $\geq \beta|C|$（跨簇边太多）
- **Poor 节点**：总度 $\leq \alpha \cdot d(\text{pivot})$（度数远小于 pivot）
- **Bad 节点**：light / heavy / poor 的并集
- **Lost 节点**：bad 邻居数 $\geq \beta \times$ good 邻居数

核心引理 (Lemma 3.2)：将 bad 和 lost 节点变为单例，期望代价仅增加 $(1+7\varepsilon)$ 倍。

### 近似保证

$$\mathbb{E}[\mathrm{cost}(\text{Sparse-Pivot})] \leq 4(1+O(\varepsilon)) \cdot \mathbb{E}[\mathrm{cost}(\text{Reference Clustering})]$$

Reference Clustering 本身是 5-近似，因此总近似比为 $4 \times 5 \times (1+O(\varepsilon)) = 20 + O(\varepsilon)$。

### 删除处理

采用懒删除策略：忽略删除操作，在累计 $\varepsilon N/6$ 次更新后全局重算。由于删除是随机的，每对节点的聚类以 $\geq 1 - 5\varepsilon/6$ 的概率保持不变。

## 实验关键数据

### 数据集

| 类型 | 数据集 | 来源 |
|------|--------|------|
| 稀疏实际图 | musae-facebook, email-Enron, ca-AstroPh, cit-HepTh | SNAP |
| 可调密度图 | drift (13910点, 129维, 阈值 $c \in \{10,15,20,25,30\}$) | UCI ML Repository |

### 主实验：Drift 数据集聚类目标值（越小越好）

| Density | Dynamic Agreement | Reference Clustering | **Sparse-Pivot** |
|---------|-------------------|---------------------|-----------------|
| 235.36 | 0.69 | 0.59 | **0.60** |
| 114.87 | 0.59 | 0.64 | **0.49** |
| 69.74 | 0.50 | 0.50 | **0.41** |
| 52.17 | 0.39 | 0.42 | **0.32** |
| 42.25 | 0.35 | 0.35 | **0.29** |

代价以 Singletons 基线归一化。Sparse-Pivot 在中低密度图上优势显著，低密度 (42.25) 时比 DA 低 17%。

### 关键发现

- 在所有数据集上 Sparse-Pivot 的近似质量均优于 Dynamic Agreement 和 Singletons 基线
- 在 SNAP 图上，Sparse-Pivot 的聚类目标值随时间步变化始终低于 DA
- Sparse-Pivot 实际运行时间也快于 Dynamic Agreement
- Break-cluster 的阈值选择是近似比从 5 放大到 20 的瓶颈，实际中可替换为更精准的实现

## 亮点与洞察

1. **显式近似因子**：首次给出节点插入 CC 问题的显式 $(20+\varepsilon)$ 近似比，相比 Cohen-Addad et al. 的未知巨大常数是质的提升
2. **精巧的节点分类**：light/heavy/poor/bad/lost 的分类体系使得理论分析能精确追踪每类节点的代价贡献
3. **采样与精确搜索的自适应切换**：根据 $\pi(u)$ 与 $L/d(u)$ 的关系自动选择策略，巧妙平衡时间与精度
4. **理论-实践一致**：不仅理论改进，实验也全面超越基线，说明设计的实用性
5. **懒删除的简洁性**：随机删除场景下直接忽略删除、定期重算，分析简洁且有效

## 局限与展望

1. **近似比仍有较大空间**：$(20+\varepsilon)$ vs. 静态最优 1.437，主要瓶颈在 Break-cluster 的 4 倍放大
2. **仅支持 soft deletion**：无法处理确定性删除，且删除必须均匀随机
3. **完全信息假设**：要求所有节点对都有相似性标签，不适用于不完全图
4. **非自适应到达序列**：假设节点到达顺序与算法决策无关
5. **实验规模有限**：未在百万级图上验证可扩展性
6. **参数 $\varepsilon$ 敏感性**：实验固定 $\varepsilon=0.1$，缺乏参数敏感性分析

## 相关工作与启发

- **Ailon et al. (2008)**: 经典 Pivot 算法 3-近似，本文的 Reference Clustering 基于其变体
- **Behnezhad et al. (2023)**: 5-近似流式 Pivot 变体，本文的直接基础
- **Cohen-Addad et al. (ICML 2024)**: 首个节点插入次线性动态算法（Dynamic Agreement），但近似常数极大
- **Dalirrooyfard et al. (2024)**: 边更新的动态 CC，$(3+\varepsilon)$-近似常数更新时间
- **Cao et al. (2024)**: 静态 CC 最优 1.437-近似

启发：动态算法中"理论近似比 vs 实际质量"的 gap 值得关注，Break-cluster 子程序是改进的关键点。

## 评分

- 新颖性: ⭐⭐⭐⭐ (节点插入动态 CC 的首个显式常数近似)
- 实验充分度: ⭐⭐⭐ (数据集较小，缺乏参数分析)
- 写作质量: ⭐⭐⭐⭐ (理论分析清晰，节点分类体系有条理)
- 价值: ⭐⭐⭐⭐ (填补了动态 CC 近似比的空白)

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/others/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/others/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ACL 2025\] Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering](../../ACL2025/others/adaptive_feature-based_low_rank_plus_sparse_decomposition_for_subspace_clusterin.md)
- [\[ICML 2025\] DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers](dsp_dynamic_sequence_parallelism_for_multi-dimensional_transformers.md)
- [\[ICML 2025\] Learning-Augmented Hierarchical Clustering](learning-augmented_hierarchical_clustering.md)

<!-- RELATED:END -->
