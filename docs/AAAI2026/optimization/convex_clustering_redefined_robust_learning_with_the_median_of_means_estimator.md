---
title: >-
  [论文解读] Convex Clustering Redefined: Robust Learning with the Median of Means Estimator
description: >-
  [AAAI 2026][优化][凸聚类] 提出 COMET（Convex Clustering with Median of Means Estimator），将中位数均值（MoM）估计器整合到凸聚类框架中，通过随机分箱、截断距离和 ADAM 优化实现对噪声和异常值的鲁棒聚类，无需预设聚类数量，在理论上证明了弱一致性，在合成和真实数据集上全面超越现有方法。
tags:
  - AAAI 2026
  - 优化
  - 凸聚类
  - 中位数均值估计
  - 鲁棒聚类
  - 异常值检测
  - ADAM优化
---

# Convex Clustering Redefined: Robust Learning with the Median of Means Estimator

**会议**: AAAI 2026  
**arXiv**: [2511.14784](https://arxiv.org/abs/2511.14784)  
**代码**: https://tinyurl.com/2v3dx75x (有)  
**领域**: Optimization / Clustering  
**关键词**: 凸聚类, 中位数均值估计, 鲁棒聚类, 异常值检测, ADAM优化

## 一句话总结
提出 COMET（Convex Clustering with Median of Means Estimator），将中位数均值（MoM）估计器整合到凸聚类框架中，通过随机分箱、截断距离和 ADAM 优化实现对噪声和异常值的鲁棒聚类，无需预设聚类数量，在理论上证明了弱一致性，在合成和真实数据集上全面超越现有方法。

## 研究背景与动机

聚类是无监督学习的基础任务，但传统方法面临多个核心挑战：

**k-means 系列**：需要预指定聚类数 k，对初始化敏感，在高维和含噪数据上退化

**凸聚类（Convex Clustering）**：通过凸优化保证全局最优解，但在高维含噪数据上面临困难——强融合正则化（大的 γ 值）会导致异常值与真实簇不当合并

**鲁棒性缺口**：现有凸聚类方法（如 Robust Convex Clustering）在权重分配上依赖 k-NN + 高斯核，任意选择带宽 ϕ 可能导致簇坍塌或任意簇形成

**核心矛盾**：凸聚类的正则化项要求融合相邻中心点，但在含噪声/异常值的数据中，异常值的"邻居"关系会误导融合过程。

**核心 idea**：利用 MoM 估计器的天然鲁棒性——将数据随机分为 L 个子集，取各子集损失的中位数而非均值，使得异常值仅污染少数子集而被中位数操作自动抑制；再结合截断距离 min(μ, ||u_i - u_j||²) 限制远距离点对的融合影响。

## 方法详解

### 整体框架

COMET 的工作流程：
1. 构建 k-NN 图和高斯权重 w_ij
2. 随机分箱（Random Binning）将数据分为 L 个子集
3. 在 MoM 目标函数上运行 ADAM 梯度下降 N 次迭代
4. 构建基于优化后中心点距离的连通图
5. 提取连通分量作为聚类结果，小簇合并为噪声簇

### 关键设计

1. **MoM 目标函数**:

    - 功能：用中位数替代均值来计算数据拟合项，抑制异常值影响
    - 核心思路：将 n 个数据点随机分为 l = O(n) 个子集 B_1, ..., B_l，每个大小为 b = ⌊n/l⌋；取各子集的凸聚类损失的中位数：
      MoM_B(U) = Median({1/(2b) Σ_{i∈B_j} ||x_i - u_i||² }_{j=1}^l)
    - 设计动机：异常值最多污染少部分子集，中位数操作天然忽略被污染的子集；MoM 估计器有严格的击穿点分析和集中度保证

2. **截断距离正则化**:

    - 功能：限制远距离点对融合项的贡献
    - 核心思路：用 min(μ, ||u_i - u_j||²) 替代 ||u_i - u_j||²，当两个中心点距离超过 μ 时，梯度归零，不再试图融合它们
    - 设计动机：防止异常值（距离远的点）通过融合正则化误导聚类中心；μ 作为超参数控制离群点检测的灵敏度

3. **随机分箱（Random Binning）**:

    - 功能：在每次迭代中随机打乱数据分配到各子集
    - 核心思路：源自随机特征方法（Rahimi & Recht），简化为固定大小分箱，每次迭代重新分配
    - 设计动机：避免固定分箱导致的偏差，随机化提供更稳定的中位数估计

### 损失函数 / 训练策略

最终目标函数：
$$C(U) = MoM_B(U) + \frac{\gamma}{2} \sum_{i,j} w_{ij} \min\{\mu, ||u_i - u_j||^2\}$$

梯度：
$$g_i = \frac{1}{b}(u_i - x_i) \mathbb{1}(i \in B_{l_t}) + \gamma \sum_j w_{ij}(u_i - u_j) \mathbb{1}(||u_i - u_j||^2 < \mu)$$

- 使用 ADAM 优化器（而非 ADMM 或 AMA），因为目标函数非凸
- 初始化 u_i = x_i（中心点从数据点出发）
- 优化完成后，用连通分量提取聚类（阈值 η₁）

## 实验关键数据

### 主实验

**真实数据集结果（10% 噪声下的 ARI）**：

| 数据集 (k) | KM | MKM | CC | RCC | RConv | RBKM | **COMET** |
|-----------|-----|------|------|------|-------|------|----------|
| Newthyroid (3) | 0.34 | 0.40 | 0.69 | 0.00 | 0.81 | 0.11 | **0.97** |
| Wisconsin (2) | 0.52 | 0.47 | 0.81 | 0.01 | 0.85 | 0.15 | **0.87** |
| Wine (3) | 0.66 | 0.59 | 0.59 | 0.00 | 0.22 | 0.01 | **0.79** |
| Dermatology (6) | 0.61 | 0.56 | 0.21 | 0.00 | 0.66 | 0.004 | **0.81** |
| Lung-Discrete (7) | 0.44 | 0.50 | 0.07 | 0.41 | 0.39 | 0.01 | **0.71** |
| ORLRaws10p (10) | 0.33 | 0.33 | 0.53 | 0.00 | 0.54 | 0.02 | **0.73** |

### 消融实验

**计算复杂度对比**：

| 算法 | 复杂度 | 说明 |
|------|-------|------|
| **COMET** | **O(Nnkd)** | N:迭代次数, n:数据点, k:邻居数, d:维度 |
| Convex Clustering | O(N(n²d + dε)) | n² 项使其在大数据上不可行 |
| RCC | O(N(n²d + nkd)) | 同样包含 n² 项 |
| RConv | O(Nnkd) | 与 COMET 相当 |

**Brain 数据集噪声鲁棒性（ARI）**：

| 噪声 % | KM | MKM | CC | RCC | RConv | **COMET** |
|--------|-----|------|------|------|-------|----------|
| 0% | 0.28 | 0.23 | 0.64 | 0.00 | 0.56 | **0.65** |
| 5% | 0.31 | 0.31 | 0.64 | 0.00 | 0.56 | **0.66** |
| 10% | 0.26 | 0.26 | 0.64 | 0.00 | 0.56 | **0.66** |
| 15% | 0.22 | - | - | 0.00 | - | **持续稳定** |

### 关键发现

- **COMET 在所有真实数据集上均显著优于其他方法**（Wilcoxon-Rank Sum 检验确认）
- **RCC 完全失效**：在几乎所有数据集上 ARI ≈ 0.00，估计簇数严重偏离（如 Newthyroid: k*=212 vs k=3）
- **COMET 的噪声鲁棒性**：随噪声增加（0%→15%），ARI 保持稳定甚至略有提升，而 KM 和 MKM 显著下降
- **自动聚类数估计**：COMET 估计的 k* 虽略大于真实 k（如 Newthyroid: k*=4.14 vs k=3），但方差极小（±0.36），远优于 CC（k*=14.14）和 RCC（k*=212）
- **高维数据的优势**：在 ORLRaws10p（d=10304）和 Lung-Discrete（d=325）等高维数据上优势更明显

## 亮点与洞察

- **理论保证完整**：Theorem 1 提供了有限样本误差界，Corollary 1.1 建立了 d=o(n) 条件下的弱一致性，Corollary 1.2 给出了 O(1/√n) 的收敛速率
- **两层鲁棒机制的叠加**：MoM 对数据拟合项做鲁棒估计 + 截断距离对融合项做鲁棒约束，双层防御使得异常值几乎无法影响聚类结果
- **无需预设 k**：通过连通分量自动确定聚类数，优于需要 Gapstat 辅助的 k-means 系列
- **实验设计的公平性**：给 k-means 系列方法也用 Gapstat 估计 k，而非直接给真实 k 值

## 局限与展望

- 目标函数非凸（因为 MoM 和 min 操作），只能保证局部最优；虽然用 ADAM 缓解，但理论全局最优性丧失
- 超参数较多（γ, μ, η₁, ϕ, k, N），调优可能需要网格搜索
- 聚类数 k* 系统性地略大于真实值，说明噪声点的过度分离仍可优化
- 时间复杂度与 RConv 相当，在超大规模数据（n > 10^6）上的可扩展性未验证
- 仅在分类标签已知的数据集上评估（用 ARI/AMI），在完全无标签场景下的实用性待验证
- 小簇合并为"噪声簇"的策略过于简化，可能误伤真实小簇

## 相关工作与启发

- MoM 估计器的鲁棒性理论（击穿点分析）为聚类鲁棒性提供了坚实的数学基础
- 凸聚类的正则化路径思想（clusterpath）可与 COMET 结合，自动选择最优 γ
- 截断距离的思想类似于 Huber 损失在回归中的应用，可以推广到其他凸优化问题的鲁棒化
- 随机分箱策略可以与 mini-batch SGD 结合，进一步提升大规模数据上的效率

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets](ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets.md)
- [Data Heterogeneity and Forgotten Labels in Split Federated Learning](data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)
- [SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)

<!-- RELATED:END -->
