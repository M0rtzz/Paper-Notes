---
title: >-
  [论文解读] LapSum -- One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection
description: >-
  [ICML2025][differentiable sorting] 提出 LapSum，基于 Laplace 分布累积密度函数之和的闭式可逆公式，统一解决可微 ranking、sorting、top-k 选择和置换矩阵四大排序问题，时间复杂度仅 $O(n\log n)$、空间 $O(n)$，在大规模场景下显著优于现有方法。
tags:
  - ICML2025
  - differentiable sorting
  - differentiable top-k
  - soft ranking
  - Laplace distribution
  - permutation learning
---

# LapSum -- One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection

**会议**: ICML2025  
**arXiv**: [2503.06242](https://arxiv.org/abs/2503.06242)  
**代码**: [github.com/gmum/LapSum](https://github.com/gmum/LapSum)  
**领域**: 可微排序 / 可微优化  
**关键词**: differentiable sorting, differentiable top-k, soft ranking, Laplace distribution, permutation learning

## 一句话总结

提出 LapSum，基于 Laplace 分布累积密度函数之和的闭式可逆公式，统一解决可微 ranking、sorting、top-k 选择和置换矩阵四大排序问题，时间复杂度仅 $O(n\log n)$、空间 $O(n)$，在大规模场景下显著优于现有方法。

## 研究背景与动机

排序 (sorting)、排名 (ranking)、top-k 选择等操作在推荐系统、多标签分类、稀疏网络提取等任务中广泛使用，但这些操作本质上是**分段常数函数**，不可微，无法直接用梯度下降优化。现有方案存在以下痛点：

- **最优传输方法** (Cuturi et al., 2019; Xie et al., 2020)：理论优美但计算量大，大规模场景难以应用
- **排列学习方法** (NeuralSort, SoftSort, SinkhornSort)：需要 Sinkhorn 迭代或复杂矩阵运算，无闭式解
- **平滑近似方法** (Berrada et al., 2018)：精度与速度存在 trade-off
- 多数方法时间复杂度为 $O(n^2)$ 或更高，在 $n$ 和 $k$ 较大时内存爆炸
- 部分方法不输出概率向量、不支持 GPU 并行

本文目标：设计一个**理论统一、闭式可解、高效可并行**的框架，一次性解决所有可微排序问题。

## 方法详解

### 核心思想：F-Sum 框架

选取任意偶对称正密度函数 $f$ 及其累积分布函数 $F$，定义带尺度参数 $\alpha$ 的 F-Sum：

$$F\text{-Sum}_\alpha(r, x) = \sum_{i=0}^{n-1} F_\alpha(x - r_i), \quad F_\alpha(x) = F\left(\frac{x}{\alpha}\right)$$

**关键性质**：$F_\alpha$ 严格单调 → $F\text{-Sum}_\alpha$ 关于 $x$ 严格单调 → 存在唯一逆函数 $F\text{-Sum}^{-1}_\alpha(r, k)$。

基于此框架，四大可微排序操作统一定义为：

| 操作 | 定义 |
|------|------|
| **Soft Ranking** | $F\text{-Rank}_\alpha(r_j) = F\text{-Sum}_\alpha(r, r_j) - \frac{1}{2}$ |
| **Soft Sorting** | $(F\text{-Sort}_\alpha(r))_l = F\text{-Sum}^{-1}_\alpha(\frac{1}{2} + l)$ |
| **Soft Top-k** | $p_i = F_\alpha(b - r_i)$，其中 $b = F\text{-Sum}^{-1}_\alpha(r, k)$ |
| **Soft Permutation** | $[F_\alpha(b_{i+1} - r_j) - F_\alpha(b_i - r_j)]_{i,j}$ |

当 $\alpha \to 0^+$ 时，所有 soft 操作均收敛到对应的 hard 操作（排序/排名/top-k/置换矩阵）。

### 为什么选 Laplace 分布？

将 $F$ 取为标准 Laplace 分布的 CDF：

$$\text{Lap}(x) = \begin{cases} \frac{1}{2}e^x & x \leq 0 \\ 1 - \frac{1}{2}e^{-x} & x > 0 \end{cases}$$

Laplace CDF 的分段指数结构使得 $\text{Lap-Sum}$ 在每个区间 $[r_j, r_{j+1}]$ 上可表示为：

$$\text{Lap-Sum}(x) = \frac{1}{2}a_j e^{(x-r_{j+1})/\alpha} - \frac{1}{2}b_{j+1}e^{(r_j - x)/\alpha} + c_{j+1}$$

其中 $a_j, b_j, c_j$ 可通过递推 $O(n)$ 计算（且支持 prefix scan 并行化）。逆函数也有**闭式解**，无需迭代求解。

### 梯度计算

Top-k 的导数矩阵 $D = \frac{\partial P}{\partial r} = s\,q^T - \text{diag}(s)$，其中：
- $s_i = \frac{1}{\alpha}\min(p_i, 1-p_i)$
- $q = \text{softmax}(-|b-r_i|/\alpha)$

直接计算为 $O(n^2)$，但利用矩阵-向量乘法 $Dv = \langle q,v\rangle s - s \odot v$，梯度传播仅需 $O(n)$。整体（含排序预处理）为 $O(n\log n)$。

## 实验关键数据

### Top-k 分类（CIFAR-100, ResNet18）

| 方法 | ACC@1 | ACC@5 |
|------|-------|-------|
| SinkhornSort | 61.89 | 86.94 |
| DiffSortNets | 62.00 | 86.73 |
| **Lap-Top-k** | **64.53** | **88.51** |

### ImageNet-21K-P 微调（ResNeXt-101）

| 方法 | ACC@1 | ACC@5 |
|------|-------|-------|
| DiffSortNets | 40.22 | 70.88 |
| **Lap-Top-k** | **40.48** | **71.05** |

### kNN 图像分类

| 方法 | MNIST | CIFAR-10 |
|------|-------|----------|
| kNN+NeuralSort | 99.5 | 90.7 |
| kNN+SOFT Top-k | 99.4 | **92.6** |
| **kNN+Lap-Top-k** | **99.4** | 92.2 |

### Large-MNIST 置换学习准确率（%）

| 方法 | n=3 | n=5 | n=7 |
|------|-----|-----|-----|
| DNeural | 93.0 | 83.7 | 73.8 |
| SNeural | 92.7 | 83.5 | 74.1 |
| **LapSum** | **94.2** | **85.3** | 74.1 |

### 时间/内存复杂度

在 $k=n/2$ 设定下，LapSum 的内存和运行时间均优于或持平所有对比方法，尤其在高维 ($n$ 大) 时优势明显。Critical difference 统计检验确认 LapSum 排名第一。

## 亮点与洞察

1. **理论统一性极强**：一个 F-Sum 框架统一推导 ranking/sorting/top-k/permutation 四大操作，优雅简洁
2. **闭式解 + 闭式梯度**：不需要 Sinkhorn 迭代，不需要 LP 求解器，实现仅 26 行伪代码
3. **$O(n\log n)$ 时间 + $O(n)$ 空间**：与排序操作本身同阶，理论近最优
4. **支持并行**：递推序列 $a_j, b_j$ 可用 prefix scan 并行化，提供 CPU 和 CUDA 实现
5. **$\alpha$ 可训练**：尺度参数可作为可学习参数，自动在 hard/soft 之间自适应
6. **输出保证为概率向量**：$\sum_i p_i = k$ 严格成立，优于部分不保证概率归一的方法

## 局限与展望

1. **置换学习在大 $n$ 时退化**：$n=9, 15$ 时 LapSum 不如 NeuralSort，可能因 Laplace 分布尾部衰减过快导致 soft permutation 矩阵在大维度时区分度不足
2. **$\alpha$ 敏感性**：实验表明 $\alpha$ 值对结果影响大，需要仔细调参或学习
3. **仅验证了分类任务**：推荐系统、NLP 排序、信息检索等实际应用场景未验证
4. **与 Gumbel-Softmax 等采样方法的关系未深入讨论**：在需要离散采样的场景中如何使用 LapSum 不明确
5. **数值稳定性**：虽然论文声称数值稳定，但在极端 $\alpha$ 或极大 $n$ 时的实际表现需进一步验证

## 相关工作与启发

- **NeuralSort** (Grover et al., 2019)：基于 Gumbel 分布的可微排列，LapSum 选 Laplace 更高效
- **DiffSortNets** (Petersen et al., 2022)：基于排序网络的可微排序，复杂度较高
- **Fast Differentiable Sorting** (Blondel et al., 2020)：基于正则化最优传输，不输出概率
- **Optimal Transport Top-k** (Xie et al., 2020)：OT 框架可微 top-k，大规模时计算瓶颈
- 启发：选择合适的概率分布（如 Laplace 的分段指数结构）可带来计算上的巨大收益，这一思路可推广到其他需要可微离散操作的问题

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — Laplace CDF 求逆思路新颖，统一框架优雅
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 top-k/排列/kNN，但应用场景偏窄
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，实验对比完整
- 价值: ⭐⭐⭐⭐⭐ — 提供开箱即用的 CPU/CUDA 实现，有望成为可微排序的标准工具

<!-- RELATED:START -->

## 相关论文

- [A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition](../../ICCV2025/others/a_hyperdimensional_one_place_signature_to_represent_them_all_stackable_descripto.md)
- [Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation](bipartite_ranking_from_multiple_labels_on_loss_versus_label_aggregation.md)
- [One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit](../../ACL2025/others/one_for_all_update_parameterized_knowledge_across_multiple_models_with_once_edit.md)
- [Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method](efficient_optimization_with_orthogonality_constraint_a_randomized_riemannian_sub.md)
- [Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)

<!-- RELATED:END -->
