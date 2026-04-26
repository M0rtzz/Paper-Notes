---
title: >-
  [论文解读] Depth-Bounds for Neural Networks via the Braid Arrangement
description: >-
  [NeurIPS 2025][ReLU网络深度下界] 本文证明了在 $\mathcal{B}_d^0$-conforming 约束下，ReLU 网络精确表示 $\max\{0, x_1, \ldots, x_d\}$ 需要 $\Omega(\log \log d)$ 层——这是首个不限制权重的非常数深度下界；同时证明 rank-(3,2) maxout 网络可以计算 7 个数的最大值，说明标准上界不紧。
tags:
  - NeurIPS 2025
  - ReLU网络深度下界
  - 分段线性函数
  - braid排列
  - maxout网络
  - 组合证明
---

# Depth-Bounds for Neural Networks via the Braid Arrangement

**会议**: NeurIPS 2025  
**arXiv**: [2502.09324](https://arxiv.org/abs/2502.09324)  
**代码**: 无  
**领域**: others (深度学习理论 / 表达力分析)  
**关键词**: ReLU网络深度下界, 分段线性函数, braid排列, maxout网络, 组合证明

## 一句话总结
本文证明了在 $\mathcal{B}_d^0$-conforming 约束下，ReLU 网络精确表示 $\max\{0, x_1, \ldots, x_d\}$ 需要 $\Omega(\log \log d)$ 层——这是首个不限制权重的非常数深度下界；同时证明 rank-(3,2) maxout 网络可以计算 7 个数的最大值，说明标准上界不紧。

## 研究背景与动机

**领域现状**：ReLU 网络可以精确表示所有连续分段线性（CPWL）函数。Arora et al. (2018) 证明 $\lceil \log_2(d+1) \rceil$ 层足够。Bakaev et al. (2025) 改进到 $\lceil \log_3(d-1) \rceil + 1$ 层。但深度**下界**几乎空白。

**核心开放问题**：精确表示 $\mathbb{R}^d$ 上所有 CPWL 函数最少需要多少层？目前最好的无条件下界仅为 2。

**关键简化**：由 Wang & Sun (2005) 知，任意 CPWL 函数可化为 $\max$ 函数的表示——因此问题归结为：计算 $d$ 个数的最大值需要多深的网络？

**研究路线**：限制 breakpoints（断点）位于 braid 排列上——即 $\mathcal{B}_d^0$-conforming。这一限制虽非完全一般，但提供了可分析的框架。

## 方法详解

### 核心概念

**Braid 排列** $\mathcal{B}_d$：由超平面 $x_i = x_j$（$1 \leq i < j \leq d$）构成的超平面配置。$\mathcal{B}_d^0$ 额外包含 $x_i = 0$。

**$\mathcal{B}_d^0$-conforming 网络**：所有神经元的断点均位于 braid 排列的超平面上。标准二叉树计算 max 的方式（逐层取两两最大值）就是 $\mathcal{B}_d^0$-conforming 的。

**关键同构**：$\mathcal{B}_d$-compatible 的 CPWL 函数空间 $\mathcal{V}_{\mathcal{B}_d}$ 与集函数空间 $\mathcal{F}_d = \mathbb{R}^{2^{[d]}}$ 同构（维度为 $2^d$），映射 $\Phi(f)(S) = f(\mathbb{1}_S)$。

**子空间层级**：$\mathcal{V}_{\mathcal{B}_d}(k) = \text{span}\{\sigma_M \mid M \subseteq [d], |M| \leq k\}$，其中 $\sigma_M(x) = \max_{i \in M} x_i$。有严格包含链：

$$\mathcal{V}_{\mathcal{B}_d}(0) \subsetneq \mathcal{V}_{\mathcal{B}_d}(1) \subsetneq \cdots \subsetneq \mathcal{V}_{\mathcal{B}_d}(d) = \mathcal{V}_{\mathcal{B}_d}$$

### 主定理：双对数下界

**定理 4.8**：对 $\ell$ 层 rank-2 maxout 网络，$\mathcal{M}_{\mathcal{B}_d}^2(\ell) \subseteq \mathcal{V}_{\mathcal{B}_d}(2^{2^\ell - 1})$

**推论 4.9**：$\max\{0, x_1, \ldots, x_{2^{2^\ell-1}}\}$ 不能被 $\ell$ 层 $\mathcal{B}_d^0$-conforming ReLU 网络表示。

即深度下界为 $\Omega(\log \log d)$。

### 证明核心思路

定义操作 $\mathcal{A}(U)$：对子空间 $U$ 中的函数施加一层 rank-2 maxout 后得到的新子空间。

**关键递推**（命题 4.7）：$\mathcal{A}(\mathcal{F}_{\mathcal{L}}(k)) \subseteq \mathcal{F}_{\mathcal{L}}(k^2 + k)$

即一层 maxout 最多将子空间级别从 $k$ 提升到 $k^2 + k$。迭代后：
- 1 层：$k=2$
- 2 层：$k=6$
- 3 层：$k=42$
- $\ell$ 层：$k = 2^{2^\ell - 1}$（双指数增长）

因此表示 $\sigma_{[d]}$（即 $d$ 个数的 max）需要 $d \leq 2^{2^\ell - 1}$，即 $\ell \geq \Omega(\log \log d)$。

证明技术要点：
1. 利用同构 $\Phi$ 转化为集函数上的分析
2. **Conforming 条件**转化为：$F \in \mathcal{C}_{\mathcal{L}}$（$F(S)$ 和 $F(T)$ 对 $S \subseteq T$ 不能反号）
3. **引理 4.5**："推低支撑"——正负支撑可被推到低层级
4. 对布尔格的子格进行分解和归纳

### $d=4$ 的组合证明

**定理 5.2**：$\mathcal{M}_{\mathcal{B}_d}^2(2) = \mathcal{V}_{\mathcal{B}_d}(4)$

即两层 $\mathcal{B}_d^0$-conforming ReLU 网络恰好能表示 $\max\{0,x_1,\ldots,x_4\}$。此前仅有基于 MIP 的计算验证——本文首次给出纯组合证明。

### Maxout 的超预期能力

**定理 6.2**：$\max\{0, x_1, \ldots, x_6\}$ 可由 rank-(3,2) maxout 网络计算。

这说明标准上界 $\prod r_i = 3 \times 2 = 6$ 不紧—— rank-(3,2) 能表示 7 个数的 max（多一个），打破了直觉。

构造方法：找到两个函数 $f_1, f_2 \in \mathcal{V}_{\mathcal{B}_7}(3)$，使得 $\max\{f_1, f_2\} \in \mathcal{V}_{\mathcal{B}_7}(7) \setminus \mathcal{V}_{\mathcal{B}_7}(6)$。

## 实验关键数据

本文是纯理论工作，无数值实验。核心结果总结：

| 结果 | 内容 |
|------|------|
| 深度下界 | $\Omega(\log \log d)$（$\mathcal{B}_d^0$-conforming） |
| 此前最佳 | 常数 2 |
| 上界 | $\lceil \log_3(d-1) \rceil + 1$（Bakaev et al.） |
| $d=4$ 精确 | 2 层 $\Leftrightarrow$ $\mathcal{V}_{\mathcal{B}_d}(4)$ |
| Maxout 超预期 | rank-(3,2) 可算 7 个数 max（标准上界仅 6） |

### 与已有下界的对比

| 方法 | 限制条件 | 下界 |
|------|---------|------|
| Averkov et al. (2025) | N-ary 分数权重 | $\Omega(\log d / \log \log N)$ |
| Haase et al. (2023) | 整数权重 | $\lceil \log_2(d+1) \rceil$ |
| Bakaev et al. (2025a) | 非负权重 | 专门下界 |
| **本文** | **断点在 braid 排列上** | $\Omega(\log \log d)$ |

本文是**唯一不限制权重**的非常数下界。

## 亮点与洞察
- **双对数下界虽然增长慢，但打破了常数下界的僵局**——是该方向数年来首个实质进展
- 证明方法优雅：将几何问题（多面体扇上的 CPWL 函数）完全转化为组合问题（布尔格上的集函数）
- $d=4$ 的组合证明替代了此前 Hertrich et al. (2023) 的 MIP 计算验证——提供了更深层的理解
- **Maxout 的超预期发现**有独立意义：说明高阶激活函数的表达力用简单乘积公式无法刻画
- 方法可潜在推广到其他扇结构，为理解一般网络深度下界提供工具

## 局限与展望
- $\mathcal{B}_d^0$-conforming 是**真正的限制**：Bakaev et al. (2025b) 已证明一般（非 conforming）2 层网络可计算 5 个数 max
- $\log \log d$ 增长极慢（$d=2^{2^{10}-1}$ 才需要 10 层），离期望的 $\Theta(\log d)$ 上界差距大
- 纯理论工作，对实际架构设计的启示有限
- 证明技术深度依赖 braid 排列结构，推广到一般 breakpoint 配置不直接

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个无权重限制的非常数深度下界，学术意义重大
- 实验充分度: ⭐⭐⭐ 纯理论工作，无数值实验（但有构造性证明替代）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示对理解布尔格分解很有帮助
- 价值: ⭐⭐⭐⭐ 推进了深度学习理论中深度 vs 表达力的核心问题

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)
- [\[NeurIPS 2025\] SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)
- [\[NeurIPS 2025\] SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks](space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)
- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](../../ICML2025/others/permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](kernel_conditional_tests_from_learning-theoretic_bounds.md)

<!-- RELATED:END -->
