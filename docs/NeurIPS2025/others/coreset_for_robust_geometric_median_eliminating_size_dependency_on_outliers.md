---
title: >-
  [论文解读] Coreset for Robust Geometric Median: Eliminating Size Dependency on Outliers
description: >-
  [NeurIPS 2025][Coreset] 首次消除鲁棒几何中位数 coreset 大小对异常值数 $m$ 的依赖：在 $n \geq 4m$ 条件下，$d=1$ 时实现最优 coreset 大小 $\tilde{\Theta}(\varepsilon^{-1/2} + \frac{m}{n}\varep…
tags:
  - "NeurIPS 2025"
  - "Coreset"
  - "鲁棒几何中位数"
  - "异常值"
  - "聚类"
  - "非逐分量误差分析"
---

# Coreset for Robust Geometric Median: Eliminating Size Dependency on Outliers

**会议**: NeurIPS 2025  
**arXiv**: [2510.24621](https://arxiv.org/abs/2510.24621)  
**领域**: 其他  
**关键词**: Coreset, 鲁棒几何中位数, 异常值, 聚类, 非逐分量误差分析

## 一句话总结
首次消除鲁棒几何中位数 coreset 大小对异常值数 $m$ 的依赖：在 $n \geq 4m$ 条件下，$d=1$ 时实现最优 coreset 大小 $\tilde{\Theta}(\varepsilon^{-1/2} + \frac{m}{n}\varepsilon^{-1})$，高维时实现 $\tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$，核心技术是新颖的**非逐分量误差分析**。

## 研究背景与动机

**领域现状**: 几何中位数（Fermat-Weber 问题）在计算几何中的核心地位不言而喻。鲁棒版本允许去除 $m$ 个异常值以抵抗噪声/对抗干扰，目标函数为 $\text{cost}^{(m)}(P, c) = \min_{|L|=m} \sum_{p \in P \setminus L} \|p - c\|$。

**现有痛点**: 所有已知 coreset 大小都包含 $O(m)$ 项——因为现有方法将异常点全部保留在 coreset 中。当 $m = \Theta(n)$（如 PageBlocks 数据集 $m \approx 0.1n$）时，coreset 失去压缩意义。

**核心问题**: 能否消除 coreset 大小中的 $O(m)$？先验结论似乎是否定的（已有 $\Omega(m)$ 下界），但该下界依赖 $m = n-1$ 的极端情形。

**本文切入角度**: 证明 $n - m = \Omega(n)$ 是消除 $O(m)$ 的充要条件，并在此条件下提出新算法和全新的非逐分量误差分析。

## 方法详解

### 下界：$n - m = \Omega(n)$ 是必要条件
**Theorem 1.1**: 存在一维数据集使得任何 $\varepsilon$-coreset 大小为 $\Omega(m/(n-m))$。当 $n - m = o(n)$ 时此界为 $\omega(1)$，因此消除 $O(m)$ 需要 $n - m = \Omega(n)$。

### 一维情形：最优 coreset (Theorem 1.2)
**核心子集 $P_M$**: 令 $P_M = \{p_{m+1}, \ldots, p_{n-m}\}$（最中间的 $n - 2m$ 个点，在 $n \geq 4m$ 时 $|P_M| \geq 2m$）。$P_M$ 中所有点对任意中心 $c$ 必为内点（inlier），因此 $\text{cost}^{(m)}(P, c) \geq \text{cost}(P_M, c)$。

**算法三阶段**:
1. **Stage 1**: 对 $P_M$ 应用 vanilla 1D geometric median 的最优 coreset 构造 [37]，产生 $\tilde{O}(\varepsilon^{-1/2})$ 个 bucket
2. **Stage 2**: 对 $P_L \cup P_R$（外围 $2m$ 个点）按距离 $c_L, c_R$ 的指数级递增区间进行分块（inner blocks + outer blocks）
3. **Stage 3**: 额外分块确保 $\text{cost}^{(m)}(P, p_{n-m}) = \text{cost}^{(m)}(S, p_{n-m})$

**非逐分量误差分析（核心技术）**: 传统方法分析每个 bucket 的误差并求和（逐分量分析），要求每个 bucket 内异常值数对齐，导致需保留所有外围点。本文**不要求**逐 bucket 误差控制，而是分析整体误差：

$$|\text{cost}^{(m)}(P, c) - \text{cost}^{(m)}(S, c)| \leq |\text{cost}^{(m)}(P, p_{n-m}) - \text{cost}^{(m)}(S, p_{n-m})| + \int_{p_{n-m}}^{c} |f'_P(x) - f'_S(x)| dx$$

**关键几何观察**: $f'_P(c)$（代价函数的导数）等于中心 $c$ 左边的内点数减右边的内点数。由此 $|f'_P(c) - f'_S(c)| \leq \sum_i |m_i - m_i'| + 2|B_c|$（$m_i, m_i'$ 为各 bucket 中 $P$ 和 $S$ 的异常值数，$B_c$ 为包含 $c$ 的 bucket）。

只要限制每个 bucket 大小为 $O(\varepsilon n)$，即可保证 $\sum_i |m_i - m_i'| \leq O(\varepsilon n)$，从而整体误差受控——**单个 bucket 可能有任意大的误差，但各 bucket 误差会互相抵消**。

**Coreset 大小分析**: $P_M$ 贡献 $\tilde{O}(\varepsilon^{-1/2})$ 个 bucket，外围点 $P_L \cup P_R$ 贡献 $O(\frac{m}{n}\varepsilon^{-1})$ 个 bucket，合计 $\tilde{O}(\varepsilon^{-1/2} + \frac{m}{n}\varepsilon^{-1})$。

**下界匹配**: 构造 $m$ 个异常值分布在 $\frac{m}{n}\varepsilon^{-1}$ 个指数递增区间中的最坏情形，若 coreset 漏掉任一区间，误差达 $2\varepsilon \cdot \text{cost}^{(m)}(P, c)$。

### 高维情形 (Theorem 1.3)
从外围 $m$ 个点中**均匀采样** $\tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$ 个点（而非全部保留）。利用 ball range space 的 VC 维 $O(d)$：采样集 $S_O$ 以高概率为 ball range space 上的 $\varepsilon$-approximation，保证 outlier-misaligned 点数仅 $O(\varepsilon m)$。

### 推广到 robust $(k,z)$-clustering (Theorem 1.5)
在 Assumption 1.4（每个内点簇大小 $\geq 4m$，无极端远点）下，coreset 大小 $\tilde{O}(k^2\varepsilon^{-2z}\min\{\varepsilon^{-2}, d\})$，消除 $O(m)$ 项。

## 实验关键数据

### 六个真实数据集上的 size-error tradeoff（robust geometric median）

| 数据集 | 方法 | Coreset 大小 | 经验误差↓ |
|-------|------|------------|----------|
| Census1990 | 本文算法 | 1000 | **0.012** |
| Census1990 | 基线 [39,40] | 2300 | 0.013 |
| PageBlocks ($m \approx 0.1n$) | 本文算法 | 500 | **低于基线** |
| PageBlocks ($m \approx 0.1n$) | 基线 [39,40] | 500 | 较高 |

### 运行时间对比
在相同经验误差下，本文算法的运行时间显著低于基线（Table 8），因为 coreset 更小。

### 数据假设的鲁棒性
Assumption 1.4 的条件 2（$\max_p \text{dist}(p, C^*)^z \leq 4k \cdot \text{avg}$）在六个真实数据集中均满足（Table 2），且即使假设被违反，算法在实践中仍表现良好。

## 亮点与洞察
- **非逐分量误差分析**是 coreset 文献中首创：允许单个 bucket 误差任意大（只要互相抵消），打破了此前所有方法的 $\Omega(m)$ 瓶颈
- **$n \geq 4m$ 的充要性**优雅地刻画了问题结构——消除 $O(m)$ 与否完全取决于内点占比
- 一维情形的**精确最优 coreset 大小** $\tilde{\Theta}(\varepsilon^{-1/2} + \frac{m}{n}\varepsilon^{-1})$ 清晰展示了 robust 与 vanilla 情形的分离点：当 $m > \sqrt{\varepsilon} n$ 时开始偏离
- Range space 论证首次应用于异常值点——将 $\varepsilon$-approximation 的工具从内点推广到外围点
- 算法实现简单高效：一维算法 $O(n)$ 时间，高维算法 $O(nd)$ 时间，均为线性复杂度

## 相关工作与对比

| 方法 | Coreset 大小 ($d > 1$, $k=1$) | 需要 $n \geq 4m$？ |
|------|-------|------|
| [39] Huang et al. | $O(m) + \tilde{O}(\varepsilon^{-3}\min\{\varepsilon^{-2}, d\})$ | 否 |
| [40] Huang et al. | $O(m) + \tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$ | 否 |
| [42] Jiang et al. | $O(m\varepsilon^{-1}) + \text{Vanilla size}$ | 否 |
| **本文 (Thm 1.3)** | $\tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$ | 是 |

## 局限与展望
- 高维情形 $\tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$ 与一维最优 $\tilde{O}(\varepsilon^{-1/2} + \frac{m}{n}\varepsilon^{-1})$ 之间有大的 gap，高维最优 coreset 大小仍开放
- Assumption 1.4 的条件 1（$\min_i |P_i^*| \geq 4m$）在不同簇大小悬殊时可能不满足
- 仅考虑欧氏空间和 $\ell_2$ 距离，更一般度量空间的推广虽提及但未详细分析
- 实验仅覆盖 robust geometric median 和 k-median，k-means ($z=2$) 的实验未展示
- 非逐分量分析的想法很强，但目前仅在一维情形充分发挥——高维版本仍回退到 range space 论证

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 非逐分量误差分析是全新技术工具，有独立研究价值
- 理论深度: ⭐⭐⭐⭐⭐ 上下界匹配 + 充要条件 + 多层推广
- 实验充分度: ⭐⭐⭐ 六个数据集验证，但限于 $z=1$ 和基本比较
- 写作质量: ⭐⭐⭐⭐ 技术概述清晰，图示有效，附录完整
- 综合: ⭐⭐⭐⭐⭐ 理论贡献突出，解决了 coreset 领域的重要开放问题

## 相关工作与启发
- **vs Huang et al. [39,40]**: 分层采样的逐分量误差分析天然需要每个 bucket 对齐异常值数，导致 $O(m)$ 不可避免。本文的非逐分量分析从根本上打破瓶颈
- **vs Jiang et al. [42]**: 通过 robust 到 vanilla 的规约得到 $O(m\varepsilon^{-1}) + \text{Vanilla size}$，仍含 $m$ 项。本文完全消除
- **vs Feldman & Langberg [27]**: 早期构造要求指数级大小 $(k+m)^{O(k+m)}$，实际不可用
- 非逐分量误差分析可推广到其他"部分数据可删除"的压缩问题
- Ball range space 论证首次用于异常值点，为 robust clustering coreset 提供新技术工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Distributionally Robust Feature Selection](distributionally_robust_feature_selection.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)
- [\[ACL 2025\] Hierarchical Bracketing Encodings for Dependency Parsing as Tagging](../../ACL2025/others/hierarchical_bracketing_dep_parsing.md)
- [\[NeurIPS 2025\] Overfitting in Adaptive Robust Optimization](overfitting_in_adaptive_robust_optimization.md)
- [\[ICML 2025\] Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](../../ICML2025/others/lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)

</div>

<!-- RELATED:END -->
