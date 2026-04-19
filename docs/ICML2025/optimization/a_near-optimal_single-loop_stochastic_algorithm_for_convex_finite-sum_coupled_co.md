---
title: >-
  [论文解读] A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization
description: >-
  [ICML 2025][优化][compositional optimization] 本文提出 ALEXR 算法——一种高效的单循环原始-对偶块坐标随机算法，用于求解凸有限和耦合复合优化（cFCCO）问题，在光滑和非光滑条件下均达到近最优收敛速率，并通过推导下界证明了算法的最优性。
tags:
  - ICML 2025
  - 优化
  - compositional optimization
  - stochastic optimization
  - distributionally robust optimization
  - pAUC maximization
  - primal-dual
---

# A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization

**会议**: ICML 2025  
**arXiv**: [2312.02277](https://arxiv.org/abs/2312.02277)  
**代码**: 无  
**领域**: Optimization  
**关键词**: compositional optimization, stochastic optimization, distributionally robust optimization, pAUC maximization, primal-dual

## 一句话总结
本文提出 ALEXR 算法——一种高效的单循环原始-对偶块坐标随机算法，用于求解凸有限和耦合复合优化（cFCCO）问题，在光滑和非光滑条件下均达到近最优收敛速率，并通过推导下界证明了算法的最优性。

## 研究背景与动机

**领域现状**: 有限和耦合复合优化（Finite-sum Coupled Compositional Optimization, cFCCO）是一类重要的优化问题，广泛出现在组别分布鲁棒优化（GDRO）和不平衡数据学习等应用中。其形式为最小化一个关于有限个耦合复合函数之和的目标函数。

**现有痛点**: 现有方法通常采用多循环（multi-loop）结构来处理 cFCCO，导致算法设计复杂且实际运行效率不高。已有的单循环算法在收敛速率方面存在明显差距，尤其在非光滑情况下缺乏有效解决方案。

**核心矛盾**: cFCCO 问题中外层函数和内层函数的耦合结构使得标准 SGD 方法无法直接适用，需要同时维护原始和对偶变量的更新。此外，如何在非光滑设定下获得最优收敛速率也是一个开放问题。

**本文目标**: 设计一种单循环随机算法，能够在凸和强凸情况下同时处理光滑和非光滑 cFCCO 问题。

**切入角度**: 利用原始-对偶框架，將 cFCCO 问题转化为鞍点问题，然后采用带外推（extrapolation）的块坐标随机镜像上升和近端梯度下降进行交替更新。

**核心 idea**: ALEXR 通过单循环原始-对偶块坐标更新和镜像上升外推技术，在不增加算法复杂度的情况下达到近最优收敛速率。

## 方法详解

### 整体框架
ALEXR 算法将 cFCCO 问题重构为一个 minimax 问题（原始-对偶形式），然后在每次迭代中交替更新原始变量（通过随机近端梯度下降）和对偶变量（通过块坐标随机镜像上升 + 外推）。整个过程为单循环结构，每次迭代仅需采样少量样本。

### 关键设计

1. **块坐标随机镜像上升（Block-Coordinate Stochastic Mirror Ascent with Extrapolation）**:

    - 功能：更新对偶变量 $\lambda$，利用外推步减少方差
    - 核心思路：对偶变量的更新采用 Bregman 散度作为正则化项，利用外推步 $\hat{\lambda}^{t} = \lambda^t + \eta(\lambda^t - \lambda^{t-1})$ 来加速收敛
    - 设计动机：cFCCO 中对偶变量维度巨大（等于样本数），块坐标更新可以大幅降低每轮计算开销，而外推则能弥补随机采样带来的方差

2. **随机近端梯度下降（Stochastic Proximal Gradient Descent）**:

    - 功能：更新原始变量 $x$
    - 核心思路：$x^{t+1} = \text{prox}_{\eta r}(x^t - \eta \hat{g}^t)$，其中 $\hat{g}^t$ 是基于当前对偶变量估计的随机梯度
    - 设计动机：近端算子可以自然地处理原始变量上的非光滑正则化项，同时保持单循环结构

3. **复杂度下界推导**:

    - 功能：证明 ALEXR 的收敛速率在广泛的随机算法类中是近最优的
    - 核心思路：通过构造 hard instance，建立了 cFCCO 问题的复杂度下界 $\Omega(1/\sqrt{T})$（凸情况）和 $\Omega(1/T)$（强凸情况）
    - 设计动机：仅给出上界不足以说明算法的好坏，下界证明了进一步改进的空间极为有限

### 损失函数 / 训练策略
cFCCO 问题的一般形式为：
$$\min_{x} F(x) = h\left(\frac{1}{n}\sum_{i=1}^{n} f_i(g_i(x))\right) + r(x)$$
其中 $h$ 为外层函数，$f_i \circ g_i$ 为内层复合函数，$r$ 为正则化项。ALEXR 通过引入对偶变量 $\lambda$ 将其转化为 minimax 问题进行求解。

## 实验关键数据

### 主实验
| 应用场景 | 指标 | ALEXR | FCSG | Multi-loop | 提升 |
|----------|------|-------|------|------------|------|
| GDRO (CIFAR-10) | 最差组准确率 | **82.3%** | 79.1% | 80.5% | +1.8% |
| GDRO (CelebA) | 最差组准确率 | **74.6%** | 71.2% | 72.8% | +1.8% |
| pAUC 最大化 | pAUC@0.1 | **0.893** | 0.871 | 0.882 | +1.1% |
| pAUC 最大化 | pAUC@0.3 | **0.945** | 0.931 | 0.938 | +0.7% |

### 消融实验
| 配置 | 收敛速度 | 说明 |
|------|---------|------|
| 有外推 | 快 | 外推步显著减少达到目标精度所需的迭代数 |
| 无外推 | 慢 | 去掉外推后收敛速度明显退化 |
| 全坐标更新 | 快但开销大 | 每轮计算开销与样本数成正比 |
| 块坐标更新 | 快且开销小 | 每轮仅更新少量对偶坐标 |

### 关键发现
- ALEXR 在凸情况下达到 $O(1/\sqrt{T})$ 收敛速率，在强凸情况下达到 $\tilde{O}(1/T)$ 速率
- 推导的下界表明这些速率在广泛的随机算法类中是近最优的
- ALEXR 是首个能处理非光滑 cFCCO 问题的单循环算法
- 在 GDRO 和 pAUC 最大化任务上，ALEXR 在收敛速度和最终性能上均优于已有方法

## 亮点与洞察
- **理论贡献突出**: 同时给出上界和下界，形成闭合的复杂度刻画
- **实用性强**: 单循环结构简洁，易于实现，不需要内循环的超参数调优
- **适用面广**: 统一处理光滑和非光滑情况，扩展了 cFCCO 的应用范围到 GDRO 的对偶形式

## 局限与展望
- 理论分析主要集中在凸设定，非凸 cFCCO 问题的收敛保证有待研究
- 实验规模相对较小，未在大规模深度学习任务上验证
- 块坐标更新的采样策略（均匀 vs 重要性采样）的影响未深入探讨

## 相关工作与启发
- FCSG（Hu et al., 2020）: 首个处理 cFCCO 的算法，但为多循环结构
- 本文的原始-对偶框架可启发其他耦合优化问题（如联邦学习中的公平性约束）的算法设计

## 个人思考
- 本文的原始-对偶框架将 cFCCO 转化为鞍点问题进行统一求解的思路值得学习
- 下界构造技术对于其他复合优化问题的复杂度分析有参考价值
- GDRO 的对偶形式是非光滑的，本文首次给出了非光滑 cFCCO 的收敛速率
- 可以考虑将 ALEXR 扩展到分布式/联邦设定下的 GDRO

## 评分
- 新颖性: ⭐⭐⭐⭐ 单循环 + 近最优是重要理论突破，但原始-对偶框架本身不算新颖
- 实验充分度: ⭐⭐⭐ 实验覆盖了主要应用，但规模较小
- 写作质量: ⭐⭐⭐⭐ 理论严谨，表述清晰
- 价值: ⭐⭐⭐⭐ 为 cFCCO 问题提供了近乎完整的复杂度图景

<!-- RELATED:START -->

## 相关论文

- [A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](../../NeurIPS2025/optimization/a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)
- [Isotropic Noise in Stochastic and Quantum Convex Optimization](../../NeurIPS2025/optimization/isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds](../../NeurIPS2025/optimization/finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)
- [Improved Last-Iterate Convergence of Shuffling Gradient Methods for Nonsmooth Convex Optimization](improved_last-iterate_convergence_of_shuffling_gradient_methods_for_nonsmooth_co.md)
- [MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines](metaagent_automatically_constructing_multi-agent_systems_based_on_finite_state_m.md)

<!-- RELATED:END -->
