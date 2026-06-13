---
title: >-
  [论文解读] Robust Estimation Under Heterogeneous Corruption Rates
description: >-
  [NeurIPS 2025][优化/理论][鲁棒估计] 本文研究了异质污染率下的鲁棒估计问题——每个样本以不同的已知概率被污染——对有界分布和高斯分布的均值估计及线性回归建立了紧的极小极大率，发现最优估计器可以简单地丢弃污染率超过某阈值的样本。 传统鲁棒统计学假设所有数据以相同的概率 $\epsilon$ 被污染（Huber…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "鲁棒估计"
  - "异质污染"
  - "极小极大率"
  - "均值估计"
  - "线性回归"
---

# Robust Estimation Under Heterogeneous Corruption Rates

**会议**: NeurIPS 2025  
**arXiv**: [2508.15051](https://arxiv.org/abs/2508.15051)  
**代码**: 暂无  
**领域**: 优化  
**关键词**: 鲁棒估计, 异质污染, 极小极大率, 均值估计, 线性回归

## 一句话总结

本文研究了异质污染率下的鲁棒估计问题——每个样本以不同的已知概率被污染——对有界分布和高斯分布的均值估计及线性回归建立了紧的极小极大率，发现最优估计器可以简单地丢弃污染率超过某阈值的样本。

## 研究背景与动机

传统鲁棒统计学假设所有数据以相同的概率 $\epsilon$ 被污染（Huber contamination model），这在理论上已经被很好地理解。然而在实际场景中——如联邦学习、众包标注、传感器网络——数据来自异质来源，不同样本的可靠性差异很大。例如大规模生物医学数据集由不同机构在不同时间收集，各部分的异常值比例自然不同。

核心矛盾在于：已有的鲁棒统计技术全部假设均匀的噪声先验，**不清楚是否能在异质污染下获得最优统计率**，甚至不清楚这些最优率是什么。

在先前的工作中，鲁棒估计的极小极大率已在均匀污染下被很好地理解（如高斯均值估计的率为 $d/n + \epsilon^2$），但这些结果无法直接推广到异质情形——因为简单地设置 $\epsilon = \max_i \lambda_i$ 会导致高度保守的估计。

作者提出了一种自然的推广——$\boldsymbol{\lambda}$-contamination模型（Definition 1）：第 $i$ 个样本以概率 $\lambda_i$ 被污染，各 $\lambda_i$ 已知且可能不同。核心技术挑战是**如何有效"混合"不同噪声水平的数据点的信息**。一个极端例子是半验证学习(semi-verified learning)：少量可信数据($\lambda_i=0$) + 大量高噪声数据($\lambda_i \approx 1$)，单独看任何一组都无法得到好的估计，但两组结合可以。

## 方法详解

### 整体框架

作者对四个基本估计问题建立了极小极大率的上下界：有界分布均值估计、一维高斯均值估计、多维高斯均值估计、高斯设计线性回归。核心发现是一个关键的"有效污染率"函数 $f(\boldsymbol{\lambda}, k)$ 贯穿所有结果。

### 关键设计

1. **$\boldsymbol{\lambda}$-contamination模型 (Definition 1)**: 形式化定义异质污染。样本 $Z_i = (1-B_i)X_i + B_i\tilde{X}_i$，其中 $B_i \sim \text{Bern}(\lambda_i)$，干净样本 $X_i \overset{iid}{\sim} P$，异常值 $\tilde{X}_i$ 可以是自适应对手选择的。关键假设是 $\boldsymbol{\lambda}$ 已知（或已知近似上界即可）。

2. **有效污染率函数**: 定义 $f(\boldsymbol{\lambda}, k) = \min_{t \in [0,1]} \left(\frac{k}{|\{i: \lambda_i \leq t\}|} + t^2\right)$。该函数通过在阈值 $t$ 上取最优平衡，同时考虑了保留样本数量（统计精度项 $k/N(t)$）和最大允许污染率（偏差项 $t^2$）。这个函数是本文所有结果的核心。

3. **阈值估计器 (上界)**: 对于有界分布，最简单的方法是——选择阈值 $t$，只使用污染率 $\lambda_i \leq t$ 的样本计算均值。最优阈值 $t^* = \arg\min_{t} (1/N(t) + t^2)$ 给出率 $r^2 f(\boldsymbol{\lambda}, 1)$。更精细地，通过加权估计器 $\sum w_i Z_i$ 并最小化 $\|w\|_2^2 + 3(w^T\lambda)^2$，可以得到相同阶的上界但实际表现可能更好。Algorithm 1 给出了近线性时间的精确最优权重算法。

4. **Le Cam两点法下界**: 构造两个假设 $P_0, P_1$（支持在 $\pm re_1$ 上的Bernoulli分布），对手策略是：对于 $\lambda_i \geq 2\delta/(1+2\delta)$ 的样本，对手可以使其分布在两个假设下完全一样（"模拟"），从而使这些样本完全无用。只有 $\lambda_i < 2\delta/(1+2\delta)$ 的样本对假设检验有帮助。通过选择合适的 $\delta$ 匹配上界。

5. **加权Tukey中位数 (高斯设定)**: 定义加权深度 $D_w(\eta, \boldsymbol{Z}) = \min_{v \in \mathbb{S}_d} \sum_i w_i \mathbb{I}\{v^T(Z_i - \eta) \geq 0\}$，通过降低高污染样本的权重来提高鲁棒性。对于高斯均值估计，上界为 $(w^T\lambda)^2 + d\|w\|_2^2$；同样可以使用阈值方法。

6. **加权回归深度 (线性回归)**: 类似地定义加权回归深度，用残差符号在各方向上的均匀性衡量估计质量，建立了相同形式的上下界。

### 上下界匹配证明

有界分布的证明最为干净：找到一个关键阈值 $\delta^*$ 使得 $N(2\delta^*) \geq \frac{1}{12\delta^{*2}}$ 而 $n(2\delta^*) \leq \frac{1}{12\delta^{*2}}$，然后将其代入上下界可得两者都是 $\Theta(r^2\delta^{*2})$。

## 实验关键数据

### 有界分布实验

| 方法 | 适用性 | 利用异质性 | MSE表现 |
|---|---|---|---|
| 均匀鲁棒估计 (baseline) | 使用最大 $\lambda$ | 否 | 最差 |
| 阈值法 (本文) | 选择最优阈值 | 部分 | 接近最优 |
| 加权法 (本文, Algorithm 1) | 逐样本加权 | 完全 | 最优 |

### 高斯均值估计对比

| 方法 | 估计误差 | 说明 |
|---|---|---|
| 均匀权重Tukey中位数 | 高 | 高污染样本拉偏深度估计 (Fig. 1A) |
| 阈值加权Tukey中位数 | 中等 | 丢弃高污染样本 (Fig. 1B) |
| 最优加权Tukey中位数 | 最低 | 平滑利用所有可用信息 (Fig. 1C) |

Figure 1直观展示了三种权重方案下Tukey深度的等高线图：最优加权方案的深度最大点（黄色星号）最接近真实均值。

### 关键发现

- 有界分布和一维高斯的极小极大率被完全确定，上下界紧致匹配
- 多维高斯和线性回归存在 $\sqrt{d}$ 的间隙（上界 $f(\boldsymbol{\lambda}, d)$，下界 $f(\boldsymbol{\lambda}, d)/\sqrt{d}$），在常数维度下紧致
- 超过最优阈值 $t^*$ 的样本可以直接丢弃，不会损失常数因子内的最优性
- 逐样本加权方法在实际中可能优于简单阈值方法，且保持理论最优性

## 亮点与洞察

- $f(\boldsymbol{\lambda}, k)$ 函数优雅地统一了所有结果，它本质上是寻找"统计精度"和"污染偏差"之间的最优平衡点
- 阈值估计器的发现极具实践价值：异质场景下不需要复杂的算法，只要选对阈值然后用标准鲁棒估计器即可
- Le Cam下界中对手策略的构造很精巧——对手可以"模拟"目标分布使得高污染样本完全无信息量
- 将异质性从鲁棒统计学的"刻意忽略"变成了"可利用的结构"

## 局限与展望

- 多维高斯和线性回归的 $\sqrt{d}$ 间隙尚未解决；作者认为这可能需要全新的复杂度度量来精确刻画异质性的贡献
- $\boldsymbol{\lambda}$ 需要已知（虽然常数因子近似即可），实际中可能需要从历史数据估计污染率的上界
- 半验证学习设定下，阈值方法的上界随 $d$ 发散，而已有工作已证明可以获得维度无关的率——说明阈值方法并非所有设定下都最优
- 仅考虑了Huber型（加法）污染模型，其他模型（如total variation, KL divergence）值得探索

## 相关工作与启发

- 与Charikar et al. (2017) 的半验证学习直接相关：本文给出了半验证学习的一个自然推广，且在半验证学习的特殊情形下可以恢复已有结果
- 与Diakonikolas & Kane (2023) 的鲁棒统计教材中的技术形成对比：标准方法假设均匀污染，本文需要新的下界技术来联合捕获维度和异质性
- 鲁棒性与隐私的深层联系（Georgiev & Hopkins, 2022）在异质设定下如何推广是有趣的开放问题
- Tukey深度/回归深度的加权版本是本文的技术核心，将经典统计量推广到异质权重设定
- 隐私文献中的异质隐私需求工作（Cummings et al., 2023）仅限于一维有界情形，本文在多维设定下取得了实质性进展

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统建立异质污染率下的极小极大率理论
- 实验充分度: ⭐⭐⭐ 以理论贡献为主，实验为补充性的合成验证
- 写作质量: ⭐⭐⭐⭐⭐ 问题形式化清晰，结果陈述干炼，关键概念(如 $f$ 函数)贯穿全文
- 价值: ⭐⭐⭐⭐ 填补了鲁棒统计学中异质污染设定的理论空白，对联邦学习、众包标注等实际场景有重要的理论指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Near-Exponential Savings for Mean Estimation with Active Learning](near-exponential_savings_for_mean_estimation_with_active_learning.md)
- [\[ICML 2026\] Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption](../../ICML2026/optimization/convex_basins_in_single-index_model_loss_landscapes_applications_to_robust_recov.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[CVPR 2025\] Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World](../../CVPR2025/optimization/convex_relaxation_for_robust_vanishing_point_estimation_in_manhattan_world.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)

</div>

<!-- RELATED:END -->
