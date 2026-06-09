---
title: >-
  [论文解读] On the Impact of the Utility in Semivalue-based Data Valuation
description: >-
  [ICLR 2026][Data Valuation] 本文通过引入"空间签名"（spatial signature）的几何表示，将数据估值中的 utility 选择问题统一建模为单位圆上的方向旋转问题，并提出了一个量化鲁棒性的指标 $R_p$…
tags:
  - "ICLR 2026"
  - "Data Valuation"
  - "Semivalue"
  - "Shapley Value"
  - "Banzhaf Value"
  - "Robustness"
---

# On the Impact of the Utility in Semivalue-based Data Valuation

**会议**: ICLR 2026  
**arXiv**: [2502.06574](https://arxiv.org/abs/2502.06574)  
**代码**: [https://github.com/taminemelissa/utility-impact](https://github.com/taminemelissa/utility-impact)  
**领域**: 数据估值 / AI理论  
**关键词**: Data Valuation, Semivalue, Shapley Value, Banzhaf Value, Robustness

## 一句话总结

本文通过引入"空间签名"（spatial signature）的几何表示，将数据估值中的 utility 选择问题统一建模为单位圆上的方向旋转问题，并提出了一个量化鲁棒性的指标 $R_p$，揭示了 Banzhaf 值在不同 utility 下表现出最高的排序稳定性。

## 研究背景与动机

**领域现状**：基于 semivalue 的数据估值是当前数据质量评估的主流方法，它利用合作博弈论的解概念（如 Shapley 值、Beta Shapley、Banzhaf 值）为每个数据点分配一个价值分数，衡量其对下游 ML 任务的贡献。这类方法被广泛用于识别高质量训练样本、数据清洗和公平数据定价。

**现有痛点**：semivalue 的计算依赖于用户选择的 utility function（效用函数），而这一选择通常是主观的。例如，训练一个猫狗分类器时，accuracy、precision、recall、F1、AUROC 都是合理的 utility，但不同 utility 可能导致完全不同的数据排序结果。作者在 8 个数据集上的实验显示，在 Titanic 数据集上使用 Shapley 值时，accuracy 和 F1 的排序相关性甚至为 -0.19，说明排序结果极不稳定。

**核心矛盾**：数据估值方法声称能客观评估数据点的重要性，但其结果高度依赖于 utility 的选择——而 utility 的选择本身并没有唯一正确答案。这使得实践者无法判断自己的数据估值结果是否可信。

**本文目标** (1) 如何统一建模"utility 变化对排序影响"的问题？(2) 如何量化这种鲁棒性？(3) 不同 semivalue（Shapley vs Banzhaf）的鲁棒性差异有多大、为什么？

**切入角度**：作者观察到，对于任意 semivalue，所有 utility 下的数据价值分数都可以通过一个低维空间中的线性泛函来表示。这意味着排序变化可以被几何化为单位圆上方向旋转时投影顺序的变化——一个简洁而可分析的问题。

**核心 idea**：将每个数据点嵌入到由 semivalue 权重和基础 utility 决定的二维空间中（spatial signature），使得 utility 变化下的排序稳定性问题转化为几何问题，从而可以精确度量和比较。

## 方法详解

### 整体框架

给定数据集 $\mathcal{D} = \{z_i\}_{i \in [n]}$、一个 semivalue 权重向量 $\omega$ 和两个基础 utility $u_1, u_2$，本文的方法分三步：(1) 将每个数据点 $z_i$ 嵌入到二维空间 $\mathbb{R}^2$ 形成 spatial signature；(2) 分析单位圆 $\mathcal{S}^1$ 上所有方向 $\bar{\alpha}$ 对应的排序如何变化；(3) 计算鲁棒性指标 $R_p$ 衡量排序稳定度。

### 关键设计

**1. 两类场景的统一建模：把所有 utility 选择都收进同一个线性组合。**

实践中 utility 的不确定性其实有两种来源，作者先证明它们可以归到同一种数学形式。一种是 utility trade-off：用户明确想在两个目标间权衡，写成 $u_\nu = \nu u^A + (1-\nu) u^B$，参数 $\nu$ 控制偏向哪个目标。另一种更隐蔽的是 multiple-valid-utility：训练同一个分类器时 accuracy、F1、precision 都"看起来合理"，没人能说哪个唯一正确。关键观察是，这些常见分类指标都能写成 true-positive rate $\lambda$ 与 positive-prediction rate $\gamma$ 的线性分式 $u(S) = \frac{c_0 + c_1\lambda(S) + c_2\gamma(S)}{d_0 + d_1\lambda(S) + d_2\gamma(S)}$，对其做一阶展开后，$u$ 关于 $(\lambda, \gamma)$ 近似为一个仿射函数。于是两种场景都坍缩成 $u_\alpha = \alpha_1 u_1 + \alpha_2 u_2$ 这一个形式——只要分析"系数 $\alpha$ 怎么变会改变数据排序"，就同时覆盖了 trade-off 和 multiple-valid-utility 两类问题，方法的适用范围因此被一次性撑开。

**2. Spatial Signature：把排序稳定性翻译成单位圆上的投影问题。**

有了统一的 $u_\alpha = \alpha_1 u_1 + \alpha_2 u_2$，接下来要回答"$\alpha$ 转动时排序会不会乱"。Proposition 3.1 给出了关键的几何映射：存在 $\psi_{\omega,\mathcal{D}}: \mathcal{D} \to \mathbb{R}^2$，使得任意 utility $u_\alpha$ 下某点的 semivalue 分数恰好是一个内积

$$\phi(z; \omega, u_\alpha) = \langle \psi_{\omega,\mathcal{D}}(z), \alpha \rangle.$$

也就是说，每个数据点被嵌入到二维平面上得到它的 spatial signature，而"用哪个 utility"等价于"朝哪个方向 $\alpha$ 投影"。数据排序就是这些二维点在 $\alpha$ 方向上的投影顺序，于是 utility 变化下排序稳不稳，被精确等价为：当 $\alpha$ 在单位圆上旋转时，投影顺序会不会翻转。这个翻译之所以有用，是因为它把"算每个 utility 下的真实分数"这种昂贵又抽象的事，换成了纯几何直觉——如果所有嵌入点近似共线，那不管朝哪个方向投影，谁在前谁在后几乎都不变，鲁棒性就最高；点摊得越开，旋转一点就越容易翻转排序。

**3. 鲁棒性指标 $R_p$：用"旋转多少角度才会乱序"量化稳定性。**

最后把上面的几何直觉做成一个可计算的数。对每一对数据点 $(z_i, z_j)$，令 $v_{ij} = \psi(z_i) - \psi(z_j)$，它们投影顺序发生翻转的临界方向就是与 $v_{ij}$ 正交的那条"切割角" $H_{ij} = \{\alpha \in \mathcal{S}^1 : \langle \alpha, v_{ij} \rangle = 0\}$。全部 $\binom{n}{2}$ 对一共产生 $2N$ 个切割点，把单位圆切成若干段弧，每段弧内部排序保持不变。定义 $\rho_p(\bar{\alpha}_0)$ 为从起始方向 $\bar{\alpha}_0$ 出发、累计发生 $p$ 次两两交换所需扫过的最小弧长——这个角度越大，说明排序越"扛转"。把它对起始方向取期望再归一化即得

$$R_p = \frac{\mathbb{E}[\rho_p]}{\pi/4},$$

分母 $\pi/4$ 是所有点完全共线时 $\rho_p$ 能取到的最大值，因此 $R_p \in [0,1]$，越接近 1 越鲁棒。它的实用之处在于可在 $O(n^2 \log n)$ 时间内精确算出，并且其高低与 Kendall 排序相关性的退化程度直接对应——$R_p$ 低就意味着换个 utility 排序就会大幅重排。

### 理论分析：为什么 Banzhaf 最稳

本文不涉及神经网络训练，整套方法是一个分析性框架，鲁棒性差异的根源由 Proposition 3.3 直接给出。它把两个基础 utility 下 semivalue 分数向量的 Pearson 相关性拆解为

$$\text{Corr}(\phi(u_1), \phi(u_2)) = \frac{\sum_j \omega_j^2 r_j}{\sqrt{\sum_j \omega_j^2 \text{Var}_j(u_1)} \sqrt{\sum_j \omega_j^2 \text{Var}_j(u_2)}},$$

其中 $\omega_j$ 是 semivalue 在 coalition size $j$ 上的权重，$r_j$ 是 size-$j$ 的对齐因子。对齐因子 $r_j$ 通常在中等 coalition size 区域最高，而 Banzhaf 权重 $\omega_j = \binom{n-1}{j-1}/2^{n-1}$ 恰好集中在这一区域，于是它系统性地把权重压在 $r_j$ 大的地方，得到更高的相关性、更共线的 spatial signature，最终拿到最高的 $R_p$；Shapley 则均匀摊到所有 size、被极端 size 的高方差拖累，因此更不稳。

## 实验关键数据

### 主实验：不同 semivalue 和数据集上的 Kendall 排序相关性

| 数据集 | Shapley | (4,1)-Beta Shapley | Banzhaf |
|--------|---------|-------------------|---------|
| Breast | 0.95 ± 0.003 | 0.95 ± 0.003 | **0.97 ± 0.008** |
| Titanic | -0.19 ± 0.007 | -0.17 ± 0.01 | **0.94 ± 0.003** |
| Credit | -0.47 ± 0.01 | -0.44 ± 0.02 | **0.87 ± 0.01** |
| Heart | 0.64 ± 0.006 | 0.68 ± 0.004 | **0.96 ± 0.003** |
| Wind | 0.81 ± 0.008 | 0.82 ± 0.008 | **0.99 ± 0.002** |
| Cpu | 0.59 ± 0.02 | 0.62 ± 0.02 | **0.86 ± 0.007** |

accuracy 与 F1 两种 utility 下的排序相关性。Banzhaf 在所有数据集上都显著优于 Shapley 和 Beta Shapley。

### 鲁棒性指标 $R_p$ 验证

| 数据集 | 场景 | Shapley $R_p$ | Banzhaf $R_p$ | 一致性 |
|--------|------|-------------|-------------|--------|
| Breast | 多 utility | 高 | 最高 | $R_p$ 与 Kendall 相关一致 |
| Titanic | 多 utility | 极低 | 高 | $R_p$ 准确反映排序不稳定 |
| Diabetes | utility trade-off | 中等 | 最高 | 回归任务同样适用 |
| Digits | utility trade-off | 中等 | 最高 | 多分类任务同样适用 |

### 关键发现

- **Banzhaf 一致性优势的几何解释**：Banzhaf 权重使得 spatial signature 中的点近乎共线，这直接最大化了 $R_p$。原因在于 Banzhaf 权重 $\omega_j = \binom{n-1}{j-1} / 2^{n-1}$ 集中在中等 coalition size，而此区域的 size-specific alignment factor $r_j$ 通常最高
- **$R_p$ 与排序相关性的一致性**：所有实验中 $R_p$ 的高低与 Kendall 相关性严格对应，验证了几何框架的实用价值
- **反直觉发现**：在某些数据集（如 Titanic）上，Shapley 和 Beta Shapley 的排序在不同 utility 下甚至负相关，意味着这些 semivalue 在该场景下作为数据估值工具完全不可靠

## 亮点与洞察

- **几何视角精彩**：将合作博弈论中的抽象排序稳定性问题转化为二维空间中的投影排序问题，直觉清晰且有精确的数学对应关系。这种从代数到几何的桥梁在 ML 理论中非常少见
- **实践指导价值高**：$R_p$ 指标可以告诉实践者"你的数据估值是否可信"——如果 $R_p$ 很低，无论用哪个 utility，排序都不稳定，不应使用 semivalue 方法
- **Banzhaf 优越性的理论解释**：过去文献已观察到 Banzhaf 经验上更稳定，本文首次从权重分布与 alignment factor 的交互角度给出了理论解释

## 局限与展望

- **线性分式近似的适用范围**：multiple-valid-utility 场景的分析基于 utility 对 $(\lambda, \gamma)$ 的一阶线性近似，对 negative log-loss 等非线性分式指标不适用
- **仅限二分类和部分多分类指标**：回归任务的 utility（如 MSE vs MAE）虽然在 trade-off 场景中得到验证，但缺乏类似的统一线性分式推导
- **计算复杂度**：$R_p$ 的精确计算需 $O(n^2 \log n)$，对超大规模数据集可能仍然昂贵
- **未讨论 utility 近似误差的传播**：线性近似引入的误差对 $R_p$ 的影响未量化

## 相关工作与启发

- **vs Data Shapley (Ghorbani & Zou, 2019)**：Data Shapley 均匀加权所有 coalition size，导致受极端 size 的高方差 marginal contribution 影响大，鲁棒性差。本文解释了为何 Banzhaf 优于 Shapley
- **vs Diehl & Wilson (2025)**：该工作同样指出 semivalue 估值在 utility 欠定义时不可靠且可被操控，但只是暴露问题。本文进一步提供了量化脆弱性的工具和选择 semivalue 的指导
- **vs Wang & Jia (2023)**：Data Banzhaf 已经证明了对学习算法随机性的鲁棒性，本文从 utility 维度扩展了鲁棒性分析

## 评分

- 新颖性: ⭐⭐⭐⭐ 几何化数据估值鲁棒性分析是全新视角，但问题设定相对窄
- 实验充分度: ⭐⭐⭐⭐ 涵盖多数据集、多 semivalue、两种场景，实验与理论高度一致
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、逻辑链完整、图示优秀，理论与实验紧密配合
- 价值: ⭐⭐⭐⭐ 对数据估值实践有直接指导意义，但受众主要限于数据估值领域

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Do We Really Need Permutations? Impact of Model Width on Linear Mode Connectivity](do_we_really_need_permutations_impact_of_model_width_on_linear_mode_connectivity.md)
- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](bayesian_influence_functions_for_hessian-free_data_attribution.md)
- [\[ICML 2026\] TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](../../ICML2026/others/tempora_characterising_the_time-contingent_utility_of_online_test-time_adaptatio.md)
- [\[ICLR 2026\] When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency](when_to_retrain_after_drift_a_data-only_test_of_post-drift_data_size_sufficiency.md)
- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](tabstruct_measuring_structural_fidelity_of_tabular_data.md)

</div>

<!-- RELATED:END -->
