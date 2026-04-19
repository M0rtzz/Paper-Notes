---
title: >-
  [论文解读] On the Impact of the Utility in Semivalue-based Data Valuation
description: >-
  [ICLR 2026][Data Valuation] 本文通过引入"空间签名"（spatial signature）的几何表示，将数据估值中的 utility 选择问题统一建模为单位圆上的方向旋转问题，并提出了一个量化鲁棒性的指标 $R_p$，揭示了 Banzhaf 值在不同 utility 下表现出最高的排序稳定性。
tags:
  - ICLR 2026
  - Data Valuation
  - Semivalue
  - Shapley Value
  - Banzhaf Value
  - Robustness
---

# On the Impact of the Utility in Semivalue-based Data Valuation

**会议**: ICLR 2026  
**arXiv**: [2502.06574](https://arxiv.org/abs/2502.06574)  
**代码**: https://github.com/taminemelissa/utility-impact (有)  
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

1. **两类场景的统一建模**:

    - 功能：将 utility trade-off 场景和 multiple-valid-utility 场景统一到同一个几何框架
    - 核心思路：Utility trade-off 场景下，$u_\nu = \nu u^A + (1-\nu) u^B$，参数 $\nu$ 控制两个目标的权衡；Multiple-valid-utility 场景下，常见分类指标（accuracy、F1、precision 等）都可以用 true-positive rate $\lambda$ 和 positive-prediction rate $\gamma$ 的线性分式形式 $u(S) = \frac{c_0 + c_1\lambda(S) + c_2\gamma(S)}{d_0 + d_1\lambda(S) + d_2\gamma(S)}$ 来近似，一阶展开后 $u$ 对 $(\lambda, \gamma)$ 近似为仿射函数。因此两种场景都可归结为 $u_\alpha = \alpha_1 u_1 + \alpha_2 u_2$ 的形式
    - 设计动机：统一框架使得一个鲁棒性指标可同时适用于两类场景，大幅扩展方法的适用范围

2. **Spatial Signature 与几何映射**:

    - 功能：将数据估值问题转化为可视化、可分析的几何问题
    - 核心思路：由 Proposition 3.1，存在映射 $\psi_{\omega,\mathcal{D}}: \mathcal{D} \to \mathbb{R}^2$，使得对任意 utility $u_\alpha$，$\phi(z; \omega, u_\alpha) = \langle \psi_{\omega,\mathcal{D}}(z), \alpha \rangle$。排序稳定性等价于所有点在 $\alpha$ 方向的投影顺序是否随方向旋转而改变。如果所有嵌入点近似共线，则旋转方向几乎不影响投影顺序，鲁棒性最高
    - 设计动机：线性内积结构使得排序变化与几何角度直接挂钩，排除了实际 utility 计算的复杂性

3. **鲁棒性指标 $R_p$**:

    - 功能：量化在 utility 变化下排序的稳定程度
    - 核心思路：对每对数据点 $(z_i, z_j)$，定义"切割角" $H_{ij} = \{\alpha \in \mathcal{S}^1 : \langle \alpha, v_{ij} \rangle = 0\}$，其中 $v_{ij} = \psi(z_i) - \psi(z_j)$。所有 $\binom{n}{2}$ 对产生 $2N$ 个切割点，将单位圆划分为排序不变的弧段。$\rho_p(\bar{\alpha}_0)$ 是从起始方向 $\bar{\alpha}_0$ 出发产生 $p$ 次配对交换所需的最小弧长。$R_p = \frac{\mathbb{E}[\rho_p]}{\pi/4}$ 归一化到 $[0,1]$，分母 $\pi/4$ 是所有点共线时的最大值
    - 设计动机：$R_p$ 可在 $O(n^2 \log n)$ 时间内精确计算，且与 Kendall 排序相关性退化程度直接对应

### 损失函数 / 训练策略

本文不涉及神经网络训练，而是一个分析性框架。核心理论结果 Proposition 3.3 表明，两个基础 utility 下 semivalue 分数向量的 Pearson 相关性可分解为 $\text{Corr}(\phi(u_1), \phi(u_2)) = \frac{\sum_j \omega_j^2 r_j}{\sqrt{\sum_j \omega_j^2 \text{Var}_j(u_1)} \sqrt{\sum_j \omega_j^2 \text{Var}_j(u_2)}}$，其中 $r_j$ 是 size-$j$ 对齐因子。Banzhaf 权重集中在 $r_j$ 较大的中间 coalition size 区域，因此系统性地获得更高的相关性和鲁棒性。

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

## 相关论文

- [When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency](when_to_retrain_after_drift_a_data-only_test_of_post-drift_data_size_sufficiency.md)
- [Impact of Layer Norm on Memorization and Generalization in Transformers](../../NeurIPS2025/others/impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)
- [OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)
- [Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)

<!-- RELATED:END -->
