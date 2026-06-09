---
title: >-
  [论文解读] Conformal Prediction for Causal Effects of Continuous Treatments
description: >-
  [NeurIPS 2025][优化/理论][conformal prediction] 首次为连续处理变量（如药物剂量）的因果效应构建共形预测区间，通过倾向性偏移参数化和分位数回归，在已知/未知倾向性两种场景下均提供有限样本 $1-\alpha$ 覆盖保证。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "conformal prediction"
  - "Continuous Treatment"
  - "Causal Inference"
  - "Propensity Score"
  - "Uncertainty Quantification"
  - "Potential Outcomes"
---

# Conformal Prediction for Causal Effects of Continuous Treatments

**会议**: NeurIPS 2025  
**arXiv**: [2407.03094](https://arxiv.org/abs/2407.03094)  
**代码**: [GitHub](https://github.com/m-schroder/ContinuousCausalCP)  
**领域**: 因果推断 / 共形预测  
**关键词**: conformal prediction, Continuous Treatment, Causal Inference, Propensity Score, Uncertainty Quantification, Potential Outcomes

## 一句话总结

首次为连续处理变量（如药物剂量）的因果效应构建共形预测区间，通过倾向性偏移参数化和分位数回归，在已知/未知倾向性两种场景下均提供有限样本 $1-\alpha$ 覆盖保证。

## 研究背景与动机

**领域现状**：共形预测（Conformal Prediction, CP）因其模型无关、无分布假设的有限样本覆盖保证，在不确定性量化领域越来越受关注。但现有因果CP方法全部局限于二元或离散处理（如"是否用药"），对连续处理（如"用药剂量"）无能为力。

**现有痛点**：将CP从传统预测任务迁移到因果推断面临三重挑战：
   - **挑战①**：干预改变了倾向性函数 $\pi(a|x)$，导致观测分布和干预分布之间产生偏移，CP核心的可交换性（exchangeability）假设被打破
   - **挑战②**：观测数据中倾向性分数通常未知，需要从数据中估计，估计误差引入额外不确定性
   - **挑战③**：连续处理下几乎不存在两个完全相同处理值的数据点，无法直接做条件校准

**核心矛盾**：CP需要可交换性保证有效覆盖，但因果干预天然破坏可交换性。如何在分布偏移下仍然提供严格的有限样本覆盖保证？

**现有方法的不足**：
   - MC Dropout：后验近似质量差，覆盖率不可靠
   - 深度集成（Deep Ensemble）：无理论覆盖保证
   - 贝叶斯方法：需要先验分布假设，对模型误指定不鲁棒
   - 现有因果CP方法（如 Lei & Candès 2021）：仅限二元处理，离散化连续处理会导致因果估计量定义不良

**本文切入角度**：将干预诱导的倾向性偏移建模为 tilting 函数族 $\mathcal{F}$，通过对该函数族的鲁棒优化构建CP区间，使得在所有可能的分布偏移下均满足覆盖保证。

## 方法详解

### 问题建模

设数据 $(X_i, A_i, Y_i)_{i=1}^n$ 由混杂因素 $X \in \mathcal{X}$、连续处理 $A \in \mathcal{A}$、结果 $Y \in \mathcal{Y}$ 组成。目标是对新样本 $X_{n+1}$ 在干预 $\Diamond$（硬干预 $a^*$ 或软干预 $A^*(X_{n+1})$）下构建预测区间 $C(X_{n+1}, \Diamond)$，使得：

$$P(Y_{n+1}(\Diamond) \in C(X_{n+1}, \Diamond)) \geq 1 - \alpha$$

数据分为训练集 $D_T$（训练预测模型 $\phi$）和校准集 $D_C$（构建CP区间）。非一致性分数（non-conformity score）取残差形式：$S_i = |Y_i - \phi(X_i, A_i)|$。

### 整体框架

两阶段流程：
1. 在训练集上训练任意因果效应预测模型 $\phi$ 并在校准集上计算非一致性分数
2. 求解倾向性偏移约束下的分位数回归问题，得到满足覆盖保证的CP区间阈值 $S^*$

### 关键设计一：倾向性偏移参数化（Tilting Function）

干预将观测倾向性 $\pi(a|x)$ 偏移为干预倾向性 $\tilde{\pi}(a|x)$，通过非负 tilting 函数 $f$ 联系两者：

$$\tilde{\pi}(a|x) = \frac{f(a,x)}{\mathbb{E}_P[f(A,X)]} \pi(a|x)$$

这一参数化将因果推断中的分布偏移问题转化为CP框架下的鲁棒优化问题。校准时不再假设可交换性，而是对 $f \in \mathcal{F}$ 做条件校准，在所有可能偏移下保持覆盖。

### 关键设计二：已知倾向性场景（Theorem 4.2）

对软干预 $A^* = A + \Delta_A$，定义函数族 $\mathcal{F} = \{\theta \frac{\pi(a+\Delta_A|x)}{\pi(a|x)} \mid \theta \in \mathbb{R}^+\}$。直接求解需对所有 $S \in \mathbb{R}$ 枚举，计算不可行。利用**强对偶性**将问题转为对偶形式：

$$\max_{\eta_i} \min_{\theta > 0} \sum_{i=m+1}^{n} \eta_i(S_i - \theta \frac{\pi(a_i + \Delta_A | x_i)}{\pi(a_i | x_i)}) + \eta_{n+1}(S - \theta \frac{\pi(a^* | x_{n+1})}{\pi(a_{n+1} | x_{n+1})})$$

约束 $-\alpha \leq \eta_i \leq 1-\alpha$。定义 $S^*$ 为满足 $\eta_{n+1}^S < 1-\alpha$ 的最大 $S$，则 $C(x_{n+1}, a^*) = \{y \mid S_{n+1}(y) \leq S^*\}$ 满足覆盖保证。

### 关键设计三：未知倾向性场景（Theorem 4.5）

硬干预 $\text{do}(a^*)$ 对应 Dirac-delta 倾向性 $\delta_{a^*}(a)$，直接处理会发散。解决方案分三步：

1. **高斯核平滑**：用高斯函数逼近 Dirac-delta，$\delta_{a^*}(a) = \lim_{\sigma \to 0} \frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{(a-a^*)^2}{2\sigma^2})$

2. **估计误差有界假设（Assumption 1）**：假设倾向性估计误差 $c_{a_i} = \hat{\pi}(a_i|x_i) / \pi(a_i|x_i) \in [1/M, M]$，其中 $M$ 由领域专家指定

3. **Type-I Invexity 保证全局最优（Lemma 4.4）**：优化问题虽非凸，但满足 Type-I invexity 和线性独立约束条件（LICQ），因此 KKT 条件既是必要的也是充分的，可找到全局最优解

最终CP区间 $C(X_{n+1}, a^*) = \{y \mid S_{n+1}(y) \leq S^*\}$，其中 $S^*$ 为 $v_{n+1}^S > 0$ 的最大 $S$。

### 损失函数与训练策略

- 阶段1：用标准 MLP 训练因果预测模型 $\phi$，MC Dropout 正则化率 0.1
- 阶段2：用 pinball loss 做分位数回归：$l_\alpha(\theta, S) = (\alpha - \mathbf{1}[\theta - S < 0])(\theta - S)$
- 未知倾向性场景用条件归一化流（conditional normalizing flows）估计 $\hat{\pi}(a|x)$

## 实验关键数据

### 合成数据实验

在两个合成数据集上评估（数据集1：分段倾向性+凹结果函数；数据集2：高斯倾向性+振荡结果函数），50次随机运行取平均：

| 方法 | 覆盖率（$\alpha=0.05$，目标0.95） | 覆盖率（$\alpha=0.1$） | 覆盖率（$\alpha=0.2$） |
|------|:---:|:---:|:---:|
| **CP（本文）** | **1.00** | **0.90-0.94** | **0.83-0.88** |
| MC-Dropout | 0.02-0.28 | 0.02-0.23 | 0.02-0.11 |
| 高斯过程 | 0.125 | 0.125 | 0.083 |
| 深度集成 | 更差 | — | — |

本文CP方法在所有设置下均达到或超过目标覆盖率，而 MC-Dropout 覆盖率仅为目标的 2%-30%。

### MIMIC-III 临床数据

在真实重症监护数据上评估（14,719名患者，8个临床混杂因素，预测机械通气时长对血压的影响）：

- CP区间在训练数据稀疏的高处理值区域自动变宽，反映真实不确定性
- MC-Dropout 区间在所有区域均偏窄，暗示覆盖率不足
- CP区间的行为符合临床直觉：罕见剂量区域不确定性大

### 关键发现

- **覆盖保证差距悬殊**：MC-Dropout 覆盖率仅为目标的 2%-30%，对安全关键应用完全不可接受
- **区间宽度语义合理**：CP区间与数据支撑程度呈反相关，数据少则区间宽
- **显著性水平 $\alpha$ 敏感性正确**：$\alpha$ 增大时区间变窄，行为符合理论预期

## 亮点与洞察

- **Tilting 函数框架的优雅性**：将因果推断中复杂的分布偏移问题统一为一个函数族上的鲁棒优化，理论框架自然通用，既处理软干预也处理硬干预
- **高斯核逼近 Dirac-delta 的数学巧妙性**：硬干预的倾向性是 Dirac-delta（不可积的广义函数），通过高斯极限绕开了这一数学障碍，并且证明了非凸优化在 Type-I invexity 下仍可全局求解
- **Assumption 1 的实用价值**：误差界 $M$ 由领域专家指定而非从数据推断，赋予从业者直接控制权。这比假设倾向性完全已知更现实，也比完全未知更有约束力
- **模型无关性**：CP区间可搭配任意因果效应预测模型，不绑定特定网络结构

## 局限性与改进方向

- **误差界 $M$ 的指定缺乏自动化方法**：过保守（$M$ 大）区间过宽失去实用性，过乐观（$M$ 小）覆盖不足。论文建议保守选择但未给出数据驱动的选择策略
- **样本分割降低数据效率**：训练/校准/测试三分在小样本场景下可能导致校准集不足
- **仅限单变量连续处理**：未讨论多维连续处理（如多种药物联合给药）
- **高斯核宽度 $\sigma$ 的敏感性**：理论上 $\sigma \to 0$，实际中选择有限 $\sigma$ 对性能的影响未充分消融
- **计算复杂度**：对大规模 CATE 向量，优化过程可能较慢

## 相关工作对比

| 方法类别 | 代表工作 | 与本文区别 |
|---------|---------|----------|
| 离散处理CP | Lei & Candès 2021; Alaa et al. 2023 | 仅限二元/离散处理，离散化连续处理导致因果量定义不良 |
| 已知倾向性CP | Jin et al. 2023 | 假设倾向性已知，实际观测数据中通常未知 |
| MC-Dropout | Gal & Ghahramani 2016 | 后验近似质量差，覆盖率不可靠，无有限样本保证 |
| 贝叶斯方法 | Alaa & van der Schaar 2017 | 需要先验假设，对模型误指定不鲁棒 |
| 分布偏移CP | Barber et al. 2023; Gibbs & Candès 2021 | 偏移已知或只给渐近保证，不适用于因果干预场景 |

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次解决连续处理因果CP问题，tilting 函数族+invexity 的数学工具运用创新
- 实验充分度: ⭐⭐⭐⭐ 合成数据+MIMIC-III真实数据，多 baseline 多 $\alpha$ 值对比，50次随机种子
- 写作质量: ⭐⭐⭐⭐ 定理-引理链条清晰，两种场景分治框架明确，证明留附录
- 实用价值: ⭐⭐⭐⭐⭐ 对医疗等安全关键应用（如化疗剂量选择）有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[NeurIPS 2025\] Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization](conformal_prediction_in_the_loop_a_feedback-based_uncertainty_model_for_trajecto.md)
- [\[ICML 2025\] Conformal Prediction as Bayesian Quadrature](../../ICML2025/optimization/conformal_prediction_as_bayesian_quadrature.md)
- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](../../CVPR2025/optimization/conformal_prediction_for_zero-shot_models.md)
- [\[ICML 2025\] On Temperature Scaling and Conformal Prediction of Deep Classifiers](../../ICML2025/optimization/on_temperature_scaling_and_conformal_prediction_of_deep_classifiers.md)

</div>

<!-- RELATED:END -->
