---
title: >-
  [论文解读] Rethinking Aleatoric and Epistemic Uncertainty
description: >-
  [ICML 2025][aleatoric uncertainty] 本文指出机器学习中 aleatoric/epistemic 不确定性二分法存在根本性概念混淆，提出基于决策理论的替代框架，将预测不确定性、可约/不可约分解、预测性能和数据分散度统一在一个连贯的理论体系中，并揭示了 BALD 作为 epistemic uncertainty 估计器的局限性。
tags:
  - ICML 2025
  - aleatoric uncertainty
  - epistemic uncertainty
  - decision theory
  - BALD
  - uncertainty decomposition
---

# Rethinking Aleatoric and Epistemic Uncertainty

**会议**: ICML 2025  
**arXiv**: [2412.20892](https://arxiv.org/abs/2412.20892)  
**代码**: 无  
**领域**: 不确定性量化 / 贝叶斯机器学习 / 决策理论  
**关键词**: aleatoric uncertainty, epistemic uncertainty, decision theory, BALD, uncertainty decomposition

## 一句话总结

本文指出机器学习中 aleatoric/epistemic 不确定性二分法存在根本性概念混淆，提出基于决策理论的替代框架，将预测不确定性、可约/不可约分解、预测性能和数据分散度统一在一个连贯的理论体系中，并揭示了 BALD 作为 epistemic uncertainty 估计器的局限性。

## 研究背景与动机

**领域现状**：在机器学习中，研究者广泛使用 aleatoric（偶然性，与数据噪声相关）和 epistemic（认知性，与模型知识状态相关）不确定性来推理模型的概率预测。这一二分法自 Gal (2016)、Kendall & Gal (2017) 以来成为主流范式，被用于主动学习、模型选择、OOD 检测等诸多场景。

**现有痛点**：文献中对这两个概念的讨论存在严重的不一致。具体表现为：
   - 模型预测与数据生成过程的界限被反复模糊（Amini et al., 2020; van Amersfoort et al., 2020）
   - 对不确定性在未见数据上如何分解做出了不合理的假设
   - 不确定性与预测精度之间建立了误导性的联系
   - 同一概念（如 epistemic uncertainty）被用密度、信息量、方差等完全不同的数学量来定义

**核心矛盾**：aleatoric-epistemic 二分法本质上只能容纳两个概念，但研究者实际需要区分的量远超两个——包括预测不确定性、不确定性的可约/不可约分解、预测性能、数据统计分散度等。将这些不同概念强行塞入两个标签之下，导致了概念过载（conceptual overloading），使本应区分的量被混为一谈。

**本文目标**
   - 揭示现有 aleatoric-epistemic 框架中的具体矛盾
   - 建立一个能连贯表达所有相关量的替代理论框架
   - 分析 BALD 作为不确定性度量的估计质量

**切入角度**：作者从决策理论出发，以一个最终决策（action）及其损失函数为起点，利用主观期望损失来形式化不确定性，避免了直接依赖 Shannon 熵或方差等特定量度。

**核心 idea**：用决策理论框架取代 aleatoric-epistemic 二分法，通过损失函数驱动的不确定性定义，连贯地统一预测不确定性、可约性分解、预测性能和数据分散度。

## 方法详解

### 整体框架

本文不是一个具体的算法，而是一个**理论框架的重建**。整体思路如下：

- **输入**：一个决策问题 $(a \in \mathcal{A}, z \in \mathcal{Z}, \ell(a,z))$ + 训练数据 $y_{1:n}$ + 预测模型 $p_n(z)$
- **输出**：对预测不确定性（predictive uncertainty）、不确定性可约性（reducibility）、预测性能（predictive performance）、数据分散度（data dispersion）四个概念的严格定义及其相互关系
- **关键中间步骤**：
  1. 定义基于 Bayes 最优行动的主观不确定性
  2. 通过推理新数据的影响，推导可约/不可约分解
  3. 连接经典统计和信息论中的分解公式
  4. 分析 BALD 在此框架下的定位

### 关键设计

1. **决策论基础（Decision-Theoretic Foundation）**:

    - 功能：以 action-loss 对 $(a, \ell)$ 为出发点定义不确定性
    - 核心思路：不确定性被定义为在信念 $p_n(z)$ 下 Bayes 最优行动的主观期望损失，即 $U_n = \min_{a \in \mathcal{A}} \mathbb{E}_{p_n(z)}[\ell(a, z)]$。当 $\ell$ 是 log loss 时退化为 Shannon 熵，当 $\ell$ 是 squared error 时退化为方差
    - 设计动机：避免先验地选择熵或方差作为不确定性度量——让损失函数决定什么是"不确定的"，从而为不同任务提供统一且任务相关的不确定性定义

2. **不确定性的可约/不可约分解（Reducible vs Irreducible Decomposition）**:

    - 功能：将预测不确定性分解为随数据增长可消除和不可消除的两部分
    - 核心思路：定义 $U_\infty = \lim_{n \to \infty} U_n$ 为数据量趋于无穷时的不可约不确定性，则可约部分为 $U_n - U_\infty$。通过考虑获取新数据后不确定性的期望变化，可以不依赖无穷数据极限来估计可约程度
    - 设计动机：这一分解替代了传统的 aleatoric = irreducible、epistemic = reducible 的简化对应，同时明确了"可约性"取决于模型类、更新方法和数据获取策略，而非仅由数据"噪声"决定
    - 与之前方法的区别：传统做法将 $\mathbb{E}_{p_n(\theta)}[H[p_n(z|\theta)]]$ 直接等同于 aleatoric/irreducible uncertainty，但这只是一个有限样本估计器，可能偏差很大

3. **预测性能与数据分散度的区分（Performance vs Dispersion）**:

    - 功能：严格区分 (a) 模型关于自身信念的不确定性, (b) 模型相对于外部 ground truth 的表现, (c) 数据源本身的统计分散程度
    - 核心思路：预测性能使用参考分布 $p_{\text{eval}}(z)$ 下的期望损失来衡量，即 $R_n = \mathbb{E}_{p_{\text{eval}}(z)}[\ell(a_n^*, z)]$；数据分散度则是数据生成过程 $p_{\text{train}}$ 本身的性质。三者虽相关但本质不同
    - 设计动机：在经典 aleatoric-epistemic 框架下，"aleatoric uncertainty" 被同时用来指代 $H[p_\infty(z)]$、$H[p_{\text{train}}]$ 和 $H[p_{\text{eval}}(z)]$ 三个不同的量——而它们一般情况下并不相等

4. **对 BALD 的重新解读（Reinterpreting BALD）**:

    - 功能：分析 BALD 分数 $\text{EIG}_\theta = H[p_n(z)] - \mathbb{E}_{p_n(\theta)}[H[p_n(z|\theta)]]$ 作为可约预测不确定性的估计质量
    - 核心思路：BALD 衡量的是观察 $z$ 后对参数 $\theta$ 的信息增益，而非预测不确定性的长期可约量。由于参数到预测的映射通常不可逆，参数空间的信息增益不一定转化为预测空间的不确定性减少
    - 关键发现：BALD 更应被理解为近似度量**短期**参数不确定性减少，而非长期可约预测不确定性。它在实践中有效，但在以预测为导向的场景下仍是次优的

### 理论分析

本文的核心贡献之一是指出经典公式 Eq.(1) 中的三大混淆源：

- **混淆 1**：将 $\mathbb{E}_{p_n(\theta)}[H[p_n(z|\theta)]]$（条件预测熵的期望）等同于不可约不确定性 $H[p_\infty(z)]$——前者在有限 $n$ 下只是后者的一个估计器，可能非常不准确
- **混淆 2**：将模型的主观预测信念与真实数据生成过程混为一谈——$p_n(z)$ 通常不等于 $p_{\text{train}}$ 或 $p_{\text{eval}}$
- **混淆 3**：将参数空间的信息增益等同于预测空间的不确定性减少——参数到预测的映射不可逆时两者不同

## 实验关键数据

本文为理论/概念性工作，不包含标准 benchmark 实验，而是通过概念分析和数学论证来支撑结论。以下整理其核心的概念对比和分析结果。

### 经典 aleatoric-epistemic 框架的混淆分析

| 概念标签 | 被赋予的含义 | 实际对应的数学量 | 问题分析 |
|---------|------------|----------------|---------|
| "Aleatoric" 含义 1 | 数据生成的固有噪声 | $H[p_{\text{train}}(y_{1:n} \mid \pi)]$ | 训练数据分布 ≠ 预测分布 |
| "Aleatoric" 含义 2 | 评估数据的不确定性 | $H[p_{\text{eval}}(z)]$ | 评估分布 ≠ 预测分布 |
| "Aleatoric" 含义 3 | 条件预测熵期望 | $\mathbb{E}_{p_n(\theta)}[H[p_n(z \mid \theta)]]$ | 仅是 $H[p_\infty(z)]$ 的有限样本估计器 |
| "Epistemic" 含义 1 | 参数的不确定性 | $H[p_n(\theta)]$ | 参数→预测映射不可逆 |
| "Epistemic" 含义 2 | 参数信息增益 (BALD) | $H[p_n(z)] - \mathbb{E}_{p_n(\theta)}[H[p_n(z \mid \theta)]]$ | 估计长期可约不确定性可能偏差大 |

### 决策论框架 vs 经典框架对照

| 特征维度 | 经典 Aleatoric-Epistemic 框架 | 本文决策论框架 |
|---------|---------------------------|-------------|
| 概念容量 | 仅 2 个标签（aleatoric / epistemic） | 4+ 个独立概念 |
| 不确定性定义 | 固定为 Shannon 熵或方差 | 由损失函数 $\ell(a,z)$ 灵活决定 |
| 可约/不可约分解 | 通过 Eq.(1) 近似（可能不准确） | 通过 $U_n - U_\infty$ 严格定义 |
| 预测性能 | 与不确定性混为一谈 | 独立定义为参考分布下的损失 |
| 数据分散度 | 等同于 aleatoric uncertainty | 独立概念，$p_{\text{train}}$ 的性质 |
| 模型假设 | 需要随机参数 $\theta$ | 适用于任意学习方法 |
| 适用范围 | 贝叶斯模型 + 信息论量 | 任意 ML 方法（含深度学习、集成等） |

### 关键发现

- **BALD 并非可约不确定性的直接度量**：它衡量的是参数信息增益，而非预测不确定性的长期可约量。两者的差异在有限数据下可能显著。
- **"Aleatoric = irreducible" 这一等式不成立**：不可约不确定性 $H[p_\infty(z)]$ 不仅取决于数据"噪声"，还依赖于模型类和更新方法。同一数据源对不同模型的不可约不确定性不同。
- **条件预测熵 $\mathbb{E}[H[p_n(z|\theta)]]$ 是 $H[p_\infty(z)]$ 的有偏估计器**：在数据不足时，这个估计器的偏差可以很大，导致对 aleatoric uncertainty 的错误评估。
- **BALD 的实际有效性来源**：它在实际中之所以有用，是因为它近似度量了短期内参数不确定性的减少，而在 active learning 等迭代场景中，短期减少量本身即有指导价值。

## 亮点与洞察

- **概念过载的精准诊断**：作者系统梳理了 aleatoric/epistemic 如何被赋予多重不兼容含义（Figure 2 的图示极为清晰），指出这不仅是语义问题，而是影响方法设计和评估的实质性障碍。这种"概念审计"的方法论本身值得借鉴。

- **损失函数驱动的不确定性定义**：不确定性 = Bayes 最优行动的期望损失，这一定义优雅地统一了方差、Shannon 熵等特例，同时自然地将不确定性与实际任务绑定。在需要任务特定不确定性度量的场景（如医学诊断中 false negative 代价高于 false positive）中尤其有启发性。

- **预测 vs 性能 vs 分散度的三角关系**：将主观信念（$p_n(z)$）、外部评估（$p_{\text{eval}}$）和数据生成（$p_{\text{train}}$）明确分离，避免了长期以来将"模型不确定"与"问题本身有噪声"混为一谈的错误。这一区分可以迁移到任何涉及模型评估的任务中。

- **BALD 的重新定位**：不否认 BALD 的实际价值，但精确指出了它的理论局限——作为长期估计器不可靠，但作为短期近似有效。这为改进 active learning 目标函数指明了方向。

## 局限与展望

- **纯理论贡献，缺少实证验证**：论文核心是概念框架重建，没有在实际数据集上验证新框架是否带来更好的方法设计或更准确的不确定性估计。未来需要设计实验展示新框架相比旧范式的实际收益。
- **$U_\infty$ 的计算困难**：虽然理论上 $U_n - U_\infty$ 是可约不确定性的严格定义，但在实践中 $U_\infty$ 通常无法直接计算，仍需依赖有限样本估计器。论文对如何在实践中近似 $U_\infty$ 讨论不足。
- **损失函数的选择**：框架要求预先指定损失函数 $\ell$，但在许多探索性场景中（如表征学习、无监督学习），最终损失函数未必明确。框架的适用性在这些场景下打折扣。
- **对现有方法的替代方案不够具体**：批评了 BALD 等现有量，但没有提出具体的替代算法或计算方案，对实践者的指导价值有限。
- **模型 agnostic 的代价**：为了适用于任意 ML 方法，框架放弃了贝叶斯推断的结构性优势（如参数后验的可解释性），可能在特定场景下不如针对性理论。

## 相关工作与启发

- **vs Kendall & Gal (2017)**：他们将 total = aleatoric + epistemic 形式化为信息论分解 Eq.(1)，本文指出该分解在有限数据下是近似的，且各项被赋予了相互矛盾的含义。本文优势在于概念连贯性，但 Kendall & Gal 提供了实际可计算的简单量。

- **vs Wimmer et al. (2023)**：Wimmer 反对 Shannon 熵作为预测不确定性度量，本文更进一步，说明 Shannon 熵只是决策论框架在 log loss 下的特例，从而将这一批评纳入更一般的理论中。

- **vs Sale et al. (2023b, 2024b)**：Sale 等人提出用逐类方差来定义不确定性，本文的框架可以将其理解为特定损失函数下的 Bayes 风险，提供了更统一的视角。

- **vs Hofman et al. (2024b)**：Hofman 等人基于 proper scoring rules 定义不确定性，与本文的损失函数驱动方法思路一致但侧重不同，两者可视为互补。

- **与 active learning 研究的关系**：本文对 BALD 的分析直接启发了 active learning 目标函数的改进——应当直接优化预测不确定性减少而非参数信息增益（Bickford Smith et al., 2023, 2024）。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 对 ML 中最基础的不确定性概念进行系统性重审，决策论视角虽非全新但在此语境下的应用极具原创性
- 实验充分度: ⭐⭐⭐ 作为理论论文无标准实验，但缺少实证验证是明显遗憾
- 写作质量: ⭐⭐⭐⭐⭐ 论证层层递进，概念图示（Figure 1, 2）清晰，每个矛盾都有具体文献和数学量的支撑
- 价值: ⭐⭐⭐⭐ 对理解不确定性的概念基础有重要意义，但实践影响有待后续工作验证

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] UnHiPPO: Uncertainty-Aware Initialization for State Space Models](unhippo_uncertainty-aware_initialization_for_state_space_models.md)
- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](../../AAAI2026/others/an_epistemic_perspective_on_agent_awareness.md)
- [\[NeurIPS 2025\] Rethinking PCA Through Duality](../../NeurIPS2025/others/rethinking_pca_through_duality.md)
- [\[NeurIPS 2025\] Uncertainty Estimation by Flexible Evidential Deep Learning](../../NeurIPS2025/others/uncertainty_estimation_by_flexible_evidential_deep_learning.md)
- [\[ACL 2025\] LoGU: Long-form Generation with Uncertainty Expressions](../../ACL2025/others/logu_longform_gen_uncertainty.md)

<!-- RELATED:END -->
