---
title: >-
  [论文解读] Towards Attributions of Input Variables in a Coalition
description: >-
  [ICML 2025][Shapley value] 本文从 AND-OR 交互的视角重新推导了 Shapley value 的计算机制，证明了不同变量划分下的归因冲突本质上源于仅覆盖联盟部分变量的交互效应，并据此定义了联盟归因指标和三个忠实度度量，实验验证其与人类直觉一致。
tags:
  - ICML 2025
  - Shapley value
  - AND-OR interaction
  - coalition attribution
  - 归因冲突
  - 可解释AI
---

# Towards Attributions of Input Variables in a Coalition

**会议**: ICML 2025  
**arXiv**: [2309.13411](https://arxiv.org/abs/2309.13411)  
**代码**: 无  
**领域**: 可解释性 / XAI  
**关键词**: Shapley value, AND-OR interaction, coalition attribution, 归因冲突, 可解释AI

## 一句话总结

本文从 AND-OR 交互的视角重新推导了 Shapley value 的计算机制，证明了不同变量划分下的归因冲突本质上源于仅覆盖联盟部分变量的交互效应，并据此定义了联盟归因指标和三个忠实度度量，实验验证其与人类直觉一致。

## 研究背景与动机

**领域现状**：Shapley value 是可解释 AI 中最经典的变量归因方法，因满足匿名性、对称性、dummy、可加性和效率性五大公理被广泛使用。SHAP、Integrated Gradients 等基于 Shapley value 的方法已成为归因分析的事实标准。

**现有痛点**：然而，Shapley value 的计算严重依赖于输入变量的划分方式（partition）。例如在图像分类中，以像素还是以局部区域作为输入变量，或在 NLP 中以 token 还是以单词作为变量，不同划分会导致截然不同的归因结果。具体来说，将一组变量 $S = \{x_1, x_2\}$ 视为一个联盟（coalition）时，联盟的归因 $\varphi(S)$ 往往不等于 $\phi(x_1) + \phi(x_2)$，这就是所谓的**归因冲突**。

**核心矛盾**：现有方法（如 Faith-Shap）试图通过工程手段（最小化平方损失）来"消除"这种冲突，但缺乏对冲突产生原因的理论解释。无论是 Shapley value 还是 Banzhaf value，都无法在任意变量分组下保证归因的一致性。

**本文目标** (1) 揭示归因冲突的数学本质：到底是哪些因素导致了不同划分下归因值的不一致？ (2) 为联盟提供有理论支撑的归因定义。 (3) 提出度量指标来评估一组变量能否形成"忠实"的联盟。

**切入角度**：作者发现 AND-OR 交互可以完整描述 DNN 中所有输入变量之间的非线性关系，而 Shapley value 本质上就是对这些交互效应的均匀再分配。这一视角为分析冲突提供了精确的数学工具。

**核心 idea**：通过将 Shapley value 重新表述为 AND-OR 交互效应的分配问题，证明归因冲突的根源是那些仅覆盖联盟部分（而非全部）变量的交互。

## 方法详解

### 整体框架

给定 AI 模型 $v(\cdot)$ 和含 $n$ 个输入变量的样本 $x$，方法的目标是：(1) 用 AND-OR 交互重新表述 Shapley value 和 Banzhaf value；(2) 基于交互定义联盟归因 $\varphi(S)$；(3) 分解归因冲突为共享分量和冲突分量；(4) 提出三个忠实度度量评估联盟质量。整个框架从理论推导到实验验证形成闭环。

### 关键设计

1. **AND-OR 交互重新推导 Shapley value**:

    - 功能：将经典 Shapley value $\phi(i)$ 分解为 AND-OR 交互效应的加权和
    - 核心思路：AND 交互 $I_{\text{and}}(S)$ 表示 $S$ 中所有变量必须同时出现才产生效应（如"raining cats and dogs"四个词同时出现才表达"大雨"），OR 交互 $I_{\text{or}}(S)$ 表示任一变量存在即触发效应。定理 3.2 证明 $\phi(i) = \sum_{S \ni i} \frac{1}{|S|} [I_{\text{and}}(S) + I_{\text{or}}(S)]$，即 Shapley value 是将每个包含变量 $i$ 的交互效应均匀分配给所有参与变量的结果
    - 设计动机：这一重新表述将"黑箱"的 Shapley value 计算与模型内部的交互模式直接关联，为后续分析冲突提供了数学基础

2. **联盟归因指标 $\varphi(S)$ 的定义**:

    - 功能：将 Shapley value 从单个变量扩展到任意变量联盟
    - 核心思路：对于联盟 $S \subseteq N$，定义 $\varphi(S) = \sum_{T \supseteq S} \frac{|S|}{|T|} [I_{\text{and}}(T) + I_{\text{or}}(T)]$。即联盟 $S$ 的归因只来自那些完全包含 $S$ 的交互 $T$（$T \supseteq S$），分配权重为 $|S|/|T|$
    - 设计动机：与之前方法用工程损失强行消除冲突不同，本文接受冲突的客观存在，通过清晰定义联盟归因来解释冲突的来源

3. **归因冲突的分解与解释**:

    - 功能：将个体变量归因之和 $\sum_{i \in S} \phi(i)$ 分解为共享分量和冲突分量
    - 核心思路：定理 3.4 证明 $\sum_{i \in S} \phi(i) = \varphi(S) + \phi_{\text{conflict}}(S)$，其中冲突分量 $\phi_{\text{conflict}}(S) = \sum_{T: \emptyset \neq T \cap S \neq S} \frac{|T \cap S|}{|T|} [I_{\text{and}}(T) + I_{\text{or}}(T)]$。即那些仅覆盖 $S$ 中部分变量的交互 $T$（$T \cap S \neq \emptyset$ 且 $T \cap S \neq S$）是冲突的直接原因。推论 3.5 进一步证明，若模型从不编码仅覆盖 $S$ 部分变量的交互，则冲突为零
    - 设计动机：这一分解首次从理论上明确了"归因冲突是不可避免的客观存在"，因为 DNN 很难保证仅以完整联盟为单位编码交互

### 忠实度度量

基于上述理论分解，提出三个评估联盟忠实度的度量：

- **$R(i)$**：评估联盟 $S$ 中特定变量 $i$ 的归因有多少来自联盟相关的交互，$R(i) = |U_{i,S}| / (|U_{i,S}| + |U_{i,\bar{S}}|)$，值越接近 1 表示联盟越忠实
- **$R'(i)$**：更细粒度的评估，用交互效应的绝对值来衡量变量 $i$ 参与联盟的显著性
- **$Q(S)$**：评估整个联盟 $S$ 的忠实度，衡量分配给完整联盟的交互强度占所有涉及 $S$ 变量的交互强度的比例

## 实验关键数据

### 主实验：理论验证——精度误差

| 验证内容 | 联盟阶数 $m$ | 近似误差 |
|----------|-------------|---------|
| 用联盟归因近似 Shapley value | $m=1$ | $3.6 \times 10^{-8}$ |
| 用联盟归因近似 Shapley value | $m=5$ | $7.6 \times 10^{-7}$ |
| 用联盟归因近似 Shapley value | $m=10$ | $2.8 \times 10^{-7}$ |
| 用联盟归因近似模型输出 | $m=1$ | $2.3 \times 10^{-7}$ |
| 用联盟归因近似模型输出 | $m=5$ | $6.3 \times 10^{-7}$ |
| 用联盟归因近似模型输出 | $m=10$ | $4.7 \times 10^{-7}$ |

所有误差在 $10^{-7}$ 数量级，证明理论推导的正确性。

### 消融实验：联盟忠实度度量在 Toy 函数上的验证

| 联盟类型 | $\mathbb{E}[R(i)]$ | $\mathbb{E}[R'(i)]$ | $\mathbb{E}[Q(S)]$ |
|---------|--------------------|--------------------|-------------------|
| 纯忠实联盟 | 0.944 | 0.936 | 0.948 |
| 部分忠实联盟 | 0.471 | 0.608 | 0.590 |
| 纯不忠实联盟 | 0.031 | 0.016 | 0.013 |

三个度量都能清晰区分忠实/不忠实联盟：纯忠实联盟接近 1，纯不忠实联盟接近 0。

### NLP 实验：SST-2 情感分类上的联盟忠实度

| 句子 | 联盟 | 模型 | $Q(S)$ | 说明 |
|------|------|------|--------|------|
| "the mesmerizing performances..." | {mesmerizing, performances} | BERT-large | 0.743 | 自然短语，忠实联盟 |
| "the mesmerizing performances..." | {mesmerizing, performances} | LLaMA | 0.746 | 跨模型一致 |
| "...easily rivaling blair witch..." | {rivaling, blair} | BERT-large | 0.425 | 拆散短语，不忠实 |
| "...easily rivaling blair witch..." | {rivaling, blair} | LLaMA | 0.312 | 跨模型一致认为不忠实 |

### 关键发现

- 归因冲突在所有测试的 DNN 中普遍存在，是模型交互编码方式的自然结果，而非计算错误
- 三个度量在 toy 函数上能完美区分三种联盟类型，在真实 NLP 模型上与人类语义直觉高度一致
- 在围棋实验中，KataGo 编码的高强度交互对应的联盟模式与专业棋手标注的经典棋形高度吻合

## 亮点与洞察

- **从交互视角统一理解归因方法**：将 Shapley value 和 Banzhaf value 都表述为 AND-OR 交互的分配问题，这一统一框架极其优雅，为理解各种归因方法提供了共同的数学语言
- **"冲突不可消除，只能解释"的哲学**：与之前方法试图工程化消除冲突不同，本文证明冲突是 DNN 交互结构的客观反映，这一思路转变为后续研究指明了方向
- **联盟归因满足五大公理**：新定义的联盟归因保持了 Shapley value 的核心理论优势，可作为任意变量分组的标准归因方法
- **围棋应用展示实用价值**：将理论应用于 KataGo 中棋形（shape pattern）的归因分析，帮助棋手理解 AI 的落子策略，是理论到应用的优秀示范

## 局限与展望

- **计算复杂度极高**：AND-OR 交互的精确计算涉及 $2^n$ 个子集的遍历，围棋实验中仅能处理 $n=10$ 个棋子。如何高效近似计算是实际应用的主要瓶颈
- **仅验证了小规模场景**：NLP 实验选取 10 个单词、围棋选取 10 颗棋子，未展示在大规模输入（如完整图像的数千像素）上的可行性
- **人工标注联盟的主观性**：NLP 和围棋实验中"自然联盟"依赖人工标注，缺乏自动发现最优联盟划分的算法
- **缺少对比基线方法的实验**：未与 Faith-Shap 等方法在相同设置下进行量化对比，仅在 Table 1 中做了定性比较

## 相关工作与启发

- **vs Faith-Shap (Tsai et al., 2023)**：Faith-Shap 用最小二乘损失 $\|v(S) - \sum_{i \in S} \phi(i)\|^2$ 迫使联盟归因接近个体之和，属于工程化消除冲突。本文证明冲突无法消除，而是提供了理论解释，思路更根本
- **vs Joint Shapley Value (Harris et al., 2021)**：Joint Shapley 提出了联合特征集的归因度量，但本质上度量的是特征集/交互的归因（类似 Shapley Taylor），而非解释冲突机制。本文从交互分解的角度给出了更清晰的理论框架
- **vs Banzhaf Value**：Banzhaf value 满足 2-efficiency 但不满足一般集合的 efficiency。本文的联盟归因同时适用于 Shapley 和 Banzhaf 框架

## 评分

- 新颖性: ⭐⭐⭐⭐ 用 AND-OR 交互统一解释归因冲突的思路新颖且理论深度高
- 实验充分度: ⭐⭐⭐ 实验覆盖合成/NLP/图像/围棋多场景，但规模偏小且缺乏与基线的定量对比
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，定理-推论链条清晰，但符号较多读起来有一定门槛
- 价值: ⭐⭐⭐⭐ 为 XAI 归因方法提供了重要的理论基础，但计算复杂度限制了实际应用

<!-- RELATED:START -->

## 相关论文

- [Minimizing False-Positive Attributions in Explanations of Non-Linear Models](../../NeurIPS2025/interpretability/minimizing_false-positive_attributions_in_explanations_of_non-linear_models.md)
- [Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](../../CVPR2026/interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)
- [ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](../../AAAI2026/interpretability/shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective](evolving_prompts_in-context_an_open-ended_self-replicating_perspective.md)
- [What Makes an Ensemble (Un)interpretable?](what_makes_an_ensemble_un_interpretable.md)

<!-- RELATED:END -->
