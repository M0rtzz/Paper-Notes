---
title: >-
  [论文解读] Leveraging Predictive Equivalence in Decision Trees
description: >-
  [ICML2025][决策树] 提出将决策树转换为最小析取范式(DNF)表示，消除"预测等价性"问题，统一表示具有相同决策边界的不同决策树，进而改善变量重要性度量、缺失数据鲁棒性和特征获取成本优化。
tags:
  - ICML2025
  - 决策树
  - 预测等价性
  - 析取范式(DNF)
  - Quine-McCluskey
  - 变量重要性
  - 缺失数据
  - Rashomon集
---

# Leveraging Predictive Equivalence in Decision Trees

**会议**: ICML2025  
**arXiv**: [2506.14143](https://arxiv.org/abs/2506.14143)  
**代码**: 待确认  
**领域**: 决策树 / 可解释机器学习  
**关键词**: 决策树, 预测等价性, 析取范式(DNF), Quine-McCluskey, 变量重要性, 缺失数据, Rashomon集

## 一句话总结

提出将决策树转换为最小析取范式(DNF)表示，消除"预测等价性"问题，统一表示具有相同决策边界的不同决策树，进而改善变量重要性度量、缺失数据鲁棒性和特征获取成本优化。

## 研究背景与动机

决策树因其清晰的决策结构广泛用于可解释机器学习，但存在一个容易被忽视的核心问题——**预测等价性(Predictive Equivalence)**：同一决策边界可由多棵结构不同的决策树表示。例如，逻辑与函数 $X_1 \wedge X_2$ 既可以先判断 $X_1$ 再判断 $X_2$，也可以反过来，两棵树的预测完全一致但结构不同。

这种现象带来四个实际挑战：

**缺失数据处理**：若 $x_1$ 缺失但 $x_2=0$，先判断 $x_1$ 的树无法遍历，但先判断 $x_2$ 的树可直接预测为 0

**变量重要性不可靠**：Gini 重要性等基于树结构的指标在等价树上给出截然不同的结果

**Rashomon 集膨胀**：等价树在近优模型集中被重复计数，导致下游分析偏差

**特征获取成本次优**：树隐含的变量评估顺序未必是成本最优的

## 方法详解

### 核心思路：决策树 → 最小 DNF

给定决策树 $\mathcal{T}$，将其转换为简化的析取范式(DNF，即"或-与"形式) $\mathcal{T}_{\text{DNF}}$，抽象掉评估顺序。

**算法流程 (Algorithm 1)**：

1. 收集所有预测正类的叶子路径 $L^+$，构造正类表达式 $PosExpr = \bigvee_{l \in L^+} l$
2. 收集所有预测负类的叶子路径 $L^-$，构造负类表达式 $NegExpr = \bigvee_{l \in L^-} l$
3. 分别用改进的 Quine-McCluskey 算法化简：

$$SimplePosExpr = \text{QuineMcCluskey}^*(PosExpr)$$
$$SimpleNegExpr = \text{QuineMcCluskey}^*(NegExpr)$$

4. 返回 $(SimplePosExpr, SimpleNegExpr)$ 作为 $\mathcal{T}_{\text{DNF}}$

### 关键理论保证

- **忠实性 (Prop 3.1)**：对完整样本 $x$，$\mathcal{T}_{\text{DNF}}(x) = \mathcal{T}(x)$
- **完备性 (Thm 3.2)**：$\mathcal{T}_{\text{DNF}}(m(x)) \neq \text{NA}$ 当且仅当对 $m(x)$ 的所有补全 $z$，$\mathcal{T}(z) = \mathcal{T}_{\text{DNF}}(m(x))$
- **简洁性 (Prop 3.3)**：解释中不含冗余变量
- **等价消解 (Thm 3.4)**：$\mathcal{T}$ 与 $\mathcal{T}'$ 预测等价 $\Leftrightarrow$ $\mathcal{T}_{\text{DNF}} = \mathcal{T}'_{\text{DNF}}$

### Blake 典范形式

最小 DNF 并不包含所有充分条件。论文进一步引入 **Blake 典范形式(BCF)**，枚举所有最小充分条件，用于简化预测逻辑——只需逐项检查是否满足，无需进一步逻辑化简。

### 预测过程 (Algorithm 2)

对含缺失值的样本 $m(x)$：

1. 检查 $SimplePosExpr$ 中是否有项被已知值满足 → 预测 1
2. 检查 $SimpleNegExpr$ 中是否有项被已知值满足 → 预测 0
3. 将已知值代入 $SimplePosExpr$ 后再次化简，若为 True/False 则返回对应预测
4. 否则返回 NA

### 成本优化

利用 Q-learning 优化特征获取顺序：给定每个特征的获取成本 $c_j$，学习最优策略以最小化达成预测所需的总成本。DNF 表示使得特征获取顺序不受树结构约束。

## 实验关键数据

### Rashomon 集去重

| 数据集 | 总树数 | 去除平凡冗余 | 去除预测等价(本文) |
|--------|--------|-------------|-------------------|
| COMPAS | 12785±3e3 | 3913±837 | **2135±448** |
| Coupon | 666±54 | 136±19 | **55±8** |
| Wine | 6936±700 | 2341±377 | **1409±256** |
| Wisconsin | 24052±9e3 | 11990±5e3 | **4657±2e3** |

Rashomon 集中大量决策树实际上是预测等价的，去重后规模大幅缩小（如 Coupon 从 666 棵降至 55 棵）。

### 变量重要性修正

- 在合成数据 $Y = X_1 X_2$、$X_1,X_2 \sim \text{Bernoulli}(0.5)$ 下，两棵等价树给出的 Gini 重要性完全相反（一棵认为 $X_0$ 重要性是 $X_1$ 的两倍，另一棵反之）
- 对 RID (Rashomon Importance Distribution) 的修正：消除等价树后，所有变量的重要性估计与真值的 1-Wasserstein 距离均下降

| 方法 | $X_1$ 距离 | $X_2$ 距离 | $X_3$ 距离 |
|------|-----------|-----------|-----------|
| 原始 RID | 0.120 | 0.136 | 0.232 |
| 修正后 RID | **0.092** | **0.105** | **0.182** |

### 缺失数据鲁棒性

- DNF 表示能在大量缺失数据场景下直接给出预测，无需插补
- 关键推论：若 $\mathcal{T}_{\text{DNF}}(m(x)) \neq \text{NA}$，则其预测等同于完美 oracle 插补的结果
- 即使在 MNAR (非随机缺失) 设定下，DNF 预测仍为无偏估计

## 亮点与洞察

1. **问题定义精准**：首次系统性地定义和量化决策树的预测等价性问题，揭示了看似不同的树实际是同一分类器的不同"写法"
2. **表示论方法新颖**：将决策树转化为布尔逻辑最小表示，跨越了树结构与决策边界之间的鸿沟
3. **一套表示解决多个问题**：同一 DNF 表示同时改善了变量重要性、缺失数据处理和成本优化三个下游任务
4. **理论保证完备**：忠实性、完备性、简洁性、等价消解四条定理构成严谨的理论基础
5. **实用价值突出**：决策树在医疗、司法等高风险领域广泛使用，消除等价性对正确解释模型至关重要

## 局限与展望

1. **计算复杂度**：寻找最小 DNF 是 NP-hard 的，当决策树叶子数较多时 Quine-McCluskey 算法开销大，论文依赖小树假设（实验中最大深度仅为 3）
2. **仅处理二分类**：虽提及可扩展到多分类，但未给出具体多分类实验
3. **仅限二值化特征**：对连续特征需先二值化，可能引入信息损失
4. **成本优化依赖 Q-learning**：在大特征空间下的收敛性和效率未充分讨论
5. **数据集规模有限**：四个主数据集特征维度较低（7-30），在高维场景下的表现待验证

## 相关工作与启发

- **决策树优化**：GOSDT (Lin et al., 2020)、DL8.5 (Aglin et al., 2020)、MurTree (Demirović et al., 2022) 等最优决策树算法
- **路径解释冗余**：Izza et al. (2022) 指出决策树路径中的冗余变量问题，本文从全局表示层面解决
- **Rashomon 集分析**：TreeFARMS (Xin et al., 2022)、RID (Donnelly et al., 2023)
- **缺失数据**：MINTY (Stempfle & Johansson, 2024) 用逻辑析取增强缺失鲁棒性，与本文思路类似但本文更通用
- **启发**：布尔逻辑简化技术（经典 EDA 领域的 Quine-McCluskey）在 ML 可解释性中的新应用

## 评分

- 新颖性: ⭐⭐⭐⭐ — 预测等价性概念的系统化和 DNF 表示在 ML 可解释性中的应用颇具原创性
- 实验充分度: ⭐⭐⭐ — 理论验证充分但数据集偏小、场景有限
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，定理表述严谨，图示直观
- 价值: ⭐⭐⭐⭐ — 对决策树的可解释性研究有实质性推进，在高风险应用领域具有实用价值

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Near-Optimal Decision Trees in a SPLIT Second](near_optimal_decision_trees_in_a_split_second.md)
- [\[NeurIPS 2025\] Empowering Decision Trees via Shape Function Branching](../../NeurIPS2025/interpretability/empowering_decision_trees_via_shape_function_branching.md)
- [\[NeurIPS 2025\] ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making](../../NeurIPS2025/interpretability/valuepilot_a_two-phase_framework_for_value-driven_decision-making.md)
- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](../../NeurIPS2025/interpretability/toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)

<!-- RELATED:END -->
