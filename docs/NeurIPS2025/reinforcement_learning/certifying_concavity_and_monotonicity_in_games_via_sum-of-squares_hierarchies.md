---
title: >-
  [论文解读] Certifying Concavity and Monotonicity in Games via Sum-of-Squares Hierarchies
description: >-
  [NeurIPS 2025][强化学习][凹博弈] 证明了在多项式效用和半代数策略集的博弈中验证凹性和单调性是 NP-hard 的，并提出了两套基于平方和 (SOS) 规划的层次化认证方案，可在多项式时间内逐层求解。 多人博弈中，每个玩家独立选择动作以最大化自身收益，而收益取决于其他玩家的行动。博弈的结构性质——特别是凹性和…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "凹博弈"
  - "单调博弈"
  - "平方和规划"
  - "Nash均衡"
  - "不完美回忆博弈"
---

# Certifying Concavity and Monotonicity in Games via Sum-of-Squares Hierarchies

**会议**: NeurIPS 2025  
**arXiv**: [2512.10292](https://arxiv.org/abs/2512.10292)  
**代码**: 无  
**领域**: 博弈论 / 多智能体系统  
**关键词**: 凹博弈, 单调博弈, 平方和规划, Nash均衡, 不完美回忆博弈

## 一句话总结

证明了在多项式效用和半代数策略集的博弈中验证凹性和单调性是 NP-hard 的，并提出了两套基于平方和 (SOS) 规划的层次化认证方案，可在多项式时间内逐层求解。

## 研究背景与动机

多人博弈中，每个玩家独立选择动作以最大化自身收益，而收益取决于其他玩家的行动。博弈的结构性质——特别是凹性和单调性——是可计算性和均衡存在性的基石：

**凹博弈 (Concave Games)**：每个玩家的策略集是紧致凸集，收益函数关于自身动作是凹函数。在凹博弈中：
- Nash 均衡总是存在
- 分散式算法可以收敛到均衡

**单调博弈 (Monotone Games)**：凹博弈的一个加强，进一步要求博弈的伪梯度算子满足单调性条件。在单调博弈中：
- 在严格假设下 Nash 均衡唯一
- 提供更强的算法收敛保证

**核心问题**：给定一个博弈（其中效用函数是多元多项式，策略集是紧致凸基本半代数集），如何验证它是否是凹的或单调的？

**困难性**：本文首先证明了一个负面结果——即使在这一表达力丰富的类（足以刻画不完美回忆的扩展式博弈）中，认证凹性或单调性也是 NP-hard 的。

**正面结果**：尽管精确认证困难，作者发展了基于平方和 (Sum-of-Squares, SOS) 规划的层次化近似方案，提供了实用的多项式时间认证。

## 方法详解

### 整体框架

本文的技术路径如下：
1. 建立复杂度下界（NP-hardness 规约）
2. 构建凹性认证的 SOS 层次
3. 构建单调性认证的 SOS 层次
4. 引入 SOS-凹/SOS-单调博弈作为全局近似
5. 应用于不完美回忆扩展式博弈

### 关键设计

**1. NP-hardness 证明**：

通过从 co-NP-complete 问题（全局多项式非负性验证）进行规约。对于任意多元多项式 $p(x)$，构造一个博弈使得验证其凹性等价于验证 $p(x) \geq 0$ 对所有 $x \in S$ 成立（其中 $S$ 是某个半代数集）。

**2. 凹性认证的 SOS 层次**：

凹性要求每个玩家 $i$ 的效用 $u_i(x_i, x_{-i})$ 关于 $x_i$ 是凹函数（即 Hessian 半负定），对所有 $x_{-i}$。

第 $r$ 层 SOS 认证：验证是否存在 SOS 分解使得
$$-\nabla^2_{x_i} u_i(x_i, x_{-i}) = \Sigma_0(x) + \sum_j \Sigma_j(x) g_j(x)$$

其中 $\Sigma_k$ 是 SOS 多项式矩阵，$g_j$ 定义了策略集的半代数约束。层次等级 $r$ 限制了 $\Sigma_k$ 的阶数，更高的 $r$ 提供更强的认证能力。

**关键定理**：几乎所有凹博弈在有限层级 $r$ 被认证。即，未被认证的凹博弈在参数空间中构成零测集。

**3. 单调性认证的 SOS 层次**：

单调性要求博弈的伪梯度映射 $F(x) = (\nabla_{x_1} u_1, \ldots, \nabla_{x_n} u_n)$ 满足：
$$\langle F(x) - F(y), x - y \rangle \leq 0 \quad \forall x, y \in S$$

这等价于要求 Jacobian $JF(x)$ 对称部分半负定。类似地，通过 SOS 层次认证此条件。

**4. SOS-凹/单调博弈**：

对于未通过 SOS 认证的博弈，定义"最近的 SOS-凹/单调博弈"：
$$\min \|u - \tilde{u}\| \quad \text{s.t. } \tilde{u} \text{ 通过第 } r \text{ 层 SOS-凹/单调认证}$$

这个问题本身可以表述为半定规划 (SDP)，在多项式时间内求解。SOS-凹/单调博弈作为凹/单调博弈的全局近似，保留了均衡存在性和唯一性等良好性质。

### 损失函数 / 训练策略

本文为理论工作，不涉及传统意义上的损失函数和训练。核心计算工具是半定规划 (SDP)：
- 每一层 SOS 认证可被转化为一个大小随层级 $r$ 多项式增长的 SDP
- SDP 可通过内点法在多项式时间内求解
- 层级 $r$ 的选择决定了认证精度和计算成本的权衡

## 实验关键数据

### 主实验：SOS 层次认证能力

| 博弈类型 | 玩家数 | 多项式阶 | 策略集维度 | SOS 层级 $r$ | 是否成功认证凹性 | 是否成功认证单调性 |
|---------|-------|--------|----------|-------------|---------------|---------------|
| 二人二项式博弈 | 2 | 2 | 各1维 | $r=1$ | 是 | 是 |
| 三人二次博弈 | 3 | 2 | 各2维 | $r=1$ | 是 | 是 |
| 二人四次博弈 | 2 | 4 | 各2维 | $r=2$ | 是 | 是 |
| 不完美回忆扩展式博弈 | 2 | 多变 | 多变 | $r=1$-$3$ | 取决于博弈 | 取决于博弈 |
| 构造的 NP-hard 实例 | 2 | 高 | 高 | 有限 $r$ | 否（需 $r \to \infty$）| — |

### 消融实验：SOS 层级与认证成功率

| SOS 层级 $r$ | SDP 变量数量 | 计算时间（相对） | 凹性认证成功率 | 单调性认证成功率 |
|-------------|------------|--------------|-------------|---------------|
| $r = 0$ | 最少 | 1x | 仅二次凹函数 | 仅线性单调 |
| $r = 1$ | 中等 | ~5x | 大多数常见博弈 | 大多数常见博弈 |
| $r = 2$ | 较多 | ~25x | 覆盖更多高次多项式 | 覆盖更多情形 |
| $r = 3$ | 大量 | ~125x | 几乎所有凹博弈 | 几乎所有单调博弈 |
| $r \to \infty$ | 极大 | 指数 | 所有（理论极限） | 所有（理论极限） |

### 关键发现

1. **复杂度精确定位**：认证凹性和单调性的精确复杂度为 NP-hard，即使对于看似"简单"的博弈类也是如此
2. **SOS 层次的实用性**：在实际博弈（包括扩展式博弈）中，低层级（$r = 1$ 或 $r = 2$）通常已足够完成认证
3. **SOS-凹/单调博弈作为近似**：最近 SOS-凹/单调博弈的计算只需一个 SDP，且保留了原博弈的关键结构性质
4. **不完美回忆博弈的应用**：框架自然适用于不完美回忆扩展式博弈，这类博弈此前缺乏系统的凹性/单调性分析工具
5. **几乎所有凹博弈可被有限层认证**：这一密度结果为 SOS 层次的实用可靠性提供了强有力的理论保证

## 亮点与洞察

1. **负面+正面结果的完美结合**：先证明问题的固有困难性（NP-hard），再提供实用的近似解（SOS 层次）
2. **代数几何与博弈论的交叉**：SOS 规划作为连接代数几何和优化理论的桥梁，在博弈论中的应用令人耳目一新
3. **几乎确定的认证保证**：在参数空间中"几乎所有"凹博弈都能被有限层认证的结果既深刻又实用
4. **扩展式博弈的新分析工具**：为不完美回忆的扩展式博弈（信息经济学中的核心对象）提供了新的计算工具

## 局限与展望

1. **SDP 的可扩展性**：虽然每个 SDP 可在多项式时间求解，但变量数随层级 $r$ 和玩家数/策略维度快速增长
2. **密度结果的非构造性**：虽然"几乎所有"凹博弈可被认证，但无法预先确定需要多高的层级
3. **仅限多项式效用**：对于更一般的效用函数类（如分段线性、非多项式平滑函数），SOS 方法不直接适用
4. **单调性认证的实际需求**：在实际博弈求解中，严格单调性可能过于强，更实用的"弱单调"概念值得探索
5. **与算法设计的衔接**：认证结果如何直接指导均衡计算算法的选择和收敛速率分析是重要的后续方向

## 相关工作与启发

- **Rosen (1965)**：凹博弈理论的奠基性工作，证明了 Nash 均衡的存在性
- **Parrilo (2003)**：SOS 规划的系统化理论，本文方法的数学基础
- **Lasserre (2001)**：多项式优化的 SOS/SDP 层次，为本文层次化方案提供了蓝图
- **Koller & Megiddo (1992)**：不完美回忆扩展式博弈的计算复杂度研究

## 评分

| 维度 | 评分 (1-5) |
|------|-----------|
| 创新性 | 5 |
| 理论深度 | 5 |
| 实验充分性 | 3 |
| 写作质量 | 4 |
| 总评 | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Near-Optimal Quantum Algorithms for Computing (Coarse) Correlated Equilibria of General-Sum Games](near-optimal_quantum_algorithms_for_computing_coarse_correlated_equilibria_of_ge.md)
- [\[ICML 2025\] Solving Zero-Sum Convex Markov Games](../../ICML2025/reinforcement_learning/solving_zero-sum_convex_markov_games.md)
- [\[AAAI 2026\] Perturbing Best Responses in Zero-Sum Games](../../AAAI2026/reinforcement_learning/perturbing_best_responses_in_zero-sum_games.md)
- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](../../ICML2026/reinforcement_learning/global_policy-space_response_oracles_for_two-player_zero-sum_games.md)
- [\[NeurIPS 2025\] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](certifying_stability_of_reinforcement_learning_policies_using_generalized_lyapun.md)

</div>

<!-- RELATED:END -->
