---
title: >-
  [论文解读] Perturbing Best Responses in Zero-Sum Games
description: >-
  [AAAI 2026][强化学习][零和博弈] 本文研究在零和博弈的最优响应预言机（BRO）中引入随机扰动，证明了随机虚拟对弈（SFP）在纯策略数量 $n$ 上可实现 $O(\frac{\log n}{\varepsilon^2})$ 的期望迭代次数，并提出了随机双重预言机（SDO）算法，在特定博弈结构下同样实现对数级收敛。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "零和博弈"
  - "Nash均衡"
  - "扰动最优响应"
  - "虚拟对弈"
  - "双重预言机"
---

# Perturbing Best Responses in Zero-Sum Games

**会议**: AAAI 2026  
**arXiv**: [2511.12523](https://arxiv.org/abs/2511.12523)  
**代码**: [github](https://github.com/geoborek/perturbing-best-responses)  
**领域**: 强化学习  
**关键词**: 零和博弈, Nash均衡, 扰动最优响应, 虚拟对弈, 双重预言机

## 一句话总结

本文研究在零和博弈的最优响应预言机（BRO）中引入随机扰动，证明了随机虚拟对弈（SFP）在纯策略数量 $n$ 上可实现 $O(\frac{\log n}{\varepsilon^2})$ 的期望迭代次数，并提出了随机双重预言机（SDO）算法，在特定博弈结构下同样实现对数级收敛。

## 研究背景与动机

在大规模两人零和博弈中计算Nash均衡（NE）是一个核心的计算挑战。虚拟对弈（FP）和双重预言机（DO）是两种基于最优响应预言机的经典算法，但它们在最坏情况下需要 $\Omega(n)$ 次迭代才能找到 $\varepsilon$-NE。理论上已知任何对数大小的 $\varepsilon$-NE 都存在（Althöfer 1994），但 Hazan & Koren 2016 证明了任何随机化BRO算法至少需要 $\Omega(\sqrt{n}/\log^3 n)$ 次迭代。

**核心动机**：为了突破BRO的下界限制，作者提出"增强"BRO的能力——通过在计算最优响应之前对效用值添加随机扰动。这种扰动最优响应预言机（PBRO）可以理解为允许算法以一定概率探索非最优但潜在有价值的策略，从而避免陷入确定性算法的线性迭代陷阱。

**关键观察**：确定性DO和FP之所以慢，是因为它们可能被对手设计的对抗性最优响应所引导，逐步遍历所有纯策略。扰动打破了这种确定性模式，使得算法有机会直接"跳到"关键策略。

## 方法详解

### 整体框架

本文提出两种随机化算法变体：
1. **随机虚拟对弈（SFP）**：在FP的每次迭代中，使用扰动后的最优响应替代确定性最优响应
2. **随机双重预言机（SDO）**：在DO的每次迭代中，使用扰动后的最优响应来扩展策略子集

两种算法的终止条件不受扰动影响——仍然使用非扰动的最优响应值来判断是否达到 $\varepsilon$-NE。

### 关键设计

#### 1. 扰动最优响应（PBRO）

核心思路是在计算最优响应前，对效用向量的每个分量添加独立同分布的随机噪声：

$$\widetilde{\text{BR}}_r(\mathbf{q}) \in \arg\min_{i \in [m]} \pi_i(\mathbf{M}\mathbf{q} - \mathbf{u})$$

其中 $\mathbf{u}$ 是随机扰动向量。关键的数学联系：当扰动服从 Gumbel 分布 $G(0, \beta)$ 时，由 Gumbel-max trick（Lemma 1），扰动最优响应等价于从 $\text{softmax}(\mathbf{M}\mathbf{q}/\beta)$ 中采样。这一联系使得SFP与随机指数加权预测器（REWF）建立了等价关系。

**设计动机**：Gumbel扰动自然地引入了"以概率正比于效用质量来选择策略"的机制，而非总是选择绝对最优响应。参数 $\beta$ 控制探索-利用的平衡。

#### 2. SFP的对数收敛性证明

**定理3**：对于矩阵博弈 $\mathbf{M} \in [0,1]^{m \times n}$ 且 $m \leq n$，使用 Gumbel 扰动的SFP可在 $O(\frac{\log n}{\varepsilon^2})$ 期望迭代次数内计算 $\varepsilon$-NE。

证明思路：
- 将SFP与Gumbel扰动对应到双方都使用REWF的场景
- 利用REWF的后悔界（Corollary 2），结合参数选择 $\eta = \sqrt{8\ln n / T}$ 和 $\beta = \frac{2 + \sqrt{2\ln n}}{\varepsilon\sqrt{8\ln n}}$
- 证明在 $T = (\frac{2+\sqrt{2\ln n}}{\varepsilon})^2$ 次迭代内，双方后悔之和以至少1/2的概率不超过 $\varepsilon$
- 通过重复执行获得对数期望迭代次数

#### 3. SDO在特定博弈上的对数收敛

对于两类Zhang & Sandholm (2024)提出的困难博弈实例：

**Example 1（矩阵$\mathbf{L}$）**：双方选择数字，选大者赢。确定性DO需 $n$ 次迭代，但SDO利用均匀扰动，在每次迭代中将最小索引的期望减半（Lemma 4），通过漂移分析（drift theory）证明 $O(\log n)$ 收敛（定理6）。

**Example 2（矩阵$\mathbf{U}$）**：$\mathbf{L}$ 的变体，最优响应唯一。扰动使期望索引不超过 $\frac{3}{4}k$（Lemma 8），同样实现 $O(\log n)$ 收敛（定理7）。

#### 4. 高效扰动机制

对于具有内部结构的博弈（如POSG），直接扰动整个效用矩阵不可行。本文提出"聚类扰动"：将终端状态分簉，对每个簇施加相同的随机扰动值：

$$\widetilde{\text{BR}}_r(\mathbf{q}) = \arg\min_{i \in [m]} \pi_i\left(\left(\mathbf{L} + \sum_{k=1}^{K} z_k \mathbf{B}_k\right)\mathbf{q}\right)$$

其中 $\mathbf{B}_k$ 是第 $k$ 个簇的掩码矩阵。这使得扰动复杂度从 $O(n^2)$ 降至 $O(K)$，$K$ 为终端状态数。

### 损失函数 / 训练策略

本文是理论与算法设计工作，不涉及深度学习训练。关键的算法策略包括：
- SFP使用Gumbel扰动参数 $\beta = \frac{2+\sqrt{2\ln n}}{\varepsilon\sqrt{8\ln n}}$
- SDO使用均匀扰动 $\mathcal{U}(-1/2, 1/2)$ 或 $\mathcal{U}(-1, 1)$
- 终止条件：$\text{BRVal}_c(\mathbf{p}) - \text{BRVal}_r(\mathbf{q}) \leq \varepsilon$

## 实验关键数据

### 主实验

SFP实验比较（$\varepsilon=0.1$）：

| 博弈类型 | FP | AFP | SFP | SAFP |
|---------|-----|-----|-----|------|
| 随机矩阵 $(n=200)$ | ~200 | ~80 | ~200 | ~200 |
| 矩阵 $\mathbf{U}^\top$ $(n=200)$ | 200 (线性) | 200 (线性) | ~25 (对数) | ~25 (对数) |
| Morra博弈 $(f=15,\ n=225)$ | >200 | >200 | ~35 | ~35 |

SDO实验比较（$\varepsilon=0.1$）：

| 博弈类型 | DO | SDO |
|---------|-----|-----|
| 矩阵 $\mathbf{L}$ $(n=200)$ | 200 | ~15 |
| 矩阵 $\mathbf{U}^\top$ $(n=200)$ | 200 | ~20 |
| Morra博弈 $(f=15)$ | ~150 | ~25 |
| Colonel Blotto (5战场) | >100 | ~20 |

### 消融实验

| 扰动类型 | 矩阵 $\mathbf{L}$ (SDO) | 矩阵 $\mathbf{U}^\top$ (SDO) | 说明 |
|---------|-------------|-------------|------|
| 无扰动 (DO) | $n$ 次 | $n$ 次 | 线性最坏情况 |
| 均匀扰动 $\mathcal{U}(-1,1)$ | ~$\log n$ | ~$\log n$ | 对数级收敛 |
| 高效聚类扰动 | 稍高于直接扰动 | 稍高于直接扰动 | 仍远优于DO |
| 随机矩阵博弈 | 无改善或退化 | - | 扰动对随机博弈无效 |

### 关键发现

1. **SFP的对数收敛**：首次证明SFP在纯策略维度上实现对数迭代复杂度，远优于FP的 $O(n)$
2. **SDO的局部有效性**：在特定结构化博弈上SDO实现对数收敛，但一般情况下的复杂度仍是开放问题
3. **随机博弈的负面结果**：在随机矩阵博弈上扰动不仅不加速，甚至可能退化——因为随机博弈的最优响应本身就具有随机性
4. **路径规划博弈的验证**：在具有内部结构的路径规划博弈上，高效扰动同样能减少迭代次数

## 亮点与洞察

- **Gumbel-softmax与博弈论的精妙联系**：通过Gumbel-max trick将PBRO与REWF建立等价关系，从而借用在线学习的后悔界来证明博弈论算法的收敛性
- **漂移分析的巧妙应用**：利用进化计算中的漂移理论来分析SDO的收敛，展示了跨领域技术迁移的力量
- **实用性考量**：高效扰动方案仅需扰动终端状态奖励，可以直接集成到现有的POSG求解器中
- **理论严谨性**：所有主要结果都有完整证明，不依赖渐近假设

## 局限与展望

1. **SDO一般复杂度未解**：SDO在任意矩阵博弈上是否总能实现亚线性甚至对数级迭代仍是开放问题
2. **随机博弈的失效**：扰动对随机矩阵博弈无效，说明方法对博弈结构有依赖
3. **扰动参数选择**：$\beta$ 的最优选择依赖于博弈的归一化范围，实际应用中可能需要调参
4. **高效扰动的局限**：聚类扰动与直接效用扰动不完全等价，理论保证有gap
5. **实验规模有限**：最大测试到 $n \approx 1000$，更大规模博弈未验证

## 相关工作与启发

- 与PSRO（Policy Space Response Oracles）系列工作密切相关，可将扰动思想应用于多智能体强化学习
- 正则化AFP（PU和OMWU）虽然也能实现 $O(\log n)$，但需要维护所有策略的概率分布，不适用于大博弈
- 启发：扰动思想可推广到序贯博弈求解、多人博弈等更广泛的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 扰动BRO本身不新，但对数复杂度证明和高效扰动方案有显著贡献
- 实验充分度: ⭐⭐⭐ — 理论验证充分，但缺乏真正大规模应用场景的实验
- 写作质量: ⭐⭐⭐⭐⭐ — 数学严谨，逻辑清晰，证明完整
- 价值: ⭐⭐⭐⭐ — 解决了博弈论中的重要理论问题，对多智能体RL有潜在启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](../../ICML2026/reinforcement_learning/global_policy-space_response_oracles_for_two-player_zero-sum_games.md)
- [\[ICML 2026\] Bilevel Optimization over Saddle Points of Zero-Sum Markov Games](../../ICML2026/reinforcement_learning/bilevel_optimization_over_saddle_points_of_zero-sum_markov_games.md)
- [\[ICML 2025\] Solving Zero-Sum Convex Markov Games](../../ICML2025/reinforcement_learning/solving_zero-sum_convex_markov_games.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](../../ICLR2026/reinforcement_learning/spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[AAAI 2026\] Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits](beyond_the_lower_bound_bridging_regret_minimization_and_best_arm_identification_.md)

</div>

<!-- RELATED:END -->
