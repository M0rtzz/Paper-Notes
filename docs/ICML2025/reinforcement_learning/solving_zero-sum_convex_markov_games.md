---
title: >-
  [论文解读] Solving Zero-Sum Convex Markov Games
description: >-
  [ICML 2025][convex Markov games] 本文首次为两人零和凸马尔可夫博弈（cMG）中的独立策略梯度方法提供了全局收敛到Nash均衡的理论保证，通过非凸正则化将问题化归为非凸-pPL min-max优化，并设计了嵌套/交替策略梯度算法。
tags:
  - ICML 2025
  - convex Markov games
  - Nash equilibrium
  - policy gradient
  - hidden convexity
  - nonconvex-pPL
---

# Solving Zero-Sum Convex Markov Games

**会议**: ICML 2025  
**arXiv**: [2506.16120](https://arxiv.org/abs/2506.16120)  
**代码**: 无  
**领域**: 强化学习 / 博弈论  
**关键词**: convex Markov games, Nash equilibrium, policy gradient, hidden convexity, nonconvex-pPL

## 一句话总结
本文首次为两人零和凸马尔可夫博弈（cMG）中的独立策略梯度方法提供了全局收敛到Nash均衡的理论保证，通过非凸正则化将问题化归为非凸-pPL min-max优化，并设计了嵌套/交替策略梯度算法。

## 研究背景与动机
**领域现状**: 多智能体强化学习（MARL）通常建模为马尔可夫博弈（MG），但传统MG要求效用函数可跨时间步加性分解。凸马尔可夫博弈（cMG）允许效用函数是状态-动作占用度量的凸函数，模型表达力更强，可涵盖创造性博弈策略发现、语言模型多步对齐、机器人多智能体探索等场景。

**现有痛点**: cMG打破了Bellman一致性——无法定义状态值函数和动作值函数，因而传统的基于值迭代/动态规划的MARL算法（如Q-learning变种）完全不适用。甚至Nash均衡的存在性证明都需要超越经典的Brouwer/Kakutani不动点定理。

**核心矛盾**: 策略空间上的效用函数天然非凸，而即使在最简单的正规形式博弈（cMG的单状态特例）中，梯度方法就会循环、产生混沌轨迹。非凸-非凹的鞍点计算在一般情况下是困难的。

**本文目标**: 策略梯度方法能否在零和cMG中收敛？

**切入角度**: 作者发现cMG中效用函数具有"隐凸性"（hidden convexity）——它是占用度量的凸函数复合一个可逆映射。利用这一结构，加上非凸正则化，可以使问题满足proximal Polyak-Łojasiewicz（pPL）条件。

**核心idea**: 通过占用度量空间的正则化把零和cMG化归为约束域上的非凸-pPL min-max优化问题，然后设计嵌套/交替梯度迭代保证收敛。

## 方法详解

### 整体框架
输入：两人零和cMG $\Gamma = (\mathcal{S}, \mathcal{A}, \mathcal{B}, \mathbb{P}, F, \gamma, \varrho)$，其中maximizer选策略 $x$，minimizer选策略 $y$，效用 $U(x,y) = F(\lambda_1(x,y), \lambda_2(x,y))$ 定义在占用度量上。输出：$\epsilon$-近似Nash均衡策略对。

方法分两步：(1) 通过正则化将cMG化归为NC-pPL min-max优化；(2) 针对NC-pPL问题设计收敛算法。

### 关键设计

1. **非凸正则化 (Hidden-Strongly-Convex Regularization)**:

    - 功能：在原始效用 $U(x,y)$ 上加一个关于maximizer占用度量的正则项 $-\frac{\mu}{2}\|\lambda_2(x,y)\|^2$，得到扰动效用 $U^\mu(x,y)$
    - 核心思路：$U$ 对每个玩家的占用度量是凹的（隐凹性），加上 $\mu$-强凸正则后变为隐强凹，从而满足pPL条件
    - 设计动机：pPL条件保证了最优响应映射 $y^\star(x)$ 关于 $x$ 是Lipschitz连续的（而非通常的 $\frac{1}{2}$-Hölder连续），这是策略梯度方法稳定迭代的关键

2. **嵌套策略梯度 (Nest-PG, Algorithm 1)**:

    - 功能：内层循环用projected gradient ascent逼近maximizer的最优响应，外层循环用projected gradient descent更新minimizer
    - 核心思路：利用NC-pPL结构，内层以线性速率收敛到近似最优响应，外层利用 $\Phi(x) = \max_y U^\mu(x,y)$ 的梯度主导性质保证下降
    - 与之前方法的区别：无需精确求解内层问题，允许不精确梯度，各玩家独立学习无需共享策略

3. **交替策略梯度 (Alt-PGDA, Algorithm 2)**:

    - 功能：两个玩家交替执行投影梯度步，但使用不对称步长（minimizer步长远小于maximizer步长）
    - 核心思路：通过时间尺度分离（step-size $\alpha_x \ll \alpha_y$），让maximizer的响应"追踪"minimizer的慢变化，同样利用pPL条件的Lipschitz最优响应性质保稳定
    - 设计动机：比嵌套方法更简单、更易实现，且同样对随机/不精确梯度鲁棒

### 损失函数 / 训练策略
目标是计算 $U^\mu$ 的鞍点：$\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} U^\mu(x,y)$。策略梯度通过REINFORCE式估计量（Definition 3）从采样轨迹中获得，需要 $\epsilon$-greedy策略保证充分探索。最终Nash均衡的近似误差由正则化强度 $\mu$ 控制。

## 实验关键数据

### 主实验（理论结果）
本文为纯理论工作，无数值实验。主结论为收敛复杂度：

| 算法 | 收敛到 $\epsilon$-NE 的迭代/样本复杂度 | 特点 |
|------|----------------------------------------|------|
| Nest-PG | $\text{poly}(1/\epsilon, |\mathcal{S}|, |\mathcal{A}|+|\mathcal{B}|, 1/(1-\gamma))$ | 嵌套循环，更强保证 |
| Alt-PGDA | $\text{poly}(1/\epsilon, |\mathcal{S}|, |\mathcal{A}|+|\mathcal{B}|, 1/(1-\gamma))$ | 单循环，更易实现 |

### 理论贡献对比

| 贡献 | 意义 |
|------|------|
| Best-response Lipschitz连续性 (Thm 4.1) | 首次证明在隐凸/NC-pPL情况下最优响应映射的Lipschitz性 |
| NC-pPL min-max全局收敛 (Thm 4.3) | 首个约束域上非凸-pPL函数的随机嵌套/交替GDA收敛保证 |
| cMG Nash均衡计算 (主定理) | 首个零和cMG的可证收敛独立策略梯度方法 |

### 关键发现
- 隐凸性 + 正则化 = pPL条件，这是统一分析的关键链条
- 两个算法都允许不精确梯度，这对独立学习至关重要（因为正则项依赖双方策略，精确梯度需要策略共享）
- 交替方法的步长比例需要精心设计，过大的比例会导致发散

## 亮点与洞察
- **隐凸性到pPL的桥梁**：这个归约思路非常巧妙——cMG效用函数看似非凸非凹，但通过占用度量视角发现其凸结构，再通过正则化激活这一结构得到梯度主导性质。这为其他具有隐凸结构的优化问题提供了方法论模板。
- **独立学习保证**：两个玩家无需交换策略信息，仅需各自估计梯度并更新，这符合MARL中去中心化学习的实际需求。
- **优化理论的独立贡献**：约束域上NC-pPL和双侧pPL的min-max收敛保证本身在优化理论中是新结果，可迁移到其他非凸-非凹min-max优化场景（如GAN训练的理论分析）。

## 局限与展望
- 纯理论工作，缺少实验验证；实际cMG场景中的数值表现未知
- 需要知道问题参数（Lipschitz常数、强凸参数等）来设置步长，实际中可能难以获取
- 复杂度关于状态/动作空间大小是多项式的，未考虑函数逼近（大规模场景）
- 正则化引入了额外的近似误差，$\mu \to 0$ 时收敛变慢

## 相关工作与启发
- **vs 传统MG算法 (Jin et al., 2021; Wei et al., 2021)**: 它们依赖值迭代和Bellman一致性，不适用于cMG；本文完全基于策略梯度绕开这些限制
- **vs cMDP优化 (Kalogiannis et al., 2024)**: 单智能体cMDP中已有隐凸性利用，但多智能体情况下占用度量耦合导致Hölder连续性降级；本文通过策略空间上的pPL条件恢复了Lipschitz性
- **vs Nesterov光滑化 (Nesterov, 2005)**: 本文方法可视为Nesterov非光滑最小化在隐凸非凸场景的推广

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次解决cMG中策略梯度收敛问题，隐凸性到pPL的归约思路原创性强
- 实验充分度: ⭐⭐ 纯理论工作，无实验验证
- 写作质量: ⭐⭐⭐⭐ 技术概览清晰，Section 1.1的技术路线图尤其有帮助
- 价值: ⭐⭐⭐⭐ 对MARL和min-max优化理论都有重要推进，开辟了cMG算法研究的新方向
- 总体: ⭐⭐⭐⭐ 扎实的理论贡献，为后续cMG实证研究奠定了基础

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Perturbing Best Responses in Zero-Sum Games](../../AAAI2026/reinforcement_learning/perturbing_best_responses_in_zero-sum_games.md)
- [Learning Utilities from Demonstrations in Markov Decision Processes](learning_utilities_from_demonstrations_in_markov_decision_processes.md)
- [Certifying Concavity and Monotonicity in Games via Sum-of-Squares Hierarchies](../../NeurIPS2025/reinforcement_learning/certifying_concavity_and_monotonicity_in_games_via_sum-of-squares_hierarchies.md)
- [Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination](cross-environment_cooperation_enables_zero-shot_multi-agent_coordination.md)
- [Pessimism Principle Can Be Effective: Towards a Framework for Zero-Shot Transfer RL](pessimism_principle_can_be_effective_towards_a_framework_for_zero-shot_transfer_.md)

<!-- RELATED:END -->
