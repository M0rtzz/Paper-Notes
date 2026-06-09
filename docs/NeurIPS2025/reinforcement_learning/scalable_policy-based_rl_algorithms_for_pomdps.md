---
title: >-
  [论文解读] Scalable Policy-Based RL Algorithms for POMDPs
description: >-
  [NeurIPS 2025][强化学习][POMDP] 提出将 POMDP 近似为有限状态的 Superstate MDP（状态为截断历史），给出更紧的最优值函数差上界（随历史长度指数衰减），并首次证明标准 TD 学习 + 策略优化在此非马尔可夫采样下的有限时间收敛保证。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "POMDP"
  - "Policy Optimization"
  - "TD Learning"
  - "Superstate MDP"
  - "Linear Function Approximation"
---

# Scalable Policy-Based RL Algorithms for POMDPs

**会议**: NeurIPS 2025  
**arXiv**: [2510.06540](https://arxiv.org/abs/2510.06540)  
**代码**: 无  
**领域**: 强化学习 / 理论  
**关键词**: POMDP, Policy Optimization, TD Learning, Superstate MDP, Linear Function Approximation

## 一句话总结

提出将 POMDP 近似为有限状态的 Superstate MDP（状态为截断历史），给出更紧的最优值函数差上界（随历史长度指数衰减），并首次证明标准 TD 学习 + 策略优化在此非马尔可夫采样下的有限时间收敛保证。

## 研究背景与动机

部分可观测马尔可夫决策过程（POMDP）是建模不确定性下决策的通用框架，广泛应用于自动驾驶、医疗诊断、博弈等领域。然而 POMDP 的求解面临根本困难：

**核心矛盾**：POMDP 可通过信念状态（belief state）转化为全观测 MDP，但信念状态是连续分布且随历史长度增长，导致 PSPACE-完全的计算复杂性。

**已有方法的局限**：
- 基于有限历史的方法（Loch & Singh, Littman）在实践中有效，但缺乏严格的理论性能保证
- Kara & Yüksel (2023) 提出 Superstate MDP 将 POMDP 近似为有限状态 MDP，但其值函数差界为 $O(1/(1-\gamma)^2)$（过松）
- Cayci et al. (2024) 使用计算开销更大的 $m$-step TD learning 作为变通方案
- Q-learning 方法面临线性函数近似下的收敛困难

**核心研究问题**：能否用标准 RL 算法（TD learning + Policy Optimization）直接近似求解 POMDP，并建立理论保证？

## 方法详解

### 整体框架

1. **POMDP → Superstate MDP 转换**：给定截断长度 $l$，将历史 $H_t$ 截断为最近 $l$ 个 (action, observation) 对，构成 Superstate $\mathcal{G}(H_t)$
2. **近似质量分析**：证明 Superstate MDP 的最优值函数与原 POMDP 的最优值函数差随 $l$ 指数衰减
3. **Policy Optimization 算法**：交替执行 TD 学习（策略评估）和指数策略更新（基于 POLITEX）

### 关键设计

1. **改进的近似保证（Theorem 2）**：
   在均匀滤波器稳定性条件（Assumption 1）下：
    $\|V^*({\pi}(H)) - \tilde{V}(\mathcal{G}(H))\|_\infty \leq \frac{2\bar{r}(1-\rho)^l}{1-\gamma} + \frac{2\bar{r}\gamma(1-\rho)^l}{(1-\gamma)((1-\gamma)+\gamma(1-\rho)^l)}$
   
   **关键改进**：先前工作的界与 $(1-\gamma)^{-2}$ 或 $(1-\gamma)^{-3}$ 成正比，本文的界在 $(1-\rho)^l$ 足够小时近似为 $\frac{2\bar{r}(1-\rho)^l}{1-\gamma}$，将 horizon 依赖从平方降至线性。且这是最坏情况界而非期望界。

2. **关键代数引理（Lemma 2）**：
   对正向量 $\mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d}$，当 $\sum a_i = \sum c_i = 1$ 时：
    $|\sum a_i b_i - \sum c_i d_i| \leq \frac{\|a-c\|_1}{2}\max(\|b\|_\infty, \|d\|_\infty) + \|b-d\|_\infty - \frac{\|a-c\|_1}{4}\|b-d\|_\infty$
   
   传统方法使用分解 + 三角不等式导致松弛，本文的不等式利用概率质量函数的归一化性质获得更紧的上界。这一引理具有独立价值，也可改进 Subramanian & Mahajan (2019) 的结果。

3. **标准 TD Learning 在非马尔可夫采样下的收敛（Lemma 4）**：
   核心难点在于采样来自真实 POMDP 的信念状态 $\pi(H_t)$，而非 Superstate MDP 的状态 $\mathcal{G}(H_t)$。作者证明在滤波器稳定性条件下，Superstate MDP 的转移矩阵具有收缩性质（Lemma 3），使得标准 TD 学习尽管在非马尔可夫环境中运行，仍能收敛到 Superstate MDP 的值函数近似。

### 损失函数 / 训练策略

**Algorithm 2**：Policy Optimization for Superstate MDP
- 外层循环（$M$ 次策略更新）：使用 POLITEX 的指数更新规则 $\mu_i(a|B) \propto \exp(\eta \sum_{j=1}^i \bar{Q}^{\mu_{j-1}}(B,a))$
- 内层循环（$\tau + l'$ 步 TD 学习）：标准 SARSA 更新 + 投影到半径 $R$ 的球
- 前 $l'$ 步作为燃烧期（burn-in），等待滤波器稳定
- 支持线性函数近似：$Q(B,a) = \phi^T(B,a)\theta$

## 实验关键数据

本文为纯理论工作，核心结果以定理呈现。

### 主结果（Theorem 3，Regret 界）

| 组成部分 | 描述 | 阶 |
|----------|------|----|
| $\xi_{\text{FA}}$ | 函数近似误差 | 取决于特征质量 |
| $\xi_{\text{HA}}$ | 历史近似误差 | $\sim (1-\rho)^l$，随 $l$ 指数衰减 |
| 主项 | regret 收敛速率 | $O(T^{3/4}\log T)$ |

总 regret：$\mathcal{R}_T \leq T \cdot (\xi_{\text{FA}} + \xi_{\text{HA}}) + O(T^{3/4}\log T)$

### 与先前工作对比

| 方法 | 近似界 | 算法 | 计算开销 | 假设 |
|------|--------|------|----------|------|
| Kara & Yüksel (2023) | $O(1/(1-\gamma)^2)$ | Q-learning | 标准 | 遍历性+渐近收敛 |
| Cayci et al. (2024) | 多项式 | m-step TD | **高** | 类似 |
| Abel et al. (2016) | $O(1/(1-\gamma)^3)$ | - | - | - |
| **本文** | $O((1-\rho)^l/(1-\gamma))$ | **标准 TD + PO** | **标准** | 滤波器稳定性 |

### 关键发现

- 标准 TD learning 首次被证明可以直接应用于真实动态为 POMDP 的场景并获得收敛保证
- 不需要 Cayci et al. 的 $m$-step TD 变体，避免了额外计算开销
- 线性函数近似的引入使方法可扩展到大状态空间
- 代数引理（Lemma 2）本身可用于改进其他POMDP近似分析

## 亮点与洞察

- **实用性**：证明了"直觉正确"的做法（将有限历史当 MDP 状态用标准 RL 算法求解）确实有理论保证
- **代数引理的通用性**：Lemma 2 对形如 $|\sum a_ib_i - \sum c_id_i|$ 的界提供了比三角不等式更紧的控制，可广泛应用
- **分离关注点**：值函数误差被分解为函数近似误差 $\xi_{\text{FA}}$（可通过更好的特征改善）和历史近似误差 $\xi_{\text{HA}}$（可通过增大 $l$ 指数衰减），这种分解使调参有明确方向

## 局限与展望

- 滤波器稳定性条件（Assumption 1）要求转移和观测核具有足够混合性，在某些确定性环境下不满足
- $O(T^{3/4}\log T)$ 的 regret 速率可能不是最优的
- 指数增长的 Superstate 空间 ($|\mathcal{Y}|^l \cdot |\mathcal{A}|^l$) 在 $l$ 较大时仍面临可扩展性挑战
- 未提供数值实验验证理论界的紧致性
- 未来可探索用 LSTM 或 Transformer 替代线性函数近似来获得更好表达能力

## 相关工作与启发

- 在 Kara & Yüksel (2023) 的 Superstate MDP 基础上显著改进了近似界和算法保证
- POLITEX 算法从平均奖励推广到折扣奖励设定
- 滤波器稳定性理论（van Handel, 2008）为信念状态收缩提供了核心工具
- 对 Subramanian & Mahajan (2019) 的近似信息状态方法提供了改进的性能界

## 评分

- 新颖性: ⭐⭐⭐⭐ 代数引理新颖实用，标准TD在POMDP下的收敛分析是重要理论突破
- 实验充分度: ⭐⭐⭐ 纯理论工作，无实验，但定理陈述清晰完整
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清楚，与先前工作的对比详尽
- 价值: ⭐⭐⭐⭐ 为POMDP的可扩展求解提供了理论基石，对实际RL系统设计有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)
- [\[NeurIPS 2025\] Horizon Reduction Makes RL Scalable](horizon_reduction_makes_rl_scalable.md)
- [\[NeurIPS 2025\] Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](../../ICLR2026/reinforcement_learning/sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[NeurIPS 2025\] Parameter-Free Algorithms for the Stochastically Extended Adversarial Model](parameter-free_algorithms_for_the_stochastically_extended_adversarial_model.md)

</div>

<!-- RELATED:END -->
