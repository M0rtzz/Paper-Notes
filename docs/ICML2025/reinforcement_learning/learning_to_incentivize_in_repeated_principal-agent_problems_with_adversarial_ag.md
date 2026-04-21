---
title: >-
  [论文解读] Learning to Incentivize in Repeated Principal-Agent Problems with Adversarial Agent Arrivals
description: >-
  [ICML2025][principal-agent] 首次研究 agent 以对抗顺序到达的重复 principal-agent 问题，在 greedy 和 smooth 两种响应模型下分别给出了紧的 regret 上下界，核心思路是将激励设计问题规约为对抗线性 bandit。
tags:
  - ICML2025
  - principal-agent
  - adversarial bandits
  - incentive design
  - regret bound
  - mechanism design
---

# Learning to Incentivize in Repeated Principal-Agent Problems with Adversarial Agent Arrivals

**会议**: ICML2025  
**arXiv**: [2505.23124](https://arxiv.org/abs/2505.23124)  
**代码**: 无  
**领域**: reinforcement_learning / 博弈论  
**关键词**: principal-agent, adversarial bandits, incentive design, regret bound, mechanism design

## 一句话总结

首次研究 agent 以对抗顺序到达的重复 principal-agent 问题，在 greedy 和 smooth 两种响应模型下分别给出了紧的 regret 上下界，核心思路是将激励设计问题规约为对抗线性 bandit。

## 研究背景与动机

重复 principal-agent 模型刻画了一类 principal 通过激励引导 agent 行为的序贯决策场景，广泛出现在电商折扣、保险合约、众包定价等场景中。已有工作要么假设 principal 反复面对同一类型的 agent，要么假设 agent 类型从固定分布中随机采样。然而现实中（如网购用户的响应序列、众包工人的羊群行为），agent 的到达顺序往往是非随机的。

本文首次研究 **adversarial agent arrivals** 设定：在 $T$ 轮交互中，$K \geq 2$ 种类型的 agent 以对抗顺序到达，principal 事先不知道每轮到达的 agent 类型。principal 在每轮选择激励向量 $\pi_t \in \mathcal{D} \subseteq [0,1]^N$，agent 根据自身偏好和激励选择 $N$ 个 arm 之一，principal 获得收益 $U(\pi_t, j_t) = v_{a(\pi_t,j_t)} - \pi_{t,a(\pi_t,j_t)}$（奖励减去激励成本）。目标是最小化对最优固定激励的 regret：

$$R_T = \sup_{\pi \in \mathcal{D}} \mathbb{E}\left[\sum_{t=1}^T \big(U(\pi, j_t) - U(\pi_t, j_t)\big)\right]$$

## 方法详解

### 两种 Agent 响应模型

**Greedy 选择模型**：agent 确定性地选择效用最大的 arm $b(\pi,j) \in \arg\max_{i} (\mu_i^j + \pi_i)$。此模型下激励的微小变化可能导致 agent 决策发生剧烈跳变。

**Smooth 选择模型**：agent 的选择概率关于激励满足 Lipschitz 条件：

$$\sum_{i=1}^N \big|\Pr[a(\pi,j)=i] - \Pr[a(\pi',j)=i]\big| \leq L \cdot \|\pi - \pi'\|_\infty$$

这可以看作 agent 在偏好向量上加入平滑噪声（如 Gumbel → Logit 模型，Gaussian → Lipschitz 模型）。

### Greedy 模型 — 不可行性结果

**定理 2.1**：若 principal 不知道 agent 的 best response 函数 $b(\cdot,\cdot)$，则任何算法在 $K=2, N=3$ 时均有 $R_T = \Omega(T)$（线性 regret）。直觉是最优激励值 $\Delta$ 的精确学习不可行。

### Greedy 模型 + 已知响应 — 离散化 + 线性 Bandit 规约

**核心思路**：

1. **离散化**：对每个 arm $i$ 和 agent 类型 $j$，构造恰好使 agent $j$ 选择 arm $i$ 的最小激励 $\pi^{i,j}$，得到 $|\Pi| = O(NK)$ 大小的有限激励集。进一步通过映射 $h: \Pi \to \{0,1\}^K$ 合并等价激励，缩减到 $|\hat\Pi| \leq \min(2KN, 2^K) + 1$。

2. **规约到对抗线性 Bandit**：将每个激励 $\pi$ 映射到 $K$ 维向量 $z^\pi$（第 $j$ 分量为 $U(\pi,j)$），每轮奖励向量 $y_t = e_{j_t}$：

$$U(\pi, j_t) = \langle z^\pi, y_t \rangle$$

直接利用 Exp3-for-linear-bandits 算法求解。

### Greedy 模型 + General 激励

当 principal 可同时激励多个 arm 时，决策空间 $\mathcal{D} = [0,1]^N$。通过识别行为等价的多面体 $\mathcal{P}_\sigma$ 及其极端点进行离散化，再结合 covering number 技巧缩减 arm 集大小至 $(6KT)^K$。

### Smooth 模型 — 均匀网格 + Tsallis-INF

对单臂激励做 $\epsilon$-网格离散化得到 $N/\epsilon$ 个 arm，运行 Tsallis-INF。离散化误差受 Lipschitz 常数控制：

$$\mathbb{E}[U(\pi^*, j)] - \mathbb{E}[U(\hat\pi, j)] \leq (2L+1)\epsilon$$

总 regret 为 $O(\sqrt{N\epsilon^{-1}T} + T(2L+1)\epsilon)$，令 $\epsilon = N^{1/3}(2L+1)^{-2/3}T^{-1/3}$ 即得最优界。

## 理论结果汇总

| Agent 行为 | 激励类型 | Upper Bound | Lower Bound |
|:---|:---|:---|:---|
| 未知 Greedy | 单臂 | N/A | $\Omega(T)$ |
| 已知 Greedy | 单臂 | $\tilde{O}(\min\{\sqrt{KT\log N},\, K\sqrt{T}\})$ | $\tilde{\Omega}(\min\{\sqrt{KT\log N},\, K\sqrt{T}\})$ |
| 已知 Greedy | 一般 | $\tilde{O}(K\sqrt{T\log(KT)})$ | — |
| 未知 Smooth | 单臂 | $\tilde{O}(L^{1/3}N^{1/3}T^{2/3})$ | $\Omega(L^{1/3}N^{1/3}T^{2/3})$ |
| 未知 Smooth | 一般 | $\tilde{O}(L^{N/(N+2)}T^{(N+1)/(N+2)})$ | — |

**关键发现**：

- Greedy 模型在单臂激励下，上下界在 $\log K$ 因子内匹配——**紧的**
- Smooth 模型在单臂激励下，上下界在多项对数因子内匹配——**紧的**
- 从 $K\sqrt{T}$ 到 $\sqrt{KT\log N}$ 的 min 结构反映了 agent 类型数与 arm 数的 tradeoff

## 亮点与洞察

1. **新问题提出**：首次系统研究 adversarial agent arrival 的 principal-agent 问题，填补了已有文献仅考虑固定或随机到达的空白
2. **不可行性刻画**：证明没有先验知识时 greedy agent 导致线性 regret，揭示了激励设计中"非平滑敏感性"的根本困难
3. **统一规约框架**：将不同设定下的激励设计问题统一规约到对抗线性 bandit，技术上干净利落
4. **紧的界**：在 greedy（已知）和 smooth 两种主要设定下均给出匹配的上下界，理论完备性强
5. **实用性桥梁**：smooth 模型可通过 Gumbel/Gaussian 噪声自然衍生自 greedy 模型，为理论到实践提供了平滑过渡

## 局限与展望

1. **纯理论工作**：没有任何实验或模拟验证，实际场景中的效果未知
2. **agent 无策略性**：假设 agent 是 myopic 的，不考虑重复博弈中 agent 可能的策略性行为（如虚报偏好）
3. **General 激励缺下界**：greedy + general 和 smooth + general 两种设定均缺少下界，紧性未知
4. **Smooth 模型需已知 $L$**：算法需要 Lipschitz 常数 $L$ 作为输入，未知 $L$ 时离散化粒度无法确定（论文指出这会导致线性依赖 $L$）
5. **离线最优基准**：regret 定义比较的是最优固定激励，更强的 dynamic regret 或与最优策略序列的比较未讨论
6. **可扩展性**：greedy + general 设定的离散化大小为 $(6KT)^K$，当 $K$ 增大时计算不可行

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 全新问题设定，adversarial arrival 在 PA 框架中首次研究
- 理论深度: ⭐⭐⭐⭐⭐ — 多个设定的匹配上下界，分析技巧精致
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机例子直观，但符号较重
- 实用价值: ⭐⭐⭐ — 纯理论贡献，距实际部署有差距

<!-- RELATED:START -->

## 相关论文

- [Principal-Agent Bandit Games with Self-Interested and Exploratory Learning Agents](principal-agent_bandit_games_with_self-interested_and_exploratory_learning_agent.md)
- [Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)
- [Learning Progress Driven Multi-Agent Curriculum](learning_progress_driven_multi-agent_curriculum.md)
- [Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination](cross-environment_cooperation_enables_zero-shot_multi-agent_coordination.md)

<!-- RELATED:END -->
