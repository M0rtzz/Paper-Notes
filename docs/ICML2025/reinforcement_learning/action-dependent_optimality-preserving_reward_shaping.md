---
title: >-
  [论文解读] Action-Dependent Optimality-Preserving Reward Shaping (ADOPS)
description: >-
  [ICML 2025][reward shaping] 提出ADOPS方法，通过查询critic网络的外在/内在值函数估计，仅在内在奖励会改变最优动作偏好时调整奖励，从而实现action-dependent的optimality-preserving reward shaping，突破了PBRS只能处理action-independent形式的限制，在Montezuma's Revenge上超越所有先前的optimality-preserving方法和baseline RND。
tags:
  - ICML 2025
  - reward shaping
  - intrinsic motivation
  - optimality preservation
  - PBRS
  - exploration
  - Montezuma's Revenge
---

# Action-Dependent Optimality-Preserving Reward Shaping (ADOPS)

**会议**: ICML 2025  
**arXiv**: [2505.12611](https://arxiv.org/abs/2505.12611)  
**代码**: https://github.com/alirezakazemipour/PPO-RND (基于此实现)  
**领域**: reinforcement_learning  
**关键词**: reward shaping, intrinsic motivation, optimality preservation, PBRS, exploration, Montezuma's Revenge

## 一句话总结

提出ADOPS方法，通过查询critic网络的外在/内在值函数估计，仅在内在奖励会改变最优动作偏好时调整奖励，从而实现action-dependent的optimality-preserving reward shaping，突破了PBRS只能处理action-independent形式的限制，在Montezuma's Revenge上超越所有先前的optimality-preserving方法和baseline RND。

## 研究背景与动机

在稀疏奖励环境中，内在动机（Intrinsic Motivation, IM）如RND、ICM等被广泛用于鼓励探索。然而IM存在严重的**reward hacking**问题——智能体可能优化内在奖励而牺牲外在奖励，导致次优策略。经典案例包括"noisy TV problem"和Montezuma's Revenge中的"dancing with skulls"现象。

已有方法试图解决这一问题：

- **PBRS（Potential-Based Reward Shaping）**：通过势函数差的形式保证不改变最优策略集。Ng et al. (1999) 提出经典形式 $F = \gamma\Phi(s') - \Phi(s)$
- **PBIM**：将PBRS扩展到有限时间步的IM场景，但需要在episode末尾补偿所有累积内在奖励
- **GRM（Generalized Reward Matching）**：PBIM的推广，通过matching function在未来时间步补偿内在奖励
- **PIES（Policy-Invariant Explicit Shaping）**：通过线性衰减IM的权重系数来保证后期策略最优

**核心问题**：这些方法在**复杂、长episode的稀疏奖励环境**中全部失败。PBIM因episode长度导致末尾惩罚项指数爆炸（$1/\gamma^{N-1} \approx 10^{19}$）；GRM面临类似延迟补偿导致的数值问题；PIES在后半程训练中不再提供IM，性能急剧下降。

## 方法详解

### 整体框架

ADOPS的核心思想是：不去要求内在回报的累积值与动作无关（PBRS的做法），而是**主动检测内在奖励是否会改变动作的最优性偏好**，仅在会导致偏好翻转时才调整内在奖励。

具体地，给定任意初始shaping reward $F$，ADOPS将其转换为 $F' = F + F_2$，其中 $F_2$ 是一个修正项。ADOPS利用agent已有的critic网络分别估计外在和内在的值函数（$\hat{V}_E^\pi, \hat{V}_I^\pi, \hat{Q}_E^\pi, \hat{Q}_I^\pi$），在每个时间步判断：

1. 如果当前动作在外在意义上是次优的（$Q_E^\pi < V_E^\pi$），确保加上IM后它仍然是次优的
2. 如果当前动作在外在意义上是最优的（$Q_E^\pi \geq V_E^\pi$），确保加上IM后它仍然是最优的

### 关键设计1：ADOPS奖励修正公式

ADOPS的核心公式（实用版本，使用当前策略的估计而非最优策略）：

$$F_2 = \begin{cases} \min(0, V_E^\pi - Q_E^\pi + V_I^\pi - \gamma V_I^\pi(s') - F - \epsilon) & \text{if } Q_E^\pi < V_E^\pi \\\\ \max(0, V_E^\pi - Q_E^\pi + V_I^\pi - \gamma V_I^\pi(s') - F) & \text{if } Q_E^\pi \geq V_E^\pi \end{cases}$$

其中 $\epsilon$ 是一个极小正常数（实验中取 $10^{-7}$）。

定义辅助量 $\Omega = V_E^\pi - Q_E^\pi + V_I^\pi - \gamma V_I^\pi(s') - F$，以及三个指示函数：

- $C_1 = \mathbf{1}(Q_E^\pi < V_E^\pi \land \Omega > 0)$：次优动作但IM使其看起来更好
- $C_2 = \mathbf{1}(Q_E^\pi \geq V_E^\pi \land \Omega < 0)$：最优动作但IM使其看起来更差
- $C_3 = \mathbf{1}(Q_E^\pi < V_E^\pi \land \Omega \leq 0)$：次优动作且IM未翻转偏好

则 $F_2 = \Omega - (C_1 + C_2)\Omega - C_3\epsilon$。

**直觉**：当IM不会导致动作偏好翻转时（$C_1 = C_2 = 0$），$F_2 = \Omega - C_3\epsilon$，agent几乎完整地收到原始IM；只有当IM会导致翻转（$C_1 = 1$ 或 $C_2 = 1$）时，ADOPS才"裁剪"奖励使其恰好不翻转。这使得ADOPS能保留尽可能多的IM信号。

### 关键设计2：ADOPES——与PIES融合的渐进版本

由于critic在训练初期估计不准确，直接使用ADOPS可能不理想。ADOPES（Action-Dependent Optimality-Preserving Explicit Shaping）结合PIES的思路：

- 引入系数 $\zeta$，但方向与PIES**相反**：从0线性增大到1
- 训练前期：$\zeta \approx 0$，$F_2$ 的修正几乎不生效，agent自由使用原始IM
- 训练后期：$\zeta \to 1$，$F_2$ 完全生效，保证最优策略不变

这样既利用了训练后期critic估计更准确的特点，又允许agent在整个训练过程中持续接收IM信号（不像PIES在后半程完全归零）。

### 关键设计3：理论保证——比PBRS更广泛的optimality preservation

传统PBRS通过使内在Q值与动作无关来保证最优性，即强制要求 $Q_I^*(a) = V_I^* \; \forall a$。这是保证最优集不变的**充分但非必要条件**。

ADOPS证明了更一般的最优性保持条件（Theorem 5.2）：

$$\text{argmax}_a Q_{IE}^\pi = \text{argmax}_a Q_E^\pi \quad \forall \pi$$

证明的关键步骤是引入**策略稳定性**概念（Assumption 5.1）：收敛后的策略不会是"不稳定的"——即不存在只差一个状态一个动作的另一策略，在任何混合下都严格更优。这个假设对任何合理的学习算法都成立。

论文还证明（Theorem B.1）存在optimality-preserving的shaping函数**无法**被写成任何GRM/PBRS形式，说明ADOPS确实覆盖了更广的函数族。

## 实验关键数据

### 表1：Montezuma's Revenge最终平均外在回报对比

| 方法 | 平均外在回报 | 与RND比较 (p值) | 是否保证最优性 |
|------|------------|----------------|--------------|
| RND (baseline) | ~7400 | — | 否 |
| PBIM (normalized) | 0 | — | 是 (理论上) |
| PBIM (non-normalized) | 0 | — | 是 (理论上) |
| GRM (normalized, D=1) | ~5600 | p=0.009 (显著更差) | 是 |
| GRM (non-normalized, D=1) | ~6200 | p=0.031 (显著更差) | 是 |
| PIES | ~5400 | p=0.059 (接近显著更差) | 是 |
| **ADOPS** | **~7800** | — | 是 |
| **ADOPES** | **~8400** | **p=0.038 (显著更好)** | 是 |
| **ADOPES w/ F/2** | **~8200** | — | 是 |

### 表2：各方法的关键特性对比

| 特性 | PBRS/PBIM | GRM | PIES | ADOPS |
|------|-----------|-----|------|-------|
| 需要环境是episodic | 是 | 是 | 否 | 否 |
| 需要future-agnostic假设 | 是 | 是 | 否 | 否 |
| 允许action-dependent的内在回报 | 否 | 否 | N/A | 是 |
| 训练后期仍可利用IM | 是 | 是 | 否 | 是 |
| 长episode环境可用 | 否 (指数爆炸) | 否 (数值问题) | 是 | 是 |
| 额外架构/计算开销 | 低 | 低 | 低 | 低（复用critic） |

## 关键发现

1. **PBIM在长episode环境中完全失败**：末尾补偿项 $1/\gamma^{N-1}$ 在Montezuma's Revenge中（$N=4500, \gamma=0.99$）达到约 $10^{19}$，导致动作概率立即饱和，agent从未获得任何外在奖励
2. **GRM的delay参数D越小越好**：D越大越接近PBIM的末尾补偿行为，D=1时表现最好但仍不如RND baseline
3. **PIES存在根本性trade-off**：前半程有IM时表现好（甚至因为隐式找到了更好的IM缩放系数），后半程停止IM后性能急剧下降且无法恢复
4. **ADOPES的PIES式前期效果揭示了IM缩放系数的重要性**：ADOPES w/ F/2 学习更快，说明PIES的早期成功主要是因为意外地降低了IM系数而非其optimality-preserving机制
5. **ADOPES显著优于RND baseline**（p=0.038），是唯一一个在保证最优性的同时超越baseline的方法

## 亮点与洞察

- **思路精妙**：不是"让IM变成potential-based形式"，而是"在动作偏好层面做裁剪"。这个视角转换突破了PBRS框架的根本限制
- **实现简洁**：只需查询已有的critic网络，不需要额外的神经网络或辅助优化，真正做到"plug-and-play"
- **ADOPES的设计巧妙**：前期让critic先学准（不施加ADOPS约束），后期critic准确后再逐步启用约束。这种"先探索再约束"的策略比PIES的"先自由后禁止"更合理
- **理论贡献扎实**：不仅证明了ADOPS的正确性，还证明了存在ADOPS能处理但PBRS/GRM永远无法处理的optimality-preserving函数，说明action-dependent形式的必要性

## 局限性

1. **依赖critic估计质量**：ADOPS的正确性依赖于外在和内在值函数估计的准确性。虽然ADOPES通过渐进启用缓解了这一问题，但在critic始终学不好的环境中可能受限
2. **Assumption 5.1的实际满足程度**：要求学习算法收敛后不执行"不稳定策略"，虽然直觉上合理但在有限训练步数下难以严格保证
3. **仅在单一环境（Montezuma's Revenge）上验证**：虽然这是该领域公认的benchmark，但未测试其他稀疏奖励环境（如Pitfall等）
4. **需要分离的外在/内在critic**：实现上需要两套critic网络分别估计外在和内在值函数，虽然RND已经这样做了，但限制了ADOPS与使用单一critic的RL算法的兼容性
5. **epsilon超参数的选择**：虽然理论上只要是正数即可，但实际中过小可能导致数值不稳定，过大可能过度压制IM

## 相关工作与启发

- **与EIPO的比较**：EIPO也自动调整IM缩放，但依赖额外神经网络且无理论保证。ADOPS更简洁且有理论支撑
- **对IM缩放系数的启示**：实验意外发现默认的IM系数1.0可能偏大，降低到0.5反而加速训练。这值得在其他IM方法中探究
- **可推广性**：ADOPS的思路不限于RND，理论上可用于任何IM方法（ICM、count-based等），值得进一步验证
- **reward shaping的新范式**：从"让shaping reward本身满足某种形式"转向"动态调整shaping reward以不改变动作偏好"，提供了思考reward shaping的新角度

## 评分

- 新颖性: 5/5 — 突破PBRS的action-independent限制，视角独特
- 理论深度: 5/5 — 完整的optimality preservation证明和无法被PBRS覆盖的反例
- 实验说服力: 4/5 — 在公认困难benchmark上达SOTA，但仅一个环境
- 实用性: 4/5 — 真正plug-and-play，但需分离critic
- 综合: 5/5 — 理论和实验双优的高质量工作

<!-- RELATED:START -->

## 相关论文

- [Automatic Reward Shaping from Confounded Offline Data](automatic_reward_shaping_from_confounded_offline_data.md)
- [Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online RL](gradual_transition_from_bellman_optimality_operator_to_bellman_operator_in.md)
- [MVR: Multi-view Video Reward Shaping for Reinforcement Learning](../../ICLR2026/reinforcement_learning/mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)
- [Action-Constrained Imitation Learning](action-constrained_imitation_learning.md)
- [Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation](demystifying_the_paradox_of_importance_sampling_with_an_estimated_history-depend.md)

<!-- RELATED:END -->
