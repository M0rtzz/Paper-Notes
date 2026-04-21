---
title: >-
  [论文解读] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning
description: >-
  [ICLR 2026][Stackelberg 博弈] 提出 SCORER 框架，将 Deep Q-Learning 中的表征学习和值函数学习建模为 Stackelberg 博弈，通过双时间尺度更新（Q 网络为 leader 慢更新、编码器为 follower 快更新）实现稳定协同适应，无需改变网络结构即可提升性能。
tags:
  - ICLR 2026
  - Stackelberg 博弈
  - 表征学习
  - Deep Q-Learning
  - 双时间尺度
  - 方差最小化
---

# Stackelberg Coupling of Online Representation Learning and Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2508.07452](https://arxiv.org/abs/2508.07452)  
**代码**: [https://github.com/fernando-ml/SCORER](https://github.com/fernando-ml/SCORER)  
**领域**: 强化学习 / 表征学习  
**关键词**: Stackelberg 博弈, 表征学习, Deep Q-Learning, 双时间尺度, 方差最小化

## 一句话总结

提出 SCORER 框架，将 Deep Q-Learning 中的表征学习和值函数学习建模为 Stackelberg 博弈，通过双时间尺度更新（Q 网络为 leader 慢更新、编码器为 follower 快更新）实现稳定协同适应，无需改变网络结构即可提升性能。

## 研究背景与动机

- **致命三要素（Deadly Triad）问题**：深度 Q-Learning 中，函数近似、自举（bootstrapping）和离策略学习的组合导致不稳定性，可能引发表征崩溃和灾难性学习失败。
- **单体网络的局限**：传统方法在单一网络中同时学习表征和值函数，导致表征必须不断适应非平稳的值目标，而值估计又依赖于变化的表征，形成恶性循环。
- **辅助损失的冲突**：引入额外辅助损失（如自监督目标）来稳定表征可能与主值学习目标产生梯度冲突。
- **核心思路**：不是简单增加辅助损失，而是从根本上重构优化问题——将其建模为层次化的 Stackelberg 博弈。

## 方法详解

### 整体框架

SCORER 将智能体内部拆分为两个策略玩家：
- **Leader（控制网络 $Q_\theta$）**：负责值估计，慢速更新，提供稳定目标
- **Follower（感知网络 $f_\phi$）**：负责表征学习，快速适应，学习对 leader 策略的最优响应

### 1. Leader 目标：最小化 MSBE

$$\min_\theta \mathcal{L}_{\text{leader}}(Q_\theta, f_{\phi^*(\theta)}) \triangleq \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} \left[(Y - Q_\theta(f_{\phi^*(\theta)}(s), a))^2\right]$$

### 2. Follower 目标：最小化 Bellman 误差方差

$$\phi^*(\theta) \in \arg\min_\phi \mathcal{L}_{\text{follower}}(f_\phi, Q_\theta) \triangleq \text{Var}_{j \in B}[\delta_j(\phi, \theta)]$$

其中 $\delta_j(\phi, \theta) = Y_j - Q_\theta(f_\phi(s_j), a_j)$ 是 Bellman 误差。选择方差而非 MSBE 的原因：方差最小化鼓励表征产生更一致的 Bellman 误差，使表征对 TD 学习中的噪声目标更鲁棒，直接对抗致命三要素的根源。

### 3. 双层优化的双时间尺度近似

形式上构成双层优化（bi-level optimization）：

$$\min_\theta \mathcal{L}_{\text{leader}}(Q_\theta, f_{\phi^*(\theta)}) \quad \text{s.t.} \quad \phi^*(\theta) \in \arg\min_\phi \mathcal{L}_{\text{follower}}(f_\phi, Q_\theta)$$

通过双时间尺度梯度下降近似 Stackelberg 均衡：
- Follower 使用较大学习率 $\alpha_{\phi,k}$（快时间尺度）
- Leader 使用较小学习率 $\alpha_{\theta,k}$（慢时间尺度）
- 满足 $\lim_{k \to \infty} \alpha_{\theta,k} / \alpha_{\phi,k} = 0$

更新规则（使用 stop-gradient 阻止梯度流动）：

$$\phi_{k+1} \leftarrow \phi_k - \alpha_{\phi,k} \nabla_\phi \mathcal{L}_{\text{follower}}(\phi_k; B_{\text{follower}}, Y, \bar{\theta_k})$$

$$\theta_{k+1} \leftarrow \theta_k - \alpha_{\theta,k} \nabla_\theta \mathcal{L}_{\text{leader}}(\theta_k; B_{\text{leader}}, Y, \bar{\phi_{k+1}})$$

### 4. 实现简洁性

SCORER 的实现极其简单——仅需为两个组件设置不同的衰减学习率，无需修改网络架构、无需额外模块。

## 实验结果

### 主实验：MinAtar 环境（最终 IQM 回报，30 seeds）

| 算法 | 方法 | Asterix | Breakout | Freeway | SpaceInvaders | 速度 |
|------|------|---------|----------|---------|---------------|------|
| DQN | Baseline | 54.95 | 19.16 | 62.70 | 127.78 | 1.00x |
| DQN | **SCORER** | **54.78** | **65.69** | **63.03** | **148.71** | 0.99x |
| DDQN | Baseline | 50.77 | 36.47 | 62.22 | 116.72 | 1.00x |
| DDQN | **SCORER** | **52.59** | **64.44** | **62.68** | **146.67** | 1.00x |
| DuelingDQN | Baseline | 39.22 | 27.81 | 61.89 | 121.21 | 1.00x |
| DuelingDQN | **SCORER** | **52.28** | **60.04** | **62.27** | **139.08** | 1.01x |

### 关键发现

- Breakout 上 DQN+SCORER 将最终得分提高了 **3 倍以上**（19.16 → 65.69）
- SCORER 使基于 replay buffer 的旧方法能与 PQN 等先进方法竞争
- 速度几乎无开销（0.99-1.01x）

### 消融实验

| Follower 目标 | 性能表现 |
|-------------|----------|
| Bellman 误差方差 | **最优** |
| MSBE | 次优 |
| 无 follower | Baseline |

- 方差最小化目标始终优于直接最小化 MSBE
- follower 使用独立 batch 采样效果更好

## 亮点与洞察

1. **博弈论视角的新颖性**：首次将值函数 RL 中的表征-控制交互建模为 Stackelberg 博弈
2. **极简实现**：仅需修改学习率调度，无需改变架构
3. **广泛适用性**：在 DQN、DDQN、Dueling DQN、R2D2、PQN 上均有效
4. **理论支撑**：双时间尺度收敛理论保证了一阶驻点收敛

## 局限性

- 需要调节两个学习率的衰减参数
- 理论分析基于一阶近似，省略了隐式梯度项的影响
- 主要在离散动作空间环境中验证，连续动作空间的效果未探索

## 相关工作

- **表征学习 in RL**：辅助任务（SPR、CURL）、对比学习、自监督方法
- **双时间尺度优化**：TTSA 理论（Borkar 1997、Hong et al. 2023）
- **值分解方法**：Dueling DQN 等架构分离，但缺乏博弈论耦合

## 评分

- **创新性**: ⭐⭐⭐⭐ — 将 Stackelberg 博弈引入表征-值函数协同学习
- **技术深度**: ⭐⭐⭐⭐ — 形式化的双层优化框架与收敛分析
- **实验充分性**: ⭐⭐⭐⭐ — 多算法多环境覆盖，消融完整
- **实用价值**: ⭐⭐⭐⭐⭐ — 实现简单，即插即用，无额外计算开销

<!-- RELATED:START -->

## 相关论文

- [Learning to Play Multi-Follower Bayesian Stackelberg Games](learning_to_play_multi-follower_bayesian_stackelberg_games.md)
- [The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)
- [Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)
- [Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)
- [Spectral Bellman Method: Unifying Representation and Exploration in RL](spectral_bellman_method_unifying_representation_and_exploration_in_rl.md)

<!-- RELATED:END -->
