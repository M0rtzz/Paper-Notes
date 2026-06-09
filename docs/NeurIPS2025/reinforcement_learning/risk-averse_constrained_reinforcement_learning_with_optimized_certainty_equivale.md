---
title: >-
  [论文解读] Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents
description: >-
  [NeurIPS 2025][强化学习][约束强化学习] 提出一种基于奖励层面(reward-based)的风险感知约束RL框架，使用优化确定性等价(OCE)风险度量同时覆盖目标和约束，建立了参数化强对偶性，并给出模块化算法——可包装标准RL求解器（如PPO）作为黑盒使用。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "约束强化学习"
  - "风险规避"
  - "优化确定性等价(OCE)"
  - "CVaR"
  - "部分拉格朗日松弛"
---

# Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents

**会议**: NeurIPS 2025  
**arXiv**: [2510.20199](https://arxiv.org/abs/2510.20199)  
**代码**: [有](https://github.com/baturaysaglam/risk-averse-constrained-RL)  
**领域**: 强化学习 / 风险规避  
**关键词**: 约束强化学习, 风险规避, 优化确定性等价(OCE), CVaR, 部分拉格朗日松弛

## 一句话总结

提出一种基于奖励层面(reward-based)的风险感知约束RL框架，使用优化确定性等价(OCE)风险度量同时覆盖目标和约束，建立了参数化强对偶性，并给出模块化算法——可包装标准RL求解器（如PPO）作为黑盒使用。

## 研究背景与动机

约束RL是处理多冲突目标的常见框架（如迷宫导航中既要到达目标又不能撞墙）。标准约束RL以期望累积奖励表达目标和约束，但这在高风险应用中不够严格：

**期望值的局限性**: 期望值无法捕捉回报分布尾部的灾难性事件。例如投资问题中，最大化平均回报可能忽略巨额亏损的风险。

现有风险规避RL的两种主要范式各有不足：

**回报层面(return-based)**: 将风险度量替换期望应用于折扣累积回报 $\rho[-\sum \gamma^\tau r]$。这种方式只捕捉整体回报分布的风险，无法感知**每个时间步**的风险。

**递归风险**: 在每个决策阶段递归评估风险度量，推广期望的tower property。计算复杂，难以扩展。

**本文提出第三种范式——奖励层面(reward-based)**: 将风险度量应用于**占据度量(occupancy measure)**，即：

$$R^* = \sup_{\nu^\pi} \frac{1}{1-\gamma} \cdot -\rho_{\nu^\pi}(-r(s,a))$$

**关键优势**: 这种范式具有**逐步鲁棒性**——同时在奖励值和时间上捕捉风险。当CVaR的 $\beta \to 0$ 时，目标退化为所有时间步所有状态动作的最差奖励的本质下确界，这是比回报层面更强的安全保证。

**核心挑战**: 
- 如何在约束RL中使用奖励层面的风险度量？
- 如何建立对偶性保证（使拉格朗日松弛是精确的）？
- 如何设计可与现有RL算法兼容的实用算法？

## 方法详解

### 整体框架

对于一般的OCE风险度量（包括CVaR、均值半方差等），约束问题写为：

$$\sup_{\pi,t_0} \mathbb{E}[\sum \gamma^\tau r_0'(s_\tau, a_\tau, t_0)] \quad \text{s.t.} \quad \sup_{t_i} \mathbb{E}[\sum \gamma^\tau r_i'(s_\tau, a_\tau, t_i)] \geq c_i$$

其中修改后的奖励 $r_i'(s,a,t) = t - \frac{1}{\beta}(t - r_i(s,a))_+$（对CVaR而言），辅助变量 $t$ 控制风险规避程度。

### 关键设计

#### 1. 参数化强对偶性与部分拉格朗日松弛

**功能**: 证明约束问题可以通过部分拉格朗日松弛精确求解。

**核心思路**: 
- 固定 $t \in \mathcal{T}$ 后，问题变为标准约束RL → 满足Slater条件时强对偶性成立(Proposition 3.3)
- 引入约束资格(Assumption 3.4)：存在一个凸紧集 $\mathcal{I} \subset \mathcal{T}$，使得对所有 $t \in \mathcal{I}$ Slater条件都成立
- 在此条件下，最终的部分对偶问题精确等价于原始约束问题

$$D_\theta^* = \sup_{t \in \mathcal{T}} \inf_{\lambda \in \Lambda} \underbrace{\sup_\theta \mathcal{L}(\pi_\theta, t, \lambda)}_{\text{黑盒RL}}$$

**设计动机**: 使对偶松弛不是近似的而是精确的，这在风险规避约束RL文献中是首次。

#### 2. 模块化算法设计（Algorithm 1）

**功能**: 将问题分离为内层RL子问题 + 外层 $(t, \lambda)$ 更新。

对于固定的 $(t, \lambda)$，内层问题是标准RL（修改后的奖励函数），可以用任何RL算法（如PPO）求解。外层通过SGDA更新：

$$\lambda^{(j+1)} \leftarrow \Pi_\Lambda(\lambda^{(j)} - \eta_\lambda \hat{\nabla}_\lambda \hat{\mathcal{L}})$$
$$t^{(j+1)} \leftarrow \Pi_\mathcal{T}(t^{(j)} + \eta_t \hat{\nabla}_t \hat{\mathcal{L}})$$

**设计动机**: 模块化使用户可以灵活选择：风险中性/风险规避的目标和/或约束的任意组合；任何现有RL算法作为黑盒子问题求解器。

#### 3. 近似最优性保证（Theorem 3.5）

在ε-universal策略参数化下：

$$P^*(t^*) \geq \sup_t \inf_\lambda \sup_\theta \mathcal{L}(\pi_\theta, t, \lambda) \geq P^*(t^*) - \mathcal{O}(\frac{\epsilon}{1-\gamma})$$

即参数化部分对偶问题与真实原始问题的差距仅取决于策略参数化误差。

### 收敛性分析

**定理3.12**: 在Lipschitz平滑性、无偏梯度预言机和近似策略求解器假设下，恢复ε-稳定点的迭代复杂度为：

$$\mathcal{O}\left(\frac{\ell^3(C^2+\sigma^2+\delta^2)(\text{diam}(\Lambda))^2 \hat{\Delta}_\Phi}{\epsilon^6}\right)$$

关键特点：只需要单条轨迹(n=1)，算法可在线运行。当不精确求解器的偏差 $\delta = \mathcal{O}(\epsilon^2)$ 时，恢复精确ε-稳定点。

## 实验关键数据

### 主实验1：安全导航（Safety-Gymnasium）

使用Point agent在Level 1难度，5-10M训练步数：

| 环境 | PPO累积成本 ↓ | MARS累积成本 ↓ | PPO奖励 ↑ | MARS奖励 ↑ |
|------|-------------|-------------|----------|-----------|
| Button | 150.76 | **0.0** | 24.29 | 2.58 |
| Circle | 206.74 | **0.0** | 60.18 | 39.19 |
| Goal | 45.09 | **0.0** | 21.89 | 13.56 |
| Push | 38.48 | **0.0** | 0.93 | **2.42** |

MARS在所有环境中实现**零约束违反**，是唯一做到这点的PPO-based方法。在Push环境中，约束甚至帮助agent获得更高奖励。

### 消融实验：安全速度约束（MuJoCo-v4）

| Agent | 速度阈值c | β-上分位数 | 收敛t值 | 两者匹配 |
|-------|----------|-----------|---------|---------|
| HalfCheetah | 1.450 | 1.419 | 1.417 | ✓ |
| Hopper | 0.373 | 0.370 | 0.370 | ✓ |
| Swimmer | 0.228 | 0.248 | 0.207 | ≈ |
| Walker2d | 1.171 | 1.133 | 1.122 | ✓ |

### 关键发现

1. **t值与CVaR一致**: 收敛后的t值精确匹配训练后速度分布的β-上分位数，验证了CVaR约束的正确工作
2. **λ稳定收敛**: 对偶变量λ在足够训练后稳定振荡在一致值附近
3. **策略可解释**: 受约束agent移动更谨慎（速度更稳定），而无约束PPO agent速度波动大
4. **评估奖励稳定性**: 随训练推进，评估奖励的波动减小，展现了风险管理的效果
5. **唯一零违反**: 在导航任务中，是文献中唯一实现严格零约束违反的基于PPO的方法

## 亮点与洞察

1. **奖励层面 vs 回报层面**: 清晰论证了reward-based风险度量如何同时在值和时间维度提供鲁棒性，这是比return-based更强的安全保证
2. **精确对偶性**: 不同于其他风险约束RL工作（如Chow等人的CVaR方法）只给出近似松弛，本文在约束资格下给出精确等价——这在理论上是重要突破
3. **实用性极强**: 黑盒包装器设计意味着可以直接用PPO/SAC/TD3等任何RL算法，降低实现门槛
4. **灵活性**: 可以混搭风险中性目标+风险规避约束（如实验中的配置），非常符合实际需求

## 局限与展望

1. **完全强对偶性未证明**: Assumption 3.4是否无条件成立仍是开放问题
2. **计算成本高**: 每次 $(t,\lambda)$ 更新都需要近似求解一个策略优化子问题，比风险中性方法慢
3. **步长敏感性**: $\eta_\lambda$ 和 $\eta_t$ 的调参对收敛至关重要，需要较长的patience
4. **动作噪声人为注入**: 实验中通过向动作注入5%高斯噪声来模拟风险，真实环境的不确定性可能更复杂
5. **仅验证CVaR**: 虽然理论覆盖一般OCE，实验仅用CVaR验证

## 相关工作与启发

- 扩展了Bonetti等人的无约束reward-based风险规避RL到约束设置
- 与Chow等人的return-based CVaR约束RL对比：本文的部分拉格朗日松弛在约束资格下是精确的
- 收敛分析框架基于Lin等人的minimax优化理论
- 可考虑扩展到多agent设置或分层RL中

## 评分
- 新颖性: ⭐⭐⭐⭐ (reward-based约束RL+精确对偶性是重要理论贡献)
- 实验充分度: ⭐⭐⭐⭐ (导航+速度场景验证，收敛曲线详尽)
- 写作质量: ⭐⭐⭐⭐⭐ (理论展开清晰，Table 1的formulation对比一目了然)
- 价值: ⭐⭐⭐⭐ (理论与实践兼具，模块化设计实用)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Risk-Averse Total-Reward Reinforcement Learning](risk-averse_total-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[ICML 2025\] Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration](../../ICML2025/reinforcement_learning/controlling_underestimation_bias_in_constrained_reinforcement_learning_for_safe_.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[NeurIPS 2025\] ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
