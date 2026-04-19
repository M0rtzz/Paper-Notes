---
title: >-
  [论文解读] Wasserstein Policy Optimization
description: >-
  [ICML2025][Wasserstein梯度流] 提出 Wasserstein Policy Optimization (WPO)，将最优传输理论中的 Wasserstein 梯度流投影到参数空间，得到一种兼具确定性策略梯度（DPG）利用动作值梯度和经典随机策略梯度（SPG）支持任意分布的闭式更新规则，无需重参数化技巧。
tags:
  - ICML2025
  - Wasserstein梯度流
  - 策略梯度
  - 最优传输
  - Actor-Critic
  - 连续动作空间
---

# Wasserstein Policy Optimization

**会议**: ICML2025  
**arXiv**: [2505.00663](https://arxiv.org/abs/2505.00663)  
**代码**: [google-deepmind/acme](https://github.com/google-deepmind/acme)  
**领域**: 策略优化 / 强化学习 / 连续控制  
**关键词**: Wasserstein梯度流, 策略梯度, 最优传输, Actor-Critic, 连续动作空间

## 一句话总结

提出 Wasserstein Policy Optimization (WPO)，将最优传输理论中的 Wasserstein 梯度流投影到参数空间，得到一种兼具确定性策略梯度（DPG）利用动作值梯度和经典随机策略梯度（SPG）支持任意分布的闭式更新规则，无需重参数化技巧。

## 研究背景与动机

连续动作空间中的策略优化是深度强化学习的核心问题。现有方法主要分两大类：

**经典随机策略梯度（SPG）**：如 REINFORCE、PPO、MPO，通过 $\nabla_\theta \log \pi_\theta(\mathbf{a}|\mathbf{s})$ 更新参数，适用于任意随机策略，但仅利用标量 $Q$ 值，高维动作空间中方差大、效率低。

**确定性策略梯度（DPG）**：如 DDPG、TD3，利用 $\nabla_\mathbf{a} Q(\mathbf{s}, \mathbf{a})$ 提供方向信息，高维中更高效，但仅限确定性策略，探索困难。扩展方案（SVG(0)、SAC）依赖重参数化技巧，限制了策略分布类型。

**核心问题**：能否设计一种策略更新方法，既利用动作值函数的梯度（如 DPG），又能学习任意随机策略（如 SPG），且不依赖重参数化技巧？

## 方法详解

### 核心思想：Wasserstein 梯度流

WPO 的出发点是概率分布空间上的梯度流理论。给定任意策略泛函 $\mathcal{J}[\pi]$，其在 2-Wasserstein 距离度量下的最速下降满足连续性方程：

$$\frac{\partial \pi}{\partial t} = -\nabla_\mathbf{a} \cdot \left(\pi \left(-\nabla_\mathbf{a} \frac{\delta \mathcal{J}}{\delta \pi}\right)\right)$$

对于 MDP 中的期望回报，泛函导数为 $\frac{\delta \mathcal{J}}{\delta \pi}(\mathbf{s}, \mathbf{a}) = \frac{1}{1-\gamma} Q^\pi(\mathbf{s}, \mathbf{a}) d^\pi(\mathbf{s})$，因此速度场为 $\nabla_\mathbf{a} Q^\pi(\mathbf{s}, \mathbf{a})$——这正是动作值函数的梯度。

### 参数化投影：从 PDE 到闭式更新

将非参数梯度流投影到神经网络参数空间 $\theta$：最小化 $D_\text{KL}[\pi_\theta \| \pi_\theta + \frac{\partial \pi}{\partial t} dt - \nabla_\theta \pi_\theta \Delta\theta]$，利用 Fisher 信息矩阵的二次近似，求解得：

$$\Delta\theta = \mathcal{F}_{\theta\theta}^{-1} \underbrace{\mathbb{E}_{\mathbf{a}\sim\pi}\left[\nabla_\theta \nabla_\mathbf{a} \log \pi_\theta(\mathbf{a}|\mathbf{s}) \cdot \nabla_\mathbf{a} Q^\pi(\mathbf{s}, \mathbf{a})\right]}_{\mathcal{F}_{t\theta}}$$

其中 $\mathcal{F}_{t\theta}$ 通过分部积分化简为上式——**WPO 核心更新公式**（公式 6）。

### 实际近似与损失函数

1. **自然梯度近似**：对高斯策略 $\pi = \mathcal{N}(\mu_\theta, \Sigma_\theta)$，利用对角 Fisher 矩阵结构，将 $\nabla_{\mu_i} \log \pi$ 乘以 $\sigma_i^2$、$\nabla_{\sigma_i} \log \pi$ 乘以 $\frac{1}{2}\sigma_i^2$ 进行方差归一化，避免策略坍缩时梯度爆炸。
2. **KL 正则化**：防止策略过快变化，采用 MPO 风格的 KL 惩罚：

$$\max_\pi \mathbb{E}\left[\sum_t \gamma^t \left(\mathbb{E}_{\mathbf{a}\sim\pi}[r_t] - \alpha D_\text{KL}[\bar{\pi}(\cdot|\mathbf{s}_t) \| \pi(\cdot|\mathbf{s}_t)]\right)\right]$$

3. **Critic 更新**：标准 $n$-step TD 目标配合目标网络。

### 与现有方法的等价性分析

在单变量高斯策略下：
- WPO 均值更新 $\Delta_\mu \theta = \mathbb{E}_\pi[\nabla_a Q(a) \nabla_\theta \mu]$，与 DPG 更新形式一致（但在采样点而非均值处取梯度）。
- WPO 方差更新 $\Delta_\sigma \theta = \mathbb{E}_\pi[\frac{a-\mu}{\sigma} \nabla_a Q(a) \nabla_\theta \sigma]$，与 SVG(0) 重参数化更新相同。
- 进一步可证明期望 WPO 更新 = 经典策略梯度 $\mathbb{E}_\pi[Q(a) \nabla_\theta \log \pi(a)]$。

**但**：（1）采样方差不同——当 $Q$ 对动作局部线性时，WPO 均值更新零方差；（2）对非高斯策略（如混合高斯），WPO 与 SPG 有质的差异，收敛更快更稳定。

## 实验关键数据

### DeepMind Control Suite（28 个任务）

| 方法 | 整体表现 | 高维任务（Humanoid CMU） | 稳定性 |
|------|---------|------------------------|--------|
| **WPO** | 几乎所有任务可比 SOTA | Walk 初期学得更快 | ✅ 最稳健 |
| MPO | 强基线 | 稳定学习 | ✅ 稳健 |
| SAC | 部分任务收敛差 | 完全无法起步 | ⚠️ 对熵权重敏感 |
| DDPG | 部分任务收敛差 | 完全无法起步 | ⚠️ 稀疏奖励困难 |

### 高维扩展实验（Combined Humanoid Stand）

| 动作维度 | 1×21 | 3×65 | 5×105 |
|----------|-------|-------|-------|
| WPO 起步速度 | 与 MPO 相当 | 显著快于 MPO | **大幅领先** |
| SAC | 收敛最慢 | 收敛最慢 | 收敛最慢 |

**关键发现**：动作维度增长时，WPO 的起步优势随之增大，暗示其在数百维动作空间中优势更显著。

### 核聚变磁约束控制（Tokamak，19维动作，93维观测）

- WPO 奖励略高于 MPO。
- WPO 策略方差随训练逐步趋零（符合完全可观测环境预期），而 MPO 保持近似恒定方差。

## 亮点与洞察

1. **理论优雅**：从最优传输理论出发，通过 Wasserstein 梯度流 → 连续性方程 → Fisher 投影，推导出简洁闭式更新，统一了 DPG 和 SPG 两大范式。
2. **通用性强**：不依赖重参数化技巧，可适用于任意连续策略分布（指数分布、混合高斯等），突破了 SAC/SVG(0) 的分布限制。
3. **低方差优势**：当 $Q$ 的动作梯度在采样点方向一致时（如局部线性），WPO 更新的方差远低于经典策略梯度。
4. **高维扩展性**：实验证明动作维度越高优势越大，在 105 维联合控制中大幅领先基线。
5. **实际可行性**：已集成到 Acme 框架开源，且工程改动简单（方差归一化 + KL 正则）。

## 局限与展望

1. **部分可观测性能下降**：Dog 域（状态 > 观测）表现不佳，暗示在 POMDP 设置中可能需要额外适配。
2. **仅验证高斯策略**：虽然理论上支持任意分布，实验仅使用对角高斯策略，非高斯策略（如混合高斯、flow-based）的实际效果待验证。
3. **自然梯度近似粗糙**：仅使用策略分布层面的对角 Fisher 缩放，忽略了网络参数维度的 Fisher 矩阵结构。
4. **超参数调优有限**：作者坦承 WPO 调优远少于 DDPG/SAC 的多年积累，可能还有性能提升空间。
5. **离散动作不适用**：理论依赖连续空间的梯度流，无法直接扩展到离散动作空间。

## 相关工作与启发

- **MPO**（Abdolmaleki et al., 2018）：WPO 的 KL 正则化沿用 MPO 方案；两者在高斯策略下更新形式等价但方差特性不同。
- **SAC**（Haarnoja et al., 2018）：重参数化版本在高斯下与 WPO 等价，但不适用于非重参数化分布。
- **Wasserstein RL 其他用法**：Abdullah et al. (2019) 用于鲁棒 MDP、Moskovitz et al. (2020) 用作预条件器——与 WPO 均不同（WPO 是速度场级别的梯度流）。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 最优传输理论与策略优化的新颖桥梁，闭式更新推导精巧
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 Control Suite + 高维扩展 + 核聚变真实任务，但缺少非高斯策略实验
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导清晰严谨，直觉图示（Fig 1-3）极佳
- 价值: ⭐⭐⭐⭐ — 为高维连续控制提供了新的算法选择，高维扩展性结果尤为亮眼

<!-- RELATED:START -->

## 相关论文

- [Extreme Value Policy Optimization for Safe Reinforcement Learning](extreme_value_policy_optimization_for_safe_reinforcement_learning.md)
- [Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](../../NeurIPS2025/reinforcement_learning/sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)
- [Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](../../AAAI2026/reinforcement_learning/behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)
- [Latent Wasserstein Adversarial Imitation Learning](../../ICLR2026/reinforcement_learning/latent_wasserstein_adversarial_imitation_learning.md)
- [Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](../../NeurIPS2025/reinforcement_learning/opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)

<!-- RELATED:END -->
