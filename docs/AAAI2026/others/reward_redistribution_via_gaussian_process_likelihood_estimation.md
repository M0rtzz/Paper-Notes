---
title: >-
  [论文解读] Reward Redistribution via Gaussian Process Likelihood Estimation
description: >-
  [AAAI 2026][稀疏奖励] 本文提出了基于高斯过程似然的奖励重分配框架 GP-LRR，通过核函数显式建模 state-action 对之间的相关性，利用 leave-one-out 策略最大化轨迹回报的边际似然来学习逐步奖励函数，理论证明传统 MSE 方法是其退化特例，并在 MuJoCo 基准上配合 SAC 实现了优越的样本效率和策略性能。
tags:
  - AAAI 2026
  - 稀疏奖励
  - 奖励重分配
  - 高斯过程
  - 似然估计
  - 信用分配
---

# Reward Redistribution via Gaussian Process Likelihood Estimation

**会议**: AAAI 2026  
**arXiv**: [2503.17409](https://arxiv.org/abs/2503.17409)  
**代码**: [GitHub](https://github.com/xiao-1120/AAAI-LRR)  
**领域**: 强化学习 / 奖励重分配  
**关键词**: 稀疏奖励, 奖励重分配, 高斯过程, 似然估计, 信用分配

## 一句话总结

本文提出了基于高斯过程似然的奖励重分配框架 GP-LRR，通过核函数显式建模 state-action 对之间的相关性，利用 leave-one-out 策略最大化轨迹回报的边际似然来学习逐步奖励函数，理论证明传统 MSE 方法是其退化特例，并在 MuJoCo 基准上配合 SAC 实现了优越的样本效率和策略性能。

## 研究背景与动机

在许多实际强化学习任务中，反馈仅在一段长序列动作结束后（即 episode 末尾）才给出，导致奖励信号**极度稀疏和延迟**。例如在航空设计中，飞行器部件的质量只有在整个制造过程完成后才能评估。这种稀疏反馈使得难以判断哪些动作对最终结果影响最大，导致学习过程收敛缓慢或陷入次优策略。

**现有方法的局限**：

**RUDDER**：使用 LSTM 预测逐步回报并通过反向传播重分配信用，但长序列 BPTT 不稳定且计算成本高。

**IRCR**：将 episode 回报均匀分配到每个时间步，计算简单但完全忽略了时间结构。

**RRD**：采样随机轨迹子序列近似最优分解，但仍忽略 state-action 对之间的依赖关系。

**GRD**：学习因果生成模型，产生可解释的代理奖励，但结构复杂。

**核心问题**：所有上述方法都**假设每步奖励是独立的**，忽视了 state-action 对之间的相互依赖性。然而，作者通过实验观察到多个环境中存在显著的**滞后-1 自相关**（如 HalfCheetah-v4 中相邻步奖励 $(r_t, r_{t+1})$ 高度相关），表明连续步之间的奖励由于 state-action 对的相关性而具有时间依赖性。忽略这种依赖会导致无效的奖励重分配，遗漏重要的动作交互，最终损害学习效率。

**GP 的适切性**：高斯过程（GP）是建模这种依赖性的天然工具——它是一种非参数方法，将未知奖励函数视为黑箱函数，通过核函数（协方差）显式建模不同 state-action 对之间的奖励相关性。尽管 GP 在 RL 中已被用于值函数近似和转移动力学建模，但**据作者所知这是首次将 GP 用于奖励重分配**。

## 方法详解

### 整体框架

GP-LRR 框架的核心思路：

1. 将每步奖励 $R(s,a)$ 建模为高斯过程的一个样本
2. 使用 leave-one-out（LOO）策略构造训练目标
3. 最大化观测到的 episode 回报的边际似然
4. 将学到的均值函数作为稠密奖励信号用于下游策略优化

### 关键设计

1. **高斯过程奖励建模**：将每步的真实奖励函数 $R_b(s,a)$ 建模为 GP 的样本：

    $R_b(s,a) \sim \mathcal{GP}(\mu_{\boldsymbol{\theta}}(s,a), k_{\boldsymbol{\phi}}((s,a),(s',a')))$

   其中 $\mu_{\boldsymbol{\theta}}(s,a)$ 是参数化均值函数（用神经网络表示），$k_{\boldsymbol{\phi}}$ 是核函数。对于一条轨迹 $\tau$，真实奖励向量服从多元高斯分布：$\mathbf{r}_b \sim \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\theta}}, \mathbf{K}_{\boldsymbol{\phi}})$。

   核函数的作用是**衡量不同 state-action 对之间的奖励相关性**。核心假设是：**相似的 state-action 对应该产生相似的奖励**。默认使用 RBF 核：

    $k_{\boldsymbol{\phi}}((s,a),(s',a')) = \sigma_f^2 \exp\left(-\frac{\|(s,a)-(s',a')\|^2}{2\ell_{\text{rbf}}^2}\right)$

2. **Leave-One-Out 目标构造**：在只有 episode 总回报 $R_{ep}(\tau)$ 而无逐步奖励的环境中，对每个时间步 $i$ 构造 LOO 目标：

    $\tilde{r}(s_i, a_i) = R_{ep}(\tau) - \sum_{t=0, t \neq i}^{T-1} \mu_{\boldsymbol{\theta}}(s_t, a_t)$

   含义是：将总回报减去其他所有步的当前估计奖励，作为当前步奖励的代理目标。这些 LOO 目标作为"含噪观测"用于构造似然函数。

3. **边际似然最大化**：给定 LOO 目标，优化如下对数边际似然：

    $\log p(\tilde{\mathbf{r}} \mid \boldsymbol{\theta}, \boldsymbol{\phi}, \sigma_\epsilon) = \underbrace{-\frac{1}{2}(\tilde{\mathbf{r}}-\boldsymbol{\mu}_{\boldsymbol{\theta}})^\top \mathbf{K}_\sigma^{-1}(\tilde{\mathbf{r}}-\boldsymbol{\mu}_{\boldsymbol{\theta}})}_{\text{数据拟合项}} \underbrace{- \frac{1}{2}\log\det(\mathbf{K}_\sigma)}_{\text{奥卡姆因子}} - \frac{|\tau|}{2}\log(2\pi)$

   其中 $\mathbf{K}_\sigma = \mathbf{K}_{\boldsymbol{\phi}} + \sigma_\epsilon^2 \mathbf{I}$。数据拟合项衡量模型对数据的解释能力，奥卡姆因子惩罚模型过拟合。

4. **与 SAC 的集成**（Algorithm 2）：

    - 维护两个缓冲区：transition buffer $\mathcal{D}$（用于 SAC 更新）和完整轨迹 buffer $\mathcal{D}_\tau$（用于 GP 训练）
    - 每 $n_{\text{update}}$ 个 episode 更新一次 GP 模型
    - SAC 更新时用学到的均值函数 $\mu_{\boldsymbol{\theta}}(s,a)$ 作为稠密奖励信号替代稀疏的 episode 奖励
    - GP 训练通过 Cholesky 分解保证数值稳定性

### 理论分析

本文提供了四个重要的理论命题：

**Proposition 1（MSE 是特例）**：当核矩阵退化为单位阵 $\mathbf{K}_{\boldsymbol{\phi}} = \mathbf{I}$ 且观测噪声消失 $\sigma_\epsilon = 0$ 时，GP-LRR 的目标函数退化为标准的 MSE 奖励重分配损失：

$$\mathcal{L}(\tau; \boldsymbol{\theta}) \propto \frac{|\tau|}{2}\left(R_{ep}(\tau) - \sum_{t=0}^{|\tau|-1}\mu_{\boldsymbol{\theta}}(s_t, a_t)\right)^2$$

这证明了 GP-LRR 严格包含传统方法。

**Proposition 2（梯度流与相关性）**：GP-LRR 对均值函数参数的梯度为：

$$\frac{\partial\mathcal{L}}{\partial\boldsymbol{\theta}} = -\frac{\partial\boldsymbol{\mu}_{\boldsymbol{\theta}}^\top}{\partial\boldsymbol{\theta}} \cdot \mathbf{K}_\sigma^{-1}(\tilde{\mathbf{r}} - \boldsymbol{\mu}_{\boldsymbol{\theta}})$$

对于特定的 $(s_i, a_i)$，其梯度贡献汇聚了**所有** state-action 对的预测误差（通过精度矩阵 $\mathbf{K}_\sigma^{-1}$ 加权），而非仅依赖自身的误差。这创造了一个"**信用分配网络**"——高度相关的 state-action 对的误差会被池化到一起进行更新。

**Proposition 3（长度尺度与平滑性权衡）**：GP 通过最大化边际似然自动调整 $\ell_{\text{rbf}}$，平衡数据拟合与模型复杂度。

**Proposition 4（噪声水平自适应）**：$\sigma_\epsilon^2$ 的梯度包含正则化项 $\frac{1}{2}\text{tr}(\mathbf{K}_\sigma^{-1})$，防止噪声消失，避免过度拟合。

### 损失函数 / 训练策略

- GP 模型通过最大化对数边际似然训练，梯度下降更新参数 $\boldsymbol{\theta}, \sigma_f^2, \ell_{\text{rbf}}, \sigma_\epsilon^2$
- SAC 部分照常使用 soft Bellman 残差（critic）、熵增强策略损失（actor）和温度自适应
- GP 训练复杂度为 $\mathcal{O}(M|\tau|^3)$，通过每 100 步更新一次（batch size 4）来摊销成本
- 奖励模型使用 2 层隐藏层 MLP（64 或 256 神经元），学习率 $10^{-3}$

## 实验关键数据

### 主实验

在四个 MuJoCo 环境的稀疏 episode 奖励设定下比较（5 次独立运行的平均回报）：

| 环境 | SAC (稀疏) | IRCR (均匀) | RRD (随机分解) | **GP-LRR (本文)** |
|------|-----------|------------|--------------|------------------|
| HalfCheetah-v4 | 低且停滞 | 缓慢上升 | 中等 | **最高且收敛最快** |
| Hopper-v4 | 低方差大 | 中等 | 中等 | **最高** |
| Swimmer-v4 | 低 | 接近 GP-LRR | 中等 | **最高** |
| Walker2d-v4 | 低 | 低 | 中等 | **最高** |

GP-LRR 在所有四个环境中一致最优，在 HalfCheetah-v4 中优势尤为显著（平滑动力学与空间相关奖励高度匹配 GP 的相关性建模）。

### 消融实验

**核函数选择**：

| 核函数 | HalfCheetah | Hopper | Swimmer | Walker2d |
|--------|-------------|--------|---------|----------|
| RBF | **最优** | **最优** | **最优** | 最差 |
| Matérn-3/2 | 中等 | 中等 | 中等 | **最优** |
| Rational Quadratic | 中等 | 中等 | 中等 | 次优 |

核函数性能与任务相关：RBF 适合平滑奖励景观，Matérn 更适合包含不连续性的 Walker2d。

**长度尺度初始化**：

| 初始化 $\ell$ | HalfCheetah | Hopper | Swimmer | Walker2d |
|--------------|-------------|--------|---------|----------|
| 小 (0.1) | 差 | 方差大 | 不敏感 | 中等 |
| 中 (1.0) | 中等 | 中等 | 不敏感 | **最优** |
| 大 (10.0) | **最优** | 中等 | 不敏感 | 中等 |
| 自适应 | 中等 | 方差大 | 不敏感 | 中等 |

无统一最优初始化——HalfCheetah 偏好大值，Walker2d 偏好中等值，Swimmer 不敏感。

### 关键发现

1. **相关性建模的价值**：GP-LRR 的优势主要来源于通过核函数池化相关 state-action 对的误差信息，而非独立地处理每步奖励。
2. **隐式不确定性正则化**：边际似然中的奥卡姆因子和噪声自适应机制防止了对探索不足区域的过度自信奖励分配。
3. **任务依赖性**：核函数和超参数的最优选择高度依赖于具体环境的奖励结构特性，没有"一刀切"的最优方案。
4. **off-policy 兼容性**：与 SAC 的无缝集成使得 GP-LRR 在样本效率上优于需要 on-policy 数据的方法。

## 亮点与洞察

- **非参数方法用于奖励重分配的新视角**：将奖励函数视为黑箱并用 GP 建模，避免了参数族假设的限制，同时通过核函数自然地引入了相关性结构。
- **理论-实践的良好结合**：四个命题清晰地揭示了 GP-LRR 与传统方法的关系（Proposition 1 的包含关系）、优势来源（Proposition 2 的信用分配网络）、以及内置的正则化机制（Proposition 3-4）。
- **LOO 策略的巧妙设计**：在只有 episode 总回报的约束下，LOO 目标既利用了全部轨迹信息，又为 GP 似然提供了合理的"观测"数据。
- **MSE 作为退化特例的证明**非常优雅：$\mathbf{K}_\phi = \mathbf{I}$（无相关性）$+ \sigma_\epsilon = 0$（无噪声）= 传统 MSE，清晰地解释了传统方法的不足。

## 局限与展望

1. **计算复杂度**：GP 训练的 $\mathcal{O}(M|\tau|^3)$ 复杂度限制了其在长轨迹任务上的应用。矩阵求逆虽可用 Cholesky 分解加速，但随轨迹长度增加仍是瓶颈。
2. **实验规模有限**：仅在四个 MuJoCo 环境上测试，且作者承认并非所有环境都适合建模 state-action 相关性。缺乏对离散动作空间、高维观测空间的验证。
3. **核函数选择缺乏指导**：不同环境最优核函数不同，缺少自动选择或组合核函数的机制。
4. **仅支持高斯噪声模型**：当奖励噪声为非高斯时可能失效，作者在结论中将非高斯扩展列为未来工作。
5. **与更多 baseline 的比较缺失**：未与 GRD、DIASter 等较新方法进行实验比较。
6. **超参数敏感性**：长度尺度初始化对结果有明显影响，增加了调参负担。

## 相关工作与启发

- GP 在 RL 中的应用从值函数近似、转移动力学扩展到奖励重分配，展现了 GP 作为灵活非参数工具的广泛适用性。
- LOO 策略可能对其他需要从全局信号推断局部贡献的场景（如多 agent 信用分配、特征归因）有启发。
- 精度矩阵提供的"信用分配网络"概念，可能启发其他基于图结构的奖励重分配方法。
- 将来可以考虑用稀疏 GP 近似（如诱导点方法）来降低计算复杂度，使其适用于更长的轨迹。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | ⭐⭐⭐⭐ | 首次将 GP 用于奖励重分配，视角新颖 |
| 实用性 | ⭐⭐⭐ | 计算复杂度和超参数敏感性限制了实际应用 |
| 理论深度 | ⭐⭐⭐⭐⭐ | 四个命题完整且优雅，MSE 特例证明尤佳 |
| 实验充分性 | ⭐⭐⭐ | 环境有限，缺少与更多近期方法的比较 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，理论与实验结合紧密 |
| 总评 | ⭐⭐⭐⭐ | 理论贡献突出，但实验验证范围需扩展 |

<!-- RELATED:START -->

## 相关论文

- [Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret in Noise-Free Gaussian Process Bandits](../../NeurIPS2025/others/gaussian_process_upper_confidence_bound_achieves_nearly-optimal_regret_in_noise-.md)
- [Expressive Temporal Specifications for Reward Monitoring](expressive_temporal_specifications_for_reward_monitoring.md)
- [Private Frequency Estimation via Residue Number Systems](private_frequency_estimation_via_residue_number_systems.md)
- [Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Cameras](spike_imaging_velocimetry_dense_motion_estimation_of_fluids_using_spike_cameras.md)
- [How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation](how_wide_and_how_deep_mitigating_over-squashing_of_gnns_via_channel_capacity_con.md)

<!-- RELATED:END -->
