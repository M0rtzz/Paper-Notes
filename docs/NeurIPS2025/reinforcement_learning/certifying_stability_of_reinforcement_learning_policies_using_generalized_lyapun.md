---
title: >-
  [论文解读] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions
description: >-
  [NEURIPS2025][Lyapunov stability] 提出 Generalized Lyapunov Function 方法，通过将 RL 值函数与神经网络残差项结合，并用多步加权下降条件替代经典的逐步严格下降要求，实现对 RL 策略的稳定性认证。
tags:
  - NEURIPS2025
  - Lyapunov stability
  - reinforcement-learning
  - stability certification
  - value function
  - region of attraction
---

# Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions

**会议**: NEURIPS2025  
**arXiv**: [2505.10947](https://arxiv.org/abs/2505.10947)  
**代码**: [GitHub](https://github.com/ExistentialRobotics/Generalized_Policy_Stability)  
**领域**: reinforcement_learning  
**关键词**: Lyapunov stability, reinforcement-learning, stability certification, value function, region of attraction

## 一句话总结

提出 Generalized Lyapunov Function 方法，通过将 RL 值函数与神经网络残差项结合，并用多步加权下降条件替代经典的逐步严格下降要求，实现对 RL 策略的稳定性认证。

## 背景与动机

强化学习策略在非线性控制任务中表现出色，但缺乏闭环系统的稳定性保证。经典 Lyapunov 方法要求函数在每一步严格递减，这一条件对于学到的策略往往难以满足。RL 的值函数天然编码了长期累积代价，是 Lyapunov 函数的候选者，但由于折扣因子 $\gamma \in (0,1)$ 的存在，折扣值函数不直接满足 Lyapunov 递减条件。

已有工作（如 Romain 2017）尝试通过给值函数加二次残差项来构建 Lyapunov 函数，但在折扣因子 $\gamma$ 较小时条件过于保守。例如标量系统 $x_{k+1}=2x_k+u_k$，真实稳定阈值为 $\gamma > 1/3$，但经典 LMI 方法只能认证 $\gamma > 0.8090$，差距巨大。

## 核心问题

1. 如何利用 RL 值函数构建有效的稳定性证书？
2. 如何降低经典 Lyapunov 条件的保守性，使更多合理的 RL 策略能被认证？
3. 如何将稳定性认证扩展到非线性系统和联合策略-证书合成场景？

## 方法详解

### 1. Generalized Lyapunov Function 定义

放松经典的逐步严格递减条件，允许 Lyapunov 函数在单步内暂时增大，只要在 $M$ 步的加权平均上递减：

$$\frac{1}{M}\sum_{i=1}^{M}\sigma_i(\mathbf{x})V(\mathbf{x}_i) - V(\mathbf{x}) < 0$$

其中 $\sigma_i(\mathbf{x}) \geq 0$ 为状态相关的非负权重，满足 $\frac{1}{M}\sum_{i=1}^M \sigma_i(\mathbf{x}) \geq 1$。论文证明了在这种放松条件下，原点仍然是渐近稳定的（Theorem 4.2）。

### 2. 线性系统（LQR）中的理论分析

对折扣 LQR 问题，将最优值函数 $J_\gamma^*(\mathbf{x})=\mathbf{x}^\top \mathbf{P}_\gamma \mathbf{x}$ 增广为：

$$V(\mathbf{x}) = J_\gamma^*(\mathbf{x}) + \frac{1}{\varpi}\mathbf{x}^\top \mathbf{S}_0 \mathbf{x}$$

通过求解一组多步 LMI 条件（Theorem 4.4）可以认证稳定性。多步公式化在权重 $\sigma_i$ 上提供了额外自由度，使得可认证的 $\gamma$ 下界显著降低。在标量例子中，$M=2$ 时下界从 0.809 改善到 0.623，随着 $M$ 增大逐渐逼近真实阈值 $\gamma^*=1/3$。

### 3. 非线性系统的 RL 策略认证（Post-hoc）

对已训练的 RL 策略 $\boldsymbol{\pi}_{\text{RL}}$，构建 Generalized Lyapunov 候选函数：

$$V(\mathbf{x};\boldsymbol{\theta}_1) = |J_\gamma^{\boldsymbol{\pi}_{\text{RL}}}(\mathbf{x}) - J_\gamma^{\boldsymbol{\pi}_{\text{RL}}}(\mathbf{0})| + |\varphi(\mathbf{x};\boldsymbol{\theta}_1) - \varphi(\mathbf{0};\boldsymbol{\theta}_1)| + \beta\|\mathbf{x}\|^2$$

其中 $\varphi$ 为神经网络残差项，$\beta\|\mathbf{x}\|^2$ 保证严格正定性。同时引入步权重网络 $\sigma(\mathbf{x};\boldsymbol{\theta}_2)$，输出层用 softmax ×M 保证权重之和为 $M$。联合训练 $\boldsymbol{\theta}_1, \boldsymbol{\theta}_2$，最小化多步下降条件违反的 ReLU 损失。

### 4. 联合策略-证书合成

扩展到同时学习神经控制器 $\boldsymbol{\pi}(\mathbf{x};\boldsymbol{\phi})$ 和 Lyapunov 证书 $V(\mathbf{x};\boldsymbol{\theta}_1)$。目标是最大化认证的吸引域（ROA）体积。训练使用 stability loss + region loss + L1 正则化，通过 PGD 进行反例挖掘（falsification）。训练完成后用 $\alpha$-$\beta$-CROWN 验证器进行形式化验证。

关键定理（Theorem 6.2）：即便广义条件不保证子水平集 $\mathcal{S}$ 的前向不变性，$\mathcal{S}$ 仍是 ROA 的有效内逼近，且原点渐近稳定。

## 实验关键数据

**Post-hoc 认证（固定策略）：**

| 环境 | RL 方法 | M | 测试点数 | 满足下降条件比例 |
|------|---------|---|----------|-----------------|
| Inverted Pendulum | PPO, SAC, TD-MPC | 15 | 10,000 | 100% |
| Cartpole | SAC, TD-MPC | 20 | 1,000,000 | 100% |

**联合合成的认证 ROA 体积：**

| 系统 | M=1 | M=2 | M=3 |
|------|-----|-----|-----|
| Inverted Pendulum | 42.9±1.2 | 76.7±1.3 | 89.2±1.2 |
| Path Tracking | 21.8±0.6 | 23.6±0.5 | 23.9±0.5 |
| 2D Quadrotor | 103.5±1.8 | 109.1±2.0 | 113.7±2.0 |

增大 $M$ 一致性地增大了认证区域，但验证时间也相应增加（如 Inverted Pendulum 从 11.7s 增至 39.2s）。

**步权重分布分析：** 学到的权重集中在 horizon 末端（80–100% 区间占 30–38%），说明网络学会了"容忍初始非单调暂态、依赖后期单调下降"的策略。

## 亮点

1. **理论优雅**：从 LQR 的精确分析出发获得直觉，再推广到非线性系统，逻辑链条清晰
2. **实用性强**：可直接对已训练的 RL 策略（PPO/SAC/TD-MPC）做 post-hoc 认证，无需重新训练策略
3. **广义条件降低保守性**：多步加权平均下降条件显著放松了经典 Lyapunov 要求，LQR 实例中认证阈值从 0.809 改善到接近真实值 0.333
4. **联合合成更大 ROA**：在 Inverted Pendulum 上 M=3 的 ROA 体积是 M=1 的两倍以上
5. **开源实现**：提供完整代码，可复现所有实验

## 局限与展望

1. **$M$ 的选择**：horizon 长度 $M$ 在训练前固定，缺乏针对给定系统自动确定最优 $M$ 的方法
2. **高维系统未验证**：实验仅涉及低维系统（最高 6 维状态），未测试类人机器人或灵巧操控等高维场景
3. **联合合成中权重固定**：由于形式化验证工具的限制，联合合成阶段 $\sigma_i$ 不能用神经网络参数化，只能通过网格搜索选取固定值
4. **验证时间随 M 增长**：$M=3$ 时 2D Quadrotor 的验证时间超过 5600 秒，扩展性是瓶颈
5. **确定性假设**：理论分析限于确定性系统，虽然实验中应用到了随机环境，但缺乏理论保证

## 与相关工作的对比

| 方法 | 特点 | 本文优势 |
|------|------|----------|
| 经典 Lyapunov (Chang 2019, Wu 2023) | 要求单步严格递减 | 多步条件更易满足，可认证更多策略 |
| Yang 2024 (Lyapunov-stable) | 联合合成 + α,β-CROWN 验证 | 本文在其框架上替换为广义条件，ROA 更大 |
| Berkenkamp 2017 | 基于 GP 的安全 RL | 需要模型先验，本文 model-free |
| k-Inductive 方法 | 仅最后一步递减 | 本文的加权平均更灵活，权重可自适应 |

## 启发与关联

- **值函数作为 Lyapunov 候选函数的增广思路**具有一般性，可推广到其他需要稳定性保证的学习控制场景
- **多步放松条件**的思想可类比 k-step return 在 RL 中的成功——更长的 horizon 提供更准确的信号
- 未来可探索**部分状态稳定性认证**（如只认证任务相关状态分量），这对高维机器人系统尤为重要
- 与 robust Lyapunov / input-to-state stability 的结合是自然的扩展方向

## 评分

- 新颖性: ⭐⭐⭐⭐ — 广义 Lyapunov 条件本身已有先例，但与 RL 值函数增广结合是新颖的
- 实验充分度: ⭐⭐⭐⭐ — 线性/非线性、post-hoc/联合合成均有验证，但缺乏高维实验
- 写作质量: ⭐⭐⭐⭐⭐ — 从 LQR 直觉到非线性推广的叙事结构非常清晰
- 价值: ⭐⭐⭐⭐ — 弥合了 RL 与控制理论稳定性分析的重要差距
