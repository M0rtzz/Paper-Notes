---
title: >-
  [论文解读] DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients
description: >-
  [AAAI2026][optimization-based control] 提出 DiffOP 框架，将优化型控制策略（如 MPC）视为可微分模块，通过隐式微分推导解析策略梯度，实现端到端强化学习训练，并给出首个非渐近收敛保证。
tags:
  - AAAI2026
  - optimization-based control
  - implicit differentiation
  - policy gradient
  - model predictive control
  - bilevel optimization
---

# DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients

**会议**: AAAI2026  
**arXiv**: [2411.07484](https://arxiv.org/abs/2411.07484)  
**代码**: [alwaysbyx/DiffOP](https://github.com/alwaysbyx/DiffOP)  
**领域**: reinforcement_learning  
**关键词**: optimization-based control, implicit differentiation, policy gradient, model predictive control, bilevel optimization

## 一句话总结

提出 DiffOP 框架，将优化型控制策略（如 MPC）视为可微分模块，通过隐式微分推导解析策略梯度，实现端到端强化学习训练，并给出首个非渐近收敛保证。

## 背景与动机

现实控制系统（电网、机器人、交通网络等）对策略的**可解释性、安全性和鲁棒性**有严格要求。基于优化的控制策略（如 MPC）通过求解带约束的优化问题生成动作，具有天然的可解释性和约束满足能力。

现有方法存在两大问题：

1. **目标不匹配（Objective Mismatch）**：传统方法将动力学模型和代价函数的学习与控制目标解耦——模型在预测精度上可能表现很好，但不一定能引导出最优控制决策。
2. **监督学习局限**：近年可微优化工作（PDP、IDOC 等）主要在监督模仿学习设定下运行，依赖专家示范，无法通过在线交互持续改进策略。
3. **现有 RL+MPC 方法**（RLMPC-TD、RLMPC-DPG）依赖值函数近似和 Q-learning 更新，往往收敛到次优解。

## 核心问题

如何在**强化学习设定**下对隐式定义的优化型控制策略进行端到端训练？具体而言，需要解决：

- 策略由优化问题的**解**隐式定义，如何高效求取策略参数的梯度？
- 能否在不依赖值函数近似的情况下，直接优化实际控制代价？
- 该学习过程是否有理论收敛保证？

## 方法详解

### 整体框架

DiffOP 将控制策略定义为参数化优化问题的解：

$$u_{0:H-1}^{\star}(x_{\text{init}};\theta) = \arg\min_u \sum_{i=0}^{H-1} c(x_i, u_i; \theta_c) + c_H(x_H; \theta_H)$$

其中动力学 $f(\cdot;\theta_f)$、阶段代价 $c(\cdot;\theta_c)$ 和终端代价 $c_H(\cdot;\theta_H)$ 均为可学习参数。策略参数 $\theta = (\theta_c, \theta_H, \theta_f)$ 联合描述了代价模型和动力学模型。

### 双层优化建模

将策略学习建模为双层优化（bi-level optimization）：

- **上层**：最小化真实系统中的期望累积代价 $C(\theta) = \mathbb{E}[\sum_t c(x_t, u_t; \phi_c)]$
- **下层**：在每个决策点求解参数化优化问题得到控制动作

为支持探索，动作通过截断高斯分布采样：$u_i \sim \mathcal{N}(u_i^\star, \sigma^2 I)$，截断范围由超参数 $\beta$ 控制。

### 隐式策略梯度推导

核心技术创新在于利用**隐式函数定理**计算优化解对参数的梯度。具体步骤：

1. 将最优解轨迹 $\zeta^\star$ 的 KKT 条件写出
2. 对 KKT 条件应用隐式微分，得到 $\nabla_\theta u_i^\star$ 的解析表达式（Proposition 1）
3. 结合 REINFORCE 梯度估计器，推导完整策略梯度（Proposition 2）：

$$\nabla_\theta C(\theta) = \mathbb{E}\left[L(\tau)\sum_{t=0}^{T} \frac{1}{\sigma^2}[\nabla_\theta u_t^\star]^{\mathsf{T}}(u_t - u_t^\star)\right]$$

实际中通过 Monte Carlo 采样 $N$ 条轨迹近似。

### 算法流程（Algorithm 1）

每轮迭代包含三步：

1. 采样 $N$ 条轨迹，每步用优化求解器（CasADi）求解策略并加噪声探索
2. 对每条轨迹计算隐式梯度 $\nabla_\theta u_t^\star$
3. 用 Monte Carlo 估计策略梯度，执行梯度下降更新 $\theta$

### 收敛性保证

在有界灵敏度假设（Assumption 1）和有界代价假设（Assumption 2）下，证明了：

$$\min_{k=0,...,K-1} \|\nabla_\theta C(\theta^{(k)})\|^2 \leq \frac{16L_C(C(\theta^{(0)}) - C(\bar\theta))}{K} + 3\epsilon$$

即 DiffOP 在 $\mathcal{O}(1/\epsilon)$ 步内收敛到 $\epsilon$-stationary point，与标准策略梯度方法的收敛率一致。

对无约束强凸情形，进一步证明了灵敏度有界条件可由 Lipschitz 平滑性和强凸性直接保证（Proposition 3）。

### 执行模式

DiffOP 支持两种部署模式：

- **DiffOP (Step)**：每步仅执行优化序列的第一个动作（类似标准 MPC 滚动执行）
- **DiffOP (Traj)**：一次生成并执行完整控制序列（开环，时序一致性更强）

## 实验关键数据

### 非线性控制任务（Cartpole / Robot Arm / Quadrotor）

| 方法 | 训练模式 | 结果 |
|------|----------|------|
| DiffOP (Step) | 在线 RL | 所有任务中收敛最快、最终代价最低 |
| DiffOP (Traj) | 在线 RL | Robot Arm 和 Quadrotor 超越 PDP (Offline) |
| RLMPC-TD | 在线 RL | 常收敛到次优解 |
| RLMPC-DPG | 在线 RL | 不稳定，部分任务代价无法持续下降 |
| PDP (Offline) | 离线监督 | 受限于专家数据质量 |

### 电压控制（IEEE 13-bus，500 场景）

| 方法 | 瞬态代价 | 稳态代价 |
|------|----------|----------|
| **DiffOP (Step)** | **-6.81** | **-0.11** |
| **DiffOP (Traj)** | **-6.80** | **-0.11** |
| TASRL | -6.76 | -0.11 |
| RLMPC-DPG | -6.11 | -0.10 |
| Stable-DDPG | -5.61 | -0.09 |
| PDP (Offline) | -5.86 | -0.09 |
| RLMPC-TD | -4.62 | -0.07 |

DiffOP 在瞬态代价上取得所有方法中最优，稳态代价与 TASRL 持平。训练后电压轨迹稳定在安全运行范围内。

## 亮点

1. **首个非渐近收敛保证**：证明了 $\mathcal{O}(1/\epsilon)$ 的收敛率，填补了优化型策略 RL 训练的理论空白
2. **通用框架**：不依赖 LQR 近似或值函数近似，适用于一般非线性约束优化问题
3. **灵活部署**：统一支持 step-wise（滚动执行）和 trajectory-level（开环执行）两种模式
4. **联合学习代价和动力学**：避免了目标不匹配问题，直接用环境反馈端到端优化
5. **约束处理能力**：在电压控制实验中展示了对硬约束的自然支持

## 局限与展望

1. **强凸假设较强**：理论收敛保证依赖代价函数对控制变量的强凸性，非凸情形下灵敏度可能无界
2. **约束边界处不光滑**：不等式约束的活跃集变化可能导致梯度不连续，论文未覆盖此情形的理论分析
3. **样本效率**：每次策略更新需采样 $N$ 条完整轨迹，计算量和样本量较大
4. **求解器依赖**：每步都需要调用优化求解器（CasADi），推理开销高于纯神经网络策略
5. **探索机制简单**：仅使用独立高斯噪声，未考虑更结构化的探索策略

## 与相关工作的对比

| 维度 | DiffOP | PDP | RLMPC-TD/DPG | Stable-DDPG/TASRL |
|------|--------|-----|-------------|-------------------|
| 训练方式 | 在线 RL | 离线监督 | 在线 RL | 在线 RL |
| 梯度计算 | 隐式微分 | PMP 微分 | Q-learning | 反向传播 |
| 值函数 | 不需要 | 不适用 | 需要 | 需要 |
| 约束支持 | 原生支持 | 支持 | 支持 | 通过设计保证 |
| 收敛保证 | 有（非渐近） | 无 | 无 | 部分（稳定性） |
| 策略形式 | 优化问题的解 | 优化问题的解 | 短视界 MPC | 神经网络 |

## 启发与关联

- **隐式微分 + RL 的范式**可推广到其他优化层嵌入的决策系统（如组合优化、调度问题）
- 双层优化视角为理解 MPC 参数学习提供了统一理论框架
- 电压控制实验展示了在安全关键系统中，优化型策略比黑盒 RL 更具可部署性
- 未来可结合 actor-critic 方法降低方差，或利用 warm-starting 加速求解器

## 评分

- 新颖性: ⭐⭐⭐⭐ — 隐式微分与策略梯度的结合有技术新意，但建立在已有 IDOC/PMP 工作之上
- 实验充分度: ⭐⭐⭐⭐ — 非线性控制和电压控制两组实验覆盖面好，但缺少高维任务和更多真实场景
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨，结构清晰
- 价值: ⭐⭐⭐⭐ — 为优化型策略的 RL 训练提供了理论基础，对安全控制方向有实际意义

<!-- RELATED:START -->

## 相关论文

- [Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)
- [HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)
- [Test-driven Reinforcement Learning in Continuous Control](test-driven_reinforcement_learning_in_continuous_control.md)
- [Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning](../../ICLR2026/reinforcement_learning/policyflow_policy_optimization_with_continuous_normalizing_flow_in_reinforcement.md)

<!-- RELATED:END -->
