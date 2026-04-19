---
title: >-
  [论文解读] HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning
description: >-
  [ICLR2026][科学计算][KKL observer] 提出 HyperKKL，用超网络（hypernetwork）编码外源输入信号并即时生成 KKL 观测器的变换映射参数，使非自治非线性系统的状态估计无需重新训练或在线梯度更新，在 Duffing、Van der Pol、Lorenz、Rössler 四个经典非线性系统上验证了方法的有效性和局限性。
tags:
  - ICLR2026
  - 科学计算
  - KKL observer
  - state estimation
  - hypernetwork
  - non-autonomous system
  - dynamical system
---

# HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning

**会议**: ICLR2026  
**arXiv**: [2602.22630](https://arxiv.org/abs/2602.22630)  
**代码**: 待确认  
**领域**: scientific_computing  
**关键词**: KKL observer, state estimation, hypernetwork, non-autonomous system, dynamical system

## 一句话总结

提出 HyperKKL，用超网络（hypernetwork）编码外源输入信号并即时生成 KKL 观测器的变换映射参数，使非自治非线性系统的状态估计无需重新训练或在线梯度更新，在 Duffing、Van der Pol、Lorenz、Rössler 四个经典非线性系统上验证了方法的有效性和局限性。

## 背景与动机

**领域现状**：状态估计（state estimation）——从部分可观测的测量中重建动力系统的完整内部状态——是控制和工程中的基础问题。KKL（Kazantzis-Kravaris/Luenberger）观测器通过将非线性动力系统浸入（immerse）一个更高维的稳定线性潜空间来实现状态估计，理论上在后向可区分性（backward distinguishability）条件下保证全局收敛。

**现有痛点**：

- KKL 观测器的核心是求解一个解析上不可解的偏微分方程（PDE）：$\frac{\partial \mathcal{T}}{\partial x}(x) f(x) = A \mathcal{T}(x) + B h(x)$
- 近年来基于神经网络的方法（PINN、自编码器等）可以近似求解这些映射，但几乎全部针对**自治系统**（autonomous system，无外部输入 $u(t)$）
- 真实系统几乎从不自治——机器人接受电机指令、生物系统响应外部刺激、工业过程受时变扰动

**核心矛盾**：扩展到非自治系统时，变换映射 $\mathcal{T}$ 需要变为输入依赖的 $\mathcal{T}(x, t)$，满足时变 PDE：

$$\frac{\partial \mathcal{T}}{\partial x}(x,t) f(x, u(t)) + \frac{\partial \mathcal{T}}{\partial t}(x,t) = A \mathcal{T}(x,t) + B h(x)$$

额外的时间偏导项 $\frac{\partial \mathcal{T}}{\partial t}$ 将变换与输入的时间演化耦合，使得静态映射无法胜任。现有学习方法要么需要针对每个输入场景重新训练，要么需要在线梯度更新，严重限制了实用性。

**本文方案**：用超网络将输入信号的历史编码为观测器参数的即时扰动，实现在推理时自适应于不同外源输入条件，无需重训练或在线优化。

## 方法详解

### 整体框架

HyperKKL 采用两阶段顺序训练：

1. **Phase 1（自治预训练）**：在无外部输入（$u \equiv 0$）条件下，使用 physics-informed 损失训练基础编码器 $\hat{\mathcal{T}}_{\theta^{\text{base}}}$ 和解码器 $\hat{\mathcal{T}}^*_{\phi^{\text{base}}}$，满足自治 KKL 条件。训练完成后冻结这些参数
2. **Phase 2（超网络训练）**：冻结基础映射，仅训练超网络参数 $\psi$，学习从输入信号到参数扰动的映射

推理时：新输入信号 → LSTM 编码 → 生成参数扰动 $\Delta\theta, \Delta\phi$ → 叠加到冻结的基础参数上 → 即时得到输入自适应的观测器。

学习目标结合重建损失和 PDE 残差：

$$\min_\psi \mathbb{E}_{(x,u) \sim \mathcal{D}} \left[ \underbrace{\| x - \hat{\mathcal{T}}^*(\hat{\mathcal{T}}(x; \theta_u), \phi_u) \|^2}_{\mathcal{L}_{\text{rec}}} + \lambda \underbrace{\left\| \frac{\partial \hat{\mathcal{T}}}{\partial x} f(x,u) + \frac{\Delta \hat{\mathcal{T}}}{\Delta t} - A\hat{\mathcal{T}} - Bh(x) \right\|^2}_{\mathcal{L}_{\text{PDE}}} \right]$$

### 关键设计 1：Dynamic HyperKKL（动态变换方法）

对于输入会持续重塑吸引子几何结构的复杂系统，需要真正的时变变换 $\mathcal{T}(x, \theta(t))$。Dynamic HyperKKL 使用残差超网络，将基础参数和输入条件扰动分离：

$$\theta_{\text{enc}}(t) = \theta_{\text{enc}}^{\text{base}} + \Delta\theta_{\text{enc}}(u_{[t-w, t]})$$
$$\phi_{\text{dec}}(t) = \phi_{\text{dec}}^{\text{base}} + \Delta\phi_{\text{dec}}(u_{[t-w, t]})$$

超网络包含三个组件：
- **共享 LSTM 编码器**：处理输入窗口 $u_{[t-w, t]}$（窗口大小 $w = 100$），输出隐状态 $h_t \in \mathbb{R}^{d_h}$
- **编码器头 MLP**：从 $h_t$ 预测 $\Delta\theta_{\text{enc}}$
- **解码器头 MLP**：从 $h_t$ 预测 $\Delta\phi_{\text{dec}}$

**分块预测策略（chunked prediction）**：直接预测完整参数扰动维度过高。将目标权重矩阵 $W \in \mathbb{R}^{m \times n}$ 分块，MLP 独立预测每个块，保持表达能力的同时控制输出维度。

**残差结构保证**：当 $u \equiv 0$ 时，LSTM 隐状态产生 $\Delta\theta = \Delta\phi = 0$，精确恢复自治观测器——确保无外部输入时不退化。

时间偏导的估计使用有限差分：

$$\frac{\Delta \hat{\mathcal{T}}}{\Delta t} \approx \frac{\hat{\mathcal{T}}(x; \theta(u_{[t, t+\Delta t]})) - \hat{\mathcal{T}}(x; \theta(u_{[t-\Delta t, t]}))}{\Delta t}$$

### 关键设计 2：Static HyperKKL（静态变换方法）

对于输入仅作为有界扰动存在的简单系统，保留自治变换 $\mathcal{T}(x)$ 不变，仅在观测器动力学中添加学习的输入注入项：

$$\dot{\hat{z}} = A\hat{z} + By + \bar{\varphi}(\hat{z}, u; \xi)$$

其中 $\bar{\varphi}$ 是小型 MLP，由 LSTM 编码的输入上下文和 $\hat{z}$ 共同输入。训练约束 $\bar{\varphi}$ 在 $u = 0$ 时输出零。

### 关键设计 3：自适应课程学习基线

作为对照，论文还评估了纯训练策略能否弥补静态架构的局限。使用课程学习将训练数据按输入复杂度分级（$\mathcal{D}_1$: 常数 → $\mathcal{D}_2$: 低频正弦 → ... → 高频混合），训练在当前级别损失停滞后推进到下一级。这一基线测试的问题是：**在不改变架构的情况下，仅通过更丰富的训练数据能否解决非自治问题？**

## 实验结果

### 主实验：四个非线性系统上的状态估计性能（RMSE / SMAPE%）

**振荡系统（Duffing、Van der Pol）**：

| 方法 | Duffing-Zero | Duffing-Sin | Duffing-Sqr | VdP-Zero | VdP-Sin | VdP-Sqr |
|:-----|:------------|:------------|:------------|:---------|:--------|:--------|
| Autonomous | 0.04 (5.6) | 0.26 (26) | 0.33 (31) | 0.15 (7.0) | 0.23 (9.8) | 0.25 (10.5) |
| Curriculum | 0.27 (33) | 0.44 (41) | 0.57 (46) | 1.10 (51.4) | 1.15 (51.5) | 1.15 (51.7) |
| Static HyperKKL | **0.04** (5.6) | **0.10↓** (9.3) | **0.17↓** (14) | 0.12↓ (5.3) | 0.24 (10.2) | 0.25 (10.8) |
| Dynamic HyperKKL | 0.08 (8.2) | 0.24↓ (25) | 0.27↓ (28) | **0.12↓** (5.0) | **0.21↓** (8.6) | **0.22↓** (9.1) |

**混沌系统（Rössler、Lorenz）**：

| 方法 | Rössler-Zero | Rössler-Sin | Rössler-Sqr | Lorenz-Zero | Lorenz-Sin | Lorenz-Sqr |
|:-----|:------------|:------------|:------------|:------------|:-----------|:-----------|
| Autonomous | 1.14 (6.7) | 1.47 (7.6) | 1.48 (8.3) | **5.56** (18) | **5.58** (18) | **5.55** (18) |
| Curriculum | 5.58 (35) | 5.94 (37) | 5.61 (38) | 11.5 (41) | 11.6 (42) | 11.6 (42) |
| Static HyperKKL | 1.14 (6.7) | 1.70 (10) | 1.75 (12) | 5.56 (18) | 16.3 (52) | 16.2 (51) |
| **Dynamic HyperKKL** | **1.01↓** (5.1) | **1.38↓** (6.0) | **1.36↓** (6.9) | 6.67 (22) | 6.67 (22) | 6.66 (22) |

核心发现：

1. **Static HyperKKL 在低维振荡系统上最优**：Duffing 正弦输入 RMSE 降低 62%（0.26 → 0.10），符合理论预期——低维振荡器的吸引子随输入平滑移动，静态变换足够
2. **Curriculum Learning 全面失败**：在所有系统的所有输入条件下性能都**劣于**自治基线（如 VdP-Zero: 0.15 → 1.10），证明瓶颈是**表征性的而非教育性的**
3. **Lorenz 系统暴露根本局限**：自治基线反而最优（RMSE ≈ 5.5），Static HyperKKL 灾难性退化（16.3），Dynamic HyperKKL 也有微弱退化（6.67）

### 消融实验：架构 vs. 训练的分离分析

| 分析维度 | 结论 | 证据 |
|:---------|:-----|:-----|
| 课程学习 vs. 不训练 | 课程学习有害 | 所有系统性能劣于自治基线 |
| Static vs. Dynamic | 系统复杂度决定选择 | 低维用 Static，混沌用 Dynamic |
| 输入编码方式 | LSTM 优于 MLP | 时序聚合对混沌系统关键 |
| $u=0$ 恢复性 | 所有超网络方法正确恢复自治性能 | $\Delta\theta \to 0$ 验证成功 |
| Lorenz 特殊性 | 高灵敏度吸引子使输入条件化引入噪声 | 小误差沿不稳定流形指数放大 |

## 评价

**评分**: ⭐⭐⭐⭐

**优点**：

- 清晰地将 KKL 观测器从自治系统扩展到非自治系统，填补了学习型 KKL 方法的实际空白
- 两阶段训练（自治预训练 + 超网络微调）和残差结构设计合理，保证了 $u=0$ 时的无退化性
- 分块预测策略平衡了超网络的表达能力和输出维度
- 诚实地报告了 Lorenz 系统上的失败案例并提供了深入的理论分析（不稳定流形 + 误差指数放大）
- Static vs. Dynamic 两种架构的对比提供了实用的选择指南

**不足**：

- 仅在 4 个经典低维系统上验证（最高 3 维状态空间），对高维实际系统的可扩展性未知
- Lorenz 系统的失败暴露了超网络条件化在高灵敏度系统上的根本局限，目前没有解决方案
- 课程学习基线的失败可能部分源于实现细节（如超参选择），而非纯粹的架构局限
- 缺少与其他非自治观测器方法（如 EKF、UKF 在非自治场景下）的对比
- 计算开销分析缺失——LSTM 超网络在推理时的延迟是否满足实时控制需求？

**与相关工作的关键区别**：

- 不同于 Niazi et al. (2025) 仅处理自治 KKL，本文通过超网络实现了非自治扩展
- 不同于 Meta-RL 方法（如 MAML）需要在线梯度更新，HyperKKL 纯前向推理即可适应
- 不同于静态变换方法，Dynamic HyperKKL 显式建模时变 PDE 的时间偏导项

<!-- RELATED:START -->

## 相关论文

- [Towards Universal Neural Operators through Multiphysics Pretraining](../../NeurIPS2025/scientific_computing/towards_universal_neural_operators_through_multiphysics_pretraining.md)
- [Hamiltonian Neural PDE Solvers through Functional Approximation](../../NeurIPS2025/scientific_computing/hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [Deep Learning for Subspace Regression](deep_learning_for_subspace_regression.md)
- [Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)

<!-- RELATED:END -->
