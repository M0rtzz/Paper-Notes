---
title: >-
  [论文解读] Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games
description: >-
  [ICML2025][heavy ball momentum] 通过连续时间ODE建模，系统揭示了Heavy Ball动量在min-max博弈中与极小化问题截然不同的行为：**更小的动量**（包括负动量）能扩大收敛步长范围并引导轨迹走向更浅梯度区域，而**交替更新**比同步更新收敛更快且放大了这一正则化效应。
tags:
  - ICML2025
  - heavy ball momentum
  - min-max games
  - 连续时间分析
  - 隐式梯度正则化
  - 交替更新
  - 负动量
  - 局部收敛
---

# Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games

**会议**: ICML2025  
**arXiv**: [2505.19537](https://arxiv.org/abs/2505.19537)  
**代码**: 无  
**领域**: 其他/优化理论  
**关键词**: heavy ball momentum, min-max games, 连续时间分析, 隐式梯度正则化, 交替更新, 负动量, 局部收敛

## 一句话总结

通过连续时间ODE建模，系统揭示了Heavy Ball动量在min-max博弈中与极小化问题截然不同的行为：**更小的动量**（包括负动量）能扩大收敛步长范围并引导轨迹走向更浅梯度区域，而**交替更新**比同步更新收敛更快且放大了这一正则化效应。

## 研究背景与动机

**现状**：Heavy Ball (HB) 动量自 Polyak (1964) 以来在极小化优化中被广泛研究，其机制通常用连续时间微分方程（描述质量在摩擦介质中运动的物理过程）来解释。HB 动量也是 Adam 等实用 min-max 算法的关键组件，被大量用于 GAN 训练和对抗训练。

**痛点**：

**理论空白**：HB 动量在 min-max 博弈中的作用几乎未被探索。极小化中的结论（如更大动量有利）不能直接搬用——Gidel et al. (2019) 发现 **负动量** 反而能改善 min-max 训练，这与直觉相悖。

**更新规则影响不明**：同步更新 (Sim-HB) 和交替更新 (Alt-HB) 对学习动态有本质不同影响，但缺乏定量分析。Gidel et al. (2019) 仅在双线性博弈的特殊情况下证明了交替+负动量才能收敛。

**全局行为理解不足**：在极小化中 HB 的全局性质（如隐式正则化）已有研究，但在 min-max 中完全缺失。

**核心问题**：动量参数 $\beta$ 和更新规则（同步 vs 交替）如何影响 min-max 博弈的学习动态？

## 方法详解

### 整体框架

作者采用**连续时间分析**框架，将离散 HB 算法建模为 ODE，从局部收敛和隐式梯度正则化两个角度展开分析。

### 1. 连续时间模型推导

对 min-max 博弈 $\min_x \max_y f(x,y)$，考虑两种 HB 算法：

- **同步 HB (Sim-HB)**：两个玩家同时更新策略
- **交替 HB (Alt-HB)**：x-玩家先更新，y-玩家再基于更新后的 $x_{n+1}$ 更新

定义修正损失函数：

$$\mathcal{F}(x,y) = \frac{1}{1-\beta} f(x,y) + \frac{h(1+\beta)}{4(1-\beta)^3}\left(\|\nabla_x f\|^2 - \|\nabla_y f\|^2\right)$$

对应的连续时间模型：

- **Continuous Sim-HB**：$\dot{x} = -\nabla_x \mathcal{F}$，$\dot{y} = \nabla_y \mathcal{F}$
- **Continuous Alt-HB**：x 方程同上，y 方程额外包含 $-\frac{h}{2(1-\beta)^2}\|\nabla_x f\|^2$ 项

**关键设计**：不同于极小化中常用的二阶 ODE 建模，本文采用**一阶 ODE**，原因在于：(a) 二阶方法耦合步长和动量参数，阻碍独立分析；(b) 二阶方法要求 $\beta > 0$ 以符合物理直觉，无法探索负动量。

**近似精度 (Theorem 3.1)**：在有限时间内，连续时间轨迹与离散算法轨迹的局部误差为 $\mathcal{O}(h^3)$，优于 Bailey et al. (2020) 的 $\mathcal{O}(h^2)$ 模型——后者无法正确预测同步 GDA 在双线性博弈中发散的行为。

### 2. 局部收敛分析

核心工具是 Jacobian 矩阵分析。梯度流的 Jacobian 可分解为：

$$\mathcal{J} = \mathcal{S} + \mathcal{A}$$

其中 $\mathcal{S}$（势能部分）为对角块矩阵，$\mathcal{A}$（哈密顿部分）为反对称矩阵，捕捉玩家交互。

**核心假设 (Assumption 4.4)**：$\mathcal{A} \gg \mathcal{S}$，即玩家交互主导博弈动态，此时 Jacobian 特征值满足 $|\text{Im}(\lambda)| > |\text{Re}(\lambda)|$。

**关键定理**：

- **Theorem 4.6 (Sim-HB 收敛条件)**：当步长满足 $h < \min_\lambda \frac{2(1-\beta)^2}{(1+\beta)} \frac{|\text{Re}(\lambda)|}{\text{Im}(\lambda)^2 - \text{Re}(\lambda)^2}$ 时，Continuous Sim-HB 实现局部收敛。
- **Corollary 4.7 (负动量扩大稳定性)**：由于 $\frac{2(1-\beta)^2}{1+\beta}$ 关于 $\beta$ 递减，**更小的 $\beta$（包括负值）允许更大的步长范围**。这与极小化中更大动量扩大步长范围的结论相反。
- **Theorem 4.8 (正动量的最优性)**：当步长足够小时，最优收敛速率对应的动量 $\tilde{\beta}(h)$ 是**正数**。
- **Theorem 4.9 (交替更新加速)**：Alt-HB 的 Jacobian 最大实部特征值满足 $\text{Re}(\lambda'_A) = \text{Re}(\lambda_S) - \frac{h}{(1-\beta)^2}|\sigma|^2 + \mathcal{O}(h^2 + h\alpha)$，比 Sim-HB 多出 $-\frac{h}{(1-\beta)^2}|\sigma|^2$ 的负项，实现**指数级加速的局部收敛**。

### 3. 隐式梯度正则化分析

分析 Continuous Sim-HB 中 $\|\nabla_x f\|^2$ 演化的两个竞争力：

- **梯度下降力** (式5)：系数 $-\frac{h(1+\beta)}{4(1-\beta)^3}$，倾向减小梯度范数
- **梯度上升力** (式6)：系数 $+\frac{h(1+\beta)}{4(1-\beta)^3}$，倾向增大梯度范数

由于 $\frac{h(1+\beta)}{4(1-\beta)^3}$ 关于 $\beta$ 递增，更小 $\beta$ 同时削弱两个力。在 $\mathcal{A} \gg \mathcal{S}$ （交互主导）条件下，上升力的减弱超过下降力的减弱，因此**更小动量引导轨迹走向更浅梯度区域**。

对 Alt-HB，$\nabla_y\|\nabla_x f\|^2$ 的系数从 $\frac{h(1+\beta)}{4(1-\beta)^3}$（始终正）变为 $\frac{h(3\beta-1)}{4(1-\beta)^3}$（当 $\beta < 1/3$ 时为负），意味着 y-玩家从最大化梯度变为**最小化梯度**，进一步放大正则化效应。

## 实验/理论验证

### 2D 测试函数实验

| 测试函数 | 动态行为 | 观察结论 |
|----------|----------|----------|
| $f(x,y) = -xy^2, x \geq 0$ | 收敛到均衡 | 更小 $\beta$ → 轨迹趋向更小梯度的均衡点 |
| $f = 3x(4y-0.45)+g(x)-g(y)$（六次多项式） | 收敛到极限环 | 更小 $\beta$ → 极限环平均斜率更低 |

两个测试函数均验证：(1) 更小动量 → 更浅梯度区域；(2) 交替更新 < 同步更新的平均斜率。

### GAN 训练实验

- **设置**：WGAN-GP，CIFAR-10，ResNet-32 架构，学习率 2e-4 线性衰减，batch size 64
- **度量**：累积平均斜率 $\text{AvgS}_\beta(t) = \frac{1}{t}\sum_{s=1}^t (\|\nabla_x f\|^2 + \|\nabla_y f\|^2)$

| 实验 | 观察 |
|------|------|
| 不同 $\beta$（交替更新） | 更小动量 → 更低平均斜率 |
| 交替 vs 同步（固定 $\beta$） | 交替更新平均斜率更低 |
| 斜率与 FID 的关系 | 更低平均斜率 → 更低 FID（更好生成质量） |

### 一致性验证

- $\beta = 0$ 时模型退化为 Rosca et al. (2021) 的 GDA 连续时间方程（一致性）
- 在双线性博弈中：Sim-HB 发散 + Alt-HB 负动量收敛 → 与 Gidel et al. (2019) 的离散结论一致（Proposition 3.2）

## 亮点与洞察

1. **核心对比发现**：min-max 博弈中动量的作用与极小化**完全相反** —— 更小/负动量有利，而非更大/正动量。这一发现有重要的实践指导意义。
2. **一阶 ODE 建模创新**：绕过传统二阶 ODE 的限制，实现步长与动量的独立分析，且支持负动量研究。
3. **$\mathcal{O}(h^3)$ 精度**：比现有 $\mathcal{O}(h^2)$ 模型更准确，能正确预测 Sim-GDA 在双线性博弈中的发散行为。
4. **理论解释实践现象**：首次从隐式正则化角度解释了为何负动量能改善 GAN 训练稳定性 —— 引导轨迹远离尖锐梯度区域。
5. **交替更新的定量优势**：Theorem 4.9 精确刻画了 Alt-HB vs Sim-HB 的收敛速率差异，给出 $\frac{h|\sigma|^2}{(1-\beta)^2}$ 的加速量。

## 局限与展望

1. **未考虑随机梯度**：所有分析基于确定性梯度，未覆盖 SGD/Adam 等随机设置，而实践中几乎都用 mini-batch。
2. **强交互假设**：核心结论依赖 Assumption 4.4（$\mathcal{A} \gg \mathcal{S}$），对弱交互博弈不一定成立。
3. **Alt-HB 加速的条件性**：Theorem 4.9 要求 $h$ 和 $\alpha$ 足够小，附录 I.4 的反例表明不满足时交替未必优于同步。
4. **维度限制**：Theorem 4.9 要求 $m = n$（两玩家维度相同），限制了一般性。
5. **隐式正则化与泛化的关联**：GAN 实验观察到更低斜率 → 更低 FID，但缺乏严格理论支撑，仅为经验观察。
6. **摄动理论的局限**：Theorem 4.9 中 $\mathcal{O}(h\alpha)$ 项的系数需计算摄动归一化特征向量，作者将其留为未来工作。

## 相关工作与启发

- **Gidel et al. (2019)**：首次经验性发现负动量在 min-max 中的好处，本文给予系统理论支撑
- **Ghosh et al. (2023)**：极小化中 HB 隐式正则化分析（更大动量 → 更浅坡度），本文证明 min-max 中结论恰好相反
- **Barrett & Dherin (2021)**：隐式梯度正则化概念的来源
- **Rosca et al. (2021)**：GDA 的连续时间方程（本文 $\beta=0$ 特例）
- **Wang & Chizat (2024)**：梯度流局部收敛的 Assumption 4.5 来源

**启发**：(1) 可将随机噪声引入框架，推导 SDE 模型；(2) 隐式正则化效应与 GAN 训练质量的关联值得深入理论化；(3) 负动量 + 交替更新的组合策略可作为实践 min-max 训练的指导。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统分析 HB 动量在 min-max 中的连续时间行为，揭示与极小化的本质差异
- 理论深度: ⭐⭐⭐⭐⭐ — 多个定理环环相扣，从 Jacobian 分析到隐式正则化形成完整理论体系
- 实验充分度: ⭐⭐⭐⭐ — 2D 函数 + GAN 训练覆盖多种动态模式，但缺少大规模实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机到理论到实验的逻辑链完整
- 实用价值: ⭐⭐⭐⭐ — 为 min-max 训练中动量选择和更新规则提供可操作的理论指导

<!-- RELATED:START -->

## 相关论文

- [Semi-infinite Nonconvex Constrained Min-Max Optimization](../../NeurIPS2025/others/semi-infinite_nonconvex_constrained_min-max_optimization.md)
- [Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)
- [Deep Continuous-Time State-Space Models for Marked Event Sequences](../../NeurIPS2025/others/deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [Bayesian Network Structural Consensus via Greedy Min-Cut Analysis](../../AAAI2026/others/bayesian_network_structural_consensus_via_greedy_min-cut_analysis.md)
- [Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics](optimal_sensor_scheduling_and_selection_for_continuous-discrete_kalman_filtering.md)

<!-- RELATED:END -->
