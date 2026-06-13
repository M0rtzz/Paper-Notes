---
title: >-
  [论文解读] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction
description: >-
  [NeurIPS 2025][时间序列][跳跃扩散] 提出 Neural MJD，用神经网络参数化非平稳 Merton 跳跃扩散模型，将预测建模为 SDE 仿真问题，结合时变 Itô 扩散（捕获连续漂移）和时变复合 Poisson 过程（建模突变跳跃），配合似然截断和 Euler-Maruyama with Restart 求解器实现可扩展学习与推理。
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "跳跃扩散"
  - "随机微分方程"
  - "非平稳"
  - "时间序列预测"
  - "金融建模"
---

# Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction

**会议**: NeurIPS 2025  
**arXiv**: [2506.04542](https://arxiv.org/abs/2506.04542)  
**代码**: [https://github.com/DSL-Lab/neural-MJD](https://github.com/DSL-Lab/neural-MJD)  
**领域**: 时间序列  
**关键词**: 跳跃扩散, 随机微分方程, 非平稳, 时间序列预测, 金融建模

## 一句话总结

提出 Neural MJD，用神经网络参数化非平稳 Merton 跳跃扩散模型，将预测建模为 SDE 仿真问题，结合时变 Itô 扩散（捕获连续漂移）和时变复合 Poisson 过程（建模突变跳跃），配合似然截断和 Euler-Maruyama with Restart 求解器实现可扩展学习与推理。

## 研究背景与动机

**领域现状**: 深度学习时间序列方法（Transformer、SSM 等）在标准基准上表现优异，但本质上是确定性模型或黑盒概率模型，缺乏对底层随机过程的显式建模。

**现有痛点**: 
   - 实际时间序列常表现为**连续趋势+突变跳跃**的混合（如股价因突发新闻暴涨暴跌、零售收入因促销骤变）
   - 经典统计模型（MJD、Lévy 过程）假设独立平稳增量，无法处理非平稳性，也难以捕获多变量间的依赖
   - 深度学习方法缺乏显式跳跃建模，生成模型（扩散模型等）计算昂贵且不直接建模突变

**核心矛盾**: 经典跳跃扩散理论精美但参数固定 → 实际需要时变参数；深度学习灵活但缺乏数学建模 → 需要两者融合。

**本文目标**: 设计一个既有显式 SDE 数学形式（可解释、可做概率推理），又能通过神经网络灵活学习时变参数的时间序列预测模型。

**切入角度**: 将经典 MJD 的固定参数 $\{\mu, \sigma, \lambda, \nu, \gamma\}$ 替换为神经网络预测的时变参数 $\{\mu_t, \sigma_t, \lambda_t, \nu_t, \gamma_t\}$，保持 SDE 框架的数学结构。

**核心 idea**: 用一个条件于历史数据和上下文的神经网络在单次前向传播中预测所有未来时刻的 SDE 参数，从而实现非平稳跳跃扩散的可扩展学习。

## 方法详解

### 整体框架

非平稳 Neural MJD 的 SDE：

$$dS_t = S_t \big((\mu_t - \lambda_t k_t)dt + \sigma_t dW_t + \int_{\mathbb{R}^d}(y-1)N(dt, dy)\big)$$

其中 $\mu_t, \sigma_t, \lambda_t, \nu_t, \gamma_t = f_\theta(S_0, S_{-1}, \ldots, S_{-T_p}, C, t)$

### 关键设计

#### 1. 时变参数的神经网络预测

**功能**: 单次前向传播输出所有未来时刻的 SDE 参数。

**核心思路**: 
- 输入：历史序列 $\{S_{-T_p}, \ldots, S_0\}$ + 可选上下文 $C$（空间图、特征等）
- 输出：$\{(\mu_\tau, \sigma_\tau, \lambda_\tau, \nu_\tau, \gamma_\tau)\}_{\tau=1}^{T_f}$
- 跳跃时刻 $N_t$ 服从时变强度 $\lambda_t$ 的 Poisson 过程
- 跳跃幅值 $\ln Y_t \sim \mathcal{N}(\nu_t, \gamma_t^2)$

**设计动机**: 不同于 PINN 把手工设计的 SDE 作为约束加入损失函数，Neural MJD 直接参数化非平稳 SDE，更灵活。

#### 2. 似然截断机制

**功能**: 使无穷级数似然可计算。

**核心问题**: 精确似然 $P(\ln S_T | \mathcal{C}) = \sum_{n=0}^{\infty} P(N_t=n) \cdot \phi(\ln S_T; a_n, b_n^2)$ 是无穷级数。

**解决方案**: 截断到最多 $\kappa$ 次跳跃。

**理论保证** (Theorem 4.1): 截断误差 $\Psi_\kappa(t, \delta)$ 以超指数速率衰减：$O(\kappa^{-\kappa})$。实验中 $\kappa = 5$ 即足够。

**分段常参数**: 将连续时间离散化，在每个整数区间 $[\tau-1, \tau)$ 内参数保持恒定，避免数值积分。

#### 3. 参数自举（Bootstrapping）训练

**功能**: 缓解训练和推理之间的分布不匹配。

**核心思路**: 计算 $\ln P(\ln S_\tau | S_{\tau-1}, \mathcal{C})$ 时，不使用真实 $S_{\tau-1}$（teacher forcing），而是用条件均值预测 $\hat{S}_{\tau-1} = \mathbb{E}[S_{\tau-1} | \mathcal{C}]$ 替代。

条件均值有闭式解：$\mathbb{E}[S_t | \mathcal{C}] = S_0 \exp(\sum_{j=1}^{\rho_t - 1} \mu_j + (t - \rho_t + 1)\mu_{\rho_t})$

#### 4. Euler-Maruyama with Restart 求解器

**功能**: 提高推理时轨迹采样的精度和稳定性。

**核心思路**: 
- 标准 EM 方法中误差随时间指数累积：$\epsilon_t^E \leq K\exp(Lt)/M$
- Restart 策略在每个整数时间点用解析条件均值重置状态
- 误差界改善为：$\epsilon_t^R \leq K\exp(L(t-\lfloor t\rfloor))/M$

**效果**: 指数项中 $t$ 变为 $t - \lfloor t\rfloor \in [0,1)$，有效抑制长时域误差累积。

### 损失函数/训练策略

总损失：$\mathcal{L} = -\sum_{\tau=1}^{T_f} \psi_\tau + \omega \|S_\tau - \hat{S}_\tau\|^2$

- 第一项：MLE 似然（截断到 $\kappa=5$ 次跳跃）
- 第二项：条件均值正则化（鼓励均值预测接近真值）
- for 循环可并行化（条件均值计算不依赖顺序步骤）

## 实验关键数据

### 合成数据（标准 MJD 生成）

| 模型 | MAE↓ | R²↑ | minMAE↓ | maxR²↑ |
|------|------|-----|---------|--------|
| ARIMA | 0.29 | -0.15 | — | — |
| MJD | 0.21 | 0.08 | 0.18 | 0.15 |
| Neural BS | 0.15 | 0.25 | 0.10 | 0.35 |
| **Neural MJD** | **0.09** | **0.32** | **0.07** | **0.39** |

Neural MJD 在所有评估协议下均最优。

### SafeGraph 商业分析数据

| 模型 | MAE↓ | R²↑ | 最优K MAE↓ | 最优K R²↑ |
|------|------|-----|-----------|----------|
| GCN | 95.2 | 0.432 | — | — |
| EDM | 57.6 | 0.525 | 49.4 | 0.556 |
| FM | 54.5 | 0.540 | 47.8 | 0.552 |
| Neural BS | 56.4 | 0.539 | 45.6 | 0.561 |
| **Neural MJD** | **54.1** | **0.549** | **42.3** | **0.565** |

### S&P 500 股票数据

| 模型 | MAE↓ | R²↑ | minMAE↓ | maxR²↑ |
|------|------|-----|---------|--------|
| MJD | 64.3 | 0.092 | 40.7 | 0.235 |
| Neural BS | — | — | — | — |
| **Neural MJD** | 最优 | 最优 | 最优 | 最优 |

### 消融实验

| 变体 | 效果 |
|------|------|
| 去掉跳跃组件（→ Neural BS） | 性能下降，尤其在有突变的序列上 |
| Teacher forcing 替代 bootstrapping | 训练-推理不匹配导致误差累积 |
| 标准 EM 替代 Restart EM | 长时域预测方差增大 |
| $\kappa = 1$ vs $\kappa = 5$ | $\kappa = 5$ 更优，更高值改进不大 |

### 关键发现

1. 显式建模跳跃比隐式建模（扩散模型、flow matching）更有效
2. Neural MJD 在确定性评估（均值）和概率评估（最佳K、似然加权）三种协议下一致胜出
3. Restart EM 显著降低长时域推理的方差
4. 参数自举训练策略有效缓解了误差累积问题
5. 模型对架构无限制——实验使用 Transformer backbone + 可选的 Graphormer 空间编码

## 亮点与洞察

- **理论与实践的有机结合**: 保留了 MJD 的数学结构（闭式似然、闭式条件均值），同时通过神经网络获得灵活性
- **似然截断误差界** (Theorem 4.1) 提供了 $O(\kappa^{-\kappa})$ 的超指数衰减保证，$\kappa=5$ 即实际精确
- **Restart EM** (Proposition 4.2) 将弱收敛误差的指数因子从 $\exp(Lt)$ 压到 $\exp(L(t-\lfloor t\rfloor))$
- 不需要跳跃标签——与时间点过程方法（需标注跳跃事件）的关键区别
- 支持连续时间仿真——虽然在离散时刻训练，但可在任意时间分辨率采样

## 局限与展望

1. 分段常参数假设（每个整数区间内参数不变）是简化，对快速变化的参数可能不够细粒度
2. 跳跃幅值假设对数正态分布，对重尾分布可能需要更灵活的参数化
3. 训练时每步需前向传播整个 backbone，长预测时域下计算量大
4. 与扩散生成模型的比较可能不完全公平（backbone 相同但任务目标不同）
5. 未探索多尺度跳跃（不同频率的跳跃过程叠加）

## 相关工作与启发

- 与 Neural ODE/SDE 的关键区别：Neural MJD 显式建模跳跃过程，不仅是连续扩散
- 与 PINN for finance 的区别：PINN 将 BS/MJD 作为 PDE 约束，Neural MJD 直接参数化 SDE
- Restart 策略与 Parareal 算法异曲同工——先粗后细的混合方法
- 可扩展到更一般的 Lévy 过程、Hawkes 过程等复杂随机过程

## 评分

⭐⭐⭐⭐

将经典金融数学（MJD）与深度学习自然融合，设计简洁有效。理论贡献（截断误差界、Restart EM 误差改善）扎实。实验全面涵盖合成和真实数据，三种评估协议严谨。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Time-Scale Invariant Population-Level Neural Representations](learning_time-scale_invariant_population-level_neural_representations.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[NeurIPS 2025\] Graph-based Neural Space Weather Forecasting](graph-based_neural_space_weather_forecasting.md)
- [\[NeurIPS 2025\] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)
- [\[NeurIPS 2025\] Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions](neural_stochastic_flows_solver-free_modelling_and_inference_for_sde_solutions.md)

</div>

<!-- RELATED:END -->
