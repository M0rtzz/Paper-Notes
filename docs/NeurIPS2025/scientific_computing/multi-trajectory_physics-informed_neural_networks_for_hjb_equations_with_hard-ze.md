---
title: >-
  [论文解读] Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data
description: >-
  [NeurIPS 2025 (Workshop on Generative AI in Finance)][科学计算][Physics-Informed Neural Networks] 针对最优交易执行中 HJB 方程的硬零终端库存约束（$X_T=0$），提出 Multi-Trajectory PINN (MT-PINN)，通过基于轨迹展开的终端损失与 $\lambda$-curriculum 训练策略，在合成数据和 SPY 实盘回测中显著优于 vanilla PINN，终端库存违规率大幅降低。
tags:
  - NeurIPS 2025 (Workshop on Generative AI in Finance)
  - 科学计算
  - Physics-Informed Neural Networks
  - Hamilton-Jacobi-Bellman
  - 最优执行
  - 终端库存约束
  - 金融量化
---

# Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data

**会议**: NeurIPS 2025 (Workshop on Generative AI in Finance)  
**arXiv**: [2512.12708](https://arxiv.org/abs/2512.12708)  
**代码**: [GitHub](https://github.com/anthimevalin/Multi-Trajectory-PINNs-Zero-Terminal-HJB)  
**领域**: scientific_computing  
**关键词**: Physics-Informed Neural Networks, Hamilton-Jacobi-Bellman, 最优执行, 终端库存约束, 金融量化

## 一句话总结
针对最优交易执行中 HJB 方程的硬零终端库存约束（$X_T=0$），提出 Multi-Trajectory PINN (MT-PINN)，通过基于轨迹展开的终端损失与 $\lambda$-curriculum 训练策略，在合成数据和 SPY 实盘回测中显著优于 vanilla PINN，终端库存违规率大幅降低。

## 研究背景与动机
**领域现状**: 最优交易执行是量化金融核心问题，需要在给定时间内清仓（$X_T=0$），涉及市场冲击、风险厌恶等因素。其对应的最优控制问题由 Hamilton-Jacobi-Bellman (HJB) PDE 刻画。

**现有痛点**: Vanilla PINNs 通过 PDE 残差 + 软边界惩罚来训练，但对于硬零终端库存约束（$\tau \to 0$ 时值函数不光滑）的执行力度不足，常产生非零终端库存和不稳定的控制策略。

**核心矛盾**: 终端条件是 $\Gamma(0,X) = 0$ if $X=0$, $+\infty$ otherwise — 这是一个奇异边界条件，PDE 残差损失难以充分捕捉其约束力。

**本文目标**: 如何在 PINN 框架下强有力地执行硬零终端库存约束，同时保持训练稳定性。

**切入角度**: 不仅在 PDE 残差层面训练，还从「控制-轨迹」视角出发，直接模拟执行轨迹并惩罚终端库存偏差。

**核心 idea**: 用 rollout-based trajectory loss + BPTT 传播终端惩罚，配合 $\lambda$-curriculum 从风险中性到风险厌恶逐步过渡。

## 方法详解

### 整体框架
MT-PINN 基于 Gatheral-Schied 最优执行模型，价格 $S_t$ 服从 GBM，库存 $X_t$ 以交易速率 $v_t$ 演化（$\dot{X}_t = -v_t$）。值函数 $\Gamma(\tau, X, S)$ 满足 reduced HJB 方程（风险中性时不依赖 $S$），最优控制 $v^* = \frac{1}{2} \partial\Gamma/\partial X$。MT-PINN 用 MLP 逼近值函数，通过自动微分计算 HJB 残差和最优控制，并额外引入轨迹展开损失。

### 关键设计
1. **Multi-Trajectory Terminal-Inventory Loss**: 从一批初始状态 $\{(X_0^{(p)}, S_0^{(p)})\}_{p=1}^P$ 和多个时间跨度 $\{T_j\}_{j=1}^J$ 出发，用前向 Euler 离散化展开轨迹（200步），得到终端库存 $x_{T_j}^{(p)}$，然后用混合惩罚函数 $\psi(x_T) = |x_T|$ (当 $|x_T| \le 1$) 或 $x_T^2$ (当 $|x_T| > 1$) 平均作为轨迹损失。梯度通过 BPTT 传播回网络参数。
2. **$\lambda$-Curriculum 训练策略**: 风险厌恶参数 $\lambda$ 从 0 逐步增加到目标值 $\lambda^*$：先在 $\lambda=0$（1D 状态空间）训练，再 warm-start 到 $\lambda > 0$（2D 状态空间）。分5个阶段 $\alpha \in (0.25, 0.50, 0.75, 0.9, 1.0)$，每阶段 5k epochs。这避免了直接在高维+强约束下训练的不稳定性。
3. **对称性与内部条件损失**: 利用值函数的对称性质（$\Gamma(\tau,X) = \Gamma(\tau,-X)$ 或 $\Gamma(\tau,X,S) = \Gamma(\tau,-X,-S)$）和内部条件（$\Gamma(\tau,0,S) \le 0$）作为额外约束项。
4. **DWA-style 自适应权重**: 各损失项权重通过 EMA 平滑后动态调整，避免某一项主导训练。

### 损失函数 / 训练策略
总损失：
$$\mathcal{L}_{\text{total}} = w_{\text{PDE}}\mathcal{L}_{\text{PDE}} + w_{\text{traj}}\mathcal{L}_{\text{traj}} + w_{\text{IC}}\mathcal{L}_{\text{IC}} + w_{\text{sym}}\mathcal{L}_{\text{sym}} + \mathbf{1}_{\{\lambda>0\}} w_0 \mathcal{L}_{\text{0-term}}$$

- $\mathcal{L}_{\text{PDE}}$: HJB 方程残差平方
- $\mathcal{L}_{\text{traj}}$: 轨迹展开终端库存惩罚（MT-PINN 核心）
- $\mathcal{L}_{\text{IC}}$: 库存为零时的内部条件
- $\mathcal{L}_{\text{sym}}$: 对称性约束
- $\mathcal{L}_{\text{0-term}}$: $\Gamma(0,0,S)=0$ 约束（仅 $\lambda>0$）

网络架构：MLP (32×32×32 for MT-PINN, 500×500×500 for baselines)，tanh 激活，JAX/FLAX 实现。优化器 AdamW，学习率 $5 \times 10^{-4}$。不使用归一化层或随机正则化器以保证导数稳定性。

训练流程分两阶段：
- **Phase A** ($\lambda=0$, 1D): 仅 $(\tau, X)$ 两个输入，30k epochs
- **Phase B** ($\lambda>0$, 2D): warm-start 到 $(\tau, X, S)$ 三个输入，5个 curriculum 阶段 × 5k epochs
- 轨迹展开参数：$P=820$ 初始状态，$J=7$ 个时间跨度 $\{T/50, T/10, T/5, 2T/5, 3T/5, 4T/5, T\}$，$N_{\text{dt}}=200$ Euler 步
- DWA 权重更新：每 1000 epochs，clip 范围 $[0.1, 2.0]$，EMA $\beta=0.95$

## 实验关键数据

### 主实验：合成基准（Gatheral-Schied 模型）
设置：$T=5.0$，$X \in [-10,10]$，$S \in [10,100]$，$\sigma=0.1$，$\kappa=0.1$，$N_{\text{PDE}}=30000$，$N_{\text{IC}}=5000$。

| 终端库存统计 ($\lambda=0.10$) | Mean $|X_T|$ ± Std | 95th pct | $p_\varepsilon$ ($\varepsilon=0.05$) |
|---|---|---|---|
| Vanilla PINN | 0.777 ± 0.444 | 1.407 | 0.055 |
| PINN + $\lambda$-curriculum | 0.164 ± 0.161 | 0.527 | 0.205 |
| **MT-PINN + $\lambda$-curriculum** | **0.073 ± 0.092** | **0.241** | **0.600** |

MT-PINN 的终端库存均值仅为 Vanilla PINN 的 1/10，满足 $|X_T| \le 0.05$ 的概率从 5.5% 提升到 60%。

### SPY 实盘回测
设置：7天 SPY 日内数据（2025.2.10-2.19），每天3个2小时窗口，5秒间隔，n=21 窗口。排除开盘/收盘各15分钟高波动期。日内波动率 $\sigma=0.0038$（≈6% 年化），永久冲击 $\kappa=0.2$，库存归一化到 $[-1,1]$，价格范围 $S \in [590, 620]$。

| 模型 | Mean Exposure | Exposure Std | Mean Cost (bps) | Cost Std (bps) |
|---|---|---|---|---|
| TWAP | 0.334 | 0.0000 | -6.35 | 12.56 |
| MT-PINN $\lambda=0.00$ | 0.336 | 0.0000 | -6.37 | 12.58 |
| MT-PINN $\lambda=0.05$ | 0.231 | 0.0004 | -5.01 | 11.02 |
| MT-PINN $\lambda=0.10$ | 0.164 | 0.0002 | -3.69 | 9.67 |

### 关键发现
- $\lambda=0$ 时 MT-PINN 几乎完美复现 TWAP（exposure 0.336 vs 0.334），验证了理论一致性
- $\lambda > 0$ 时形成清晰的 risk-cost frontier：exposure 下降（0.334 → 0.164），成本上升但标准差也降低
- 在下跌行情窗口中，$\lambda > 0$ 的 MT-PINN 前置执行（front-load），表现优于 TWAP
- MT-PINN 网络仅需 32×32×32 宽度即超越 500×500×500 的 vanilla PINN，说明轨迹损失才是关键
- 在 $\lambda \in \{0, 0.05, 0.10\}$ 三个设置下 MT-PINN 均一致优于 baselines（详见论文 Table 3）
- 计算资源极为轻量：1× TPU v6e-1，典型运行 1-4 分钟

## 亮点与洞察
- **轨迹视角的约束执行**：不依赖 PDE 残差间接满足约束，而是直接展开轨迹 + BPTT 惩罚终端库存，思路简洁有效
- **Curriculum regularization**：从低维/简单问题 warm-start 到高维/复杂问题，是 PINN 训练稳定性的实用技巧
- **极小网络超越大网络**：MT-PINN 用 32 宽度 MLP 胜过 500 宽度 baseline，说明归纳偏置（轨迹损失）比模型容量更重要
- **理论-实践一致**：$\lambda=0$ 时与 TWAP 一致、与解析解对齐，建立了可信度

## 局限与展望
- 仅考虑单资产、线性永久冲击模型，未涉及多资产、非线性冲击、订单簿动态
- 不包含交易费用和隔夜风险
- 轨迹损失的计算复杂度为 $\mathcal{O}(J \times N_{\text{dt}} \times P)$，规模受限
- SPY 数据仅7天21个窗口，统计显著性有限
- 未建模市场微观结构（订单簿深度、买卖价差、订单流不平衡）

## 相关工作与启发
- **vs Vanilla PINN**: MT-PINN 的核心优势在于轨迹损失直接约束终端库存，而非软惩罚
- **vs TWAP**: $\lambda=0$ 等价于 TWAP，$\lambda > 0$ 在下跌行情中明显优于 TWAP
- **vs Krishnapriyan et al.**: 借鉴了 curriculum learning 思路用于 PINN 训练稳定化
- **启发**: 对于带硬约束的 PDE 控制问题，「展开轨迹 + BPTT」是通用思路，可推广到其他金融/控制场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 轨迹损失+BPTT 执行终端约束是 PINN 领域的新颖组合
- 实验充分度: ⭐⭐⭐ 合成+实盘扎实但数据规模偏小，消融不够系统
- 写作质量: ⭐⭐⭐⭐ Workshop paper 结构清晰，数学推导完整
- 价值: ⭐⭐⭐⭐ 对 PINN 在金融控制问题中的约束执行提供了可复现的实用方案

<!-- RELATED:START -->

## 相关论文

- [Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](../../ICML2025/scientific_computing/differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)
- [Neuro-Spectral Architectures for Causal Physics-Informed Networks](neuro-spectral_architectures_for_causal_physics-informed_networks.md)
- [Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](../../ICLR2026/scientific_computing/empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)
- [Universal Neural Optimal Transport](../../ICML2025/scientific_computing/universal_neural_optimal_transport.md)

<!-- RELATED:END -->
