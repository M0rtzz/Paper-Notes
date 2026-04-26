---
title: >-
  [论文解读] Hamiltonian Neural PDE Solvers through Functional Approximation
description: >-
  [NeurIPS 2025][科学计算][Hamiltonian mechanics] 基于 Riesz 表示定理，用可学习核积分（Integral Kernel Functional）近似无限维 Hamiltonian 泛函，通过自动微分获取泛函导数，实现保能量的神经 PDE 求解器（HNS），在 1D/2D PDE 上展现出优越的稳定性和泛化能力。
tags:
  - NeurIPS 2025
  - 科学计算
  - Hamiltonian mechanics
  - 偏微分方程
  - 泛函近似
  - 神经场
  - 能量守恒
---

# Hamiltonian Neural PDE Solvers through Functional Approximation

**会议**: NeurIPS 2025  
**arXiv**: [2505.13275](https://arxiv.org/abs/2505.13275)  
**代码**: [GitHub](https://github.com/anthonyzhou-1/hamiltonian_pdes)  
**领域**: Scientific Computing / Neural PDE Solvers  
**关键词**: Hamiltonian mechanics, PDE求解, 泛函近似, 神经场, 能量守恒

## 一句话总结

基于 Riesz 表示定理，用可学习核积分（Integral Kernel Functional）近似无限维 Hamiltonian 泛函，通过自动微分获取泛函导数，实现保能量的神经 PDE 求解器（HNS），在 1D/2D PDE 上展现出优越的稳定性和泛化能力。

## 研究背景与动机

**领域现状**：神经 PDE 求解器（FNO、Unet 等）在参数化 PDE 求解上取得了显著进展，但绝大多数方法工作在 Newtonian 框架下——直接预测下一时刻的状态或时间导数，没有利用物理系统的守恒结构。

**现有痛点**：Hamiltonian Neural Networks（HNN）已在离散粒子系统中证明了保守恒律的能力，但 HNN 局限于有限维系统（如 N 体问题），其 Hamiltonian 是一个函数 $\mathcal{H}(\mathbf{q}, \mathbf{p}): \mathbb{R}^{2n} \to \mathbb{R}$。然而，大多数实际 PDE 描述的是连续场（流体、波动、弹性体），需要**无限维** Hamiltonian 力学，其中 Hamiltonian 是泛函 $\mathcal{H}[u]: \mathcal{F}(\Omega) \to \mathbb{R}$，演化由泛函导数 $\delta\mathcal{H}/\delta u$ 驱动。

**核心矛盾**：将 Hamiltonian 框架推广到 PDE 有两大难题：(a) 需要逼近从函数空间到标量的映射（泛函），传统神经网络不是为此设计的；(b) 逼近后的泛函必须具有准确的泛函导数，以便用于 Hamilton 方程的时间演化。

**本文目标** 设计一种能够学习 Hamiltonian 泛函并准确计算泛函导数的神经网络架构，以此构建遵循 Hamiltonian 框架的神经 PDE 求解器。

**切入角度**：利用泛函分析中的 **Riesz 表示定理**——任何连续线性泛函都可以表示为内积 $\mathcal{H}[u] = \langle u, \kappa_\theta \rangle$，从而将泛函逼近问题转化为函数逼近问题，而神经网络天然擅长后者。

**核心 idea**：用神经场参数化的核积分来逼近 Hamiltonian 泛函，结合自动微分获取泛函导数，构建保守恒律的 PDE 求解器。

## 方法详解

### 整体框架

HNS（Hamiltonian Neural Solver）的工作流程：

1. **前向传播**：给定当前场状态 $\mathbf{u}^t$ 和坐标 $\mathbf{x}$，通过核积分泛函（IKF）计算 Hamiltonian 标量 $\mathcal{H}_\theta$
2. **反向传播**：利用自动微分计算泛函导数 $\delta\mathcal{H}_\theta / \delta u$
3. **时间演化**：通过已知的线性算子 $\mathcal{J}$（如 $\partial_x$）得到时间导数 $\partial u / \partial t = \mathcal{J}(\delta\mathcal{H}_\theta / \delta u)$
4. **数值积分**：用 2 阶 Adams-Bashforth 方法更新状态 $\mathbf{u}^{t+1}$

### 关键设计

#### 1. 积分核泛函（Integral Kernel Functional, IKF）

- **功能**：将泛函 $\mathcal{H}[u]$ 表示为核积分形式
- **核心思路**：基于 Riesz 表示定理，$\mathcal{H}[u] = \int_\Omega \kappa_\theta(x, u(x)) \cdot u(x) \, dx$，用 Riemann 求和离散化为 $\mathcal{H}_\theta \approx \sum_i \kappa_\theta(x_i, u_i) \cdot u_i \cdot \mu_i \Delta x$
- **设计动机**：与神经算子中的积分核算子类似但本质不同——算子输出函数需对每个查询点求和，开销大到需要 Fourier 技巧截断；而泛函输出标量，Riemann 求和只算一次，可以完整保留精度

#### 2. SIREN + FiLM 条件化核参数化

- **功能**：用神经场架构参数化核函数 $\kappa_\theta$
- **核心思路**：采用正弦表示网络（SIREN）作为核架构，每层用 FiLM（Feature-wise Linear Modulation）进行条件化：$\kappa_\theta^{(l)}(x_i, u_i) = \gamma_\theta^{(l)}(u_i) \sin(Wx_i + b) + \beta_\theta^{(l)}(u_i)$
- **设计动机**：SIREN 不仅拟合函数效果好，其梯度表示也很准确（正弦函数求导仍为正弦），有利于精确计算泛函导数。FiLM 可实现局部/全局条件化，让核函数依赖于输入场

#### 3. 局部 vs 全局条件化

- **局部条件化**：FiLM 参数仅依赖于 $u_i = u(x_i)$，适用于 Hamiltonian 仅包含局部项的简单 PDE（如 Advection）
- **全局条件化**：FiLM 参数依赖于全场 $\mathbf{u}^t$（通过浅层 1D CNN），适用于包含非局部项（如 $u_{xx}$）的 PDE（如 KdV、SWE）

#### 4. 隐式泛函导数学习

- **关键发现**：IKF 只需在泛函值 $\mathcal{H}[u]$ 上训练（标量监督），就能通过自动微分隐式学到准确的泛函导数 $\delta\mathcal{H}/\delta u$
- **直觉**：对线性 IKF，$\nabla_\mathbf{u} \sum_i \kappa_\theta(x_i) u_i \mu_i \Delta x = [\kappa_\theta(x_1), \ldots, \kappa_\theta(x_n)]$，即泛函导数恰好是学到的核函数的离散化

### 损失函数 / 训练策略

- 训练损失：$\mathcal{L} = \|\delta\mathcal{H}_\theta / \delta u - \delta\mathcal{H} / \delta u\|^2$，直接在泛函导数域优化
- 标签来自解析 Hamiltonian 的泛函导数公式 + 有限差分计算
- 选择在泛函导数域而非时间域训练，因为更直接优化核函数 $\kappa_\theta$

## 实验关键数据

### 主实验

#### Toy：线性/非线性泛函逼近（Table 1）

| 指标 | MLP | FNO | IKF |
|------|-----|-----|-----|
| 线性泛函 $\mathcal{F}_l[u]$ (Base) | 2.47e-5 | 2.76e-4 | **3.00e-16** |
| 线性泛函导数 (Base) | 0.083 | 0.066 | **1.15e-3** |
| 线性泛函 (OOD) | 0.046 | 0.268 | **2.13e-7** |
| 非线性泛函 $\mathcal{F}_{nl}[u]$ (Base) | 0.029 | 0.016 | **2.05e-3** |
| 非线性泛函导数 (Base) | 1.33 | 2.10 | **0.045** |

关键结论：IKF 在泛函值和泛函导数上均大幅优于 MLP 和 FNO，且参数最少（4.3K vs MLP 7.5K vs FNO 10.9K）。

#### 1D PDE：Advection + KdV（Table 2）

| 模型 | Adv Rollout Err ↓ | KdV Corr Time ↑ |
|------|-------------------|-----------------|
| FNO | 0.83 ± 0.17 | 68.0 ± 10.5 |
| Unet | 0.40 ± 0.29 | 134.0 ± 10.6 |
| FNO(du/dt) | 0.048 ± 0.007 | 77.6 ± 3.6 |
| Unet(du/dt) | 0.057 ± 0.024 | 113.3 ± 15.0 |
| **HNS** | **0.0039 ± 0.0008** | **151.1 ± 3.0** |

关键结论：HNS 用约一半参数（Adv: 32K vs 65K; KdV: 87K vs 135K）超越所有基线，Advection Rollout Error 比最好基线低一个数量级。

#### 2D 浅水方程（Table 3）

| 模型 | Sines (in-dist) ↓ | Pulse (OOD) ↓ | 参数量 |
|------|-------------------|---------------|-------|
| Transolver | 0.084 | 0.122 | 4M |
| FNO | 0.057 | 0.117 | 7M |
| PINO | 0.053 | 0.114 | 7M |
| Unet | **0.010** | 0.042 | 3M |
| **HNS** | 0.026 | **0.021** | 3M |

关键结论：HNS 在 OOD 泛化（Pulse 初始条件）上大幅领先（0.021 vs Unet 0.042），同时保持能量守恒。

### 消融实验

| 消融内容 | 主要发现 |
|---------|---------|
| 核类型（线性 vs 非线性） | 非线性核对复杂 Hamiltonian 必要 |
| 条件化（局部 vs 全局） | KdV 等含 $u_{xx}$ 项的 PDE 需要全局条件化 |
| SIREN vs MLP 核 | SIREN 梯度表示更准确，性能更优 |
| 积分求积方式 | 梯形法则已足够精确 |

### 关键发现

1. **能量守恒**：HNS 预测的轨迹 Hamiltonian 随时间几乎恒定，而 FNO/Unet 的 Hamiltonian 显著漂移
2. **时间外推**：训练于短时间窗口（Adv: 0-4s, KdV: 0-25s），HNS 可稳定外推到长时间（Adv: 0-20s, KdV: 0-100s）
3. **OOD 泛化**：在 2D SWE 上，HNS 甚至在 OOD 初始条件下也能保持 Hamiltonian 守恒，展示强归纳偏置
4. **参数效率**：核权重在所有输入点 $(x_i, u_i)$ 间共享，HNS 参数量约为基线的一半

## 亮点与洞察

1. **优雅的理论根基**：从 Riesz 表示定理出发，将泛函逼近化归为函数逼近，是一个非常自然且深刻的 insight
2. **隐式导数学习**：仅需标量监督训练即可获得准确的泛函导数，这是其他架构（MLP/FNO）做不到的——这使得 Hamiltonian 框架在 PDE 领域成为可能
3. **统一了多个视角**：IKF 同时联系了泛函分析、神经场和神经算子文献
4. **实用的参数效率**：权重共享机制使得 HNS 在参数更少的情况下性能更好
5. **有趣的负面结果**：在 Hamiltonian 值上加辅助 loss 反而有害（模型可能学恒等映射），这揭示了 Hamiltonian 守恒约束的微妙性

## 局限与展望

1. **非线性泛函理论不完整**：Riesz 定理只保证线性泛函的逼近，非线性情况缺乏严格理论支撑
2. **推理开销**：需要反向传播计算泛函导数 + 评价线性算子 $\mathcal{J}$，推理比直接预测方法慢
3. **可扩展性**：依赖数值积分（Riemann 求和），对高分辨率网格的扩展性存疑
4. **仅适用于 Hamiltonian 系统**：不能直接处理耗散系统，虽然存在扩展（如 GENERIC 框架），但需额外工作
5. **$\mathcal{J}$ 的数值实现**：用有限差分近似 $\mathcal{J}$ 可能引入数值误差，特别是在高阶导数项上
6. **未来方向**：可结合辛积分器；扩展到 3D 或更复杂几何；探索学习 $\mathcal{J}$

## 相关工作与启发

- **HNN 系列**：Greydanus et al. (2019) 的离散 HNN → 本文自然推广到无限维
- **Neural Operator**：FNO 中的积分核算子与 IKF 有深刻联系（IKF 可看作算子在单点求值的泛函版本）
- **神经场**（SIREN、NeRF 等）：为核函数参数化提供了成熟工具
- **ML for DFT**：密度泛函理论中学习能量泛函的思路与 IKF 异曲同工
- **启发**：Hamiltonian 框架不仅提供了守恒保证，更提供了一种"学习能量而非力"的范式——这可能对分子动力学、气象模拟等领域有重要启示

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将 Hamiltonian 框架推广到无限维神经 PDE 求解，理论视角新颖
- 实验充分度: ⭐⭐⭐⭐ — 从 toy 到 2D SWE 逐步验证，但缺少 3D 实验和更大规模验证
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导清晰，从离散到连续的叙事线非常流畅
- 价值: ⭐⭐⭐⭐ — 为 physics-informed ML 提供了新范式，但实际应用场景受限于 Hamiltonian 系统

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)
- [\[NeurIPS 2025\] INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers](inc_an_indirect_neural_corrector_for_auto-regressive_hybrid_pde_solvers.md)
- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](towards_universal_neural_operators_through_multiphysics_pretraining.md)
- [\[NeurIPS 2025\] DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)
- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](../../ICLR2026/scientific_computing/one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)

<!-- RELATED:END -->
