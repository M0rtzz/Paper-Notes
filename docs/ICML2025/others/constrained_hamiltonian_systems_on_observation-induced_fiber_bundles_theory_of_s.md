---
title: >-
  [论文解读] Constrained Hamiltonian Systems on Observation-Induced Fiber Bundles: Theory of Symmetry and Integrability
description: >-
  [ICML2025][Hamiltonian系统] 提出"观测诱导纤维丛"几何框架，将部分可观测系统中的观测不确定性从外部扰动内化为纤维坐标的内禀变化，在此结构上统一处理状态约束与观测约束，建立了完整的辛几何、可积性、对称性与守恒律理论。
tags:
  - ICML2025
  - Hamiltonian系统
  - 纤维丛
  - 观测约束
  - Dirac约束理论
  - 辛结构
  - 可积性
  - Lax对
  - Noether定理
  - 安全控制
---

# Constrained Hamiltonian Systems on Observation-Induced Fiber Bundles: Theory of Symmetry and Integrability

**会议**: ICML2025  
**arXiv**: [2505.22824](https://arxiv.org/abs/2505.22824)  
**代码**: 无  
**领域**: 物理/几何（约束动力学、纤维丛理论、辛几何）  
**关键词**: Hamiltonian系统, 纤维丛, 观测约束, Dirac约束理论, 辛结构, 可积性, Lax对, Noether定理, 安全控制

## 一句话总结
提出"观测诱导纤维丛"几何框架，将部分可观测系统中的观测不确定性从外部扰动内化为纤维坐标的内禀变化，在此结构上统一处理状态约束与观测约束，建立了完整的辛几何、可积性、对称性与守恒律理论。

## 研究背景与动机

经典约束 Hamilton 理论（Dirac 约束量子化）假设系统状态完全可观测，即约束条件 $\phi(x)=0$ 直接作用于精确已知的状态 $x \in M$。然而在现代控制系统、机器人学和量子测量中，**观测不完备性**（传感器噪声、测量精度限制）使得这一理想化假设不再成立。

现有处理方式的局限：

| 方法 | 核心思路 | 不足 |
|------|---------|------|
| 概率约束方法 | 随机优化处理不确定性 | 丧失动力系统几何结构，难以保辛 |
| 鲁棒控制 | 有界不确定性的最坏情形分析 | 过于保守，缺乏几何洞察 |
| 扩展 Kalman 滤波 | 将不确定性当作外部扰动估计和消除 | 未能利用系统内禀结构 |

**核心洞察**：当系统状态 $x \in M$ 只能通过观测映射 $h: M \to Y$（带不确定性 $\epsilon$）间接获取时，观测不确定性不是待消除的噪声，而是系统几何结构的**内禀组成部分**。

## 方法详解

### 1. 观测诱导纤维丛的构造

给定 Hamilton 系统 $(M, \omega, H)$ 和观测映射 $h: M \to Y$，构造观测诱导纤维丛 $\pi: E \to M$：

- **底流形**：$2n$ 维辛流形 $M$（相空间）
- **纤维**：每个状态 $x$ 对应的纤维 $\pi^{-1}(x) = \{\xi \in T^*_{h(x)}Y : \|\xi\|_{\rho_x} \leq \delta(x)\}$
- **不确定性函数** $\delta: M \to (0, \Delta]$：编码各位置观测精度差异
- **总空间**：$E = \{(x, \xi) : x \in \mathcal{W}, \|\xi\|_{\rho_x} \leq \delta(x)\}$

关键要素包括：
- 工作区域 $\mathcal{W} \subset M$ 为连通开集，$\bar{\mathcal{W}}$ 紧致
- 观测映射在 $\mathcal{W}$ 上满足局部微分同胚条件 $\text{rank}(dh) = k$
- 纤维度量族 $\rho = \{\rho_x\}$ 在 $\bar{\mathcal{W}}$ 上光滑变化

### 2. 观测自适应联络与曲率

定义保持纤维结构的线性联络 $\nabla$，满足：
- **度量相容性**：$\nabla \rho = 0$
- **观测相容性**：保持水平-垂直分解 $TE = H_\nabla E \oplus VE$
- **无挠率**：底流形方向上 $T^\nabla = 0$
- **曲率控制**：$\|R^\nabla\|_{L^\infty(\mathcal{W})} < \infty$

联络系数的具体形式：
- 底流形部分：$\Gamma^k_{ij} = \Gamma^{LC,k}_{ij}$（Levi-Civita 联络）
- 纤维部分：$\Gamma^a_{bc} = 0$（纤维度量不依赖纤维坐标）
- 混合部分：$\Gamma^a_{bi} = \frac{1}{2}\rho^{ac}\partial_i \rho_{bc}$

混合曲率简化为：

$$R^\nabla{}^a{}_{bij} = \partial_i \Gamma^a_{jb} - \partial_j \Gamma^a_{ib}$$

### 3. 纤维丛上的辛结构

总空间上的辛形式分解为三部分：

$$\omega_E = \pi^* \omega_{\mathcal{W}} + \omega_{\text{fib}} + \Omega_{\text{mix}}$$

- **底流形辛形式**：$\pi^*\omega_\mathcal{W} = \sum_\alpha dq^\alpha \wedge dp_\alpha$
- **纤维标准辛形式**：$\omega_{\text{fib}} = \sum_a d\xi_a \wedge d\pi_a$
- **曲率混合项**：$\Omega_{\text{mix}} = \sum_{i,a}(K_{ia}\,dx^i \wedge d\xi_a + L_{ia}\,dx^i \wedge d\pi_a)$

边界处理采用光滑截断函数 $\chi$ 实现辛结构的退化过渡，保证 Hamilton 方程在边界附近仍然良定义。

### 4. 可积性与 Lax 对

建立了观测约束下完全可积性的**几何充要条件**，并扩展了 Arnold-Liouville 定理到纤维丛设定。以 $n$ 粒子 Toda 晶格为例，构造观测 Lax 对：

$$L(\lambda, \epsilon) = L_0(\epsilon) + \lambda L_1(\epsilon)$$

其中 $L_0$ 为带观测误差的三对角矩阵（对角元为动量 $p_i$，次对角元为 $e^{y_i + \epsilon_i}$），$L_1$ 为不确定性对角矩阵。

**可积性保持条件**：当观测误差满足 $\|\epsilon\|_\infty \leq \epsilon_{\text{crit}}$ 时，修正积分 $I^{\text{obs}}_k = I^{\text{classical}}_k + \epsilon \delta I_k + O(\epsilon^2)$ 仍保持对合，不变环面仅发生 $O(\epsilon)$ 的正则形变。

### 5. 对称性与 Noether 定理

建立了观测对称群的几何结构刻画和主丛表示理论，证明了纤维丛上的 Noether 定理，能够处理传统理论无法涵盖的**依赖观测的守恒律**，并构造了纤维丛上的矩映射理论。

### 6. 保几何数值算法

提出保持纤维丛几何结构的数值积分方法：
- 利用观测自适应联络计算 Hamilton 向量场
- 约束违反时采用几何投影（而非代数修正）
- 时间步长自适应于观测不确定性 $\delta(x)$ 的局部变化
- 几何损失函数设计：$L_{\text{geo}} = \|\Phi(x,\xi)\|^2 + \alpha\|\nabla_i \rho_{jk}\|^2 + \beta\|\omega_E - \omega_{\text{ref}}\|^2$

## 实验关键数据

本文为纯理论工作，无传统数值实验。通过三个来自 [25] 的应用验证理论有效性：

| 应用场景 | 底流形 $M$ | 观测空间 $Y$ | 纤维丛特点 |
|---------|-----------|-------------|-----------|
| 软体机器人（MPM） | 连续体构型空间 | $\mathbb{R}^6$（6传感器） | 满足 (C4c) 衰减条件 |
| 7-DOF 机械臂 | $\mathbb{R}^{14}$（关节相空间） | $SE(3)$（末端位姿） | 避奇异构型，$SO(3)$ 对称 |
| 四旋翼导航 | $\mathbb{R}^3 \times SO(3) \times \mathbb{R}^6$ | $\mathbb{R}^4$（4深度传感器） | 满足 (C4b)，$\beta=2$（功率衰减） |

保几何算法收敛性（Theorem 26）：
- 约束保持误差：$|\Phi(x_n, \xi_n)| \leq C_1 h^2$
- 辛结构保持误差：$|\omega_{E,n} - \omega_E| \leq C_2 h^2$
- 全局收敛：$\|(x_n,\xi_n) - (x(t_n),\xi(t_n))\|_E \leq Che^{Lt_n}$
- 7-DOF 机械臂：步长 $h \sim 10^{-3}$s，约束误差 $\sim 10^{-6}$

## 亮点与洞察

1. **观测不确定性的几何内化**：将观测误差从外部扰动转化为纤维坐标的内禀变化，这一范式转换类似于广义相对论中等效原理的思想——观测不确定性可以在适当几何结构下被局部"规范掉"
2. **统一框架**：同一理论覆盖 Dirac 约束、辛约化、可积性、Noether 定理、Control Barrier Functions，实现经典力学与现代安全控制的桥接
3. **适量性条件 (Properness) 的完整刻画**：给出了纤维丛上约束适量性的完整充要条件（Theorem 7），包含全局不确定性有界、势函数二次下界、Lipschitz 连续性和渐近控制等四个条件
4. **工作区域设定的工程合理性**：避免奇异构型的局部化处理既保证理论严格性，又符合实际系统的物理限制
5. **Toda 晶格可积性保持**：明确给出了观测扰动下保持可积结构的临界条件 $\epsilon_{\text{crit}}$

## 局限与展望

1. **确定性不确定性假设**：当前框架仅处理确定性观测不确定性，无法直接处理 Wiener 过程等真正的随机噪声，需发展随机微分几何
2. **局部化限制**：理论严格性主要在避免奇异构型的工作区域内保证，向全局理论的推广需要解决复杂的拓扑和几何问题
3. **光滑性要求过强**：理论需要系统具有足够的光滑性（$C^3$），在某些实际应用中可能过于restrictive
4. **缺乏数值实验**：未给出保几何算法的实际计算结果与传统方法的定量对比
5. **应用验证间接性**：三个应用案例来自配套工作 [25]，本文未独立进行端到端验证

## 相关工作与启发

- **Dirac 约束理论** [5,6]：本文将其从约束子流形推广到纤维丛设定
- **Marsden-Weinstein 辛约化** [13]：本文扩展到观测约束下的版本
- **Arnold-Liouville 定理** [3]：推广到纤维丛上的完全可积性
- **Control Barrier Functions** [2]：本文为其提供了纤维丛几何基础
- **[25] "Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures"**：应用验证的配套 ICML 工作
- **Yang-Mills 理论** [24]：观测自适应联络与规范场联络具有类似几何地位

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 开创性地将观测不确定性几何化为纤维丛结构，提供全新的理论范式
- 实验充分度: ⭐⭐ — 纯理论工作，应用验证来自配套论文，缺乏独立数值实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、证明详尽，但篇幅极长（2000+行），部分冗余
- 价值: ⭐⭐⭐⭐ — 为约束动力学与安全控制搭建了统一几何桥梁，理论深度突出但实际影响待验证

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)
- [\[ICML 2025\] Symmetry-Robust 3D Orientation Estimation](symmetry-robust_3d_orientation_estimation.md)
- [\[ICML 2025\] Understanding Mode Connectivity via Parameter Space Symmetry](understanding_mode_connectivity_via_parameter_space_symmetry.md)
- [\[ACL 2025\] Length-Induced Embedding Collapse in PLM-based Models](../../ACL2025/others/length-induced_embedding_collapse_in_plm-based_models.md)
- [\[ICML 2025\] Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry](sparse_training_from_random_initialization_aligning_lottery_ticket_masks_using_w.md)

<!-- RELATED:END -->
