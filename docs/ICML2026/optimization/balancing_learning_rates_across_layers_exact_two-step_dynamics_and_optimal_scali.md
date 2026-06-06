---
title: >-
  [论文解读] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks
description: >-
  [ICML 2026][优化/理论][逐层学习率] 本文在两层和三层线性神经网络中推导出梯度下降一步和两步后测试损失的精确闭式表达式，揭示了一个相变现象：第一步更新时非对称学习率最优，而第二步后对称（平衡）学习率变为局部最优，为逐层学习率调度提供了理论基础。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "逐层学习率"
  - "线性网络"
  - "梯度分解"
  - "训练动力学"
  - "学习率平衡"
---

# Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks

**会议**: ICML 2026  
**arXiv**: [2606.00340](https://arxiv.org/abs/2606.00340)  
**代码**: https://github.com/TDCSZ327/Layer-Balancing  
**领域**: 优化理论  
**关键词**: 逐层学习率, 线性网络, 梯度分解, 训练动力学, 学习率平衡  

## 一句话总结

本文在两层和三层线性神经网络中推导出梯度下降一步和两步后测试损失的精确闭式表达式，揭示了一个相变现象：第一步更新时非对称学习率最优，而第二步后对称（平衡）学习率变为局部最优，为逐层学习率调度提供了理论基础。

## 研究背景与动机

**领域现状**：深度网络训练中，逐层学习率调度（如 LARS、LAMB、TempBalance、Adam-mini）已被广泛采用以加速收敛和提升泛化。这些方法通过为不同层分配不同学习率来适应各层梯度特性的差异。

**现有痛点**：现有逐层学习率策略主要基于启发式设计或渐近分析，缺乏将学习率选择与测试损失直接关联的精确公式。连续时间梯度流分析和 NTK 近似都无法刻画离散有限步设置中信号-残差耦合效应。层间交互使得学习率分配对泛化的影响难以量化。

**核心矛盾**：在多层网络中，各层梯度的信号分量和自交互分量的范数不同，学习率在决定训练动力学中起关键作用。现有理论框架要么假设无穷小步长（mean-field / µP），要么独立分析各层（忽略跨层耦合），无法精确刻画有限步训练中逐层学习率对泛化的影响。

**本文目标**：在线性网络中推导一步和两步梯度下降后测试损失关于逐层学习率的精确闭式表达式，从而精确刻画何时应使用非对称 vs. 对称学习率。

**切入角度**：利用正交初始化的代数结构，将梯度分解为信号对齐项 $A_\ell^t$（主导学习信号）和自交互项 $B_\ell^t$（权重间耦合），证明在临界学习率阈值以下自交互项可忽略，从而获得可解析的代理损失。

**核心 idea**：逐层学习率的最优分配是动态的——早期训练受益于非对称分配以利用层级特异性信号传播，后期训练则倾向平衡分配以促进跨层协调。

## 方法详解

### 整体框架

考虑正交初始化的线性网络：两层模型 $f(x) = \frac{1}{h}x^\top W_1 W_2$（输入输出均为 $\mathbb{R}^h$）和三层模型 $f^*(x) = \frac{1}{\sqrt{h}}x^\top W_1 W_2 a$（标量输出，$a$ 固定）。训练数据由线性教师模型生成，优化目标为 MSE 损失。分析流程为：(1) 梯度分解为信号项与残差项；(2) 证明残差项在宽网络中可忽略；(3) 用信号-only 轨迹推导测试损失闭式解；(4) 分析最优学习率分配的对称性。

### 关键设计

1. **梯度分解与信号主导性证明**:

    - 功能：将精确梯度分解为信号对齐项和自交互项，并证明前者主导
    - 核心思路：对两层网络第 $\ell$ 层的梯度 $G_\ell^t = B_\ell^t - A_\ell^t$，其中 $A_1^t = \frac{1}{h}M W_2^{t\top}$ 捕获标签信号，$B_1^t = \frac{1}{h^2}W_1^t W_2^t W_2^{t\top}$ 为权重自交互。Proposition 5.1 证明当 $\eta_1, \eta_2 \leq O(h\sqrt{h})$ 时，$\|G_\ell^t - A_\ell^t\| \leq \|G_\ell^t\| / (\sqrt{h}-1)$，即残差项被 $1/\sqrt{h}$ 因子压制
    - 设计动机：该分解将不可解析的精确梯度替换为结构简洁的信号项，使闭式测试损失推导成为可能，同时识别出临界学习率尺度 $\eta = \Theta(h\sqrt{h})$（两层）和 $\eta = \Theta(h)$（三层）

2. **精确两步测试损失公式**:

    - 功能：给出测试损失关于 $\eta_1, \eta_2$ 的精确多项式表达式
    - 核心思路：两层网络一步测试损失为 $L^{(1)} = \frac{\eta_1^2}{h^4} + \frac{\eta_2^2}{h^4} + \frac{2\eta_1\eta_2}{h^4} + \frac{\eta_1^2\eta_2^2}{h^7} - \frac{2\eta_1}{h^2} - \frac{2\eta_2}{h^2} + \frac{1}{h} + \frac{2\eta_1\eta_2}{h^5} + 1$，包含线性改善项、二次交互项和残差方差项。两步损失则含 $(1+\eta_1\eta_2/h^3)$ 的高次幂，反映层间乘性表示学习
    - 设计动机：精确公式使得可以严格分析学习率对称点 $\eta_1 = \eta_2$ 是否为局部最小值，从而揭示相变现象

3. **非对称到平衡的相变定理**:

    - 功能：证明最优学习率分配随训练步数发生质变
    - 核心思路：在约束 $\eta_1 + \eta_2 = 2h^\alpha$ 下，Corollary 5.4 证明：(a) 对任意 $0 < \alpha \leq 3/2$，对称点 $\eta_1 = \eta_2$ 不是一步损失的局部最小值；(b) 对 $1 < \alpha \leq 3/2$ 且 $h$ 足够大，$\eta_1 = \eta_2$ 是两步损失的局部最小值。三层网络有类似结论但临界尺度降为 $O(h)$
    - 设计动机：首次从理论上解释了为什么逐层学习率调度在训练早期应非对称（利用表示层与读出层的角色差异），后期应趋向平衡（促进跨层协调对齐）

## 实验关键数据

### 主实验：理论预测 vs 实际测试损失

| 设置 | 网络 | 步数 | 理论-实验偏差 | 对称性结论 |
|------|------|------|--------------|-----------|
| $h=1000$, 正交初始化 | 2层 | 1步 | 紧密吻合 | 对称 LR 非最优 |
| $h=1000$, 正交初始化 | 2层 | 2步 | 紧密吻合 | 对称 LR 局部最优 |
| $h=1000$, 正交初始化 | 3层 | 1步 | 紧密吻合 | 对称 LR 非最优 |
| $h=1000$, 正交初始化 | 3层 | 2步 | 紧密吻合 | 对称 LR 局部最优 |
| $h=1000$, 高斯初始化 | 2层/3层 | 多步 | 一致趋势 | 同上，相变仍存在 |

### 扩展实验：泛化性验证

| 扩展条件 | 关键发现 |
|----------|---------|
| 标签噪声 $\xi \sim \mathcal{N}(0,\rho)$ | 非对称→平衡相变仍然成立 |
| 4层/8层深线性网络 | 一步非对称、两步平衡的转变保持 |
| 3层非线性网络（ReLU） | 曲线对称性稍弱但相变趋势一致 |
| 多步训练（至 512 步） | 平衡学习率在后续步骤中持续局部最优 |
| Frobenius 范数驱动的 LR 调度器 | 比均匀基线取得更低的训练/测试损失 |

### 关键发现
- 临界学习率阈值：两层网络为 $\eta = O(h\sqrt{h})$，三层为 $\eta = O(h)$，超过此阈值梯度近似不再成立
- 三层网络两步损失对 $\eta_1\eta_2$ 的高阶项依赖更强（出现 $\eta_1^4\eta_2^4/h^6$ 项），反映更深层的跨层耦合
- 基于 $\|W_1\|_F, \|W_2\|_F$ 设计的自适应逐层 LR 调度器验证了理论预测：范数差 $\|W_1\|_F - \|W_2\|_F \to 0$ 对应学习率趋向平衡，同时收敛到更平坦的极小值

## 亮点与洞察
- **信号-残差梯度分解**是本文最巧妙的工具：通过证明自交互项 $B_\ell^t$ 被 $1/\sqrt{h}$ 压制，将不可解析的精确动力学简化为信号-only 轨迹，使闭式测试损失推导成为可能。这一分解思路可迁移到其他需要分析有限步梯度下降的理论工作中
- **"先非对称后平衡"的相变视角**为实践中逐层学习率调度器（LARS、LAMB、TempBalance）提供了统一的理论解释：早期各层角色不同（表示 vs. 读出），应差异化对待；后期跨层协调占主导，应趋向平衡
- 范数驱动的 LR 调度器设计 $\eta_{W_i}^{(t)} = \frac{2\|W_j^t\|_F}{\|W_1^t\|_F + \|W_2^t\|_F} \cdot lr$ 是理论到实践的直接映射，简洁且有效

## 局限与展望
- 理论分析仅限于线性网络和正交/高斯初始化，虽然非线性 ReLU 实验显示类似趋势但缺乏理论保证
- 仅分析了一步和两步动力学，对长时间训练的最优学习率分配只有实验观察
- 假设 $n = h = d$，未考虑过参数化或欠参数化的不同体制
- 未涉及随机梯度下降（SGD）和 mini-batch 设置，与实际训练仍有差距
- 未来可将分析扩展到 Transformer 等实际架构，特别是探索注意力层和 FFN 层之间的最优学习率分配

## 相关工作与启发
- **LARS/LAMB** (You et al., 2017; 2018)：基于"信任比率"的逐层 LR，本文为其早期非对称分配提供了理论支持
- **TempBalance** (Zhou et al., 2023; Liu et al., 2024)：基于权重谱重尾性的逐层 LR，本文的范数平衡视角与之互补
- **Adam-mini / Blockwise-LR** (Zhang et al., 2024; Wang et al., 2025)：基于 Hessian 块结构的逐层 LR
- **Du et al. (2018)**：证明深度齐次模型的层范数自动平衡，本文进一步揭示了学习率在驱动此平衡中的角色
- **Kunin et al. (2024)**：研究不平衡初始化下的特征学习，发现层间平衡的学习率促进快速特征学习

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)
- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](../../AAAI2026/optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)
- [\[ICML 2026\] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](muon_in_associative_memory_learning_training_dynamics_and_scaling_laws.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[ICLR 2026\] Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers](../../ICLR2026/optimization/pinet_optimizing_hard-constrained_neural_networks_with_orthogonal_projection_lay.md)

</div>

<!-- RELATED:END -->
