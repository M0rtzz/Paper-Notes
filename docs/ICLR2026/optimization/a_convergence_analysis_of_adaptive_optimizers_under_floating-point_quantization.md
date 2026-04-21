---
title: >-
  [论文解读] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization
description: >-
  [ICLR 2026][优化][低精度训练] 本文建立了首个在浮点量化下分析自适应优化器收敛性的理论框架，对梯度、权重和优化器状态（动量、二阶矩）同时施加相对误差量化模型，证明了量化 Adam 和 Muon 在尾数长度仅需对数增长于迭代次数时即可保持与全精度相同的 $\tilde{O}(T^{-1/4})$ 收敛率，并揭示了 Adam 对权重和二阶矩量化高度敏感而 Muon 更为鲁棒的理论机制。
tags:
  - ICLR 2026
  - 优化
  - 低精度训练
  - Adam
  - Muon
  - 浮点量化
  - 收敛分析
---

# A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization

**会议**: ICLR 2026  
**arXiv**: [2510.21314](https://arxiv.org/abs/2510.21314)  
**代码**: 无  
**领域**: Optimization  
**关键词**: 低精度训练, Adam, Muon, 浮点量化, 收敛分析

## 一句话总结
本文建立了首个在浮点量化下分析自适应优化器收敛性的理论框架，对梯度、权重和优化器状态（动量、二阶矩）同时施加相对误差量化模型，证明了量化 Adam 和 Muon 在尾数长度仅需对数增长于迭代次数时即可保持与全精度相同的 $\tilde{O}(T^{-1/4})$ 收敛率，并揭示了 Adam 对权重和二阶矩量化高度敏感而 Muon 更为鲁棒的理论机制。

## 研究背景与动机
大语言模型（LLM）的快速规模扩展使得低精度训练成为降低内存、提高效率的关键技术。BF16、FP8等低精度格式已在实际的万亿 token 级训练中被广泛使用（如 DeepSeek-V3、FP8-LM 等），并且在经验上未观察到显著的精度损失。

然而，**理论理解严重滞后于实践**。现有的量化优化器收敛理论存在多个关键缺口：

**只分析梯度量化**: 大多数理论工作仅考虑随机梯度下降（SGD）中梯度的量化，而现代低精度训练同时量化权重、梯度和优化器状态

**不切实际的假设**: 现有分析要么假设无偏量化（unbiased quantization），要么依赖误差反馈（error feedback）机制——前者不符合浮点量化的特性，后者在大规模LLM训练中因内存开销而不实际

**忽略优化器状态量化**: Adam 的一阶矩和二阶矩在实践中也被量化以节省内存（如 8-bit Adam），但理论分析中这一环节被完全忽略

**未涵盖新型优化器**: Muon 等基于矩阵视角的新兴优化器在低精度下的理论保证为空白

**核心问题**: 为什么在所有组件都被激进量化的情况下，自适应优化器仍然能有效收敛？

## 方法详解

### 整体框架
本文提出了一个**解析式低精度训练框架（Analytical Low-Precision Training Framework）**，明确建模以下量化操作：
- Master 维护全精度权重 $\mathbf{W}_t$，但向 worker 传输量化版本 $\mathbf{W}_t^Q$
- Worker 使用 $\mathbf{W}_t^Q$ 进行前向和反向传播，计算梯度后量化并回传
- Master 反量化梯度，更新量化的优化器状态（动量、二阶矩），应用优化器更新后重新量化存储

框架的关键特征：使用**相对误差模型（Relative Error Model）**代替传统的无偏量化假设。

### 关键设计
1. **浮点量化的相对误差建模（Assumption 3.1）**: 对于任意标量 $x$，量化后的值 $x^Q$ 满足 $|x^Q - x| \leq q|x|$，其中 $q = \Theta(2^{-M})$，$M$ 为目标浮点格式的尾数长度。设计动机：浮点量化（如 FP32 → BF16）截断尾数位但保持符号位和指数位不变，因此量化误差与数值本身的量级成比例——这正是相对误差的特征。该假设在实践中通过 per-tensor/per-channel scaling 技术得到良好满足。

2. **组件级分离量化误差**: 框架对四种量化误差分别建模并追踪其对收敛的影响：

    - $q_W$（权重量化误差）
    - $q_G$（梯度量化误差）
    - $q_M$（一阶矩/动量量化误差）
    - $q_V$（二阶矩量化误差）
   
   这种分离使得理论可以精确刻画**不同组件的量化对收敛的差异化影响**。

3. **量化 Adam 的收敛定理（Theorem 4.5）**: 在标准假设（无偏随机梯度、有界梯度、$L$-光滑）下，当设置 $\eta = \Theta(1/\sqrt{T})$, $1 - \beta_2 = \Theta(1/T)$, 且量化误差满足 $q_G, q_M = O(1/T)$, $q_W, q_V = O(1/T^2)$ 时，量化 Adam 达到 $\tilde{O}(T^{-1/4})$ 收敛率——匹配全精度 Adam 的已知最优率。

   **关键发现**: Adam 对二阶矩 ($q_V$) 和权重 ($q_W$) 的精度要求更严格（需要 $O(1/T^2)$），而对梯度和一阶矩的要求较宽松（$O(1/T)$）。这是因为 $\beta_2 \to 1$ 时二阶矩的累积误差通过逆平方根非线性放大。

4. **量化 Muon 的收敛定理（Theorem 4.6）**: 对于 Muon，所有组件只需 $q_G = q_W = q_M = O(T^{-1/2})$ 即可保持 $O(T^{-1/4})$ 收敛率。这一条件**显著弱于 Adam**（$O(T^{-1/2})$ vs $O(T^{-1})$ 和 $O(T^{-2})$）。理论解释：Muon 使用基于 SVD 的 sign 操作符，避免了由二阶矩逆平方根引起的量化误差放大。

### 损失函数 / 训练策略
理论分析在以下标准假设下进行：
- **Assumption 4.1**: 无偏随机梯度
- **Assumption 4.2**: 梯度有界（Adam: $\ell_\infty$ 有界；Muon: 方差有界）
- **Assumption 4.3**: 目标函数 $L$-光滑
- **Assumption 4.4**: 有界初始化

量化使用模拟方式实现：保持指数和符号位不变，截断尾数位至 $M$ 位，并应用随机舍入。

## 实验关键数据

### 主实验（合成实验 - Rosenbrock 函数）

| 优化器 | 尾数长度 M | 收敛行为 | 梯度范数 |
|--------|-----------|---------|---------|
| Adam | M=23 (FP32) | 基线，最佳收敛 | 最小 |
| Adam | M=10 | 接近全精度 | 略大 |
| Adam | M=7 (BF16) | 接近全精度 | 略大 |
| Adam | M=3 | 收敛变慢 | 明显增大 |
| Adam | M=1 | 严重退化 | 发散 |
| Muon | M=7 (BF16) | 接近全精度 | 略大 |
| Muon | M=3 | 仍可收敛 | 轻微退化 |
| Muon | M=2 | 开始退化 | 明显增大 |

### 真实数据实验（CIFAR-10，4层全连接网络）

| 优化器 | 尾数长度 M | 梯度范数收敛 | 与全精度对比 |
|--------|-----------|-----------|------------|
| Adam | M≥7 | 接近全精度 | 差距极小 |
| Adam | M=3 | 退化 | 可见差距 |
| Adam | M=1-2 | 严重退化 | 无法匹配 |
| Muon | M≥3 | 接近全精度 | 差距极小 |
| Muon | M=2 | 轻微退化 | 小幅差距 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅量化梯度 | 影响最小 | 梯度对量化最鲁棒 |
| 仅量化权重 | Adam 敏感，Muon 较鲁棒 | 验证了 $q_W$ 的差异化影响 |
| 仅量化二阶矩 | Adam 最敏感 | $\beta_2 \to 1$ 导致误差放大 |
| 仅量化一阶矩 | 中等影响 | 衰减机制提供了一定保护 |
| Adam vs Muon 鲁棒性 | Muon 更鲁棒 | 验证了 $O(T^{-1/2})$ vs $O(T^{-2})$ 的理论预测 |

### 关键发现
- **尾数长度仅需对数增长**: $M = \Omega(\log T)$ 即可保证全精度收敛率，这与现有硬件精度（BF16 的 $M=7$, FP8 的 $M=3$）完全一致
- **Adam 的二阶矩和权重是瓶颈**: $q_V$ 和 $q_W$ 需要 $O(1/T^2)$ 精度，而 $q_G, q_M$ 仅需 $O(1/T)$——验证了 FP8-LM 中二阶矩需要略高精度的经验观察
- **Muon 需要的误差控制更弱**: 所有组件只需 $O(T^{-1/2})$，理论解释了 Liu et al. (2025) 观察到的 Muon 在低精度下表现更优的经验现象
- **相对误差模型比无偏假设更合理**: 浮点量化天然满足相对误差性质，不需要额外的误差反馈机制

## 亮点与洞察
- **填补了重要的理论空白**: 首次在实际的浮点量化模型下对自适应优化器（包括 Adam 和新兴的 Muon）给出了收敛保证
- **可解释的组件级灵敏度分析**: 精确量化了不同组件对收敛的差异化影响，为混合精度训练策略的设计提供了理论指导（如：二阶矩和权重需要更高精度）
- **Adam vs Muon 的定量对比**: 理论上清晰解释了 Muon 为何在低精度下更鲁棒（$O(T^{-1/2})$ vs $O(T^{-2})$），为优化器选择提供了依据
- **实际意义显著**: 结果直接证明了 BF16 和 FP8 训练的理论合理性，为工业界的低精度训练实践提供了理论背书
- **不依赖误差反馈机制**: 与之前需要per-parameter error feedback 的理论不同，本文的框架更贴合实际的大规模训练流程

## 局限与展望
- **标准光滑性假设**: 分析假设 $L$-光滑，而实际深度学习目标可能仅满足更弱的 $(L_0, L_1)$-光滑条件，作者将其列为未来方向
- **精确算术假设**: 分析假设量化状态的运算在精确算术下完成，未考虑 FP8 矩阵乘法等低精度运算的额外误差
- **未考虑通信效率**: 低精度训练的另一重要动机是分布式训练中的通信压缩，本文未涉及
- **实验规模较小**: 仅在 Rosenbrock 函数和 CIFAR-10 上的小规模网络验证，未在大规模 Transformer/LLM 训练中实测
- **$q_W = O(1/T^2)$ 条件可能过严**: 作者指出此条件来自证明中对权重范数无界增长的 worst-case 处理，在权重范数有界的实际场景中可放松至 $O(1/T)$

## 相关工作与启发
- **自适应优化理论**: Défossez et al. (2022) 的全精度 Adam 收敛性分析提供了本文的理论基础骨架
- **量化SGD/SGDM**: Alistarh et al. (2017) 的 QSGD 工作和后续的误差反馈方法（Karimireddy et al., 2019）仅处理了 SGD 场景
- **量化Adam的先前工作**: Chen et al. (2021) 需要误差反馈；Modoranu et al. (2024) 的 MicroAdam 忽略了优化器状态量化
- **低精度训练实践**: DeepSeek-V3 (FP8), FP8-LM, COAT 等工作展示了低精度训练的实际可行性
- **Muon 优化器**: Jordan et al. (2024) 提出的基于矩阵 SVD 的优化器，Shen et al. (2025) 给出了全精度收敛保证
- **启发**: 浮点量化的"相对误差"特性是一种自然优势（相对于整数量化的"绝对误差"），在设计量化方案时应充分利用这一结构。Adam 的 $\beta_2 \to 1$ 设置虽然对收敛必要，但会放大量化误差——这暗示对 Adam 的量化策略需要特别注意二阶矩的精度。

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication](../../AAAI2026/optimization/a_unified_convergence_analysis_for_semi-decentralized_learni.md)
- [Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](../../CVPR2026/optimization/fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)
- [Non-Asymptotic Analysis of Efficiency in Conformalized Regression](non-asymptotic_analysis_of_efficiency_in_conformalized_regression.md)
- [Rethinking Consistent Multi-Label Classification Under Inexact Supervision](rethinking_consistent_multi-label_classification_under_inexact_supervision.md)
- [Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](../../NeurIPS2025/optimization/adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)

<!-- RELATED:END -->
