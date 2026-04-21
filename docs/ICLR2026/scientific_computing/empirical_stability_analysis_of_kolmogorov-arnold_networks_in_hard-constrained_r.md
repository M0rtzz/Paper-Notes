---
title: >-
  [论文解读] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery
description: >-
  [ICLR2026][科学计算][KAN] 在硬约束递归物理信息架构（HRPINN）中系统评估vanilla KAN替代MLP作为残差分支的效果——通过3项互补研究×100随机种子发现KAN在单变量可分残差（Duffing的 $-0.3x^3$）上的表现具有竞争力，但在乘法耦合残差（Van der Pol的 $(1-x^2)v$）上系统性失败且超参数极度脆弱，标准MLP在几乎所有配置下稳定性远优。
tags:
  - ICLR2026
  - 科学计算
  - KAN
  - physics-informed
  - oscillator
  - HRPINN
  - neural ODE
  - 残差发现
---

# Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery

**会议**: ICLR2026  
**arXiv**: [2602.09988](https://arxiv.org/abs/2602.09988)  
**代码**: 未开源  
**领域**: 科学计算/物理信息神经网络  
**关键词**: KAN, physics-informed, oscillator, HRPINN, neural ODE, 残差发现

## 一句话总结

在硬约束递归物理信息架构（HRPINN）中系统评估vanilla KAN替代MLP作为残差分支的效果——通过3项互补研究×100随机种子发现KAN在单变量可分残差（Duffing的 $-0.3x^3$）上的表现具有竞争力，但在乘法耦合残差（Van der Pol的 $(1-x^2)v$）上系统性失败且超参数极度脆弱，标准MLP在几乎所有配置下稳定性远优。

## 研究背景与动机

**领域现状**：硬约束递归物理信息架构（HRPINN）将已知物理嵌入递归积分器，神经网络只学习残差动力学——这确保了物理一致性并已在网络物理系统中验证有效。同时，Kolmogorov-Arnold Networks（KAN）基于Kolmogorov-Arnold表示定理，将多元函数分解为单变量函数之和 $\Phi(\mathbf{x}) = \sum_q \phi_q(\sum_p \psi_{q,p}(x_p))$，用可学习的B-样条替代MLP的固定激活函数，在科学ML中展现潜力。

**现有痛点**：
- KAN在Neural ODE和灰箱设置中展现了符号发现潜力（KAN-ODEs、SKANODEs），但这些工作使用的是无约束的连续ODE设置
- 没有人在硬约束递归架构中测试过KAN——递归设置中误差会累积，对稳定性要求更高
- KAN的加法归纳偏置（$\phi(x) + \phi(v)$）理论上适合可分物理定律，但在实际中是否成立？

**核心矛盾**：KAN的加法结构天然适合加法可分函数，但许多物理定律包含乘法耦合项（如Van der Pol的 $(1-x^2)v$）。理论上KAN可通过深层组合表示乘法（$xy = \frac{1}{4}((x+y)^2 - (x-y)^2)$），但这需要更深的层→深层KAN在递归误差累积下是否仍然稳定？

**本文目标**：为vanilla KAN在硬约束递归物理架构中建立基线评估。

**切入角度**：选择两个具有对比残差结构的经典振荡器——Duffing（单变量多项式）和Van der Pol（乘法耦合）——作为加法可分性的边界测试。

**核心 idea**：通过精心对照的实验设计，揭示KAN的加法归纳偏置在递归物理约束下的实际成功边界。

## 方法详解

### 整体框架：HRPINN + KAN/MLP残差分支对照

HRPINN框架中，残差分支 $R_\theta(x, v)$ 接收归一化状态 $[x, v]$，分别用标准ReLU MLP和B-spline KAN实现。已知物理和积分器固定在递归更新规则中，网络仅学习残差流形。Performance用测试MSE和Discovery $R^2$（网格密度相关性）评估，后者在 $100 \times 100$ 网格上对 $x, v \in [-2.5, 2.5]$ 的相空间计算。采用统一的候选拟合方法（非KAN特有的符号剪枝）确保公平对比。

### 关键设计1：配置消融——超参数敏感性系统评估

设计7种KAN网格配置（grid size $G$ 和 spline order $k$ 的不同组合），固定训练设置，每种配置100个随机种子。结果揭示：

- 粗网格配置（Config F, $G=3, k=3$）在Duffing上达到 $R^2 = 0.862$，与MLP（$0.957$）缩小差距
- 但多数配置在Van der Pol上产生负 $R^2$（发散解），如Config C的 $R^2 = -5.229 \pm 5.091$
- MLP（337参数）稳定达到 Duffing $R^2 = 0.957$、VdP $R^2 = 0.768$，方差极小

**设计动机**：通过穷举配置空间并用大量种子统计，区分"KAN的某个配置碰巧好"和"KAN作为架构宽鲁棒地好"→结论是前者。

### 关键设计2：参数规模消融×两种训练范式

固定配置、变化参数量（Very Small 120 → Deep 880），分别在单步Teacher Forcing和BPTT两种训练范式下测试：

| 训练范式 | KAN行为 | MLP行为 |
|---------|---------|---------|
| Teacher Forcing | 小KAN在Duffing竞争力强、VdP随规模迅速退化 | 平稳扩展 |
| BPTT | 最小KAN达到VdP最佳 $R^2 \approx 0.74$（长时域监督有帮助）、深KAN不稳定 | 所有规模稳定优越 |

对比揭示了关键瓶颈：MLP的稠密矩阵乘法在第一层就实现变量交互（$w_i x + w_j v$），而KAN的加法偏置（$\phi(x) + \phi(v)$）需要通过深层组合来近似乘法→在递归误差累积下深层组合不稳定。

### 关键设计3：定性验证——残差流形可视化

展示KAN和MLP学到的残差曲面与真实解的对比：
- Duffing：KAN准确再现立方流形，候选拟合得到 $-0.234x^3$（真值 $-0.3x^3$），$R^2 = 0.91$
- Van der Pol：KAN曲面坍缩为近似线性形式→未能捕捉 $(1-x^2)v$ 的抛物线调制结构

这种定性验证与定量统计互相印证：KAN的加法偏置在单变量上是优势，在变量耦合上是瓶颈。

## 实验关键数据

### 配置消融主表（95% Bootstrap CI, N=100 seeds）

| 配置 | Duffing $R^2$ | Van der Pol $R^2$ |
|------|:---:|:---:|
| KAN Config A ($G=5, k=3$) | 0.835 ± 0.030 | 0.667 ± 0.037 |
| KAN Config C (Sparse-Low) | 0.595 ± 0.033 | **-5.229 ± 5.091** |
| KAN Config E (Aggressive-Grid) | 0.794 ± 0.067 | 0.699 ± 0.065 |
| KAN Config F (Coarse-Grid) | **0.862 ± 0.037** | 0.639 ± 0.302 |
| KAN Config G (Fine-Grid) | 0.745 ± 0.099 | -0.174 ± 0.691 |
| **MLP (337 params)** | **0.957 ± 0.009** | **0.768 ± 0.015** |

### 参数规模消融（Mean ± 95% CI, N=100 seeds）

| 架构 | 参数 | Duffing(TF) | VdP(TF) | Duffing(BPTT) | VdP(BPTT) |
|------|:---:|:---:|:---:|:---:|:---:|
| KAN Very Small | 120 | 0.836±0.032 | 0.464±0.166 | 0.914±0.061 | 0.743±0.061 |
| KAN Small | 240 | 0.777±0.079 | 0.322±0.292 | 0.874±0.080 | 0.785±0.073 |
| KAN Wide | 480 | 0.845±0.025 | 0.232±0.570 | 0.468±0.773 | -0.602±2.842 |
| KAN Deep | 880 | -3.146±7.106 | -0.303±1.579 | (不稳定) | 0.754±0.079 |
| MLP Tiny | 105 | 0.914±0.026 | 0.593±0.048 | 0.906±0.092 | 0.622±0.173 |
| MLP Small | 337 | 0.957±0.009 | 0.768±0.015 | 0.937±0.047 | 0.879±0.032 |
| MLP Medium | 1185 | 0.960±0.013 | 0.805±0.014 | 0.951±0.033 | 0.879±0.019 |
| **MLP Large** | **4417** | **0.965±0.009** | **0.843±0.010** | **0.932±0.063** | **0.898±0.017** |

### 关键发现

- KAN在Duffing上可发现立方结构（$-0.234x^3$，真值 $-0.3x^3$，$R^2=0.91$），38%种子成功→有潜力但不可靠
- KAN在Van der Pol上系统性失败→加法偏置无法稳定学习乘法耦合
- BPTT的长时域监督帮助最小KAN缓解VdP问题（$R^2$ 从0.464升至0.743），但MLP仍全面占优
- KAN超参数敏感度远高于MLP——VdP上从0.699到-5.229→实践中不实用
- 深层KAN（880参数）在递归设置中灾难性不稳定（$R^2 = -3.146$）

## 亮点与洞察

- **诚实的"负面结果"**——清楚展示了当前vanilla KAN在物理约束递归架构中的实际边界，为KAN社区提供了重要警示
- **加法偏置vs乘法耦合的精准诊断**：选择Duffing和Van der Pol作为恰好跨越加法可分性边界的测试对→诊断直击KAN设计核心假设
- **大规模种子统计的可信度**：每项实验100个随机种子+95%置信区间→结论不依赖于幸运的初始化
- **递归误差累积的独特洞察**：KAN在无约束ODE中可能表现尚可，但在硬约束递归设置中误差快速放大→揭示了设置依赖性

## 局限与展望

- 仅测试vanilla KAN——改进变体（SKANODEs、Hybrid KAN-MLP、DeepOKAN）可能克服乘法限制
- 仅两个振荡器系统——更复杂/混沌系统（Lorenz吸引子）待测试
- 未与SINDy等成熟符号发现方法对比
- 未探究KAN独有的符号剪枝能力——直接通过样条结构提取符号表达式
- 未分析梯度条件数/优化景观——仅展示了"什么失败"但未完全解释"为什么失败"

## 相关工作与启发

- **vs KAN-ODEs (Koenig et al., 2024)**：在无约束连续ODE中表现好→本文揭示硬约束递归设置下的脆弱性→设置依赖性是关键
- **vs SKANODEs (Liu et al., 2025)**：结构化KAN可能通过算子链接（分别表示 $1-x^2$ 再与 $v$ 交互）缓解乘法问题→启发混合方案
- **启发**：可否设计"乘法感知KAN"→在KAN基础层引入显式乘法门→保留加法偏置的可解释性同时处理耦合项？

## 评分

⭐⭐⭐⭐ (4/5)

综合评价：系统性的负面结果论文，100种子×3项研究的极其充分实证支撑了关于KAN加法偏置边界的精确论断——虽非新方法，但为KAN在物理信息应用中的实践提供了不可或缺的校准参考。

<!-- RELATED:START -->

## 相关论文

- [Astral: Training Physics-Informed Neural Networks with Error Majorants](astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](../../AAAI2026/scientific_computing/pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)
- [Neuro-Spectral Architectures for Causal Physics-Informed Networks](../../NeurIPS2025/scientific_computing/neuro-spectral_architectures_for_causal_physics-informed_networks.md)
- [Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](../../ICML2025/scientific_computing/differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data](../../NeurIPS2025/scientific_computing/multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)

<!-- RELATED:END -->
