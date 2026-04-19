---
title: >-
  [论文解读] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems
description: >-
  [NeurIPS 2025][Neural ODE] 系统实证研究 Neural ODE (NODE) 在动力系统中的外推能力和 Symbolic Regression (SR) 的方程恢复能力，发现 NODE 在动态相似条件下可外推到新边界条件，并提出 NODE→SR 流水线：仅用 10% 原始数据训练 NODE 生成增强数据，SR 即可恢复 2/3 的控制方程和 1/3 的良好近似。
tags:
  - NeurIPS 2025
  - Neural ODE
  - symbolic regression
  - dynamical systems
  - extrapolation
  - scientific discovery
---

# An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems

**会议**: NeurIPS 2025  
**arXiv**: [2601.20637](https://arxiv.org/abs/2601.20637)  
**代码**: 有 (基于 JAX/Diffrax + PySR)  
**领域**: 科学计算 / 科学发现  
**关键词**: Neural ODE, symbolic regression, dynamical systems, extrapolation, scientific discovery

## 一句话总结

系统实证研究 Neural ODE (NODE) 在动力系统中的外推能力和 Symbolic Regression (SR) 的方程恢复能力，发现 NODE 在动态相似条件下可外推到新边界条件，并提出 NODE→SR 流水线：仅用 10% 原始数据训练 NODE 生成增强数据，SR 即可恢复 2/3 的控制方程和 1/3 的良好近似。

## 研究背景与动机

**科学发现的数据驱动范式**。从实验数据中自动发现动力系统的控制方程是加速科学发现的核心问题。Neural ODE 因其连续时间特性天然适合建模微分方程描述的动力系统，已在流体力学、药物动力学等领域成功应用。

**NODE 的外推盲区**。现有研究主要关注 NODE 架构改进和鲁棒性评估，但对其在有噪声合成数据或真实条件下的关键能力——特别是对新边界条件（新初始条件）和新时间范围的外推能力——研究不足。这是实际应用中最关键的需求：我们不可能穷举所有初始条件来训练。

**SR 的数据饥渴问题**。Symbolic Regression (SR) 能发现可解释的符号方程，但通常需要大量高质量数据，这在实验科学中往往不可得。核心矛盾是：NODE 能从少量数据学习动力学但输出黑箱模型，SR 能输出可解释方程但需要大量数据。本文的核心 idea 是将两者组合——用 NODE 作为数据增强工具，从少量真实数据生成大量轨迹供 SR 做方程恢复。

## 方法详解

### 整体框架

提出 NODE→SR 科学发现流水线：(1) 从含噪声的稀疏真实数据训练 NODE；(2) 用训练好的 NODE 生成大量覆盖多种条件的轨迹（数据增强）；(3) 对增强数据用 PySR 做符号回归恢复控制方程。同时系统评估 NODE 的外推/内插能力和 SR 在不同数据条件下的方程恢复成功率。

### 关键设计

1. **NODE 外推能力的系统评估**:

    - 功能：研究 NODE 在何种条件下能泛化到训练分布之外
    - 核心思路：在 cart-pole 和生物系统两个 damped oscillation 系统上，分别训练仅覆盖部分初始条件/时间范围的 NODE，然后测试在训练域外的表现。Cart-pole Model B 仅训练小范围初始条件，但在相空间中与训练数据共享轨迹的区域也表现良好
    - 设计动机：揭示 NODE 外推成功的关键条件——"动态相似性"（dynamic similarity）：新初始条件的轨迹与训练数据在相空间中共享相同的动力学特性时外推成功，否则失败

2. **NODE 作为去噪器和数据增强器**:

    - 功能：从有噪声的稀疏数据生成大量干净的增强轨迹
    - 核心思路：NODE 仅用 10% 的原始模拟数据（2 小时 × 12 种 shift 条件 × 每小时 10 个数据点）训练，然后生成完整的 8 小时轨迹供 SR 使用。实验发现 NODE 起到了去噪滤波器的作用——在含 5% 噪声的数据上训练的 NODE 生成的数据让 SR 恢复了比直接在噪声数据上做 SR 更多的方程
    - 设计动机：实验科学中数据总是有噪声且有限的，NODE 的连续性和平滑性天然提供了去噪能力

3. **SR 对输入变量选择的敏感性分析**:

    - 功能：揭示 SR 成功发现方程的前提条件
    - 核心思路：在生物模型上，当提供中间变量 $\lambda$ 作为输入时 SR 成功恢复全部 3 个方程；仅提供 3 个主状态变量时只恢复 1 个。原因是 $\lambda$ 中的有理项 $\frac{\psi_A \phi_R}{\psi_A + k_\alpha}$ 在数据范围内被近似为 $\phi_R$（因为 $\psi_A \gg k_\alpha$），遮蔽了真实结构
    - 设计动机：说明 SR 不仅需要足够的数据，还需要合适的特征工程——输入变量的选择会根本性地影响方程发现的成败

### 损失函数 / 训练策略

NODE 使用标准的 MSE 损失训练，基于 JAX 的 Diffrax 库实现。cart-pole Model A 训练数据为 35 个初始条件 × 前 1 秒 × 25 Hz 采样；Model B 使用更小的训练窗口。生物模型 NODE 训练数据为 12 种 nutrient shift × 前 2 小时 × 10 Hz。SR 使用 PySR 工具包，默认搜索设置。

## 实验关键数据

### 主实验
| 系统 | 数据源 | 方程2 | 方程3 | 方程4 | 说明 |
|------|--------|:---:|:---:|:---:|------|
| Bio-model (真实数据, 无噪声, 含λ) | Ground Truth | ✓ | ✓ | ✓ | SR全部恢复 |
| Bio-model (真实数据, 5%噪声, 含λ) | Ground Truth | ✗ | ✓ | ✗ | 噪声严重影响发现 |
| Bio-model (NODE数据, 无噪声, 含λ) | NODE增强 | ~✓(近似) | ✓ | ✓ | NODE→SR 恢复2/3+近似1/3 |
| Bio-model (NODE数据, 5%噪声, 含λ) | NODE增强 | ~✓(近似) | ✓ | ✓ | NODE去噪效果显著 |
| Cart-pole (Model A) | NODE | - | - | - | 训练1秒→外推5秒成功 |
| Cart-pole (Model B) | NODE | - | - | - | 训练小范围→外推到动态相似区域成功 |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| NODE 训练采样频率 (5-100 Hz) | 8小时MSE无显著差异 | 长期预测对采样频率不敏感 |
| NODE 训练采样频率 (5-100 Hz) | 1小时MSE在<10Hz时偏高 | 极稀疏数据(<6点/变量)影响内插精度 |
| SR 含中间变量λ | 3/3方程恢复(无噪声) | 正确的输入变量是关键 |
| SR 仅主状态变量 | 1/3方程恢复 | 有理项被数据范围遮蔽 |
| SR on NODE (无噪声) vs Ground Truth (5%噪声) | NODE: 2/3+近似 vs GT: 1/3 | NODE起到去噪作用 |

### 关键发现
- **NODE 外推的关键条件是"动态相似性"**：不是初始条件在数值上接近训练数据，而是在相空间中共享相同的动力学轨迹。这意味着训练集应追求动力学多样性而非密集采样
- **NODE 充当去噪滤波器**：在含 5% 噪声数据上训练的 NODE 生成的增强数据，让 SR 恢复了更多方程（2/3），而直接在噪声数据上 SR 只恢复了 1/3
- **SR 对方程结构有"遮蔽"问题**：当数据范围使某些数学结构（如有理分式）退化为更简单形式时，SR 无法发现真实结构
- **采样频率对长期预测影响很小**：5 Hz 到 100 Hz 训练的 NODE 在 8 小时预测上 MSE 无显著差异——少量数据即可训练可用的 NODE

## 亮点与洞察
- **NODE→SR 流水线**是数据稀缺场景下的实用科学发现方案：用 10% 数据训练 NODE，生成增强数据给 SR 做方程发现
- "动态相似性"概念对 NODE 训练集设计有指导意义：追求轨迹多样性而非初始条件的密集覆盖
- NODE 的去噪效果是意外发现：连续时间模型的平滑归纳偏置自然过滤了测量噪声
- 对 SR 的 failure mode 分析（变量选择敏感性、数据范围遮蔽）有实际参考价值

## 局限与展望
- 仅在 2 个相对简单的 damped oscillation 系统上验证（cart-pole、细菌营养适应），混沌系统、高维系统未探索
- SR 分析仅使用单次 shift 数据（单一初始条件→终态），多条件联合分析可能改善方程发现
- NODE 架构使用基础版本，augmented NODE 或 Neural CDE 等改进架构可能提升外推能力
- SR 框架限于 PySR，SINDy 等结合物理先验的方法可能更适合
- 生物模型的方程 (2) 始终未被完美恢复（即使从 NODE 数据），说明流水线对特定方程结构仍有局限

## 相关工作与启发
- 与 SINDy 的关系：SINDy 使用稀疏回归发现方程，需要预定义函数库；PySR 搜索空间更大但对数据需求也更高。两者互补
- 与 augmented NODE 的关系：本文使用基础 NODE，augmented NODE 通过扩展状态空间可能改善外推
- 启发：NODE→SR 流水线可推广到其他实验科学领域——只要有少量时序测量数据，就可能自动发现控制方程

## 评分
- 新颖性: ⭐⭐⭐ 组合已有方法（NODE + SR），但"动态相似性"发现和 NODE 去噪效应有价值
- 实验充分度: ⭐⭐⭐ 两个系统偏少，但消融分析（采样频率、噪声、变量选择）较系统
- 写作质量: ⭐⭐⭐⭐ 实验设计清晰，failure mode 分析诚恳深入
- 价值: ⭐⭐⭐ 对数据稀缺的科学发现流水线有参考意义，但需在更复杂系统上验证

<!-- RELATED:START -->

## 相关论文

- [Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)
- [Generalization Bounds for Rank-sparse Neural Networks](generalization_bounds_for_rank-sparse_neural_networks.md)
- [Stochastic Self-Organization in Multi-Agent Systems](../../ICLR2026/llm_pretraining/stochastic_self-organization_in_multi-agent_systems.md)
- [Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)
- [Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding](learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)

<!-- RELATED:END -->
