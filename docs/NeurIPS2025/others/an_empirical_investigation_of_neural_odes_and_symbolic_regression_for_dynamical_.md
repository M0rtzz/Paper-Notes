---
title: >-
  [论文解读] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems
description: >-
  [NeurIPS 2025][Neural ODE] 本文系统研究了 Neural ODE (NODE) 在含噪合成数据上的外推能力，并探索了将 NODE 作为数据增强工具、与符号回归 (SR) 结合以从有限数据中恢复动力学方程的流水线，结果表明该组合方案能从仅 10% 的仿真数据中恢复三个控制方程中的两个及第三个的良好近似。
tags:
  - NeurIPS 2025
  - Neural ODE
  - Symbolic Regression
  - 动态系统
  - 外推能力
  - 科学发现
---

# An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems

**会议**: NeurIPS 2025  
**arXiv**: [2601.20637](https://arxiv.org/abs/2601.20637)  
**代码**: 有（基于 JAX/Diffrax + PySR）  
**领域**: Scientific Discovery / Dynamical Systems  
**关键词**: Neural ODE, Symbolic Regression, 动态系统, 外推能力, 科学发现

## 一句话总结

本文系统研究了 Neural ODE (NODE) 在含噪合成数据上的外推能力，并探索了将 NODE 作为数据增强工具、与符号回归 (SR) 结合以从有限数据中恢复动力学方程的流水线，结果表明该组合方案能从仅 10% 的仿真数据中恢复三个控制方程中的两个及第三个的良好近似。

## 研究背景与动机

准确建模复杂系统动力学或发现其控制微分方程是加速科学发现的核心任务。随着数据爆炸和机器学习技术的发展，如何利用实验数据推断复杂系统的动力学成为关键问题。

**Neural ODE** 因其连续时间特性天然契合微分方程建模，已在湍流预测、药代动力学等领域成功应用。但现有研究主要聚焦架构改进和鲁棒性评估，对 NODE 在含噪数据下的**边界条件外推能力**研究不足。

**符号回归 (SR)** 能直接发现精确的控制方程，但通常需要大量数据，在实验科学中常常难以满足。

本文的核心动机是：能否用训练好的 NODE 作为**数据增强工具**，将有限的实验数据扩展为更完整的数据集，从而帮助 SR 恢复底层物理方程？这为**数据稀缺场景**下的科学发现提供了一条实用路径。

## 方法详解

### 整体框架

本文的研究框架分为两个相互衔接的部分：

1. **NODE 外推能力研究**：使用含噪合成数据训练 NODE，评估其对新边界条件和更长时间段的泛化能力
2. **NODE + SR 流水线**：用仅 10% 原始仿真数据训练 NODE → NODE 生成完整数据集 → SR 从中恢复控制方程

研究采用两个阻尼振荡系统作为测试 benchmark：
- **Cart-pole 系统**：无外力作用的带摩擦倒立摆，用角度 θ 和角速度 ω 描述状态
- **Bio-model**：细菌适应营养环境变化的生物模型，包含三个状态变量 (ψ_A, ϕ_R, χ_R) 和三个耦合微分方程

### 关键设计

**NODE 训练与外推策略**：

对于 Cart-pole 系统，设计了两个模型：
- **Model A**：35 组边界条件（初始角度 × 角速度的组合），采样率 25 Hz，仅用第一秒数据训练
- **Model B**：仅在有限的初始条件区域内训练，用于评估外推能力

对于 Bio-model：
- **Model 2A**：仅在两组上升偏移 (up-shift) 仿真数据上训练
- 还训练了不同采样频率 (5-100 数据点/小时) 的多个模型

**符号回归分析**：
- 使用 PySR 框架进行符号回归
- 分别在**真实数据**和 **NODE 生成数据**上进行 SR 分析
- 对比有/无噪声条件下的方程恢复效果
- 核心发现是输入变量选择对 SR 成功率有决定性影响

### 损失函数 / 训练策略

- NODE 使用 JAX-based Diffrax 库实现，训练采用标准的常微分方程拟合目标
- 合成数据添加 ±5% 的均匀分布噪声，模拟真实实验条件
- Cart-pole 仿真运行 10 秒，时间步长 0.01 秒
- Bio-model 仿真运行 8 小时，时间步长 0.01 小时

## 实验关键数据

### 主实验

**表1：SR 分析结果 — 不同数据源对三个控制方程的恢复情况**

| 方程 | 真实数据 (无噪声) | 真实数据 (5%噪声) | NODE数据 (无噪声) | NODE数据 (5%噪声) |
|------|:-:|:-:|:-:|:-:|
| 方程2 | ✓ | ✗ | ✗* | ✗* |
| 方程3 | ✓ | ✓ | ✓ | ✓ |
| 方程4 | ✓ | ✗ | ✓ | ✓ |

*注：✗* 表示未找到精确方程但找到了较好近似*

该表揭示了三个关键发现：
1. 在包含 λ 作为输入变量时，无噪声真实数据上 SR 能恢复全部三个方程
2. 5% 噪声严重影响 SR 在真实数据上的表现（仅恢复 1/3）
3. NODE 数据上 SR 恢复了 2/3 的方程，且在噪声数据上表现更好 — NODE 起到了**去噪滤波器**的作用

**表2：方程2的 SR 恢复详情**

| 数据集 | MSE Loss | SR 结果 |
|--------|----------|---------|
| 真实数据 (无噪声) | 2.09×10⁻⁸ | 2.076 - 3.77×ϕ_R - λ - λ×ψ_A |
| 真实数据 (5%噪声) | 8.12×10⁻³ | 1.410 - λ（过度简化）|
| NODE (无噪声) | 1.91×10⁻⁵ | 2.024 - 3.54×ϕ_R - λ - ψ_A |
| NODE (5%噪声) | 2.16×10⁻⁴ | 2.006 - 3.67×ϕ_R - λ |

方程2的真实形式为 `2.079 - 3.78×ϕ_R - λ - λ×ψ_A`。可以看到：
- 无噪声真实数据几乎完美恢复
- NODE 无噪声数据遗漏了 λ×ψ_A 项中的 λ 系数，用 ψ_A 代替
- NODE 有噪声数据进一步丢失了 ψ_A 项，但将其均值吸收进了常数（2.006 vs 2.079）

### 消融实验

**采样频率对模型性能的影响**：

在 Bio-model 上，训练了 6 个不同采样频率的模型（5, 10, 20, 33, 50, 100 数据点/小时），每个模型仅用第一小时数据训练。关键发现：

- **8 小时 MSE 高度一致**：不同采样频率间无显著差异，说明高质量的长期预测可通过极少量训练数据实现
- **1 小时 MSE**：最低两个采样频率（5 和 10 点/小时）显著更高，因为每个变量每个偏移仅有约 6 个训练点，噪声影响放大
- 结果跨 19 个偏移和 10 次独立运行取平均

**输入变量选择的影响**：

当 SR 的输入变量仅包含三个主状态变量 (ψ_A, ϕ_R, χ_R) 而非包含 λ 时，仅能恢复方程4。原因在于 λ 中的有理项 ψ_A·ϕ_R/(ψ_A + k_a) 被数据"掩盖" — 由于 ψ_A 平均比 k_a 大 8 倍，分母近似等于 ψ_A，使得整个有理项约等于 ϕ_R，隐藏了真实结构。

### 关键发现

1. **NODE 的相空间外推**：Model B 在训练区域外展现出低 MSE 区域，分析发现这些区域与训练数据**共享相同的相空间轨迹**。这表明 NODE 泛化的关键是捕获多样化的动力学特征，而非密集采样初始条件

2. **跨偏移泛化**：Model 2A 仅在两组 up-shift 数据上训练，却能准确预测 down-shift 响应（误差 < 5%），展示了 NODE 对未见过的动力学模式的泛化能力

3. **时间外推**：Model A 仅用第一秒数据训练，能准确外推 5 倍训练时长（5秒）的动力学

4. **NODE 的去噪效应**：在含 5% 噪声的真实数据上 SR 仅恢复 1/3 方程，而在同样含噪声数据训练的 NODE 输出上 SR 恢复 2/3 + 1个近似，说明 NODE 能有效滤除噪声

## 亮点与洞察

- **"多样化动力学 > 密集采样"** 的核心洞察对实验设计有直接指导意义：在设计训练集时，应优先覆盖不同的动力学行为模式，而非简单增加初始条件数量
- **NODE 作为数据增强器 + 去噪器** 的双重角色是一个优雅的发现：一个模型同时解决了数据稀缺和数据质量两个问题
- 仅用 10% 的仿真数据就能通过 NODE→SR 流水线恢复 2/3 的控制方程，展示了该方法在数据稀缺科学领域的实用潜力

## 局限与展望

1. SR 分析仅基于单次偏移仿真，扩展到多条件、多样化数据可能显著提升方程恢复率
2. NODE 的训练数据未经优化以最大化泛化能力，可进一步探索训练集设计策略
3. 可引入更先进的架构（如 Neural CDE），可能改善数据质量和外推范围
4. SR 框架本身可通过物理先验（如单位匹配）或替代框架（如 SINDy）加强
5. 方程2的恢复失败部分源于低信噪比项（λ·ψ_A 量级仅 0.01-0.13），需要更精细的信号处理策略

## 相关工作与启发

- **Neural ODE** [Chen et al., 2018]：本文基于此进行外推能力的系统评估
- **PySR** [Cranmer, 2023]：用于符号回归的核心工具
- **SINDy** [Brunton et al., 2016]：另一种从数据发现动力学方程的方法，可作为替代 SR 框架
- **Neural CDE** [Kidger et al., 2020]：可处理不规则时间序列的高级 NODE 变体

本文为 NODE 与 SR 结合提供了初步但有说服力的证据，启发了将深度学习模型作为"数据放大器"来辅助可解释性方法的研究方向。

## 评分

- **新颖性**: ⭐⭐⭐ — 方法组件均已有，但 NODE→SR 流水线的组合及去噪效应发现有新意
- **技术深度**: ⭐⭐⭐ — 实验设计系统且分析细致，但无理论分析
- **实验质量**: ⭐⭐⭐⭐ — 两个系统、多组消融、误差分析完整
- **实用性**: ⭐⭐⭐⭐ — 对数据稀缺的实验科学有直接应用价值
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表信息量大

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Modeling Neural Activity with Conditionally Linear Dynamical Systems](modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)
- [\[NeurIPS 2025\] Lagrangian neural ODEs: Measuring the existence of a Lagrangian with Helmholtz metrics](lagrangian_neural_odes_measuring_the_existence_of_a_lagrangian_with_helmholtz_me.md)
- [\[NeurIPS 2025\] Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems](efficient_parametric_svd_of_koopman_operator_for_stochastic_dynamical_systems.md)
- [\[NeurIPS 2025\] Regression Trees Know Calculus](regression_trees_know_calculus.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)

<!-- RELATED:END -->
