---
title: >-
  [论文解读] Symbolic Regression Is All You Need: From Simulations to Scaling Laws in Binary Neutron Star Mergers
description: >-
  [NeurIPS 2025 (AI for Science Workshop)][物理/科学计算][Symbolic Regression] 利用符号回归（Symbolic Regression）从数值相对论模拟数据中自动发现双中子星并合后吸积盘质量的解析标定关系…
tags:
  - "NeurIPS 2025 (AI for Science Workshop)"
  - "物理/科学计算"
  - "Symbolic Regression"
  - "Binary Neutron Stars"
  - "Gravitational Waves"
  - "Kilonova"
  - "Equation of State"
---

# Symbolic Regression Is All You Need: From Simulations to Scaling Laws in Binary Neutron Star Mergers

**会议**: NeurIPS 2025 (AI for Science Workshop)  
**arXiv**: [2511.08784](https://arxiv.org/abs/2511.08784)  
**代码**: 使用 PySR 和 PyOperon 开源框架  
**领域**: 科学计算  
**关键词**: Symbolic Regression, Binary Neutron Stars, Gravitational Waves, Kilonova, Equation of State  

## 一句话总结

利用符号回归（Symbolic Regression）从数值相对论模拟数据中自动发现双中子星并合后吸积盘质量的解析标定关系，所得紧凑表达式在预测精度、泛化能力和可解释性上全面超越文献中已有的经验拟合公式。

## 研究背景与动机

**领域背景**: 双中子星（BNS）并合是引力波和电磁对应体（kilonova）的核心来源。Kilonova 亮度由动力学抛射物质量 $M_{\text{dyn}}$ 和并合后吸积盘质量 $M_{\text{disk}}$ 共同决定，准确建模这些量对推断中子星状态方程（EoS）至关重要。

**数值模拟的昂贵**: 建立 $M_{\text{disk}}$ 与源参数（质量、致密度、潮汐可变形性）间关系依赖数值相对论（NR）模拟，但每次模拟极其耗时且对 EoS 高度敏感。

**已有经验公式的局限**: Radice18、Coughlin19、Dietrich20、Krüger20、Lund25 等先后提出参数化拟合公式，但模型间差异常超过报告不确定性，外推到标定范围外易产生非物理结果。

**泛化与鲁棒性问题**: 现有公式对微物理和 EoS 选择高度敏感，面对 GW190425 类系统时常用动力学抛射模型预测失败。

**黑箱 ML 的不足**: 传统黑箱机器学习模型虽可能提高精度，但缺乏可解释性，难以提供物理洞察或在无训练数据区域泛化。

**本文动机**: 符号回归可自主搜索模型结构和参数空间，发现紧凑、可解释的解析表达式，兼具精度和物理可解释性。

## 方法详解

### 整体框架

以 NR 模拟数据为训练集，使用两个符号回归框架（PySR、PyOperon）自动搜索 $M_{\text{disk}}$ 的闭合形式表达式。包含三种模式：自由发现（Free-form）、物理先验模板（Physics-informed Template）、以及 LLM 辅助模板（ChatGPT-4.0 生成）。所有模型在 Pareto 前沿上按精度-复杂度权衡选择最优表达式。

### 关键设计

#### 模块一：数据集与评估策略

- **训练集 (Krüger20)**: 56 次 NR 模拟，盘质量 $10^{-4}$–$0.234 M_\odot$，质量比 $q = 0.77$–$1.0$
- **测试集 (Nedora21)**: 包含训练集 + 63 次额外模拟，盘质量 $3 \times 10^{-5}$–$0.30 M_\odot$，质量比 $0.55$–$1.0$
- 关键点: 测试集覆盖更广的物理参数空间，用于评估模型的泛化（外推）能力
- 输入变量: $M_1, M_2, C_1, C_2, q, \tilde{\Lambda}$

#### 模块二：符号回归框架

- **PyOperon**: 基于遗传编程的 SR 方法，算子集 {+, -, *, /, sin, sqrt, log, tanh}，默认超参数
- **PySR**: 进化式 SR 框架，通过多目标损失平衡精度与复杂度，支持 GPU 加速；测试了三组算子集:
    - Core: {+, -, *, /; sqrt, log, exp, abs}
    - Extended: 增加 max, min, square, cube
    - All: 增加 sin, cos, tan, sinh, cosh, tanh, pow
- **训练配置**: 3000 迭代，最大表达式大小 20（模板模式为 10），树深度上限 5
- **模型选择**: 在 Pareto 前沿上选复杂度 < 10 且 MSE 稳定的表达式

#### 模块三：物理先验模板

- **手工模板**: $M_{\text{disk,pred}} = a \cdot f(C_1, M_1) + b \cdot g(C_2, M_2) + c \cdot h(q, \tilde{\Lambda}) + d$，允许隔离各物理参数的影响
- **LLM 模板**: 将文献综述输入 ChatGPT-4.0，让其基于物理直觉生成模板结构: $M_{\text{disk}} = a_1 f(q_0, C_1) + a_2 g(C_1, \tilde{\Lambda}) + a_3 h(M_1, M_2) + a_4 t(q_0) + a_5$
- SR 在模板约束下搜索子表达式 $f, g, h, t$，结合了领域知识与数据驱动的优势

### 损失函数

- 标准多目标优化: 最小化 MSE 的同时惩罚表达式复杂度（每个算子/变量/常数贡献 1 单位复杂度）
- 模型评估指标: MSE, MAE, $R^2$, BIC (贝叶斯信息准则)

## 实验关键数据

### 主实验：SR 模型 vs 文献拟合公式 (Table 1)

| 模型 | MSE | MAE | $R^2$ | BIC | 复杂度 | 依赖变量 |
|------|-----|-----|-------|-----|--------|----------|
| **PyOperon** | **0.002697** | **0.035435** | **0.612** | **-660.9** | 9 | $C_1$ |
| PySR_Free_Ext | 0.002751 | 0.035714 | 0.604 | -658.6 | 9 | $C_1, M_1$ |
| PySR_Free_M12 | 0.002789 | 0.037436 | 0.599 | -637.8 | 13 | $M_1, M_2, \tilde{\Lambda}$ |
| PySR_Template_Ext | 0.002827 | 0.036365 | 0.593 | -602.8 | 20 | $C_1, C_2, \tilde{\Lambda}$ |
| Lund25 (文献) | 0.002910 | 0.036369 | 0.581 | -637.6 | 12 | $C_1$ |
| PySR_Free_All | 0.003029 | 0.036696 | 0.564 | -651.9 | 8 | $C_1$ |
| Krüger20 (文献) | 0.003665 | 0.039238 | 0.473 | -595.8 | 15 | $C_1, M_1$ |
| Radice18 (文献) | 0.006179 | 0.057595 | 0.111 | -524.1 | 12 | $\tilde{\Lambda}$ |

PyOperon 以复杂度仅 9 的表达式 $M_{\text{disk}} = 0.1188 - 0.1430 \times \sin(\sin(40.90 \times C_1))$ 在全部指标上超越文献最优 Lund25，BIC 低 23 点以上。

### 中间盘质量区间性能对比

| 模型 | MSE ($M_{\text{disk}} \sim 0.05$–$0.20 M_\odot$) |
|------|--------------------------------------------------|
| PyOperon | 2.12 × 10⁻³ |
| PySR_Free_Ext | 2.09 × 10⁻³ |
| Lund25 | 1.95 × 10⁻³ |
| Krüger20 | 3.31 × 10⁻³ |

### 关键发现

1. **9 个 SR 表达式中有 4 个在全部指标上超越文献最新 Lund25**，且 PyOperon 的 BIC 优势达 23 点，统计显著性极强
2. **SR 自动重新发现了 $C_1$（轻中子星致密度）的主导作用**: 尽管训练时提供了全部 6 个输入变量，多数最优模型仍选择以 $C_1$ 为核心
3. **替代预测变量集的可能性**: PySR_Free_M12 仅依赖 $(M_1, M_2, \tilde{\Lambda})$——不含 $C_1$——性能媲美 Lund25，这为从电磁观测中约束此前不可及的 EoS 参数开启了新路径
4. **高盘质量区域泛化**: Lund25 无法预测 $M_{\text{disk}} > 0.20 M_\odot$ 的情形，而 PySR_Free_M12 和 PySR_Template_LLM 有效泛化到此区间
5. **LLM 辅助设计模板可行**: ChatGPT-4.0 生成的物理 Ansatz 在模板约束下发现了有竞争力的表达式

## 亮点与洞察

- **从黑箱到白箱**: 用 SR 替代传统 ML，兼顾预测精度与科学可解释性，发现的表达式可直接写入物理论文
- **稀疏数据下的泛化**: 仅用 56 个训练样本标定，即可在更广参数空间（119 个样本）上泛化
- **LLM + SR 的混合范式**: 利用 ChatGPT 的领域知识生成物理模板，再由 SR 搜索具体函数形式，是 AI for Science 的有趣协作模式
- **多目标 Pareto 选择**: 通过复杂度-精度权衡自动避免过拟合，比手工调模型更系统化

## 局限性

1. **仅聚焦吸积盘质量 $M_{\text{disk}}$**: 未涵盖动力学抛射质量 $M_{\text{dyn}}$ 和平均抛射速度等同等重要的量
2. **训练集规模极小（56 个模拟）**: NR 模拟样本稀缺，SR 表达式的统计可靠性仍有待更多数据验证
3. **测试集 Nedora21 与训练集部分重叠**: 包含全部训练样本 + 63 个新样本，并非严格的独立测试
4. **未分析极端区域的物理行为**: 论文承认尚未探索大中子星半径和高致密度极端状态下表达式的定性合理性
5. **SR 表达式的物理可解释性仍有限**: 如 PyOperon 的 $\sin(\sin(\cdot))$ 形式虽精度好，但缺乏直观物理含义

## 相关工作与启发

- **经验拟合公式谱系**: Radice18 → Coughlin19 → Dietrich20 → Krüger20 → Lund25，揭示了吸积盘质量建模的十年进展
- **SR 方法**: PySR (Cranmer, 2023) 和 PyOperon (Burlacu, 2020) 是当前主流的开源 SR 工具
- **AI-Feynman (Udrescu, 2020)** 和 **SciMED (Keren, 2023)**: 论文计划未来使用的替代 SR 算法
- **启发**: SR 作为"AI 驱动的科学发现"工具，在数据稀疏、需要可解释性的物理问题中具有独特优势，可推广到宇宙学、粒子物理等领域的标定关系发现

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 SR 系统性引入 BNS 并合领域，结合 LLM 模板是新颖的混合思路
- 实验充分度: ⭐⭐⭐ — 对比全面（9 个 SR 变体 + 3 个文献公式），但训练/测试集规模小且有重叠
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，公式完整，附录详尽（含 ChatGPT prompt 全文）
- 价值: ⭐⭐⭐⭐ — 对多信使天文学有直接应用价值，发现的替代预测变量集可能开启新的 EoS 约束路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)
- [\[ICLR 2026\] Deep Learning for Subspace Regression](../../ICLR2026/physics/deep_learning_for_subspace_regression.md)
- [\[CVPR 2025\] Improve Representation for Imbalanced Regression through Geometric Constraints](../../CVPR2025/physics/improve_representation_for_imbalanced_regression_through_geometric_constraints.md)
- [\[NeurIPS 2025\] EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale](eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)
- [\[NeurIPS 2025\] From Simulations to Surveys: Domain Adaptation for Galaxy Observations](from_simulations_to_surveys_domain_adaptation_for_galaxy_observations.md)

</div>

<!-- RELATED:END -->
