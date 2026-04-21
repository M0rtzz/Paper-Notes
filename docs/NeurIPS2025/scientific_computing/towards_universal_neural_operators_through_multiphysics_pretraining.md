---
title: >-
  [论文解读] Towards Universal Neural Operators through Multiphysics Pretraining
description: >-
  [NeurIPS 2025][科学计算][神经算子] 提出基于 adapter 的多物理场预训练框架，通过将 lifting/projection 层作为问题特定适配器、冻结共享的核积分算子层，实现跨 PDE 问题的迁移学习，显著降低微调成本并提升泛化能力。
tags:
  - NeurIPS 2025
  - 科学计算
  - 神经算子
  - 迁移学习
  - 偏微分方程
  - multiphysics
  - foundation model
---

# Towards Universal Neural Operators through Multiphysics Pretraining

**会议**: NeurIPS 2025  
**arXiv**: [2511.10829](https://arxiv.org/abs/2511.10829)  
**代码**: 待确认  
**领域**: 科学计算 / 神经算子  
**关键词**: neural operator, transfer learning, PDE, multiphysics, foundation model

## 一句话总结

提出基于 adapter 的多物理场预训练框架，通过将 lifting/projection 层作为问题特定适配器、冻结共享的核积分算子层，实现跨 PDE 问题的迁移学习，显著降低微调成本并提升泛化能力。

## 研究背景与动机

**核心问题：** 神经算子（Neural Operator, NO）在数据驱动的物理仿真中已广泛使用，但训练代价高昂，且每个新 PDE 问题通常需要从零训练一个独立模型。如何构建一个通用的神经算子基础模型，使其能够在多种 PDE 问题上预训练后高效迁移到新问题？

**现有方案的不足：**

**PINN（物理信息神经网络）：** 需要显式的 PDE 公式，且精度仅在训练网格节点上有保证，泛化能力有限

**传统神经算子（FNO/DeepONet）：** 虽然能近似函数空间映射，具有离散化不变性，但每个问题独立训练，无法复用已学到的物理知识

**已有的预训练方法：** 多数局限于特定类型方程（如稳态方程），或仅在同类 PDE 内做参数外推，缺乏跨物理场景的通用迁移能力

**CoDA-NO：** 虽引入了 codomain attention 做多物理迁移，但其预训练和微调流程仍不够灵活

**本文动机：** 受大语言模型中 adapter 微调范式的启发，将神经算子的 lifting 和 projection 层视为轻量级适配器，核积分算子层（Fourier 层/Transformer 层）视为共享的"主干"。预训练阶段在多个不同物理问题上联合训练全部参数，微调阶段仅更新新问题的 adapter，从而实现：(a) 跨 PDE 知识迁移；(b) 大幅降低微调计算量；(c) 支持不同输入函数集的 PDE 问题。

## 方法详解

### 整体框架

本文的核心架构遵循标准神经算子的 **Lifting → Operator Blocks → Projection** 三阶段设计，但在此基础上引入了 **adapter-based 多物理场预训练/微调** 策略：

- **Lifting 层 $\mathcal{L}$：** 将输入函数 $\mathbf{a} = \{a_1, \dots, a_{n\_in}\}$ 映射到高维隐空间，参数为 $\theta_\mathcal{L} = \{A_\mathcal{L}, b_\mathcal{L}\}$
- **核积分算子层 $\mathcal{F}$：** 包含 $n_{\text{layers}}$ 个叠加的积分核算子块，每层计算 $\mathcal{F}_t(x) = \sigma\left(A_t v_t(x) + \int_{D_i} \kappa_t(x,y) v_t(y) dy + b_t(x)\right)$
- **Projection 层 $\mathcal{P}$：** 将最后一层隐表示投影回输出函数空间

**关键思想：** 不同 PDE 问题拥有各自独立的 lifting 和 projection 层（adapter），而核积分算子层在所有问题间共享。预训练时联合优化全部参数 $(\theta_{\mathcal{P}_1}, \dots, \theta_{\mathcal{P}_N}, \theta_\mathcal{F}, \theta_{\mathcal{L}_1}, \dots, \theta_{\mathcal{L}_N})$；微调时冻结 $\theta_\mathcal{F}$，仅训练新问题的 $(\theta_{\mathcal{P}_{ft}}, \theta_{\mathcal{L}_{ft}})$。

### 关键设计：两类增强架构

为提升神经算子在迁移学习中的泛化能力，文章探索了两种结构改进：

**1. Mamba-SSM 增强的 FNO（MambaFNO）：**

在 lifting 层之后插入 Mamba 状态空间模块 $\mathcal{M}_\phi$，对提升后的特征做因果卷积：

$$\widetilde{v}_0(x,t) = (\mathcal{M}_\phi v_0)(x,t) = \sum_{\tau \leq t} K_\tau v_0(x, t-\tau)$$

其中 $K_\tau$ 为可学习卷积核。Mamba 模块的作用是**隐空间预条件化**：在进入 Fourier 层前，将 embedding 与主导动力学模式（输运、扩散、振荡）对齐，降低输入信号的频谱秩和变异性，从而使后续 $\mathcal{F}_t \circ \mathcal{M}_\phi$ 的训练更稳定，预训练表示在微调时更高效迁移。

**2. Perceiver IO 增强的神经算子：**

引入 Perceiver IO 的对称交叉注意力机制：
- **编码阶段：** 输入经 FNO 映射得到 $K_1 = \text{FNO}_{K_1}(X)$, $V_1 = \text{FNO}_{V_1}(X)$，与可学习的潜变量 $Q_1 = L$ 做交叉注意力
- **处理阶段：** 潜表示间的自注意力
- **解码阶段：** 用输入的查询与变换后的潜表示做交叉注意力输出

Perceiver 的优势在于用更少的潜特征数组编码信息，操作更抽象的特征表示，参数量可控。

此外，文章还对比了 **Codomain Attention（CoDA-NO）** 和 **Swin-v2 Transformer** 两种基线，其中 codomain attention 的相似度计算在特征维度（而非样本维度）上进行点积，更适合神经算子场景。

### 损失函数

采用范围归一化平均绝对误差（NMAE）作为训练和评估指标：

$$\text{NMAE}(\theta) = \frac{1}{|\mathcal{D}^{\text{test}}|} \sum_{(\mathbf{a},u) \in \mathcal{D}^{\text{test}}} \frac{\|\mathcal{G}_\theta(\mathbf{a}) - u\|_{1,G}}{\max_G u - \min_G u + \varepsilon}$$

该指标通过输出值域归一化消除了不同物理量级的影响，适合跨物理场景的公平比较。

## 实验关键数据

### 实验一：参数外推场景（Out-of-Sample Parameter Values）

在 Burgers 方程、Gray-Scott 反应扩散和 Navier-Stokes 不可压流上，预训练后微调 vs 从零训练的对比：

| 模型 | MSE | NMAE (%) | 平均 epoch 时间(s) | 参数量 |
|------|-----|----------|-------------------|--------|
| MambaFNO (预训练) | 1.009×10⁻⁷ | 0.0120 | 21.91 | ~10⁷ |
| MambaFNO (从零) | 1.193×10⁻⁷ | 0.0213 | 40.14 | ~10⁷ |
| Perceiver (预训练) | 1.425×10⁻⁷ | 0.0169 | 3.21 | ~10⁸ |
| Perceiver (从零) | 1.981×10⁻⁷ | 0.0219 | 204.73 | ~10⁸ |
| FNO (从零) | 1.774×10⁻⁷ | 0.0204 | 7.44 | ~10⁶ |
| Swin-v2 (预训练+从零) | 4.391×10⁻⁸ | 0.0092 | 101.3 | ~10⁹ |
| CoDA-NO (预训练) | 2.881×10⁻⁷ | 0.0343 | 62.91 | ~10⁸ |
| CoDA-NO (从零) | 4.912×10⁻⁷ | 0.0712 | 63.29 | ~10⁸ |

**关键发现：** 预训练方法在所有架构上均优于从零训练；Perceiver 预训练的微调加速最为显著（从 205s 降至 3.2s/epoch，约 **64 倍加速**）；MambaFNO 在预训练后 NMAE 降低约 44%。

### 实验二：输入扩展 + 跨物理场景迁移

从对流/Burgers 方程迁移到反应扩散（PDEBench 数据集），以及热方程扩展为对流-扩散方程：

| 模型 | MSE | NMAE (%) | 平均 epoch 时间(s) |
|------|-----|----------|-------------------|
| MambaFNO (预训练) | 3.91×10⁻⁶ | 0.0041 | 131.2 |
| MambaFNO (从零) | 4.291×10⁻⁶ | 0.0054 | 261.1 |
| Perceiver (预训练) | 4.107×10⁻⁶ | 0.0051 | 20.4 |
| Perceiver (从零) | 6.315×10⁻⁶ | 0.0074 | 804.0 |
| FNO (从零) | 7.286×10⁻⁶ | 0.0121 | 41.3 |
| CoDA-NO (预训练) | 1.043×10⁻⁵ | 0.013 | 185.1 |
| CoDA-NO (从零) | 1.239×10⁻⁵ | 0.018 | 181.9 |

**关键发现：** 跨物理场景更具挑战性，但预训练仍持续有效；Perceiver 预训练微调加速约 **39 倍**（804s→20.4s），同时 NMAE 降低 31%；MambaFNO 在精度和速度上均有提升，且相比从头训练 epoch 快约 2 倍。

## 亮点与洞察

1. **Adapter 思路的简洁性：** 将 lifting/projection 类比为 LLM 中的 adapter，核积分层类比为预训练 backbone，这一类比自然且有效，设计简洁、实现成本低
2. **Mamba 模块作为"隐空间预条件化器"：** 通过因果卷积将 embedding 与通用动力学模式对齐，降低频谱变异性，这个设计思路值得在其他 PDE 迁移场景中借鉴
3. **Perceiver 的极端加速效果：** 微调阶段 39-64 倍的加速说明潜变量表示能有效压缩信息，使得仅更新少量 adapter 参数即可适配新问题
4. **纯数据驱动：** 刻意回避物理信息（PINN），专注评估神经算子从数据中学习通用动力学的能力，实验设计清晰

## 局限性

1. **问题维度限制：** 所有实验均为相同空间维度的 PDE，未验证跨维度迁移的可行性（如 1D→2D 或 2D→3D）
2. **数据集规模和多样性：** 实验涉及的物理场景仍较有限（对流、Burgers、反应扩散、NS），未测试更复杂的多物理耦合问题
3. **缺乏与最新基础模型的比较：** 未与 POSEIDON 等近期工作做直接对比，难以判断在真实大规模场景中的竞争力
4. **网格依赖性未深入探讨：** 虽然提到"mesh-agnostic"，但实际实验中不同 PDE 是否使用了统一分辨率未说明
5. **Swin-v2 的参数量为 ~10⁹：** 与其他方法不在同一量级，对比的公平性存疑

## 相关工作与启发

- **POSEIDON** [Herde et al., 2024]：层级视觉 Transformer + shifted windows，在 Euler/NS 间迁移，是本文的重要对比对象
- **CoDA-NO** [Rahman et al., 2024]：codomain attention 在特征维度做相似度计算，本文将其作为基线之一
- **DeepONet 迁移学习** [Goswami et al., 2022]：在条件偏移下的算子迁移，强调了 NO 迁移的理论可行性
- **PDEBench** [Takamoto et al., 2022]：提供了标准化的 PDE 基准数据集，本文的跨物理实验基于此

**启发：** adapter-based 多任务预训练策略可推广至更多科学计算场景。后续可考虑：(a) 基于 Lie 对称性的数据增强；(b) 引入物理先验作为正则化而非硬约束；(c) 在更大规模异构数据集上训练真正的基础模型。

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐ |
| 综合评分 | ⭐⭐⭐ |

**总评：** 工作思路清晰、实验设计合理，adapter-based 预训练微调范式在 PDE 迁移学习中验证有效。但整体贡献更偏"工程验证"而非"方法创新"——核心思想（adapter 解耦 + 共享 backbone）是 NLP 领域已成熟的范式在科学计算中的直接应用，新颖性有限。实验规模偏小，距离论文标题中"Universal"的愿景还有一定距离。

<!-- RELATED:START -->

## 相关论文

- [DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)
- [Hamiltonian Neural PDE Solvers through Functional Approximation](hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [Universal Neural Optimal Transport](../../ICML2025/scientific_computing/universal_neural_optimal_transport.md)
- [Accurate Differential Operators for Hybrid Neural Fields](../../CVPR2025/scientific_computing/accurate_differential_operators_for_hybrid_neural_fields.md)
- [Maximal Update Parametrization and Zero-Shot Hyperparameter Transfer for Fourier Neural Operators](../../ICML2025/scientific_computing/maximal_update_parametrization_and_zero-shot_hyperparameter_transfer_for_fourier.md)

<!-- RELATED:END -->
