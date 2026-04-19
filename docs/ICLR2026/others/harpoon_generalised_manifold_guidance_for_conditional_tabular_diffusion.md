---
title: >-
  [论文解读] Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion
description: >-
  [ICLR 2026][表格数据] 将流形理论从图像扩展到表格数据扩散模型，证明任意可微推理时损失的梯度都位于数据流形切线空间中（不限于平方误差损失），据此提出Harpoon方法在推理时沿流形引导无条件样本满足多样化表格约束。
tags:
  - ICLR 2026
  - 表格数据
  - 流形引导
  - 条件生成
  - 推理时引导
  - 不等式约束
---

# Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion

**会议**: ICLR 2026  
**arXiv**: [2602.07875](https://arxiv.org/abs/2602.07875)  
**代码**: [GitHub](https://github.com/adis98/Harpoon)  
**领域**: 扩散模型/表格数据  
**关键词**: 表格数据, 流形引导, 条件生成, 推理时引导, 不等式约束

## 一句话总结
将流形理论从图像扩展到表格数据扩散模型，证明任意可微推理时损失的梯度都位于数据流形切线空间中（不限于平方误差损失），据此提出Harpoon方法在推理时沿流形引导无条件样本满足多样化表格约束。

## 研究背景与动机

**领域现状**：表格扩散模型可以生成高质量表格数据，但条件生成（缺失值填补、不等式约束等）是核心需求。现有条件方法分为训练时（难以泛化到新约束）和推理时（仅限填补任务）两类。

**现有痛点**：(1) 训练时方法（条件输入/分类器引导/无分类器引导）无法泛化到训练时未见的约束；(2) 推理时方法仅支持填补不支持不等式约束；(3) 图像扩散的流形理论假设连续特征+平坦几何，不适用于混合类型表格数据。

**核心矛盾**：需要一次训练、推理时适应任意约束的方法，但现有流形引导理论只对平方误差损失+平坦流形有保证。

**切入角度**：证明两个更强的理论结果：(1) Theorem 3.1: 去噪映射 $Q_t$ 在 $\bar{\alpha}_t \to 1$ 时收敛到流形正交投影（不需平坦假设）；(2) Theorem 3.2: 任意可微损失的梯度都在切线空间中（不限于平方误差）。

**核心 idea**：证明推理时任意可微目标的梯度与流形对齐，据此交替做无条件去噪和切向校正来满足多样化约束。

## 方法详解

### 整体框架
一次训练无条件扩散模型→推理时交替执行：(1) 无条件去噪一步；(2) 用推理时损失 $\mathcal{L}_{\text{inf}}$ 的梯度做切向校正。支持填补、不等式约束等多种条件。

### 关键设计

1. **Theorem 3.1 (正交投影)**:

    - 内容：MSE训练的去噪器在 $\bar{\alpha}_t \to 1$ 时等价于到流形 $\mathcal{M}_0$ 的正交投影
    - 贡献：推广了Chung等人的结果——不需要平坦流形假设，弯曲流形也成立
    - 实际意义："dirty estimate" $\hat{x}_0 = Q_t(x_t)$ 落在流形上

2. **Theorem 3.2 (切线空间梯度)**:

    - 内容：对任意可微推理时损失 $\mathcal{L}_{\text{inf}}$，其梯度 $\nabla_{x_t}\mathcal{L}_{\text{inf}}(\hat{x}_0, c) \in T_{\hat{x}_0}\mathcal{M}_0$
    - 贡献：从"仅平方误差"推广到任意可微损失（交叉熵、L1、ReLU不等式等）
    - 实际意义：推理时用任何合理损失做梯度校正都不会把样本推离流形

3. **Harpoon算法**:

    - 功能：每步先无条件去噪再切向校正
    - 核心思路：$x_{t-1} = x_{t-1}' - \eta \cdot \nabla_{x_t}\mathcal{L}_{\text{inf}}(\hat{x}_0, c)$
    - 支持的约束：填补（部分观测）、范围约束（Age>=10）、分类约束（Gender=Male）、合取/析取

### 损失函数 / 训练策略
- 训练：标准MSE去噪损失（一次训练）
- 推理时损失可选：MAE（默认，稀疏诱导适合表格）、MSE、交叉熵、ReLU不等式
- 引导强度 $\eta$ 控制约束满足程度

## 实验关键数据

### 主实验 - 填补 (MAR, 50%缺失)
| 方法 | Adult | Bean | California | Magic | 平均 |
|------|-------|------|-----------|-------|------|
| GAIN | 1.86 | 1.41 | 15.06 | 1.27 | 高 |
| DiffPuter (SOTA) | 中 | 中 | 中 | 中 | 中 |
| Harpoon | **低** | **低** | **低** | **低** | **SOTA** |

### 不等式约束
| 约束类型 | 违反率↓ | α-score↑ | 效用↑ |
|---------|---------|----------|-------|
| 范围约束 | **最低** | 高 | 高 |
| 分类约束 | **最低** | 高 | 高 |
| 合取(and) | **最低** | 高 | 高 |
| 析取(or) | **最低** | 高 | 高 |

### 关键发现
- 实验验证推理时梯度确实与"dirty estimate"近似正交（~90°），即使在较大时步也成立
- 不同推理时损失（MSE/MAE/CE）在同一训练目标下行为一致→实证验证Theorem 3.2
- MAE损失对表格数据效果最好（稀疏诱导特性适合离散特征）
- 一次训练，多种推理时约束→比训练时条件化方法灵活得多

## 亮点与洞察
- **理论贡献是核心**：两个定理显著推广了图像扩散的流形引导理论——弯曲流形+任意可微损失。这个理论不仅适用于表格，对其他模态也有指导意义。
- **"一次训练，任意约束"**：训练无条件模型→推理时加任意约束，这是条件生成的理想范式。Harpoon证明了这在表格数据中可行且有理论保证。
- **MAE优于MSE的发现**：表格数据的离散特征更适合稀疏诱导的L1损失，这个领域特异的insight有实用价值。

## 局限与展望
- 正交投影保证仅在 $\bar{\alpha}_t \to 1$ 时严格成立，实际大时步可能有偏差
- 表格数据的连续嵌入（如one-hot）是近似，离散-连续gap仍存在
- 引导强度 $\eta$ 需要调参
- 仅在UCI数据集上验证，更大规模表格数据的scalability未知

## 相关工作与启发
- **vs DiffPuter**: DiffPuter是训练时条件化，Harpoon是推理时引导——前者更专一后者更灵活
- **vs Chung等人的图像流形引导**: Harpoon推广了理论（弯曲流形+任意损失），并首次应用到表格
- **vs CTGAN/TabDDPM**: 这些不支持推理时条件化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 流形理论的推广是重要理论贡献，表格适配自然
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多任务(填补+不等式)、理论验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，直觉解释到位
- 价值: ⭐⭐⭐⭐ 理论影响超出表格领域，对扩散模型引导有通用意义

<!-- RELATED:START -->

## 相关论文

- [Compositional Diffusion with Guided Search for Long-Horizon Planning](compositional_diffusion_long_horizon_planning.md)
- [Tab-PET: Graph-Based Positional Encodings for Tabular Transformers](../../AAAI2026/others/tab-pet_graph-based_positional_encodings_for_tabular_transformers.md)
- [Local Guidance for Configuration-Based Multi-Agent Pathfinding](../../AAAI2026/others/local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [Exploring Guided Sampling of Conditional GANs](../../ECCV2024/others/exploring_guided_sampling_of_conditional_gans.md)
- [Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](../../ACL2025/others/neodiff_unified_text_diffusion.md)

<!-- RELATED:END -->
