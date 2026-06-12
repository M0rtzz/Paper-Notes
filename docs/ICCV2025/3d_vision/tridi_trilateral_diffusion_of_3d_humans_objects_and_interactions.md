---
title: >-
  [论文解读] TriDi: Trilateral Diffusion of 3D Humans, Objects, and Interactions
description: >-
  [ICCV 2025][3D视觉][3D人体-物体交互] 提出 TriDi，首个建模人体(H)、物体(O)和交互(I)三变量联合分布的统一扩散模型，一个网络覆盖 7 种条件生成模式，超越各专用单向基线。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D人体-物体交互"
  - "联合概率建模"
  - "三路扩散"
  - "接触图"
  - "多模态生成"
---

# TriDi: Trilateral Diffusion of 3D Humans, Objects, and Interactions

**会议**: ICCV 2025  
**arXiv**: [2412.06334](https://arxiv.org/abs/2412.06334)  
**代码**: [https://virtualhumans.mpi-inf.mpg.de/tridi/](https://virtualhumans.mpi-inf.mpg.de/tridi/)  
**领域**: 3D Vision / Human-Object Interaction  
**关键词**: 3D人体-物体交互, 联合概率建模, 三路扩散, 接触图, 多模态生成

## 一句话总结

提出 TriDi，首个建模人体(H)、物体(O)和交互(I)三变量联合分布的统一扩散模型，一个网络覆盖 7 种条件生成模式，超越各专用单向基线。

## 研究背景与动机

**领域现状**：3D 人体-物体交互（HOI）建模对于 AR/VR、虚拟人生成等应用至关重要。现有方法以**单向条件**方式运作：有的从物体恢复人体姿态 P(H|O)，有的从人体恢复物体位姿 P(O|H)，每种模式需要独立的模型架构、训练流程和设计方案。

**核心矛盾**：针对每种条件组合训练专用模型(1) 不可扩展，(2) 忽略三个模态间的相互依赖关系，(3) 无法做到无条件联合生成。给定人体和物体，存在多种可能的交互方式（坐、提、推等），全面的模型应同时捕捉这些模态之间的交互关系。

**本文切入角度**：将 HOI 建模从"单向条件分布"范式转向"三变量联合分布"范式。受 UniDiffuser（双模态扩散）启发，将其扩展到三模态，一个紧凑架构建模 P(H,O,I)，自然产生 $2^3 - 1 = 7$ 种操作模式。

**核心 idea**：(1) 基于 Transformer 的三路扩散过程，为每个模态分配独立时间步，通过 token 级自注意力发现跨模态细粒度关系；(2) 将文本描述和身体接触图嵌入到共享潜空间，兼顾用户可控性与空间表达力。

## 方法详解

### 整体框架

TriDi 将三模态 (H, O, I) 分别参数化：
- **人体 H** = (姿态 θ ∈ R^{51×3}, 体型 β ∈ R^{10}, 全局位姿 g_H ∈ R^9)，基于 SMPL+H 模型
- **物体 O** = (全局位姿 g_O ∈ R^9)，几何由 PointNeXt 特征 + 类别 one-hot 编码
- **交互 I** = (潜变量 z_I ∈ R^{128})，一个联合嵌入接触图和文本描述的紧凑编码

模型接收带噪的三模态 token 和三个独立时间步 (t^H, t^O, t^I)，预测去噪后的原始样本。

### 关键设计

1. **Contact-Text 交互表示**：训练两个编码器（接触图编码器 E_ϕ 和文本编码器 E_T）映射到共享 128 维潜空间，以及一个接触图解码器 D_ϕ。损失函数包含：接触图自编码 BCE、文本到接触图编码 BCE、文本-接触图潜空间对齐 L2。这样用户可以用文本或接触图引导生成。

2. **三路扩散公式化**：扩展 UniDiffuser 至三模态，每个模态有独立的噪声时间步 (t^H, t^O, t^I)。训练目标为：

    $\min_\psi \mathbb{E}_p \mathbb{E}_t \mathbb{E}_q \| \text{TriDi}_\psi(H^{t^H}, O^{t^O}, I^{t^I}; t^H, t^O, t^I; C_O) - (H^0, O^0, I^0) \|_2$

   通过调节时间步可切换任意模式：t=0 表示条件，t=T 表示生成目标。

3. **基于重建的引导机制**：在去噪过程中，利用预测的接触图计算人体-物体接触距离监督函数 F，通过梯度引导修正预测：

    $F(\hat{H}, \hat{O}, \hat{I}) = \sum_j |\hat{\phi}_{I_j} \hat{d}_j|$

   其中 $\hat{d}_j$ 为人体顶点到物体最近点距离，该引导在最后 200 步施加，权重 λ=2.0。

### 损失函数 / 训练策略

总损失包含 6 项：参数空间损失（人体 L1、物体 L1、交互 L2）+ 顶点空间损失（人体顶点 L2、物体顶点 L2、人-物距离 L2），加权系数分别为 (2, 1, 1, 6, 2, 4)。

训练细节：15M 参数，batch=1024，lr=1e-4 余弦衰减，AdamW 优化器，RTX4090 上训练约 20 小时。通过 ZY 平面镜像增强解决数据中的右手偏置问题，有效增加训练多样性。

## 实验关键数据

### 主实验

在 BEHAVE 和 GRAB 两个数据集上评估分布质量和几何一致性：

| 方法 | 1-NNA (→50) | COV↑ | MMD↓ | MPJPE↓ | MPJPE-PA↓ |
|------|------------|------|------|--------|-----------|
| GNet (BEHAVE, H,I\|O) | 80.01 | 40.71 | 1.789 | 35.6 | 14.6 |
| ObjPOP (BEHAVE, O,I\|H) | 81.36 | 35.02 | 0.329 | - | - |
| **TriDi (H,I\|O)** | **67.89** | **47.81** | **1.352** | **20.8** | **12.3** |
| **TriDi (O,I\|H)** | **63.72** | **51.71** | **0.166** | - | - |

TriDi 在 1-NNA 接近理想值 50、COV 提升最高 47% 的同时，几何一致性也优于专用基线。

### 消融实验

| 配置 | 1-NNA | COV | MPJPE | Acc_cont |
|------|-------|-----|-------|----------|
| 无增强 | 较差 | 较低 | - | - |
| 无 I 模态 | - | - | 较差 | 较低 |
| 无引导 | - | - | 几何一致较差 | 较低 |
| **完整模型** | **最优** | **最优** | **最优** | **最优** |

增强提升分布质量(1-NNA)；引导和交互模态对几何一致性贡献显著；三模态联合比二模态表现更好。

### 关键发现

- 联合训练的 TriDi 优于或持平专门针对单一模式训练的变体（s-TriDi-HI、s-TriDi-OI），证明联合建模有助于改善泛化。
- 用户研究（40 人）中，TriDi 输出被偏好于基线的比例约 89%，与 GT 接近（~52%）。
- TriDi 能泛化到未见几何（如椅子和凳子），并支持从 RGB 图像间接重建交互。

## 亮点与洞察

- **统一性**：一个 15M 参数模型覆盖 7 种操作模式，涵盖所有先前工作的专用场景，并扩展出新用例（如联合生成 H+O+I）。
- **Contact-Text 表示**很巧妙：将精确但不直观的接触图与易用但粗略的文本描述统一到同一潜空间，兼顾可控性和精度。
- **左右对称增强**简单但有效：之前的 HOI 方法都没处理数据的右手偏置问题。

## 局限与展望

- 受限于训练数据偏斜，对高频物体效果好，对功能性差异大的新物体（轮椅、自行车）泛化有限。
- 当前仅处理单帧静态交互，扩展到动态序列是重要方向。
- 单人单物交互，扩展到多人多物是未来趋势。

## 相关工作与启发

- 与 UniDiffuser [Bao 2023] 从双模态扩展到三模态的思路一致，展示了多模态联合扩散的可扩展性。
- CG-HOI [Diller 2024] 需要强文本条件且一次只训练一个数据集，TriDi 更通用。
- 对场景填充、虚拟现实内容创作领域有直接应用价值。

## 评分

- 新颖性：⭐⭐⭐⭐ — 三模态联合扩散 + Contact-Text 统一表示
- 技术深度：⭐⭐⭐⭐ — 严谨的概率建模和引导机制
- 实验充分度：⭐⭐⭐⭐⭐ — 多数据集、消融、用户研究、下游应用
- 实用性：⭐⭐⭐⭐ — AR/VR 场景填充、交互重建

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Event-Driven Storytelling with Multiple Lifelike Humans in a 3D Scene](event-driven_storytelling_with_multiple_lifelike_humans_in_a_3d_scene.md)
- [\[ICCV 2025\] ETCH: Generalizing Body Fitting to Clothed Humans via Equivariant Tightness](etch_generalizing_body_fitting_to_clothed_humans_via_equivariant_tightness.md)
- [\[ICCV 2025\] SceneMI: Motion In-betweening for Modeling Human-Scene Interactions](scenemi_motion_in-betweening_for_modeling_human-scene_interaction.md)
- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [\[CVPR 2025\] UnCommon Objects in 3D](../../CVPR2025/3d_vision/uncommon_objects_in_3d.md)

</div>

<!-- RELATED:END -->
