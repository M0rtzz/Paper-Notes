---
title: >-
  [论文解读] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis
description: >-
  [ICCV 2025][人体-物体交互] 提出 SyncDiff，一个统一的多体人体-物体交互运动合成框架，通过对齐分数（alignment scores）和显式同步策略实现多体运动的精确同步，并引入频域分解来建模高频交互语义。
tags:
  - "ICCV 2025"
  - "人体-物体交互"
  - "运动合成"
  - "扩散模型"
  - "多体同步"
  - "频率分解"
---

# SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis

**会议**: ICCV 2025  
**arXiv**: [2412.20104](https://arxiv.org/abs/2412.20104)  
**代码**: [https://syncdiff.github.io/](https://syncdiff.github.io/)  
**领域**: 其他  
**关键词**: 人体-物体交互, 运动合成, 扩散模型, 多体同步, 频率分解

## 一句话总结

提出 SyncDiff，一个统一的多体人体-物体交互运动合成框架，通过对齐分数（alignment scores）和显式同步策略实现多体运动的精确同步，并引入频域分解来建模高频交互语义。

## 研究背景与动机

现有的人体-物体交互（HOI）运动合成方法通常局限于特定的交互配置（如单手-单物体、双手-单物体），缺乏对任意数量人体、手部和物体组成的通用多体场景的处理能力。多体场景下存在两大核心挑战：

**同步性需求高**：不同身体的运动之间存在高度相关性和相互影响，简单地将所有运动拼接成高维表示并用单一扩散模型建模，只能隐式地描述个体运动间的关联，难以保证多体间的精确对齐（如接触一致性、避免穿透）。

**高频交互被淹没**：物体之间的高频、小振幅交互（如刷洗时刷子与茶壶之间的周期性摩擦）往往被大尺度的低频运动（如物体的整体移动和接触）所掩盖，导致生成的动作缺乏语义上关键的细节。

为解决这些问题，作者提出需要一套专门的对齐分数来促进运动同步，以及频域分解技术来显式建模高频运动成分。

## 方法详解

### 整体框架

SyncDiff 在图模型上定义扩散过程，图的节点表示个体运动（人/手/物体），边表示成对身体间的相对运动。模型操作在包含所有个体运动和相对运动的高阶表示上，是首个能处理任意数量身体的统一多体 HOI 合成框架。框架包含三个核心设计：频率分解、对齐分数（训练时）、显式同步（推理时）。

### 关键设计

1. **运动表示**：对于骨骼体（人/手），用 3D 关节位置表示其运动 $x_h \in \mathbb{R}^{N \times 3D}$；对于刚体（物体），用平移 + 四元数表示 $x_o \in \mathbb{R}^{N \times 7}$。相对运动 $x_{b_2 \to b_1}$ 通过坐标变换计算 $b_2$ 在 $b_1$ 坐标系中的运动。所有个体和相对运动拼接为高阶表示 $x \in \mathbb{R}^{N \times D_{sum}}$。

2. **频率分解**：通过 FFT 将运动分解为低频（$x_{dc}$）和高频（$x_{ac}$）成分，设截止频率 $L=16$，丢弃过高频率噪声。低频成分在时域监督，高频成分在频域表示为 $x_F$（傅里叶系数），分别送入 Transformer backbone 去噪。最终重组：$\hat{x} = \hat{x}_{dc} + \hat{x}_{ac}$。这样做的设计动机是防止高频语义成分被低频大运动淹没。

3. **对齐分数与对齐损失**：核心创新之一。类比数据样本分数引导去噪重建，对齐分数用于促进图模型中每条边上个体运动与相对运动的一致性。具体地，对齐损失定义为：

$$\mathcal{L}_{align} = \sum_{j_1 \neq j_2} \|\hat{x}_{o_{j_2} \to o_{j_1}} - \text{rel}(\hat{x}_{o_{j_1}}, \hat{x}_{o_{j_2}})\|_2^2 + \sum_{i,j} \|\hat{x}_{h_i \to o_j} - \text{rel}(\hat{x}_{o_j}, \hat{x}_{h_i})\|_2^2$$

即要求模型预测的相对运动与从个体运动计算得到的相对运动一致，数学上等价于对对齐似然的负对数进行优化。

4. **显式同步（推理时）**：每隔 $s=50$ 步执行一次同步操作（总步数 $T=1000$），通过最大化数据样本似然与对齐似然的联合分布，推导出解析形式的同步更新公式。个体运动的同步结果是扩散预测均值与相对运动推导结果的加权平均，权重由超参 $\bar{\lambda}$ 和噪声尺度 $\sigma$ 控制。作者证明该公式等价于新高斯分布上的最大似然采样。

### 损失函数 / 训练策略

总损失为四部分的加权和：

$$\mathcal{L} = \lambda_{dc}\mathcal{L}_{dc} + \lambda_{ac}\mathcal{L}_{ac} + \lambda_{align}\mathcal{L}_{align} + \lambda_{norm}\mathcal{L}_{norm}$$

- $\mathcal{L}_{dc}$, $\mathcal{L}_{ac}$：分别监督低频和高频成分的重建
- $\mathcal{L}_{align}$：对齐损失，促进多体同步
- $\mathcal{L}_{norm}$：约束刚体旋转四元数的范数接近 1

模型架构采用 latent diffusion 范式，用 CLIP 编码动作/物体标签、BPS 编码物体几何。

## 实验关键数据

### 主实验

在 5 个数据集上评估（TACO, CORE4D, GRAB, OAKINK2, BEHAVE），覆盖手-物体和人-物体交互场景。

| 方法 | CSIoU(%) ↑ | IV(cm³) ↓ | FID ↓ | RA(%) ↑ |
|------|-----------|----------|-------|---------|
| MACS | 56.81 | 13.18 | 10.56 | 58.40 |
| DiffH2O | 62.29 | 10.25 | 4.34 | 61.40 |
| **SyncDiff** | **73.00** | **6.64** | **2.70** | **73.28** |

*TACO Test1 结果。SyncDiff 在接触质量和语义准确率上均大幅领先。*

| 方法 | CRR(%) ↑ | FID ↓ | RA(%) ↑ |
|------|---------|-------|---------|
| OMOMO | 5.31 | 13.22 | 68.02 |
| CG-HOI | 5.74 | 12.16 | 70.05 |
| **SyncDiff** | **6.15** | **6.45** | **92.89** |

*CORE4D Test1 结果。动作语义识别准确率超越 SOTA 方法约 23 个百分点。*

### 消融实验

| 消融设置 | CSIoU(%) | FID | RA(%) |
|---------|---------|-----|-------|
| SyncDiff (完整) | 73.00 | 2.70 | 73.28 |
| w/o all | 62.96 | 10.63 | 57.39 |
| w/o 频率分解 | 68.86 | 6.44 | 56.60 |
| w/o $\mathcal{L}_{align}$ + 显式同步 | 63.74 | 4.13 | 64.47 |
| w/o $\mathcal{L}_{align}$ | 70.39 | 2.90 | 67.82 |
| w/o 显式同步 | 65.51 | 3.39 | 67.27 |

*TACO Test1 消融结果。三个核心组件的移除均导致不同程度的性能下降，显式同步的影响最大。*

### 关键发现

- 频率分解对需要周期性相对运动的场景（如刷洗动作）尤为关键，移除后物体倾向于相对静止
- 单独引入频率分解但不加同步机制反而可能导致接触指标下降，说明高频建模对同步的需求更高
- 用户研究（150 人，10 个数据集划分）显示 SyncDiff 在多体场景下的优势随身体数量增加而愈发显著

## 亮点与洞察

- 首次将多体 HOI 合成形式化为图模型上的运动同步问题，并推导出对齐分数的理论基础
- 显式同步策略有严格的数学推导支撑（等价于最大似然采样），而非启发式设计
- 频域分解是一个优雅且有效的方案，解决了高频语义被低频运动掩盖的普遍问题

## 局限与展望

- 显式同步步骤引入额外计算开销（每 50 步执行一次）
- 当前仅在 mocap 数据集上验证，未涉及真实场景下的在线部署
- 相对运动表示仅限于刚体作为参考坐标系，骨骼体之间的相对表示被省略

## 相关工作与启发

- 与 CG-HOI 等方法相比，SyncDiff 不需要预定义的接触引导
- 频率分解策略受 GID 启发，将其从场景生成扩展到多体运动合成
- 图模型 + 扩散模型的组合框架对其他需要多体协调的生成任务（如多人舞蹈、机器人协作）有借鉴意义

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer](layertracer_cognitive-aligned_layered_svg_synthesis_via_diffusion_transformer.md)
- [\[ICCV 2025\] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks](a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)
- [\[ACL 2025\] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction](../../ACL2025/others/visual_cues_enhance_predictive_turn-taking_for_two-party_human_interaction.md)
- [\[ICCV 2025\] Jigsaw++: Imagining Complete Shape Priors for Object Reassembly](jigsaw_imagining_complete_shape_priors_for_object_reassembly.md)
- [\[ICLR 2026\] MOSIV: Multi-Object System Identification from Videos](../../ICLR2026/others/mosiv_multi-object_system_identification_from_videos.md)

</div>

<!-- RELATED:END -->
