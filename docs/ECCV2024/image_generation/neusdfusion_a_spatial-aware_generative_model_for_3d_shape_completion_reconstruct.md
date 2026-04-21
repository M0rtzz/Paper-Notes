---
title: >-
  [论文解读] NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation
description: >-
  [ECCV 2024][图像生成] 提出 NeuSDFusion，一个基于混合三平面 SDF 表示（NeuSDF）和空间感知 Transformer 自编码器的 3D 形状生成框架，通过保持三平面间的空间对应关系，在无条件生成、多模态形状补全、单视图重建和文本到 3D 生成等任务上达到 SOTA 性能。
tags:
  - ECCV 2024
  - 图像生成
---

# NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation

**会议**: ECCV 2024  
**arXiv**: [2403.18241](https://arxiv.org/abs/2403.18241)  
**领域**: 图像生成

## 一句话总结

提出 NeuSDFusion，一个基于混合三平面 SDF 表示（NeuSDF）和空间感知 Transformer 自编码器的 3D 形状生成框架，通过保持三平面间的空间对应关系，在无条件生成、多模态形状补全、单视图重建和文本到 3D 生成等任务上达到 SOTA 性能。

## 研究背景与动机

- **三平面表示的兴起**：三平面（tri-plane）用三个正交 2D 平面紧凑表示 3D 信息，相比体素/点云更高效
- **现有方法忽略空间一致性**：
    - NFD、DiffusionSDF 等将三个平面在通道维拼接当作 RGB 图像处理，忽略不同平面同一坐标之间没有显式关系
    - Rodin 的 3D 感知卷积使用池化聚合特征，丢失上下文信息，无法生成平滑表面
- **3D 表示的内存限制**：T-SDF 用 3D 体素存储距离场，内存消耗大，难以捕获精细形状特征
- **NFD 的表示局限**：三平面从离散占用网格学习，表示能力受网格分辨率限制
- **核心目标**：设计一个高效的 3D 表示和生成框架，能够在保持空间一致性的同时生成高保真、多样化的 3D 形状

## 方法详解

### 整体框架

三阶段流水线：
1. **NeuSDF 拟合**：将每个 3D 物体编码为三平面 + MLP 的混合 SDF 表示
2. **空间感知自编码器**：将原始三平面压缩为紧凑的潜在三平面表示，保持空间对应关系
3. **潜在扩散模型**：在压缩的潜在空间中进行条件/无条件生成

### 关键设计

**NeuSDF 表示**：
- 将 3D 形状用三个轴对齐平面（XY、YZ、XZ）表示
- 查询任意 3D 点 $p$ 时，投影到三个平面通过双线性插值获取特征 $F_{xy}, F_{xz}, F_{yz}$，逐元素求和后经 MLP 解码为 SDF 值
- 优化方法：对每个物体联合优化三平面和 MLP 参数
- 采样策略：表面点（$\Omega_0$）+ 空间填充点（$\Omega$），并采样法线向量作为额外监督

**空间感知 Transformer 自编码器**：
- **问题**：现有方法（roll-out、通道拼接）使用 CNN 处理三平面，导致跨平面空间关系丢失
    - Roll-out 在三平面边界处卷积跨越两个平面，产生边界缺陷
    - 通道拼接忽略不同平面同一位置间缺乏关系的事实
- **解决方案**：U 形 Transformer 编码器-解码器结构
    - 每个阶段先用分组卷积独立下采样各平面（参数更高效）
    - 再将三平面展平为 1D token 序列 $x \in \mathbb{R}^{C \times 3HW}$ 输入 Transformer
    - Transformer 注意力学习三个平面间的全局关系
- **空间感知位置嵌入（SAPE）**：为三个平面各创建独立的可学习位置嵌入，引入归纳偏置使每个 token 能区分其他 token 属于同一平面还是不同平面
- **线性注意力**：使用线性注意力机制将复杂度从 $O(n^2)$ 降为 $O(n)$，使得可以直接处理高分辨率三平面（第一阶段 $3 \times 64 \times 64 = 12288$ tokens）

**条件扩散模型**：
- 在潜在三平面空间训练 U-Net 扩散模型
- 条件注入：通过 cross-attention 层将条件编码器输出（图像/文本/点云特征）注入 U-Net 的每个Block
- Classifier-free guidance：10% 概率将条件替换为零掩码

### 损失函数

**NeuSDF 拟合**：$\mathcal{L}_{geo} = \mathcal{L}_{sdf} + \mathcal{L}_{normal} + \mathcal{L}_{eikonal}$
- SDF 损失：表面点距离为零 + 空间点与真值 SDF 值一致
- 法线损失：梯度方向与真值法线一致
- Eikonal 损失：SDF 梯度模长约束为 1（保持 SDF 的物理特性）

**自编码器**：$\mathcal{L}_{ae} = \mathcal{L}_{rec} + \mathcal{L}_{KL} + \mathcal{L}_{geo}$

**潜在扩散**：$\mathcal{L}_{ldm} = \|\Psi(z_t, \gamma(t)) - z_0\|^2$

## 实验关键数据

### 主实验

**无条件生成（ShapeNet，1-NNA↓）**：

| 方法 | 表示 | Airplane CD/EMD | Chair CD/EMD | Car CD/EMD |
|------|------|----------------|-------------|-----------|
| IM-GAN | Occupancy | 79.48/82.94 | 58.59/69.05 | 95.69/94.79 |
| LION | Point Cloud | 67.41/61.23 | 53.70/52.34 | 53.41/51.14 |
| 3DQD | TSDF | 56.29/54.78 | 55.61/52.94 | 55.75/52.80 |
| **NeuSDFusion** | **NeuSDF** | **52.33/52.47** | **51.95/52.60** | **53.06/51.11** |

**多模态形状补全（×10²）**：

| 方法 | Bottom Half MMD↓ | Bottom Half AMD↓ | Octant MMD↓ | Octant AMD↓ |
|------|-----------------|-----------------|------------|------------|
| AutoSDF | 3.51 | 8.20 | 5.72 | 12.79 |
| 3DQD | 2.93 | 6.30 | 4.69 | 10.93 |
| **NeuSDFusion** | **2.29** | **5.90** | **3.03** | **9.59** |

### 消融实验

**单视图重建（Pix3D）**：

| 方法 | CD↓ | F-Score↑ |
|------|-----|---------|
| Pix2Vox | 3.00 | 0.39 |
| AutoSDF | 2.28 | 0.42 |
| SDFusion | 1.85 | 0.43 |
| **NeuSDFusion** | **0.92** | **0.61** |

**文本引导生成**：

| 方法 | PMMD↓ | CLIP-S↑ | FPD↓ | TMD↑ |
|------|-------|---------|------|------|
| 3DQD | 1.49 | 32.11 | 59.00 | 2.80 |
| **NeuSDFusion** | **1.49** | **32.52** | **55.01** | **3.20** |

### 关键发现

1. **表示能力**：NeuSDF 在所有类别的无条件生成上均优于 TSDF（3DQD）、点云（LION）等表示，证明混合三平面 SDF 表示的优越性
2. **空间一致性**：与 roll-out 和通道拼接方法对比，空间感知 Transformer 自编码器生成的物体表面更完整光滑，无边界伪影
3. **单视图重建**：CD 从 SDFusion 的 1.85 降至 0.92（**50% 提升**），F-Score 从 0.43 提升至 0.61（**42% 提升**），得益于 NeuSDF 表示能捕获更精细的形状细节
4. **多模态补全**：在质量指标（MMD/AMD）上领先，同时保持有竞争力的多样性（TMD），说明潜在空间的结构化有利于条件生成

## 亮点与洞察

- **混合表示的优雅设计**：三平面 + MLP 的 NeuSDF 表示既享有三平面的高效性，又通过连续 SDF 学习避免了离散网格的分辨率限制
- **空间感知位置嵌入（SAPE）**：简单而有效的归纳偏置，使 Transformer 能区分三个平面间的空间关系
- **线性注意力的关键作用**：使得直接在 12288 长度的 token 序列上做全局注意力成为可能，突破了 LRM 等方法 32 分辨率的限制
- **通用的条件生成框架**：同一个管线支持无条件生成、形状补全、单视图重建和文本到 3D，展示了表示和架构的通用性

## 局限性

- 第一阶段的 NeuSDF 拟合需要对每个物体独立优化，数据集规模扩大时耗时较长
- Marching Cubes 提取形状的分辨率仍然受限
- 线性注意力虽降低了复杂度，但在特别复杂的几何细节上可能不如完整的自注意力

## 评分

⭐⭐⭐⭐ (4/5) — 三阶段管线设计清晰完整，NeuSDF 表示和空间感知 Transformer 均有实质创新，在四个任务上的全面评估令人信服，对 3D 生成领域有重要推进

<!-- RELATED:START -->

## 相关论文

- [ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)

<!-- RELATED:END -->
