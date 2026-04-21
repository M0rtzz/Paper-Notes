---
title: >-
  [论文解读] DIVE: Taming DINO for Subject-Driven Video Editing
description: >-
   提出DIVE框架，利用预训练DINOv2模型的语义特征作为隐式对应关系来引导主体驱动的视频编辑，通过DINO特征进行时序运动建模和目标主体身份注册，实现高质量的主体替换同时保持运动一致性。
tags:

---

# DIVE: Taming DINO for Subject-Driven Video Editing

## 基本信息
- **会议**: ICCV 2025
- **arXiv**: 2412.03347
- **代码**: 未公开
- **领域**: 图像/视频生成 / 视频编辑
- **关键词**: 视频编辑, DINO特征, 主体驱动, 运动一致性, LoRA, 扩散模型

## 一句话总结

提出DIVE框架，利用预训练DINOv2模型的语义特征作为隐式对应关系来引导主体驱动的视频编辑，通过DINO特征进行时序运动建模和目标主体身份注册，实现高质量的主体替换同时保持运动一致性。

## 研究背景与动机

- **问题定义**：主体驱动的视频编辑需要在源视频中替换主体（动物、物体等），使编辑后的视频保持原始运动轨迹和时序一致性，同时精确呈现目标主体身份。
- **现有方法局限**：
    - 基于注意力的方法（Tune-A-Video、FateZero等）：通过注意力图/特征注入传递运动信息，但存储的特征可能保留源主体外观信息，导致源和目标外观混合
    - 基于稠密对应关系的方法（光流、深度图、边缘图）：高密度导致视觉内容不连贯和闪烁
    - VideoSwap使用稀疏语义点作为对应关系：能精确对齐运动轨迹，但需要手动标注
- **关键发现**：DINOv2特征在视频帧间具有高语义一致性，同时包含极少外观信息，天然适合作为鲁棒的视频对应关系（Fig.2展示了DINO特征三大优势：运动跟踪、外观稀疏、语义一致但帧内有区分度）

## 方法详解

### 整体框架

DIVE分为三个阶段：（1）时序运动建模 → （2）主体身份注册 → （3）推理。

### Stage 1：时序运动建模

**目标**：从源视频中提取运动信息，作为编辑时的运动引导。

- 对源视频每帧提取DINOv2（ViT-g/14）语义特征$\mathbf{F}_d \in \mathbb{R}^{N \times h \times w \times c}$
- 利用PCA降维+阈值化自动生成前景mask，分离前景主体特征
- 通过可学习MLP集合$\boldsymbol{\psi} = \{\boldsymbol{\psi}_l | l \in \{1,2,3,4\}\}$将DINO特征投影到扩散模型特征空间
- 投影后特征通过逐元素相加注入U-Net编码器各下采样块的中间特征：

$$\mathbf{F}_l^t \leftarrow \mathbf{F}_l^t + \lambda \mathbf{F}_l^s$$

- 优化目标：仅优化MLP参数$\boldsymbol{\psi}$，使扩散模型在前景区域的去噪能力更强：

$$\min_{\boldsymbol{\psi}} \mathbb{E}_{\epsilon,t} \|[\epsilon - \epsilon_{\theta'}(\mathbf{Z}^t, t, \mathbf{c}, \boldsymbol{\psi}(\mathbf{F}_d))] \odot \mathbf{M}\|_2^2$$

- 仅在$t \in [T/2, T]$的较高时间步训练，避免过拟合到主体低级细节

### Stage 2：主体身份注册

**目标**：从参考图像中学习目标主体身份。

- 使用3-5张目标主体参考图像
- 提取参考图像的DINOv2前景特征$\tilde{\mathbf{F}}_d$，通过另一组MLP $\boldsymbol{\phi}$投影到扩散空间
- 与Stable Diffusion中间特征融合，提供准确的身份引导
- 训练LoRA $\Delta\theta$注册目标身份：

$$\min_{\boldsymbol{\phi}, \Delta\theta} \mathbb{E}_{\epsilon,t} \|[\epsilon - \epsilon_{\theta+\Delta\theta}(\mathbf{I}^t, t, \tilde{\mathbf{c}}, \boldsymbol{\phi}(\tilde{\mathbf{F}}_d))] \odot \tilde{\mathbf{M}}\|_2^2$$

- DINO特征提供高级语义匹配（如部件级对应），弥补SD特征仅捕获低级空间信息的不足

### Stage 3：推理

- DDIM反演获取源视频初始噪声
- 在$T$到$T/2$的去噪步骤中注入Stage 1学到的运动引导特征
- 使用Stage 2的预训练LoRA提供身份引导
- 替换文本prompt中的源主体词（如"cat"→"dog"）
- 利用前景mask进行latent blending保持背景不变

## 实验关键数据

### 定量评估（参考图像引导）

| 方法 | Text Align↑ | Image Align↑ | Temporal Consist↑ | Video Quality↑ | 用户偏好↑ |
|------|------------|-------------|-------------------|---------------|----------|
| Slicedit | 28.21 | 64.57 | 91.09 | 0.592 | 6.73% |
| AnyV2V | 28.13 | 78.26 | 90.52 | 0.439 | 13.2% |
| FLATTEN | 28.79 | 69.32 | 92.09 | 0.683 | 8.67% |
| RAVE | 28.26 | 66.25 | 91.71 | 0.646 | 5.80% |
| **DIVE** | **29.43** | **84.27** | **92.33** | **0.775** | **65.6%** |

DIVE在所有指标上均优于竞争方法，用户偏好高达65.6%（远超第二名的13.2%）。

### 消融实验

| 配置 | Text Align↑ | Image Align↑ | Temporal Consist↑ | Video Quality↑ |
|------|------------|-------------|-------------------|---------------|
| w/o DINO, w/ learnable motion | 29.91 | 67.49 | 92.18 | 0.737 |
| w/ DINO λ=0 (仅LoRA) | - | - | - | 运动不一致 |
| w/ DINO λ=0.5 | - | - | - | 运动部分缺失 |
| **w/ DINO λ=1.0 (完整)** | **29.43** | **84.27** | **92.33** | **0.775** |

**运动引导消融**：不使用DINO特征、仅依赖AnimateDiff的时序建模会导致源主体外观残留和图像不对齐。λ值影响运动引导强度。

**身份引导消融**：不含DINO引导的身份注册存在语义错误（如有尾vs无尾的柯基犬），DINO提供部件级语义引导使模型更忠实于参考图像。

### 文本引导编辑结果

| 方法 | Text Align↑ | Temporal Consist↑ | Video Quality↑ | 用户偏好↑ |
|------|------------|-------------------|---------------|----------|
| Slicedit | 31.24 | 92.95 | 0.562 | 5.50% |
| AnyV2V | 31.05 | 93.73 | 0.533 | 19.9% |
| FLATTEN | 31.55 | 95.35 | 0.567 | 11.9% |
| RAVE | 31.57 | 95.12 | 0.588 | 9.90% |
| **DIVE** | **32.29** | **95.89** | **0.614** | **52.8%** |

## 亮点与洞察

1. **DINO特征的双重价值**：同时用于运动建模和身份注册，核心在于DINO特征的三大特性——运动跟踪能力、外观稀疏性、语义一致性
2. **运动与外观解耦**：通过DINO的外观稀疏性自然实现了运动和外观的解耦，避免了注意力方法中的外观泄露问题
3. **无需手动标注**：相比VideoSwap需要手动定义语义点，DINO特征自动提供鲁棒的隐式对应关系
4. **PCA自动分割前景**：利用DINO特征PCA降维+阈值化自动生成前景mask，无需额外分割模型
5. **用户偏好碾压式领先**：65.6%用户偏好率远超其他方法，说明视觉质量差异显著

## 局限性

- 基于Stable Diffusion 1.5，生成质量受基础模型限制
- 每个视频需独立训练Stage 1和Stage 2（50-100 + 800-1000次迭代），不够高效
- 仅在16帧短视频上验证，长视频的表现未知
- 参考图像引导需3-5张目标主体图像，单张图像场景未探讨
- 依赖DDIM反演质量，复杂背景或大运动时可能不稳定

## 相关工作与启发

- VideoSwap：使用稀疏语义点但需手动标注，DIVE用DINO特征自动化了这一过程
- TokenFlow：通过帧间对应关系传播扩散特征，但仍有外观泄露问题
- RAVE：基于深度图的对应关系，稠密导致闪烁
- **启发**：DINO特征在视频域的应用空间巨大，PCA+阈值化的简单前景分割极为实用

## 评分

- **新颖性**: ⭐⭐⭐⭐ （DINO特征用于视频对应关系的动机清晰、应用巧妙）
- **实验**: ⭐⭐⭐⭐ （全面的定量对比和消融，用户研究有力）
- **写作**: ⭐⭐⭐⭐ （PCA可视化等分析直观，pipeline图清晰）
- **价值**: ⭐⭐⭐⭐ （首次系统探索DINO在视频编辑中的潜力，开辟新方向）

<!-- RELATED:START -->

## 相关论文

- [Taming Teacher Forcing for Masked Autoregressive Video Generation](../../CVPR2025/video_generation/taming_teacher_forcing_for_masked_autoregressive_video_generation.md)
- [DACoN: DINO for Anime Paint Bucket Colorization with Any Number of Reference Images](dacon_dino_for_anime_paint_bucket_colorization_with_any_number_of_reference_imag.md)
- [Multi-subject Open-set Personalization in Video Generation](../../CVPR2025/video_generation/multi-subject_open-set_personalization_in_video_generation.md)
- [VACE: All-in-One Video Creation and Editing](vace_all-in-one_video_creation_and_editing.md)
- [Generative Inbetweening through Frame-wise Conditions-Driven Video Generation](../../CVPR2025/video_generation/generative_inbetweening_through_frame-wise_conditions-driven_video_generation.md)

<!-- RELATED:END -->
