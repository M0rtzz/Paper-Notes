---
title: >-
  [论文解读] MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model
description: >-
  [ECCV 2024][视频生成] 提出 MOFA-Video，通过设计多个领域感知运动场适配器（MOFA-Adapter）为冻结的视频扩散模型（SVD）添加可控运动能力，支持手绘轨迹、人脸关键点等多种控制信号及其组合，实现开放域可控图像动画。 - 领域内动画方法：（如 SadTalker）可精细控制特定类别（人脸、流体）…
tags:
  - "ECCV 2024"
  - "视频生成"
---

# MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model

**会议**: ECCV 2024  
**arXiv**: [2405.20222](https://arxiv.org/abs/2405.20222)  
**领域**: 图像生成

## 一句话总结

提出 MOFA-Video，通过设计多个领域感知运动场适配器（MOFA-Adapter）为冻结的视频扩散模型（SVD）添加可控运动能力，支持手绘轨迹、人脸关键点等多种控制信号及其组合，实现开放域可控图像动画。

## 研究背景与动机

- **领域内动画方法**（如 SadTalker）可精细控制特定类别（人脸、流体），但受限于特定领域，无法泛化到开放域
- **扩散式 I2V 模型**（如 SVD、AnimateDiff）可处理开放域图像动画，但生成内容可能偏离输入图像，且仅支持文本或简单闲置动画，控制能力弱
- **已有控制方法的不足**：DragNUWA 通过自适应归一化建模轨迹但空间对应性差；MotionCtrl 依赖 T2V 模型缺少世界坐标系
- **核心问题**：如何构建一个统一框架，在开放域图像上实现来自多个运动领域的精细可控动画？

## 方法详解

### 整体框架

MOFA-Video 在冻结的 Stable Video Diffusion（SVD）上附加 MOFA-Adapter 作为运动控制模块，类似 ControlNet 的思路。核心是将不同领域的控制信号统一为稀疏运动向量表示，再通过统一的适配器结构生成视频。

### 关键设计

**1. MOFA-Adapter 结构**：
- **稀疏到稠密（S2D）运动生成网络**：接受第一帧图像和稀疏运动提示，生成稠密光流场，采用 CMP 网络结构
- **参考图像编码器**：多尺度卷积特征编码器，提取第一帧的多尺度特征用于 warp
- **融合编码器**：SVD 编码器的可训练副本，将 warp 后的特征与 SVD 解码器的特征融合

**2. 领域感知运动控制**：
- **开放域轨迹**：从视频光流中采样稀疏运动向量进行训练，推理时接受手绘轨迹
- **人脸关键点**：将面部关键点位移转化为稀疏运动向量，统一表示简化框架
- **多适配器组合**：不同领域的 MOFA-Adapter 可零样本联合使用，通过掩码感知策略融合不同区域的控制信号

**3. 长视频生成**：提出周期采样策略，每个扩散步内将帧分组（14帧/组，7帧重叠），对重叠帧的预测噪声取平均，实现更长视频的时序一致性

### 损失函数

冻结 SVD 参数，仅优化 MOFA-Adapter 参数 $\theta_{\mathcal{M}}$：

$$\mathcal{L} = \| \mathcal{S}(\mathcal{V}_t, t, \mathcal{M}(\mathcal{V}_t, t, I, F^s; \theta_{\mathcal{M}})) - \mathcal{V} \|^2$$

其中 $\mathcal{S}$ 为冻结的 SVD，$\mathcal{V}$ 为视频潜在表示。

## 实验关键数据

### 主实验

**轨迹控制对比（vs DragNUWA）**：

| 方法 | 帧一致性↑ | LPIPS↓ | FID↓ | FVD↓ | 控制精度(用户)↑ | 视觉质量(用户)↑ |
|------|-----------|--------|------|------|----------------|----------------|
| DragNUWA | 0.9302 | 0.2705 | 19.66 | 91.38 | 2.76 | 3.18 |
| **MOFA-Video** | **0.9390** | **0.2274** | **16.82** | **86.76** | **3.58** | **3.42** |

**人像动画对比（vs SadTalker, StyleHEAT）**：

| 方法 | CPBD↑ | ID↑ | 保真度(用户)↑ | 自然度(用户)↑ | 视觉质量(用户)↑ |
|------|-------|-----|--------------|--------------|----------------|
| SadTalker | 0.3218 | 0.9188 | 4.15 | 3.12 | 3.97 |
| StyleHEAT | 0.2577 | 0.7993 | 3.26 | 3.65 | 3.70 |
| **MOFA-Video** | **0.4075** | **0.9293** | **4.80** | **3.97** | **4.52** |

### 消融实验

**网络结构消融（轨迹控制）**：

| 变体 | LPIPS↓ | FID↓ | FVD↓ |
|------|--------|------|------|
| w/o warping（纯稀疏条件） | 0.2619 | 18.80 | 184.27 |
| w/o S2D（稀疏 warp） | 0.2376 | 16.87 | 81.80 |
| w/o tuning（直接用重建模型） | 0.2163 | 16.97 | 102.17 |
| **完整模型** | **0.2274** | **16.82** | **86.76** |

### 关键发现

- 稀疏条件模型无法精确控制目标物体轨迹，因为缺乏空间 warp 操作导致空间不对齐
- 稀疏 warp 模型能控制轨迹但由于缺少稠密光流指导，生成结果有严重伪影
- 不同领域的 MOFA-Adapter 必须分别训练，直接用开放域模型做人脸动画会导致不自然表情
- 周期采样策略显著优于朴素帧分组方法，有效解决了长视频的误差累积和时序不一致问题

## 亮点与洞察

- 将多领域运动控制统一为稀疏运动向量问题，设计优雅且可扩展
- 显式的稀疏到稠密光流生成 + 特征 warp 策略在控制精度和生成质量之间取得了良好平衡
- 多 MOFA-Adapter 的零样本组合能力使得同时控制人脸表情和背景运动成为可能
- 相比 DragNUWA 的隐式轨迹建模，显式光流方法能更好地限定运动区域

## 局限性

- 无法控制/生成远离输入图像的新内容（受限于 SVD 的短视频训练数据）
- 大运动引导下可能出现模糊或结构损失等视觉伪影
- 视频长度受限于 SVD 的 14 帧窗口，长视频需要额外的周期采样策略

## 评分

- **新颖性**: 7/10 — 适配器思路源自 ControlNet，核心创新在于运动场的统一建模和多领域组合
- **技术深度**: 8/10 — S2D + warp 的显式运动建模设计扎实，多适配器组合方案合理
- **实验充分度**: 8/10 — 对比实验和消融实验较全面，但缺少定量的长视频评估
- **影响力**: 7/10 — 为可控视频生成提供了实用的统一框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)
- [\[ECCV 2024\] Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion](videoshop_localized_semantic_video_editing_with_noise-extrapolated_diffusion_inv.md)
- [\[ECCV 2024\] MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)
- [\[ECCV 2024\] SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](sv3d_novel_multi-view_synthesis_and_3d_generation_from_a_single_image_using_late.md)
- [\[ECCV 2024\] Evaluating Text-to-Visual Generation with Image-to-Text Generation](evaluating_text-to-visual_generation_with_image-to-text_generation.md)

</div>

<!-- RELATED:END -->
