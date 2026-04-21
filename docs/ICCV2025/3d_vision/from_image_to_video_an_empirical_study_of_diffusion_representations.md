---
title: >-
  [论文解读] From Image to Video: An Empirical Study of Diffusion Representations
description: >-
  [ICCV 2025][3D视觉][扩散模型] 系统对比了相同架构(WALT)在图像 vs 视频生成目标下训练的扩散模型在下游视觉理解任务上的表现，发现视频扩散模型在所有任务上一致优于图像对应物，尤其在需要运动和3D空间理解的任务上优势显著（点跟踪+68%、相机位姿+60%）。
tags:
  - ICCV 2025
  - 3D视觉
  - 扩散模型
  - 视频表示学习
  - 图像vs视频扩散
  - 运动理解
  - WALT
---

# From Image to Video: An Empirical Study of Diffusion Representations

**会议**: ICCV 2025  
**arXiv**: [2502.07001](https://arxiv.org/abs/2502.07001)  
**代码**: 无公开代码  
**领域**: 3d_vision  
**关键词**: 扩散模型, 视频表示学习, 图像vs视频扩散, 运动理解, WALT

## 一句话总结

系统对比了相同架构(WALT)在图像 vs 视频生成目标下训练的扩散模型在下游视觉理解任务上的表现，发现视频扩散模型在所有任务上一致优于图像对应物，尤其在需要运动和3D空间理解的任务上优势显著（点跟踪+68%、相机位姿+60%）。

## 研究背景与动机

扩散模型在图像和视频生成领域取得了巨大成功，其内部表示已被证明对图像理解任务有强大潜力（如分割、深度估计、关键点匹配等）。然而，一个关键问题尚未回答：

**视频扩散表示几乎未探索**：尽管视频扩散模型已能生成高质量视频内容，但其表示的理解能力完全未知。相比之下，图像扩散表示已有大量研究。

**缺乏公平对比**：要比较图像vs视频扩散的表示质量，需要相同架构在不同训练目标下的对比。但现有模型（如Stable Diffusion 2.1 vs SVD）参数量差异太大（865M vs 1.5B），无法得出有意义的结论。

**时间信息的作用未知**：视频训练引入的时序理解如何影响表示质量？运动信息是否增强了语义理解、3D感知或物体追踪能力？

**核心洞察**：WALT模型的混合架构恰好允许公平对比——相同参数量的模型可以分别用于图像和视频生成训练，仅在注意力窗口中是否包含时间维度上有所区别。这为首次直接对比提供了理想平台。

## 方法详解

### WALT模型架构

WALT (Windowed-Attention Latent Transformer) 是一个基于Transformer的潜在扩散模型，关键特性包括：

- **共享tokenizer**：使用MAGVIT-v2的因果3D CNN编码器，将图像和视频压缩到共享潜在空间。视频被编码为 $z \in \mathbb{R}^{(1+m) \times h \times w \times c}$，其中第一帧独立编码，后续 $m=4$ 个latent表示16帧
- **窗口注意力**：交替使用时空窗口块(spatio-temporal)和空间窗口块(spatial-only)
- **图像模式兼容**：图像生成时，时空块中每个latent仅自注意（等效于identity mask）

### I-WALT vs V-WALT

为公平对比，构建了两个模型：
- **V-WALT**（视频模型）：标准WALT，时空窗口块同时做空间和时间注意力
- **I-WALT**（图像模型）：将时空窗口注意力块替换为仅空间窗口注意力块（相同参数量）

V-WALT在图像+视频数据集上训练；I-WALT在相同数据集上训练，但视频中随机抽帧作为独立图像。两者共享完全相同的架构和参数量。

### 特征提取与探测框架

将WALT作为冻结骨干，训练轻量级readout head评估下游任务：

1. **特征提取**：对输入添加噪声到时间步 $t$，前向传播一次（无需完整去噪），提取中间Transformer块的激活
2. **readout架构**：根据任务选择不同readout头
    - 分类任务（ImageNet、K400等）：attentive readout（可学习query token + cross-attention + MLP）
    - 深度估计：Scene Representation Transformer解码器（cross-attention + 逐像素MLP）
    - 点/框跟踪：MooG风格的循环readout（预测-校正机制）

### 评估任务覆盖

从纯语义到时空理解的完整谱系：
- **图像分类**：ImageNet（物体）、Places365（场景）、iNat2018（细粒度）
- **动作识别**：K400/K700（外观主导）、SSv2（运动敏感）
- **单目深度估计**：ScanNet
- **相机位姿估计**：RealEstate10k
- **视觉对应**：点跟踪(Perception Test)、框跟踪(Waymo Open)

## 实验

### 主实验：视频 vs 图像扩散表示

| 任务 | I-WALT(基线100%) | V-WALT相对提升 | 任务性质 |
|------|-----------------|---------------|---------|
| Places365 | 100% | +0.6% | 纯语义 |
| ImageNet | 100% | +1.8% | 纯语义 |
| iNat2018 | 100% | +11% | 细粒度语义 |
| K400 | 100% | +8% | 外观理解 |
| K700 | 100% | +12% | 外观理解 |
| SSv2 | 100% | +42% | 运动理解 |
| Depth | 100% | +16% | 3D感知 |
| Cam. Pose | 100% | **+60%** | 空间理解 |
| Obj. Tracks | 100% | +23% | 时空对应 |
| PointTracks | 100% | **+68%** | 精确定位 |

**关键发现**：
- V-WALT在所有10个任务上一致优于I-WALT，但提升幅度差异巨大（0.6%-68%）
- 纯语义任务（Places365）几乎无差异，而需要运动/空间理解的任务（PointTracks、Cam. Pose、SSv2）提升最为显著
- iNat2018的+11%提升出乎意料，可能是视频训练增强了对细粒度视觉差异的敏感性

### 消融实验：噪声水平与网络层选择

| 设计选择 | 最优配置 | 影响观察 |
|---------|---------|---------|
| 噪声水平 $t$ | 大多数任务t=200最优 | 高噪声普遍有害；跟踪任务对噪声更敏感(t=0-100最优) |
| 网络块 $l$ | ~2/3深度处(l=11-16)最优 | 模型隐含分为编码器/解码器，最佳表示在交界处 |
| 训练进度 | 20%训练即达>90%性能 | 识别任务随训练持续提升；跟踪和深度任务较早达到峰值 |
| 模型规模 | 284M→1.9B显著提升 | 分类任务受益最大（类别数多时尤为明显），PointTracks例外 |

**关键发现**：
- 少量噪声(t=200)有助于大多数任务，但跟踪等低层任务偏好零或极少噪声
- 最优特征层位于模型深度的~2/3处，暗示扩散模型存在隐式的编码器-解码器结构
- 相机位姿估计在26%训练后性能下降，暗示长时间训练可能导致特征偏向生成而非理解
- PCA可视化揭示V-WALT选择性关注运动区域，而I-WALT关注所有显著区域（包括静态物体）

### 与其他表示模型的对比

| 模型 | 类型 | 参数量 | 相对性能(vs I-WALT) |
|------|------|--------|-------------------|
| DINOv2 | 对比学习(图像) | 300M | 语义任务强，跟踪弱 |
| SigLIP | 图文对齐(图像) | 400M | 语义任务最强 |
| V-JEPA | 特征重建(视频) | 300M | 与V-WALT互有胜负 |
| VideoMAE | 像素重建(视频) | 300M | 整体稍弱 |
| V-WALT(1.9B) | 扩散(视频) | 1.9B | 大部分任务显著提升 |

V-WALT在深度和运动理解任务上具竞争力，但在纯语义任务上被DINOv2和SigLIP主导，揭示了生成式扩散模型在语义理解上的核心短板。

## 亮点与洞察

1. **首次公平的图像vs视频扩散表示对比**：利用WALT的混合架构，在完全控制变量的条件下得出结论——视频训练一致性地改善表示质量
2. **运动敏感性的发现**：PCA可视化和brick wall实验巧妙地揭示了V-WALT对运动区域的选择性关注，解释了其在时空任务上的优势
3. **噪声和层深度的系统研究**：为从扩散模型提取特征提供了实用指南——少量噪声(~200)、~2/3深度
4. **训练动态的有趣观察**：生成质量(FVD)持续改善，但部分下游任务在早期就达到峰值甚至开始下降，暗示生成能力与表示能力可能存在trade-off

## 局限性

- 仅使用WALT单一模型架构，结论的泛化性有待验证（尽管作者给出了充分理由）
- 使用Google内部数据集和模型训练，结果难以完全复现
- 未探索微调扩散模型（仅使用冻结特征+轻量探测），可能低估了扩散表示的潜力
- readout head的选择可能影响结论，虽然采用了与先前工作一致的设计
- 未提供代码和模型权重

## 相关工作

- **图像扩散表示**：DDPM-Seg、DIFT、零样本分类、多任务理解（分割+深度）等
- **视频表示学习**：V-JEPA（对比）、VideoMAE（掩码重建）、VideoPrism（混合策略）
- **扩散模型架构**：WALT（窗口注意力潜在Transformer）、Stable Diffusion/SVD（U-Net系列）

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首次系统对比但属实证研究，无新方法提出
- **技术质量**: ⭐⭐⭐⭐⭐ — 实验设计严谨，变量控制到位，10个下游任务覆盖广
- **实验充分度**: ⭐⭐⭐⭐⭐ — 主实验+噪声/层消融+训练动态分析+模型规模+对比模型，极为全面
- **表达清晰度**: ⭐⭐⭐⭐⭐ — 图表简洁直观，结论清晰，讨论深入
- **综合评分**: 8.0/10

<!-- RELATED:START -->

## 相关论文

- [PolarAnything: Diffusion-based Polarimetric Image Synthesis](polaranything_diffusion-based_polarimetric_image_synthesis.md)
- [Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)
- [Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_a.md)

<!-- RELATED:END -->
