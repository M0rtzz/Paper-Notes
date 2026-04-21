---
title: >-
  [论文解读] S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis
description: >-
  [ECCV 2024][3D视觉][说话头合成] 提出 S³D-NeRF，利用分层面部外观编码器、跨模态面部形变场和唇音同步判别器，实现了仅需一张图片即可由语音驱动生成高保真说话头视频的 NeRF 方法，在视频质量和唇音同步方面超越了现有单图方法。
tags:
  - ECCV 2024
  - 3D视觉
  - 说话头合成
  - 神经辐射场
  - 语音驱动
  - 单图驱动
  - 唇音同步
---

# S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis

**会议**: ECCV 2024  
**arXiv**: [2408.09347](https://arxiv.org/abs/2408.09347)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 说话头合成, 神经辐射场, 语音驱动, 单图驱动, 唇音同步

## 一句话总结
提出 S³D-NeRF，利用分层面部外观编码器、跨模态面部形变场和唇音同步判别器，实现了仅需一张图片即可由语音驱动生成高保真说话头视频的 NeRF 方法，在视频质量和唇音同步方面超越了现有单图方法。

## 研究背景与动机

**领域现状**：说话头合成（Talking Head Synthesis）是计算机视觉中的重要任务，当前基于 NeRF 的方法（如 AD-NeRF、ER-NeRF）在驱动说话头视频方面表现优异，但大多为单人专属模型（identity-specific），需要对每个新身份进行长时间训练，且通常依赖中间表示（如 3DMM 系数）而非直接使用音频信号。

**现有痛点**：(1) 单人 NeRF 方法泛化能力差，遇到新身份时必须重新训练，训练成本高达数十小时；(2) 现有单图驱动方法（如 SadTalker、StyleTalk）虽然泛化性好，但图像质量和唇形精度不尽人意；(3) 大多数方法通过中间表示（3DMM、关键点等）间接映射音频信号，引入了信息损失。

**核心矛盾**：直接从音频到面部形变的映射是非平凡的——音频信号仅与嘴唇区域强相关，与面部其他区域（眉毛、眼睛）的关联较弱，但说话头合成需要整张脸的协调运动。这导致了"局部信号 vs 全局运动"之间的矛盾。

**本文目标** (1) 如何从单张图片学习到具有足够表达力的身份外观特征？(2) 如何根据音频信号准确建模不同面部区域的运动？(3) 如何保证嘴唇区域的时序一致性？

**切入角度**：作者观察到，不同面部区域与语音信号的相关性差异显著——嘴唇区域相关性最强，而上半脸几乎不受音频影响。利用跨注意力机制计算音频-视觉相关性得分，可以作为先验知识指导面部形变预测，实现区域感知的精确动画。

**核心 idea**：用多尺度三平面表示 + 音频-视觉跨注意力形变场 + 唇音同步判别器，构建首个单图语音驱动的泛化 NeRF 说话头方法。

## 方法详解

### 整体框架
输入是一张单帧源图像和一段驱动音频序列，输出是高保真的多视角说话头视频。整体 pipeline 分为三个阶段：(1) 分层面部外观编码器从源图像提取多尺度特征并构建三平面表示；(2) 跨模态面部形变场根据音频信号和视觉特征的相关性预测 3D 点的位移；(3) 体渲染生成粗糙面部图像后，通过超分辨率模块补充细节和背景。

### 关键设计

1. **分层面部外观编码器 (Hierarchical Facial Appearance Encoder)**:

    - 功能：从单张源图像提取多尺度特征，构建高效的三平面表示以建模任意说话人的外观
    - 核心思路：采用特征金字塔（FPN）结构，通过下采样卷积网络提取四个尺度的特征图 $\mathbf{D}_0$ 到 $\mathbf{D}_3$，再通过上采样和拼接得到分层特征 $\mathbf{F}_i$。每个尺度的特征图被 reshape 为三个正交子平面（$\mathbf{F}_{xy}$, $\mathbf{F}_{yz}$, $\mathbf{F}_{xz}$），形成多尺度三平面表示。渲染时，通过源姿态和目标姿态的相机变换定位 3D 点在三平面上的投影位置，用双线性插值提取特征向量
    - 设计动机：单一尺度的特征图无法同时捕获粗粒度的结构信息和细粒度的纹理细节。多尺度三平面表示允许不同分辨率的特征互补，使得即使只有一张图片也能准确建模陌生人的面部外观

2. **跨模态面部形变场 (Cross-modal Facial Deformation Field)**:

    - 功能：根据语音信号预测面部各区域的 3D 形变，实现精确的语音动画
    - 核心思路：首先将多尺度视觉特征通过 Slot Attention 聚合为统一的视觉嵌入 $\mathbf{F}_{agg}$，然后用 1D 全卷积网络处理语音信号得到音频特征 $\mathbf{a}_{dri}$。通过多头跨注意力（MHCA）计算视觉嵌入（query）与音频特征（key/value）之间的相关性分数 $\mathbf{F}_{cm} = \text{MHCA}(\mathbf{F}_{agg}, \mathbf{a}_{dri})$。该相关性分数作为先验知识输入 U-Net 结构的形变预测模块，结合 3D 位置和音频特征预测位移 $\Delta\mathbf{x} = \text{Deform}(\mathbf{x}, \mathbf{a}_{dri}, \mathbf{F}_{cm})$
    - 设计动机：朴素的形变模块直接从音频预测全脸形变，容易产生模糊结果，因为音频只与嘴部区域强相关。跨注意力机制的热力图显示下半脸区域激活最大，证实了这种区域感知的设计能正确分配不同区域的运动幅度

3. **唇音同步判别器 (Lip-sync Discriminator)**:

    - 功能：约束生成视频的嘴唇运动与驱动音频在时序上保持同步
    - 核心思路：判别器包含视觉分支和音频分支，分别从 $T$ 帧唇部区域和音频片段提取嵌入 $\mathbf{e}_l$ 和 $\mathbf{e}_a$，通过余弦相似度 $\cos(\mathbf{e}_l, \mathbf{e}_a)$ 判断同步程度。训练时使用对比三元组损失，在更大规模混合数据集（HDTF+LRS2）上预训练。生成器的唇音同步损失为 $\mathcal{L}_{sync}^{gen} = \cos(\mathbf{e}_l^{gen}, \mathbf{e}_{gt})$，冻结判别器后反向传播梯度
    - 设计动机：NeRF 生成的视频缺乏对嘴唇时序一致性的显式约束，容易出现嘴型不正确的帧。引入唇音同步判别器作为外部监督，比 Wav2Lip 的同步专家更具区分力

### 损失函数 / 训练策略
总损失为 $\mathcal{L} = \mathcal{L}_{pix} + 0.01\mathcal{L}_{per} + \mathcal{L}_{adv} + 0.5\mathcal{L}_{sync}^{gen} + 0.001\mathcal{L}_{deform}$，其中包括像素级 L2 重建损失、感知损失、StyleGAN 判别器的对抗损失、唇音同步损失和形变正则化损失。采用粗到精策略：先用 NeRF 渲染内脸区域的粗糙图像，再通过超分辨率模块补充外脸和背景细节，减轻 NeRF 同时建模多种纹理的难度。

## 实验关键数据

### 主实验

**与单图方法对比 (HDTF 数据集)**:

| 方法 | SSIM↑ | LPIPS↓ | F-LMD↓ | M-LMD↓ | CPBD↑ | Sync↑ |
|------|-------|--------|--------|--------|-------|-------|
| Wav2Lip | 0.749 | 0.332 | 3.582 | 3.652 | 0.195 | 8.365 |
| SadTalker | 0.776 | 0.289 | 3.623 | 3.273 | 0.234 | 6.188 |
| StyleTalk | 0.678 | 0.386 | 4.621 | 3.654 | 0.192 | 6.473 |
| **S³D-NeRF** | **0.819** | **0.258** | **2.799** | **2.929** | **0.263** | 6.514 |

**与 NeRF 方法对比 (AD-NeRF 提供的视频)**:

| 方法 | SSIM↑ | M-LMD↓ | Sync↑ | FPS | Fit Time |
|------|-------|--------|-------|-----|----------|
| AD-NeRF | 0.846 | 1.982 | 5.626 | 0.13 | 36h |
| ER-NeRF | 0.884 | 1.659 | 6.781 | 29 | 4.5h |
| **S³D-NeRF** | 0.852 | **1.493** | **7.118** | 9.5 | **<0.01h** |

### 消融实验

| 配置 | SSIM↑ | LPIPS↓ | F-LMD↓ | M-LMD↓ | Sync↑ |
|------|-------|--------|--------|--------|-------|
| w/o Deform | 0.556 | 0.457 | N/A | N/A | 0.240 |
| Naive Deform (ND) | 0.744 | 0.278 | 3.475 | 3.507 | 5.201 |
| Feat-Concatenate Deform | 0.834 | 0.242 | 2.855 | 3.196 | 6.106 |
| w/o Lip Sync | 0.801 | 0.263 | 3.276 | 3.342 | 5.649 |
| **S³D-NeRF (full)** | **0.829** | **0.258** | **2.799** | **2.929** | **6.514** |

### 关键发现
- 跨模态形变场的贡献最大：去掉形变模块后完全无法生成说话动画，Naive Deform 产生模糊的平均脸效果
- 跨注意力相关性分数作为先验显著优于直接拼接特征的方案（FCD），Sync 分数提升约 6.7%
- 唇音同步判别器带来约 15% 的 Sync 提升和嘴部关键点距离的显著改善
- S³D-NeRF 在泛化性上具有绝对优势：无需针对新身份重新训练（Fit Time < 0.01h），而单人 NeRF 方法需要数小时到数十小时

## 亮点与洞察
- **跨注意力相关性得分作为形变先验**：这是一个非常巧妙的设计——利用注意力热力图直接反映音频和面部不同区域的相关程度，相当于给形变网络提供了"哪里该动、哪里不该动"的空间指引，解决了音频作为局部信号难以驱动全局运动的核心问题
- **多尺度三平面表示泛化到新身份**：相比 3DMM 等参数化表示，从原始图像直接构建三平面表示保留了更多身份细节，且多尺度设计使单张图片也能提供足够丰富的外观信息
- **粗到精的分离渲染策略**可以迁移到其他 NeRF 任务——将重要区域和背景区域分开建模，降低 NeRF 的学习难度

## 局限与展望
- 不支持背景更换——源图像的背景固定
- 当头部姿态过大时面部轮廓容易模糊
- 渲染速度为 9.5 FPS，虽优于 AD-NeRF 但远不如 ER-NeRF 的 29 FPS
- 未使用 3D Gaussian Splatting 等更高效的 3D 表示，可能在速度和质量上有进一步提升空间
- 训练数据集（HDTF）规模有限，扩大数据规模或引入预训练可能进一步提升跨身份泛化能力

## 相关工作与启发
- **vs AD-NeRF/ER-NeRF**: 这些方法是单人专属 NeRF，图像质量极高但必须针对每个身份单独训练。S³D-NeRF 牺牲少量图像质量换取了极强的泛化能力（Fit Time < 0.01h vs 4.5~36h）
- **vs SadTalker/StyleTalk**: 这些是基于 3DMM 中间表示的单图方法，S³D-NeRF 跳过中间表示直接从音频驱动 NeRF，在图像质量（SSIM +5%）和嘴部精度（F-LMD 23%↓）上全面领先
- **vs Wav2Lip**: Wav2Lip 在唇音同步上表现极佳（Sync > GT），但生成的面部非常模糊（LPIPS 高、CPBD 低），S³D-NeRF 在清晰度上有显著优势

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个单图语音驱动的泛化 NeRF 说话头方法，跨注意力形变场设计新颖
- 实验充分度: ⭐⭐⭐⭐ 与单图方法和 NeRF 方法都进行了全面对比，消融实验覆盖所有组件
- 写作质量: ⭐⭐⭐⭐ 问题分解清晰，方法描述层次分明
- 价值: ⭐⭐⭐⭐ 在泛化性和质量之间取得了良好平衡，具有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)
- [TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting](talkinggaussian_structure-persistent_3d_talking_head_synthesis_via_gaussian_spla.md)
- [Dynamic Neural Radiance Field from Defocused Monocular Video](dynamic_neural_radiance_field_from_defocused_monocular_video.md)
- [Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation](mesh2nerf_direct_mesh_supervision_for_neural_radiance_field_representation_and_g.md)
- [Invertible Neural Warp for NeRF](invertible_neural_warp_for_nerf.md)

<!-- RELATED:END -->
