---
title: >-
  [论文解读] LightMover: Generative Light Movement with Color and Intensity Controls
description: >-
  [CVPR 2026][视频生成][光源操控] LightMover 利用视频扩散先验，将光源编辑建模为序列到序列预测问题，通过统一的控制token表示实现光源位置、颜色和亮度的精确操控，并提出自适应token剪枝机制将控制序列长度减少41%，在光源移动和物体移动任务上均超越现有方法。
tags:
  - "CVPR 2026"
  - "视频生成"
  - "光源操控"
  - "视频扩散模型"
  - "光照编辑"
  - "自适应token剪枝"
  - "物理渲染数据"
---

# LightMover: Generative Light Movement with Color and Intensity Controls

**会议**: CVPR 2026  
**arXiv**: [2603.27209](https://arxiv.org/abs/2603.27209)  
**代码**: [项目页面](https://gengzezhou.github.io/LightMover/)  
**领域**: 视频生成  
**关键词**: 光源操控, 视频扩散模型, 光照编辑, 自适应token剪枝, 物理渲染数据

## 一句话总结

LightMover 利用视频扩散先验，将光源编辑建模为序列到序列预测问题，通过统一的控制token表示实现光源位置、颜色和亮度的精确操控，并提出自适应token剪枝机制将控制序列长度减少41%，在光源移动和物体移动任务上均超越现有方法。

## 研究背景与动机

从单张图像精确编辑光照是一个极具挑战性的任务，因为光照与几何、材质、遮挡之间存在复杂的全局交互。现有方法分为两类：(1) 逆渲染方法（如重建几何、材质、光照后重新渲染），从单张图像求解是高度病态的且计算昂贵；(2) 扩散模型编辑方法（如LightLab）可以调节色调和环境光，但无法建模光源的空间移动。通用图像编辑模型（SDEdit、InstructPix2Pix、Gemini等）缺乏显式的光照参数化表示，无法实现物理合理的光照控制。

核心矛盾在于：现有方法要么缺乏对光源空间位置的建模能力，要么将光照隐式地融入物体移动框架中，无法正确传播阴影、反射和光照衰减。本文的核心idea是将ObjectMover的token序列框架扩展到光照编辑领域，为颜色和亮度设计专门的控制token，并通过"2.5D"学习范式（用视频扩散模型在2D图像上近似3D光传输）实现物理一致的光源操控。

## 方法详解

### 整体框架

LightMover基于5B参数的视频扩散Transformer，将所有输入编码为伪视频帧序列。输入序列包含：(1) 参考图像 $I_{\text{ref}}$；(2) 目标物体裁剪 $I_{\text{obj}}$；(3) 移动地图 $I_{\text{move}}$（用R通道编码源区域、GB通道编码目标区域）；(4) 颜色控制帧 $I_{\text{color}}$；(5) 亮度控制帧 $I_{\text{intensity}}$；(6) 待生成的噪声帧 $X^t$。所有帧通过VAE编码为latent token后由扩散Transformer联合处理。

### 关键设计

1. **多信号位置编码 (MSPE)**:
    - 功能：确保扩散Transformer正确区分不同输入帧的语义角色
    - 核心思路：将四种正交位置子空间整合——空间编码(W,H)保持帧内空间结构、时间编码(T)反映序列顺序、条件类型编码(C)区分不同模态（参考/物体/移动/颜色/亮度/输出）、帧角色编码(R)区分条件输入和预测输出。四者投影、加性组合后通过类RoPE调制
    - 设计动机：标准位置编码无法区分语义上完全不同的输入帧，MSPE使模型能联合推理空间对齐和条件依赖关系

2. **自适应Token剪枝**:
    - 功能：在保持编辑精度的同时显著减少控制序列长度
    - 核心思路：对空间控制信号（如移动地图），根据bounding box面积比决定是否保留全分辨率或按比例下采样；对非空间控制（颜色、亮度），使用可学习的下采样率压缩token数量。整体将控制token长度减少41%
    - 设计动机：朴素地将颜色和亮度编码为额外的全分辨率帧会让序列长度膨胀，限制模型效率和分辨率上限

3. **物理解耦渲染数据管线**:
    - 功能：生成大规模配对训练数据，覆盖光源位置、颜色、亮度的系统变化
    - 核心思路：在Blender中使用25个室内场景和100个光源资产，通过蒙特卡洛路径追踪将每帧分解为环境基底 $I_{\text{amb}}$ 和直接光贡献 $I_{\text{light}}$，然后通过公式 $I_{\text{relit}}(\alpha, G_{\text{illum}}, \mathbf{c}_t) = \alpha I_{\text{amb}} + G_{\text{illum}} I_{\text{light}} \odot \mathbf{c}_t$ 在后处理中生成无穷多光照变体。其中亮度增益 $G_{\text{illum}} = 2^{S_{\text{EV}}}$ 以摄影曝光值(EV)为单位
    - 设计动机：物理解耦使模型能学习光照的因果效应（阴影如何移动、反射如何变亮、间接光如何传播），而非仅学习表面像素变化

### 损失函数 / 训练策略

使用flow matching目标训练：$\mathcal{L} = \mathbb{E}_{t,X^0,X^1}[\|v(S^t, t; \theta)_{[6]} - V^t\|^2]$

训练采用多任务混合策略，合成与真实数据比例10:1，七类任务（光源移动:物体移动:颜色变化:亮度变化:联合变化:光源移除:光源插入）比例为6:3:3:3:1:1。训练分辨率混合512×512和1024×1024（1:1比例）。

## 实验关键数据

### 主实验

**光源移动 (LightMove-A)**:

| 方法 | PSNR ↑ | DINO ↑ | CLIP ↑ |
|------|--------|--------|--------|
| Qwen-Image | 19.01 | 69.94 | 87.27 |
| Gemini-2.5-Flash | 19.59 | 72.46 | 89.72 |
| ObjectMover | 19.49 | 78.12 | 90.48 |
| **LightMover** | **20.38** | **81.27** | **91.85** |

**光源颜色/亮度变化 (LightMove-B)**:

| 方法 | 颜色PSNR | 亮度PSNR | 组合PSNR |
|------|----------|----------|----------|
| Gemini-2.5-Flash | 22.14 | 25.09 | 18.42 |
| **LightMover** | **24.06** | **27.12** | **19.97** |

**通用物体移动 (ObjMove-A)**:

| 方法 | PSNR ↑ | DINO ↑ | CLIP ↑ |
|------|--------|--------|--------|
| ObjectMover | 25.27 | 85.07 | 93.16 |
| **LightMover** | **25.73** | **88.59** | **91.86** |

### 消融实验

| 配置 | PSNR | DINO | CLIP | 说明 |
|------|------|------|------|------|
| 无Light Aug/Color/Intensity | 19.88 | 79.93 | 91.06 | 基线，无辅助任务 |
| +Light Aug | 20.07 | 79.73 | 91.62 | 光照增强提升质量 |
| +All（完整模型） | 20.38 | 81.27 | 91.85 | 多任务联合训练最佳 |
| 无frame-as-condition | 19.53 | 77.32 | 90.01 | 帧编码优于嵌入编码 |
| 无adaptive downsample | 19.38 | 75.62 | 89.81 | 自适应剪枝不可或缺 |

### 关键发现

- 多任务联合训练产生强互增强效应：位置、颜色、亮度信号相互正则化，提升所有光照任务的准确性
- 自适应token剪枝减少41%控制序列长度，性能几乎无损（启用剪枝后PSNR 20.39 vs 20.38）
- LightMover不仅在光源操控上领先，在通用物体移动任务上也超越ObjectMover，说明统一框架的泛化能力

## 亮点与洞察

- "2.5D"范式非常巧妙——用视频扩散模型在2D空间近似3D光传输，避免了完整3D重建的计算代价
- 将光照编辑统一到物体移动框架中的思路很自然，控制token的设计使一个模型同时支持物体和光源编辑
- 物理解耦渲染管线是数据工程的典范：通过分离ambient和direct light，在后处理中生成无穷变体

## 局限与展望

- 当前仅支持可见光源的操控，对于不可见光源（如来自窗外的自然光）的控制能力有限
- 合成数据仅使用25个室内场景，场景多样性可进一步提升
- 光源移动时对远距离光传输效应（如焦散、次表面散射）的建模可能不够准确
- 仅支持单张图像输入，尚未扩展到视频序列的光源操控

## 相关工作与启发

- **vs ObjectMover**: LightMover扩展了ObjectMover的token框架，增加了光照特定的控制信号，在物体移动任务上也更优
- **vs LightLab**: LightLab支持光照色调和开关控制但不支持空间移动，LightMover首次实现光源位置的精确控制
- **vs Gemini-2.5-Flash-Image**: 通用LLM编辑器缺乏光照参数化，在光照传播的物理一致性上明显不如LightMover
- **启发**: 将非空间属性（颜色、亮度）也编码为帧token的思路值得在其他控制生成任务中借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将光源空间移动与颜色/亮度控制统一在视频扩散框架中，自适应token剪枝设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖光源移动/颜色/亮度/联合控制/物体移动/插入/移除，消融全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，物理建模公式严谨，pipeline可视化质量高
- 价值: ⭐⭐⭐⭐ 对照片后期编辑和虚拟场景制作有直接应用价值，推进了精细光照控制的研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters](fastlightgen_fast_and_light_video_generation_with_fewer_steps_and_parameters.md)
- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[CVPR 2026\] PhysVid: Physics Aware Local Conditioning for Generative Video](physvid_physics_aware_local_conditioning_for_generative_video_models.md)
- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](../../ICML2026/video_generation/light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)

</div>

<!-- RELATED:END -->
