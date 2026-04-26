---
title: >-
  [论文解读] AvatarArtist: Open-Domain 4D Avatarization
description: >-
  [CVPR 2025][图像生成][4D头像生成] 提出 AvatarArtist，通过 GAN 和扩散模型协同构建多域 image-triplane 数据集，训练 DiT 生成参数化三平面 + 运动感知跨域渲染器，实现从任意风格单张肖像生成可驱动的 4D 头像。
tags:
  - CVPR 2025
  - 图像生成
  - 4D头像生成
  - 参数化三平面
  - GAN
  - 跨域渲染
  - 表情驱动
---

# AvatarArtist: Open-Domain 4D Avatarization

**会议**: CVPR 2025  
**arXiv**: [2503.19906](https://arxiv.org/abs/2503.19906)  
**代码**: https://kumapowerliu.github.io/AvatarArtist （代码/数据/模型即将公开）  
**领域**: 扩散模型 / 3D视觉  
**关键词**: 4D头像生成, 参数化三平面, GAN-扩散协同, 跨域渲染, 表情驱动

## 一句话总结
提出 AvatarArtist，通过 GAN 和扩散模型协同构建多域 image-triplane 数据集，训练 DiT 生成参数化三平面 + 运动感知跨域渲染器，实现从任意风格单张肖像生成可驱动的 4D 头像。

## 研究背景与动机

**领域现状**：单张图像驱动的头像动画（avatarization）分为 2D 和 4D 两条路线。2D 方法（如 LivePortrait）利用扩散模型先验可处理多风格输入，但缺乏 3D 结构理解，大角度旋转时出现几何畸变。4D 方法（如 Portrait4D、InvertAvatar）通过 NeRF/3DGS 等神经渲染保证几何一致性，但受限于 4D 数据的匮乏，只能处理真实人脸域。

**现有痛点**：4D 方法的核心瓶颈是缺乏多域 4D 训练数据。要构建 image-4D 表示对需要多视角多表情的 4D 采集数据，这对卡通、游戏角色、雕塑等非写实域几乎不可能获取。现有 4D GAN（如 Next3D）可以无监督地从 2D 图像学习 4D 表示，但 GAN 的模式坍塌问题使其难以覆盖多样化的视觉域。

**核心矛盾**：需要大规模、多域的 4D 数据来训练通用模型，但 4D 数据获取极其困难；GAN 能生成 4D 数据但只能覆盖单一域；扩散模型能覆盖多域但不能直接生成 4D 表示。

**本文目标** 如何构建一个跨域的 4D 头像生成系统，能从任意风格（写实、卡通、游戏角色、雕塑等）的单张肖像生成可驱动的 4D 头像。

**切入角度**：GAN 和扩散模型各有所长——扩散模型擅长跨域 2D 图像生成，GAN 擅长无监督 2D→4D 转换。两者协同可以解决数据问题：先用扩散模型生成多域 2D 数据（保持表情/姿态一致以复用 3DMM mesh），再用 GAN 将 2D 数据转为 4D 表示对。

**核心 idea**：用扩散模型帮 GAN 扩展到 28 个非写实域来合成 56 万 image-triplane 对，再训练 DiT + 跨域渲染器实现开放域 4D 头像生成。

## 方法详解

### 整体框架
整个系统分三个阶段。**阶段一（数据构建）**：用 SDEdit + ControlNet 将 FFHQ 写实人脸图像迁移到 28 个非写实域（动漫、乐高等），保持表情/姿态一致以复用 3DMM mesh，然后为每个域微调 Next3D GAN，随机采样生成 56 万 image-参数化三平面对。**阶段二（4D 生成）**：训练 Triplane VAE 压缩三平面表示，再训练 image-conditioned DiT 从单张图像生成参数化三平面的隐变量。**阶段三（渲染）**：设计基于 ViT 的运动感知跨域渲染器，融合源图像特征和生成的三平面，通过隐式运动嵌入驱动表情变化，最终体渲染输出。

### 关键设计

1. **GAN-扩散协同数据构建流水线**

    - 功能：生成覆盖 28 个视觉域的 56 万 image-参数化三平面训练数据对
    - 核心思路：首先，对 FFHQ 写实人脸图像加噪后用 StableDiffusion（配合域特定 prompt）去噪，结合 landmark ControlNet 控制表情，将图像迁移到目标域（如动漫、雕塑）。因为输出图像保持了原始的姿态和表情，可以直接复用写实域提取的 3DMM mesh 标签。每个域生成 6000 张图像用于微调独立的 Next3D GAN，然后每个 GAN 采样 20K 个 image-triplane 对（28 域共 560K 对）。SDEdit 保持整体结构，ControlNet 保持表情精度，缺一不可
    - 设计动机：非写实域无法准确提取 3DMM mesh，直接训练 GAN 不可行。通过从写实域迁移并复用 mesh，巧妙绕过了这一瓶颈

2. **Image-Conditioned Diffusion Transformer**

    - 功能：从单张肖像图像生成参数化三平面的隐表示
    - 核心思路：先训练 VAE 将三平面 $\in \mathbb{R}^{256 \times 256 \times 4 \times 32}$ 压缩到 $\mathbb{R}^{64 \times 64 \times 4 \times 8}$。然后训练 DiT-XL/2（28 层 DiT blocks），输入为加噪的 triplane latent 展平序列、条件图像通过 CLIP 提取语义嵌入（cross-attention 注入）和 DINO 提取细节 token（与 latent 拼接后 self-attention）。训练用 IDDPM 目标预测噪声和方差，10% 概率随机 drop 条件图像支持 classifier-free guidance。推理时用 19 步 DPMSolver
    - 设计动机：双条件注入（CLIP 语义 + DINO 细节）确保生成的三平面既语义正确又保留细节。VAE 压缩减少 DiT 计算量

3. **运动感知跨域渲染器（Motion-Aware Cross-Domain Renderer）**

    - 功能：从生成的参数化三平面和源图像渲染出高质量的目标表情/姿态图像，同时保持身份
    - 核心思路：用编码器 $E_I$ 提取源图像特征，送入 ViT。在 ViT 的 self-attention 中注入 DiT 生成的参数化三平面来中和面部表情、规范化姿态（消除源图像的表情信息）。然后通过 cross-attention 注入隐式运动嵌入（motion embedding）来赋予目标表情。ViT 输出解码后与栅格化的三平面融合，最终经体渲染 + 超分辨率生成输出图像。隐式运动嵌入不包含空间信息，避免了身份泄露
    - 设计动机：Next3D 原始的 CNN 渲染器在跨域场景下表现很差，出现严重身份泄露和表情不匹配。ViT 的注意力机制更适合跨域特征融合，隐式运动嵌入避免了 mesh 不准确导致的伪影

### 损失函数 / 训练策略
VAE 训练使用三平面重建的 $\mathcal{L}_1$ loss + 渲染图像的 $\mathcal{L}_1$ 和 LPIPS loss（不用对抗 loss 因会导致训练不稳定）。DiT 使用 IDDPM loss 预测噪声和方差。渲染器训练参考 Portrait4D 等方法的 loss 组合，在 1200 万张多域图像上训练。

## 实验关键数据

### 主实验

| 方法 | 自驱动LPIPS↓ | 自驱动ID↑ | 跨域AKD↓ | 跨域APD↓ | 跨域FID↓ | 跨域CLIP↑ |
|------|-------------|----------|---------|---------|---------|----------|
| LivePortrait (2D) | 0.27 | 0.65 | 4.92 | 139.35 | 100.3 | 0.91 |
| XPortrait (2D) | 0.31 | 0.63 | 10.67 | 237.4 | 78.6 | 0.89 |
| Portrait4Dv2 (4D) | 0.29 | 0.58 | 7.13 | 63.3 | 140.5 | 0.75 |
| InvertAvatar (4D) | 0.42 | 0.32 | 20.78 | 134.9 | 194.7 | 0.64 |
| **AvatarArtist (Ours)** | **0.26** | **0.69** | **2.58** | **52.3** | 89.3 | 0.84 |

### 消融实验

| 配置 | FID↓ | CLIP↑ | AKD↓ | APD↓ |
|------|------|-------|------|------|
| Next3D CNN渲染器 | 130.72 | 0.73 | 5.89 | 42.93 |
| **完整模型（ViT渲染器）** | **68.69** | **0.86** | **2.56** | **40.89** |

### 关键发现
- 在跨域驱动任务（非写实源→写实目标）中，本文在运动精度指标（AKD 2.58, APD 52.3）上大幅超越所有方法，说明 4D 表示在姿态/表情传递上的优势
- 自驱动任务中本文在 LPIPS (0.26) 和 ID (0.69) 上均最优，说明身份保持能力强
- 替换 ViT 渲染器为 CNN 后 FID 从 68.69 暴涨到 130.72，CLIP 从 0.86 降到 0.73，跨域渲染器是关键模块
- 数据构建中 SDEdit 和 ControlNet 缺一不可：去掉 SDEdit 表情偏差大，去掉 ControlNet 姿态保持但表情不一致

## 亮点与洞察
- **GAN 和扩散模型的优势互补**非常巧妙：扩散模型提供多域数据能力，GAN 提供无监督 2D→4D 转换能力，合力解决了"多域 4D 数据不存在"的根本问题。这种协同策略可迁移到其他需要稀缺 3D/4D 数据的任务
- **复用 3DMM mesh 的跨域迁移思路**：通过保持表情/姿态一致的域迁移来避免非写实域的 mesh 提取困难，是一种实用的工程技巧
- **隐式运动嵌入替代显式 mesh 驱动**：避免了 mesh 提取不准确导致的伪影，对跨域泛化至关重要

## 局限与展望
- 自驱动 FID (52.62) 略差于 LivePortrait (46.49)，2D 纹理质量有提升空间
- 跨域 CLIP ID 分数 (0.84) 低于 2D 方法 LivePortrait (0.91)，跨域身份保真度仍有差距
- 需要训练 28 个独立的域特定 GAN（每个 300K 迭代），数据流水线训练成本高
- 依赖 FaceVerse 3DMM mesh 提取，对非标准面部结构（如侧脸、遮挡）可能失效
- 渲染器需要在 1200 万张图像上训练，资源需求大

## 相关工作与启发
- **vs LivePortrait**: LivePortrait 是 2D 方法，身份保持好但大角度旋转有几何畸变。AvatarArtist 通过 4D 表示解决了几何一致性，代价是略低的纹理质量
- **vs Portrait4Dv2**: 都是 4D 方法，但 Portrait4Dv2 在非写实域泛化差（CLIP 0.75 vs 0.84）。关键区别是 AvatarArtist 通过多域数据构建实现了开放域泛化
- **vs Rodin**: Rodin 做静态头像，本文做可驱动的动态 4D 头像，复杂度更高。但 Rodin 的 image-3D 数据构建思路启发了本文的数据流水线设计

## 评分
- 新颖性: ⭐⭐⭐⭐ GAN-扩散协同数据构建是亮点，但各组件技术本身较成熟
- 实验充分度: ⭐⭐⭐⭐ 定量定性对比充分，消融验证各组件，但缺少用户研究和更多域的测试
- 写作质量: ⭐⭐⭐⭐ 整体流程清晰，图表丰富，数据流水线描述详细
- 价值: ⭐⭐⭐⭐ 首次实现开放域 4D 头像生成，但训练成本高限制了实际应用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] DNF: Unconditional 4D Generation with Dictionary-Based Neural Fields](dnf_unconditional_4d_generation_with_dictionary-based_neural_fields.md)
- [\[CVPR 2025\] OpenSDI: Spotting Diffusion-Generated Images in the Open World](opensdi_spotting_diffusion-generated_images_in_the_open_world.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](../../NeurIPS2025/image_generation/model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)
- [\[CVPR 2025\] DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles](doracycle_domain-oriented_adaptation_of_unified_generative_model_in_multimodal_c.md)
- [\[CVPR 2025\] Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment](everything_to_the_synthetic_diffusion-driven_test-time_adaptation_via_synthetic-.md)

<!-- RELATED:END -->
