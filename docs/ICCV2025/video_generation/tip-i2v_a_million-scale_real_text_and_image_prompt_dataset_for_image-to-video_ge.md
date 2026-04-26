---
title: >-
  [论文解读] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation
description: >-
  [ICCV 2025][图像到视频] 构建了首个百万规模的真实用户文本和图像Prompt数据集TIP-I2V（170万+唯一prompt对），包含5个SOTA图像到视频模型的生成视频，并基于此提出了TIP-Eval评估基准、用户偏好分析、以及视频真伪检测等多个研究方向。
tags:
  - ICCV 2025
  - 图像到视频
  - 提示学习
  - 用户偏好分析
  - 视频生成评估
  - 虚假视频检测
---

# TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation

**会议**: ICCV 2025  
**arXiv**: [2411.04709](https://arxiv.org/abs/2411.04709)  
**代码**: [GitHub](https://tip-i2v.github.io/)  
**领域**: 视频生成/数据集  
**关键词**: 图像到视频, Prompt数据集, 用户偏好分析, 视频生成评估, 虚假视频检测

## 一句话总结

构建了首个百万规模的真实用户文本和图像Prompt数据集TIP-I2V（170万+唯一prompt对），包含5个SOTA图像到视频模型的生成视频，并基于此提出了TIP-Eval评估基准、用户偏好分析、以及视频真伪检测等多个研究方向。

## 研究背景与动机

图像到视频（I2V）生成是视频生成领域的重要方向，相比文本到视频（T2V），它提供了更好的可控性、一致性和实用性。然而，当前I2V研究面临一个基础性缺失：**没有专门的I2V prompt数据集**。

现有prompt数据集的局限：

**VidProM**（T2V）和**DiffusionDB**（T2I）仅包含文本prompt，缺乏配对的图像prompt
2. T2V和T2I的prompt语义与I2V根本不同：前者描述要生成的场景（如"a dragon flies over a city"），而I2V的文本prompt是对图像中已有物体的动作指令（如"make the hair dance""the flowers fall"）
3. 现有I2V评估基准（VBench-I2V、I2V-Bench等）覆盖的主题数量有限（仅11-16个主题），且prompt由专家设计而非来自真实用户

这些差距导致研究者无法准确了解用户的真实需求，模型评估也无法反映实际使用场景。因此，一个大规模、来自真实用户的I2V prompt数据集具有迫切需求。

## 方法详解

### 整体框架

TIP-I2V的构建流程：数据收集（Pika Discord频道） → Prompt提取与去重 → 图像Prompt解析 → 元信息标注（UUID/UserID/时间戳/主题/方向/NSFW/嵌入） → 多模型视频生成 → 下游研究应用。

### 关键设计

1. **大规模真实Prompt收集**：通过DiscordChatExporter从Pika官方Discord频道收集2023年7月至2024年10月的聊天消息。利用正则表达式提取文本prompt和对应视频链接，去重后获得**1,701,935**条唯一文本prompt。由于原始图像prompt不可直接获取，通过解析Pika生成视频的第一帧来恢复图像prompt（因为Pika将用户输入图像作为视频首帧）。

2. **丰富的元数据标注**：每个数据点包含：

    - UUID和匿名UserID
    - 时间戳
    - **主题和方向**：使用GPT-4o自动推断用户要将哪个主体（subject）做什么动作（direction）
    - 文本嵌入（text-embedding-3-large）和图像嵌入（CLIP）
    - NSFW状态：文本通过Detoxify检测，图像通过nsfw_image_detection检测

3. **多模型视频生成**：除Pika原始视频外，使用100K随机采样prompt为以下模型生成视频：

    - Stable Video Diffusion
    - Open-Sora
    - I2VGen-XL
    - CogVideoX-5B
   
   每个模型生成100,000条视频，用于跨模型比较研究。

4. **TIP-Eval评估基准**：从最受欢迎的1,000个主题中，每个主题配10对真实用户文本+图像prompt，共10,000条prompt用于模型评估。相比现有基准（VBench-I2V仅11个主题/355条prompt），TIP-Eval在覆盖面和实用性上大幅提升。

### 损失函数 / 训练策略

- 本文为数据集工作，不涉及新模型训练
- 视频真伪检测器使用标准分类训练方案，基于CLIP features的线性分类器（用于强检测器）和CNN（用于泛化实验）

## 实验关键数据

### 主实验

TIP-Eval上5个I2V模型的多维评估（10维评估标准化后比较）：

| 评估维度 | Pika | SVD | Open-Sora | I2VGen-XL | CogVideoX-5B |
|----------|------|-----|-----------|-----------|---------------|
| 美学质量 | 较高 | 中等 | 较低 | 低 | 中等 |
| 一致性 | 高 | 高 | 低 | 低 | 中等 |
| 动态性 | 低 | 低 | 高 | 中等 | 高 |
| 视频-文本对齐 | 0.26（最高） | - | - | - | 略低 |

关键发现：**早期商用模型Pika在10个维度中的8个上超过最新开源CogVideoX-5B**，与基于专家设计prompt的评估结论不同。

### 消融实验

视频真伪检测（现有方法在I2V场景的泛化性）：

| 方法 | Pika | SVD | Open-Sora | I2VGen-XL | CogVideoX | 平均 |
|------|------|-----|-----------|-----------|-----------|------|
| Blind Guess | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |
| CNNSpot | 50.7 | 50.3 | 50.7 | 50.3 | 50.3 | 50.5 |
| UnivFD | 48.5 | 52.0 | 53.4 | 60.9 | 50.7 | 53.1 |
| 强检测器（同域） | 93.2 | 97.3 | 96.9 | 97.9 | 96.2 | **96.3** |
| 强检测器（跨域） | 84.5 | 92.1 | 93.4 | 73.6 | 92.2 | **87.2** |

现有假图像检测方法完全无法泛化到I2V视频检测（近似随机猜测），强调了专门训练I2V检测器的必要性。

### 关键发现

- **用户偏好高度不均衡**：Top-3主题（person、astronaut、portrait painting）全部与人相关；Top-1动作为"move"
- **长尾效应显著**：要覆盖80%用户偏好仅需2,721个主题和309个方向
- **I2V prompt语义独特**：WizMap可视化显示TIP-I2V的文本prompt分布与VidProM和DiffusionDB明显不同
- **三类视频判别**（真实/T2V生成/I2V生成）：强检测器同域准确率96.3%，跨域87.2%，说明I2V生成的视频具有可检测特征

## 亮点与洞察

- 首次构建I2V prompt数据集，填补了重要的研究空白
- 170万规模的真实用户数据提供了独特的用户行为洞察
- TIP-Eval基准在覆盖面（1000主题）和实用性（真实用户prompt）上远超现有基准
- "用户视角的评估可能颠覆专家视角的结论"这一发现对模型开发具有指导意义
- 数据集设计考虑了扩展性——研究者可用新模型和现有prompt生成更多视频

## 局限与展望

- 图像prompt通过视频首帧恢复，可能存在质量损失（压缩伪影等）
- 数据来源单一（仅Pika Discord），用户群体可能存在偏差
- GPT-4o推断的主题/方向存在语义重叠（如person/people/man）
- 受限于计算资源，每个模型仅生成100K视频（占总prompt的约6%）
- 缺乏视频质量的人工标注

## 相关工作与启发

- 延续了VidProM（T2V）和DiffusionDB（T2I）的数据收集范式，但开创了I2V方向
- I2V的"指令式"prompt与T2V/T2I的"描述式"prompt形成鲜明对比
- 为未来的用户导向视频训练数据集构建提供了方法论基础
- 视频溯源（追踪源图像）是一个有实际价值的安全研究方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个I2V prompt数据集，问题定义清晰
- **实验充分度**: ⭐⭐⭐⭐ 涵盖多模型比较、用户分析、检测实验，丰富全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，可视化丰富
- **价值**: ⭐⭐⭐⭐⭐ 数据集+基准的组合对社区有持久贡献，170万规模的prompt具有广泛研究价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)
- [\[ICCV 2025\] DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation](dh-facevid-1k_a_large-scale_high-quality_dataset_for_face_video_generation.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)
- [\[ICCV 2025\] Versatile Transition Generation with Image-to-Video Diffusion](versatile_transition_generation_with_image-to-video_diffusion.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)

<!-- RELATED:END -->
