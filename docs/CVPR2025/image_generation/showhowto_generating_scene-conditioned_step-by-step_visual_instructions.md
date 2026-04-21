---
title: >-
  [论文解读] ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions
description: >-
  [CVPR 2025][图像生成][视觉指令生成] 本文提出 ShowHowTo，一个视频扩散模型，能够根据用户提供的初始场景图像和分步文字指令，生成与场景一致的逐步视觉指令序列；同时构建了包含57.8万条序列的大规模教学数据集，通过全自动管线从网络教学视频中采集。
tags:
  - CVPR 2025
  - 图像生成
  - 视觉指令生成
  - 视频扩散模型
  - 教学视频
  - 场景一致性
  - 分步生成
---

# ShowHowTo: Generating Scene-Conditioned Step-by-Step Visual Instructions

**会议**: CVPR 2025  
**arXiv**: [2412.01987](https://arxiv.org/abs/2412.01987)  
**代码**: https://soczech.github.io/showhowto/  
**领域**: 扩散模型 / 图像生成  
**关键词**: 视觉指令生成, 视频扩散模型, 教学视频, 场景一致性, 分步生成

## 一句话总结

本文提出 ShowHowTo，一个视频扩散模型，能够根据用户提供的初始场景图像和分步文字指令，生成与场景一致的逐步视觉指令序列；同时构建了包含57.8万条序列的大规模教学数据集，通过全自动管线从网络教学视频中采集。

## 研究背景与动机

**领域现状**：当前大语言模型能可靠地生成个性化的分步文字指导，但将文字指令转化为视觉指令仍极具挑战。视频生成模型专注于短视频片段生成，图像生成模型每次只产出单张图像，都不能直接生成长跨度的多步骤视觉指令序列。

**现有痛点**：（1）现有方法只能生成单步视觉指令或不与用户场景关联——生成的图像展示的是任意场景而非用户的实际环境；（2）迭代式生成方法（逐步用上一步输出作为输入）会累积误差和产生漂移；（3）缺乏大规模的分步视觉指令训练数据。

**核心矛盾**：要同时满足三个需求——（a）每步图像忠实于文字指令，（b）全序列与输入场景一致，（c）跨步骤的时间连贯性——而现有方法无法兼顾。

**本文目标**：（1）建立可扩展的分步视觉指令数据采集管线；（2）训练一个能一次性生成全序列的扩散模型，同时条件化于场景图像和每步文字指令。

**切入角度**：利用教学视频中解说与视觉演示的天然对齐关系，自动挖掘高质量的图像-文字对序列作为训练数据。

**核心 idea**：将分步视觉指令生成建模为条件视频扩散问题，每帧接收独立的文字条件，同时通过跨帧注意力保证场景一致性。

## 方法详解

### 整体框架

ShowHowTo 基于潜空间视频扩散模型（SVD），输入为用户提供的场景图像 $I_0$ 和 $n$ 条分步文字指令 $\{\tau_i\}_{i=0}^{n}$，输出 $n$ 张对应的指令图像。场景图像通过 VAE 编码后拼接到每帧噪声上，去噪过程中 U-Net 的空间和时间注意力层跨帧交互以保证一致性，交叉注意力层让每帧独立地关注对应的文字指令。

### 关键设计

1. **自动化数据集构建管线**:

    - 功能：从百万级网络教学视频中自动提取高质量的分步图像-文字对序列
    - 核心思路：四阶段流程——（1）用 WhisperX 高质量转录视频解说；（2）用 LLaMA-3.1 过滤非教学视频（如产品评测、vlog）；（3）用 LLM 从转录文本中提取结构化的分步指令及其时间区间；（4）在每个时间区间内用 DFN-CLIP 进行跨模态对齐，选出最佳代表帧，同时保证时间顺序的一致性。最终从百万视频中获得57.8万条序列、450万图像-文字对，涵盖25,026种 HowTo 任务。
    - 设计动机：手动标注成本高昂且无法扩展，现有数据集（WikiHow 手绘插图、HowToStep 粗粒度片段）质量不足。全自动管线是可扩展的关键。

2. **逐帧独立文字条件注入**:

    - 功能：使每帧图像精确对应其文字指令
    - 核心思路：不同于标准视频扩散模型使用单一全局文字提示，ShowHowTo 为序列中每帧 $i$ 独立注入其对应文字指令 $\tau_i$，通过 U-Net 的交叉注意力层实现。每帧独立条件化使模型能够在一次生成中同时处理"切蔬菜"和"煎肉"这样语义跨度很大的步骤。
    - 设计动机：消融实验表明，单一提示（拼接或摘要）的 Step Faithfulness 仅为0.20-0.21，而逐帧独立提示达到0.52，差异巨大。视觉指令序列中相邻步骤的语义变化远大于普通视频中的帧间变化。

3. **可变长度序列训练**:

    - 功能：支持生成任意长度的指令序列（1到15步）
    - 核心思路：训练时在不同 batch 中使用不同的序列长度（同一 batch 内保持一致以便计算效率）。如果数据集序列更长，随机选择起始帧并取连续 $k$ 帧。训练长度上限设为8帧，平衡了场景一致性和步骤准确性。
    - 设计动机：消融表明短序列训练（≤4帧）提高场景一致性但降低步骤准确性，而长序列训练（≤16帧）则相反。8帧上限取得了最佳平衡。连续采样比随机采样更好，因为保持了场景的连续变化。

### 损失函数 / 训练策略

基于 SVD 的标准去噪扩散损失。从 WebVid10M 预训练的检查点初始化，在自建数据集上微调整个 U-Net。场景图像 $I_0$ 通过 VAE 编码后在通道维与噪声拼接作为输入，同时通过独立的交叉注意力层全局条件化。

## 实验关键数据

### 主实验

| 方法 | Step Faith. | Scene Consist. | Task Faith. |
|---|---|---|---|
| InstructPix2Pix | 0.25 | 0.17 | 0.25 |
| GenHowTo | 0.49 | 0.13 | 0.27 |
| StackedDiffusion | 0.43 | 0.02 | 0.42 |
| **ShowHowTo** | **0.52** | **0.34** | **0.42** |
| 原始视频序列 | 0.50 | 1.00 | 0.56 |

ShowHowTo 在步骤忠实度上甚至超过了源视频序列（0.52 vs 0.50），同时在场景一致性上远超序列生成方法（StackedDiffusion 仅0.02）。

### 消融实验

| 文字条件类型 | Step Faith. | Scene Consist. | Task Faith. |
|---|---|---|---|
| 单提示（拼接） | 0.21 | 0.29 | 0.38 |
| 单提示（摘要） | 0.20 | 0.30 | 0.40 |
| **逐帧独立提示** | **0.52** | **0.34** | **0.42** |

| 训练数据 | Step Faith. | Scene Consist. | Task Faith. |
|---|---|---|---|
| WikiHow-VGSI | 0.55 | 0.12 | 0.30 |
| HowToStep | 0.39 | 0.33 | 0.29 |
| **ShowHowTo 数据集** | **0.52** | **0.34** | **0.42** |

### 关键发现

- 逐帧独立文字条件是效果最关键的设计，Step Faithfulness 提升了150%以上
- ShowHowTo 数据集显著优于 WikiHow 和 HowToStep，主要归因于更精准的帧选择和更干净的指令提取
- 用户研究中 ShowHowTo 在步骤和场景维度上42%的情况下优于源视频序列
- 在 WikiHow 零样本测试上也展现了强泛化能力

## 亮点与洞察

1. **数据集构建管线是最大贡献**：全自动、可扩展、高质量的57.8万条序列数据集为这个方向奠定了基础
2. **逐帧独立条件化的洞察**：视觉指令序列不是普通视频——帧间语义跳跃大，需要独立引导
3. **生成质量超过真实视频**：在步骤忠实度上超过源视频——因为真实视频中动作可能不可见或被遮挡
4. **应用价值广泛**：不仅服务于人类用户指导，也可以为机器人策略学习生成中间目标图像

## 局限与展望

- 模型难以在长序列中维持物体状态一致性（如已煮熟→又变回生的）
- 对罕见物体（如电子元件）可能生成物理不合理的配置
- 继承了底层视频扩散模型的固有限制（偶尔的模糊、伪影）
- 未来方向：引入物体状态跟踪、扩展到更多领域（装配、维修等）

## 相关工作与启发

- 与 GenHowTo 的关系：GenHowTo 也从教学视频提取数据，但用图像到图像的迭代方式生成，会累积误差
- 与 StackedDiffusion 的关系：StackedDiffusion 在 WikiHow 手绘插图上训练，场景一致性极差（0.02）
- AURORA 做局部编辑，保场景但丢步骤信息
- 启发：同时满足精确条件化和全局一致性是视觉生成的核心挑战

## 评分

- **新颖性**: 8/10 — 数据集管线的自动化设计和逐帧独立条件化都有创新
- **实验充分度**: 9/10 — 多数据集评测、用户研究、详尽消融
- **写作质量**: 9/10 — 问题定义清晰，贡献层次分明
- **价值**: 8/10 — 对视觉指令生成和机器人规划都有启发意义

<!-- RELATED:START -->

## 相关论文

- [OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)
- [OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching](../../ACL2025/image_generation/ozspeech_one-step_zero-shot_speech_synthesis_with_learned-prior-conditioned_flow.md)
- [SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion](swiftedit_lightning_fast_text-guided_image_editing_via_one-step_diffusion.md)
- [TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)
- [Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)

<!-- RELATED:END -->
