---
title: >-
  [论文解读] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes
description: >-
  [CVPR 2025][图像分割][音视频分割] SAM2-LOVE 通过设计多模态融合 Transformer 将文本、音频、视觉三模态信息压缩为可学习 token 来提示 SAM2，结合 token 传播与累积策略增强时空一致性，在 Ref-AVS 基准上以 $\mathcal{J\&F}$ 58.5% 的成绩超越 SOTA（EEMC）8.5个百分点。
tags:
  - CVPR 2025
  - 图像分割
  - 音视频分割
  - SAM2
  - 多模态融合
  - 参考分割
  - 时空一致性
---

# SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes

**会议**: CVPR 2025  
**arXiv**: [2506.01558](https://arxiv.org/abs/2506.01558)  
**代码**: https://github.com/yuji-wang/SAM2-LOVE  
**领域**: 分割  
**关键词**: 音视频分割, SAM2, 多模态融合, 参考分割, 时空一致性

## 一句话总结

SAM2-LOVE 通过设计多模态融合 Transformer 将文本、音频、视觉三模态信息压缩为可学习 token 来提示 SAM2，结合 token 传播与累积策略增强时空一致性，在 Ref-AVS 基准上以 $\mathcal{J\&F}$ 58.5% 的成绩超越 SOTA（EEMC）8.5个百分点。

## 研究背景与动机

**领域现状**：参考音视频分割（Ref-AVS）是一个新兴任务，要求模型根据文本表达和音频信号从视频中持续分割目标物体——这需要在"语言辅助的音视频场景"（LAVS）中实现像素级场景理解。现有方法分为双模态方法（文本-视觉或音频-视觉）和三模态方法（EEMC）。

**现有痛点**：(1) 双模态方法由于缺少第三模态信息，无法准确定位目标——文本-视觉方法（EVF-SAM）在静音场景中无法区分发声物体，音频-视觉方法（GAVS）无法理解文本中的动态控制信号；(2) 现有三模态方法 EEMC 虽然同时建模三模态，但时空一致性不足——尽管有 memory cache，模型仍然无法持续追踪目标位置和形状，导致分割区域随时间漂移。

**核心矛盾**：Ref-AVS 任务要求同时具备三模态理解能力和视频级时空一致性。EEMC 在三模态理解上表现不错，但缺乏强大的视频追踪能力。而 SAM2 拥有强大的视频分割和追踪能力，但缺乏文本和音频理解能力。

**本文目标**：将 SAM2 的强大视频分割能力与三模态理解整合，实现 LAVS 中的像素级理解。

**切入角度**：SAM2 遵循"提示-传播"范式——在关键帧用提示定位目标，然后传播到全视频。关键在于如何将三模态信息压缩为有效的提示来驱动 SAM2。

**核心 idea**：设计融合 Transformer 将三模态信息压缩到可学习的 [seg] token 中，用该 token 提示 SAM2 的第一帧，然后利用 SAM2 的零样本 VOS 能力传播到全视频。通过 token 传播（前向知识传递）和 token 累积（后向知识传递）策略增强时空理解。

## 方法详解

### 整体框架

SAM2-LOVE 的 pipeline 包含三个主要组件：(1) **多模态编码器**：VGGish 编码音频、ViT 编码视频帧、DistilRoBERTa 编码文本，各经 MLP 投影到统一维度；(2) **多模态融合 Transformer**：6层双向 Transformer，将三模态序列和可学习 [seg] token 融合，输出压缩后的多模态表示；(3) **SAM2**：以 [seg] token 作为提示在第一帧定位目标，通过 memory attention 传播到全视频。训练时仅在第一帧计算损失，推理时 SAM2 处理整个视频序列。

### 关键设计

1. **多模态融合模块**:

    - 功能：将文本、音频、视觉三模态信息压缩为单个可学习 token 来提示 SAM2
    - 核心思路：定义可学习的 [seg] token，将其前置于多模态序列：$F_M^i = \text{Concat}([[seg]; \hat{F}_A; [aud]; \hat{F}_T; [vis]; \hat{F}_V^i])$。其中 [aud] 和 [vis] 是固定的模态指示 token。该序列送入6层双向 Transformer encoder，[seg] token 通过自注意力与所有模态信息交互，输出时取第一个元素作为更新后的 [seg] 嵌入，包含了压缩的三模态信息
    - 设计动机：SAM2 的提示接口设计为接受稀疏提示（点/框/掩码），单个 token 是最自然的适配方式。双向注意力使 [seg] 能同时感知所有模态的信息

2. **Token 传播策略（前向知识传递）**:

    - 功能：使 [seg] token 在视频帧间传播，自适应捕获帧内空间特征和帧间连续性
    - 核心思路：处理第 $i$ 帧时，使用上一帧输出的 [seg] token 作为当前帧的输入。音频和文本特征保持不变，仅替换视觉特征为当前帧。这样 [seg] token 逐帧传播，不断积累时序信息。最终使用全视频传播完毕的 [seg] token 提示 SAM2 的第一帧
    - 设计动机：常规做法是对每帧独立融合三模态信息，但这忽略了帧间的时序关系。通过传播，token 可以建模视频的时空动态

3. **Token 累积策略（后向知识传递）**:

    - 功能：防止 [seg] token 在长视频传播过程中遗忘早期帧的信息
    - 核心思路：维护历史 token 序列 [his]，每处理完一帧就将该帧 ViT 的全局 [cls] token 追加到 [his] 中：$[his]^i = \text{Concat}([[cls]^0; [cls]^1; ...; [cls]^i])$。在处理第 $i+1$ 帧时，[his] 被附加到 $F_M^{i+1}$ 中，使 [seg] 可以通过注意力回顾所有历史帧的全局表示
    - 设计动机：随着传播次数增加，早期帧信息会被稀释。累积策略提供了一种"回放"机制，与传播形成互补——传播是前向知识传递，累积是后向知识传递

### 损失函数 / 训练策略

- **损失**：$\mathcal{L}_{mask} = \lambda_{bce} \text{BCE}(\hat{M}, M) + \lambda_{dice} \text{DICE}(\hat{M}, M)$，$\lambda_{bce}=\lambda_{dice}=1.0$
- **训练策略**：融合模块+[seg] token 访问完整视频序列，SAM2 仅在第一帧接受监督。推理时 SAM2 接收全视频利用零样本 VOS 能力
- **优化**：DeepSpeed ZeRO-2, AdamW, lr=1e-4, batch=8, grad accumulation=2, 2×A100
- **冻结策略**：音频/视觉/文本编码器 + SAM2 图像编码器/memory attention 冻结；融合 Transformer + SAM2 prompt encoder/mask decoder 可训练

## 实验关键数据

### 主实验

| 方法 | Seen $\mathcal{J\&F}$ | Unseen $\mathcal{J\&F}$ | Mix $\mathcal{J\&F}$ |
|------|----------------------|------------------------|---------------------|
| GAVS+text | 39.4 | 39.8 | 39.6 |
| ReferFormer+audio | 40.7 | 39.6 | 40.2 |
| EEMC | 42.8 | 57.2 | 50.0 |
| **SAM2-LOVE** | **47.7** | **69.4** | **58.5** |
| vs EEMC | +4.9 | +12.2 | **+8.5** |

### 消融实验

| 设计 | Seen $\mathcal{J\&F}$ | Unseen $\mathcal{J\&F}$ |
|------|----------------------|------------------------|
| CLIP编码器(无[cls]累积) | 46.8 | 68.2 |
| RoBERTa+ViT(无[cls]累积) | 46.7 | 69.0 |
| RoBERTa+ViT(有[cls]累积) | **47.7** | **69.4** |
| 1层Transformer | 45.4 | 68.1 |
| 6层Transformer | 46.7 | 69.0 |
| 12层Transformer | 46.4 | 70.5 |

### 关键发现

- SAM2 的零样本 VOS 能力是性能飞跃的关键——在 Unseen 分类上提升12.2%，远超 Seen 的4.9%
- RoBERTa 比 CLIP 文本编码器更适合 LAVS——因为 Ref-AVS 的表达是控制信号式的（如"发出声音的那个"），而非具体语义（如"猫"）
- [cls] token 累积策略几乎无额外计算开销但稳定提升性能
- 单个 [seg] token 就足够有效，增加到4/8个时 Seen 下降而 Unseen 上升——符合"额外 token 作为 register 存储全局信息利于新任务但可能伤害已知任务"的洞察

## 亮点与洞察

- **设计简洁有效**：核心创新就是"融合成 token + 提示 SAM2"，没有复杂的多模态融合架构
- **知识传递的统一视角**：将 token 传播解读为前向知识传递、token 累积解读为后向知识传递，概念优雅
- **训练-推理不对称**：训练只用第一帧监督，推理时 SAM2 零样本传播全视频，充分利用了 SAM2 的预训练能力
- **在 Unseen 类别上提升巨大**（+12.2%），说明 SAM2 的泛化能力被有效激活

## 局限与展望

- NULL 设置（表达指向不存在的物体）表现较差（0.23），说明模型难以判断"无目标"场景
- 训练只在第一帧监督，可能导致对非首帧的误差累积
- 音频编码器 VGGish 较旧，可能限制了音频理解能力
- 可探索更强的音频编码器（如 AudioMAE）和端到端训练 SAM2 图像编码器

## 相关工作与启发

- **EEMC**：之前唯一的三模态 Ref-AVS 方法，同时建模三模态但时空一致性差
- **EVF-SAM**：将 SAM 扩展到文本提示的工作，但仅限图像且缺少音频
- **GAVS**：将 SAM 扩展到音频-视觉的工作，但缺少文本理解
- **SAM2**：强大的视频分割基础，本文将其扩展到三模态场景

## 评分

- **新颖性**: 7/10 — 核心思路（压缩到 token 提示 SAM2）较直觉，但 token 传播/累积策略有新意
- **实验充分度**: 8/10 — 在 Ref-AVS 基准上全面评测，消融覆盖各组件，但仅一个基准数据集
- **写作质量**: 7/10 — 方法描述清晰，但部分符号和概念可更精练
- **价值**: 8/10 — 大幅推进了 Ref-AVS 任务的 SOTA，展示了 SAM2 在多模态场景中的潜力

<!-- RELATED:START -->

## 相关论文

- [Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [A Distractor-Aware Memory for Visual Object Tracking with SAM2](a_distractor-aware_memory_for_visual_object_tracking_with_sam2.md)
- [OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](../../ICCV2025/segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)
- [EdgeTAM: On-Device Track Anything Model](edgetam_on-device_track_anything_model.md)
- [Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics](dynamic_derivation_and_elimination_audio_visual_segmentation_with_enhanced_audio.md)

<!-- RELATED:END -->
