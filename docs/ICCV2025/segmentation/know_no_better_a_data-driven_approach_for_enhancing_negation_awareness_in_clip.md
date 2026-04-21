---
title: >-
  [论文解读] Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP
description: >-
  [ICCV 2025][图像分割][CLIP] 通过分析 CLIP 预训练数据中否定表达的稀缺和错位问题，设计两条基于 LLM/MLLM 的否定数据生成管线来微调 CLIP 文本编码器，开发出 NegationCLIP，在增强否定理解能力的同时保持通用性能，并提出 NegRefCOCOg 基准用于全面评估否定理解。
tags:
  - ICCV 2025
  - 图像分割
  - CLIP
  - 否定理解
  - 数据生成
  - 视觉-语言模型
  - 指代图像分割
---

# Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP

**会议**: ICCV 2025  
**arXiv**: [2501.10913](https://arxiv.org/abs/2501.10913)  
**代码**: [GitHub](https://github.com/parkquasar/NegationCLIP)  
**领域**: 图像分割  
**关键词**: CLIP, 否定理解, 数据生成, 视觉-语言模型, 指代图像分割

## 一句话总结

通过分析 CLIP 预训练数据中否定表达的稀缺和错位问题，设计两条基于 LLM/MLLM 的否定数据生成管线来微调 CLIP 文本编码器，开发出 NegationCLIP，在增强否定理解能力的同时保持通用性能，并提出 NegRefCOCOg 基准用于全面评估否定理解。

## 研究背景与动机

CLIP 作为视觉-语言模型的基石，被广泛用于文本到图像生成、指代图像分割等下游任务。然而 CLIP 存在一个严重但被忽视的缺陷——**无法正确理解否定**。例如，"parking" 和 "no parking" 对 CLIP 来说几乎没有区别。

本文通过系统实验揭示了这一问题的根源：

**CelebA 实验暴露问题**：在 40 个二元属性上设计正/负提示对（如 "wearing glasses" vs "not wearing glasses"），CLIP ViT-L/14 的平均平衡准确率仅为 60.8%——对于二分类任务，随机猜测就有 50%，这与 ImageNet 1000 类分类 73.4% 的准确率形成鲜明对比

**根因分析——训练数据缺失**：分析 LAION-400M 数据集发现，仅 0.70% 的 caption 包含否定词，否定词仅占总词汇的 0.08%。更关键的是，即使 caption 中包含否定（如 "no smoking"），往往与图像内容完全无关或错位（图像中可能恰恰有人在吸烟）。这是因为图像描述的本质是描述"看到了什么"，而非"没看到什么"

**现有方案不足**：CoN-CLIP 等先前工作生成否定 caption 后检索相似图像配对，但这种"先文字后图像"的方式导致否定 caption 与检索到的图像之间的语义对齐质量差

## 方法详解

### 整体框架

NegationCLIP 的训练流程分为两步：(1) 通过两条数据生成管线创建高质量的否定数据，(2) 冻结视觉编码器，仅微调文本编码器。这种设计保留了原始图像嵌入空间，使 NegationCLIP 可以即插即用地替换现有模型中的 CLIP 文本编码器。

### 关键设计

1. **管线一：基于物体缺失的否定生成（Pipeline 1 - Object Absence）**：利用已有的图像描述数据集（COCO）。三步流程：(a) 将 caption 提供给 LLM（Llama-3-8B），识别"在 caption 中未提及但合理可能存在的物体"（plausible objects）；(b) 使用 MLLM（LLaVA-1.6）验证这些物体确实**不**在图像中出现；(c) 让 LLM 将"该物体不存在"的否定信息自然地融入原始 caption 中。核心设计动机是"从图像出发"——先有图像，后生成与之**视觉对齐**的否定描述，而非反过来。这比随机选择物体进行否定更有效，因为上下文相关的否定才对模型学习有意义。

2. **管线二：扩展否定多样性（Pipeline 2 - Diversity Expansion）**：管线一仅覆盖物体存在性否定。为扩大否定类型范围，利用 VQA 数据集（VQAv2）中回答为"no"的问答对。这些问答涵盖了行为（"is the person running?"→"no"）、属性（"is the car red?"→"no"）等多样化否定。流程：(a) 筛选答案为"no"的三元组；(b) 让 LLM 将问题和否定答案融入原始 caption。总共生成 229K 否定数据（P1: 147K, P2: 82K）。

3. **NegRefCOCOg 基准**：现有否定评估基准存在严重偏见——CREPE Negate 和 CC-Neg 假设所有包含否定的 caption 都是错误的（即盲模型只需检测否定词就能"答对"）。NegRefCOCOg 基于 RefCOCOg 构建：(a) 筛选包含否定词的 prompt；(b) 确定正样本 $P^+$（与 prompt 匹配的目标区域）和硬负样本 $P^-$（同类物体但不符合否定描述的区域）；(c) 评估模型是否能正确将否定 prompt 匹配到 $P^+$ 而非 $P^-$。支持多种否定词（"no", "not", "without"）和否定位置（物体、动作、属性）。

### 损失函数 / 训练策略

使用标准 InfoNCE 损失微调文本编码器，学习率 $1e{-6}$，AdamW 优化器。视觉编码器完全冻结，仅更新文本编码器参数。这一设计确保：(1) 图像嵌入空间不变，微调后的文本编码器可直接替换原始编码器用于下游任务；(2) 训练高效，无需接触视觉侧参数。

## 实验关键数据

### 主实验

**否定理解与通用性能（多架构对比）**

| 模型 | 架构 | VALSE ↑ | NegRefCOCOg ↑ | ImageNet ↑ | COCO ↑ |
|------|------|---------|--------------|-----------|--------|
| CLIP | ViT-B/32 | 70.97 | 57.73 | 62.02 | 54.78 |
| CLIP-bnl | ViT-B/32 | 76.78 | 62.05 | 53.33 | 55.47 |
| CoN-CLIP | ViT-B/32 | 71.72 | 55.45 | 63.08 | 55.66 |
| **NegationCLIP** | ViT-B/32 | **80.15** | **64.09** | 60.97 | **68.00** |
| CLIP | ViT-L/14 | 66.85 | 57.27 | 73.44 | 59.99 |
| **NegationCLIP** | ViT-L/14 | **79.59** | **62.95** | 73.91 | **72.77** |

NegationCLIP 在否定基准上全面领先 9-13 个点，同时在 ImageNet 和 COCO 上保持甚至超过原始 CLIP 的性能。

### 消融实验

**数据配置对否定理解的影响（ViT-B/32）**

| 数据配置 | VALSE ↑ | NegRefCOCOg ↑ | 说明 |
|---------|---------|--------------|------|
| Original CLIP | 70.97 | 57.73 | 基线 |
| + Rand-P1（随机物体） | 73.78 | 62.05 | 随机否定有一定帮助 |
| + P1（合理物体） | **80.15** | 63.18 | 上下文相关否定更有效 |
| + P2（VQA 否定） | 76.78 | 64.32 | 多样化否定类型 |
| + **P1 + P2** | **80.15** | **64.09** | 两条管线互补，最佳组合 |

Rand-P1 consistently 低于 P1，验证了"合理物体选择"的重要性。P1 + P2 在 NegRefCOCOg 上最优，说明 NegRefCOCOg 比 VALSE 更好地测试了否定表达的多样性。

**下游任务验证**

| 模型 | PhraseCut mIoU | RefCOCOg(Neg) mIoU |
|------|--------------|-------------------|
| CLIPSeg | 0.562 | 0.267 |
| CoN-CLIPSeg | 0.539 | 0.123 |
| **NegationCLIPSeg** | 0.561 | **0.288** |

| 模型 | T2I TIFA ↑ | T2I Neg Score ↑ |
|------|-----------|----------------|
| SD-1.4 | 0.786 | 0.295 |
| SD-1.4 + CoN-CLIP | 0.783 | 0.243 |
| **SD-1.4 + NegationCLIP** | **0.790** | **0.449** |

### 关键发现

- **数据质量 > 数据量**：229K 张高质量否定数据即可显著改善 CLIP 的否定理解
- **"从图像出发"策略**：NegationCLIP 先从图像生成匹配的否定 caption，优于 CoN-CLIP 的"先 caption 后检索图像"策略
- **插拔式增强**：仅替换文本编码器即可改善 T2I 生成的否定能力（Neg Score 从 0.295 提升至 0.449）和 RIS 的否定分割
- **否定基准偏差**：现有基准（CREPE, CC-Neg）对否定有偏见，盲模型可轻松作弊，NegRefCOCOg 通过硬负样本设计解决了这一问题
- **CoN-CLIP 反而有害**：在某些任务上 CoN-CLIP 的性能甚至低于原始 CLIP，源于其否定 caption 与检索图像的错位

## 亮点与洞察

- **问题定义精准**：通过 CelebA 实验和 LAION-400M 数据分析，清晰地揭示了 CLIP 否定理解缺陷的根因
- **实用性极强**：NegationCLIP 作为即插即用的文本编码器，无需修改下游模型即可提升否定处理能力
- **评估基准设计考究**：NegRefCOCOg 避免了现有基准的偏见问题，支持多样化否定类型
- **跨任务验证**：在图像-文本匹配、T2I 生成、指代图像分割三个不同任务上都验证了有效性

## 局限与展望

- 仅微调文本编码器，视觉编码器未做适配，可能限制了更深层的否定语义理解
- 否定数据生成依赖 LLM/MLLM 的能力，存在生成噪声和成本问题
- SDXL 双文本编码器场景下 TIFA 略有下降，说明替换策略在复杂架构下需进一步优化
- 未探索其他否定类型（如双重否定、隐性否定）
- 数据量（229K）相比 LAION-400M 仍很小，更大规模的否定数据能否带来更多增益值得探索
- 可将此方法推广到视频 VLM 和多语言场景

## 相关工作与启发

- CLIP-bnl 和 CoN-CLIP 是最直接的先前工作，但受限于数据生成方式（先文本后图像 vs 先图像后文本）
- 硬否定挖掘在 NLP 中有广泛应用（BERT 的否定理解研究），本文将这一思路迁移到多模态领域
- 否定理解对 AI 安全至关重要（如 "no weapons" vs "weapons" 的区分），本文的方法具有实际安全意义

## 评分

- **新颖性**: ⭐⭐⭐⭐ 数据驱动的否定增强方法清晰有效，NegRefCOCOg 基准设计精巧
- **实验充分度**: ⭐⭐⭐⭐⭐ 4 种架构、3 个任务全面验证，消融实验细致
- **写作质量**: ⭐⭐⭐⭐⭐ 问题分析深入透彻，从根因到方案再到评估的逻辑链完整
- **价值**: ⭐⭐⭐⭐⭐ 解决了 CLIP 的关键盲点，即插即用特性使其具有广泛实用价值

<!-- RELATED:START -->

## 相关论文

- [Enhancing Transformers Through Conditioned Embedded Tokens](enhancing_transformers_through_conditioned_embedded_tokens.md)
- [Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation](know_your_attention_maps_class-specific_token_masking_for_weakly_supervised_sema.md)
- [A Plug-and-Play Physical Motion Restoration Approach for In-the-Wild High-Difficulty Motions](a_plug-and-play_physical_motion_restoration_approach_for_in-.md)
- [DeBias-CLIP: CLIP Is Shortsighted — Paying Attention Beyond the First Sentence](../../CVPR2026/segmentation/clip_shortsighted_beyond_first_sentence.md)
- [SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree](sam2long_enhancing_sam_2_for_long_video_segmentation_with_a.md)

<!-- RELATED:END -->
