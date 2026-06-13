---
title: >-
  [论文解读] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos
description: >-
  [CVPR 2025][多模态VLM][视觉定位] VideoGLaMM 是一个视频大型多模态模型，通过双视觉编码器（空间+时间）、可调 V→L 和 L→V 适配器、以及时空像素解码器，实现了视频中的像素级细粒度视觉定位，同时构建了首个 38K 视频 grounded QA 数据集。 领域现状：图像领域的 grounded…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视觉定位"
  - "像素级标注"
  - "视频分割"
  - "多模态大模型"
  - "时空对齐"
---

# VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos

**会议**: CVPR 2025  
**arXiv**: [2411.04923](https://arxiv.org/abs/2411.04923)  
**代码**: [https://mbzuai-oryx.github.io/VideoGLaMM](https://mbzuai-oryx.github.io/VideoGLaMM)  
**领域**: 多模态VLM  
**关键词**: 视觉定位, 像素级标注, 视频分割, 多模态大模型, 时空对齐

## 一句话总结

VideoGLaMM 是一个视频大型多模态模型，通过双视觉编码器（空间+时间）、可调 V→L 和 L→V 适配器、以及时空像素解码器，实现了视频中的像素级细粒度视觉定位，同时构建了首个 38K 视频 grounded QA 数据集。

## 研究背景与动机

**领域现状**：图像领域的 grounded LMM（如 GLaMM）已经能把文本响应与像素级 mask 关联起来，但视频领域的 LMM（如 VideoChat、Video-ChatGPT）只能做全局理解和对话，无法将回答中提到的对象定位到具体像素。

**现有痛点**：(1) 现有 Video LMM 使用单投影层对齐视觉和语言，足以理解全局内容但无法捕捉局部对象细节；(2) 缺少带有像素级 mask 标注的视频指令微调数据集；(3) PG-Video-LLaVA 虽尝试视频 grounding，但组合了预训练模块无法端到端训练，缺乏细粒度时空建模能力。

**核心矛盾**：视频中的视觉定位需要同时理解空间（每帧内容）和时间（跨帧变化）信息，但现有架构要么只用图像编码器忽略时间，要么只用视频编码器丢失空间细节。

**本文目标**：构建一个端到端可训练的视频 LMM，能在生成文本回复的同时，为提及的每个实体输出时空一致的像素级分割 mask。

**切入角度**：图像和视频编码器提供互补信息——图像编码器给出局部空间细节，视频编码器给出全局时序语义。同时需要双向对齐：视觉→语言、语言→视觉。

**核心 idea**：双编码器 + 双向适配器 + 时空像素解码器的三位一体架构，配合首创的 grounded 视频 QA 数据集。

## 方法详解

### 整体框架

输入视频 $V \in \mathbb{R}^{T \times H \times W \times C}$ 分两路处理：图像编码器逐帧提取空间特征 $f_g$，视频编码器分段提取时序特征 $f_h$。两路特征分别经 V→L 适配器投射到 LLM 空间，与文本 token 拼接后送入 LLM。LLM 输出文本中包含 `<SEG>` token，其对应的 last-layer embedding 经 L→V 适配器投射到视觉空间，与 frame encoder 处理的多尺度帧特征一起送入时空像素解码器，输出最终的像素级 mask。

### 关键设计

1. **时空双编码器（Spatio-Temporal Dual Encoder）**:

    - 功能：分别提取帧级空间特征和视频级时序特征
    - 核心思路：图像编码器用预训练 CLIP ViT-L/14 (336×336) 逐帧处理，输出局部空间特征 $f_g$。视频编码器用 InternVideo-v2 (224×224) 分段采样处理，对视频按 K 个 segment 分组，每段 $s=T/K$ 帧，输出全局时序特征 $f_h$
    - 设计动机：消融实验表明只用图像编码器 mIoU=60.06、CLAIR=18.9，只用视频编码器 mIoU=64.62 但 CLAIR=26.5，双编码器分别捕捉局部和全局特征，取得最佳平衡（mIoU=62.34, CLAIR=28.2）

2. **双向适配器（V→L 和 L→V Adapters）**:

    - 功能：实现视觉-语言的双向对齐
    - 核心思路：V→L 适配器 $\mathcal{W}_g$ 和 $\mathcal{W}_h$ 分别将图像/视频特征投射到 LLM 空间，得到 $Z_g$ 和 $Z_h$，与文本 token $Z_{text}$ 拼接成 $\mathcal{Z} = [Z_g, Z_h, Z_{text}]$ 送入 LLM。L→V 适配器 $\mathcal{W}_p$ 将 LLM 输出的 `<SEG>` token embedding 投射到像素解码器空间
    - 设计动机：现有方法只有 V→L 单向对齐，无法将语言侧的丰富时空语义传回视觉侧做精确 mask 生成。双向适配器使得文本理解能反哺定位精度

3. **时空像素解码器**:

    - 功能：根据 LLM 的语言-视觉特征生成精细 mask
    - 核心思路：基于 SAM2 的编码器-解码器初始化。grounded frame encoder $\mathcal{P}$ 提取输入帧的多尺度视觉特征。L→V 适配器输出的 $e_{seg}^p$ 经 prompt encoder $\mathcal{H}$ 编码后作为 mask decoder $\mathcal{D}$ 的 prompt，结合帧特征 $\mathcal{P}(V)$ 预测 mask：$M = \mathcal{D}(\mathcal{P}(V), \mathcal{H}(e_{seg}^p))$。支持时空一致的分割
    - 设计动机：SAM2 本身具备跨帧传播能力，作为像素解码器的初始化可以直接利用其时序建模先验，比从零训练时空分割头高效得多

### 损失函数 / 训练策略

- 总损失：$\mathcal{L}_{total} = CE + \mathcal{L}_{masked}$，CE 为 LLM 文本生成的交叉熵损失，$\mathcal{L}_{masked}$ 为 mask 预测的 IoU 损失
- 训练策略：渐进训练。前 20 个 epoch 在图像/视频分割数据集上训练，20-30 epoch 引入 GCG 数据集。如需 referring segmentation，30-40 epoch 继续微调
- 冻结双编码器和像素解码器/frame encoder，可训练组件：V→L 适配器、L→V 适配器、LLM 的 LoRA 参数
- LLM：Phi3-Mini-3.8B；训练硬件：4x A100 40GB + DeepSpeed
- 额外训练数据：ADE20K, COCO-Stuff, RefCOCO 系列, LLaVA-Instruct-150k, ReasonSeg, GranDf, Ref-DAVIS17 等

## 实验关键数据

### 主实验

| 任务 | 指标 | VideoGLaMM | GLaMM+SAM2 | PG-Video-LLaVA |
|------|------|------------|------------|-----------------|
| GCG | mIoU | **62.34** | 28.60 | 24.03 |
| GCG | CIDEr | **0.59** | 0.15 | 0.01 |
| GCG | CLAIR | **28.2** | 22.9 | 15.0 |
| MeViS (Ref-VOS) | J&F | **45.15** | 38.66 | 18.87 |
| Ref-DAVIS-17 | J&F | **69.5** | - | - |
| VidSTG (VG) | mIoU | **39.66** | 38.63 | 34.20 |

| Ref-DAVIS-17 | 方法 | J | F | J&F |
|-------------|------|---|---|-----|
| | VideoGLaMM | **73.3** | **65.6** | **69.5** |
| | VideoLISA | 72.7 | 64.9 | 68.8 |
| | TrackGPT-13B | 70.4 | 62.7 | 66.5 |

### 消融实验

| 编码器配置 | mIoU | CLAIR | 说明 |
|-----------|------|-------|------|
| 仅图像编码器 | 60.06 | 18.9 | 空间好但时序差 |
| 仅视频编码器 | 64.62 | 26.5 | mIoU 最高 |
| 双编码器 | 62.34 | **28.2** | 空间+时间平衡 |

| 解码器配置 | mIoU | CLAIR | 说明 |
|-----------|------|-------|------|
| 仅空间解码器 | 59.68 | 26.7 | 缺少时序 context |
| 时空解码器 | **62.34** | **28.2** | 时空一致性更好 |

| 解码器帧数 | mIoU | METEOR | CLAIR | 说明 |
|-----------|------|--------|-------|------|
| 4 帧 | **63.82** | 0.094 | 27.2 | mask 好但对话弱 |
| 8 帧 | 62.34 | **0.103** | **28.2** | 对话质量更好 |

### 关键发现

- 在 GCG 任务上 mIoU 从 28.60 (GLaMM+SAM2) 跃升至 62.34，说明端到端训练远优于模块拼接
- 在 MeViS 运动引导分割上超越 VideoLISA（后者还需后处理），验证了时空建模的有效性
- 双编码器的互补性明显：图像编码器擅长空间细节但缺时序，视频编码器有时序但细节弱
- 时空像素解码器比纯空间版本提升约 3% mIoU
- 更多监督帧（8 vs 4）牺牲少量 mIoU 但显著提升对话质量

## 亮点与洞察

- **端到端架构设计思路清晰**：双编码器→双向适配器→时空解码器，每个组件解决一个明确的子问题。模块化但端到端可训练
- **数据集构建 pipeline 半自动化**很实用：利用 Gemini-Pro + GPT-4o + SAM 组合，从已有视频数据集中自动生成 grounded caption + mask，是一种可复制的标注范式
- **首次在视频 LMM 中实现像素级 grounded 对话**，填补了图像 GLaMM 到视频的空白
- SAM2 作为像素解码器初始化的选择非常聪明，直接复用其时序传播能力

## 局限与展望

- GCG 数据集虽有 38K 样本，但视频对主要来自短中时长片段，长视频理解未验证
- 数据标注含噪（部分由半自动 pipeline 生成），视频描述未穷举场景中所有实体
- 对不同粒度对象（大/小/细碎物体）的分割能力不均，可能因训练数据分布不平衡
- 不同粒度的 grounding 质量在定量上未细分评估
- 未来方向：长视频支持、更高质量密集标注、多粒度分割均衡

## 相关工作与启发

- **vs GLaMM**: 图像级 grounded LMM，直接扩展到视频效果差（需拼接 SAM2），说明图像→视频不是简单架构复用
- **vs PG-Video-LLaVA**: 模块拼接方法，使用音频转录辅助理解但非端到端，grounding 精度低（mIoU 24）
- **vs VideoLISA**: 仅做 referring segmentation 不做 GCG，且需后处理提升性能；VideoGLaMM 统一了三个任务
- **vs Video-ChatGPT/Video-LLaMA**: 只做全局对话无 pixel grounding 能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个视频像素级 grounded LMM，架构设计系统性强
- 实验充分度: ⭐⭐⭐⭐ 三任务评测 + 多维消融，但缺少长视频和细粒度分析
- 写作质量: ⭐⭐⭐⭐ 架构描述清晰，pipeline 图信息量足
- 价值: ⭐⭐⭐⭐ 数据集和模型对视频 grounding 社区有实际推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output](mimo_a_medical_vision_language_model_with_visual_referring_multimodal_input_and_.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)
- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](../../ICCV2025/multimodal_vlm/dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[CVPR 2025\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)

</div>

<!-- RELATED:END -->
