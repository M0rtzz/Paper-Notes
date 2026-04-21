---
title: >-
  [论文解读] StoryGPT-V: Large Language Models as Consistent Story Visualizers
description: >-
  [CVPR 2025][图像分割][故事可视化] 本文提出StoryGPT-V，通过两阶段训练——先训练角色感知的潜在扩散模型（Char-LDM）实现高质量角色生成，再将LLM输出与Char-LDM输入空间对齐实现指代消解和上下文一致性——在故事可视化任务上生成准确、高质量且时间一致的角色图像，内存消耗低。
tags:
  - CVPR 2025
  - 图像分割
  - 故事可视化
  - 大语言模型
  - 潜在扩散模型
  - 指代消解
  - 角色一致性
---

# StoryGPT-V: Large Language Models as Consistent Story Visualizers

**会议**: CVPR 2025  
**arXiv**: [2312.02252](https://arxiv.org/abs/2312.02252)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 故事可视化, 大语言模型, 潜在扩散模型, 指代消解, 角色一致性

## 一句话总结

本文提出StoryGPT-V，通过两阶段训练——先训练角色感知的潜在扩散模型（Char-LDM）实现高质量角色生成，再将LLM输出与Char-LDM输入空间对齐实现指代消解和上下文一致性——在故事可视化任务上生成准确、高质量且时间一致的角色图像，内存消耗低。

## 研究背景与动机

故事可视化（Story Visualization）是一个比单张图片生成更复杂的任务：它需要根据一系列叙事描述生成多帧图像，且要求角色和背景保持跨帧一致性。该任务面临两个核心挑战：

**挑战一：角色生成质量**。现有文生图模型虽能生成高质量单帧图像，但在故事可视化中难以精确生成特定角色。单纯的文本提示不足以提供角色外观的详细信息。

**挑战二：指代消解（Anaphora Resolution）**。故事描述中频繁使用代词（"他"、"她"、"他们"），模型需要从上下文推断代词指代的具体角色。Story-LDM虽首次引入指代消解，但其注意力记忆模块在CLIP空间中交互，丢失了细粒度语言理解能力，且需要保持所有先前帧的像素级表征，导致内存需求随帧数急剧增长。

本文的核心动机是：**利用LLM强大的推理能力来解决指代消解问题，同时将视觉信息压缩为token级表征来降低内存消耗**。通过LLM的因果语言建模能力，模型可以从交错输入的图像和文本上下文中隐式推断代词指代。

## 方法详解

### 整体框架

StoryGPT-V采用两阶段训练：
- **第一阶段**：训练角色感知的潜在扩散模型（Char-LDM），通过融合角色视觉特征到文本嵌入，并用角色分割掩码引导交叉注意力图，提升角色生成的准确性和保真度
- **第二阶段**：将LLM（OPT-6.7B或Llama2-7B）的输出与Char-LDM的输入空间对齐，使LLM能接收交错的图像-文本输入并产生视觉输出，利用因果语言建模实现指代消解

### 关键设计

1. **角色感知融合嵌入（Character-Augmented Fused Embedding）**:
    - 功能：将角色视觉特征融合到文本嵌入中，为扩散模型提供角色外观信息
    - 核心思路：对文本描述中每个角色名token，将其CLIP文本嵌入与对应角色图像的CLIP视觉嵌入拼接后通过MLP映射：$c^k = \text{MLP}(\text{concat}(\psi(S[i_c^k]), \phi(I_c^k)))$
    - 设计动机：纯文本描述（如"Fred"）无法传达角色的视觉外观细节，融合视觉特征后，扩散模型在去噪过程中能获得更丰富的角色信息

2. **交叉注意力掩码引导（Cross-Attention Control with Segmentation Masks）**:
    - 功能：引导扩散模型中角色token的注意力集中到对应角色的空间区域
    - 核心思路：使用SAM获取角色分割掩码 $M_k$，设计正则化损失 $\mathcal{L}_{reg} = \frac{1}{K}\sum_{k=1}^K (A_k^- - A_k^+)$，其中 $A_k^+$ 是角色区域内的平均注意力，$A_k^-$ 是角色区域外的平均注意力
    - 设计动机：标准LDM中一个像素可以无约束地与所有文本token交互，导致角色生成位置和外观不精确。掩码引导强制角色token主要影响其对应的空间区域

3. **LLM对齐用于指代消解（LLM Alignment for Reference Resolution）**:
    - 功能：利用LLM的推理能力隐式解析代词指代，并生成与Char-LDM输入空间对齐的视觉输出
    - 核心思路：LLM接受交错的(I₁, S₁, .., I_{n-1}, S_{n-1}, S_n)作为输入，图像通过CLIP编码后映射为4个token嵌入。LLM生成R个[IMG] token，通过Transformer-based Mapper投影到Char-LDM的输入空间，训练目标为对齐损失 $\mathcal{L}_{align} = \|\text{Mapper}_{LDM}(h_{[IMG_{1:R}]}, q_1,...q_L) - c_i\|_2^2$，其中 $c_i$ 是非指代文本的融合嵌入
    - 设计动机：LLM通过因果建模和上下文记忆能力，从"They are talking"中推断出"Fred and Wilma are talking"，并生成包含正确角色信息的视觉条件

### 损失函数 / 训练策略

**第一阶段**：
- 标准扩散损失 + 注意力正则化损失 $\mathcal{L}_{reg}$
- 训练策略：10%无条件训练（classifier-free guidance）、10%纯文本训练、80%角色增强融合训练
- 冻结CLIP文本编码器，微调其余模块，25k步，lr=1e-5

**第二阶段**：
- Token生成损失 $\mathcal{L}_{gen}$（[IMG] token的NLL）+ 对齐损失 $\mathcal{L}_{align}$
- 使用OPT-6.7B或Llama2-7B，冻结LLM主体，训练Mapper和额外token嵌入
- 预计算非指代融合嵌入作为对齐目标

## 实验关键数据

### 主实验

FlintstonesSV数据集（含指代文本）：

| 模型 | Char-Acc↑ | Char-F1↑ | BG-Acc↑ | FID↓ | BLEU4↑ | CIDEr↑ |
|------|-----------|----------|---------|------|--------|--------|
| StoryDALL-E | 61.83 | 78.36 | 48.10 | 44.66 | 0.4460 | 1.3373 |
| LDM | 75.37 | 87.54 | 52.57 | 32.36 | 0.4911 | 1.5103 |
| Story-LDM | 77.23 | 88.26 | 54.97 | 36.34 | 0.4585 | 1.4004 |
| **StoryGPT-V** | **87.96** | **94.17** | **56.01** | **21.71** | **0.5070** | **1.6607** |

PororoSV数据集（含指代文本）：

| 模型 | Char-Acc↑ | Char-F1↑ | FID↓ | BLEU4↑ | CIDEr↑ |
|------|-----------|----------|------|--------|--------|
| StoryDALL-E | 21.03 | 50.56 | 40.39 | 0.2295 | 0.3666 |
| Story-LDM | 29.14 | 57.56 | 26.64 | 0.2420 | 0.4581 |
| **StoryGPT-V** | **36.06** | **62.70** | **19.56** | **0.2586** | **0.5279** |

### 消融实验

第一阶段消融（FlintstonesSV，无指代文本）：

| 配置 | Char-Acc↑ | FID↓ | 说明 |
|------|-----------|------|------|
| w/o $\mathcal{L}_{reg}$ | 88.86 | 23.51 | 无掩码引导 |
| w/o augmented text | 87.45 | 21.27 | 无角色视觉增强 |
| freeze vis | 88.67 | 22.01 | 冻结视觉编码器 |
| Default (w/ img) | **90.36** | 21.13 | 完整Char-LDM |

第二阶段消融：

| 配置 | Char-Acc↑ | FID↓ | 说明 |
|------|-----------|------|------|
| Caption-Emb_text | 69.70 | 21.32 | 仅文本输入+文本嵌入对齐 |
| Interleave-Emb_text | 86.10 | 21.30 | 交错输入+文本嵌入对齐 |
| Interleave-Emb_fuse | **87.96** | 21.71 | 交错输入+融合嵌入对齐 |

### 关键发现

- 交错图文输入（Interleave）相比纯文本输入（Caption）带来巨大提升（+18.26% Char-Acc），说明视觉上下文对指代消解至关重要
- 融合嵌入（Emb_fuse）相比纯文本嵌入（Emb_text）进一步提升准确率，证明角色视觉信息的融合有效
- 长序列（40+帧）生成时，Story-LDM内存超限(80G A100上42帧)，StoryGPT-V可超过50帧
- 更强的LLM骨干（Llama2-7B vs OPT-6.7B）带来一致性提升（Char-Acc: 89.08% vs 87.96%）

## 亮点与洞察

- **高效的上下文记忆**：将视觉信息压缩为4个token嵌入（$n \times 4 \times d$），而非Story-LDM的像素级表征（$n \times h \times w \times d$），内存效率提升数量级
- **两阶段解耦设计**：第一阶段专注角色生成质量，第二阶段专注指代消解，分工清晰，便于独立优化
- **掩码引导的注意力控制**：通过分割掩码显式引导交叉注意力，比纯隐式学习更高效地建立角色token与空间区域的对应关系
- **多模态生成能力**：模型不仅能根据文本生成图像，还能自主生成文本续写故事并配合生成图像
- **人类评估一致验证**：MTurk评估中StoryGPT-V在视觉质量、文本对齐、角色准确性和时间一致性上全面优于Story-LDM

## 局限与展望

- 当前仅在FlintstonesSV和PororoSV两个卡通数据集上验证,未扩展到更真实的故事可视化场景
- 对于角色数量多或角色外观差异小的场景，准确率可能下降
- 推理时需要角色参考图像，这限制了完全开放式的故事生成
- 第二阶段的对齐损失为L2距离，可能不是语义对齐的最优度量
- 未来可探索更大规模LLM、扩展到视频连续生成、以及与更先进扩散模型（如SDXL）的结合

## 相关工作与启发

- **vs Story-LDM**: Story-LDM使用注意力记忆模块在像素空间保持上下文，内存消耗大且指代消解能力有限；StoryGPT-V利用LLM在token空间高效处理上下文
- **vs StoryDALL-E**: StoryDALL-E需要额外的源帧输入，且不处理指代文本；StoryGPT-V通过LLM自动处理指代
- **vs GILL/Emu等多模态生成模型**: 这些通用模型不针对故事可视化优化，缺乏角色一致性保证；StoryGPT-V通过Char-LDM专门优化角色生成

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将LLM推理能力与角色感知扩散模型结合用于故事可视化指代消解
- 实验充分度: ⭐⭐⭐⭐ 定量、定性、人类评估、消融实验和长序列分析全面
- 写作质量: ⭐⭐⭐ 整体清晰但公式排版稍显拥挤，部分细节需看补充材料
- 价值: ⭐⭐⭐⭐ 为故事可视化任务提供了高效实用的解决方案，高效的上下文记忆机制可推广到其他序列生成任务

<!-- RELATED:START -->

## 相关论文

- [F-LMM: Grounding Frozen Large Multimodal Models](f-lmm_grounding_frozen_large_multimodal_models.md)
- [Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)](assessing_and_learning_alignment_of_unimodal_vision_and_language_model.md)
- [GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation](glus_global-local_reasoning_unified_into_a_single_large_language_model_for_video.md)
- [VISA: Reasoning Video Object Segmentation via Large Language Models](../../ECCV2024/segmentation/visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [MatAnyone: Stable Video Matting with Consistent Memory Propagation](matanyone_stable_video_matting_with_consistent_memory_propagation.md)

<!-- RELATED:END -->
