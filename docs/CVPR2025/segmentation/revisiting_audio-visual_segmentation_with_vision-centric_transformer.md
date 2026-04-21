---
title: >-
  [论文解读] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer
description: >-
  [CVPR 2025][图像分割][音频-视觉分割] 本文提出以视觉为中心的 Transformer（VCT）框架来解决音频-视觉分割任务，用从视觉特征衍生的 query 替代传统的音频衍生 query，配合原型提示查询生成模块（PPQG），在 AVSBench 三个子集上达到新 SOTA，尤其在最具挑战性的 AVSS 子集上取得显著提升。
tags:
  - CVPR 2025
  - 图像分割
  - 音频-视觉分割
  - Transformer
  - 原型提示
  - query设计
  - 发声物体分割
---

# Revisiting Audio-Visual Segmentation with Vision-Centric Transformer

**会议**: CVPR 2025  
**arXiv**: [2506.23623](https://arxiv.org/abs/2506.23623)  
**代码**: https://github.com/spyflying/VCT_AVS  
**领域**: 图像分割 / 多模态  
**关键词**: 音频-视觉分割, 视觉中心Transformer, 原型提示, query设计, 发声物体分割

## 一句话总结

本文提出以视觉为中心的 Transformer（VCT）框架来解决音频-视觉分割任务，用从视觉特征衍生的 query 替代传统的音频衍生 query，配合原型提示查询生成模块（PPQG），在 AVSBench 三个子集上达到新 SOTA，尤其在最具挑战性的 AVSS 子集上取得显著提升。

## 研究背景与动机

**领域现状**：音频-视觉分割（AVS）旨在利用视频的音频信号，对画面中发声物体进行像素级分割。现有主流方法采用音频中心 Transformer 架构——将音频特征作为或融入 object query，通过 Transformer decoder 层层交互来定位发声物体。代表方法包括 COMBO、AQFormer、CATR 等。

**现有痛点**：音频中心 Transformer 存在两个根本限制。（1）**感知模糊性**：真实场景中的音频通常是多个声源的混合，包括画面内外的声音。例如一段音频同时包含人声、吉他声和画面外的汽车声，基于这种混合音频衍生的 query 相互干扰，难以区分不同的发声物体，且画面外的噪声可能导致假阳性预测。（2）**密集预测能力减弱**：AVS 本质上是一个视觉中心的密集预测任务，query 需要同时包含抽象的音频语义（判断物体是否发声）和具体的视觉细节（精确勾勒轮廓）。但音频衍生的 query 初始只有音频语义，视觉信息的延迟整合导致细节丢失。

**核心矛盾**：音频信号的混合性和视觉密集预测的精细性之间存在矛盾。从混合音频出发去寻找视觉中的发声物体，不如从视觉区域出发去匹配对应的声音信息来得直接和准确。

**本文目标**：重新设计 AVS 的 query 机制，将视觉信息置于中心地位，让 query 自然地包含丰富视觉细节的同时逐步获取音频语义。

**切入角度**：将 query 从音频域迁移到视觉域——每个 query 初始聚焦于图像的不同区域，通过与音频和视觉特征的多层交互逐步变得"音频感知"。这样每个 query 能独立地从混合音频中提取自己对应的声音信息，避免干扰。

**核心 idea**：用视觉衍生的 query 替代音频衍生的 query，配合音频原型提示和像素上下文分组，实现更准确的发声物体区分和轮廓描绘。

## 方法详解

### 整体框架

给定 T 帧视频和音频片段，首先用视觉编码器（Swin Transformer）提取多尺度视觉特征 $\{V_i\}_{i=2}^5$，用 VGGish 提取音频特征 $A \in \mathbb{R}^{T \times S \times C^a}$。将最大分辨率视觉特征 $V_2$ 送入 PPQG 模块生成 N 个视觉衍生 query。这些 query 在迭代音频-视觉 Transformer decoder 中与音频特征和多尺度视觉特征交替交互，最终通过分类头和 mask 头输出分割结果。

### 关键设计

1. **原型提示查询生成模块（PPQG）**:

    - 功能：生成既包含丰富视觉细节又具有音频语义感知能力的视觉衍生 query
    - 核心思路：三步生成。**第一步（视觉嵌入聚合）**：将高分辨率特征 $V_2$ 通过卷积层和 MLP 进行投影和空间信息聚合，得到 N 个视觉嵌入 $V^e \in \mathbb{R}^{N \times C^h}$。**第二步（音频原型提示）**：定义 K 个可学习的音频原型 $P \in \mathbb{R}^{K \times C^h}$（K 为音频事件类别数），通过 cross-attention 将音频类别先验注入视觉嵌入：$\bar{V}^e = V^e + \text{Softmax}(\frac{(V^e W_1^q)(P W_1^k)^T}{\sqrt{C^h}})(P W_1^v)$。同时设计原型-音频对比损失 $\mathcal{L}_{pac}$ 确保原型学习到正确的音频语义。**第三步（像素上下文分组）**：使用 Gumbel-Softmax 实现硬且可微的分配，将图像像素上下文分组到各 query 中，使其聚焦不同图像区域
    - 设计动机：音频原型提示让 query 在进入 decoder 之前就知道场景中可能出现哪些声音事件，从而在后续交互中更有针对性地提取音频信息。Gumbel-Softmax 硬分配确保不同 query 聚焦不同区域，增强可区分性

2. **迭代音频-视觉 Transformer Decoder**:

    - 功能：让视觉衍生 query 逐步获取对应的声音信息和精细视觉特征
    - 核心思路：decoder 由交互单元 $\mathcal{U} = \{A_t, V_5, V_4, V_3\}$ 重复 D 次构成。每个单元包含一个**音频信息提取 block**（query 与当前帧音频特征做 cross-attention，音频作为 key/value）和三个**视觉信息增强 block**（query 依次与 $V_5, V_4, V_3$ 做 cross-attention）。音频 block 让每个 query 获取其代表区域的声音信息；视觉 block 捕获更精细的视觉特征以精确预测 mask。遵循 Mask2Former 的做法，用上一层预测的 mask 作为当前层的 attention mask
    - 设计动机：视觉衍生 query 聚焦不同视觉区域，可以独立地从混合音频中提取各自对应的声音信息，避免了音频衍生 query 相互干扰的问题。从低分辨率到高分辨率的逐步视觉增强确保精细的轮廓预测

3. **原型-音频对比损失（PAC Loss）**:

    - 功能：确保随机初始化的音频原型学习到不同音频事件类别的语义信息
    - 核心思路：将音频特征投影并全局池化后与各原型做内积，得到各类别的匹配预测 $M \in \mathbb{R}^K$。利用数据集标注获取真实的音频事件类别作为 ground truth $M^*$。用 BCE 损失训练：$\mathcal{L}_{pac} = \frac{1}{K} \sum_k \mathcal{L}_{bce}(M_k, M_k^*)$。该损失拉近音频特征与对应原型的距离，推远与不相关原型的距离
    - 设计动机：没有损失约束的随机初始化原型无法学习有意义的音频先验（消融实验证实），而通过与音频特征而非视觉特征的对比学习，原型能获得更显式和清晰的音频事件类别先验

### 损失函数 / 训练策略

总损失为 $\mathcal{L} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{pac}\mathcal{L}_{pac}$，其中 $\lambda_{cls}=2, \lambda_{mask}=5, \lambda_{pac}=1$。分类损失为 CE loss，mask 损失包含 BCE + Dice loss。使用 AdamW 优化器，学习率 $1e^{-4}$。训练迭代次数：S4 子集 45K，MS3 子集 40K，AVSS 子集 45K。视觉衍生 query 数 N=100，decoder 重复次数 D=2。

## 实验关键数据

### 主实验

| 方法 | Backbone | AVSS $\mathcal{M_J}$ | AVSS $\mathcal{M_F}$ | S4 $\mathcal{M_J}$ | MS3 $\mathcal{M_J}$ |
|------|----------|-----|-----|-----|-----|
| COMBO | PVT-v2 | 42.1 | 46.1 | 84.7 | 59.2 |
| AVSBias | Swin-B(384) | 44.4 | 49.9 | 83.3 | 67.2 |
| TeSO | Swin-B(384) | 39.0 | 45.1 | 83.3 | 66.0 |
| **VCT (Ours)** | PVT-v2(224) | **44.7** | **49.5** | **84.8** | 62.0 |
| **VCT (Ours)** | Swin-B(224) | **47.9** | **52.9** | 84.7 | **67.5** |
| **VCT (Ours)** | Swin-B(384) | **51.2** | **55.5** | **86.2** | **67.6** |

### 消融实验（AVSS 子集，ResNet-50）

| 配置 | $\mathcal{M_J}$ | $\mathcal{M_F}$ | 说明 |
|------|-----|-----|------|
| ACT (audio-derived queries) | 33.2 | 37.0 | 音频中心基线 |
| VCT + Naive Vision Queries | 35.2 | 39.3 | 仅视觉嵌入聚合 |
| + Cross-Attention | 35.8 | 39.8 | 正常 softmax |
| + Group-Attention (Gumbel) | 36.3 | 40.5 | Gumbel-Softmax 硬分配 |
| + Audio Prototypes (PAC) | **37.5** | **42.2** | 完整 PPQG |

### 关键发现

- **视觉中心 vs 音频中心的根本优势**：仅使用最简单的视觉嵌入作为 query（35.2 vs 33.2），就已经超越音频衍生 query 基线，证明视觉中心范式的方向性正确
- **AVSS 子集提升最大**：在最具挑战性的语义分割子集上，VCT (Swin-B, 384) 达到 51.2 $\mathcal{M_J}$，比 AVSBias 的 44.4 高出 6.8 个点，说明视觉衍生 query 在需要区分多个发声物体类别时优势最大
- **PVT-v2 (224) 可以匹敌 Swin-B (384)**：VCT 用 PVT-v2 和 224 分辨率（44.7 $\mathcal{M_J}$）就超过了 AVSBias 用 Swin-B 和 384 分辨率（44.4），展示了架构设计的高效性
- **PAC loss 必须与音频特征对比**：与视觉 query 对比学习的效果（36.5）远不如与音频特征对比（37.5），说明原型需要从音频信号而非视觉关联中学习语义
- **直接融合音频特征到 query 是次优的**：音频特征与视觉特征的 multiply/concat/add 融合（33.9-36.3）均不如 VCT 的完整方案（37.5），验证了延迟融合优于早期融合的假设

## 亮点与洞察

- **query 设计范式的转变**：从"音频去找视觉"转变为"视觉去找音频"，这种视角转换虽然简单但非常有效。核心洞察是：多个视觉区域的 query 可以独立地从混合音频中提取各自的信息，而混合音频衍生的 query 本身就已经纠缠在一起
- **Gumbel-Softmax 的巧妙运用**：用硬分配让不同 query 聚焦不同图像区域，同时保持可微性。这种做法借鉴了 GroupViT 但在多模态分割场景中取得了新的效果
- **音频原型的双重作用**：既作为 PPQG 中的类别先验提示 query 生成，又通过 PAC loss 自身学习音频语义，是一个优雅的自监督设计

## 局限与展望

- 音频编码器使用较旧的 VGGish，换用更强的音频模型（如 AudioMAE、BEATs）可能进一步提升
- 当前的原型数量固定为 K（音频事件类别数），对于开放世界场景需要动态原型机制
- 未考虑多帧间的时序建模，当前逐帧处理可能遗漏时序一致性
- 在 MS3 子集上部分设置下提升不如 AVSS 显著，说明在简单场景中优势有限

## 相关工作与启发

- **vs COMBO**: COMBO 将音频特征加到可学习 query 上再与视觉特征双向融合。VCT 完全以视觉为中心，让音频信息通过 decoder 交互逐步获得，避免了早期融合的信息纠缠
- **vs AQFormer**: AQFormer 直接用音频作为 query 聚合视觉特征。本文反其道行之用视觉作为 query 提取音频信息，logit map 可视化清楚地展示了视觉衍生 query 关注更多样化区域的优势
- **vs GAVS**: GAVS 使用 ViT-B 在 S4 上达到 80.1 和 MS3 上的 63.7，VCT 用 Swin-B 在两者上均超越

## 评分

- 新颖性: ⭐⭐⭐⭐ query 设计范式的转变是核心贡献，虽然思路简单但效果显著
- 实验充分度: ⭐⭐⭐⭐⭐ 三个子集全面评估，消融实验覆盖每个组件，可视化分析有说服力
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，图示专业，方法描述完整
- 价值: ⭐⭐⭐⭐ 为 AVS 领域提供了新的设计范式，PPQG 模块有迁移潜力

<!-- RELATED:START -->

## 相关论文

- [Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](mambavision_a_hybrid_mamba-transformer_vision_backbone.md)
- [Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics](dynamic_derivation_and_elimination_audio_visual_segmentation_with_enhanced_audio.md)
- [Rethinking Query-Based Transformer for Continual Image Segmentation](rethinking_query-based_transformer_for_continual_image_segmentation.md)
- [Audio-Visual Instance Segmentation](audio-visual_instance_segmentation.md)

<!-- RELATED:END -->
