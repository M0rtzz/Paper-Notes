---
title: >-
  [论文解读] CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models
description: >-
  [CVPR2026][多模态VLM][Concept Bottleneck Model] 提出 TextUnlock 方法将任意冻结视觉分类器的输出分布对齐到视觉-语言对应空间，进而构建无需CLIP、无需标签、无需训练线性探针的全无监督概念瓶颈模型 (U-F²-CBM)…
tags:
  - "CVPR2026"
  - "多模态VLM"
  - "Concept Bottleneck Model"
  - "可解释性"
  - "知识蒸馏"
  - "无监督分类"
  - "视觉-语言对齐"
  - "零样本图像描述"
---

# CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models

**会议**: CVPR 2026 Findings  
**arXiv**: [2503.10981](https://arxiv.org/abs/2503.10981)  
**代码**: 待确认  
**领域**: 多模态VLM  
**关键词**: Concept Bottleneck Model, 可解释性, 知识蒸馏, 无监督分类, 视觉-语言对齐, 零样本图像描述

## 一句话总结

提出 TextUnlock 方法将任意冻结视觉分类器的输出分布对齐到视觉-语言对应空间，进而构建无需CLIP、无需标签、无需训练线性探针的全无监督概念瓶颈模型 (U-F²-CBM)，在40+模型上超越有监督CLIP-based CBM。

## 研究背景与动机

**概念瓶颈模型 (CBM) 的价值**：CBM 将稠密特征映射为人类可解释的概念激活，再线性组合预测类别，是重要的可解释性工具，但现有方法严重依赖 CLIP 提供图像-概念标注。

**CLIP 依赖的弊端**：使用 CLIP 生成概念标注时，CBM 被锚定在 CLIP 嵌入空间，遗留模型需通过 CLIP 的相似性概念来解释，而非其自身学到的表征；还会引入 CLIP 的偏见（如排版偏见）。

**遗留专家模型难以替代**：真实场景中常存在高性能的任务专用遗留模型，在 CLIP 大规模图文语料上重训不现实（计算成本高、数据需求量大）。

**人工标注成本高昂**：不使用 CLIP 的方法需要人工标注图像-概念关联，耗时且昂贵。

**所有 CBM 都需训练线性探针**：现有所有 CBM 方法均需在概念激活之上训练线性分类器将概念映射到类别，无法做到全无监督。

**重训会改变模型决策分布**：对遗留模型进一步微调会改变其原始决策过程，通常不希望如此。

## 方法详解

### 整体框架

方法的目标是：不依赖 CLIP、不依赖标签、也不训练任何线性探针，就把一个**冻结的现成视觉分类器**改造成可解释的概念瓶颈模型。它分两阶段——先用 TextUnlock 训一个轻量 MLP，把冻结分类器的特征投射到文本嵌入空间且保持原分类分布不变；再在这个对齐后的空间里直接做概念发现和概念-类别预测，整个第二阶段不需要任何额外训练。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    I["输入图像 I"] --> FV["冻结视觉编码器 Fᵥ → 特征 f"]
    subgraph TU["TextUnlock：把冻结分类器搬进文本空间"]
        direction TB
        FV --> MLP["可训练 MLP<br/>投射 f → f̃"]
        CN["类名经模板编码<br/>U = T(an image of a class)"] --> SIM["余弦相似度<br/>S = f̃ · Uᵀ"]
        MLP --> SIM
        SIM -.->|"对齐"| DIST["自蒸馏损失<br/>S 对齐原软分布 o = softmax(f·W)<br/>仅训练 MLP，编码器全冻结"]
    end
    subgraph CBM["U-F²-CBM：免训练推导概念-类别权重"]
        direction TB
        CSET["概念集：20K 常用词过滤后<br/>文本编码为 C"] --> ACT["概念激活<br/>f̃ · Cᵀ"]
        CSET --> WCON["概念-类别权重<br/>W_con = C · Uᵀ（免训练）"]
        ACT --> PRED["总体预测<br/>S_cn = f̃ · CᵀC · Uᵀ"]
        WCON --> PRED
    end
    MLP --> ACT
    PRED --> OUT["可解释类别预测<br/>+ 概念干预 / 零样本描述"]
```

### 关键设计

**1. TextUnlock：把冻结分类器搬进文本空间，又不动它的决策**

CBM 的老问题是必须借 CLIP 来给图像-概念打标，于是模型被锚死在 CLIP 空间、还继承了 CLIP 的偏见。TextUnlock 改用分类器自己的输出分布来对齐：给定冻结的视觉分类器 $F$（特征提取器 $F_v$ + 线性头 $W$）和任意文本编码器 $T$，先训一个 MLP 把视觉特征 $f = F_v(I)$ 投射为与文本同空间的 $\tilde{f} = \text{MLP}(f) \in \mathbb{R}^m$；再把 $K$ 个类名用模板 "an image of a {class}" 编码成 $U \in \mathbb{R}^{K \times m}$ 当作新分类头；最后计算投射特征与类名的余弦相似度 $S = \tilde{f} \cdot U^T$，把它对齐到原始分类器的软分布 $o = \text{softmax}(f \cdot W)$。整个过程只动 MLP，编码器和原分类头全冻结，因此既进了文本空间又没改原模型的决策分布。

**2. U-F²-CBM：不训练探针，直接用文本相似度推出概念-类别权重**

所有现有 CBM 都得在概念激活之上训一个线性分类器，做不到全无监督。本文的洞察是：既然类名权重 $U$ 和概念集都来自同一个文本编码器，那概念到类别的映射可以直接「算」出来。具体地，选 $Z = 20K$ 个英语常用词、严格过滤（去掉与类名匹配、父/子类、同义词等）后编码为 $C \in \mathbb{R}^{Z \times m}$，图像的概念激活就是 $\tilde{f} \cdot C^T \in \mathbb{R}^Z$；概念-类别分类器无需训练，直接令 $W^{con} = C \cdot U^T \in \mathbb{R}^{Z \times K}$，权重即每个概念与每个类名的文本相似度。于是总体预测为

$$S_{cn} = (\tilde{f} \cdot C^T) \cdot (C \cdot U^T) = \tilde{f} \cdot \underbrace{C^T C}_{\text{Gram矩阵}} \cdot U^T$$

有意思的是，当 Gram 矩阵 $C^T C$ 退化为单位阵时，整式回到原始特征分类器 $\tilde{f} \cdot U^T$——也就是说，CBM 转换本质上只是在原分类器里插了一个概念的 Gram 矩阵，这给「加可解释性而几乎不掉精度」提供了直观解释。

### 损失函数

$$L = -\sum_{i=1}^{K} o_i \log\left(\frac{e^{s_i}}{\sum_{j=1}^{K} e^{s_j}}\right)$$

该损失等价于原始分布 $o$ 与预测分布间的 KL 散度（差一个常数熵项），可视为一种自蒸馏——把原模型的分布蒸到它的视觉-语言对应分布里。关键在于：全程无需任何真实标签，仅需类名文本。

## 实验

### 主要结果

**TextUnlock 分类精度保持**（ImageNet-1K 验证集，17 个模型）：

| 模型 | TextUnlock Top-1 | 原始 Top-1 | Δ |
|------|---------|-------|------|
| ResNet50 | 75.80 | 76.13 | −0.33 |
| EfficientNetv2-M | 84.95 | 85.11 | −0.16 |
| ViT-B/16 | 80.70 | 81.07 | −0.37 |
| Swinv2-Base | 83.72 | 84.11 | −0.39 |
| BeiT-L/16 | 87.22 | 87.34 | −0.12 |
| DINOv2-B | 84.40 | 84.22 | **+0.18** |

40 个模型平均精度下降仅约 **0.2 个百分点**。

**U-F²-CBM vs 有监督 CLIP-based CBM**（ImageNet-1K）：

| 方法 | 模型 | Top-1 |
|------|------|-------|
| LF-CBM (有监督) | CLIP ViT-B/16 | 75.4 |
| DN-CBM (有监督) | CLIP ViT-B/16 | 79.5 |
| DCBM-SAM2 (有监督) | CLIP ViT-L/14 | 77.9 |
| **U-F²-CBM (无监督)** | ViT-B/16v2 | **83.2** |
| **U-F²-CBM (无监督)** | ConvNeXtV2-B@384 | **86.4** |

即使是 ImageNet-only 训练的 ResNet50 (73.9) 也超过了 400M 图文对训练的 CLIP ResNet50 CBM (72.9)。

### 跨数据集泛化

| 数据集 | 方法 | 模型 | 精度 |
|--------|------|------|------|
| Places365 | CDM (CLIP) | CLIP-RN50 | 52.70 |
| Places365 | **Ours** | DenseNet161 | **53.42** |
| EuroSAT | Baseline (CLIP) | CLIP-ViT-B/16 | 88.57 |
| EuroSAT | **Ours** | ResNet50 | **94.22** |
| DTD | Baseline (CLIP) | CLIP-ViT-B/16 | 61.86 |
| DTD | **Ours** | ResNet50 | **68.88** |

在领域特定（场景/卫星/纹理）和细粒度、小类别数数据集上同样有效。

### 消融与关键发现

- **训练效率**：仅训练轻量 MLP（视觉编码器、文本编码器、线性分类头全部冻结），可在标准硬件上完成，数据需求远小于 CLIP 训练。
- **概念集灵活性**：概念集可在推理时任意替换（on-the-fly），因为只需用文本编码器编码新概念集即可。
- **概念干预**：通过显式干预瓶颈层概念可控制预测、修复偏见（如 "dumbbell" 类中手臂偏见的文本化解释）。
- **零样本图像描述**：将 TextUnlock 与 ZeroCap 结合，使任意视觉分类器可做零样本图像描述，ConvNeXtV2@384 在 COCO 上 CIDEr=17.9、SPICE=6.9，超越 CLIP-based 方法 (CIDEr=14.6, SPICE=5.5)。

## 亮点

- **三重 "Free"**：同时实现 CLIP-free、Label-free、无监督概念-类别分类器，是首个全无监督 CBM。
- **优雅的数学洞察**：CBM 转换等价于在原始分类器中插入概念 Gram 矩阵，当 Gram 矩阵为单位阵时退化回原始分类器。
- **极强的通用性**：适用于 CNN/Transformer/混合架构共 40+ 模型，架构无关。
- **高数据效率**：仅用 ImageNet-1K 训练即超越 400M 图文对训练的 CLIP 模型。
- **推理时灵活切换概念集**：无需重训即可更换概念集构建 CBM。

## 局限性

- 概念发现质量依赖于文本编码器的语义空间质量，弱文本编码器可能导致概念激活不够精确。
- Gram 矩阵引入的概念冗余可能在概念数量极大时影响性能。
- 零样本图像描述在 BLEU-4/METEOR 等 n-gram 指标上不及基于 CLIP 的方法（需额外组合式描述策略弥补）。
- 方法仅在分类任务上验证，尚未扩展到检测/分割等更复杂任务。
- 概念过滤依赖手动规则（去父类/同义词等），可能遗漏某些语义泄露。

## 相关工作

- **传统 CBM**：Koh et al. [ICML 2020] 提出原始 CBM，需人工概念标注。
- **Label-Free CBM (LF-CBM)**：利用 CLIP 提供图像-概念标注，免去人工标注但依赖 CLIP。
- **直接构建于 CLIP 之上的 CBM**：LaBo、CDM、DN-CBM 等直接在 CLIP 嵌入空间中计算概念激活。
- **视觉特征到文本的解码**：DeVIL、LIMBER 训练自回归生成器解码视觉特征为文本，但依赖标注数据且改变分类器分布。
- **T2C**：训练线性层将任意分类器映射到 CLIP 视觉空间，但仍依赖 CLIP 且丢弃原始类分布。
- **本文 U-F²-CBM**：完全不依赖 CLIP/VLM，不需标注数据，不改变原始决策分布，概念-类别分类器无监督推导。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 三重 Free 的全无监督 CBM 构建是全新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ — 40+ 模型、4 个数据集、消融/干预/零样本描述
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰，Gram 矩阵洞察优雅
- 价值: ⭐⭐⭐⭐⭐ — 解除可解释 CBM 对 CLIP 的依赖，通用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](coat_cbm_concept_wise_attention.md)
- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)
- [\[ICML 2026\] CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models](../../ICML2026/multimodal_vlm/clip_tricks_you_training-free_token_pruning_for_efficient_pixel_grounding_in_lar.md)
- [\[CVPR 2026\] LLMind: Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models](llmind_bio-inspired_training-free_adaptive_visual_representations_for_vision-lan.md)

</div>

<!-- RELATED:END -->
