---
title: >-
  [论文解读] Learning Visual Generative Priors without Text
description: >-
  [CVPR 2025][图像生成][图像到图像生成] 提出Lumos框架，通过纯视觉的图像到图像（I2I）自监督预训练学习视觉生成先验，然后仅用1/10的文本-图像对微调即可达到甚至超越现有T2I模型的效果，并在文本无关的视觉任务（I2V、NVS）上展现出优于T2I先验的性能。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "图像到图像生成"
  - "视觉生成先验"
  - "自监督学习"
  - "预训练"
  - "扩散模型"
---

# Learning Visual Generative Priors without Text

**会议**: CVPR 2025  
**arXiv**: [2412.07767](https://arxiv.org/abs/2412.07767)  
**代码**: [https://ant-research.github.io/lumos](https://ant-research.github.io/lumos) (项目页)  
**领域**: 图像生成  
**关键词**: 图像到图像生成, 视觉生成先验, 自监督学习, 预训练, 扩散模型

## 一句话总结
提出Lumos框架，通过纯视觉的图像到图像（I2I）自监督预训练学习视觉生成先验，然后仅用1/10的文本-图像对微调即可达到甚至超越现有T2I模型的效果，并在文本无关的视觉任务（I2V、NVS）上展现出优于T2I先验的性能。

## 研究背景与动机

1. **领域现状**：当前文本到图像（T2I）模型是视觉生成的主流先验，被广泛用作下游任务（如视频生成、3D合成）的初始化权重。
2. **现有痛点**：T2I模型严重依赖高质量文本-图像数据对。实验表明文本噪声比例从10%增至90%时，CLIP score下降约1.0。扩大高质量配对数据的标注成本极高，限制了模型的尺度化。
3. **核心矛盾**：T2I模型同时需要学习两件困难的事——纹理建模和文本-图像对齐。嘈杂的文本不仅影响对齐，还会干扰纹理建模的学习。
4. **本文目标** 能否将纹理建模和跨模态对齐解耦——先用海量无标注图像学习纯视觉生成先验，再用少量配对数据做对齐微调？
5. **切入角度**：跨模态对齐并非"好的视觉生成先验"的必要条件，视觉先验的核心在于纹理建模。I2I生成可以从无标注图像中自监督学习。
6. **核心 idea**：用预训练视觉编码器（DINO）提取图像特征作为条件，在1.9亿无标注图像上训练I2I扩散模型作为更基础的视觉先验。

## 方法详解

### 整体框架
分两阶段：(1) I2I预训练——给定图像，用冻结的视觉编码器（DINO-B）提取特征，以此为条件训练DiT-XL/2扩散模型进行图像重建；(2) 下游迁移——加载I2I预训练权重，将条件从图像特征切换为文本编码器（T5-XXL）的输出，在少量文本-图像对上微调实现T2I，或直接迁移到NVS、I2V等文本无关任务。

### 关键设计

1. **纯视觉I2I训练框架**:
    - 功能：在无标注图像上以自监督方式学习视觉生成先验
    - 核心思路：给定图像 $x$，先用预训练VAE编码到潜空间 $z = \mathcal{E}(x)$，同时用冻结的DINO-B提取视觉语义特征 $\tau^{\text{img}}(x) \in \mathbb{R}^{M \times d}$。以这些特征为条件，通过cross-attention注入DiT骨干网络，用标准扩散去噪目标训练。关键点：编码器和VAE在整个预训练过程中保持冻结。
    - 设计动机：DINO等自监督视觉编码器已被证明能提取比有监督模型更丰富的特征。利用它们作为条件，I2I模型可以充分利用互联网上海量的无标注图像。

2. **条件特征选择策略**:
    - 功能：确定I2I模型使用全局或局部视觉特征作为条件
    - 核心思路：对比了三种特征：全局CLS token、局部patch tokens、全部tokens。实验发现局部特征显著加速I2I收敛，但对下游T2I微调反而不利（因为模型对条件的依赖过强）；全局特征虽然I2I收敛慢，但下游迁移效果更好。最终选用全局特征。
    - 设计动机：全局特征提供语义级别的约束而非像素级约束，给下游任务留有更大的调整空间——这一发现揭示了"上下游不一致性"。

3. **视觉编码器选择：DINO vs CLIP**:
    - 功能：验证纯视觉编码器与多模态编码器对I2I先验的影响
    - 核心思路：对比DINO、MoCoV3（纯视觉）与CLIP（多模态）。DINO/MoCoV3在I2I阶段收敛更快、FID更好。在下游T2I微调时，CLIP在早期有优势（因为具有文本对齐能力），但DINO最终反超——作者称其为"后起之秀"（late bloomer）。T5在T2I阶段作为文本编码器优于CLIP编码器。
    - 设计动机：证明了纯视觉先验不仅可行，而且最终效果更好。I2I先验增强了纹理建模，简化了T2I的学习过程。

### 损失函数 / 训练策略
I2I阶段使用标准扩散去噪损失，在1.9亿图像上训练。T2I阶段在3000万文本-图像对上微调65K步即达到竞争性结果。支持条件dropout以启用classifier-free guidance。

## 实验关键数据

### 主实验

| 模型 | T&I Pairs | Steps | FID-30K↓ |
|------|-----------|-------|----------|
| SDv1.5 | 2000M | 1026k | 9.62 |
| PixArt-α | 24M | 240k | 7.32 |
| Imagen | 860M | 5000k | 7.27 |
| **Lumos-T2I** | **30M** | **65k** | **12.20** |
| **Lumos-T2I (长字幕)** | **30M** | **65k** | **6.44** |

使用长字幕时Lumos以仅30M数据和65K步超越所有现有方法。在GenEval上Overall=0.57，DPG-Bench上Average=79.9，与同量级模型相当甚至更好。

### 消融实验

| I2I数据规模 | I2I FID↓ | T2I FID↓ | 说明 |
|------------|----------|----------|------|
| 10M | 较高 | 较高 | 数据少 |
| 50M | 中等 | 中等 | 持续提升 |
| 200M | 最低 | 最低 | 规模化有效 |

| 先验类型 | NVS PSNR↑ | NVS SSIM↑ | NVS LPIPS↓ |
|---------|-----------|-----------|------------|
| 无先验 | 较低 | 较低 | 较高 |
| T2I先验 | 中等 | 中等 | 中等 |
| **I2I先验** | **19.63** | **0.8439** | **0.1526** |

### 关键发现
- **上下游不一致性**：I2I模型本身FID越好，不代表下游T2I迁移越好。局部特征I2I好但T2I差，全局特征反之——这揭示了预训练和下游的目标不完全一致。
- **I2I先验可规模化**：从10M到200M图像，I2I和下游T2I的FID持续下降，证明纯视觉先验的学习可以有效利用数据规模。
- **I2I先验在文本无关任务上优于T2I先验**：在NVS和I2V任务上，I2I先验一致性地优于T2I先验，因为不需要设计文本提示。
- **DINO是"后起之秀"**：虽然CLIP在I2I早期收敛快，但DINO在最终步数上反超，且下游T2I效果更好。

## 亮点与洞察
- **解耦纹理建模与对齐**：将T2I的两个难题拆开——先用海量无标注数据学纹理（I2I），再用少量配对数据学对齐（T2I微调）。这是一种本质上的效率提升思路。
- **上下游不一致性的发现**：预训练模型的"好"不等于下游任务的"好"——这为选择预训练策略提供了重要启示。
- **数据效率**：仅用1/10的文本-图像对就达到了竞争性能，大幅降低了高质量配对数据的需求。对资源有限的研究者非常友好。

## 局限与展望
- 当前仅在DiT-XL/2（~0.8B参数）上验证，更大模型的表现未知
- I2I预训练的图像筛选标准和数据来源对结果的影响未充分探讨
- T2I微调仍需30M配对数据，能否进一步减少到百万级别？
- 未探索将I2I先验应用到图像编辑、图像修复等更多下游任务

## 相关工作与启发
- **vs PixArt-α**: PixArt-α使用ImageNet上的class-to-image预训练，仍依赖人工标注的类别标签；Lumos完全自监督，无需任何标注
- **vs RCG**: RCG也做I2I生成但仅在ImageNet上训练且用adaLN-Zero注入条件；Lumos在1.9亿图像上用cross-attention训练，FID从12.70降到4.82
- **vs DALL·E2**: DALL·E2也有I2I中间桥梁的思路，但其unCLIP需要CLIP对齐；Lumos证明纯视觉编码器更优

## 评分
- 新颖性: ⭐⭐⭐⭐ 明确提出纯视觉生成先验的概念并系统验证
- 实验充分度: ⭐⭐⭐⭐⭐ T2I/NVS/I2V三个下游任务，大量消融实验，分析非常详尽
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富
- 价值: ⭐⭐⭐⭐ 降低了数据标注依赖，对大规模视觉生成预训练有重要参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Generative Image Layer Decomposition with Visual Effects](generative_image_layer_decomposition_with_visual_effects.md)
- [\[ICML 2025\] Visual Generation Without Guidance](../../ICML2025/image_generation/visual_generation_without_guidance.md)
- [\[ICML 2025\] Learning Single Index Models with Diffusion Priors](../../ICML2025/image_generation/learning_single_index_models_with_diffusion_priors.md)
- [\[CVPR 2025\] FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)
- [\[CVPR 2025\] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](ctrl-o_language-controllable_object-centric_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
