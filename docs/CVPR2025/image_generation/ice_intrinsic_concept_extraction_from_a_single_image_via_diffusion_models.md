---
title: >-
  [论文解读] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models
description: >-
  [CVPR 2025][图像生成][concept learning] 提出 ICE 两阶段框架，仅用单个 T2I 扩散模型从单张图像自动定位物体级概念并分解为内在属性（类别、颜色、材质），实现无标注、无额外模型的层次化视觉概念提取。
tags:
  - CVPR 2025
  - 图像生成
  - concept learning
  - intrinsic concepts
  - 扩散模型
  - triplet loss
  - concept decomposition
---

# ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2503.19902](https://arxiv.org/abs/2503.19902)  
**代码**: [https://visual-ai.github.io/ice](https://visual-ai.github.io/ice)  
**领域**: image_generation  
**关键词**: concept learning, intrinsic concepts, diffusion model, Stable Diffusion, triplet loss, concept decomposition

## 一句话总结

提出 ICE 两阶段框架，仅用单个 T2I 扩散模型从单张图像自动定位物体级概念并分解为内在属性（类别、颜色、材质），实现无标注、无额外模型的层次化视觉概念提取。

## 研究背景与动机

**领域现状**: T2I 扩散模型（如 Stable Diffusion）积累了丰富的视觉世界知识，已被用于图像分类、分割、语义对应等任务。生成式概念学习旨在从图像中分解和理解组成复杂场景的基本元素。

**现有痛点**: (1) Textual Inversion/DreamBooth 需要多张同概念图像学习，无法从单图提取；(2) Break-A-Scene 依赖人工标注 mask；(3) ConceptExpress 等方法只提取物体级概念，忽略颜色/材质等内在属性；(4) LangInt 需要为每个概念轴训练独立编码器，扩展性差。

**核心矛盾**: 视觉概念定义本身具有模糊性和层次性 — 一个"红色金属球"包含物体类别（球）、颜色（红色）、材质（金属）等多层内在概念，现有方法缺乏系统化的分解机制。

**本文切入角度**: 完全利用预训练 T2I 模型的内在能力（CLIP 编码器做文本检索、自注意力层做分割），无需引入额外预训练模型。

## 方法详解

### 整体框架

两阶段架构：
1. **Stage 1 - 自动概念定位**: 零训练地从图像中提取文本概念 + 对应 mask
2. **Stage 2 - 结构化概念学习**: 分两个 phase 学习物体级概念和内在概念

### 关键设计

**1. 自动概念定位（Stage 1, 无训练）**
- **功能**: 迭代地从图像中提取概念和 mask，直到图像中无剩余物体。
- **核心流程**:
  1. Image-to-Text Retriever $\mathcal{T}(\mathbf{x})$: 用 CLIP 编码器的 SpLiCE 框架将密集嵌入分解为稀疏可解释的语义概念，获取 top-1 文本概念 $c_i$
  2. Zero-shot Segmentor $\mathcal{S}(\mathbf{x}, c_i)$: 利用 Stable Diffusion 的自注意力层（DiffSeg）生成对应 mask $\mathbf{m}_i$
  3. 迭代移除: $\mathbf{x}' = \mathbf{x} \odot (1 - \mathbf{m}_i)$，更新图像后重复
- **设计动机**: 完全复用 T2I 模型内部组件，不引入外部模型，确保框架的轻量性和一致性。

**2. 物体级概念学习（Stage 2, Phase 1）**
- **功能**: 为每个物体级概念 $c_i$ 学习两个 token: concept-specific $c_i^{conspec}$（表示通用语义类别）和 instance-specific $c_i^{inspec}$（表示实例特有属性）。
- **核心思路**: 用三元组损失把 concept-specific token 拉近 anchor（Stage 1 提取的文本概念），推远 instance-specific token:
  $$\mathcal{L}_{triplet}^{obj} = \max(0, \|\mathcal{E}(anchor) - \mathcal{E}(c_i^{conspec})\|_2^2 - \|\mathcal{E}(anchor) - \mathcal{E}(c_i^{inspec})\|_2^2 + \gamma)$$
- **设计动机**: 三元组损失的 anchor 来自 Stage 1 的高质量文本概念初始化，提供了比随机初始化更好的学习起点。

**3. 内在概念学习（Stage 2, Phase 2）**
- **功能**: 将 instance-specific token 进一步分解为内在属性 token $c_j^{intrinsic}$（如颜色、材质）。
- **核心思路**: 为每种内在概念构造专属 anchor 文本（如"the colour of $c_i^{inspec}$"），用内在三元组损失约束:
  $$\mathcal{L}_{triplet}^{intrinsic} = \max(0, \|\mathcal{E}(anchor_j) - \mathcal{E}(c_j^{intrinsic})\|_2^2 - \|\mathcal{E}(anchor_j) - \mathcal{E}(c_k^{intrinsic})\|_2^2 + \gamma)$$
- **概念精炼**: 在 Phase 2 后对 U-Net 和文本编码器进行少量 fine-tuning（300步），确保内在概念与视觉属性精确对齐。
- **设计动机**: 层次化分解（物体级→内在概念级）比一次性学习所有概念要稳定得多。

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{att}\mathcal{L}_{att} + \lambda_{triplet}\mathcal{L}_{triplet}$$

- $\mathcal{L}_{recon}$: 去噪重建损失
- $\mathcal{L}_{att}$: 注意力 mask 损失（Wasserstein距离对齐注意力图与分割mask）
- $\mathcal{L}_{triplet}$: 三元组损失（物体级或内在级，按 phase 切换）
- 超参: $\lambda_{att} = 1 \times 10^{-5}$, $\lambda_{triplet} = 1$

训练配置: Phase 1 400步 → Phase 2 400步 → Refinement 300步，单 3090 GPU。

## 实验关键数据

### 主实验 — UCE 基准（CLIP 编码器）

| 方法 | SIMI↑ | SIMC↑ | ACC1↑ | ACC3↑ |
|---|---|---|---|---|
| Break-A-Scene | 0.627 | 0.773 | 0.174 | 0.282 |
| ConceptExpress | 0.689 | 0.784 | 0.263 | 0.385 |
| **ICE** | **0.738** | **0.822** | **0.325** | **0.518** |

### DINO 编码器评估

| 方法 | SIMI↑ | SIMC↑ | ACC1↑ | ACC3↑ |
|---|---|---|---|---|
| ConceptExpress | 0.319 | 0.568 | 0.324 | 0.470 |
| **ICE** | **0.677** | **0.755** | **0.476** | **0.638** |

ICE 在 DINO 特征空间的优势更加显著（SIMI 提升 112%），说明学到的概念具有跨编码器的泛化性。

### 消融实验

| 变体 | CLIP SIMI↑ | CLIP SIMC↑ | DINO SIMI↑ | DINO ACC3↑ |
|---|---|---|---|---|
| ConceptExpress | 0.689 | 0.784 | 0.319 | 0.470 |
| ICE w. mask only | 0.710 | 0.781 | 0.493 | 0.604 |
| ICE w/o Stage Two | 0.726 | 0.807 | 0.501 | 0.604 |
| ICE w/o text init | 0.722 | 0.814 | 0.548 | 0.627 |
| **ICE Full** | **0.738** | **0.822** | **0.677** | **0.638** |

### Mask 质量对比

| 方法 | mIoU↑ | Recall↑ | Precision↑ |
|---|---|---|---|
| ConceptExpress | 0.483 | 0.676 | 0.657 |
| **ICE** | **0.635** | **0.893** | **0.720** |

ICE 的 Recall 从 0.676 提升到 0.893，表明自动概念定位能捕获更多物体区域。

### 关键发现

1. **每个模块都有贡献**: 消融显示 mask、text init、Stage Two 学习各自提供增量收益。
2. **Mask质量是基础**: ICE 的 mIoU=0.635 远超 ConceptExpress 的 0.483，更好的定位带来更好的概念学习。
3. **DINO空间优势更大**: 说明 ICE 学到的概念不只是 CLIP 语义，而是更通用的视觉属性。
4. **无需额外模型**: 全部复用 T2I 模型内部组件（CLIP编码器+自注意力分割），实现端到端一致性。

## 亮点与洞察

- 概念层次化分解（物体→类别+实例→内在属性）的设计思路清晰优雅
- Stage 1 完全零训练，仅利用 T2I 模型内部组件完成概念定位和分割
- 三元组损失 + 文本 anchor 的设计巧妙地将概念约束嵌入学习过程
- 是首个从单张图像同时提取物体级和内在级概念的无监督框架

## 局限与展望

- 依赖 DiffSeg 的零样本分割质量，复杂重叠场景可能提取不完整
- 内在概念类型（颜色、材质）需要预定义，无法自动发现新属性维度
- 训练步数较少（共 1100 步），概念精炼可能不够充分
- 仅在 Unsplash 数据集上评估，缺乏多样化场景的验证
- 概念分解的可解释性仍有提升空间（某些内在概念可能纠缠）

## 相关工作与启发

- **Textual Inversion / DreamBooth**: 概念学习的先驱，但需多图 + 单概念
- **ConceptExpress**: 当前最强的单图多概念基线，但仅物体级
- **Inspiration Tree**: 尝试概念分解，但缺乏结构化引导
- **启发**: T2I 模型的内部组件（CLIP、自注意力）蕴含丰富的可利用信号，"模型即工具"的范式值得深入

## 评分

⭐⭐⭐⭐ — 框架设计精巧，从 T2I 模型中挖掘概念分解能力的路径新颖；实验在 UCE 基准上全面领先，但内在概念类型需预设是局限。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Intrinsic Concept Extraction Based on Compositional Interpretability](../../CVPR2026/image_generation/intrinsic_concept_extraction_based_on_compositional_interpretability.md)
- [\[CVPR 2025\] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)
- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [\[CVPR 2025\] DeClotH: Decomposable 3D Cloth and Human Body Reconstruction from a Single Image](decloth_decomposable_3d_cloth_and_human_body_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] CustAny: Customizing Anything from A Single Example](custany_customizing_anything_from_a_single_example.md)

<!-- RELATED:END -->
