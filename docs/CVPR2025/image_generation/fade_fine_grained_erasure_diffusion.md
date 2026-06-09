---
title: >-
  [论文解读] FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models
description: >-
  [CVPR 2025][图像生成][概念遗忘] 提出 FADE（Fine-grained Attenuation for Diffusion Erasure），首次解决文生图扩散模型中概念遗忘的邻近性问题——精确擦除目标概念的同时保留语义相近概念的生成能力，在保留性能上比 SOTA 提升至少 12%。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "概念遗忘"
  - "细粒度擦除"
  - "邻近感知"
  - "扩散模型安全"
  - "LoRA"
---

# FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models

**会议**: CVPR 2025  
**arXiv**: [2503.19783](https://arxiv.org/abs/2503.19783)  
**代码**: [https://iab-rubric/unlearning/FG-Un](https://iab-rubric/unlearning/FG-Un)  
**领域**: 图像生成  
**关键词**: 概念遗忘, 细粒度擦除, 邻近感知, 扩散模型安全, LoRA

## 一句话总结

提出 FADE（Fine-grained Attenuation for Diffusion Erasure），首次解决文生图扩散模型中概念遗忘的邻近性问题——精确擦除目标概念的同时保留语义相近概念的生成能力，在保留性能上比 SOTA 提升至少 12%。

## 研究背景与动机

**领域现状**：文生图扩散模型需要选择性地移除特定概念（如有害内容），现有遗忘方法（ESD、UCE、SPM）关注局部性但忽视邻近性。

**现有痛点**：擦除"金毛寻回犬"后，其他寻回犬品种的生成也会受损；擦除特定花种后相似花种也无法正确生成。

**核心矛盾**：目标概念与相邻概念在特征空间中距离近，粗粒度遗忘会波及相邻概念。

**核心 idea**：通过 Concept Neighborhood 识别语义相邻概念集合，用 Mesh Modules（LoRA）实现精确擦除同时保护相邻概念。

## 方法详解

### 整体框架

三组件：(1) Concept Neighborhood 用图像嵌入相似度找到目标概念的 top-K 相邻概念；(2) Mesh Modules（LoRA 适配器）通过三项损失精确修改模型；(3) ERB 评分指标同时衡量遗忘效果和邻近保留。

### 关键设计

1. **Concept Neighborhood**:

    - 功能：自动识别与目标概念语义相近的概念集合
    - 核心思路：用原始模型为每个概念生成 m 张图像，用预训练图像编码器计算均值特征向量，按余弦相似度选取 top-K 最相似概念。理论证明 latent space 中 k-NN 分类器在一定条件下收敛到最优贝叶斯分类器
    - 设计动机：在无语义标注（如 WordNet）时自动构建邻近集合

2. **三项损失的 Mesh Modules**:

    - 功能：平衡擦除目标概念与保护相邻概念
    - 核心思路：Erasing Loss 用 triplet 形式拉远目标概念与其相邻概念的噪声预测距离；Guidance Loss 将目标概念的噪声预测引向 null 概念；Adjacency Loss 约束相邻概念的噪声预测不变：$\mathcal{L}_{FADE} = \lambda_{er}\mathcal{L}_{er} + \lambda_{adj}\mathcal{L}_{adj} + \lambda_{guid}\mathcal{L}_{guid}$
    - 设计动机：三项损失分别负责擦除、引导和保护，实现精细平衡

3. **ERB 评估指标**:

    - 功能：统一评估遗忘效果和邻近保留
    - 核心思路：Erasing-Retention Balance Score 同时量化目标概念的遗忘程度和相邻概念的保留程度
    - 设计动机：现有指标只关注遗忘效果，忽视了邻近保留这个关键维度

### 损失函数 / 训练策略

仅训练 LoRA 参数（Mesh Modules），保持原始模型权重冻结。训练数据由模型自身生成。

## 实验关键数据

### 主实验

在 Stanford Dogs、Oxford Flowers、CUB、I2P、Imagenette、ImageNet-1k 上：
- 邻近保留性能比 SOTA 提升 ≥ 12%
- 目标概念擦除效果与 SOTA 持平或更优

### 关键发现

- Adjacency Loss 对保护相邻概念至关重要
- K=5 的邻近集合大小在多数场景下效果最佳
- 细粒度数据集（如 Stanford Dogs）上优势更明显

## 亮点与洞察

- 首次正式定义并解决概念遗忘的"邻近性"问题
- Concept Neighborhood 方法简洁且有理论支撑
- 提出的 ERB 指标填补了评估维度的空白

## 局限与展望

- Concept Neighborhood 依赖图像编码器的质量
- LoRA 适配器的表达能力可能限制复杂遗忘场景
- 对组合概念（如"穿裙子的金毛犬"）的处理待研究

## 评分

- 新颖性：8/10 — 邻近感知遗忘的首次形式化
- 技术深度：8/10 — 三项损失设计有理论依据
- 实验充分度：8/10 — 6 个数据集广泛验证
- 写作质量：8/10 — 问题定义清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [\[CVPR 2025\] FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs](finelip_clip_long_text_fine_grained.md)
- [\[CVPR 2025\] Six-CD: Benchmarking Concept Removals for Text-to-Image Diffusion Models](six-cd_benchmarking_concept_removals_for_text-to-image_diffusion_models.md)
- [\[CVPR 2025\] SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models](sleepermark_towards_robust_watermark_against_fine-tuning_text-to-image_diffusion.md)

</div>

<!-- RELATED:END -->
