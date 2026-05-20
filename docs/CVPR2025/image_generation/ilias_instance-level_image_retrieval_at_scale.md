---
title: >-
  [论文解读] ILIAS: Instance-Level Image Retrieval At Scale
description: >-
  [CVPR 2025][图像生成][实例级检索] ILIAS 是一个包含 1000 个实例对象、1 亿干扰图像的大规模实例级图像检索测试基准，通过全面 benchmarking 揭示了当前基础模型在特定物体识别上的能力与不足，为该领域提供了一个远未饱和的评测标准。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "实例级检索"
  - "大规模评测"
  - "基础模型"
  - "视觉语言模型"
  - "数据集基准"
---

# ILIAS: Instance-Level Image Retrieval At Scale

**会议**: CVPR 2025  
**arXiv**: [2502.11748](https://arxiv.org/abs/2502.11748)  
**代码**: [https://github.com/ilias-vrg/ilias](https://github.com/ilias-vrg/ilias)  
**领域**: 图像检索 / 基准数据集  
**关键词**: 实例级检索, 大规模评测, 基础模型, 视觉语言模型, 数据集基准

## 一句话总结
ILIAS 是一个包含 1000 个实例对象、1 亿干扰图像的大规模实例级图像检索测试基准，通过全面 benchmarking 揭示了当前基础模型在特定物体识别上的能力与不足，为该领域提供了一个远未饱和的评测标准。

## 研究背景与动机

**领域现状**：实例级图像检索（Instance-Level Image Retrieval）旨在从海量图库中找到包含与查询图像完全相同物体的图像。随着 DINOv2、SigLIP 等视觉基础模型的崛起，通用视觉表示的质量大幅提升。然而，现有评测基准（如 Oxford Buildings、Paris 等）规模小、领域单一（主要是地标建筑），且性能已接近饱和，无法有效区分不同方法的优劣。

**现有痛点**：现有数据集存在三个核心问题——(1) 规模有限：最大的检索基准库仅百万级，无法测试十亿级实际场景下的检索能力；(2) 领域单一：偏重地标和产品检索，无法评估模型在多样化物体类型上的泛化能力；(3) 性能饱和：顶尖模型已达 90%+ 的检索精度，区分度不够。此外，ground truth 标注中的假阴性问题也会干扰评估可靠性。

**核心矛盾**：基础模型的快速进步与评测基准的滞后之间存在鸿沟。现有基准无法回答"当前最好的基础模型在任意物体的实例级检索上究竟有多好"这一关键问题。

**本文目标**：构建一个大规模、多领域、ground truth 精确、且远未饱和的实例级检索测试基准，全面评估当前基础模型和检索技术。

**切入角度**：作者采用了一个巧妙的 ground truth 保证策略——只收集 2014 年之后出现的物体实例作为查询对象。由于 YFCC100M 数据集编译于 2014 年，这些物体一定不会出现在 YFCC100M 的 1 亿张干扰图像中，从而在无需额外标注的前提下消除假阴性问题。

**核心 idea**：构建一个时间分隔的大规模检索基准（查询物体 post-2014，干扰图像 pre-2014），确保零假阴性下的精确评估。

## 方法详解

### 整体框架
ILIAS 的构建包括三个阶段：数据收集、标注验证和全面 benchmarking。数据集包含 1000 个物体实例的 1232 张查询图像（干净/均匀背景）、4715 张正样本图像（真实环境中拍摄，含遮挡、杂乱背景、尺度变化）、1000 条文本查询描述，以及来自 YFCC100M 的 1 亿张干扰图像。评估流程涵盖特征提取、kNN 搜索、线性适配和重排等完整检索 pipeline。

### 关键设计

1. **时间分隔的 Ground Truth 策略（Temporal Separation for GT）**:

    - 功能：在无需人工逐一验证 1 亿张干扰图像的前提下，保证零假阴性
    - 核心思路：所有 1000 个查询物体实例都被确认是 2014 年之后新出现的（如新建筑、新产品、新艺术品等）。YFCC100M 的图片全部采集于 2014 年之前。因此这些物体在理论上不可能出现在干扰集中，任何未被标注为正样本的图像都一定是真正的负样本
    - 设计动机：传统方法评估大规模检索时面临假阴性困境——在百万/亿级图库中很难穷尽标注所有包含目标物体的图像。时间分隔策略从根本上消除了这个问题，同时避免了昂贵的大规模人工标注

2. **多领域物体实例收集（Multi-Domain Instance Collection）**:

    - 功能：确保评估涵盖多样化的物体类型，不偏向特定领域
    - 核心思路：1000 个物体实例跨越建筑、雕塑、产品、标志、自然地标、艺术品等多个领域。对每个实例，查询图像在干净背景下拍摄（类似"产品照"），正样本图像在真实环境中捕获，包含各种挑战性条件（遮挡、杂乱背景、视角变化、尺度变化、部分可见等）。这种设计模拟了"用产品图搜商品"或"用参考图搜实地照片"的真实应用场景
    - 设计动机：现有基准偏重地标导致"在地标上调优的模型在 ILIAS 上失败"——实验证实了领域特化模型缺乏泛化能力。多领域设计正是为了暴露这一问题

3. **全面的 Benchmarking 框架（Comprehensive Benchmarking Pipeline）**:

    - 功能：提供标准化的评估流程和多维度对比分析
    - 核心思路：评估覆盖四个维度——(a) 基础图-图检索：直接使用模型特征进行 kNN 搜索；(b) 线性适配：在模型特征上训练线性层进行多领域类别监督；(c) 重排：使用局部描述子（如 AMES、SP 等）对初始检索结果进行重排；(d) 文-图检索：使用文本查询和视觉-语言模型进行 text-to-image 检索。评估指标为 mAP@100 和 mAP (full)
    - 设计动机：不同的检索 pipeline 环节（特征提取、适配、重排）对最终性能的影响不同，全面评估能揭示每个环节的贡献

### 损失函数 / 训练策略
ILIAS 作为测试基准不涉及模型训练。线性适配部分使用多领域分类监督（multi-domain class supervision）训练线性投影层，采用交叉熵损失。

## 实验关键数据

### 主实验（Image-to-Image 检索）

| 模型 | 训练方式 | mAP@100 | mAP (full) |
|------|---------|---------|------------|
| DINOv3-L | SSL | **31.1** | **26.5** |
| PE-L@336 | VLA | 27.1 | 22.0 |
| DINOv3-B | SSL | 26.4 | 22.0 |
| SigLIP2-L@512 | VLA | 25.3 | 20.8 |
| DINOv2-L | SSL | 18.5 | 15.3 |
| OpenCLIP-L | VLA | 12.7 | 9.8 |

### 线性适配后检索

| 模型 | 适配前 mAP@100 | 适配后 mAP@100 | 提升 |
|------|---------------|---------------|------|
| PE-L@336 | 27.1 | **39.6** | +12.5 |
| SigLIP2-L@512 | 25.3 | 37.3 | +12.0 |
| SigLIP-L@384 | 24.2 | 34.3 | +10.1 |
| DINOv3-L | 31.1 | 32.9 | +1.8 |
| DINOv2-L | 18.5 | 23.5 | +5.0 |

### 关键发现
- **性能远未饱和**：最好的模型 mAP 也只有约 31%（基础检索）或 40%（适配后），大量提升空间存在。这与现有地标基准上 90%+ 的性能形成鲜明对比
- **领域特化模型泛化差**：在地标或产品检索上效果很好的模型在 ILIAS 的多领域测试中表现不佳
- **VLM 的线性适配效果显著**：视觉-语言模型经过简单的线性适配后性能大幅提升（+12.5 点），而 SSL 模型 DINOv3 提升有限（+1.8），说明 VLM 特征的潜力未被充分发挥但可通过少量监督释放
- **局部描述子的重排仍然关键**：在严重背景杂乱场景下，局部描述子重排可将 mAP 从 34 提升到 39，证明空间几何验证仍不可替代
- **文-图检索接近图-图检索**：VLM 的 text-to-image 检索性能（SigLIP2-L 的 24.7）与 image-to-image（25.3）差距很小，这个发现出人意料

## 亮点与洞察
- **时间分隔策略**是一个非常巧妙的 ground truth 保证机制——利用物体出现时间与数据集编译时间的天然分隔，以零标注成本实现零假阴性。这个思路可以迁移到任何需要大规模负样本集的评测场景
- 在 1 亿干扰图像级别的评测揭示了基础模型的真实能力边界。最好的模型也只有 ~31% mAP，说明实例级检索仍是一个极具挑战性的开放问题
- 线性适配的巨大效果（VLM 上 +12 点）暗示了一个低成本的实用改进方向：只需少量多领域标注数据，就能显著提升通用模型的检索能力

## 局限与展望
- ILIAS 是纯测试集，不提供训练数据，限制了在其上开发新方法的可能性
- 只包含 2014 年后的物体可能引入隐性偏差——这些物体可能更"现代"、更具特征性
- 1000 个实例的规模虽大于现有基准但对于某些领域仍显不足（如自然类物体的多样性有限）
- 文本查询的质量和粒度会影响 text-to-image 评估的公平性

## 相关工作与启发
- **vs Oxford/Paris Buildings**: Oxford5k 和 Paris6k 是经典地标检索基准，但规模小（5k/6k 图像）且性能已饱和。ILIAS 的干扰集大 4-5 个数量级，领域覆盖更广，性能远未饱和
- **vs GLDv2（Google Landmarks）**: GLDv2 专注于地标领域，ILIAS 刻意多领域化。实验证实地标特化模型在 ILIAS 上性能骤降
- **vs ROxford/RParis**: 这些是 Oxford/Paris 的重新标注版本，修复了标注问题但仍受限于小规模和单一领域
- 该数据集来自 CTU Prague 的 VRG 实验室，与 Oxford/Paris 基准来自相同的检索研究传统，在数据集设计上有明显的传承和改进

## 评分
- 新颖性: ⭐⭐⭐⭐ 时间分隔 ground truth 策略巧妙，大规模多领域设计填补了评测空白
- 实验充分度: ⭐⭐⭐⭐⭐ 73+ 个模型的全面 benchmarking，覆盖 4 种检索设置，发现多个有价值的洞察
- 写作质量: ⭐⭐⭐⭐ 数据集设计决策描述清晰，benchmarking 分析深入
- 价值: ⭐⭐⭐⭐⭐ 为实例级检索社区提供了一个急需的、远未饱和的大规模评测标准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Instance-Level Composed Image Retrieval](../../NeurIPS2025/image_generation/instance-level_composed_image_retrieval.md)
- [\[CVPR 2025\] RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories](rayflow_instance-aware_diffusion_acceleration_via_adaptive_flow_trajectories.md)
- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)
- [\[CVPR 2025\] PatchDPO: Patch-level DPO for Finetuning-free Personalized Image Generation](patchdpo_patch-level_dpo_for_finetuning-free_personalized_image_generation.md)
- [\[CVPR 2025\] GIF: Generative Inspiration for Face Recognition at Scale](gif_generative_inspiration_for_face_recognition_at_scale.md)

</div>

<!-- RELATED:END -->
