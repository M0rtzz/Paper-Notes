---
title: >-
  [论文解读] EmoEdit: Evoking Emotions through Image Manipulation
description: >-
  [CVPR 2025][图像生成][情感图像操纵] 本文提出 EmoEdit，首个通过内容修改（而非仅颜色/风格调整）来唤起指定情感的图像操纵框架，构建了 40,120 对的 EmoEditSet 数据集，设计了可即插即用的 Emotion Adapter，在结构保持和情感唤起之间取得了显著平衡。 领域现状：情感图像操纵（A…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "情感图像操纵"
  - "情感适配器"
  - "扩散模型编辑"
  - "内容感知"
  - "视觉情感分析"
---

# EmoEdit: Evoking Emotions through Image Manipulation

**会议**: CVPR 2025  
**arXiv**: [2405.12661](https://arxiv.org/abs/2405.12661)  
**代码**: 无  
**领域**: 扩散模型  
**关键词**: 情感图像操纵, 情感适配器, 扩散模型编辑, 内容感知, 视觉情感分析

## 一句话总结

本文提出 EmoEdit，首个通过内容修改（而非仅颜色/风格调整）来唤起指定情感的图像操纵框架，构建了 40,120 对的 EmoEditSet 数据集，设计了可即插即用的 Emotion Adapter，在结构保持和情感唤起之间取得了显著平衡。

## 研究背景与动机

**领域现状**：情感图像操纵（Affective Image Manipulation, AIM）旨在修改用户提供的图像以唤起特定情感。现有方法主要通过调整颜色和风格来实现，如 CLVA 和 AIF 将真实图像转为艺术风格。扩散模型在图像编辑中表现出色，但在情感操纵方面缺乏知识。

**现有痛点**：(1) 颜色/风格调整无法唤起精确和深层的情感变化——心理学研究表明视觉内容（而非仅颜色）是关键情感刺激；(2) 现有 AIM 方法多限于二元情感分类（正/负），粒度不够；(3) DALL-E 3 能传达情感但不保持原图结构，IP2P 保持结构但缺乏情感表达——情感唤起和结构保持天然矛盾。

**核心矛盾**：情感唤起需要有意义的内容修改（如添加蝴蝶传达满足感），但过大的修改会破坏原图结构。需要找到自动选择"恰当的、与上下文匹配的"情感语义修改的方法。

**本文目标** (1) 缺乏大规模 AIM 数据集——如何自动构建高质量的情感对照数据？(2) 如何让扩散模型具备情感意识？(3) 如何在不指定具体编辑指令的情况下，仅凭情感词就自动选择合适的内容修改？

**切入角度**：基于心理学中"视觉内容是情感刺激"的洞察，对 EmoSet 做聚类构建八种情感的"情感因子树"，每种情感对应多种语义表示（如"满足"→书与花、彩虹、蝴蝶等），然后数据驱动地训练 Emotion Adapter 学习基于上下文的语义选择。

**核心 idea**：构建情感因子树和大规模数据集教会扩散模型"什么样的内容修改能唤起什么情感"，实现仅需情感词即可驱动的内容感知编辑。

## 方法详解

### 整体框架

EmoEdit 分两大步：(1) 构建 EmoEditSet 数据集——从 EmoSet 做聚类提取情感因子树，用 IP2P 生成源-目标图像对，经四重指标筛选和人工审核；(2) 训练 Emotion Adapter——基于 Q-Former 架构设计情感适配器，结合 diffusion loss 和 instruction loss 训练，使之可即插即用到各种扩散模型中。推理时只需提供输入图像和目标情感词。

### 关键设计

1. **情感因子树与 EmoEditSet 数据集构建**:

    - 功能：提供大规模、语义多样的情感操纵配对数据
    - 核心思路：对 EmoSet 中八种情感（amusement、awe、contentment、excitement、anger、disgust、fear、sadness）分别用 CLIP 语义嵌入做聚类，提取代表性视觉因子。用 GPT-4V 为每个聚类生成内容摘要并分类为物体/场景/动作/表情四类，构建层次化的"情感因子树"。然后收集 15,531 张源图（来自 MagicBrush、MA5K、Unsplash），用 IP2P 以情感因子为指令生成目标候选，通过 CLIP 图像相似度（0.75-0.9）、CLIP 文本相似度（>0.25）、情感分数（>0.3）和美学分数四重过滤 + 人工审核，最终获得 40,120 对数据，平均每张图 2.6 个情感方向
    - 设计动机：缺乏大规模 AIM 数据是根本瓶颈。通过聚类而非人工标注获取情感因子，结合自动生成和多维度过滤，可扩展地构建高质量数据集

2. **Emotion Adapter（情感适配器）**:

    - 功能：使扩散模型具备情感意识，自动选择最适合输入图像的情感语义表示
    - 核心思路：基于 Q-Former 构建。可学习查询 $q$ 作为"情感字典"，目标情感 $e_t$ 和输入图像 $e_i$ 作为索引。自注意力层先根据目标情感选择相关语义：$A_s = \text{softmax}(\frac{[q;e_t]W_q^s([q;e_t]W_k^s)^T}{\sqrt{d_k}})[q;e_t]W_v^s$。交叉注意力层再结合图像信息选择最匹配的表示：$A_c = \text{softmax}(\frac{A_s W_q^c(e_i W_k^c)^T}{\sqrt{d_k}})e_i W_v^c$。迭代多次后输出情感嵌入 $c_e$，作为条件注入 IP2P 的去噪过程
    - 设计动机：每种情感对应多种语义表示（如"恐惧"可以是鬼怪、黑暗、暴风雨等），需要根据输入图像的内容动态选择。Q-Former 的查询机制天然适合这种"根据条件从字典中检索"的操作

3. **Instruction Loss（指令损失）**:

    - 功能：捕捉情感数据对中的语义变化，避免模型仅依赖像素颜色调整
    - 核心思路：在标准 diffusion loss $\mathcal{L}_{LDM}$ 之外增加 instruction loss：$\mathcal{L}_{ins} = \frac{1}{M}\|c_e - \mathcal{E}_{txt}(t_{ins})\|_2^2$，将 Emotion Adapter 输出的情感嵌入 $c_e$ 与数据集中对应的内容指令（如"添加彩色蝴蝶"）的文本编码 $\mathcal{E}_{txt}(t_{ins})$ 对齐。总损失为 $\mathcal{L} = \mathcal{L}_{LDM} + \mathcal{L}_{ins}$
    - 设计动机：仅用 diffusion loss 训练时模型倾向于做颜色纹理级调整（像素级最优解），产生颜色伪影但缺乏有意义的内容修改。Instruction loss 通过语义级监督强制模型学习内容变化

### 损失函数 / 训练策略

训练时冻结 IP2P 参数，仅训练 Emotion Adapter。使用 diffusion loss 保证像素级保真度，instruction loss 保证语义级情感表达。推理时可通过调节 image guidance scale 控制情感强度和结构保持的平衡。

## 实验关键数据

### 主实验

在 405 张测试图（8 个情感方向，共 3,240 对）上评估：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP-I↑ | Emo-A↑ | Emo-S↑ |
|------|-------|-------|--------|---------|--------|--------|
| SDEdit | 15.43 | 0.415 | 0.459 | 0.638 | 38.21% | 0.221 |
| PnP | 14.41 | 0.436 | 0.381 | 0.851 | 23.83% | 0.095 |
| ControlNet | 11.98 | 0.292 | 0.603 | 0.686 | 36.33% | 0.213 |
| CLVA | 12.61 | 0.397 | 0.479 | 0.757 | 14.04% | 0.017 |
| AIF | 14.05 | 0.537 | 0.493 | 0.828 | 12.74% | 0.004 |
| **EmoEdit** | **16.62** | **0.571** | **0.289** | **0.828** | **50.09%** | **0.335** |

### 人类评估

| 方法 | 结构保持↑ | 情感忠实↑ | 综合平衡↑ |
|------|---------|---------|---------|
| SDEdit | 11.71% | 10.85% | 5.07% |
| BlipDiff | 15.12% | 8.35% | 4.88% |
| **EmoEdit** | **70.12%** | **75.73%** | **89.12%** |

### 关键发现

- EmoEdit 在所有像素级指标（PSNR 16.62, SSIM 0.571, LPIPS 0.289）上最优，同时情感准确率（50.09%）远超所有方法
- 情感增量分数 Emo-S（0.335）比次优 SDEdit（0.221）高 52%，说明情感修改更有效
- 消融证实 Emotion Adapter 是必要的（去掉后图像几乎不变），instruction loss 保证语义清晰，diffusion loss 保证结构
- Emotion Adapter 可直接插入 ControlNet、BlipDiff 等其他模型增强其情感能力
- 可扩展到风格化图像生成（与 Composable Diffusion 结合），保持艺术风格的同时唤起情感

## 亮点与洞察

- **开创性地引入内容修改做情感操纵**：超越了颜色/风格调整的限制，从心理学出发构建了系统的"情感因子树"。这个数据构建范式可扩展到其他抽象属性的编辑
- **Emotion Adapter 的即插即用设计**：训练一次后可直接插入任何基于 IP2P 或 Stable Diffusion 的编辑/生成模型。这种模块化情感增强的思路有广泛应用价值
- **EmoEditSet 数据集贡献**：40,120 对带情感方向和内容指令的数据，可作为 AIM 领域的基础 benchmark

## 局限与展望

- 仅支持 Mikels 的 8 种情感类别，现实中情感远更复杂和细粒度
- 情感因子树高度依赖 EmoSet 的覆盖范围，可能存在偏差
- CLIP 图像相似度作为结构保持的代理指标不够精确
- 数据集构建依赖 IP2P 生成，受限于 IP2P 本身的编辑能力和质量上限
- 用户无法指定具体修改内容，完全依赖模型自动选择，有时会选择不恰当的语义

## 相关工作与启发

- **vs CLVA / AIF**: 这些是风格迁移类 AIM 方法，仅改变颜色风格但情感效果有限（Emo-S < 0.02）。EmoEdit 通过内容修改实现更强的情感唤起
- **vs InstructPix2Pix**: IP2P 能做具体指令的编辑但不理解情感；EmoEdit 的 Emotion Adapter 为 IP2P 注入情感知识，只需情感词即可工作
- **vs SDEdit**: SDEdit 有一定情感理解能力（Emo-A 38.21%），但严重破坏图像结构（PSNR 15.43, SSIM 0.415）。EmoEdit 在更好保持结构的同时情感效果更强

## 评分

- 新颖性: ⭐⭐⭐⭐ 从心理学出发做内容感知 AIM 是新方向，情感因子树+Adapter 设计有创意
- 实验充分度: ⭐⭐⭐⭐ 定量+定性+用户研究+消融+跨模型迁移，全面但缺少大规模对比
- 写作质量: ⭐⭐⭐⭐ 故事线清晰，图表直观，数据集构建过程描述详细
- 价值: ⭐⭐⭐⭐ 开辟了内容感知 AIM 新方向，数据集和 Adapter 有社区价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Make Me Happier: Evoking Emotions Through Image Diffusion Models](../../ICCV2025/image_generation/make_me_happier_evoking_emotions_through_image_diffusion_models.md)
- [\[CVPR 2025\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)
- [\[CVPR 2025\] Interpretable Generative Models through Post-hoc Concept Bottlenecks](interpretable_generative_models_through_post-hoc_concept_bottlenecks.md)
- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)
- [\[CVPR 2025\] TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)

</div>

<!-- RELATED:END -->
