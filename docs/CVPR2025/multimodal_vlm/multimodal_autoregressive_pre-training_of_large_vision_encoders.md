---
title: >-
  [论文解读] Multimodal Autoregressive Pre-training of Large Vision Encoders
description: >-
  [CVPR 2025 (Highlight)][多模态VLM][视觉编码器] Apple提出AIMV2系列视觉编码器，通过将ViT编码器与一个多模态自回归解码器配对——同时生成原始图像patch和文本token作为预训练目标，在保持简洁训练流程的同时实现了跨任务的通用性能…
tags:
  - "CVPR 2025 (Highlight)"
  - "多模态VLM"
  - "视觉编码器"
  - "自回归预训练"
  - "多模态预训练"
  - "AIMV2"
  - "大规模视觉模型"
---

# Multimodal Autoregressive Pre-training of Large Vision Encoders

**会议**: CVPR 2025 (Highlight)  
**arXiv**: [2411.14402](https://arxiv.org/abs/2411.14402)  
**代码**: [https://github.com/apple/ml-aim](https://github.com/apple/ml-aim)  
**领域**: 多模态VLM  
**关键词**: 视觉编码器, 自回归预训练, 多模态预训练, AIMV2, 大规模视觉模型

## 一句话总结
Apple提出AIMV2系列视觉编码器，通过将ViT编码器与一个多模态自回归解码器配对——同时生成原始图像patch和文本token作为预训练目标，在保持简洁训练流程的同时实现了跨任务的通用性能，AIMV2-3B在ImageNet冻结主干评估中达到89.5%，并在多模态理解基准上全面超越CLIP和SigLIP。

## 研究背景与动机

**领域现状**：大规模视觉编码器的预训练是视觉和多模态AI的基础。当前主流路线有两条：(1) 对比学习路线（CLIP、SigLIP等），通过图文对比损失学习视觉-语言对齐的表征，在零样本分类和多模态理解上表现出色；(2) 自回归路线，如AIMv1，通过预测图像patch的下一个token来学习视觉表征，在纯视觉任务（分类、定位）上表现强劲。

**现有痛点**：对比学习模型擅长多模态但在纯视觉任务上不及专用模型（如DINOv2在定位任务上更优）；自回归模型擅长视觉但缺乏语言理解能力。目前没有一种预训练方式能让视觉编码器同时在**纯视觉**和**多模态**任务上都达到SOTA水平。此外，对比学习依赖大量高质量图文对且对负样本设计敏感，扩展性受限。

**核心矛盾**：对比目标学习的是图文之间的全局语义对齐（哪些图文匹配），但缺乏对视觉细节的建模；自回归目标学到丰富的视觉细节但没有语言接地(grounding)。如何让一个编码器同时具备两种能力？

**本文目标**：设计一种简洁、可扩展的预训练方法，使得单个视觉编码器在分类、定位、grounding、多模态理解等diverse任务上都表现优异。

**切入角度**：作者观察到，如果将自回归预训练从纯视觉扩展到多模态（同时生成图像patch和文本），编码器就需要同时理解图像的细粒度视觉内容和对应的高层语义描述，自然获得双重能力。关键是让解码器足够简单，避免解码器"偷懒"导致编码器学不到有效表征。

**核心idea**：用一个轻量级自回归解码器作为预训练任务的载体，输入为编码器的视觉特征，输出为图像本身的raw patches和对应文本的tokens。解码器在预训练完成后丢弃，只保留编码器。

## 方法详解

### 整体框架
AIMV2的训练架构分为编码器(encoder)和解码器(decoder)两部分。编码器为标准ViT（从L到3B参数量），将输入图像分成 $14 \times 14$ 的patch后编码为特征序列。解码器为autoregressive Transformer，接收编码器的特征序列作为前缀(prefix)，然后自回归地生成两种类型的token：(1) 图像raw patch token——直接回归每个patch的RGB像素值；(2) 文本token——对应图像的文本描述（如alt-text或caption）。两种token在同一个序列中交替或顺序排列。

### 关键设计

1. **多模态自回归解码目标 (Multimodal Autoregressive Objective)**:

    - 功能：同时学习视觉细节重建和语言语义理解
    - 核心思路：给定一张图像，编码器输出特征序列$\{z_1, ..., z_N\}$。解码器以此为prefix，依次预测：先是所有图像patch的raw pixel values（回归损失），再是对应文本的token序列（交叉熵损失）。总训练损失为$\mathcal{L} = \mathcal{L}_{patch} + \mathcal{L}_{text}$。图像patch预测使用均方误差（MSE），直接在像素空间回归；文本生成使用标准的next-token prediction
    - 设计动机：图像patch重建迫使编码器保留细粒度视觉信息（纹理、边缘、颜色），文本生成迫使编码器理解高层语义（物体类别、场景描述、动作等）。两个目标共同约束使编码器特征既细粒度又语义丰富。相比对比学习，这种方法不需要负样本设计，训练过程更简洁直接

2. **轻量级解码器设计 (Lightweight Decoder)**:

    - 功能：作为预训练的辅助模块，迫使编码器承担主要的信息提取责任
    - 核心思路：解码器参数量远小于编码器（大约是编码器的1/4到1/8）。这种不对称设计至关重要——如果解码器太强大，它可以仅凭自己的capacity完成patch重建和文本生成，编码器就退化为信息传递通道而非信息提取器。通过限制解码器容量，信息瓶颈被推向编码器，迫使编码器学到高质量的通用表征
    - 设计动机：这是"辅助任务训练视觉表征"的经典设计哲学：辅助模块（解码器）越弱，主模块（编码器）被迫学到的表征越好。类似思路在MAE（使用浅层解码器）中也被验证过

3. **从图像到文本的统一序列建模 (Unified Sequence Modeling)**:

    - 功能：消除图像和文本之间的模态边界，使模型自然地学习跨模态语义关联
    - 核心思路：图像patch和文本token被放在同一个自回归序列中，解码器按统一的next-token prediction方式处理它们。这意味着在生成文本token时，解码器可以attend到之前生成的图像patch token，建立图像细节到文本描述的直接联系。这种设计比分别用两个head处理图文更能学习到细粒度的图文对应关系
    - 设计动机：统一建模消除了多任务训练中常见的"任务冲突"问题——两个目标共享同一个解码器和训练流程，天然协调而非互相竞争

### 训练策略
预训练数据为大规模图文对数据集（类似CLIP使用的web-crawled数据）。训练采用标准的AdamW优化器，配合cosine learning rate scheduler。关键超参为patch重建损失和文本生成损失的权重比例。编码器从random initialization开始训练（不使用MAE/DINO等预训练权重作为初始化）。模型系列从0.3B (ViT-L)扩展到2.7B (ViT-3B)。

## 实验关键数据

### 主实验

| 任务/数据集 | AIMV2-L (0.3B) | AIMV2-3B (2.7B) | CLIP-L | SigLIP-L | DINOv2-L |
|-----------|---------------|----------------|--------|----------|----------|
| ImageNet-1k (冻结trunk) | 87.9 | 89.5 | 80.2 | 82.1 | 86.3 |
| COCO Detection (AP) | 56.2 | 59.4 | 48.7 | 50.3 | 57.8 |
| RefCOCO (grounding) | 81.1 | 83.6 | 72.4 | 74.8 | 79.2 |
| VQAv2 | 80.2 | 82.7 | 78.5 | 79.1 | 71.3 |
| TextVQA | 72.6 | 75.3 | 68.2 | 70.1 | 58.7 |
| MMBench | 74.5 | 76.3 | 69.8 | 71.2 | 62.5 |

### 消融实验

| 配置 | ImageNet Acc | VQAv2 Acc | 说明 |
|------|-------------|----------|------|
| Full AIMV2 (patch + text) | 87.9 | 80.2 | 完整多模态目标 |
| 仅patch重建 (w/o text) | 86.4 | 67.5 | 退化为AIMv1，多模态能力大降 |
| 仅text预测 (w/o patch) | 82.3 | 79.8 | 视觉细节能力下降 |
| 大解码器 (1/2编码器) | 86.1 | 77.4 | 解码器太强，编码器退化 |
| 小解码器 (1/8编码器) | 87.5 | 79.6 | 轻量解码器效果近似最优 |

### 关键发现
- **双目标协同效应显著**：去掉任一目标都导致对应能力大幅下降——去掉text目标，VQAv2从80.2降到67.5（-12.7%）；去掉patch目标，ImageNet从87.9降到82.3（-5.6%）。这证明了多模态自回归目标的必要性
- **编码器-解码器不对称性至关重要**：解码器参数量为编码器的1/4时效果最佳；增大到1/2时编码器性能反而下降，验证了"弱解码器逼出强编码器"的设计假设
- **扩展性极佳**：从0.3B到2.7B，ImageNet准确率从87.9稳步提升到89.5，没有饱和迹象。这说明自回归预训练的scaling law依然成立
- AIMV2在纯视觉任务（检测、grounding）上超越DINOv2，在多模态任务（VQA、MMBench）上超越CLIP/SigLIP，真正实现了"一个编码器走天下"
- 蒸馏版AIMV2-L-distilled从3B蒸馏，在0.3B参数量下多模态性能进一步提升（MMBench 76.3 vs 74.5）

## 亮点与洞察
- **"简洁即力量"的典范**：AIMV2的预训练过程极其简洁——一个编码器+一个解码器，两个损失函数，不需要负样本构造、动量编码器、teacher-student蒸馏等复杂技巧。这种简洁性不仅降低了工程复杂度，更重要的是使模型scaling变得直接
- **弱解码器设计的深刻洞察**：通过限制解码器容量来迫使编码器学到好表征，这一思想虽非首创但在多模态自回归场景下被验证得非常彻底。对其他需要辅助任务的场景有很好的指导意义
- **统一自回归建模消除模态壁垒**：将图像patch和文本token放在同一个自回归序列中，是一种优雅的多模态融合方式。这为构建真正统一的多模态基础模型提供了思路

## 局限与展望
- 预训练数据为web-crawled图文对，数据质量和偏见问题未被讨论
- 编码器在预训练后是冻结使用的，未评估微调场景下与对比学习模型的性能差异
- 解码器在推理时被丢弃，其知识未被利用；能否将解码器保留用于生成任务值得探索
- 仅评估了图像理解任务，未涉及视频理解，temporal维度的扩展有待验证
- 3B规模的编码器在推理时计算开销较大，实际部署需要考虑效率

## 相关工作与启发
- **vs CLIP / SigLIP**: 对比学习方法在多模态对齐上优势明显但缺乏视觉细粒度。AIMV2通过自回归patch重建弥补了这一不足，在多模态和纯视觉任务上双双超越
- **vs AIMv1**: AIMv1仅做图像自回归，缺乏语言能力。AIMV2通过加入文本生成目标，以极小成本获得了强大的多模态理解能力，同时纯视觉性能也有提升
- **vs DINOv2**: DINOv2基于自监督蒸馏在视觉任务上表现优异，但缺少语言接地。AIMV2-3B在检测和grounding上超越DINOv2-L，同时具备DINOv2不具备的多模态理解能力
- **vs MAE**: 同为自回归/重建式预训练，MAE在像素空间重建，AIMV2也回归raw patch但额外加入了文本目标，获得了MAE不具备的语义理解能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 将自回归预训练扩展到多模态的思路清晰自然，弱解码器设计有深度
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖分类/检测/grounding/多模态理解等众多任务，消融全面，scaling分析充分
- 写作质量: ⭐⭐⭐⭐⭐ 论述简洁有力，实验图表清晰
- 价值: ⭐⭐⭐⭐⭐ Apple开源3B视觉编码器，支持PyTorch/JAX/MLX，实用价值极高，是视觉编码器预训练领域的重要里程碑

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](../../ICCV2025/multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[ECCV 2024\] MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](../../ECCV2024/multimodal_vlm/mm1_methods_analysis_and_insights_from_multimodal_llm_pre-training.md)
- [\[CVPR 2025\] BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models](stealthy_backdoor_attack_in_self-supervised_learning_vision_encoders_for_large_v.md)
- [\[ICCV 2025\] OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning](../../ICCV2025/multimodal_vlm/openvision_a_fully-open_cost-effective_family_of_advanced_vision_encoders_for_mu.md)

</div>

<!-- RELATED:END -->
