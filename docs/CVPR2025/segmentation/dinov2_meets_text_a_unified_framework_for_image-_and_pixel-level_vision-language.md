---
title: >-
  [论文解读] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment
description: >-
  [CVPR 2025][图像分割][DINOv2] 提出 dino.txt，通过冻结 DINOv2 视觉编码器 + 从头训练文本编码器的 LiT 策略，创新性地用 [CLS]+平均池化拼接作为图像表征，结合文本+图像双模态数据平衡，仅用 50K 迭代（CLIP 训练成本的几分之一）即在零样本分类和开放词汇分割上达到 SOTA。
tags:
  - CVPR 2025
  - 图像分割
  - DINOv2
  - 视觉语言对齐
  - LiT
  - 零样本分类
  - 开放词汇分割
---

# DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment

**会议**: CVPR 2025  
**arXiv**: [2412.16334](https://arxiv.org/abs/2412.16334)  
**代码**: 无（Meta FAIR 内部项目）  
**领域**: 分割 / 多模态VLM  
**关键词**: DINOv2, 视觉语言对齐, LiT, 零样本分类, 开放词汇分割

## 一句话总结

提出 dino.txt，通过冻结 DINOv2 视觉编码器 + 从头训练文本编码器的 LiT 策略，创新性地用 [CLS]+平均池化拼接作为图像表征，结合文本+图像双模态数据平衡，仅用 50K 迭代（CLIP 训练成本的几分之一）即在零样本分类和开放词汇分割上达到 SOTA。

## 研究背景与动机

**领域现状**：DINOv2 等自监督视觉基础模型能产生强大的通用特征，在分类、分割、匹配等下游任务中表现优异。然而，这些模型的特征空间缺乏与语言的对齐，无法直接用于零样本识别和开放词汇分割等需要文本接口的任务。另一方面，CLIP 等视觉语言模型虽然对齐了视觉和语言，但从头训练成本极高，且 patch-level 的密集特征质量不如自监督模型。

**现有痛点**：直接将 Locked-image Text Tuning (LiT) 应用到 DINOv2 上效果不理想——分类尚可，但密集任务（分割、检索）表现很差。原因有二：(1) CLIP/LiT 训练范式仅对比全局图像-文本表征，梯度无法传到 patch 特征；(2) 冻结视觉编码器导致视觉预训练数据与 LiT 训练数据之间存在域差距。

**核心矛盾**：如何在保留 DINOv2 强大密集特征质量的同时，低成本地为其添加语言接口？如何在一个训练目标下同时实现图像级和像素级的视觉-文本对齐？

**本文目标** (1) 全局对齐（分类/检索）+密集对齐（分割）的统一方案；(2) 冻结视觉编码器下的域差距弥合；(3) 高效训练数据策略。

**切入角度**：不需要从头训练或微调 DINOv2 的视觉编码器，只需用正确的图像表征（CLS+平均池化拼接）来训练文本编码器对齐，再加上轻量的可训练视觉块来弥合域差距。同时通过文本+图像双模态数据平衡来提升训练效率。

**核心 idea**：用 [CLS] 拼接 patch 平均池化作为对齐目标，让一个损失同时驱动全局和密集特征的文本对齐。

## 方法详解

### 整体框架

dino.txt 的架构分为三个部分：冻结的 DINOv2 视觉编码器、2 层可训练视觉 Transformer 块（弥合域差距）、从头训练的文本编码器。图像通过冻结的 DINOv2 得到 [CLS] token 和 N 个 patch token，经过 2 层可训练视觉块后，将更新后的 [CLS] 与 patch 平均池化拼接成 2D 维的全局描述符 g，与文本 [EOS] token 做对比学习。推理时，分类用全局描述符 g，分割用每个 patch 的输出特征与文本的 patch 对齐部分做余弦相似度。

### 关键设计

1. **[CLS]+平均池化拼接表征 (CLS-AvgPool Concatenation)**:

    - 功能：统一全局和密集对齐的图像表征
    - 核心思路：将 [CLS] token（全局语义）与所有 patch token 的平均池化（密集信息的聚合）拼接成 $\mathbf{g} = [\mathbf{c'}; \sigma([\mathbf{f_1'}, \cdots, \mathbf{f_N'}])]$，维度为 $2D$。对比损失在 g 和文本 embedding 之间计算，梯度同时流向 [CLS] 和每个 patch token
    - 设计动机：单用 [CLS] 分类好但分割差，单用平均池化分割好但分类差。拼接后两者兼得——这是本文最关键的发现。表 2 消融显示 [CLS avg] 组合在分类、检索、分割三项任务上均为最优

2. **可训练视觉块 (Learnable Vision Blocks)**:

    - 功能：弥合冻结视觉编码器的预训练数据与 LiT 训练数据之间的域差距
    - 核心思路：在冻结的 DINOv2 顶部添加 2 层可训练 Transformer 块 $\psi$，保持输出维度 D 不变。这些块可以让视觉特征适应新的训练数据分布
    - 设计动机：直接 LiT 训练 DINOv2 在检索和分割上效果差，加 2 层视觉块后显著提升。比微调整个视觉编码器参数少得多

3. **文本+图像双模态数据平衡 (Text and Image Based Curation)**:

    - 功能：构建概念均衡的训练数据集
    - 核心思路：文本侧用 MetaCLIP 的 WordNet 查询平衡策略，图像侧用基于 DINOv2 特征的层次化 k-means 聚类平衡策略（3 级，20M/800K/80K 质心），最终取两者交集。从 23 亿图文对池中每 epoch 采样约 6.5 亿对
    - 设计动机：互联网图文对的标题噪声严重，仅靠文本平衡不足以保证视觉概念的均匀覆盖。实验证明双模态平衡比单模态平衡提升 1-2 个点

### 损失函数 / 训练策略

- 使用标准 CLIP 对比损失，batch size 32K-65K
- 仅训练 50K 迭代（约看 16-32 亿图文对），远少于 CLIP 的训练量
- 冻结视觉编码器节省显存，允许更大 batch size
- 分割推理时使用滑窗策略，高分辨率版本采样不同大小的 crop 并聚合

## 实验关键数据

### 主实验

零样本分类：

| 方法 | 视觉编码器 | IN1K | IN-v2 | ObjNet |
|------|-----------|------|-------|--------|
| CLIP ViT-L/14 | 训练 | 75.3 | 69.8 | 57.1 |
| OpenCLIP ViT-G/14 | 训练 | 80.1 | 73.6 | 63.8 |
| **dino.txt ViT-L/14** | **冻结** | **81.1** | **74.3** | **65.2** |

开放词汇分割：

| 方法 | ADE20K | Cityscapes | COCO-Stuff | VOC20 |
|------|--------|-----------|------------|-------|
| TCL | 24.3 | 30.4 | 19.6 | 77.5 |
| GroupViT | 10.6 | 11.1 | 15.3 | 79.7 |
| **dino.txt ViT-L/14** | **29.5** | **40.0** | **24.6** | **73.6** |
| **dino.txt (高分辨率)** | **37.2** | - | - | - |

### 消融实验

| 池化方式 | IN1K 分类 | COCO 检索 | ADE 分割 |
|---------|----------|----------|---------|
| [CLS] 仅 | 78.8 | 30.2 | 8.3 |
| [avg] 仅 | 74.7 | 32.7 | 13.3 |
| [CLS avg] 拼接 | **79.2** | **34.7** | **18.2** |

| 数据平衡策略 | IN1K | COCO | ADE |
|------------|------|------|-----|
| 仅文本平衡 | 79.2 | 34.7 | 18.2 |
| 仅图像平衡 | 78.9 | 33.9 | 18.0 |
| **文本+图像双平衡** | **80.3** | **37.5** | **20.3** |

### 关键发现

- [CLS]+平均池化拼接是本文最关键的设计：分类+4.5 vs 仅用 [avg]，分割+9.9 vs 仅用 [CLS]
- 加 2 层视觉块后分割从 18.2→20.7，检索从 34.7→39.2，效果显著
- 双模态数据平衡比单模态一致地提升 1-2 个点
- DINOv2 冻结后作为 LiT 的视觉编码器效果优于其他自监督模型（MAE、I-JEPA、DINO v1）
- 50K 迭代即可收敛，训练效率极高

## 亮点与洞察

- **简洁优雅的统一表征**：一个拼接操作同时解决了全局和密集对齐的矛盾，避免了复杂的多任务损失或专门的密集对齐模块。DINOv2 的 patch 特征本身就有很好的空间定位能力，只需要让梯度能流过去即可
- **训练效率的极致**：仅 50K 迭代（CLIP 需要数十万迭代），得益于冻结编码器允许超大 batch + 高效数据平衡。这大幅降低了语言-视觉对齐的准入门槛
- **图像侧数据平衡的重要性**：揭示了互联网图文对中文本和图像的概念分布严重不匹配，仅靠文本平衡是不够的。这个洞察对所有基于网络爬取数据的 VLM 训练都有指导意义

## 局限与展望

- 分割推理质量高度依赖滑窗策略的 crop 数量和分辨率，高分辨率推理需要约 800 个 crop，耗时 10 秒
- 检索性能与 CLIP 仍有差距，冻结编码器不适应新数据可能是根本限制
- 当 DINOv2 的 patch 特征对某些概念的空间定位本身不好时，dino.txt 也无能为力
- 没有探索在分割上使用 DINOv2 的中间层特征（多尺度融合可能进一步提升）
- 仅使用弱监督（图文对），若结合像素级标注的半监督训练可能效果更好

## 相关工作与启发

- **vs CLIP/OpenCLIP**: CLIP 从头训练两个编码器，成本极高；dino.txt 冻结视觉编码器只训练文本侧，效率高一个数量级，且利用了 DINOv2 的强大密集特征
- **vs CLIPpy**: CLIPpy 也尝试微调 SSL backbone 配合文本对齐，但分类性能下降。dino.txt 通过 [CLS]+avg 拼接避免了这个 trade-off
- **vs MaskCLIP**: MaskCLIP 需要特殊的推理适配（取 Value embedding 绕过注意力），而 dino.txt 直接使用最终 patch token 就能做分割，更简洁

## 评分

- 新颖性: ⭐⭐⭐⭐ [CLS]+avg 拼接的想法简单但效果惊人，双模态数据平衡有实际价值
- 实验充分度: ⭐⭐⭐⭐⭐ 分类、检索、分割三类任务全面评估，消融详尽
- 写作质量: ⭐⭐⭐⭐ 论述清晰，动机链条完整，但部分实验细节分散
- 价值: ⭐⭐⭐⭐⭐ 为 DINOv2 等 SSL 模型解锁语言接口，实际应用价值极高

<!-- RELATED:START -->

## 相关论文

- [Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)](assessing_and_learning_alignment_of_unimodal_vision_and_language_model.md)
- [SAIL: Assessing and Learning Alignment of Unimodal Vision and Language Models](assessing_and_learning_alignment_of_unimodal_vision_and_language_models.md)
- [UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation](fine-grained_image-text_correspondence_with_cost_aggregation_for_open-vocabulary.md)
- [Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)

<!-- RELATED:END -->
