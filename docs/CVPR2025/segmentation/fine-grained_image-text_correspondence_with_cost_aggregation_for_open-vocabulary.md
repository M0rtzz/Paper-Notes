---
title: >-
  [论文解读] Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation
description: >-
  [CVPR 2025][语义分割][开放词汇分割] PartCATSeg 通过将物体级和部件级的图文代价体积解耦聚合、引入组合损失约束部件构成关系、并利用 DINO 特征提供结构引导，在多个开放词汇部件分割基准上将 h-IoU 提升超过 10%。 领域现状：开放词汇部件分割 (OVPS) 是一个新兴方向…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "开放词汇分割"
  - "部件分割"
  - "代价聚合"
  - "视觉语言模型"
  - "DINO结构引导"
---

# Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation

**会议**: CVPR 2025  
**arXiv**: [2501.09688](https://arxiv.org/abs/2501.09688)  
**代码**: [https://github.com/jihochoi/PartCATSeg](https://github.com/jihochoi/PartCATSeg)  
**领域**: 分割  
**关键词**: 开放词汇分割, 部件分割, 代价聚合, 视觉语言模型, DINO结构引导

## 一句话总结

PartCATSeg 通过将物体级和部件级的图文代价体积解耦聚合、引入组合损失约束部件构成关系、并利用 DINO 特征提供结构引导，在多个开放词汇部件分割基准上将 h-IoU 提升超过 10%。

## 研究背景与动机

**领域现状**：开放词汇部件分割 (OVPS) 是一个新兴方向，旨在对训练时未见过的类别进行细粒度部件级分割。现有方法主要基于预训练视觉语言模型 (VLM)，如 VLPart 利用 DINO 特征建立基类和新类之间的部件对应，OV-PARTS 利用物体级上下文增强部件分割，PartCLIPSeg 通过注意力机制联合训练物体和部件。

**现有痛点**：OVPS 面临两大核心挑战。第一，部件级文本与视觉特征的对齐远比物体级困难——在 VLM 预训练数据中，部件级图文对的比例远低于物体级，导致部件级特征获得的监督更弱、噪声更大。CLIP 的可视化清楚表明，"head" 和 "wing" 等部件文本的图文相似性远弱于 "bird" 等物体级文本。第二，现有方法缺乏对部件间结构关系的理解，经常出现将"腿"误分为"尾巴"、将"喙"放到"尾部"等荒谬的分割错误。

**核心矛盾**：部件级监督信号的稀缺性与部件分割对精细对齐的高要求之间存在根本矛盾；同时，纯粹基于视觉特征匹配的方法忽略了物体内部件之间的结构约束，导致"部分-整体错觉"问题。

**本文目标** (1) 如何增强部件级图文对应而不被物体级信息所主导？(2) 如何在有限的部件标注下捕获部件-物体的组合关系？(3) 如何引入空间结构先验来区分外观相似但位置不同的部件？

**切入角度**：作者从代价聚合 (Cost Aggregation) 的视角出发，将物体和部件的图文对应分别建模为独立的代价体积，然后通过解耦的聚合策略分别优化。这个角度有希望是因为代价聚合在密集匹配任务中已被证明能有效减少匹配误差并增强泛化能力。

**核心 idea**：通过解耦物体级和部件级的代价体积并独立聚合，结合组合损失的部件-物体约束和 DINO 的结构引导，大幅提升开放词汇部件分割的图文对齐精度。

## 方法详解

### 整体框架

PartCATSeg 基于 CAT-Seg 架构进行扩展。输入一张图像和类别文本，首先用 CLIP 的视觉编码器和文本编码器分别提取密集视觉嵌入和类别文本嵌入。然后将类别名拆分为物体级名称 (如 "cat") 和部件级名称 (如 "paw")，分别计算两组代价体积。两组代价体积经过独立的空间聚合和类别聚合 Transformer 进行精炼，再通过线性投影融合为物体-部件联合代价体积，最终解码生成分割掩码。训练过程中同时施加解耦损失、联合损失和组合损失。

### 关键设计

1. **解耦代价聚合 (Disentangled Cost Aggregation)**:

    - 功能：将物体级和部件级的图文对应分开处理，避免部件级信号被物体级主导
    - 核心思路：将类别名解析为物体类 $\mathbf{C}_{\text{Obj}}$ 和部件类 $\mathbf{C}_{\text{Part}}$，分别计算余弦相似度代价体积 $\mathbb{C}_{\text{Obj}}$ 和 $\mathbb{C}_{\text{Part}}$。每个代价体积独立经过空间聚合 Transformer（基于 Swin Transformer 块捕获局部连续性）和类别聚合 Transformer（建模类间关系），产生精炼后的特征 $F''_{\text{Obj}}$ 和 $F''_{\text{Part}}$。两组预测分别用 BCE 损失监督
    - 设计动机：CLIP 预训练中部件级图文对比例远低于物体级，直接在统一代价体积上操作会导致部件信号被淹没。解耦处理让每个层级都有独立的表示空间来精炼对应关系

2. **物体感知的部件代价聚合 (Object-aware Part Cost Aggregation)**:

    - 功能：将物体级语义上下文注入部件级预测，形成"cat's paw"这样的物体特定部件表示
    - 核心思路：将精炼后的物体和部件特征通过线性投影融合为联合特征 $F_{\text{Obj-Part}}(i) = \text{Linear}([F''_{\text{Obj}}(i); F''_{\text{Part}}(i)])$，然后与物体特定部件文本嵌入（如 "cat's paw"）计算相似度，获得第三组代价体积 $\mathbb{C}_{\text{Obj-Part}}$。该代价体积再经过空间和类别聚合 Transformer 进一步精炼后送入解码器
    - 设计动机：部件名称（如 "leg"）在不同物体上的视觉外观可能完全不同，需要物体级上下文来消除歧义

3. **组合损失 (Compositional Loss)**:

    - 功能：利用"部件组成物体"的归纳偏置弥补部件级标注不足的问题
    - 核心思路：在每个空间位置，对物体代价体积和物体-部件代价体积分别做 softmax 归一化得到分布 $\mathbb{P}_{\text{Obj}}$ 和 $\mathbb{P}_{\text{Obj-Part}}$。通过预定义的部件-物体映射 $M$，将部件分布聚合回物体级分布 $\tilde{\mathbb{P}}_{\text{Obj}}$。然后用 Jensen-Shannon 散度约束聚合分布与直接预测的物体分布一致：$\mathcal{L}_{\text{comp}} = \frac{1}{2}(D_{\text{KL}}(\mathbb{P}_{\text{Obj}} \| \tilde{\mathbb{P}}_{\text{Obj}}) + D_{\text{KL}}(\tilde{\mathbb{P}}_{\text{Obj}} \| \mathbb{P}_{\text{Obj}}))$
    - 设计动机：部件标注稀缺，但物体标注充足。通过组合约束，可以利用物体级信息间接监督部件预测，确保部件分配覆盖整个物体区域，特别有助于小部件的识别

### 损失函数 / 训练策略

总损失为三部分之和：$\mathcal{L} = \mathcal{L}_{\text{Obj-Part}} + \mathcal{L}_{\text{disen}} + \lambda_{\text{comp}} \mathcal{L}_{\text{comp}}$。其中 $\mathcal{L}_{\text{disen}}$ 包含物体级和部件级的 BCE 损失，$\mathcal{L}_{\text{Obj-Part}}$ 是联合代价体积的 BCE 损失。训练沿用 CAT-Seg 的策略，微调 CLIP 编码器的 Query 和 Value 头。

## 实验关键数据

### 主实验

| 数据集 | 设置 | 指标 | PartCATSeg | 之前SOTA | 提升 |
|--------|------|------|------------|----------|------|
| Pascal-Part-116 | Pred-All | h-IoU | 45.77 | 30.67 (PartCLIPSeg) | +15.10 |
| Pascal-Part-116 | Oracle-Obj | h-IoU | 50.41 | 38.79 (PartCLIPSeg) | +11.62 |
| ADE20K-Part-234 | Pred-All | h-IoU | 24.19 | 11.38 (PartCLIPSeg) | +12.81 |
| ADE20K-Part-234 | Oracle-Obj | h-IoU | 49.96 | 41.83 (PartGLEE) | +8.13 |
| PartImageNet | Pred-All | h-IoU | 55.12 | 25.94 (PartCLIPSeg) | +27.79 |
| PartImageNet | Oracle-Obj | h-IoU | 72.66 | 53.85 (PartCLIPSeg) | +18.81 |

### 消融实验

| 配置 | Pred-All h-IoU | Oracle-Obj h-IoU | 说明 |
|------|----------------|------------------|------|
| Cost Agg (baseline) | 31.94 | 37.66 | 仅代价聚合 |
| + DINO | 43.28 | 48.55 | 加入结构引导，+11.34 |
| + DINO + L_comp (L1) | 43.36 | 49.57 | L1归一化的组合损失 |
| + DINO + L_comp (SM) | 45.77 | 50.41 | Softmax归一化效果最好 |

| 结构引导位置 | Pred-All h-IoU | Oracle-Obj h-IoU |
|------------|----------------|------------------|
| 无 | 33.65 | 37.60 |
| 仅物体级 | 38.61 | 42.25 |
| 仅部件级 | 44.41 | 51.35 |
| 两级都加 | 45.77 | 50.41 |

### 关键发现
- DINO 结构引导贡献最大，从 baseline 到加 DINO 提升了 11.34% h-IoU
- 组合损失在 Softmax 归一化下比 L1 归一化效果更好，因为 Softmax 鼓励每个位置主要归属于一个部件类别
- 结构引导在部件级比物体级更有效，说明 DINO 的价值在于捕获物体内部的细粒度结构信息而非仅仅区分前景背景
- 跨数据集评估中也保持显著优势，PartImageNet OOD 上无监督类别 mIoU 达 40.17%，比 PartCLIPSeg 提升 20.34%

## 亮点与洞察
- **解耦代价体积的思路**非常巧妙——与其在统一空间中让部件和物体竞争注意力，不如分开建模再融合；这个思路可以迁移到任何需要多粒度匹配的任务（如细粒度检索、层级分类）
- **组合损失是一种无需额外标注的自监督信号**——通过"部件之和等于物体"的先验，将充足的物体级标注间接转化为部件级监督，这个 trick 非常优雅且通用
- DINO 作为结构先验的使用方式有启发性——不是用来做特征匹配（VLPart 的做法），而是将其像素级特征作为空间聚合的 Query/Key 引导，让聚合过程感知空间结构

## 局限与展望
- 依赖预训练 CLIP 和 DINO 的特征质量，对于 VLM 未充分覆盖的领域（如医学影像的部件分割），效果可能受限
- 组合损失假设部件-物体映射是预定义的，无法处理动态发现新部件的场景
- 推理时需要计算三组代价体积和多个 Transformer 聚合，计算开销较大，论文未讨论效率问题
- 仅在语义分割场景验证，未探索实例级或全景分割等更复杂的设定

## 相关工作与启发
- **vs CAT-Seg**: CAT-Seg 是本文直接基于的框架，但只处理物体级 OVSS；本文通过解耦代价体积和组合损失将其扩展到部件级
- **vs PartCLIPSeg**: PartCLIPSeg 用注意力机制联合训练物体和部件，但缺乏显式的代价解耦和结构引导；PartCATSeg 在所有数据集上大幅超越
- **vs VLPart**: VLPart 用 DINO 做基类-新类的图像间匹配，本文则将 DINO 用于图文代价体积的空间聚合引导，利用方式更直接有效

## 评分
- 新颖性: ⭐⭐⭐⭐ 解耦代价聚合和组合损失的思路简洁有效，但核心框架仍基于 CAT-Seg 的扩展
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集 + 跨数据集评估 + 详尽的消融实验，结果令人信服
- 写作质量: ⭐⭐⭐⭐ 结构清晰，可视化丰富，动机阐述到位
- 价值: ⭐⭐⭐⭐⭐ 在所有基准上实现了 10%+ 的 h-IoU 提升，为 OVPS 领域设立了新的强基线

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation](../../CVPR2026/segmentation/pca-seg_revisiting_cost_aggregation_for_openvocabulary_semantic_and_part_segmentat.md)
- [\[CVPR 2025\] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)
- [\[CVPR 2025\] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](declip_decoupled_learning_for_open-vocabulary_dense_perception.md)

</div>

<!-- RELATED:END -->
