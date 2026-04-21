---
title: >-
  [论文解读] Cross-Architecture Distillation Made Simple with Redundancy Suppression
description: >-
  [ICCV 2025][模型压缩][知识蒸馏] 提出RSD（Redundancy Suppression Distillation），通过跨架构不变性最大化和特征去相关来提取架构无关知识，仅用一个简单的RSD损失和轻量MLP解耦模块，在CIFAR-100和ImageNet-1k上大幅超越跨架构蒸馏先驱方法OFA，且参数开销仅为其小部分。
tags:
  - ICCV 2025
  - 模型压缩
  - 知识蒸馏
  - cross-architecture
  - redundancy suppression
  - feature decorrelation
  - CNN-ViT-MLP
---

# Cross-Architecture Distillation Made Simple with Redundancy Suppression

**会议**: ICCV 2025  
**arXiv**: [2507.21844](https://arxiv.org/abs/2507.21844)  
**代码**: 无  
**领域**: model_compression  
**关键词**: knowledge distillation, cross-architecture, redundancy suppression, feature decorrelation, CNN-ViT-MLP

## 一句话总结
提出RSD（Redundancy Suppression Distillation），通过跨架构不变性最大化和特征去相关来提取架构无关知识，仅用一个简单的RSD损失和轻量MLP解耦模块，在CIFAR-100和ImageNet-1k上大幅超越跨架构蒸馏先驱方法OFA，且参数开销仅为其小部分。

## 研究背景与动机
知识蒸馏（KD）旨在将预训练教师模型的能力转移给轻量学生模型。传统KD大多在同架构间进行（如CNN→CNN），但随着ViT、MLP-Mixer等新架构的出现，跨架构知识蒸馏（CAKD）变得越来越重要（最佳模型往往不适合部署）。→ 核心挑战：异构特征具有不同维度、不同甚至冲突的模式特征，强制学生盲目吸收教师特征会导致性能退化。→ 先驱方法OFA需要为不同架构设计定制的投影模块（如CNN用深度可分离卷积、ViT用注意力块），将特征投影到"架构无关"的logit空间，但这导致复杂设计和巨大参数开销（ConvNeXt-T→Swin-N时投影器参数是学生的3倍）。→ 本文的核心insight：不需要复杂的投影，只需通过冗余信息抑制来提取异构表示间的共性知识。

## 方法详解

### 整体框架
RSD作用于教师和学生的倒数第二层嵌入（penultimate-layer embeddings），通过一个轻量AAD解耦模块对齐维度后，计算跨架构的Pearson相关矩阵并施加RSD损失。总损失为$\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{RSD}$。训练结束后AAD丢弃，推理无额外开销。

### 关键设计
1. **冗余抑制蒸馏（RSD）损失**:

    - 功能：提取教师和学生表示中的架构无关共性知识
    - 核心思路：构建教师特征$\mathbf{z}^t$和学生特征$\mathbf{z}^s$间的Pearson相关矩阵$\mathbf{P} \in \mathbb{R}^{D \times D}$，优化目标为恒等矩阵$\mathbf{T} = I$。(1) 对角线元素→1：最大化同维度间的跨架构不变性（提取共性知识）；(2) 非对角线元素→0：去相关不同特征维度间的互信息（抑制冗余架构特有信息）。损失为$\mathcal{L}_{RSD} = d(\mathbf{P}(h(\mathbf{z}^s), \mathbf{z}^t), \mathbf{T})$，使用MSE距离。非对角线损失可加权系数κ调节
    - 设计动机：受经典无监督特征学习理论（Barlow Twins信息最大化、特征去相关）启发，互信息最小化等价于提取统计独立的架构无关特征

2. **架构无关知识解耦（AAD）模块**:

    - 功能：缓冲学生内部表示不被RSD目标完全覆盖，保留学生架构独有的有益能力
    - 核心思路：两层FC（expander + adaptor），中间接BatchNorm和GeLU激活。expander将学生嵌入映射到高维空间，adaptor对齐到教师嵌入维度
    - 设计动机：不同架构有独特的优势（如CNN的局部纹理敏感性是ViT不具备的），完全用架构无关知识覆盖会丢失这些能力。AAD作为缓冲层，让RSD优化作用在投影后的表示上，而非直接修改学生内部表示

3. **选择倒数第二层嵌入的设计考量**:

    - 功能：避免中间特征的复杂维度对齐问题
    - 核心思路：倒数第二层嵌入始终是1D向量（非feature map或token），无需架构特定操作（深度可分离卷积/token操作等）。比中间特征更接近网络输出，架构特有性更弱，更适合提取架构无关信息
    - 设计动机：这正是OFA复杂性的根源——其需要对不同架构的中间特征设计不同的投影模块

### 损失函数 / 训练策略
RSD损失可用约8行PyTorch代码实现：归一化特征→计算互相关矩阵→对角线MSE+非对角线加权MSE。遵循OFA的训练配置。RSD也可应用在logit空间，作为logit蒸馏器同样表现优异。

## 实验关键数据

### 主实验
CIFAR-100（12对异构教师-学生，部分展示）：

| 教师→学生 | From Scratch | KD | OFA | RSD | RSD vs OFA |
|-----------|-------------|-----|-----|-----|------------|
| Swin-T→ResNet18 | 74.01 | 78.74 | 80.54 | **83.92** | +3.38 |
| ViT-S→MobileNetV2 | 73.68 | 72.77 | 78.45 | **81.68** | +3.23 |
| ConvNeXt-T→DeiT-T | 68.00 | 72.99 | 75.76 | **82.46** | +6.70 |
| ConvNeXt-T→ResMLP-S12 | 66.56 | 72.25 | 81.22 | **84.21** | +2.99 |
| **平均增益** | - | +3.17 | +7.47 | **+10.69** | **+3.22** |

ImageNet-1k（15对异构教师-学生，部分展示）：

| 教师→学生 | From Scratch | OFA | RSD | RSD vs OFA |
|-----------|-------------|-----|-----|------------|
| Swin-T→ResNet18 | 69.75 | 71.85 | **72.13** | +0.28 |
| ConvNeXt-T→Swin-N | 75.53 | 77.50 | **77.70** | +0.20 |
| ConvNeXt-T→ResMLP-S12 | 76.65 | 77.53 | **78.41** | +0.88 |
| **平均增益** | - | +2.20 | **+2.34** | **+0.14** |

### 消融实验

| 配置 | Swin-T→ResNet18 | ConvNeXt-T→ResMLP-S12 | 说明 |
|------|-----------------|----------------------|------|
| Baseline (scratch) | 74.01 | 76.65 | 无蒸馏 |
| + RSD-corr only | 80.65 | 83.40 | 仅不变性最大化 |
| + RSD-decorr | **83.92** | **84.21** | 加入去相关（完整RSD） |

AAD效果（CIFAR-100/ImageNet）：

| 配置 | ViT-S→ResMLP | ConvNeXt-T→Mixer |
|------|-------------|-----------------|
| RSD完整 | 82.94 | 80.73 |
| w/o AAD | 82.26 (-0.68) | 79.93 (-0.80) |

参数开销对比（ConvNeXt-T→Swin-N @ ImageNet）：

| 方法 | 学生参数 | 额外参数 | 额外/学生比 |
|------|---------|---------|------------|
| OFA | 9.6M | 28.2M | 2.94x |
| **RSD** | 9.6M | ~2.8M | **0.29x** |

RSD作为logit蒸馏器：

| Logit Loss | Swin-T→ResNet18 | ConvNeXt-T→ResMLP |
|-----------|-----------------|-------------------|
| KD | 78.74 | 72.25 |
| DKD | 80.26 | 73.22 |
| OFA (仅logit部分) | 80.60 | 78.87 |
| **RSD on logits** | **83.23** | **81.15** |

### 关键发现
- RSD在CIFAR-100上平均增益+10.69%，显著超越OFA的+7.47%
- ConvNeXt-T→DeiT-T上RSD领先OFA 6.70%，几乎等于OFA与无蒸馏的差距
- RSD单独作为logit蒸馏器就能超越OFA的完整框架（含所有复杂投影器）
- 去相关目标在大多数情况下进一步提升性能
- RSD可与OFA互补：替换OFA中的所有loss为RSD loss带来额外提升
- CKA可视化表明RSD在中层和深层显著增加了异构架构间的特征相似性

## 亮点与洞察
- "简而有效"的典范：8行代码的RSD损失超越了复杂得多的OFA，参数开销仅为1/10
- 冗余抑制视角对跨架构蒸馏问题的重新定义非常精准：不是学习如何对齐异构特征，而是学习如何去除架构特有的冗余信息
- 选择倒数第二层嵌入而非中间特征是一个聪明的设计决策，彻底规避了异构特征对齐难题
- AAD的"保留学生独有能力"设计体现了对蒸馏本质的深入理解

## 局限与展望
- 在ImageNet上优势不如CIFAR-100显著（+2.34% vs +10.69%），大规模数据集上还有提升空间
- 仅使用1D嵌入，无法利用2D特征图的丰富空间信息，限制了向检测等空间敏感任务的扩展
- 超参数λ和κ有一定敏感性，需要调优
- 未探索更多下游任务（如目标检测、语义分割）的跨架构蒸馏

## 相关工作与启发
- Barlow Twins的信息最大化+去相关原理被巧妙地转化为跨架构蒸馏目标
- 与域泛化中的"域不变表示学习"有概念层面的联系，但上下文和方法论本质不同
- RSD的简洁性和通用性使其有潜力成为CAKD领域的基准方法

## 评分
- 新颖性: ⭐⭐⭐⭐ 冗余抑制视角新颖，但核心技术（相关矩阵+去相关）借鉴自SSL领域
- 实验充分度: ⭐⭐⭐⭐⭐ 12+15对异构模型，CIFAR+ImageNet验证，消融/兼容性/可视化全面
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，动机推导自然，与OFA的对比分析透彻
- 价值: ⭐⭐⭐⭐⭐ 简单有效的方法对社区价值极大，有望成为CAKD的强基线

<!-- RELATED:START -->

## 相关论文

- [Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification](competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif.md)
- [Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](../../NeurIPS2025/model_compression/universal_cross-tokenizer_distillation_via_approximate_likelihood_matching.md)
- [DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation](../../CVPR2025/model_compression/delt_a_simple_diversity-driven_earlylate_training_for_dataset_distillation.md)
- [Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)
- [Simple Unsupervised Knowledge Distillation With Space Similarity](../../ECCV2024/model_compression/simple_unsupervised_knowledge_distillation_with_space_similarity.md)

<!-- RELATED:END -->
