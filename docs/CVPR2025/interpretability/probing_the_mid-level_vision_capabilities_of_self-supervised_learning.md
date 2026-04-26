---
title: >-
  [论文解读] Probing the Mid-Level Vision Capabilities of Self-Supervised Learning
description: >-
  [CVPR 2025][自监督学习] 本文从儿童视觉发育的视角出发，系统评估了 22 种自监督学习（SSL）模型在中层视觉任务（深度估计、表面法线、物体分割、几何对应等）上的能力，发现尽管 SSL 模型在高层语义任务上与监督模型存在较大差距，但在 3D 空间感知等中层视觉能力上差距显著更小。
tags:
  - CVPR 2025
  - 自监督学习
  - 中层视觉
  - 深度估计
  - 3D感知
  - 表征评估
---

# Probing the Mid-Level Vision Capabilities of Self-Supervised Learning

**会议**: CVPR 2025  
**arXiv**: [2411.17474](https://arxiv.org/abs/2411.17474)  
**代码**: 无  
**领域**: 自监督学习  
**关键词**: 自监督学习, 中层视觉, 深度估计, 3D感知, 表征评估

## 一句话总结

本文从儿童视觉发育的视角出发，系统评估了 22 种自监督学习（SSL）模型在中层视觉任务（深度估计、表面法线、物体分割、几何对应等）上的能力，发现尽管 SSL 模型在高层语义任务上与监督模型存在较大差距，但在 3D 空间感知等中层视觉能力上差距显著更小。

## 研究背景与动机

**领域现状**：自监督学习在 ImageNet 分类等高层语义任务上已取得了监督学习约 70% 的性能。然而，中层视觉能力——包括 3D 空间感知（深度、表面法线）、物体分割、几何对应等——在 SSL 评估中被严重忽视。这些能力在人类视觉发展中至关重要：婴儿在 1 岁前就发展出成熟的 3D 空间感知，远早于语义理解。

**现有痛点**：(1) SSL 模型的评估几乎完全集中在分类、检测等高层任务上，对中层视觉能力的研究空白；(2) 不同 SSL 方法（对比学习 vs 掩码建模 vs 聚类等）在中层任务上的优劣未知；(3) 缺乏系统性的基准来衡量 SSL 表征的 3D 空间感知质量。

**核心矛盾**：SSL 领域追求的是语义级表征质量（用分类精度衡量），忽视了视觉表征中的空间/几何信息，而后者对于机器人操控、导航、AR 等应用至关重要。

**本文目标**：用涵盖"3D 理解的第一前线"的多个中层视觉任务全面评估 SSL 模型，揭示哪些 SSL 方法最擅长学习 3D 空间表征。

**切入角度**：从发展心理学的洞察出发——婴儿从头戴相机视角的视觉经验中用极少监督就能发展出空间感知。如果用 200 小时婴儿头戴相机视频训练 SSL 模型（模拟儿童视觉），能否学到类似的中层视觉能力？

**核心 idea**：在 22 种 SSL 模型上系统评估 6 项中层视觉任务（物体分割、深度估计、表面法线、物体几何对应、场景几何对应、中层图像相似度），发现 SSL 模型的中层视觉能力远比高层语义能力更接近监督模型，且不同 SSL 方法在不同任务上表现各异。

## 方法详解

### 整体框架

纯评估性研究。选取 22 种主流 SSL 模型（覆盖 Jigsaw/RotNet/NPID/SimCLR/MoCo v2-v3/BYOL/SimSiam/SwAV/DINO/iBOT/MAE/MaskFeat 等），使用 ResNet-50 和 ViT-B/16 两种骨干，在 ImageNet-1K 上预训练。评估 6 项中层视觉任务，冻结特征提取器仅训练线性探测头或轻量级解码器。

### 关键设计

1. **全面的中层视觉评估体系**:

    - 功能：从多个维度评估 SSL 表征的空间/几何感知能力
    - 核心思路：6 项任务覆盖了中层视觉的核心能力：(a) **通用物体分割**（VOC07/VOC12，前景-背景二值分割，mIoU/F1/Acc）；(b) **深度估计**（NYU 室内深度和 NAVI 物体深度，$\delta_i$ 阈值精度和 RMSE）；(c) **表面法线估计**（角度误差和阈值精度）；(d) **物体几何对应**（3D 度量误差下的 recall）；(e) **场景几何对应**（2D 投影误差下的 recall）；(f) **中层图像相似度**（判断哪张图像在中层特征上更相似）。
    - 设计动机：中层视觉介于低层（边缘检测）和高层（分类）之间，是构建统一3D世界表征的关键。这6项任务从2D分组（分割）到3D几何（深度/法线/对应）再到相似度判断，层次递进地评估了空间感知的不同方面。

2. **多类别 SSL 方法对比**:

    - 功能：识别哪类 SSL 范式最有利于中层视觉表征学习
    - 核心思路：22 种 SSL 方法分为5大类：(a) 前置任务方法（Jigsaw、RotNet）——预测旋转角度/拼图排列；(b) 实例判别（NPID、PIRL）——将每张图像视为独立类别；(c) 对比学习（SimCLR、MoCo v2/v3、BYOL、SimSiam、Barlow Twins）——拉近同图像增强间距离；(d) 聚类方法（SwAV、DeepCluster-v2、SeLa-v2、ClusterFit）——在特征空间聚类分配伪标签；(e) 掩码建模（MAE、MaskFeat、iBOT）——重建被遮蔽的图像 patch。统一使用 ImageNet-1K 预训练以控制数据变量。
    - 设计动机：不同 SSL 范式的学习目标差异巨大——对比学习鼓励全局不变性，掩码建模鼓励局部重建，聚类鼓励语义聚合。这些不同的归纳偏置对中层视觉能力的影响需要被系统揭示。

3. **儿童视觉经验模拟实验**:

    - 功能：探索"类婴儿"视觉经验训练的 SSL 模型是否能获得中层视觉能力
    - 核心思路：使用 SAYCam 数据集中单个儿童 200 小时头戴相机视频（6-25 个月龄），训练嵌入模型和生成模型。评估这些模型在中层视觉任务上的表现，作为与 ImageNet 训练模型的对照。
    - 设计动机：如果从儿童视觉经验中就能学到实用的 3D 感知表征，这将揭示中层视觉能力可能不需要大规模多样化数据，而是源于时序一致性等更基本的信号。

### 损失函数 / 训练策略

纯评估工作，每个 SSL 模型使用其原始论文的预训练检查点。下游任务评估使用线性探测（冻结特征+线性头）或 DPT 解码器。

## 实验关键数据

### 主实验（SSL 模型中层视觉表现，ViT-B/16 骨干，选取代表性方法）

| 方法 | VOC12 mIoU | NYU 深度 $\delta_1$↑ | 表面法线 $\delta_1$↑ | 几何对应 Recall↑ |
|------|-----------|------------------|-------------------|----------------|
| MAE | 69.63 | 中等 | 中等 | 较低 |
| MoCo v3 | 74.11 | 中等 | 中等 | 中等 |
| DINO | **79.94** | 较高 | 较高 | **最高** |
| iBOT | **84.72** | **最高** | **最高** | 高 |

### 消融实验（高层 vs 中层差距对比）

| 任务层级 | SSL vs 监督模型性能比 |
|---------|-------------------|
| 高层语义（分类）| ~70% |
| 中层视觉（3D空间感知）| **~85-90%** |

### 关键发现

- **SSL 的中层视觉能力远优于预期**：虽然在高层语义任务上 SSL 仅达监督方法的 ~70%，但在 3D 空间感知任务上差距显著缩小（~85-90%），说明 SSL 天然倾向于学习空间结构信息。
- **iBOT 和 DINO 在中层任务上一致领先**：结合自蒸馏+掩码建模的 iBOT 在物体分割（mIoU 84.72%）和深度/法线估计上均最优。DINO 在几何对应上表现最佳。
- **掩码建模方法（MAE/MaskFeat）表现相对较差**：虽然在分类 fine-tuning 后表现好，但冻结特征的中层视觉能力不足，说明其学到的掩码重建信号更偏向低层纹理而非中层空间结构。
- **前置任务方法（Jigsaw/RotNet）出乎意料地有竞争力**：Jigsaw 通过预测 patch 排列直接学习了空间关系，在某些几何任务上不输对比学习方法。
- DenseCL（像素级对比学习）在需要空间精度的任务上有优势，验证了密集级自监督目标对中层视觉的益处。

## 亮点与洞察

- **从发展心理学视角审视 SSL**的跨学科框架非常有启发性：人类幼儿不需要语义标签就能发展3D空间感知，这与 SSL 的无标签学习范式高度一致。SSL 在中层视觉上的良好表现进一步支持了这一类比。
- **不同 SSL 目标→不同中层能力**的发现为 SSL 方法选择提供了实用指南：需要3D感知的应用应选择自蒸馏方法（DINO/iBOT），需要密集预测的应用应选择像素级方法（DenseCL）。
- 评估框架本身是对 SSL 社区的重要贡献——提供了超越分类精度的多维度表征质量评估。

## 局限与展望

- 仅使用线性探测/轻量解码器评估，未探索 full fine-tuning 下的中层视觉能力差异。
- 婴儿视频实验的规模较小（200小时单个儿童），结论的普遍性有待验证。
- 未包含近期的大规模 SSL 方法（如 DINOv2、I-JEPA 等）。
- 未来可探索专门为中层视觉能力设计的 SSL 目标函数，以及多任务 SSL 预训练的效果。

## 相关工作与启发

- **vs DINO**: DINO 被广泛认为是最好的 SSL 特征提取器之一，本文进一步证实了它在中层视觉上的优势，并揭示其自蒸馏机制是关键。
- **vs MAE**: MAE 在 fine-tuning 后的分类性能出色，但冻结特征的中层视觉能力较差，说明掩码重建目标学到的更多是纹理而非空间结构。
- **vs DUSt3R/MASt3R**: 近期的跨视图几何预训练方法在3D任务上表现出色，本文的发现暗示将自蒸馏（DINO）与跨视图几何目标结合可能是最优策略。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统评估 SSL 的中层视觉能力，跨学科视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 22 种模型、6 项任务、多个数据集，极其全面
- 写作质量: ⭐⭐⭐⭐ 发展心理学的引入充实了叙事
- 价值: ⭐⭐⭐⭐ 为 SSL 社区提供了重要的评估框架和见解

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/interpretability/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [\[ICCV 2025\] AIM: Amending Inherent Interpretability via Self-Supervised Masking](../../ICCV2025/interpretability/aim_amending_inherent_interpretability_via_self-supervised_masking.md)
- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)
- [\[CVPR 2025\] Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)

<!-- RELATED:END -->
