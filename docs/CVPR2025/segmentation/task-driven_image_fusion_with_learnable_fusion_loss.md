---
title: >-
  [论文解读] Task-driven Image Fusion with Learnable Fusion Loss
description: >-
  [CVPR 2025][图像分割][多模态图像融合] 本文提出 TDFusion，通过元学习方式训练一个损失生成模块，使融合损失函数能够根据下游任务（语义分割或目标检测）自适应地调整，从而让红外-可见光融合图像在下游任务上表现最优。
tags:
  - CVPR 2025
  - 图像分割
  - 多模态图像融合
  - 元学习
  - 可学习损失函数
  - 语义分割
  - 目标检测
---

# Task-driven Image Fusion with Learnable Fusion Loss

**会议**: CVPR 2025  
**arXiv**: [2412.03240](https://arxiv.org/abs/2412.03240)  
**代码**: https://github.com/HaowenBai/TDFusion  
**领域**: 分割/图像融合  
**关键词**: 多模态图像融合, 元学习, 可学习损失函数, 语义分割, 目标检测

## 一句话总结

本文提出 TDFusion，通过元学习方式训练一个损失生成模块，使融合损失函数能够根据下游任务（语义分割或目标检测）自适应地调整，从而让红外-可见光融合图像在下游任务上表现最优。

## 研究背景与动机

**领域现状**：多模态图像融合将红外与可见光图像的信息聚合为一张融合图像，广泛应用于语义分割、目标检测等下游任务。现有方法通常使用预定义的融合损失（如强度损失、梯度损失），或者通过级联下游任务网络引入任务损失来约束融合。

**现有痛点**：即使引入了下游任务，现有框架仍依赖固定的融合损失项，缺乏动态适应能力。手动定义的损失函数施加了预设的先验约束，无法根据具体图像对和任务需求灵活调整融合偏好。这导致融合结果对特定任务的适配性有限。

**核心矛盾**：融合损失的设计目标（视觉质量）与下游任务的需求（语义特征、检测精度）之间存在 gap。预定义损失无法动态捕捉不同任务对源图像信息的不同偏好——分割任务偏好边界和纹理，检测任务偏好目标区域的对比度。

**本文目标**：设计一个框架，使融合损失本身成为可学习的，让下游任务的损失直接驱动融合过程的优化方向。

**切入角度**：作者观察到融合损失的本质是控制"保留多少来自各源图像的强度信息"，将其参数化为逐像素权重 $w_a, w_b$，用一个神经网络（损失生成模块）来预测这些权重。

**核心 idea**：用元学习（MAML 风格）来训练损失生成模块——内层更新用融合损失更新融合网络的副本，外层用下游任务损失更新损失生成模块，使得融合损失的生成始终朝着最小化任务损失的方向演化。

## 方法详解

### 整体框架

TDFusion 包含三个模块：融合网络 $\mathcal{F}$（负责将红外和可见光图像融合为一张图）、下游任务网络 $\mathcal{T}$（如 SegFormer 或 YOLOv8）、损失生成模块 $\mathcal{G}$（输出可学习融合损失的参数）。训练时，融合网络和损失生成模块交替更新：先通过元学习流程（内层+外层）优化损失生成模块，然后用优化后的融合损失训练融合网络。

### 关键设计

1. **可学习融合损失（Learnable Fusion Loss）**:

    - 功能：根据下游任务需求自适应地生成逐像素的融合权重
    - 核心思路：融合损失由强度项和梯度项组成 $\mathcal{L}_f = \mathcal{L}_f^{int} + \alpha \mathcal{L}_f^{grad}$。强度项中，损失生成模块对输入的红外和可见光图像预测逐像素权重 $\{w_a, w_b\} = \mathcal{G}(I_a, I_b)$，Softmax 保证 $w_a^{ij} + w_b^{ij} = 1$，控制融合图像应更接近哪个源图像。梯度项用 Sobel 算子提取梯度，要求融合图像保留源图像中较大的梯度值。
    - 设计动机：传统方法的融合损失权重是固定的（如均分 $1/2$），无法区分不同任务、不同区域的信息偏好。可学习权重使模型能够针对每个像素决定偏好红外还是可见光信息。

2. **元学习训练策略（Meta-learning Training）**:

    - 功能：让下游任务损失驱动融合损失的优化
    - 核心思路：采用 MAML 风格的两阶段更新。**内层更新**：克隆融合网络 $\mathcal{F}'$ 和任务网络 $\mathcal{T}'$，用当前融合损失在 meta-training set 上各更新一步，得到中间参数 $\theta_{\mathcal{F}'}$ 和 $\theta_{\mathcal{T}'}$。**外层更新**：用 $\mathcal{F}'$ 在 meta-test set 上生成融合图像，计算下游任务损失 $\mathcal{L}_t$，反向传播更新损失生成模块 $\theta_{\mathcal{G}}$。关键在于内层更新保留了 $\theta_{\mathcal{F}'}$ 对 $\theta_{\mathcal{G}}$ 的计算图，使得二阶梯度可以传回损失生成模块。
    - 设计动机：直接用任务损失训练融合网络会使融合退化为任务特定的特征提取，丧失融合图像的视觉质量。元学习策略让任务损失间接地通过优化融合损失来指导融合，保持了融合损失框架的通用性。

3. **数据分割与交替训练**:

    - 功能：避免过拟合，确保损失生成模块的泛化能力
    - 核心思路：每个 epoch 从融合训练集中随机采样不重叠的 meta-training set 和 meta-test set（各 $M$ 对图像）。损失生成模块在这些子集上完成 $M$ 步元学习后，融合网络在完整训练集上用更新后的融合损失训练 $N$ 步。两者交替进行 $L$ 个 epoch。
    - 设计动机：meta-train/meta-test 的分离确保了损失生成模块不是在记忆特定图像对，而是学到了通用的任务偏好模式。交替更新保证融合损失在融合网络的不同训练阶段都能保持最优。

### 损失函数 / 训练策略

融合损失 $\mathcal{L}_f$ 包含可学习强度损失和固定梯度损失，权重 $\alpha=1$。下游任务损失 $\mathcal{L}_t$ 取决于具体任务：语义分割用交叉熵损失，目标检测用 YOLO 损失。使用 Adam 优化器，学习率 $1 \times 10^{-4}$，batch size 为 2。融合网络基于 Restormer Block 构建，损失生成模块同样使用 Restormer Block 加 Softmax 输出。

## 实验关键数据

### 主实验

在四个数据集（MSRS、FMB、M3FD、LLVIP）上进行融合和下游任务评估：

| 方法 | MSRS mIoU↑ | FMB mIoU↑ | M3FD mAP50↑ | LLVIP AP50↑ |
|---|---|---|---|---|
| TarDAL | 71.35 | 55.33 | 83.16 | 93.79 |
| SegMIF | 74.25 | 58.41 | 83.61 | 93.95 |
| EMMA | 74.48 | 56.28 | 83.71 | 94.00 |
| MRFS | 74.50 | 55.71 | 83.28 | 93.03 |
| TIMFusion | 73.58 | 57.24 | 83.22 | 93.76 |
| **TDFusion** | **75.09** | **60.50** | **86.27** | **95.00** |

### 消融实验

| 配置 | EN↑ | SF↑ | SCD↑ | VIF↑ | $Q^{AB/F}$↑ | SSIM↑ |
|---|---|---|---|---|---|---|
| 固定 $w_a=w_b=0.5$ | 6.60 | 13.73 | 1.58 | 0.39 | 0.60 | 0.72 |
| 去掉梯度损失 | 6.77 | 11.65 | 1.63 | 0.37 | 0.64 | 0.73 |
| $\theta_\mathcal{F}$ 同时受任务损失影响 | 6.80 | 13.85 | 1.70 | 0.41 | 0.66 | 0.73 |
| 去掉融合独立学习 | 6.82 | 14.07 | 1.72 | 0.41 | 0.67 | 0.72 |
| 用加权平均替代融合网络 | 6.75 | 11.49 | 1.65 | 0.38 | 0.62 | 0.73 |
| **完整 TDFusion** | **6.86** | **14.16** | **1.76** | **0.43** | **0.68** | **0.75** |

### 关键发现

- 固定权重 $w_a=w_b=0.5$ 相比可学习权重，SCD 从 1.76 降至 1.58，说明自适应权重是核心贡献
- 去掉梯度损失后 SF（空间频率）从 14.16 骤降至 11.65，说明梯度项对保留纹理细节至关重要
- 直接用任务损失训练 $\theta_\mathcal{F}$（Exp III）反而不如通过元学习间接优化，验证了元学习策略的必要性
- 可视化显示语义分割偏好可见光纹理+红外低光照信息，检测偏好红外中的高亮目标区域，两个任务的融合权重分布有明显差异

## 亮点与洞察

1. **"学习如何学习损失函数"的范式**：不是设计更好的损失函数，而是让元学习自动生成最适合下游任务的损失函数，这个思路具有很强的通用性，可迁移到其他无 ground truth 的任务
2. **融合权重的可解释性**：逐像素的 $w_a, w_b$ 可视化清楚地展示了不同任务对信息的偏好差异，为理解多模态融合提供了直观工具
3. **架构无关性**：框架对融合网络和任务网络的架构没有要求，可以即插即用地替换为任意网络

## 局限与展望

- 元学习的二阶梯度计算增加了训练成本，单 GPU（RTX 3090）训练依赖较小的 batch size
- 仅验证了红外-可见光融合场景，未扩展到 CT-MRI 等医学图像融合
- 损失生成模块的 Softmax 约束限制了 $w_a + w_b = 1$，无法表达"同时强调两个源"的情况
- 未来可以探索将可学习损失思路扩展到更多无监督任务（如超分辨率、去雾等）

## 相关工作与启发

- **vs TIMFusion**：TIMFusion 通过 NAS 搜索最优融合架构初始化，但仍使用固定融合损失；TDFusion 直接学习损失函数，更灵活
- **vs SegMIF**：SegMIF 将高级视觉任务特征嵌入融合过程，属于特征级引导；TDFusion 在损失级别引导，两者互补
- **vs ReFusion**：同一作者的前序工作，通过重建源图像来学习融合损失；TDFusion 改为直接用任务损失驱动，更加 end-to-end
- 元学习驱动损失设计的思路可以迁移到多模态学习的其他场景，如视觉-语言模型的对齐损失设计

## 评分

- 新颖性: 8/10 — 将元学习用于学习融合损失函数的想法新颖且优雅，但 MAML 框架本身已有成熟应用
- 实验充分度: 8/10 — 四个数据集+两个下游任务+完整消融，实验设计全面
- 写作质量: 7/10 — 公式推导详细但文章较长，核心思想可以更简洁地表达
- 价值: 7/10 — 对多模态融合领域有价值，但应用范围限于红外-可见光融合

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation](semantic_library_adaptation_lora_retrieval_and_fusion_for_open-vocabulary_semant.md)
- [\[NeurIPS 2025\] HCLFuse: Revisiting Generative Infrared and Visible Image Fusion Based on Human Cognitive Laws](../../NeurIPS2025/segmentation/revisiting_generative_infrared_and_visible_image_fusion_based_on_human_cognitive.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[CVPR 2026\] GeoSURGE: Geo-localization using Semantic Fusion with Hierarchy of Geographic Embeddings](../../CVPR2026/segmentation/geosurge_geo-localization_using_semantic_fusion_with_hierarchy_of_geographic_emb.md)
- [\[CVPR 2026\] EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](../../CVPR2026/segmentation/erecu_pseudolabel_evolution_unsupervised_camouflage.md)

<!-- RELATED:END -->
