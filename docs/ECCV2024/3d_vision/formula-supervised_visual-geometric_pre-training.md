---
title: >-
  [论文解读] Formula-Supervised Visual-Geometric Pre-training (FSVGP)
description: >-
  [ECCV 2024][3D视觉][视觉-几何表征学习] 提出FSVGP，利用分形几何的数学公式自动生成对齐的合成图像和点云，通过公式监督一致性标签在统一Transformer上实现跨模态视觉-几何预训练，在图像和3D物体的分类、检测、分割六项任务上均超越单模态FDSL方法。
tags:
  - ECCV 2024
  - 3D视觉
  - 视觉-几何表征学习
  - 合成数据预训练
  - 分形几何
  - Transformer
  - 公式驱动监督学习
---

# Formula-Supervised Visual-Geometric Pre-training (FSVGP)

**会议**: ECCV 2024  
**arXiv**: [2409.13535](https://arxiv.org/abs/2409.13535)  
**代码**: https://ryosuke-yamada.github.io/fdsl-fsvgp/ (项目页面)  
**领域**: 3D视觉  
**关键词**: 视觉-几何表征学习, 合成数据预训练, 分形几何, 统一Transformer, 公式驱动监督学习

## 一句话总结

提出FSVGP，利用分形几何的数学公式自动生成对齐的合成图像和点云，通过公式监督一致性标签在统一Transformer上实现跨模态视觉-几何预训练，在图像和3D物体的分类、检测、分割六项任务上均超越单模态FDSL方法。

## 研究背景与动机

**领域现状**: 图像（视觉）和点云（几何）的融合对增强视觉模型的3D场景理解至关重要，但目前视觉-几何表征学习的研究大多分别focus在提升图像或3D物体识别上，缺乏能同时增强两种模态的统一模型。

**现有痛点**: 大规模图像-点云配对数据集极度稀缺，高质量3D数据的采集成本高昂，图像与点云的对齐需要大量预处理，且标注通常需要专家手动标注复杂的3D空间信息。此外，真实数据集面临版权和伦理偏见问题。

**核心矛盾**: 现有FDSL（公式驱动监督学习）方法如VisualAtom和PC-FractalDB各自只在单一模态（图像或点云）上工作，无法实现跨模态的表征学习。

**本文目标**: 如何在不依赖真实数据的前提下，通过合成数据实现视觉和几何两种模态的统一预训练，使单一模型同时提升图像和3D物体的多种下游任务性能。

**切入角度**: 利用分形几何的数学公式同时生成图像和点云，并通过公式天然提供的跨模态对齐标签实现监督预训练。

**核心 idea**: 数学公式既能生成多模态数据，又天然提供跨模态的一致性标签，这比人工标注和对齐更经济高效。

## 方法详解

### 整体框架

FSVGP包含两个核心组件：(1) VG-FractalDB数据集构建——使用3D迭代函数系统(3D-IFS)生成分形点云，将其投影到2D平面获得分形图像，并利用公式参数得到跨模态一致性标签；(2) 统一Transformer预训练——对ViT和PointT做最小修改，同时输入分形图像和分形点云，用交叉熵损失进行分类预训练。

### 关键设计

1. **VG-FractalDB (视觉-几何分形数据库)**: 数据集定义为 $\mathcal{D} = \{(X_j, I_j, y_j)\}_{j=1}^{N}$，其中 $X_j$ 是分形点云，$I_j$ 是分形图像，$y_j$ 是公式监督一致性标签。分形点云通过3D-IFS生成，由仿射变换 $t_i(\mathbf{x}) = \mathbf{r}_i \mathbf{x} + \mathbf{b}_i$ 迭代产生 $T=8192$ 个3D坐标点。分形图像则通过虚拟相机投影 $I_j = \mathcal{F}_{\text{RGB}}(X_j; \mathbf{c})$ 得到。由于图像和点云共享同一个3D-IFS参数 $\Theta^c$，跨模态标签天然对齐，无需额外标注成本。设计动机：利用分形几何的自相似性生成足够复杂的视觉-几何结构，同时避免真实数据的版权和隐私问题。

2. **公式监督一致性标签 (Formula-Supervised Consistency Label)**: 每个分形类别 $c$ 由3D-IFS $\Theta^c$ 定义，3D点云和2D投影图像共享同一个标签 $y_j \in \{1, 2, \cdots, C\}$。通过方差阈值准则（阈值0.05）沿每个坐标轴过滤无效类别，并使用FractalNoiseMix技术（混入20%随机噪声点）增加实例多样性。设计动机：数学公式天然定义了跨模态的对应关系，省去了传统方法中昂贵的像素-点对应预处理步骤。

3. **统一Transformer模型**: 对ViT和PointT仅修改输入处理部分。分形图像被嵌入为图像token $\mathbf{z}_i = [x_{\text{class}}, \mathbf{z}_i^1, \dots, \mathbf{z}_i^{M_i}]$，分形点云被嵌入为点云token $\mathbf{z}_p = [x_{\text{class}}, \mathbf{z}_p^1, \dots, \mathbf{z}_p^{M_p}]$。两种模态共享class token $x_{\text{class}}$ 和用于分类的MLP层。设计动机：保持模型结构尽可能简洁，避免设计复杂的跨模态模块，确保预训练模型对多种下游任务的适用性。

### 损失函数 / 训练策略

- **损失函数**: 交叉熵损失 $\mathcal{L}_{\text{ce}}(f(\mathcal{D})) = -\frac{1}{N} \sum_{j=1}^{N} \sum_{c=1}^{C} y_{j,c} \log \hat{y}_{j,c}$，其中 $\hat{y}_j = f(X_j, I_j)$ 是统一模型对两种模态的联合输出。
- **训练配置**: VG-FractalDB-1k（1000个类别，每类1000个实例），使用AdamW优化器，batch size 64/GPU，初始学习率5e-4，权重衰减5e-2，训练200个epoch。16块NVIDIA V100约需60小时。

## 实验关键数据

### 主实验：FDSL方法对比（6项任务）

| 预训练数据集 | 图像分类(Acc.) | 图像检测(AP50) | 图像分割(AP50) | 3D分类(Acc.) | 3D检测(mAP25) | 3D分割(mIoU) |
|---|---|---|---|---|---|---|
| VisualAtom-21k | 91.3 | 66.3 | 63.3 | ✗ | ✗ | ✗ |
| PC-FractalDB-1k | ✗ | ✗ | ✗ | 83.3 | 63.0 | 83.7 |
| **VG-FractalDB-1k** | **92.0** | **68.3** | **65.6** | **83.7** | **63.7** | **84.1** |

### 图像分类详细比较

| 方法 | 类型 | C10 | C100 | Cars | Flowers | VOC12 | P30 | IN100 | 平均 |
|---|---|---|---|---|---|---|---|---|---|
| ImageNet-1k | SL | 99.0 | 89.6 | 81.9 | 99.1 | 86.5 | 82.1 | 93.1 | 90.2 |
| ImageNet-1k (MAE) | SSL | 99.1 | 90.1 | 91.3 | 99.8 | 90.2 | 82.8 | 94.1 | 92.5 |
| VisualAtom-21k | FDSL | 97.7 | 86.7 | 89.2 | 99.0 | 82.4 | 81.6 | 91.3 | 89.7 |
| **VG-FractalDB-1k** | FDSL | 98.1 | 85.9 | 89.2 | 99.5 | 83.5 | 81.7 | 92.0 | **90.0** |

### 消融实验

| 配置 | IN100(Acc.) | M40(Acc.) | 说明 |
|------|------------|-----------|------|
| ShapeNet (50k) | 87.3 | 92.7 | 使用CAD模型 |
| VG-FractalDB (50k) | 87.9 | 92.8 | 分形数据更优 |
| VG-PN-1k (Perlin噪声) | 90.7 | 92.6 | 替换生成规则 |
| VG-FractalDB-1k (分形) | **92.0** | **92.9** | 分形几何最优 |
| VG-FDB-1k (MAE自监督) | 80.3 | 92.8 | SSL不如FDSL |
| VG-FDB-1k (FSVGP) | **92.0** | **92.9** | 公式监督更有效 |

### 关键发现

- FSVGP用1/21的数据量（1M vs 21M）即超越VisualAtom-21k平均+0.3%
- VG-FractalDB-21k在384分辨率下达到83.8% ImageNet-1k精度，接近JFT-300M（84.2%），但仅用1/14数据
- 视觉+几何双模态预训练(V+G)在两种模态的下游任务上均优于单模态预训练
- 在3D检测mAP25上超越MaskPoint等SSL方法（63.7 vs 63.4）

## 亮点与洞察

- **极简但有效的跨模态设计**: 不需要复杂的跨模态注意力或对比学习模块，仅通过共享class token和分类MLP即实现跨模态学习
- **公式即对齐**: 数学公式天然提供了跨模态的对齐信号，完全绕过了传统方法中昂贵的人工标注和配对数据收集
- **合成数据的隐私优势**: 无版权、无隐私泄露、无社会偏见问题，这在数据合规日益重要的当下有显著意义
- **统一预训练范式**: 一次预训练同时服务6项下游任务（2模态 × 3任务），展示了视觉-几何表征学习的通用潜力

## 局限与展望

- 与ImageNet上的SSL方法（MAE、DINO）仍有差距，合成数据与真实数据的domain gap仍需解决
- 线性探测（linear probing）性能较弱，需要设计更高效的微调策略
- 仅使用单一虚拟相机视角（$v=1$）投影，多视角投影可能进一步丰富视觉模态的多样性
- 可扩展到自动驾驶中的鸟瞰图（BEV）和3D形状检索等更复杂应用场景

## 相关工作与启发

- **VisualAtom / PC-FractalDB**: 本文的直接前身，分别在图像和点云单模态上做FDSL预训练
- **CrossPoint / Pri3D**: 传统的视觉-几何表征学习方法，依赖真实配对数据和对比学习
- **启发**: 数学公式生成的合成数据可能是解决多模态对齐标注瓶颈的一个通用范式——不止适用于图像-点云，也可能推广到其他跨模态场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 用数学公式同时生成多模态数据+标签的思路非常巧妙且独特
- 实验充分度: ⭐⭐⭐⭐ 6项任务13个数据集的全面评估，消融实验探讨了生成规则、监督类型等核心问题
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，但部分符号定义略显冗余
- 价值: ⭐⭐⭐⭐ 在数据合规时代，合成预训练的路径很有前景，但与真实数据SSL的差距限制了直接实用性

<!-- RELATED:START -->

## 相关论文

- [E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](../../CVPR2026/3d_vision/e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)
- [4D Visual Pre-training for Robot Learning](../../ICCV2025/3d_vision/4d_visual_pretraining_for_robot_learning.md)
- [Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training](improving_domain_generalization_in_self-supervised_monocular_depth_estimation_vi.md)
- [Deep Patch Visual SLAM](deep_patch_visual_slam.md)
- [TCC-Det: Temporarily Consistent Cues for Weakly-Supervised 3D Detection](tcc-det_temporarily_consistent_cues_for_weakly-supervised_3d_detection.md)

<!-- RELATED:END -->
