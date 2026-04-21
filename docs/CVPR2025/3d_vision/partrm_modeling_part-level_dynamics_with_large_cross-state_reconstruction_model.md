---
title: >-
  [论文解读] PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model
description: >-
  [CVPR 2025][3D视觉][部件级动力学] PartRM 提出了一个基于大规模3D高斯重建模型的4D重建框架，能够从多视图图像同时建模物体的外观、几何和部件级运动，通过构建 PartDrag-4D 数据集、多尺度拖拽嵌入模块和两阶段训练策略，在部件级运动学习上达到 SOTA，并可应用于机器人操作任务。
tags:
  - CVPR 2025
  - 3D视觉
  - 部件级动力学
  - 4D重建
  - 3D高斯
  - 拖拽交互
  - 机器人操作
---

# PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model

**会议**: CVPR 2025  
**arXiv**: [2503.19913](https://arxiv.org/abs/2503.19913)  
**代码**: https://PartRM.c7w.tech/  
**领域**: 3D视觉  
**关键词**: 部件级动力学, 4D重建, 3D高斯, 拖拽交互, 机器人操作

## 一句话总结

PartRM 提出了一个基于大规模3D高斯重建模型的4D重建框架，能够从多视图图像同时建模物体的外观、几何和部件级运动，通过构建 PartDrag-4D 数据集、多尺度拖拽嵌入模块和两阶段训练策略，在部件级运动学习上达到 SOTA，并可应用于机器人操作任务。

## 研究背景与动机

**领域现状**：世界模型（World Model）需要根据当前观测和动作预测未来状态，其中部件级动力学建模（如抽屉滑动、门旋转）对机器人操作、AR/VR 等应用至关重要。现有方法如 Puppet-Master 通过微调大规模视频扩散模型来实现拖拽控制下的物体运动生成。

**现有痛点**：Puppet-Master 等方法存在两个核心缺陷：（1）输出仅为单视图视频，无法直接提供模拟器所需的3D表示，需要额外使用单目重建模型，引入误差；（2）扩散去噪过程耗时数分钟，无法满足快速试错生成操作策略的需求。

**核心矛盾**：2D视频表示与3D应用需求之间的鸿沟，以及生成速度与实时交互需求之间的矛盾。

**本文目标**：同时建模物体的外观、几何和部件级运动，生成可从任意视角渲染的3D表示，且推理速度快。

**切入角度**：作者观察到大规模3D高斯重建模型（如 LGM）已经具备了静态物体的外观和几何建模先验，部件级运动与几何天然关联（如抽屉沿法线方向滑动），因此可以在重建模型基础上扩展运动建模能力。

**核心 idea**：在预训练的大规模3D高斯重建模型上扩展4D能力，通过拖拽条件建模部件运动，用两阶段训练避免灾难性遗忘。

## 方法详解

### 整体框架

给定单视图观测图像和2D拖拽交互信息，PartRM 首先通过微调的 Zero123++ 生成多视图图像，然后通过拖拽传播模块将单个拖拽扩展到运动部件的整个区域。多视图图像和拖拽信息被送入基于 LGM 的 U-Net 网络，输出表示变形后状态的3D高斯表示。整个流程采用两阶段训练：第一阶段学运动，第二阶段学外观。

### 关键设计

1. **PartDrag-4D 数据集**:

    - 功能：提供部件级动力学的多视图训练数据
    - 核心思路：基于 PartNet-Mobility 数据集，选取 738 个跨 8 个类别的铰接物体网格，对每个物体的可动部件在极限位置之间设置 6 个阶段，同时随机化其他部件位置，共产生 20,548 个状态。每个状态用 Blender 渲染 12 个视图，并在运动部件表面采样拖拽点
    - 设计动机：现有4D数据要么缺少3D信息，要么使用 Objaverse 中的通用动画数据（包含变形等不符合运动学动力学的操作），需要一个专门符合铰接运动学的数据集

2. **拖拽传播与多尺度嵌入模块**:

    - 功能：将单个拖拽交互扩展为覆盖整个运动部件的拖拽提案，并在多个分辨率尺度上嵌入到 U-Net 中
    - 核心思路：传播阶段使用 SAM 对运动部件进行分割，在分割 mask 上采样新的起始点，保持与原始拖拽相同的方向和强度。嵌入阶段对每个拖拽点用 Fourier 编码 + 3层 MLP 得到特征嵌入，构建与 U-Net 下采样块输出尺寸匹配的多尺度拖拽图 $M_{t,l}$，通过拼接和卷积与特征图交互：$I_{l+1} = O_l + \text{Conv}(M_{t,l} \oplus O_l)$
    - 设计动机：单个拖拽条件有歧义，会导致模型幻觉；多尺度嵌入让网络在不同粒度上理解拖拽运动——大尺度捕获局部精细信息，小尺度捕获全局运动模式

3. **两阶段训练策略**:

    - 功能：在微调中防止预训练的外观和几何建模能力灾难性遗忘
    - 核心思路：第一阶段（运动学习）使用知识蒸馏方法，将预训练 LGM 在目标状态观测上推理得到的高斯参数作为监督信号，直接对 splatter image 的 14 维参数施加 L2 损失；第二阶段（外观学习）使用 MSE + LPIPS + alpha MSE 的渲染损失联合优化外观、几何和运动
    - 设计动机：如果只用渲染损失监督（Stage 2），模型倾向于利用损失函数漏洞，不真正学习运动；先学运动再联合优化，实现由粗到精的训练

### 损失函数 / 训练策略

Stage 1 使用 splatter image 像素级 L2 损失：$\mathcal{L}_1 = \sum \|\mathcal{GS}_i - \mathcal{GS}_j\|_2^2$，其中 $i, j$ 是对应像素。Stage 2 使用渲染损失：$\mathcal{L}_2 = L_{\text{mse}} + \lambda_1 L_{\text{lpips}} + \lambda_2 L_{\text{mse}}^{\alpha}$，$\lambda_1 = \lambda_2 = 1.0$。

## 实验关键数据

### 主实验

| 方法 | 设置 | PSNR↑ | SSIM↑ | LPIPS↓ | 时间 |
|------|------|-------|-------|--------|------|
| DiffEditor | Drag-First | 22.34 | 0.9174 | 0.0918 | 128.8s |
| DragAPart | Drag-First | 24.91 | 0.9454 | 0.0567 | 119.4s |
| Puppet-Master | Drag-First | 24.42 | 0.9475 | 0.0528 | 361.5s |
| **PartRM (Ours)** | - | **28.15** | **0.9531** | **0.0356** | **4.2s** |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Only Stage 1 | 22.05 | 0.8624 | 0.1274 |
| Only Stage 2 | 25.87 | 0.9387 | 0.0537 |
| Stage 1+2 | **28.15** | **0.9531** | **0.0356** |
| 1 drag | 27.06 | 0.9466 | 0.0452 |
| 5 drags | 27.56 | 0.9483 | 0.0448 |
| 10 drags | **28.15** | **0.9531** | **0.0356** |

### 关键发现

- 两阶段训练相比单阶段提升巨大：Stage 1+2 比 Only Stage 2 在 PSNR 上高 2.28dB，说明运动学习阶段对模型学到正确运动至关重要
- 拖拽数量从 1 增加到 10，PSNR 从 27.06 提升到 28.15，更多拖拽提供了更明确的运动指引
- 多尺度拖拽嵌入（128+32+8）优于任何单一尺度，因为不同尺度捕获不同粒度的运动信息
- PartRM 推理仅需 4.2s，比 Puppet-Master 快约 86 倍

## 亮点与洞察

- **用3D高斯替代2D视频作为世界模型的状态表示**，天然支持多视角渲染和下游机器人应用，是一个很有前瞻性的设计思路
- **拖拽传播利用 SAM 分割**将单个交互扩展为密集运动条件，巧妙解决了拖拽条件歧义问题，这个思路可迁移到其他条件生成任务
- **两阶段训练中的知识蒸馏策略**——用预训练模型自身的输出作为目标进行持续学习，既保留了泛化能力又加速了训练

## 局限与展望

- 对偏离训练分布较远的铰接物体（如互联网数据中的非典型物体）泛化能力有限
- 数据集仅包含 8 个类别的铰接运动，缺乏软体变形等更复杂的运动类型
- 当前每次只能处理单个部件的运动，多部件联动场景尚未涉及
- 可以探索将 PartRM 扩展到更通用的物体动力学建模，结合语言指令实现更灵活的交互

## 相关工作与启发

- **vs Puppet-Master**: Puppet-Master 通过微调视频扩散模型实现拖拽控制，输出单视图视频；PartRM 直接输出3D高斯表示，速度快 86x，且天然支持多视角
- **vs DragAPart**: DragAPart 只做2D图像级拖拽变形，对复杂运动模式难以捕捉，且需要额外的3D重建步骤
- **vs L4GM**: L4GM 从单视图视频生成动态3D表示，但非动作条件化，不支持部件级动力学

## 评分

- 新颖性: ⭐⭐⭐⭐ 将大规模重建模型扩展到4D部件动力学是新颖的框架设计
- 实验充分度: ⭐⭐⭐⭐ 包含主实验、多组消融、泛化测试和机器人应用
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机-方法-实验逻辑连贯
- 价值: ⭐⭐⭐⭐ 在铰接物体操作和3D世界模型方向有较高的应用价值

<!-- RELATED:START -->

## 相关论文

- [LIM: Large Interpolator Model for Dynamic Reconstruction](lim_large_interpolator_model_for_dynamic_reconstruction.md)
- [Continuous 3D Perception Model with Persistent State](continuous_3d_perception_model_with_persistent_state.md)
- [SUM Parts: Benchmarking Part-Level Semantic Segmentation of Urban Meshes](sum_parts_benchmarking_part-level_semantic_segmentation_of_urban_meshes.md)
- [Matrix3D: Large Photogrammetry Model All-in-One](matrix3d_large_photogrammetry_model_all-in-one.md)
- [Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](mesh_mamba_a_unified_state_space_model_for_saliency_prediction_in_non-textured_a.md)

<!-- RELATED:END -->
