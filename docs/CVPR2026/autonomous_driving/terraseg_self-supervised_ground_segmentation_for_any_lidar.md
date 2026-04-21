---
title: >-
  [论文解读] TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR
description: >-
  [CVPR 2026][自动驾驶][地面分割] 本文提出 TerraSeg，首个自监督的域无关 LiDAR 地面分割模型，通过构建统一的 OmniLiDAR 大规模数据集（12个公开基准、15种传感器、近2200万次扫描）和创新的 PseudoLabeler 自监督伪标签生成模块，在不使用任何人工标注的情况下在 nuScenes、SemanticKITTI 和 Waymo 上达到 SOTA。
tags:
  - CVPR 2026
  - 自动驾驶
  - 地面分割
  - 自监督学习
  - 跨传感器泛化
  - LiDAR感知
  - 伪标签
---

# TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR

**会议**: CVPR 2026  
**arXiv**: [2603.27344](https://arxiv.org/abs/2603.27344)  
**代码**: 已公开（Apache 2.0）  
**领域**: 自动驾驶 / 3D点云分割  
**关键词**: 地面分割、自监督学习、跨传感器泛化、LiDAR感知、伪标签

## 一句话总结

本文提出 TerraSeg，首个自监督的域无关 LiDAR 地面分割模型，通过构建统一的 OmniLiDAR 大规模数据集（12个公开基准、15种传感器、近2200万次扫描）和创新的 PseudoLabeler 自监督伪标签生成模块，在不使用任何人工标注的情况下在 nuScenes、SemanticKITTI 和 Waymo 上达到 SOTA。

## 研究背景与动机

**领域现状**：LiDAR 地面分割是自动驾驶感知栈的基础任务，用于物体发现、自由空间估计和定位建图。现有方法分为两类——手工几何方法（如 RANSAC、PatchWork++）和监督学习方法（如 GndNet）。

**现有痛点**：手工方法虽然快速且不需要标注，但依赖简单地形假设（如全局平面）和传感器特定调参，换到新环境/传感器就需要重新调参，泛化性差。监督学习方法泛化性更好，但依赖昂贵的逐点人工标注，可扩展性极差。

**核心矛盾**：快速+免标注的手工方法缺乏泛化性，而有泛化性的学习方法需要昂贵标注——理想方案应兼具：免标注、跨传感器零样本泛化、实时运行。

**本文目标** (1) 如何在完全不使用人工标注的情况下训练出高质量地面分割模型；(2) 如何让单一模型泛化到不同传感器、不同场景和不同天气条件。

**切入角度**：受 NLP/CV 大规模预训练成功的启发，但不追求多任务通用系统，而是走单任务域无关路线——在极多样的几何数据上自监督训练，实现零样本跨域迁移。

**核心 idea**：汇聚12个数据集、15种传感器的近2200万次扫描构建 OmniLiDAR，用自监督伪标签（PseudoLabeler）训练一个基于 Point Transformer v3 的域无关地面分割模型。

## 方法详解

### 整体框架

TerraSeg 框架包含三个核心组件：(1) OmniLiDAR 数据集——统一标准化12个公开驾驶数据集的原始LiDAR扫描；(2) PseudoLabeler——通过自监督运行时优化生成每帧的地面/非地面伪标签；(3) TerraSeg 模型——基于 Point Transformer v3 的实时域无关地面分割网络，用伪标签训练。输入为原始3D点云坐标，输出为逐点的地面/非地面置信度分数。

### 关键设计

1. **OmniLiDAR 统一数据集**:

    - 功能：提供前所未有的传感器多样性，支持跨域泛化训练
    - 核心思路：汇聚 nuScenes、SemanticKITTI、Waymo、Argoverse 2 等12个公开驾驶数据集的近2200万次原始扫描，覆盖15种不同LiDAR硬件。对所有数据进行三步标准化：(1) 下采样到 0.2Hz 确保多样性；(2) 通过传感器特定变换对齐坐标系使 $z=0$ 近似地面对齐、$x$ 轴朝前；(3) 按传感器型号去除自车点。仅保留标准化的 $(x,y,z)$ 坐标和元数据标签
    - 设计动机：单一数据集的传感器和场景多样性不足以训练泛化模型，需要大规模聚合来覆盖不同地形、天气和传感器配置

2. **PseudoLabeler 自监督伪标签生成**:

    - 功能：无需人工标注，为每帧原始LiDAR扫描生成高质量地面/非地面标签
    - 核心思路：将地面分割转化为鸟瞰图高程图估计的代理任务。用 MLP 参数化高程图 $g_\theta: \mathbb{R}^2 \to \mathbb{R}$，对每个点计算垂直残差 $\Delta d_i = z_i - g_\theta(x_i, y_i)$。利用地面是场景中最低连续表面的几何先验，设计不对称损失：低于预测面的点用二次惩罚，高于的用 Huber 损失。包含三阶段流水线：预处理（去除负噪声）→ 运行时优化（SiLU激活+AdamW+EMA早停）→ 后处理（柱状细化，恢复被误分为地面的物体底部点）
    - 设计动机：利用"地面是最低表面"这个简单但强大的几何先验实现完全自监督，同时通过精心设计的前后处理解决实际问题（如多路径反射噪声、车辆轮胎误分等）

3. **TerraSeg 域无关模型设计**:

    - 功能：实时推理、零样本跨传感器地面分割
    - 核心思路：基于 Point Transformer v3 骨干网络，做了三个关键域无关适配：(1) 禁用数据集特定的归一化，强迫模型学习通用几何先验；(2) 用 Group Normalization 替代 Batch Normalization 以应对多传感器混合批次的分布不稳定；(3) 仅使用3维输入特征——常数1特征、归一化高度、归一化水平距离，原始坐标仅用于构建体素网格。提供 Base（精确）和 Small（高效）两个版本
    - 设计动机：通过限制输入特征和禁用域特定模块，迫使模型只学习通用的几何判据而非传感器特定的伪影模式

### 损失函数 / 训练策略

训练损失为 Binary Cross-Entropy + 对称 Lovász-Softmax 损失 $\mathcal{L} = \mathcal{L}_{BCE} + \lambda \mathcal{L}_{Lovász}$（$\lambda=1.0$）。BCE 项使用动态正类权重（通过 EMA 跟踪的地面/非地面点比例），自适应处理跨场景的类别不平衡。使用 AdamW 优化器（lr=2e-3, weight decay=5e-3），线性 warm-up + cosine decay。体素化分辨率 0.05，有效批大小 256，自定义 epoch 长度 20,000 帧。

## 实验关键数据

### 主实验

nuScenes 验证集结果（无人工标注训练）：

| 方法 | 标注 | Ground IoU | Non-Ground IoU | mIoU | 吞吐量(Hz) |
|------|------|-----------|---------------|------|-----------|
| RANSAC | 无 | 89.14 | 83.97 | 86.55 | 255.0 |
| PatchWork++ | 无 | 86.19 | 81.42 | 83.80 | 30.0 |
| TRAVEL | 无 | 89.76 | 87.16 | 88.46 | 365.7 |
| GndNet | **有** | 82.54 | 78.72 | 80.62 | 484.3 |
| **TerraSeg-B** | **无** | **93.50** | **91.45** | **92.47** | 28.0 |
| **TerraSeg-S** | **无** | 92.40 | 90.49 | 91.45 | 49.8 |
| 监督上界 | 有 | 95.96 | 94.65 | 95.31 | 28.0 |

### 消融实验

PseudoLabeler 各组件消融（nuScenes）：

| 配置 | 效果 |
|------|------|
| 无预处理 | 负噪声导致高程图下沉，地面过分割 |
| 无后处理 | 车辆底部/轮胎被误分为地面 |
| 完整 PseudoLabeler | mIoU 90.63（作为训练标签时 TerraSeg 达 92.47） |

### 关键发现

- **自监督超越监督基线**：TerraSeg-B（无标注）的 mIoU 92.47 远超使用标注训练的 GndNet 的 80.62
- **接近监督上界**：与全监督版本（95.31）差距仅约3个百分点
- **跨数据集一致性**：在 nuScenes、SemanticKITTI、Waymo 三个完全不同的数据集上均达到 SOTA
- **实时推理**：TerraSeg-S 达到 ~50Hz，TerraSeg-B 达到 ~28Hz，满足在线部署需求
- **学生优于教师**：TerraSeg 模型（mIoU 92.47）反超其训练标签来源 PseudoLabeler（90.63），体现了模型的泛化和去噪能力

## 亮点与洞察

- **大规模聚合策略**：12个数据集、15种传感器的统一聚合是非常有工程价值的工作，OmniLiDAR 本身就是重要贡献
- **自监督方案设计精巧**：将地面分割巧妙转化为高程图估计的代理任务，利用简单几何先验实现免标注训练
- **"学生超越教师"现象**：说明大规模多样性数据+神经网络的泛化能力可以有效过滤伪标签中的噪声
- **实用性极强**：不需要任何标注数据、支持任何LiDAR传感器、实时运行、代码开源，是非常落地的工作

## 局限与展望

- 仅处理单帧点云，未利用时序信息，复杂地形下可能受限
- 伪标签质量在极端场景（如陡坡、高植被）可能退化
- 仅输出二分类（地面/非地面），未提供语义信息（如可通行性等级）
- 可考虑扩展到语义地面分割（区分道路、草地、泥土等不同地面类型）

## 相关工作与启发

- PatchWork++ 的同心区域极坐标网格思想可与学习方法结合使用
- Chodosh et al. 的自监督运行时优化是 PseudoLabeler 的直接前身，本文在此基础上做了关键改进
- Point Transformer v3 作为骨干网络的成功应用验证了其在点云任务上的通用性

## 评分

- **新颖性**: ⭐⭐⭐⭐ — OmniLiDAR 数据集和自监督域无关训练范式都是首创
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个主流 benchmark 全面验证，消融实验详尽，基线对比完整
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，三个核心组件的关系描述明确
- **价值**: ⭐⭐⭐⭐⭐ — 实用价值极高，直接解决了自动驾驶中地面分割的标注困难和传感器泛化问题

<!-- RELATED:START -->

## 相关论文

- [BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images](bev-sld_self-supervised_scene_landmark_detection_for_global_localization_with_li.md)
- [Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](../../CVPR2025/autonomous_driving/exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)
- [Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens](le_mumo_jepa_multi-modal_self-supervised_representation_learning_with_learnable_.md)
- [HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles](horizonforge_driving_scene_editing_with_any_trajectories_and_any_vehicles.md)
- [PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds](../../CVPR2025/autonomous_driving/psa-ssl_pose_and_size-aware_self-supervised_learning_on_lidar_point_clouds.md)

<!-- RELATED:END -->
