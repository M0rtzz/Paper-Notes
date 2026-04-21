---
title: >-
  [论文解读] Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection
description: >-
  [ECCV 2024][自动驾驶][LiDAR-相机融合] 本文从数据标注过程中总结出"回归任务不应使用图像特征"的基本原则，提出 DAL 范式——将检测过程类比为标注过程，用点云特征独立完成回归预测、用融合特征完成分类预测，结合简洁的训练流程，在 nuScenes 上以 74.0 NDS（val）和 74.8 NDS（test）大幅刷新 SOTA。
tags:
  - ECCV 2024
  - 自动驾驶
  - LiDAR-相机融合
  - 3D目标检测
  - 数据标注范式
  - 过拟合抑制
  - 速度精度权衡
---

# Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection

**会议**: ECCV 2024  
**arXiv**: [2311.07152](https://arxiv.org/abs/2311.07152)  
**代码**: https://github.com/HuangJunJie2017/BEVDet  
**领域**: 自动驾驶 / 3D目标检测  
**关键词**: LiDAR-相机融合, 3D目标检测, 数据标注范式, 过拟合抑制, 速度精度权衡

## 一句话总结

本文从数据标注过程中总结出"回归任务不应使用图像特征"的基本原则，提出 DAL 范式——将检测过程类比为标注过程，用点云特征独立完成回归预测、用融合特征完成分类预测，结合简洁的训练流程，在 nuScenes 上以 74.0 NDS（val）和 74.8 NDS（test）大幅刷新 SOTA。

## 研究背景与动机

**领域现状**：LiDAR-相机融合的3D目标检测是自动驾驶感知的核心任务。近年来涌现了大量融合方法（TransFusion、BEVFusion、CMT 等），在 nuScenes 排行榜上竞争激烈。然而，这些方法普遍存在过拟合问题，需要依赖复杂的多阶段预训练和特殊学习率策略来缓解。

**现有痛点**：(1) 所有现有方法都将来自图像的特征参与到回归任务（如预测3D框的中心、尺寸、朝向）中，违反了数据标注的基本规则。(2) 由于单目深度估计的固有不适定性，图像特征在回归几何属性时不够鲁棒，导致模型过拟合。(3) 为对抗过拟合，现有方法采用复杂的训练流水线——多阶段预训练（在 ImageNet、nuScenes、nuImages 等多个数据集上）、定制学习率策略，增加了额外代价和不确定性。(4) 图像分支参与回归也限制了图像空间数据增强的范围，因为需要保持图像特征与目标预测的一致性。

**核心矛盾**：图像和 LiDAR 在3D检测中扮演不同角色。LiDAR 点云是精确的"尺子"，能准确定位3D框的边界；图像是"经验丰富的赌徒"，擅长识别和分类但在几何回归上不可靠。现有方法未区分这两种模态在不同子任务上的角色差异。

**切入角度**：作者从数据标注流程中获得启发——标注人员遵循两条规则：(A) 图像与点云结合搜索候选目标并确定类别；(B) 3D框仅根据点云标注。现有算法违反了规则B。DAL 通过模仿标注过程来构建检测流水线。

**核心 idea**：将检测过程类比为数据标注过程，回归任务仅使用点云特征，分类任务使用融合特征，从根本上消除过拟合源。

## 方法详解

### 整体框架

DAL 采用 dense-to-sparse 范式。密集感知阶段：分别用图像编码器和点云编码器提取特征 $F_I$ 和 $F_P$，将图像特征通过 LSS 变换到 BEV 空间，拼接后生成密集热图，选取 Top-$K$ 候选。稀疏感知阶段：对每个候选，用其点云特征通过 FFN 预测回归目标（中心、尺寸、朝向、速度），同时融合图像特征、图像 BEV 特征和点云 BEV 特征进行分类预测。关键在于回归分支完全不使用图像特征。

### 关键设计

1. **回归-分类分模态预测（Modality-specific Task Assignment）**:

    - 功能：从根本上消除由图像特征参与回归导致的过拟合
    - 核心思路：在稀疏感知阶段，回归目标（中心、尺寸、朝向、速度）仅由点云特征经过简单 FFN 预测。分类任务则融合图像特征、图像 BEV 特征和点云 BEV 特征来完成。密集感知阶段融合两种模态的 BEV 特征生成热图进行候选搜索。与 BEVFusion 的关键区别在于：(1) 推迟融合——BEV 编码器之后再融合而非之前；(2) 移除稀疏实例与 BEV 特征之间的注意力；(3) 回归仅使用点云特征
    - 设计动机：模仿数据标注规则B——3D框的几何属性应仅根据点云确定。图像特征的单目深度估计不适定性使其在回归中引入系统性噪声

2. **简洁训练流水线**:

    - 功能：消除对复杂预训练和特殊学习率策略的依赖
    - 核心思路：仅加载 ImageNet 预训练的图像骨干权重，端到端训练 20 个 epoch，使用 CBGS 数据采样和循环学习率策略（初始值 $2.0 \times 10^{-4}$），不需要在 nuScenes、nuImages 等数据集上预训练 LiDAR 骨干。总损失为 $L_{\text{DAL}} = L_{\text{aux}} + L_{\text{TransFusion}}$，其中 $L_{\text{aux}}$ 是基于图像特征的辅助分类头损失
    - 设计动机：回归任务不涉及图像特征后，图像分支的梯度不再受不精确的深度估计影响，使得简单的端到端训练成为可能。这也使得大范围图像 resize 增强变得可行（因为不再需要保持图像尺寸与回归目标的一致性）

3. **速度增强策略（Velocity Augmentation）**:

    - 功能：解决训练数据中速度分布极度不平衡的问题
    - 核心思路：nuScenes 中大多数车辆实例是静止的，导致速度分布严重偏斜。对部分静止目标随机赋予预定义速度，并据此调整其多帧点云的位置，制造"移动"效果。仅对静止目标执行此增强，因为可以通过标注框准确获取其完整点云
    - 设计动机：不平衡的速度分布使得模型在速度预测上表现不佳。速度预测对自动驾驶的规划模块至关重要。消融实验显示速度增强将 AVE 指标降低约 25%

### 损失函数 / 训练策略

DAL 共享 TransFusion 和 BEVFusion 的目标设计和损失函数设计，额外增加一个辅助分类头——基于标注目标重力中心提取图像稀疏特征并进行分类，其损失直接加到总损失上，不做重加权。辅助分类头弥补了密集和稀疏感知阶段对图像分支的监督缺陷。

## 实验关键数据

### 主实验

| 数据集 | 指标 | DAL-Large | 之前SOTA (UniTR) | 之前SOTA (CMT) | 提升 |
|--------|------|-----------|----------|------|------|
| nuScenes val | NDS | 74.0 | 73.3 | 72.9 | +0.7 |
| nuScenes val | mAP | 71.5 | 70.9 | 72.0 | +0.6 (vs CMT) |
| nuScenes test | NDS | 74.8 | 74.5 | 74.1 | +0.3 |
| nuScenes test | mAP | 72.0 | 70.5 | 72.0 | +1.5 (vs UniTR) |

DAL-Tiny 在 16.55 FPS 下达到 71.3 NDS，比相似速度的 CMT-R50 (14.2 FPS, 70.8 NDS) 快且准。

### 消融实验

| 配置 | Pipeline | 辅助分类 | 图像resize范围 | 速度增强 | mAP | NDS |
|------|----------|---------|------------|---------|-----|-----|
| A (LiDAR only) | BEVFusion | - | - | - | 63.67 | 69.00 |
| B | BEVFusion | ✗ | 0.36-0.55 | ✗ | 63.59 | 68.71 |
| F | DAL | ✓ | 0.36-0.55 | ✗ | 64.16 | 69.52 |
| G | DAL | ✓ | 0.36-0.88 | ✗ | 68.07 | 70.87 |
| H | DAL | ✓ | 0.36-0.88 | ✓ | 68.50 | 71.94 |

### 关键发现

- BEVFusion 使用 DAL 的简单训练流程时（config B），性能不如 LiDAR-only 基线（config A），说明其依赖复杂预训练来利用图像模态
- DAL 的流水线使大范围 resize 增强可行（config F→G），带来 +3.91 mAP 提升
- 速度增强将 mAVE 从 25.80 降到 19.31，降低约 25%
- DAL 推荐用小图像分支 + 大 LiDAR 分支的配置，因为分类任务对图像分支要求较低

## 亮点与洞察

- **从标注规则推导算法设计**：将数据标注的行业规范提升为算法设计原则，视角独特且令人信服
- **简洁而强大**：仅用最经典的元素（ResNet + VoxelNet + FPN + SECOND），不使用注意力机制，却达到 SOTA
- **训练流程极简**：仅需 ImageNet 预训练的图像骨干，一阶段端到端训练，无需定制学习率策略
- **速度-精度的帕累托最优**：在不同配置下均提供优于现有方法的速度-精度权衡

## 局限与展望

- 未考虑 LiDAR 范围之外的物体（这些物体在 nuScenes 中不被标注）
- nuScenes 仅有 10 个类别，简单分类任务无法充分利用先进图像骨干（如 SwinTransformer）的能力
- DAL 当前使用无注意力的流水线，可以进一步引入 DSVT、DETR 等注意力机制增强
- 未在 Waymo 等其他数据集上验证泛化性

## 相关工作与启发

- **BEVFusion (MIT/ADLab)**：BEV 空间融合的代表方法，DAL 的基线
- **TransFusion**：基于 Transformer 的融合方法，DAL 共享其目标和损失设计
- **CMT / UniTR**：基于注意力的 SOTA 方法
- 启发：算法设计应尊重数据生成过程的基本规则；"越简单越好"在正确的设计原则指导下是成立的

## 评分

- 新颖性: ⭐⭐⭐⭐（从标注规则推导设计的视角非常新颖）
- 实验充分度: ⭐⭐⭐⭐⭐（详尽的消融、速度-精度分析、多个配置对比）
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐（为LiDAR-相机融合提供了清晰的设计原则和强基线）

<!-- RELATED:START -->

## 相关论文

- [RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](../../CVPR2025/autonomous_driving/racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation](mapdistill_boosting_efficient_camera-based_hd_map_construction_via_camera-lidar_.md)
- [CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](../../ICCV2025/autonomous_driving/cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)

<!-- RELATED:END -->
