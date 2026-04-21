---
title: >-
  [论文解读] MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation
description: >-
  [ECCV 2024][自动驾驶][HD地图] 首次将知识蒸馏引入 HD 地图构建任务，提出 MapDistill 框架，通过双 BEV 变换模块、跨模态关系蒸馏、双层特征蒸馏和 Map Head 蒸馏，将相机-LiDAR 融合教师模型的知识迁移至轻量纯相机学生模型，在 nuScenes 上实现 **+7.7 mAP** 或 **4.5倍加速**。
tags:
  - ECCV 2024
  - 自动驾驶
  - HD地图
  - 知识蒸馏
  - BEV感知
  - 跨模态蒸馏
  - 轻量化部署
---

# MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation

**会议**: ECCV 2024  
**arXiv**: [2407.11682](https://arxiv.org/abs/2407.11682)  
**代码**: 无（未公开）  
**领域**: 自动驾驶 / HD地图构建 / 知识蒸馏  
**关键词**: HD地图, 知识蒸馏, BEV感知, 跨模态蒸馏, 轻量化部署

## 一句话总结

首次将知识蒸馏引入 HD 地图构建任务，提出 MapDistill 框架，通过双 BEV 变换模块、跨模态关系蒸馏、双层特征蒸馏和 Map Head 蒸馏，将相机-LiDAR 融合教师模型的知识迁移至轻量纯相机学生模型，在 nuScenes 上实现 **+7.7 mAP** 或 **4.5倍加速**。

## 研究背景与动机

在线高清地图（HD Map）构建是自动驾驶系统中的关键任务，为规划和导航提供精确的静态环境信息。近年来，基于多视角相机的方法因低成本部署优势受到关注，但面临**核心矛盾**：

**缺乏深度信息**：相机图像天然缺少显式3D几何信息，导致当前方法必须依赖大模型（如 ResNet50 + Swin Transformer）来弥补这一缺陷

**性能-效率权衡**：实验证明骨干网络的表示能力与模型性能直接相关——越大的模型结果越好，但推理越慢，牺牲了相机方法的成本优势

**KD 方法的空白**：BEV 空间的知识蒸馏已在 3D 目标检测中取得成功（BEVDistill、UniDistill），但**HD 地图构建的 KD 方法尚属空白**

**为什么不能直接沿用 3D 检测的 KD 方法？** 两个关键差异：
- 3D 检测的 head 输出目标的分类+定位，而 Map head 输出地图元素的分类+点位回归，形式不同
- 3D 检测 KD 通常对齐前景目标特征、抑制背景，这在 HD 地图构建中显然不适用（地图元素本身就分布在整个 BEV 空间）

实验验证：直接将 BEVDistill 应用于 HD 地图仅获得 +1.2 mAP 提升，UniDistill 获得 +2.3 mAP，远低于 MapDistill 的 +7.7 mAP。

## 方法详解

### 整体框架

MapDistill 采用教师-学生架构：
- **教师模型**：基于 MapTR 的相机-LiDAR 融合模型（ResNet50 + SECOND），训练完毕后冻结参数
- **学生模型**：基于 MapTR 相机分支的轻量模型（ResNet18），配备**双 BEV 变换模块**
- **蒸馏目标**：三路蒸馏损失（跨模态关系 + 双层特征 + Map Head），仅在训练阶段使用

推理时只需学生模型（纯相机），享受轻量高效的部署优势。

### 关键设计

#### 1. 双 BEV 变换模块 (Dual BEV Transform)

**功能**：将多视角相机特征转换到两个不同的 BEV 子空间，模拟教师模型的双模态（相机+LiDAR）BEV 特征。

**核心思路**：
- **子空间1**：使用 GKT（基于注意力的2D-to-BEV变换）生成 $\mathbf{F}_{C_{sub1}}^S \in \mathbb{R}^{H \times W \times C}$
- **子空间2**：使用 LSS（基于深度估计的投影变换）生成 $\mathbf{F}_{C_{sub2}}^S \in \mathbb{R}^{H \times W \times C}$
- 两个子空间特征拼接后经全卷积网络融合为 $\mathbf{F}_{fused}^S$

**设计动机**：教师模型有相机 BEV 和 LiDAR BEV 两个独立分支，学生模型需要对应的"双路"结构才能有效模仿教师的跨模态交互。使用两种不同的 2D-to-BEV 变换（GKT 关注语义、LSS 关注几何），让两个子空间捕获互补信息，类比教师的相机-LiDAR 双流。

#### 2. 跨模态关系蒸馏 ($\mathcal{L}_{relation}$)

**功能**：让学生模型学习教师模型中相机与 LiDAR 之间的跨模态注意力关系。

**核心思路**：
- 教师端：计算相机-到-LiDAR 注意力 $A_{c2l}^T$ 和 LiDAR-到-相机注意力 $A_{l2c}^T$

$$A_{c2l}^T = \text{softmax}\left(\frac{\mathbf{Fp}_{C_{bev}}^T \cdot \text{Transpose}(\mathbf{Fp}_{L_{bev}}^T)}{\sqrt{D_k}}\right)$$

- 学生端：用双 BEV 子空间的 patch 特征计算对应注意力 $A_{c2l}^S$，$A_{l2c}^S$
- 用 KL 散度对齐两方注意力：

$$\mathcal{L}_{relation} = D_{KL}(A_{c2l}^T || A_{c2l}^S) + D_{KL}(A_{l2c}^T || A_{l2c}^S)$$

**设计动机**：跨模态交互是教师模型优势的核心来源，学生模型通过模仿这种交互模式可以间接获得类LiDAR的3D感知能力。消融实验证实，跨模态关系比单模态关系更有效（mAP: 53.6 vs 52.0）。

#### 3. 双层特征蒸馏 ($\mathcal{L}_{feature}$)

**功能**：在统一 BEV 空间中同时对齐低层和高层特征表示。

**核心思路**：
- 低层蒸馏：对齐融合 BEV 特征 $\mathcal{L}_{low} = \text{MSE}(\mathbf{F}_{fused}^T, \mathbf{F}_{fused}^S)$
- 高层蒸馏：对齐 Map Encoder 输出 $\mathcal{L}_{high} = \text{MSE}(\mathbf{F}_{high}^T, \mathbf{F}_{high}^S)$

$$\mathcal{L}_{feature} = \mathcal{L}_{low} + \mathcal{L}_{high}$$

**设计动机**：低层特征包含丰富的空间和几何信息，高层特征包含语义信息。双层对齐比单层更全面，消融显示双层 (53.6 mAP) > 仅高层 (52.9) > 仅低层 (52.7)。

#### 4. Map Head 蒸馏 ($\mathcal{L}_{head}$)

**功能**：让学生模型的最终预测（地图元素分类 + 点位位置）逼近教师模型的输出。

**核心思路**：以教师预测作为伪标签监督学生：

$$\mathcal{L}_{head} = \mathcal{L}_{Focal}(\mathbf{F}_{cls}^T, \mathbf{F}_{cls}^S) + \mathcal{L}_{p2p}(\mathbf{F}_{point}^T, \mathbf{F}_{point}^S)$$

- 分类用 Focal Loss
- 点位回归用曼哈顿距离(Manhattan Distance)

**设计动机**：特征级蒸馏是隐式的，Head 级蒸馏提供直接的输出对齐信号，两者互补。消融显示同时使用分类和点位蒸馏 (53.6 mAP) 优于仅用其一 (51.8/51.9)。

### 损失函数 / 训练策略

总训练损失：

$$\mathcal{L} = \mathcal{L}_{map} + \lambda_1 \mathcal{L}_{relation} + \lambda_2 \mathcal{L}_{feature} + \lambda_3 \mathcal{L}_{head}$$

- $\mathcal{L}_{map}$：MapTR 原始地图损失（分类 + 点到点 + 边方向）
- 训练配置：8 块 A6000 GPU，batch=64，AdamW 优化器，初始学习率 $4 \times 10^{-3}$
- 教师模型：ResNet50 + SECOND，24 epoch 预训练
- 学生模型：ResNet18，110 epoch 蒸馏训练

## 实验关键数据

### 主实验

**nuScenes val 集（mAP）**：

| 方法 | 学生模态 | 教师模态 | Backbone | AP_ped | AP_div | AP_bou | mAP |
|------|---------|---------|----------|--------|--------|--------|-----|
| MapTR | C | — | R50 | 45.3 | 51.5 | 53.1 | 50.3 |
| MapTR | C&L | — | R50&Sec | 55.9 | 62.3 | 69.3 | 62.5 |
| MapTR (baseline) | C | — | R18 | 39.6 | 49.9 | 48.2 | **45.9** |
| BEVDistill | C | L | R18 | 42.4 | 48.5 | 50.2 | 47.1 (+1.2) |
| UniDistill | C | C&L | R18 | 43.9 | 48.6 | 52.1 | 48.2 (+2.3) |
| **MapDistill (C teacher)** | C | C | R18 | 43.3 | 48.8 | 51.9 | 48.0 (+2.1) |
| **MapDistill (L teacher)** | C | L | R18 | 45.9 | 50.7 | 53.6 | 50.1 (+4.2) |
| **MapDistill (C&L teacher)** | C | C&L | R18 | **49.2** | **54.5** | **57.1** | **53.6 (+7.7)** |

### 消融实验

**各蒸馏损失组件贡献**：

| 设置 | $\mathcal{L}_{relation}$ | $\mathcal{L}_{feature}$ | $\mathcal{L}_{head}$ | mAP |
|------|:-:|:-:|:-:|------|
| Baseline (无蒸馏) | ✗ | ✗ | ✗ | 45.9 |
| (a) 仅关系蒸馏 | ✔ | ✗ | ✗ | 48.8 (+2.9) |
| (b) 仅特征蒸馏 | ✗ | ✔ | ✗ | 48.4 (+2.5) |
| (c) 仅Head蒸馏 | ✗ | ✗ | ✔ | 49.0 (+3.1) |
| (d) 关系+特征 | ✔ | ✔ | ✗ | 50.3 |
| (e) 特征+Head | ✗ | ✔ | ✔ | 50.8 |
| (f) 关系+Head | ✔ | ✗ | ✔ | 51.1 |
| **(g) 全部** | **✔** | **✔** | **✔** | **53.6** |

**跨模态关系蒸馏消融**：

| 关系类型 | mAP |
|---------|-----|
| 无关系蒸馏 | 50.8 |
| 单模态关系 | 52.0 |
| 混合关系 | 52.4 |
| **跨模态关系（ours）** | **53.6** |

### 关键发现

1. **融合教师 >> 单模态教师**：相机-LiDAR 融合教师带来 +7.7 mAP，远超纯 LiDAR 教师 (+4.2) 和纯相机教师 (+2.1)
2. **三路蒸馏损失互补**：每个损失单独贡献约 +2.5~3.1 mAP，三者联合达到 +7.7，说明存在协同效应
3. **跨模态关系是关键**：跨模态注意力蒸馏优于单模态关系蒸馏 (53.6 vs 52.0)，甚至优于混合关系 (52.4)
4. **双 BEV 变换有效**：使用不同变换方式（GKT+LSS）的双路结构最优，单一变换或相同变换均不如

## 亮点与洞察

- **填补领域空白**：首次系统性地将知识蒸馏应用于 HD 地图构建，指出任务差异导致 3D 检测 KD 方法不可直接迁移
- **双 BEV 变换的巧妙设计**：用两种不同的 2D-to-BEV 变换（注意力 vs 深度估计）模拟教师的相机-LiDAR 双流，纯相机实现"伪多模态"
- **部署友好**：所有蒸馏损失仅在训练阶段存在，推理时完全不增加计算开销
- **全面消融**：对每个损失组件、每种蒸馏关系、每个 BEV 变换组合都做了详尽消融

## 局限与展望

- 训练需要同时运行教师和学生模型，**训练成本较高**
- 学生骨干固定为 ResNet18，未探索更轻量的架构（如 MobileNet）或 NAS 搜索
- 仅在 nuScenes 上验证，未涉及其他数据集（如 Argoverse2）
- 教师模型的质量上限决定了学生的性能天花板，可探索更强的融合教师
- 双 BEV 变换模块引入了额外参数，可探索参数共享方案

## 相关工作与启发

- **MapTR**：提供了教师和学生模型的基础架构，其统一的地图元素表示便于蒸馏
- **BEVDistill / UniDistill**：BEV 空间跨模态蒸馏的先驱，但未针对 HD 地图任务优化
- **GKT / LSS**：两种互补的 2D-to-BEV 变换方法，恰好适合构建双路结构模拟双模态特征

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将 KD 引入 HD 地图构建，双 BEV 变换和跨模态关系蒸馏设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 极其详尽的消融：三路损失、关系类型、特征层级、BEV变换组合、超参分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，问题分析到位，与 3D 检测 KD 的差异阐述准确
- **价值**: ⭐⭐⭐⭐ — +7.7 mAP 或 4.5 倍加速的实际意义大，可作为 KD-based HD map 构建的 strong baseline

<!-- RELATED:START -->

## 相关论文

- [Stream Query Denoising for Vectorized HD-Map Construction](stream_query_denoising_for_vectorized_hd-map_construction.md)
- [Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection](detecting_as_labeling_rethinking_lidar-camera_fusion_in_3d_object_detection.md)
- [SafeMap: Robust HD Map Construction from Incomplete Observations](../../ICML2025/autonomous_driving/safemap_robust_hd_map_construction_from_incomplete_observations.md)
- [MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping](maptracker_tracking_with_strided_memory_fusion_for_consistent_vector_hd_mapping.md)
- [TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion](../../CVPR2025/autonomous_driving/tacodepth_towards_efficient_radar-camera_depth_estimation_with_one-stage_fusion.md)

<!-- RELATED:END -->
