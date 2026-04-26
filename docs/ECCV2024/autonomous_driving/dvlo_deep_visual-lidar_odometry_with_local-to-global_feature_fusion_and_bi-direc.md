---
title: >-
  [论文解读] DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion and Bi-directional Structure Alignment
description: >-
  [ECCV2024][自动驾驶][Visual-LiDAR Odometry] 提出基于聚类的 Local-to-Global 融合网络 DVLO，通过双向结构对齐（图像→伪点云 + 点云→伪图像）解决视觉与 LiDAR 的数据结构不一致问题，在 KITTI 里程计和 FlyingThings3D 场景流任务上均取得 SOTA。
tags:
  - ECCV2024
  - 自动驾驶
  - Visual-LiDAR Odometry
  - 多模态
  - Clustering-based Fusion
  - Bi-Directional Structure Alignment
  - Scene Flow
---

# DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion and Bi-directional Structure Alignment

**会议**: ECCV2024  
**arXiv**: [2403.18274](https://arxiv.org/abs/2403.18274)  
**代码**: [IRMVLab/DVLO](https://github.com/IRMVLab/DVLO)  
**领域**: autonomous_driving  
**关键词**: Visual-LiDAR Odometry, Multi-Modal Fusion, Clustering-based Fusion, Bi-Directional Structure Alignment, Scene Flow

## 一句话总结

提出基于聚类的 Local-to-Global 融合网络 DVLO，通过双向结构对齐（图像→伪点云 + 点云→伪图像）解决视觉与 LiDAR 的数据结构不一致问题，在 KITTI 里程计和 FlyingThings3D 场景流任务上均取得 SOTA。

## 背景与动机

视觉/LiDAR 里程计是自动驾驶和 SLAM 中的基础任务，需要从连续帧中估计相对位姿变换。图像提供细粒度的纹理信息，点云提供丰富的几何信息，二者互补性强。然而，视觉-LiDAR 融合的核心困难在于**数据结构的天然不一致**：

- 图像像素是**规则、密集**的二维网格
- LiDAR 点云是**无序、稀疏**的三维点集

现有方法的不足：

1. **基于 CNN 的融合**：感受野受限于卷积核大小，无法建立全局对应关系
2. **基于 Attention 的融合**：虽能全局交互，但计算复杂度为二次方，推理时间过长
3. **单层级融合**：仅做全局或局部融合，无法同时保留细粒度信息和全局上下文

## 核心问题

如何设计一种既能保留**局部细粒度像素-点对应关系**、又能实现**全局信息交互**的高效多模态融合策略，同时解决图像与点云之间的数据结构不一致问题？

## 方法详解

### 整体架构

DVLO 包含四个核心模块：层级特征提取、Local Fuser（局部融合）、Global Fuser（全局融合）、迭代位姿估计。

### 1. 层级特征提取

**点云特征提取**：将 LiDAR 点云通过柱面投影（Cylindrical Projection）转换为伪图像（$64 \times 1800$），投影公式为：

$$u = \arctan2(y/x) / \Delta\theta, \quad v = \arcsin(z / \sqrt{x^2+y^2+z^2}) / \Delta\phi$$

每个二维位置填充对应的原始三维坐标，既能转换为伪图像结构，又保留了三维几何信息。然后用层级卷积网络提取多尺度点云特征 $F_P \in \mathbb{R}^{H_P \times W_P \times D}$。

**图像特征提取**：对相机图像（填充至 $384 \times 1280$）使用卷积特征金字塔提取多尺度图像特征 $F_I \in \mathbb{R}^{H_I \times W_I \times C}$。

### 2. Local Fuser：基于聚类的局部融合

这是本文最核心的创新。受 Context Clusters 启发，提出了首个**基于聚类的多模态融合模块**，不使用 CNN 或 Transformer。

**图像→伪点云**（Image-to-Point 结构对齐）：将图像特征 $F_I$ reshape 为伪点集 $F_{pp} \in \mathbb{R}^{M \times C}$（$M = H_I \times W_I$），使图像具备与 LiDAR 点云相同的数据结构。

**伪点聚类**：将 LiDAR 点投影到图像平面获取二维坐标作为**聚类中心**，通过双线性插值获取中心特征 $F_c$，然后根据中心特征与伪点特征之间的**余弦相似度**将伪点分配到最近的中心，形成 $N$ 个簇。

**局部特征聚合**：在每个簇内，根据相似度动态聚合伪点特征：

$$F_L^i = \frac{1}{X}\left(F_c^i + \sum_{j=1}^{k} \text{sigmoid}(\alpha s_{ij} + \beta) \cdot F_{pp}^j\right)$$

其中 $\alpha, \beta$ 是可学习参数，$s_{ij}$ 是相似度分数。局部融合特征 $F_L$ 维度与原始 LiDAR 点数相同。

### 3. Global Fuser：自适应全局融合

局部融合的感受野有限，因此引入全局自适应融合机制。

**点云→伪图像**（Point-to-Image 结构对齐）：通过柱面投影将点云转为伪图像结构。

**自适应融合**：对局部融合特征 $F_L$ 和点云特征 $F_P$ 分别通过 MLP + Sigmoid 生成自适应权重 $A_L, A_P$，加权融合：

$$F_G = \frac{A_P \odot F_P + A_L \odot F_L}{A_P + A_L}$$

### 4. 迭代位姿估计

使用 Attentive Cost Volume 在最粗层关联两帧的全局融合特征，生成嵌入特征 $E$。通过可学习掩码加权后，用 FC 层回归旋转四元数 $q$ 和平移向量 $t$，再逐层迭代细化。

### 5. 损失函数

多层级监督损失，对每层的平移和旋转误差用可学习标量 $k_x, k_q$ 自适应平衡：

$$\mathcal{L}^l = \|t_{gt} - t^l\| \exp(-k_x) + k_x + \|q_{gt} - q^l\|_2 \exp(-k_q) + k_q$$

## 实验关键数据

### KITTI 里程计

- **训练集**：序列 00-06；**测试集**：序列 07-10
- 测试集平均 $t_{rel}$：**0.82%**，$r_{rel}$：**0.41°/100m**
- 对比纯视觉 SOTA（Cho et al.）：$t_{rel}$ 下降 63.4%，$r_{rel}$ 下降 43.8%
- 对比纯 LiDAR SOTA（EfficientLO）：$t_{rel}$ 下降 4.9%，旋转误差持平
- 对比多模态 SOTA（H-VLO，训练用了更多数据 00-08）：$t_{rel}$ 下降 47.0%
- 对比传统多模态方法（PL-LOAM）：全序列平均 $t_{rel}$ 下降 28.7%（0.67 vs 0.94）

### FlyingThings3D 场景流

- EPE2D：**1.69**（CamLiRAFT 1.73）
- EPE3D：**0.048**（CamLiRAFT 0.049）
- 在 2D 和 3D 指标上均超越专为场景流设计的 CamLiRAFT

### 推理效率

- 推理时间：**98.5ms**，唯一满足 10Hz LiDAR 实时要求（<100ms）的多模态方法
- 对比 Attention 方法（183.76ms）：仅需约一半推理时间
- 与 CNN 方法（87.24ms）相当，但精度更高

### 消融实验

| 配置 | 测试集平均 $t_{rel}$ | 测试集平均 $r_{rel}$ |
|------|---------------------|---------------------|
| 仅 Global Fuser | 0.93 | 0.47 |
| 仅 Local Fuser | 1.00 | 0.50 |
| **完整 DVLO** | **0.82** | **0.41** |

## 亮点

1. **首个基于聚类的多模态融合方法**：既非 CNN 也非 Transformer，提供了全新的融合范式
2. **双向结构对齐**：同时做图像→伪点云和点云→伪图像的结构转换，最大化跨模态互补
3. **Local-to-Global 层级设计**：局部保留细粒度像素-点对应，全局实现大感受野信息交互
4. **高效实时**：98.5ms 推理时间满足 10Hz 实时约束，优于所有其他多模态方法
5. **泛化性强**：融合模块可直接迁移到场景流估计任务且超越专用 SOTA

## 局限与展望

1. **仅在 KITTI 验证**：缺少 nuScenes、Waymo 等更大规模数据集的评估
2. **单目图像限制**：仅使用单目左相机，未探索立体视觉带来的深度信息增益
3. **无 mapping 后端**：当前仅是里程计前端，未集成完整 SLAM 系统（回环检测等）
4. **聚类策略固定**：采用最近邻分配，未探索软分配或可微聚类的潜力
5. **柱面投影的信息损失**：远距离点的投影分辨率降低，可能影响长距离场景

## 与相关工作的对比

| 方法 | 模态 | 融合策略 | KITTI 07-10 $t_{rel}$ | 推理时间 |
|------|------|---------|----------------------|---------|
| EfficientLO | LiDAR | 无融合 | 0.86 | — |
| H-VLO | 视觉+LiDAR | CNN 融合 | 1.36 | — |
| TransLO | LiDAR | Transformer | 0.99 | — |
| PL-LOAM | 视觉+LiDAR | 传统 | — | 200ms |
| **DVLO（本文）** | **视觉+LiDAR** | **聚类 Local-to-Global** | **0.82** | **98.5ms** |

DVLO 的核心优势在于：不仅精度全面领先，而且推理速度也是最快的，兼顾了性能和效率。

## 启发与关联

1. **聚类作为融合原语**：Context Clusters 已证明聚类可作为视觉骨干，本文将其拓展到多模态融合，这一范式值得在更多多模态任务（如 3D 检测、BEV 感知）中探索
2. **双向结构对齐思想**：处理异构数据融合时，双向互相对齐比单向投影更有效，可推广到 radar-camera、thermal-RGB 等融合场景
3. **层级融合策略**：先局部后全局的融合顺序可以应用到其他需要兼顾细粒度和全局信息的任务中

## 评分
- 新颖性: ⭐⭐⭐⭐ （首个基于聚类的多模态融合，双向结构对齐设计新颖）
- 实验充分度: ⭐⭐⭐⭐ （KITTI 里程计 + FlyingThings3D 场景流 + 详细消融，但缺少更多数据集）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，公式完整，图示直观）
- 价值: ⭐⭐⭐⭐ （实时性强、泛化性好，对多模态融合研究有参考价值）

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion](dvlo_deep_visuallidar_odometry_with_localtoglobal_featu.md)
- [\[ECCV 2024\] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)
- [\[ECCV 2024\] LiDAR-Event Stereo Fusion with Hallucinations](lidar-event_stereo_fusion_with_hallucinations.md)
- [\[ECCV 2024\] Approaching Outside: Scaling Unsupervised 3D Object Detection from 2D Scene](approaching_outside_scaling_unsupervised_3d_object_detection_from_2d_scene.md)
- [\[ECCV 2024\] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)

<!-- RELATED:END -->
