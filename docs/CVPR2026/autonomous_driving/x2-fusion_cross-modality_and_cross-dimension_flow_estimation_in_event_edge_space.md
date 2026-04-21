---
title: >-
  [论文解读] x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space
description: >-
  [CVPR 2026][自动驾驶][光流] 提出 x2-Fusion，以事件相机的时空边缘信号为锚构建统一的 Event Edge Space，将图像/LiDAR/事件特征对齐到同质边缘空间后进行可靠性感知自适应融合和跨维度对比学习，同时估计 2D 光流和 3D 场景流，在合成和真实数据上达到 SOTA。
tags:
  - CVPR 2026
  - 自动驾驶
  - 光流
  - 场景流
  - 事件相机
  - 多模态融合
  - 边缘空间
---

# x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space

**会议**: CVPR 2026  
**arXiv**: [2603.16671](https://arxiv.org/abs/2603.16671)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 光流, 场景流, 事件相机, 多模态融合, 边缘空间

## 一句话总结

提出 x2-Fusion，以事件相机的时空边缘信号为锚构建统一的 Event Edge Space，将图像/LiDAR/事件特征对齐到同质边缘空间后进行可靠性感知自适应融合和跨维度对比学习，同时估计 2D 光流和 3D 场景流，在合成和真实数据上达到 SOTA。

## 研究背景与动机

光流和场景流是动态场景理解的核心工具。现有多模态融合方法将图像/LiDAR/事件保持在各自异构特征空间中融合，带来三个问题：

**高复杂度**：无共享通道基础，需逐对模态对齐，导致模块过多

**信息侵蚀**：异构空间延迟融合到晚期，早期失真难以修正

**高脆弱性**：无共同表示基础，退化条件下对齐本身崩溃

核心洞察：事件相机天然提供时空边缘信号——像素级亮度变化精确标记运动边缘——可作为统一所有模态的"边缘锚"。

## 方法详解

### 整体框架

Event Edge Encoder 预训练 → 冻结为边缘原型 → 图像/LiDAR 编码器对齐到 Event Edge Space → 可靠性感知自适应融合 → 跨维度对比学习 → 2D/3D 流输出。

### 关键设计

#### 1. Event Edge Space

**为什么是边缘？** 边缘是模态无关的结构信息，在不同传感器中保持一致。

**为什么用事件？** 事件相机在运动边缘上精确触发，与图像共享 2D 坐标，与 LiDAR 共享稀疏异步采样特性——是连接两者的天然桥梁。

**Event Edge Encoder 预训练**：体素化事件流 → 稀疏 3D CNN → 多尺度特征金字塔。自监督预训练：从过去事件预测未来边缘强度。

边缘强度定义：$e^E(x,y) = \tilde{A}^E(x,y)(1 - \tilde{\sigma}_t(x,y))$，结合归一化事件活跃度和时序方差。

#### 2. 图像-LiDAR 对齐

冻结事件编码器，其特征作为固定边缘原型。图像和 LiDAR 编码器通过投影头映射到相同维度空间。

边缘锚定对称正则化：在 2D（像素级）和 3D（点级）上分别计算三模态间的 L1 距离，以事件边缘图 $e^E$ 作为权重：
$$\mathcal{L}_{align} = \lambda_{2D} \cdot \mathcal{L}_{align}^{2D} + \lambda_{3D} \cdot \mathcal{L}_{align}^{3D}$$

#### 3. 可靠性感知自适应融合

双层可靠性估计：
- **全局可靠性** $\omega_m$：通过时空分解（时序差分+空间梯度）衡量各模态与事件运动信号的一致性
- **局部可靠性** $\mathcal{A}_m(x)$：高通滤波+平均池化+分组卷积后 softmax

融合：$F_{fused}(x) = \sum_m \frac{\omega_m \mathcal{A}_m(x)}{\sum_n \omega_n \mathcal{A}_n(x)} Z_m(x)$

跨注意力 Transformer 进一步增强融合特征。

#### 4. 跨维度对比学习

显式约束帧间运动一致性和 2D-3D 几何一致性，使光流和场景流互相增强。

## 实验关键数据

### EKubric 合成数据

| 方法 | EPE_2D ↓ | ACC_1px ↑ | EPE_3D ↓ | ACC_.05 ↑ |
|------|----------|-----------|----------|-----------|
| RPEFlow | 0.439 | 95.99% | 0.027 | 95.33% |
| **x2-Fusion** | **0.430** | **96.86%** | **0.024** | **96.78%** |

### DSEC 真实数据

| 方法 | EPE_2D ↓ | ACC_1px ↑ | EPE_3D ↓ |
|------|----------|-----------|----------|
| RPEFlow | 0.326 | 95.28% | 0.103 |
| **x2-Fusion** | **0.305** | **95.60%** | **0.092** |

### 退化场景

| 条件 | 提升幅度 |
|------|---------|
| 极端光照 | 显著改善 |
| LiDAR 稀疏 | 显著改善 |

### 消融实验

| 配置 | EPE_2D | EPE_3D | 说明 |
|------|--------|--------|------|
| 无 Event Edge Space | +0.05 | +0.003 | 同质空间对融合至关重要 |
| 无可靠性融合 | +0.03 | +0.002 | 自适应权重在退化条件下尤其重要 |
| 无跨维度对比 | +0.02 | +0.003 | 2D-3D 互相增强有效 |

### 关键发现

- Event Edge Space 是首个将三种模态统一到同质边缘空间的设计
- 可靠性感知融合在退化场景下优势最大
- 跨维度对比使 2D 和 3D 任务互相促进

## 亮点与洞察

1. Event Edge Space 的设计理念优雅——用事件相机的天然边缘信号作为"通用锚"
2. 将融合从"异构空间逐对对齐"简化为"同质空间内权重分配"
3. 边缘强度作为对齐权重——在边缘处精确对齐，非边缘处放松约束

## 局限与展望

1. 事件编码器预训练增加了训练流程复杂度
2. 冻结事件编码器可能限制了自适应能力
3. 当前未处理动态物体遮挡
4. 对事件相机硬件的依赖限制了纯图像+LiDAR 场景的应用

## 相关工作与启发

- 相比 RPEFlow（阶段式融合）：统一空间设计更简洁
- 相比 VisMoFlow（手工物理空间）：Event Edge Space 数据驱动且更一般化
- 事件相机作为"边缘传感器"的视角值得在更多任务中探索

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Event Edge Space 概念新颖，同质化融合范式独创
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据，退化场景验证
- 写作质量: ⭐⭐⭐⭐ 架构图清晰，对比范式图直观
- 价值: ⭐⭐⭐⭐⭐ 对多模态融合流估计领域提供全新思路

<!-- RELATED:START -->

## 相关论文

- [x²-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](../../ICLR2026/autonomous_driving/x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration](lirec-net_a_target-free_and_learning-based_network_for_lidar_rgb_and_event_calib.md)
- [FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision](flashcap_millisecond-accurate_human_motion_capture_via_flashing_leds_and_event-b.md)
- [Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multi_modal_learning_in_3d_human_pose_estimation.md)
- [LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)

<!-- RELATED:END -->
