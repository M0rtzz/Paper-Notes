---
title: >-
  [论文解读] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency
description: >-
  [CVPR 2026][自动驾驶][单目深度估计] 本文提出PTC-Depth，一个结合光流三角化和轮式里程计的单目深度估计框架，通过递归贝叶斯更新追踪深度基础模型的度量尺度，实现时间一致的度量深度预测，在KITTI、TartanAir和热红外等多个数据集上展现强泛化能力。
tags:
  - CVPR 2026
  - 自动驾驶
  - 单目深度估计
  - 时间一致性
  - 贝叶斯尺度融合
  - 光流三角化
  - 轮式里程计
---

# PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency

**会议**: CVPR 2026  
**arXiv**: [2604.01791](https://arxiv.org/abs/2604.01791)  
**代码**: https://ptc-depth.github.io  
**领域**: 自动驾驶 / 深度估计  
**关键词**: 单目深度估计, 时间一致性, 贝叶斯尺度融合, 光流三角化, 轮式里程计

## 一句话总结

本文提出PTC-Depth，一个结合光流三角化和轮式里程计的单目深度估计框架，通过递归贝叶斯更新追踪深度基础模型的度量尺度，实现时间一致的度量深度预测，在KITTI、TartanAir和热红外等多个数据集上展现强泛化能力。

## 研究背景与动机

1. **领域现状**：单目深度估计（MDE）已广泛应用于自动驾驶和移动机器人，深度基础模型（如Depth Anything v2）在零样本泛化上取得了巨大进展，但大多只能预测相对深度（缺乏绝对度量尺度）。

2. **现有痛点**：(a) 单帧深度估计在连续帧间存在严重的时间不一致性（抖动和突变）；(b) 视频深度模型（如VDA）改善了一致性但仍不提供度量深度；(c) 深度补全方法（如OGNI-DC）需要额外的LiDAR深度，不适用于仅有相机+里程计的场景。

3. **核心矛盾**：相对深度模型有好的结构保留和泛化能力，但缺乏度量尺度；度量深度模型有绝对尺度但泛化差（如UniDepth在out-of-distribution场景下大幅退化）。两者的优势难以简单结合。

4. **本文目标** 仅使用单目相机和轮式里程计（无需LiDAR/深度传感器），将深度基础模型的相对深度转换为时间一致的度量深度。

5. **切入角度**：观察到轮式里程计提供的度量基线和光流共同约束了深度的度量尺度，利用连续帧间的三角化获取稀疏度量深度，再通过递归贝叶斯框架跟踪全局/局部尺度因子。

6. **核心 idea**：将相对深度到度量深度的转换建模为尺度场$S$的贝叶斯递归估计问题，用超像素分割实现局部尺度适配。

## 方法详解

### 整体框架

输入为连续视频帧和轮式里程计数据。Pipeline分四步：(1) 计算连续帧间光流；(2) 利用光流和相对深度通过RANSAC估计相机位姿，轮式里程计提供度量基线；(3) 基于位姿和光流进行三角化获取稀疏度量深度；(4) 通过递归贝叶斯更新融合三角化深度和上一帧传播的先验深度，得到最终的度量深度图。

### 关键设计

1. **基于运动场的鲁棒位姿估计**:

    - 功能：从光流和相对深度中恢复相机旋转$\boldsymbol{\Omega}$和平移方向$\hat{\boldsymbol{T}}$
    - 核心思路：利用Longuet-Higgins运动场公式，将光流分解为旋转项$\mathbf{B}\boldsymbol{\Omega}$和平移项$\frac{1}{\alpha d^{rel}}\mathbf{A}\boldsymbol{T}$。先假设相对深度$d^{rel}$通过单一尺度因子$\alpha$转换为度量深度，将位姿恢复建模为超定线性系统。使用RANSAC + 分区采样（图像划分为网格，每格等量采样）排除动态物体的光流外点，最终用IRLS+Huber权重精炼
    - 设计动机：动态物体的光流不反映相机运动，必须排除。分区采样确保RANSAC假设覆盖整个视场，避免偏向特定区域

2. **基于Sampson残差的三角化质量评估**:

    - 功能：为每个三角化深度点分配可靠性权重
    - 核心思路：对每个光流对应点进行三角化获取度量深度$z^{tri}$，同时计算Sampson残差$\rho$衡量该对应关系满足对极约束的程度。Sampson残差小说明匹配可靠、三角化准确；大则标记为不可靠。这个逐像素的可靠性分数直接用于贝叶斯融合中的观测不确定性 $V^{obs} = \sigma^2 \frac{\rho}{f_x f_y}$
    - 设计动机：三角化可能因光流误差、动态物体或位姿不准而失败，需要逐像素的可靠性度量而非全局阈值

3. **递归贝叶斯尺度融合（核心）**:

    - 功能：将三角化的稀疏度量深度和上一帧传播的先验融合为时间一致的度量深度
    - 核心思路：不在深度空间直接融合，而是估计潜在尺度场$S$使得$Z = S \cdot d^{rel}$。从上一帧传播先验尺度$S^{prior} = Z^{prior}/d^{rel}$，从三角化获取观测尺度$S^{obs} = Z^{tri}/d^{rel}$。执行逐像素的Kalman更新：计算归一化创新量$\gamma$进行异常值检测（chi-square检验），用一致性约束的Kalman增益$\kappa$融合先验和观测。此外，当帧级几何质量差时（中位Sampson残差大），自适应膨胀先验方差
    - 设计动机：在尺度空间而非深度空间操作保留了相对深度$d^{rel}$的结构连贯性，避免直接融合产生的平滑伪影。约束增益$\kappa$防止了先验和观测弱一致时的过度更新

4. **超像素级别尺度整合**:

    - 功能：解决仿射不变深度模型的shift分量问题
    - 核心思路：使用Felzenszwalb分割将图像划分为超像素，边界跟随$d^{rel}$的几何结构。在每个超像素$\Lambda_\ell$内，取后验尺度的中位数$\bar{s}_\ell$作为该区域的统一尺度，低拟合误差的区域使用局部尺度，否则回退到全局尺度估计。最终度量深度为 $Z^{post} = S^{seg} \cdot d^{rel}$
    - 设计动机：单一全局尺度无法完全补偿仿射不变模型的shift分量，局部尺度估计能更好地适应场景中不同深度区域的尺度差异

### 损失函数 / 训练策略

本方法是**无需训练**的推理框架，不涉及神经网络训练或微调。深度基础模型（Depth Anything v2）以冻结方式使用，所有计算都是解析的（光流、RANSAC、贝叶斯更新）。

## 实验关键数据

### 主实验

全范围（0-80m）深度估计：

| 数据集 | 方法 | AbsRel ↓ | δ<1.25 ↑ | TAE ↓ |
|--------|------|----------|----------|-------|
| KITTI | UniDepth | **0.047** | **0.977** | 4.34 |
| KITTI | **Ours** | 0.137 | 0.877 | 5.35 |
| TartanAir | **Ours** | **0.427** | **0.688** | 5.42 |
| TartanAir | UniDepth | 0.503 | 0.176 | 11.11 |
| Roadside | **Ours** | **0.309** | **0.725** | 5.27 |
| Roadside | UniDepth | 0.465 | 0.201 | 11.92 |
| MS2 (热红外) | **Ours** | 0.247 | 0.700 | 5.29 |
| MS2 (热红外) | DA v2 metric | 0.405 | 0.187 | 4.87 |

近距离（0-20m）深度估计：

| 数据集 | 方法 | AbsRel ↓ | δ<1.25 ↑ |
|--------|------|----------|----------|
| TartanAir | **Ours** | **0.339** | **0.712** |
| TartanAir | UniDepth | 0.485 | 0.202 |
| Roadside | **Ours** | **0.165** | **0.860** |
| Roadside | UniDepth | 0.432 | 0.241 |

### 消融实验（三角化位姿来源对比）

| 方法 | 位姿来源 | KITTI AbsRel | TartanAir AbsRel | Roadside δ1 |
|------|----------|-------------|-------------------|-------------|
| MADPose | UniDepth度量深度 | 0.115 | 0.481 | 0.222 |
| **Ours** | 里程计+光流 | **0.115** | **0.239** | **0.649** |
| GT Pose | 真实位姿 | 0.130 | 0.168 | - |

### 关键发现

- 在KITTI（in-distribution）上UniDepth最强，但在所有out-of-distribution数据集上本文方法显著占优（TartanAir AbsRel 0.427 vs 0.503，Roadside 0.309 vs 0.465）
- MADPose依赖UniDepth的泛化能力，在OOD数据集上三角化精度大幅下降（TartanAir AbsRel 0.481），而本方法仅依赖里程计恢复尺度，保持一致的高精度
- 近距离（0-20m）三角化效果最好，因为基线足够大形成良好的三角化几何；远距离（20-80m）视差减小导致三角化退化，这是所有基于几何的方法的固有局限
- VDA达到了好的时间一致性（低TAE），但其度量精度在OOD场景大幅退化（Roadside AbsRel 2.198），说明一致地错也能有低TAE

## 亮点与洞察

- **无需训练的通用框架**：不依赖特定数据集训练，仅需一个冻结的相对深度模型+轮式里程计，即可在RGB和热红外上工作。这种即插即用的设计非常适合机器人部署场景
- **尺度空间融合而非深度空间**：在$S = Z/d^{rel}$空间做贝叶斯融合，保留了基础模型预测的边界清晰度和空间结构，这个insight可迁移到所有需要将相对预测转换为绝对预测的问题
- **Sampson残差作为逐像素可靠性度量**：不需要训练一个专门的置信度网络，直接用几何约束的满足程度作为权重，既优雅又高效

## 局限与展望

- 远距离（>20m）的三角化精度受限于短基线的视差消失问题，这是几何方法的固有限制
- 对轮式里程计精度有一定依赖——在不平坦地形或轮胎打滑时里程计误差会传播到尺度估计
- MS2数据集上的性能受限于热红外图像的光流质量和里程计同步精度
- 超像素分割的参数（Felzenszwalb的阈值）对不同场景可能需要调整
- 仅处理了仿射不变模型的尺度分量，对shift分量的局部补偿依赖超像素大小

## 相关工作与启发

- **vs UniDepth**: UniDepth在训练域内（KITTI）最强，但泛化差。本文方法利用几何约束避免了对特定度量深度训练的依赖，泛化性更好
- **vs VDA (Video Depth Anything)**: VDA是视频深度模型，时间一致性好但无度量尺度且OOD退化严重（Roadside AbsRel 2.198）。本方法在提供度量深度的同时保持合理的时间一致性
- **vs MADPose**: MADPose用UniDepth做度量位姿估计，本质上继承了UniDepth的泛化瓶颈。本文仅用里程计恢复尺度，彻底解耦了对度量深度模型的依赖

## 评分

- 新颖性: ⭐⭐⭐ 贝叶斯尺度融合框架是已知技术的精心组合，核心idea（用里程计约束尺度+Kalman融合）不算全新，但超像素局部尺度和Sampson残差加权是nice的设计
- 实验充分度: ⭐⭐⭐⭐ 覆盖RGB+热红外、真实+合成、多个OOD场景，三角化和深度估计分别评估，近/远距离分析深入
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰完整，框架图直观，实验分析有层次
- 价值: ⭐⭐⭐⭐ 对机器人和自动驾驶的实际部署价值高——无需训练、无需额外深度传感器、跨模态工作

<!-- RELATED:START -->

## 相关论文

- [Distilling Monocular Foundation Model for Fine-grained Depth Completion](../../CVPR2025/autonomous_driving/distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)
- [Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multimodal_learning_in_3d_human_p.md)
- [InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset](incarpose_in-cabin_relative_camera_pose_estimation_model_and_dataset.md)
- [Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth-_and_region-guided_3d_occupancy_from_surround-view_cameras_for_auton.md)
- [Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](../../CVPR2025/autonomous_driving/helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)

<!-- RELATED:END -->
