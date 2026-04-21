---
title: >-
  [论文解读] PRaDA: Projective Radial Distortion Averaging
description: >-
  [CVPR 2025][3D视觉][相机标定] PRaDA 提出了一种完全在射影空间中工作的径向畸变标定方法，通过将多对图像的畸变估计在函数空间中进行加权平均，无需 3D 点重建或相机位姿估计即可获得高精度的畸变校正，在多个具有严重畸变的数据集上显著超越 COLMAP、GLOMAP 等传统方法。
tags:
  - CVPR 2025
  - 3D视觉
  - 相机标定
  - 径向畸变
  - 射影几何
  - 畸变平均
  - 自动标定
---

# PRaDA: Projective Radial Distortion Averaging

**会议**: CVPR 2025  
**arXiv**: [2504.16499](https://arxiv.org/abs/2504.16499)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 相机标定, 径向畸变, 射影几何, 畸变平均, 自动标定

## 一句话总结

PRaDA 提出了一种完全在射影空间中工作的径向畸变标定方法，通过将多对图像的畸变估计在函数空间中进行加权平均，无需 3D 点重建或相机位姿估计即可获得高精度的畸变校正，在多个具有严重畸变的数据集上显著超越 COLMAP、GLOMAP 等传统方法。

## 研究背景与动机

**领域现状**：精确的相机模型是 SfM、SLAM 和新视角合成等几何视觉算法的基础。当前畸变标定主要有三条路线：(1) 基于 SfM 的方法（如 COLMAP/GLOMAP）在 bundle adjustment 中联合估计畸变参数；(2) 学习方法（如 GeoCalib、DeepCalib）直接从单张图像回归参数；(3) N-point 求解器在图像对上估计基础矩阵和畸变。

**现有痛点**：基于 SfM 的方法将畸变估计与 3D 重建耦合，没有好的初始化就容易收敛失败，特别是鱼眼等严重畸变的场景。学习方法缺乏精度和鲁棒性，训练数据中缺乏多样的鱼眼畸变样本。两视图求解器在整个图像范围内通常不够准确，因为对应点可能不覆盖全部区域。

**核心矛盾**：准确估计畸变参数要么需要解决完整的 SfM 问题（复杂且容易失败），要么依赖学习方法（精度不足）。能否在保持 SfM 方法精度的同时避开其复杂性？

**本文目标**：将畸变标定从 3D 重建中解耦，在射影空间中仅通过两视图关系完成高精度畸变估计。

**切入角度**：在射影空间中，几何关系由基础矩阵描述，它可以吸收除畸变之外的所有相机参数（焦距等）。这意味着可以不需要估计焦距、3D 点或位姿就进行畸变优化。同时，学习型匹配器在处理畸变图像时已经足够鲁棒。

**核心 idea**：从每对图像独立估计畸变参数，然后在函数空间中对这些不一致的估计进行加权平均融合，得到单一一致的相机模型。

## 方法详解

### 整体框架

方法的流程分为四步：(1) 对每对图像用特征匹配获取对应点，通过鲁棒求解器估计单参数畸变模型；(2) 对每对进行非线性精炼，升级到高阶多项式模型；(3) 对同一相机的所有两视图估计进行畸变平均，融合为单一模型；(4) 全局精炼跨所有图像对优化 Sampson 误差。全程无需 3D 点或相机位姿。

### 关键设计

1. **两视图初始化与非线性精炼**:

    - 功能：从每对图像获取初始畸变估计并提升到高阶模型
    - 核心思路：使用 LO-RANSAC + F10 求解器获取单参数 division model $d_\lambda(r) = 1/(1+\lambda r^2)$ 初始估计，同时完成内点筛选。然后通过最小化 Sampson 误差进行非线性精炼：$\text{argmin}_{F,\theta} \sum_l r^2_{\text{sampson}}(p_l, q_l, F, \theta_1, \theta_2)$，并将模型升级为 k 阶多项式 $h_\theta(r) = \sum_{i=0}^k \theta_i r^i$。基础矩阵使用 $SO(3) \times S^1 \times SO(3)$ 的 7-DoF 最小参数化
    - 设计动机：单参数模型虽有高效的最小求解器但表达力不足，通过非线性优化升级到高阶模型可以捕获更复杂的真实畸变模式。Sampson 误差只依赖极线约束，无需 3D 点

2. **函数空间中的畸变平均**:

    - 功能：将同一相机的多个不一致畸变估计融合为单一一致模型
    - 核心思路：每个两视图估计只在对应点覆盖的区域可靠，在其他区域可能不一致。将平均问题形式化为加权最小二乘：$\bar{\theta} = \text{argmin}_\theta \sum_i \omega_i \int_0^R \|1/h_\theta(r) - 1/h_{\theta_i}(r)\|^2 r^3 dr$，用加权平均 $\bar{\theta} = \sum \omega_i \theta_i / \sum \omega_i$ 初始化后数值求解。径向对称性使得积分从 2D 图像域简化为 1D 径向域
    - 设计动机：这是论文最核心的创新。单对图像的畸变估计受限于对应点分布，只有将多对估计融合才能覆盖整个图像并获得全局一致的模型。这也使得可以生成任意阶多项式

3. **多项式正则化与全局精炼**:

    - 功能：约束无对应点覆盖区域的畸变行为，并跨所有图像全局一致优化
    - 核心思路：正则化通过最小化 undistortion 函数的变化率 $\min \int_0^R \|dU_\theta(r)/dr\|^2 dr$ 保证单调性。全局精炼联合优化所有图像对的鲁棒 Sampson 误差：$\text{argmin}_{\{F_{ij}\}, \{\theta_k\}} \sum_{l,i,j} \rho(r_{\text{sampson}}(\cdot))$，使用 Cauchy 损失函数处理异常值
    - 设计动机：高阶多项式在无约束区域可能产生不合理行为（如振荡），正则化保证物理一致性。全局精炼让同一相机在所有图像对中共享参数，进一步提升一致性

### 损失函数 / 训练策略

非学习方法，核心度量为 Sampson 误差（极线约束的近似最小像素调整量）。全局精炼使用 Cauchy 鲁棒损失。像素坐标按图像对角线归一化以提高数值稳定性。

## 实验关键数据

### 主实验

焦距调整后的重投影误差 (FA-RE，像素)：

| 方法 | ScanNet++ Mean | ETH3D cam4 Mean | ETH3D cam5 Mean | KITTI-360 cam2 Mean | KITTI-360 cam3 Mean |
|------|----------------|-----------------|-----------------|---------------------|---------------------|
| COLMAP | 2.0 | 26.0 | 25.1 | 125.5 | 112.4 |
| GLOMAP | 1.8 | 18.4 | 19.6 | 122.0 | 113.3 |
| DroidCalib | 1.2 | 36.3 | 46.4 | 102.2 | 128.1 |
| GeoCalib | 4.6 | 35.8 | 34.6 | 125.5 | 123.1 |
| DeepCalib | 10.8 | 20.9 | 18.2 | 160.5 | 153.2 |
| **Ours (SIFT)** | **0.6** | **5.3** | **14.4** | **44.8** | **50.2** |

WoodScape (180° 鱼眼)：前向 51.2 vs DeepCalib 9.7 / GeoCalib 98.0

### 消融实验

PRaDA 初始化 + GLOMAP vs 纯 GLOMAP（ScanNet++ 稀疏测试集）：

| 指标 | PRaDA + GLOMAP | GLOMAP |
|------|---------------|--------|
| 旋转误差 Min/Mean/Max | 0.18/4.51/44.56 | 0.25/28.99/118.6 |
| 平移方向误差 Min/Mean/Max | 0.26/8.70/81.07 | 0.33/27.39/95.76 |

### 关键发现

- 在 KITTI-360 等严重畸变数据集上，PRaDA 的误差约为 COLMAP/GLOMAP 的 1/3 到 1/2
- 在 ScanNet++ 上亚像素误差集中度远高于 SfM 方法
- 解耦标定为 GLOMAP 提供初始化后，平均旋转误差从 28.99° 降至 4.51°，显著改善了 3D 重建
- 前向/后向摄像头是退化情况（极线都是直线），此时畸变估计较差
- 学习型匹配器（LOFTR）在严重畸变的 WoodScape 上也能提供有效对应点

## 亮点与洞察

- 射影空间中的畸变解耦极为优雅：基础矩阵吸收了焦距等参数，使畸变估计成为独立问题
- 函数空间中的畸变平均是一个创新且有数学美感的设计，将多个局部准确的估计融合为全局一致模型
- 不需要跨图像的特征轨迹（point tracks），降低了数据要求
- 方法论上证明了"先精确标定畸变，再做 SfM"比"在 SfM 中联合估计"更可靠

## 局限与展望

- 假设径向对称畸变，无法处理非径向对称的畸变模型（如 ETH3D/KITTI-360 的 GT 模型）
- 依赖匹配器质量，如果匹配器主要在针孔设置下训练，在严重畸变区域可能引入非高斯误差
- 使用预定义的 RANSAC 阈值，作者建议未来可用 σ++ 共识方法完全避免阈值
- 未估计焦距，需要额外步骤获取完整相机模型

## 相关工作与启发

- Fitzgibbon [12] 的 division model 和 Kukelova [30] 的 F10 求解器是方法的基石
- 与 GLOMAP [41] 的全局 SfM 思路有相似之处，但 PRaDA 不需要 3D 点
- 启发：几何原理可以用传统方法精确处理，不需要用神经网络重新学习——这与 Sarlin 等人的观点一致

## 评分

- **新颖性**: 9/10 — 射影框架下的畸变平均是全新概念，数学推导优雅
- **实验充分度**: 8/10 — 四个包含严重畸变的数据集，与五种基线对比全面
- **写作质量**: 8/10 — 数学推导清晰，动机明确，但部分符号较密集
- **价值**: 8/10 — 为自动标定提供了新的高精度工具，作为 SfM 预处理步骤实用价值高

<!-- RELATED:START -->

## 相关论文

- [Deformable Radial Kernel Splatting](deformable_radial_kernel_splatting.md)
- [Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)
- [Volumetrically Consistent 3D Gaussian Rasterization](volumetrically_consistent_3d_gaussian_rasterization.md)
- [SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation](sharpdepth_sharpening_metric_depth_predictions_using_diffusion_distillation.md)
- [Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)

<!-- RELATED:END -->
