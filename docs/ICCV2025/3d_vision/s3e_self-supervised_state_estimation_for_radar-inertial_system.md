---
title: >-
  [论文解读] S3E: Self-Supervised State Estimation for Radar-Inertial System
description: >-
  [ICCV 2025][3D视觉][毫米波雷达] 提出S3E，首次实现从雷达信号频谱和惯性数据的互补自监督状态估计，通过基于旋转的跨融合技术增强有限角分辨率下的空间结构信息。
tags:
  - ICCV 2025
  - 3D视觉
  - 毫米波雷达
  - IMU融合
  - 自监督
  - 状态估计
  - 雷达频谱
---

# S3E: Self-Supervised State Estimation for Radar-Inertial System

**会议**: ICCV 2025  
**arXiv**: [2509.25984](https://arxiv.org/abs/2509.25984)  
**代码**: 未公开  
**领域**: 3D视觉  
**关键词**: 毫米波雷达, IMU融合, 自监督, 状态估计, 雷达频谱

## 一句话总结

提出S3E，首次实现从雷达信号频谱和惯性数据的互补自监督状态估计，通过基于旋转的跨融合技术增强有限角分辨率下的空间结构信息。

## 研究背景与动机

毫米波雷达在恶劣条件（雾/雨/雪）下具有独特可靠性，但现有方案面临挑战：

**点云稀疏**：CFAR检测器提取的点云稀疏且含鬼点

**多路径效应**：高反射功率产生假阳性"鬼点"

**角分辨率有限**：单芯片雷达天线数量少

关键动机：
- 使用信息更丰富的距离-方位频谱(RAS)替代稀疏点云
- 雷达提供外感知（补偿IMU漂移），IMU提供内感知（蒸馏运动一致地标）
- 旋转分量在RAS中表现为沿方位轴的线性平移

## 方法详解

### 基于旋转的跨融合 (Rotation-based Cross Fusion)

核心发现：相邻RAS之间主能量的移动量取决于旋转分量。将第$k$帧的最大功率沿方位角线性平移运动角度$\vartheta$，其峰值恰好与第$k+1$帧重合。

IMU预积分获得帧间旋转 $\boldsymbol{q}_{k+1}^k$。利用此旋转增强RAS：

$$\boldsymbol{M}_{k+1}^k = \text{Softmax}\left(-\frac{(\mathbf{1}\boldsymbol{\eta}^T - \vartheta\mathbf{11}^T - \boldsymbol{\eta}\mathbf{1}^T)^2}{\kappa}\right) \cdot \boldsymbol{M}_k$$

$$\boldsymbol{M}_{k+1}' = \boldsymbol{M}_{k+1}^k \oplus \boldsymbol{M}_{k+1}$$

### 一致地标提取器

基于U-Net的多头架构：
- **位置头**：输出检测分数 $L \in \mathbb{R}^{H \times W}$，提取子像素位置
- **分数头**：归一化可信度权重，消除鬼点影响
- **描述子头**：248维特征用于跨帧地标匹配

子像素位置通过Softmax加权：
$$u_k = \sum_{(i,j) \in \mathcal{U}_k} u_{ij}[\text{Softmax}(L_{\mathcal{U}_k})]_{ij}$$

### 可微速度估计

利用静态地标的多普勒速度与车辆速度的余弦约束，建立超定方程：

$$-\begin{bmatrix} v_1^r \\ \vdots \\ v_N^r \end{bmatrix} = \begin{bmatrix} \cos\alpha_1 & \sin\alpha_1 \\ \vdots & \vdots \\ \cos\alpha_N & \sin\alpha_N \end{bmatrix} \begin{bmatrix} v^R\cos\beta \\ v^R\sin\beta \end{bmatrix}$$

通过可微最小二乘求解：$(\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T\mathbf{B}$

### 自监督损失

三个约束联合优化：
- **几何约束** $\mathcal{L}_1$：地标对满足IMU变换矩阵
- **运动学约束** $\mathcal{L}_2$：静态地标满足多普勒余弦约束
- **速度对齐** $\mathcal{L}_3$：IMU+雷达观测速度与变换速度一致

$$\mathcal{L}_{total} = \mathcal{L}_1 + \lambda_1 \mathcal{L}_2 + \lambda_2 \mathcal{L}_3$$

## 实验

### ColoRadar数据集

| 方法 | longboard Trans.↓ | edgar_classroom Trans.↓ | outdoors Trans.↓ |
|------|-------------------|------------------------|-----------------|
| EKF-RIO | - | 5.32 | 4.65 |
| PG-RIO | - | 2.61 | 7.67 |
| Milliego | 9.14 | 2.32 | 2.02 |
| **S3E** | **5.69** | 最优 | 最优 |

S3E在大多数场景中取得最优或极具竞争力的性能。

### 自采集数据集泛化性

在未见场景上评估，S3E作为自监督方法比有监督的Milliego展现更好的泛化能力。

## 亮点与洞察

1. **首次融合雷达频谱+IMU**：绕过稀疏点云，直接使用信息更丰富的RAS
2. **自监督无需定位真值**：通过雷达-IMU互补构建约束实现自监督
3. **旋转-频谱对应关系**：巧妙利用旋转在RAS中表现为线性平移的物理特性
4. **可微速度估计**：从多普勒观测到自车速度的可微求解支持端到端训练

## 局限性

- 仅适用于单芯片低分辨率雷达，4D成像雷达可能不需如此复杂处理
- 高速场景下IMU预积分精度下降
- 静态地标假设在高动态环境下可能不足
- 未进行大规模长期SLAM评估

## 相关工作

- EKF-RIO, PG-RIO: 基于模型的雷达里程计
- Milliego: 监督学习雷达-IMU融合
- RadarHD: 雷达频谱增强

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (雷达频谱+IMU自监督融合首创)
- 技术深度: ⭐⭐⭐⭐⭐ (跨融合+可微速度估计+三重自监督)
- 实验充分度: ⭐⭐⭐⭐ (公开+自采数据集)
- 实用价值: ⭐⭐⭐⭐ (恶劣条件导航的实用方案)

<!-- RELATED:START -->

## 相关论文

- [Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics](self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)
- [No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](../../NeurIPS2025/3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians](sheap_self-supervised_head_geometry_predictor_learned_via_2d_gaussians.md)

<!-- RELATED:END -->
