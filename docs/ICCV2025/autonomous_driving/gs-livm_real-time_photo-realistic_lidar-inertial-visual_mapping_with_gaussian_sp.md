---
title: >-
  [论文解读] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting
description: >-
  [ICCV 2025][自动驾驶][3D高斯溅射] 提出 GS-LIVM，首个为大规模无界室外场景设计的实时光真实感 LiDAR-惯性-视觉建图框架，通过体素级高斯过程回归（Voxel-GPR）解决 LiDAR 点云稀疏不均匀问题，利用协方差中心化设计快速初始化 3D 高斯参数，在多个室外数据集上达到 SOTA 的建图效率和渲染质量。
tags:
  - ICCV 2025
  - 自动驾驶
  - 3D高斯溅射
  - LiDAR-IMU-视觉融合
  - 实时建图
  - 高斯过程回归
  - 新视角合成
---

# GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: [2410.17084](https://arxiv.org/abs/2410.17084)  
**代码**: [GitHub](https://github.com/xieyuser/GS-LIVM)  
**领域**: 自动驾驶  
**关键词**: 3D高斯溅射, LiDAR-IMU-视觉融合, 实时建图, 高斯过程回归, 新视角合成

## 一句话总结
提出 GS-LIVM，首个为大规模无界室外场景设计的实时光真实感 LiDAR-惯性-视觉建图框架，通过体素级高斯过程回归（Voxel-GPR）解决 LiDAR 点云稀疏不均匀问题，利用协方差中心化设计快速初始化 3D 高斯参数，在多个室外数据集上达到 SOTA 的建图效率和渲染质量。

## 研究背景与动机
SLAM 是机器人和自动驾驶的基石技术。传统 SLAM 使用稀疏特征表示，虽然定位有效但无法实现高质量3D重建和光真实感渲染。NeRF 和 3DGS 等神经场景表示方法的出现改变了这一格局，但将其应用于大规模无界室外场景仍面临重大挑战：

**点云稀疏不均匀**：多线旋转 LiDAR 和非重复扫描 LiDAR 产生的点云分布不均，直接用于 3DGS 会导致内存低效和优化困难

**方向偏置**：室外 SLAM 主要是单向运动，导致 3D 高斯优化偏向相机方向，新视角渲染质量显著下降

**实时性困境**：现有方法（如 Gaussian-LIC）每帧迭代约 1 秒，远达不到实时要求（100ms），且无法处理完整室外序列（GPU 内存溢出）

**离线方法过拟合**：离线优化方法过度拟合监督视角，新视角合成质量差

**核心 idea**：以协方差为中心，用体素级 GPR 生成均匀点云 → 利用 GPR 输出的协方差初始化 3D 高斯的尺度和旋转 → 迭代更新 GPR 参数和高斯地图结构，实现实时高质量建图。

## 方法详解

### 整体框架
系统分为跟踪线程和建图线程。跟踪线程使用 ESIKF 算法融合 LiDAR/IMU/相机数据进行实时状态估计（IMU 频率输出里程计）。建图线程接收着色点云，通过 Voxel-GPR 生成均匀分布的预测点云，初始化 3D 高斯后整合进稠密地图进行渲染优化。

### 关键设计
1. **体素级高斯过程回归（Voxel-GPR）**:

    - 功能：将稀疏不均匀的 LiDAR 点云转换为均匀分布的网格点云
    - 核心思路：对每个体素内的点云 $\mathcal{P}_\alpha$ 进行 PCA 分析确定主方向，将3D问题降为2D→1D的 GPR 问题。选择与特征向量最小夹角的轴为值轴 $\mathbf{f}_\alpha$，其余两轴为参数轴 $\mathbf{x}_\alpha$。在参数平面上生成均匀网格 $\mathbf{x}_{\alpha*}$，通过 GPR 预测值轴坐标：
    $\mathbf{f}_{\alpha*} = \boldsymbol{\mu}_{\alpha*} = \mathbf{K}_{\alpha*}^\top (\mathbf{K}_\alpha + \sigma^2 \mathbf{I})^{-1} \mathbf{f}_\alpha$
      以及协方差 $\boldsymbol{\Sigma}_{\alpha*}$。使用 CUDA 并行处理数百个体素，单帧耗时 < 30ms
    - 设计动机：不均匀点云导致梯度监督不均，快速移动时 3DGS 难以收敛；GPR 生成的均匀点云使用更少的 3D 高斯即可保持/提升性能，显著降低 GPU 内存消耗

2. **协方差驱动的 3D 高斯快速初始化**:

    - 功能：利用 GPR 输出的协方差快速估计每个 3D 高斯的尺度和旋转参数
    - 核心思路：将每个体素分为 $n_s \times n_s$ 个子网格。对第 $\beta$ 个子网格中的预测点，以 GPR 协方差的倒数为权重计算加权中心 $\mathbf{p}^\beta$ 和加权协方差矩阵 $\Phi^\beta$：
    $\Phi^\beta = \frac{\mathbf{Q}^\top \cdot \text{diag}(w_1^\beta, ..., w_{n_r^2}^\beta) \cdot \mathbf{Q}}{\sum w_i^\beta}$
      尺度参数 $S^\beta = \text{diag}(\Phi^\beta)$，旋转初始化为单位四元数。颜色通过将位置重投影到当前图像获取
    - 设计动机：原始 3DGS 初始化依赖最近邻距离计算尺度、旋转统一初始化为常数，收敛慢。协方差低的区域（信心高）权重大，确保几何可靠的初始化

3. **迭代优化框架（地图扩展 + 协方差更新 + 相似性正则化）**:

    - 功能：持续更新 GPR 参数和 3D 高斯地图结构
    - 核心思路：将体素分为四类（未探索/待处理/活跃/已收敛），仅处理待处理和活跃体素以提高效率。提出两个正则化损失：
        - **Delta 深度相似性损失** $\mathcal{L}_d$：约束相邻帧渲染深度的一致性，增强新视角合成的鲁棒性
        - **结构相似性损失** $\mathcal{L}_p$：测量当前 3D 高斯地图与最新 LiDAR 帧的欧氏距离，加速优化
    - 设计动机：仅靠光度损失在室外单向运动场景容易过拟合

### 损失函数 / 训练策略
$$\mathcal{L} = (1-\lambda_s)\|C - C_{gt}\|_1 + \lambda_s \mathcal{L}_{ssim} + \lambda_d \sum \mathcal{L}_d(\mathcal{F}_*, \mathcal{F}_{*+1}) + \lambda_p \mathcal{L}_p$$

使用当前相机窗口 $\mathcal{Q}_{curr}$ 和随机采样的历史帧 $\mathcal{Q}_{hist}$ 联合优化，防止灾难性遗忘。

## 实验关键数据

### 主实验

| 数据集/序列 | 指标 | GS-LIVM | 3DGS(离线) | MonoGS | NeRF-SLAM |
|------------|------|---------|-----------|--------|-----------|
| hku_campus_seq_00 | PSNR↑/SSIM↑/LPIPS↓ | **22.43/0.719/0.247** | 21.74/0.719/0.302 | 12.14/0.368/0.608 | 13.23/0.410/0.653 |
| Visual_Challenge | PSNR↑ | **21.81** | 18.55 | 13.48 | 12.98 |
| eee_02 (Ouster-16) | PSNR↑ | **20.72** | 20.39 | 11.63 | 8.32 |
| 1005_00 (Botanic) | PSNR↑ | 21.12 | **22.17** | 14.56 | 9.79 |

### 消融实验

| 配置 | PSNR | SSIM | 说明 |
|------|------|------|------|
| 无 Voxel-GPR（原始点云） | 较差 | 较差 | 不均匀分布导致优化困难 |
| 无协方差初始化 | 需 1.3x 时间 | - | 收敛速度慢 |
| 无结构相似性损失（SS） | 需 1.3x 时间 | - | 细节表示不足 |
| $n_s=3$, $\eta=0.3$, +SS, +DDS | 19.29 | 0.640 | 基础完整配置 |
| $n_s=4$, $\eta=0.3$, +SS, +DDS | **20.14** | **0.672** | 更多高斯提升质量 |
| $\eta=0.1$（更严格收敛） | 18.49 | 0.637 | 时间增 6x 但质量无提升 |

### 关键发现
- 所有序列均能在 8GB GPU 上完成完整建图（$n_s=3$），最大内存消耗约 3.5GB
- Botanic Garden 1018_13 序列（208s）的建图时间始终低于实时限制
- Voxel-GPR 使高斯数量减少的同时保持或提升渲染质量
- $n_s$ 选择存在精度-速度权衡：小场景提升显著，大场景可能 GPU 内存溢出

## 亮点与洞察
- **首个实时光真实感 LiDAR-惯性-视觉 SLAM 框架**：在大规模室外场景实现了真正的实时（满足 $t \leq \mathcal{D}/\mathcal{C}$）
- **协方差中心化设计**：GPR 的协方差同时服务于 3D 高斯初始化和 GPR 参数更新，一箭双雕
- **CUDA 并行化**：Voxel-GPR 处理数百个体素仅需 < 30ms，实际可部署
- 体素类型分类（a/b/c/d）减少不必要的计算，是工程上的重要优化

## 局限与展望
- $n_s$ 在大场景下可能导致内存问题，需要自适应调整策略
- 仅使用零阶球谐函数（SH），限制了视角依赖的颜色表达
- 不支持动态物体建模，运动物体会产生伪影
- 定位精度依赖前端里程计（FAST-LIVO/R3LIVE），闭环优化能力有限
- Delta 深度相似性损失在快速运动时可能引入假阳性约束

## 相关工作与启发
- Voxel-GPR 的思路可推广到其他需要点云补全的任务（如稀疏 LiDAR 语义分割）
- 协方差初始化的方法在其他 3DGS 应用中也有价值（如城市级重建）
- 迭代更新体素状态的策略类似增量式 SLAM 的关键帧管理

## 评分
- 新颖性: ⭐⭐⭐⭐ Voxel-GPR + 协方差初始化的组合新颖，但各组件均有前驱工作
- 实验充分度: ⭐⭐⭐⭐ 多 LiDAR 配置验证，消融详细，但缺少与最新 Gaussian-LIC 的直接数值对比
- 写作质量: ⭐⭐⭐⭐ 算法伪代码清晰，系统设计描述完整，图表信息丰富
- 价值: ⭐⭐⭐⭐⭐ 实时大规模室外光真实感建图是机器人和自动驾驶的刚需，开源代码加分

<!-- RELATED:START -->

## 相关论文

- [Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [RTMap: Real-Time Recursive Mapping with Change Detection and Localization](rtmap_real-time_recursive_mapping_with_change_detection_and_localization.md)
- [6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dope-gs_online_6d_object_pose_estimation_using_gaussian_splatting.md)
- [GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)

<!-- RELATED:END -->
