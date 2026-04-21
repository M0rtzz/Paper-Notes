---
title: >-
  [论文解读] MonoSOWA: Scalable Monocular 3D Object Detector Without Human Annotations
description: >-
  [ICCV2025][自动驾驶][单目3D检测] 提出首个完全不依赖人工标注（包括 2D 和 3D）的单目 3D 物体检测方法，通过新提出的局部目标运动模型（LOMM）解耦帧间运动来源，自动标注速度比前人快 700 倍，并通过规范目标空间（COS）融合不同相机设置的多数据集训练。
tags:
  - ICCV2025
  - 自动驾驶
  - 单目3D检测
  - 弱监督
  - 自动标注
  - 伪LiDAR
  - 无人工标注
---

# MonoSOWA: Scalable Monocular 3D Object Detector Without Human Annotations

**会议**: ICCV2025  
**arXiv**: [2501.09481](https://arxiv.org/abs/2501.09481)  
**代码**: [github.com/jskvrna/MonoSOWA](https://github.com/jskvrna/MonoSOWA)  
**领域**: autonomous_driving  
**关键词**: 单目3D检测, 弱监督, 自动标注, 伪LiDAR, 无人工标注

## 一句话总结

提出首个完全不依赖人工标注（包括 2D 和 3D）的单目 3D 物体检测方法，通过新提出的局部目标运动模型（LOMM）解耦帧间运动来源，自动标注速度比前人快 700 倍，并通过规范目标空间（COS）融合不同相机设置的多数据集训练。

## 研究背景与动机

- 单目 3D 目标检测是自动驾驶的关键组件，传统方法依赖 LiDAR 传感器和大量人工 3D 标注
- 标注过程极其耗时、昂贵，限制了训练数据的多样性；每当相机设置变化就需要重新采集标注
- 现有弱监督方法仍需 2D 实例掩码（如 WeakMono3D、VSRD）或 LiDAR 数据（如 WeakM3D、Autolabels）
- VSRD 每帧标注需约 15 分钟，对 Waymo 的 15.8 万帧需要约 4 年才能标注完，实际上无法扩展
- WeakMono3D 和 VSRD 都无法处理运动目标：只能丢弃或给低置信度，浪费大量训练信息
- 本文目标：**完全消除人工标注和 LiDAR 依赖**，使大多数仅配备单目相机的量产车辆数据可被直接利用

## 方法详解

### 整体流水线

输入仅需：**图像序列 + 自车运动数据（GPS/IMU）+ 已知相机参数**（内参和外参）。不需要任何人工标注，甚至不需要 2D 标注。使用两个通用预训练模型：
1. **2D 目标检测器**：MViT2-Huge（Detectron2 框架，COCO 预训练）——提供实例分割掩码
2. **单目深度估计器**：Metric3D v2（选择它是因为零样本度量深度的泛化性最好）

自动标注流水线将 7-DOF 的 3D 包围框估计问题分解为朝向、位置、尺寸三个子问题分步求解，避免了直接联合优化的困难。

### Step 1: 伪 LiDAR 聚合

- 对每帧图像推理度量深度图，反投影为 3D 点云
- 利用 2D 检测器的实例分割掩码，提取每个目标对应的点云 $P_{i,j}$
- 在 3D 世界坐标系中进行目标跟踪：
    - 用目标点云的中位数近似空间位置
    - 基于物理运动模型预测下一帧位置
    - 最近邻 + 距离阈值匹配
- 将多帧点云用自车运动数据变换到参考帧坐标系

### Step 2: 局部目标运动模型（LOMM）

这是论文的核心创新，解决了一个关键问题：由于自车也在行驶中，所有目标在帧间都会发生相对位移，如何区分哪些是真正在运动的目标？

- **核心创新**：解耦帧间目标位置变化为**自车运动**和**目标自身运动**两个来源
- **分类方法**：对每个目标实例计算帧间位置差分序列 $\Delta_{i,j} = L_{i,j} - L_{i-1,j}$
    - 均值 $\mu_j = \frac{1}{l-k} \sum \Delta_{i,j}$，标准差 $\sigma_j$（除以 $\sqrt{2}$ 归一化）
    - 比值 $z_j = \|\mu_j\|_2 / \|\sigma_j\|_2$
    - 静止车辆：位移变化来自伪 LiDAR 噪声抖动（类似随机游走），$\|\mu\|_2 \ll \|\sigma\|_2$
    - 运动车辆：存在持续方向性运动，$\|\mu\|_2 \gg \|\sigma\|_2$
    - 判定条件：$z > T_z = 0.2$ 且净位移距离 $> T_m = 5\text{m}$
- **对静止目标**：直接聚合所有帧的点云 $A_j = \{P_{i,j}\}$（因为已变换到参考帧），得到更密集的表示，且能从遮挡中恢复
- **对运动目标**：利用已知轨迹，根据物理约束直接计算朝向（车辆沿运动方向行驶）
- 对比前人方法（WeakMono3D、VSRD 丢弃或降权运动目标），LOMM 可**同时利用静止和运动目标的信息**

### Step 3: 分步自动标注

将 7-DOF 位姿估计分解为三个独立子问题：

**朝向估计**：
- 运动目标：直接由相邻帧位置差计算 $\theta = \text{atan2}(\Delta z / \Delta x)$，取前后各 5 帧 yaw 的中位数增强鲁棒性
- 静止目标：在 BEV 中迭代所有角度 $\theta \in [0, \pi/2)$，使用新提出的**饱和紧密度准则**（SCC）选择最优角度
    - 改进 Zhang 的 Closeness Criterion：(1) 用 $\sigma(\alpha \cdot x)$（$\alpha=10$）做 sigmoid 饱和，抑制离群点影响；(2) 用第 10/90 百分位替代 min/max 作为边界参考点
    - 最终取两个互相垂直轴上每个点到最近边界距离的最小值累加为准则值
    - 因算法不区分前后，产生两个朝向假设

**尺寸估计**：
- 从 SCC 算法输出中获取，异常值（超出典型车辆尺寸）用先验尺寸替代
- 对视角导致的不可观测维度进行检测：当车辆朝向与观察角度差接近 $0, \pi/2, \pi, 3\pi/2$ 时，用先验尺寸替代

**位置精修**：
- 以已知朝向和粗位置为基础，沿 x、z 轴施加小扰动（≤2m）
- 使用通用车辆模板点云的 Template Fitting Loss（TFL）选择最优位置（TFL 比 Chamfer Distance 更抗离群点）
- 同时确定前后朝向（测试 $\theta$ 和 $\theta + \pi$ 两个假设，取 TFL 更低者）

### Step 4: 规范目标空间（COS）

- **问题**：不同相机焦距下，同一目标在同一距离的图像大小不同；极端情况下网络需要对相同像素大小的目标预测截然不同的距离
- **解决方案**：选择规范焦距 $f^C = 750$，仅对标签的 3D 坐标缩放：$\omega_i = f^C / f_i$，$(x^C, y^C, z^C) = (x \cdot \omega_i, y \cdot \omega_i, z \cdot \omega_i)$
- 训练时模型在 COS 中学习，推理时用目标帧的 $\omega_j$ 逆变换回世界坐标
- 数据增强（如图像缩放）时需同步调整感知焦距
- 受 Metric3D 和 Omni3D 启发，但仅变换标签坐标而非整个图像或深度图，设计最小侵入
- 这使得**一个模型即可在不同相机设置上训练和推理**，无需针对每种相机单独训练

### 实现细节

- 检测器：MonoDETR 作为最终的 3D 检测模型，使用 AdamW 优化器（lr=2e-4, wd=1e-4）
- 聚合窗口：100 帧（每个目标最多聚合前后各 50 帧的点云信息）
- 阈值设定：$T_z = 0.2$（运动/静止分类），$T_m = 5$m（最小运动距离），$\alpha = 10$（SCC 陡度）
- 所有三个数据集使用相同超参数，体现方法的泛化性

## 实验关键数据

### KITTI-360 测试集

| 方法 | 人工标注 | AP_BEV@0.5 (Easy/Hard) | AP_3D@0.5 (Easy/Hard) | 标注速度 |
|------|----------|------------------------|-----------------------|----------|
| MonoFlex (全监督) | 3D框 | 50.82/41.78 | 43.11/34.43 | - |
| MonoDETR (全监督) | 3D框 | 47.21/36.05 | 41.01/30.38 | - |
| Autolabels | LiDAR+掩码 | 20.18/14.33 | 4.69/2.79 | 6s/帧 |
| VSRD | 掩码 | 29.07/22.83 | 21.77/16.46 | 15min/帧 |
| **MonoSOWA（无标注）** | **无** | **38.41/35.26** | **29.98/27.56** | **1.3s/帧** |

### Waymo 验证集（Level 2）

| 方法 | AP_BEV@0.5 All | AP_3D@0.5 All |
|------|----------------|---------------|
| MonoDETR (全监督) | 23.63 | 21.41 |
| **MonoSOWA（无标注）** | **18.98** | **13.46** |

### 伪标签预训练 + 少量人工标注微调（KITTI）

| 预训练 | 人工标注比例 | AP_BEV@0.7 Easy | AP_3D@0.7 Easy |
|--------|-------------|-----------------|----------------|
| 无 | 25% | 31.72 | 21.76 |
| **MonoSOWA** | **25%** | **39.99** | **32.64** |
| 无 | 100% | 37.99 | 29.36 |

- 使用 MonoSOWA 预训练 + 仅 25% 人工标注即超越 100% 人工标注的全监督训练
- 15% 人工标注 + MonoSOWA 预训练 ≈ 100% 人工标注性能 → 节省 85% 标注成本

### 跨数据集训练

| 训练数据 | AP_BEV@0.5 (Easy) | AP_3D@0.5 (Easy) |
|----------|--------------------|--------------------|
| KITTI 伪标签 | 61.24 | 53.22 |
| K360 伪标签 | 57.62 | 47.39 |
| KITTI+K360 伪标签 | **64.29** | **58.97** |
| KITTI 人工标签 | 67.44 | 65.09 |

多数据集伪标签联合训练可接近人工标签性能。在 Hard@0.3 指标上甚至超越了人工标签。

### 消融实验

| LOMM | SCC | AP_BEV@0.5 Easy |
|------|-----|-----------------|
| ✗ | ✗ | 20.41 |
| ✗ | ✓ | 20.37 |
| ✓ | ✗ | 35.89 |
| ✓ | ✓ | **39.22** |

LOMM 是性能提升的关键因素；SCC 在有 LOMM 基础上进一步提升 ~3.3 AP。

## 优势与局限

**优势**：
- 首个完全无人工标注的单目 3D 检测方法（不需要 2D 和 3D 标注）
- 标注速度约 1.3s/帧，比 VSRD 快约 700 倍
- LOMM 首次允许利用运动目标的时序信息（前人方法只能丢弃运动目标）
- COS 使单一模型可跨数据集训练和推理
- 作为预训练工具可节省 85% 人工标注成本

**局限**：
- 对远距离目标（几个像素高度）检测精度较低，这是单目检测的固有限制而非标注流水线问题
- 依赖 Metric3D 的零样本深度估计质量
- AP 指标在 IoU=0.3 时与 VSRD 仍有差距，原因是 KITTI-360 人工标签为 amodal（包含被遮挡部分），这天然有利于使用人工掩码的 VSRD；当使用相同输入时 MonoSOWA 全面超越 VSRD
- 仅验证了车辆（car）类别，行人、骑行者等非刚体/小目标类别未涉及

## 个人思考

- 整体设计哲学非常实用：将复杂的 7-DOF 估计分解为朝向、位置、尺寸三个更易处理的子问题，每步都有针对性的解法
- LOMM 的核心洞察（$\mu/\sigma$ 比值区分静止/运动）简单而有效，是 signal-to-noise ratio 的经典应用
- COS 的引入使得"一个模型适配所有相机"成为可能，这对规模化部署意义重大
- 700 倍的速度提升使方法真正可应用于大规模数据集，这是从研究到工程的关键突破
- 25% 标注即超 100% 全监督的预训练效果说明伪标签质量已相当可靠
- 不需要 LiDAR 意味着可以利用普通行车记录仪数据，数据获取成本大幅降低
- 潜在改进方向：替换 Metric3D 为 Depth Anything V2 等更新模型、引入自监督深度一致性约束、扩展到行人和骑行者等类别

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个完全无人工标注的单目3D检测系统，LOMM和SCC都有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 三个大规模数据集验证，消融充分，预训练和跨数据集实验说服力强
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详尽，流水线各步骤逻辑连贯
- 价值: ⭐⭐⭐⭐⭐ 对自动驾驶数据标注成本问题有重大实际意义，700倍速度提升使大规模应用可行

<!-- RELATED:START -->

## 相关论文

- [Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d.md)
- [Monocular Occupancy Prediction for Scalable Indoor Scenes](../../ECCV2024/autonomous_driving/monocular_occupancy_prediction_for_scalable_indoor_scenes.md)
- [LabelAny3D: Label Any Object 3D in the Wild](../../NeurIPS2025/autonomous_driving/labelany3d_label_any_object_3d_in_the_wild.md)
- [MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection](../../ECCV2024/autonomous_driving/monowad_weather-adaptive_diffusion_model_for_robust_monocular_3d_object_detectio.md)
- [Difficulty-Aware Label-Guided Denoising for Monocular 3D Object Detection](../../AAAI2026/autonomous_driving/difficulty-aware_label-guided_denoising_for_monocular_3d_object_detection.md)

<!-- RELATED:END -->
