---
title: >-
  [论文解读] Simultaneous Motion And Noise Estimation with Event Cameras
description: >-
  [ICCV 2025][视频理解][事件相机] 首次提出事件相机运动估计与噪声估计的联合方法，利用对比度最大化（CMax）框架中运动补偿后的局部对比度对每个事件评分，通过交替优化同时获得运动参数和信号/噪声分类，在 E-MLB 去噪基准上达到 SOTA。
tags:
  - ICCV 2025
  - 视频理解
  - 事件相机
  - 去噪
  - 运动估计
  - 对比度最大化
  - 联合估计
---

# Simultaneous Motion And Noise Estimation with Event Cameras

**会议**: ICCV 2025  
**arXiv**: [2504.04029](https://arxiv.org/abs/2504.04029)  
**代码**: [GitHub](https://github.com/tub-rip/ESMD)  
**领域**: 视频理解  
**关键词**: 事件相机, 去噪, 运动估计, 对比度最大化, 联合估计

## 一句话总结

首次提出事件相机运动估计与噪声估计的联合方法，利用对比度最大化（CMax）框架中运动补偿后的局部对比度对每个事件评分，通过交替优化同时获得运动参数和信号/噪声分类，在 E-MLB 去噪基准上达到 SOTA。

## 研究背景与动机

事件相机是新型视觉传感器，能克服传统相机的运动模糊、有限动态范围等缺陷，但由于其工作在低功耗（亚阈值）条件下，会产生大量噪声（尤其是背景活动 BA 噪声）。

**现有方法的关键问题**：

**去噪与运动估计割裂**：现有去噪方法通常独立设计，将运动估计作为单独的后续任务。然而，**运动是事件数据的本质属性**——没有运动就无法感知场景边缘。二者应该是协同的。

**Ground Truth 获取困难**：学习方法需要 GT 噪声标签，但在真实数据中无法定义。现有方案要么依赖仿真，要么通过激进的预过滤来获取"纯信号"数据，但这种处理可能改变事件的信号/噪声特性。

**循环依赖问题**：去噪需要知道真实运动（将信号事件与噪声分开），而准确的运动估计又需要信号事件（因为噪声不携带运动信息）。

**核心洞察**：运动信息可以帮助去噪（反之亦然）。不如将二者整合到一个统一框架中同时求解。本文是**首个同时估计运动（自运动、光流等多种形式）和噪声**的方法。

## 方法详解

### 整体框架

基于对比度最大化（CMax）框架的迭代交替优化：

1. 用当前信号事件估计运动（CMax 的一步优化）
2. 用估计的运动对所有事件进行运动补偿
3. 根据补偿后的局部对比度对每个事件评分
4. 按评分排序+阈值化分类为信号/噪声
5. 更新信号事件集，重复步骤 1

### 关键设计

1. **对比度最大化（CMax）基础**

   CMax 框架假设事件由移动边缘产生，根据运动模型 $\mathbf{W}$ 变换事件坐标，将事件集 $\mathcal{E} = \{e_k\}_{k=1}^{N_e}$ 变换到参考时刻：

   $$e_k = (\mathbf{x}_k, t_k, p_k) \mapsto e'_k = (\mathbf{x}'_k, t_{ref}, p_k)$$

   变换后的事件在像素网格上聚合生成 warped event image (IWE)：

   $$I(\mathbf{x}; \boldsymbol{\theta}) = \sum_{k=1}^{N_e} \delta(\mathbf{x} - \mathbf{x}'_k)$$

   其中 Dirac delta 用高斯近似。优化目标是最大化 IWE 的对比度（图像方差），从而找到使事件最大程度对齐的运动参数。

   本文支持两种运动模型：**旋转运动**（3-DOF 角速度估计）和**稠密光流**（逐像素速度估计，$2N_p$ DOF）。

2. **基于局部对比度的去噪**

   **核心思想**：信号事件经过正确运动补偿后会聚集在边缘位置，产生高 IWE 值；噪声事件因随机分布不会聚集，IWE 值低。

   对每个事件 $e_k$ 计算得分 $c_k$：

   $$c_k = I(\mathbf{x}'_k)$$

   即该事件在运动补偿后的 IWE 中的局部值（局部对比度）。IWE 值越高，说明更多事件支持同一场景边缘，该事件越可能是信号。

   按得分排序后阈值化分类：

   $$\mathcal{E}_{signal} = \{e_k \in \mathcal{E} \mid c_k > T(\eta)\}$$
   $$\mathcal{E}_{noise} = \mathcal{E} \setminus \mathcal{E}_{signal}$$

   其中 $\tau = 1 - \eta$ 为信号事件比例，$\eta$ 为噪声比例（先验或估计值）。

   **不变性**：分类结果对 $c_k$ 的单调递增变换不变（如取对数、指数等），因为排序保持不变。

   **对不同边缘强度的鲁棒性**：IWE 中的高斯核控制对边缘强度的敏感度，增大高斯核可以更好保留低 IWE 强度区域的信号事件。

3. **交替优化**

   信号/噪声分类和运动估计构成循环依赖——分类需要真实运动，运动估计需要信号事件。

   解决方案是**迭代交替优化**：
   - 初始化：随机划分事件为信号和噪声集
   - 每次迭代：①用当前信号事件做 CMax 运动估计（1步即可）→ ②用估计运动对所有事件做 warping → ③计算所有事件的评分 $c_k$ → ④重新划分信号/噪声集
   - 收敛判据：运动参数收敛

   **计算复杂度**：单次迭代 $O(N_p + N_e \log N_e)$，仅比原始 CMax 的 $O(N_p + N_e)$ 多了一个排序的 $\log$ 因子。

   **灵活性**：运动估计步骤中的 CMax 可被替换为任何其他运动估计器（包括深度神经网络），方法具有良好的可扩展性。

### 损失函数 / 训练策略

本方法为**无监督方法**，不需要训练损失。运动估计的目标函数是 IWE 的方差（对比度）：

$$\text{Var}(I(\mathbf{x}; \boldsymbol{\theta})) = \frac{1}{|\Omega|}\int_{\Omega}(I(\mathbf{x}; \boldsymbol{\theta}) - \mu_I)^2 d\mathbf{x}$$

通过最大化该目标来找到最优运动参数 $\boldsymbol{\theta}^*$。整个过程为优化而非学习，无需GT标签。

## 实验关键数据

### 主实验

E-MLB 去噪基准（MESR↑，值越高越好）：

| 方法 | 类别 | Day ND1 | Day ND4 | Day ND16 | Day ND64 | Night ND1 |
|------|------|---------|---------|----------|----------|-----------|
| BAF | 模型 | 0.861 | 0.869 | 0.876 | 0.890 | 0.946 |
| IETS | 模型 | 0.772 | 0.785 | 0.777 | 0.753 | 0.950 |
| MLPF | 学习 | 0.851 | 0.855 | 0.846 | 0.840 | 0.926 |
| EDformer | 学习 | 0.952 | 0.955 | 0.956 | 0.942 | 1.048 |
| **Ours** | **模型** | **0.938** | **0.958** | **0.986** | **0.950** | **1.037** |

在模型基方法中排名第一或第二，部分条件下甚至超越需要 GT 训练的学习方法。

DND21 去噪基准（AUC↑）：

| 方法 | hotel 1Hz | driving 1Hz | hotel 5Hz | driving 5Hz |
|------|-----------|-------------|-----------|-------------|
| BAF | 0.9535 | 0.8479 | 0.8916 | 0.7930 |
| TS | 0.9716 | 0.9307 | 0.9606 | 0.9270 |
| EDformer | 0.9928 | 0.9541 | 0.9845 | 0.9424 |
| **Ours** | **1.014** | **0.882** | **0.963** | **0.855** |

### 消融实验

运动估计改善（ECD 数据集，旋转运动）：

| 配置 | 方法 | 效果 | 说明 |
|------|------|------|------|
| 无去噪 | CMax 原始 | 依赖初始值 | 容易陷入局部最优 |
| BAF 预处理 | CMax + BAF | 部分改善 | 简单滤波不够 |
| **联合估计** | **Ours** | **显著改善** | 降低对初始值的依赖 |

光流估计（MVSEC 数据集）+ 去噪组合：

| 配置 | 说明 |
|------|------|
| 基于深度学习的运动估计器 | 本方法可将 CMax 替换为 DNN 进行联合估计 |
| 图像重建质量 | 去噪后事件重建的图像质量显著优于原始事件 |

### 关键发现

- **去噪改善运动估计**：联合方法降低了 CMax 框架对初始值的依赖，使旋转运动估计更鲁棒。
- **运动改善去噪**：正确的运动补偿使信号事件聚集度更高，噪声分类更准确。
- **无监督超越有监督**：在 E-MLB 的多个条件下，无监督方法超越了依赖 GT 标签训练的学习方法。
- **方法灵活性**：可与深度学习运动估计器结合，不限于 CMax 框架。
- **实际应用**：去噪后的事件用于图像强度重建，产生更少伪影和更高质量的图像。

## 亮点与洞察

- **第一性原理驱动**：从"噪声与运动不相关"的物理原理出发，将两个看似独立的问题统一为联合估计。
- **无监督、无需 GT 标签**：突破了学习方法对标注数据的依赖，在真实场景中更具实用性。
- **理论优雅**：基于 IWE 的评分-排序-阈值化流程简洁清晰，分类结果对评分的单调变换不变。
- **计算高效**：仅比原始 CMax 多一个排序操作的计算开销。
- **开源**：提供了完整的开源实现。

## 局限与展望

- 需要先验知识或估计噪声比例 $\eta$，不同场景下最优 $\eta$ 可能不同。
- 不处理闪烁或主动光源导致的非 BA 噪声。
- 交替优化可能收敛到局部最优，尤其在高噪声率场景下。
- DND21 数据集中"纯信号"通过激进过滤获得，可能与真实信号分布有偏差，导致评估值偏低。

## 相关工作与启发

- CMax 框架在事件相机领域广泛应用，本文展示了其在去噪中的新用途。
- "同时估计"的思想可推广到其他传感器融合问题（如 LiDAR 去噪+运动估计）。
- 局部对比度作为信号指标的思路可启发其他基于事件的任务（如目标检测的置信度估计）。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次联合运动+噪声估计，第一性原理驱动
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证+多任务应用展示
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，物理直觉强
- 价值: ⭐⭐⭐⭐ 开辟了事件相机研究的新方向

<!-- RELATED:START -->

## 相关论文

- [Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)
- [egoPPG: Heart Rate Estimation from Eye-Tracking Cameras in Egocentric Systems to Benefit Downstream Vision Tasks](egoppg_heart_rate_estimation_from_eye-tracking_cameras_in_egocentric_systems_to_.md)
- [EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](../../CVPR2025/video_understanding/edcflow_exploring_temporally_dense_difference_maps_for_event-based_optical_flow_.md)
- [Motion-prior Contrast Maximization for Dense Continuous-Time Motion Estimation](../../ECCV2024/video_understanding/motion-prior_contrast_maximization_for_dense_continuous-time_motion_estimation.md)

<!-- RELATED:END -->
