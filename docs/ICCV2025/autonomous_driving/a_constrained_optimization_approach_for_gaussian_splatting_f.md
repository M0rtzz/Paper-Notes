---
title: >-
  [论文解读] A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds
description: >-
  [ICCV 2025][自动驾驶][3D Gaussian Splatting] 提出一种无需SfM的约束优化方法，通过相机位姿分解、灵敏度预调节、对数障碍约束和几何约束，从多相机SLAM系统输出的粗糙位姿和噪声点云中联合优化相机参数与3DGS场景重建。
tags:
  - ICCV 2025
  - 自动驾驶
  - 3D Gaussian Splatting
  - 相机位姿优化
  - 约束优化
  - 多相机SLAM
  - Lidar点云
---

# A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds

**会议**: ICCV 2025  
**arXiv**: [2504.09129](https://arxiv.org/abs/2504.09129)  
**代码**: 无（计划发布数据集）  
**领域**: 自动驾驶 / 3D重建 / 3D Gaussian Splatting  
**关键词**: 3D Gaussian Splatting, 相机位姿优化, 约束优化, 多相机SLAM, Lidar点云

## 一句话总结

提出一种无需SfM的约束优化方法，通过相机位姿分解、灵敏度预调节、对数障碍约束和几何约束，从多相机SLAM系统输出的粗糙位姿和噪声点云中联合优化相机参数与3DGS场景重建。

## 研究背景与动机

3D Gaussian Splatting (3DGS) 是一种高效的3D重建技术，但它严重依赖精确的相机位姿和高质量稀疏点云作为初始化，通常来自耗时的Structure-from-Motion (SfM)算法（如COLMAP）。这限制了3DGS在实际场景和大规模重建中的应用。

在实际的机器人/AR/VR场景中，多相机SLAM系统可以快速获取相机位姿和点云，但存在以下问题：

**位姿噪声**：SLAM输出的设备位姿因传感器噪声和Lidar里程计漂移而不准确

**时间异步**：RGB图像与设备位姿采集存在时间偏移（可达50ms），导致估计位姿与真实位姿偏差

**标定误差**：相机内参和Lidar-相机外参标定不完美引入额外误差

**COLMAP耗时**：COLMAP处理需4-12小时，且在重复纹理场景可能失败

直接使用这些噪声输入会导致模糊的重建和退化的几何。本文的目标是在不依赖SfM的前提下，从多相机SLAM系统的不精确输入中实现高质量3DGS重建。

## 方法详解

### 整体框架

方法的核心是将相机位姿优化问题分解并施加多种约束。给定N张RGB图像、粗糙的相机位姿和噪声点云，联合优化相机内外参和3DGS场景表示。整体框架包含：位姿分解 → 灵敏度预调节 → 对数障碍约束 → 几何约束（极线+重投影）→ 测试时适应。

### 关键设计

#### 1. 相机位姿分解（Camera Pose Decomposition）

将每个相机位姿分解为两个变换的组合：

$$\mathcal{P}^{(j,t)} = \hat{\mathcal{P}}^t \times \mathcal{E}^j$$

其中 $\hat{\mathcal{P}}^t$ 是第 $t$ 时刻设备到世界的位姿，$\mathcal{E}^j$ 是第 $j$ 个相机到设备的外参。

- 对于4相机、10k图像的系统，原始自由度为60k
- 分解后只需优化2500个独立设备位姿 + 4个共享外参 = 15024个自由度
- 通过学习小偏移量 $\vec{\phi}^t$ 和 $\vec{\rho}^j$（各6维，3旋转+3平移）来修正位姿

关键选择：采用**右乘**误差矩阵（$f(\hat{\mathcal{P}}^t, \vec{\phi}^t) = \hat{\mathcal{P}}^t \times \Phi^t$）而非左乘。左乘会使相机位置围绕世界原点旋转（通常远离初始值），导致优化不稳定。右乘则围绕初始相机位置局部旋转，更加稳定。

#### 2. 相机内参优化（Intrinsic Refinement）

现有方法通常假设内参已知，但在多相机系统中，内参误差无法像单相机那样通过外参偏移补偿（因为多相机共享设备位姿，调整一个相机的外参会影响所有相机）。通过修改3DGS光栅化器，推导解析梯度实现焦距和主点的端到端优化：
- $\partial u/\partial f_x = \vec{u}^x_{\text{cam}} / \vec{u}^z_{\text{cam}}$，$\partial u/\partial c_x = 1$

#### 3. 灵敏度预调节（Sensitivity-based Pre-conditioning）

受Levenberg-Marquardt算法启发，计算投影函数对各参数的Jacobian矩阵 $\mathcal{J}$，根据 $(\mathcal{J}^\top \mathcal{J})^{-1/2}$ 的对角元素比值（即Hessian逆平方根的近似）自适应调整各参数组的学习率。即使1%的参数变化也可能导致截然不同的渲染结果，因此不同参数需要不同步长。

#### 4. 对数障碍约束（Log-barrier Constraint）

为防止敏感参数超出可行域，引入对数障碍函数：

$$\mathcal{L}_{\text{barrier}} = \frac{1}{\mathcal{T}} \sum_{i=1}^m \log(-h_i(x))$$

- 温度 $\mathcal{T}$ 从小到大递增：初期强约束（接近边界时梯度 $-1/(\tau h_i(x))$ 极大），后期放松允许充分探索
- 内参约束：焦距和主点偏差不超过 ±2%
- 外参约束：设备位姿旋转 ±0.625°、平移 ±0.125m；相机外参旋转 ±2.5°、平移 ±0.5m

#### 5. 几何约束（Geometric Constraints）

使用LoFTR关键点匹配获取相邻帧半稠密匹配点对（每对数百个），提出两种互补约束：

**软极线约束**：通过基础矩阵 $\mathbb{F}$ 计算Sampson距离，约束相对位姿满足极线几何。不考虑深度但提供强先验。

**重投影误差正则化**：将传统SfM的Bundle Adjustment扩展为几何正则，利用匹配点对和深度双向投影。深度通过射线交点（line intersection）精确计算，而非不稳定的alpha-blending方式。

### 损失函数 / 训练策略

总损失函数：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pixel}} + 0.2 \cdot \mathcal{L}_{\text{ssim}} + 0.1 \cdot \mathcal{L}_{\text{barrier}} + 10^{-3} \cdot \mathcal{L}_{\text{epipolar}} + 5 \times 10^{-4} \cdot \mathcal{L}_{\text{reproj}}$$

- 内参学习率 $8\times10^{-4}$，外参基础学习率 $5\times10^{-3}$（按Jacobian调整）
- 余弦学习率衰减 + 3次重启（在1st、max_iter/6、max_iter/2迭代处）
- 训练48k迭代，禁用Gaussian pruning，67%训练量后启用densification
- **测试时适应（TTA）**：冻结3DGS，以 $5\times10^{-4}$ 学习率优化500步位姿 + 曝光补偿（YCbCr空间低频亮度偏移）

## 实验关键数据

### 主实验

**自采数据集**：4个场景（Cafeteria/Office/Laboratory/Town），使用4鱼眼相机+IMU+Lidar的自研设备。

| 方法 | 预处理时间 | Cafeteria PSNR/SSIM | Office PSNR/SSIM | Lab PSNR/SSIM | Town PSNR/SSIM |
|------|-----------|---------------------|-------------------|---------------|----------------|
| Direct reconst. | 3 min | 19.23/0.789 | 17.49/0.758 | 18.35/0.798 | 16.12/0.615 |
| Pose optimize | 5 min | 26.89/0.872 | 23.96/0.837 | 26.11/0.867 | 20.18/0.685 |
| 3DGS-COLMAP | 4-12 hrs | 17.03/0.768 | 25.82/0.883 | 28.30/0.908 | 24.07/0.830 |
| 3DGS-COLMAP△ | 2-3 hrs | 26.51/0.838 | 23.91/0.839 | 23.76/0.816 | 23.51/0.809 |
| CF-3DGS | 1 min | 15.44/0.541 | 16.53/0.756 | 16.44/0.756 | 15.45/0.541 |
| MonoGS | 1 min | 8.27/0.468 | 9.56/0.496 | 13.08/0.601 | 12.74/0.309 |
| InstantSplat | 50 min | 19.86/0.774 | 23.30/0.872 | 20.89/0.862 | 21.48/0.738 |
| **Ours** | **5 min** | **29.05/0.917** | **26.07/0.885** | **28.64/0.910** | **24.52/0.826** |

**公开数据集对比（多模态方法）**：

| 方法 | GarageWorld G0 PSNR | GarageWorld G6 PSNR | Waymo S002 PSNR | Waymo S031 PSNR |
|------|-------|-------|-------|-------|
| 3DGS | 25.43 | 21.23 | 25.84 | 24.42 |
| LetsGo | 25.29 | 21.72 | 26.11 | 24.79 |
| Street-GS | 24.20 | 20.52 | 27.96 | 25.04 |
| **Ours** | **26.06** | **23.76** | **29.75** | **28.48** |

### 消融实验

**相机分解 + 预调节策略**（CVG%为SSIM达95%峰值时的训练阶段，越小收敛越快）：

| C.D. | P.C. | Cafeteria PSNR/SSIM | CVG% | Lab PSNR/SSIM | CVG% |
|------|------|---------------------|------|---------------|------|
| ✗ | ✗ | 26.91/0.866 | 34.38 | 27.00/0.881 | 31.25 |
| ✗ | ✓ | 26.45/0.858 | 22.92 | 26.07/0.865 | 18.76 |
| ✓ | ✗ | 28.87/0.915 | 43.10 | 28.52/0.909 | 39.58 |
| ✓ | ✓ | **29.05/0.917** | **15.65** | **28.64/0.910** | **16.67** |

**几何约束消融（不同噪声水平）**：

| 噪声 | E.P. | R.P. | PSNR/SSIM | Ep-e↓ | RP-e↓ |
|------|------|------|-----------|-------|-------|
| - | ✗ | ✗ | 27.05/0.895 | 1.14 | 2.52 |
| - | ✓ | ✓ | 27.31/0.915 | 1.08 | 1.88 |
| 0.2° | ✗ | ✗ | 26.04/0.890 | 1.23 | 2.56 |
| 0.2° | ✓ | ✓ | 26.84/0.905 | 1.11 | 2.00 |
| 0.5° | ✗ | ✗ | 24.80/0.858 | 1.72 | 3.92 |
| 0.5° | ✓ | ✓ | 25.20/0.867 | 1.21 | 2.32 |

**其他关键消融**：
- 内参优化：PSNR从27.40提升至29.05（Cafeteria），文字等细节明显清晰
- 对数障碍：仅约束±2%即提升SSIM 6.8%
- 相机数量：1/2/4相机下改进分别为 +2.30/+2.24/+3.07 dB PSNR
- TTA消融：位姿优化+曝光补偿联合使用效果最佳（Cafeteria 28.58→仅位姿23.04，仅曝光22.65，不用19.80）

### 关键发现

1. **COLMAP并非万能**：在Cafeteria场景因重复纹理导致失败（PSNR仅17.03），而本方法达29.05
2. **预处理时间优势巨大**：本方法仅需5分钟，COLMAP需4-12小时
3. **相机分解贡献最大**：PSNR提升约2dB，且结合预调节后CVG%从43%降至16%，收敛更稳定
4. **几何约束在噪声越大时改进越明显**（0.5°噪声下极线误差从1.72降至1.21像素）
5. 增量式SLAM方法（CF-3DGS、MonoGS）在低共视性场景严重退化（SSIM 0.3-0.75）

## 亮点与洞察

1. **问题定义精准**：首个系统解决多相机SLAM输出用于3DGS的工作，从分析到约束设计环环相扣
2. **分解思想优雅**：将60k自由度降至15k，共享外参提供全局约束
3. **约束设计有理有据**：灵敏度预调节源自LM算法，对数障碍源自凸优化理论，几何约束源自SfM
4. **右乘vs左乘的细节**：看似小的技术选择实际决定优化稳定性
5. **曝光补偿模块**：在YCbCr空间仅修改亮度通道的低频分量，用tinycudann实现，设计简洁高效
6. **深度计算方式**：用射线交点精确求解而非不稳定的alpha-blending

## 局限与展望

1. 仅处理静态场景，依赖YOLOv8检测并排除行人区域
2. 对数障碍的边界（±2%等）为经验设定，不同场景可能需调整
3. 关键点匹配质量依赖LoFTR性能
4. 数据集仅4个场景，泛化能力有待更大规模验证
5. 需要预先做鱼眼去畸变，畸变参数的联合优化值得探索
6. 未考虑场景语义信息辅助优化

## 相关工作与启发

- **InstantSplat**：用3D foundation model提供相对位姿，但GPU内存限制最多30张图
- **CF-3DGS/MonoGS**：增量式SLAM+3DGS，在低共视性时严重退化
- **BARF**：粗到细位置编码位姿联合优化，但限于NeRF
- **Street-GS**：自动驾驶多模态方法，独立优化各相机位姿，无内参优化
- **LetsGo**：同样利用Lidar+相机，但假设精确位姿

约束优化框架可推广到其他需要从噪声初始化恢复的3D任务。

## 评分

- **创新性**：⭐⭐⭐⭐ — 将约束优化理论系统性引入3DGS位姿优化，位姿分解思想优雅
- **实用性**：⭐⭐⭐⭐⭐ — 5分钟替代12小时COLMAP，工业部署价值极高
- **实验**：⭐⭐⭐⭐ — 自建数据集+2个公开基准，消融充分，GarageWorld噪声鲁棒性实验设计合理
- **写作**：⭐⭐⭐⭐ — 动机充分，公式推导完整，补充材料详尽
- **综合**：8.5/10

<!-- RELATED:START -->

## 相关论文

- [GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multiview_robust_physical_adver.md)
- [IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation](igl-nav_incremental_3d_gaussian_localization_for_image-goal_navigation.md)
- [Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [Counting Stacked Objects](counting_stacked_objects.md)

<!-- RELATED:END -->
