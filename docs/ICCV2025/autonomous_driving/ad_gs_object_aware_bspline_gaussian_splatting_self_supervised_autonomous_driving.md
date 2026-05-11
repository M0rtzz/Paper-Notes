---
title: >-
  [论文解读] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving
description: >-
  [ICCV 2025][自动驾驶][自动驾驶场景渲染] 本文提出 AD-GS，一种自监督的自动驾驶场景渲染框架，通过结合局部感知的可学习 B 样条曲线和全局感知的三角函数来建模动态物体运动，并利用简化的伪 2D 分割进行场景分解，在不依赖人工 3D 标注的情况下显著超越现有自监督方法，接近有标注方法的性能。
tags:
  - ICCV 2025
  - 自动驾驶
  - 自动驾驶场景渲染
  - 高斯溅射
  - B样条曲线
  - 自监督
  - 动态场景重建
---

# AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving

**会议**: ICCV 2025  
**arXiv**: 无 (CVF Open Access)  
**代码**: [https://jiaweixu8.github.io/AD-GS-web/](https://jiaweixu8.github.io/AD-GS-web/)  
**领域**: 自动驾驶  
**关键词**: 自动驾驶场景渲染, 高斯溅射, B样条曲线, 自监督, 动态场景重建

## 一句话总结

本文提出 AD-GS，一种自监督的自动驾驶场景渲染框架，通过结合局部感知的可学习 B 样条曲线和全局感知的三角函数来建模动态物体运动，并利用简化的伪 2D 分割进行场景分解，在不依赖人工 3D 标注的情况下显著超越现有自监督方法，接近有标注方法的性能。

## 研究背景与动机

1. **领域现状**：自动驾驶场景渲染旨在从 LiDAR 点云和多相机图像重建动态驾驶环境，支持新视角和新时刻的渲染。主流方法分两类：(1) 依赖人工 3D 标注（物体边界框和轨迹），如 4DGF、StreetGS 等，质量高但标注成本极高；(2) 自监督方法，不需标注但性能明显落后。

2. **现有痛点**：自监督方法在运动建模和场景分解两方面存在不足。运动建模方面，神经网络方法计算开销大且难以捕捉局部运动细节；三角函数方法速度快但全局拟合特性使其难以表达局部运动变化。场景分解方面，使用实例分割或特征监督的方法在噪声伪标签下容易产生重建伪影。

3. **核心矛盾**：自监督方法需要在**全局运动拟合的稳定性**和**局部运动细节的精确性**之间取得平衡。全局三角函数在噪声监督下稳定但不够精确，局部拟合方法精确但容易过拟合噪声。同时，场景分解的粒度也需要在鲁棒性和精确性之间权衡。

4. **本文目标**：设计一种兼具全局稳定性和局部精确性的运动表示方法，并提出鲁棒的场景分解策略，使自监督方法在不依赖 3D 标注的情况下接近有标注方法的渲染质量。

5. **切入角度**：B 样条曲线天然具有局部控制特性（每个曲线上的点只受附近控制点影响），而三角函数具有全局拟合特性。将两者结合可以同时实现局部细节拟合和全局运动捕获。场景分解只需简单的"物体/背景"二分类，避免细粒度分割的噪声问题。

6. **核心 idea**：用 B 样条位置曲线 + B 样条四元数曲线 + 三角函数共同建模动态高斯体的运动，并引入双向时间可见性掩码处理物体的突然出现和消失。

## 方法详解

### 整体框架

AD-GS 基于 3D Gaussian Splatting，将场景分解为物体和背景两部分。背景高斯保持静态，物体高斯通过 B 样条曲线和三角函数进行时间相关的位置和旋转变形。输入为 LiDAR 点云和多视角图像序列，通过 SAM 生成简化的物体/背景二值分割作为伪标签。使用光流、单目深度、分割和天空遮罩等多种伪监督信号进行自监督训练。

### 关键设计

1. **可学习 B 样条运动曲线**:

    - 功能：精确建模动态物体的局部运动细节。
    - 核心思路：使用均匀 B 样条曲线表示高斯体位置的时间变形：$\mu' = \mu + p(t) + \sum_{l=1}^L a_l \sin(t \cdot l\vartheta) + b_l \cos(t \cdot l\vartheta)$，其中 $p(t)$ 为 B 样条曲线。B 样条的关键特性是局部控制——每个时间点的曲线值只由附近 $k$ 个控制点决定，当某帧数据有噪声时只影响局部而非全局。使用矩阵形式 $p(t) = [1, u, u^2, ..., u^{k-1}] M_k [p_{i-k+1}, ..., p_i]^T$ 替代递归计算以提高效率。旋转则用 B 样条四元数曲线 $q(t)$ 直接建模。
    - 设计动机：三角函数的全局优化特性使其在噪声自监督下稳定，但难以捕捉局部运动变化（如急刹车、转弯）。B 样条的局部拟合特性恰好互补。

2. **简化伪 2D 分割的场景分解**:

    - 功能：将场景鲁棒地分解为物体和背景两部分。
    - 核心思路：不使用复杂的实例分割，而是将所有类别简化为"物体"（车辆等可能运动的类别）和"背景"两类。使用 SAM 生成二值分割 $M_{obj}$，通过 LiDAR 投影初始化高斯分类。训练时通过 $\hat{M}_{obj} = \sum_i \mathbb{I}\{G_i \in \Omega_{obj}\} \alpha_i T_i$ 渲染物体掩码，用 BCE 损失 $L_{obj}$ 使两类高斯保持在各自区域。
    - 设计动机：细粒度实例分割（如每个车辆单独分割）在噪声伪标签下过于脆弱，简化为二分类大大提高了鲁棒性。

3. **双向时间可见性掩码**:

    - 功能：处理动态物体在序列中的突然出现和消失。
    - 核心思路：对每个物体高斯 $G \in \Omega_{obj}$ 的不透明度施加时间衰减：$\omega'(t) = \omega \cdot e^{-(t-\mu_t)^2 / 2s^2}$，其中 $\mu_t$ 固定为 LiDAR 采集时间戳（作为物体"何时可见"的先验），$s_0$（$t < \mu_t$）和 $s_1$（$t \geq \mu_t$）为可学习的前后方向缩放参数。引入扩展正则化 $L_s = \|2\bar{f} / (s_0 + s_1)\|_1$ 防止掩码过窄。
    - 设计动机：驾驶场景中车辆会突然进入和离开视野。不加时间掩码，已离开的车辆的高斯仍会在后续帧中产生"幽灵"伪影。

### 损失函数 / 训练策略

总损失：$L = (1-\varsigma_c)L_1 + \varsigma_c L_{D-SSIM} + \varsigma_d L_d + \varsigma_f L_f + \varsigma_{obj} L_{obj} + \varsigma_{sky} L_{sky} + \varsigma_r L_r + \varsigma_s L_s$

- 图像重建损失：L1 + D-SSIM
- 逆深度监督 $L_d$：使用 DPTv2 生成伪标签，scale-and-shift 对齐
- 光流监督 $L_f$：使用 CoTracker3 生成伪标签，仅对物体区域监督
- 物理刚体正则化 $L_r$：KNN 邻域内高斯变形参数方差约束

## 实验关键数据

### 主实验

| 数据集 | 方法 | 标注 | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|------|-------|-------|--------|
| KITTI-75% | 4DGF | 有 | 31.34 | 0.945 | 0.026 |
| KITTI-75% | PVG | 无 | 27.13 | 0.895 | 0.049 |
| KITTI-75% | **AD-GS** | **无** | **29.16** | **0.920** | **0.033** |
| Waymo | StreetGS | 有 | 33.97 | 0.926 | 0.227 |
| Waymo | EmerNeRF | 无 | 31.32 | 0.881 | 0.301 |
| Waymo | **AD-GS** | **无** | **33.91** | **0.927** | **0.228** |
| nuScenes | Grid4D | 无 | 30.29 | 0.920 | 0.172 |
| nuScenes | **AD-GS** | **无** | **31.06** | **0.925** | **0.164** |

### 消融实验

| 配置 | PSNR↑(动态) | PSNR↑(全场景) | 说明 |
|------|------------|-------------|------|
| sin&cos only | 24.28 | 32.61 | 仅三角函数 |
| + B-spline | 26.65 | 33.65 | 加入B样条 +2.37 |
| + t-mask (full) | **27.41** | **33.91** | 加入时间掩码 +0.76 |

| 损失配置 | PSNR↑ | 说明 |
|---------|-------|------|
| 基础 | 26.52 | 仅图像损失 |
| + obj&sky | 26.98 | +分割监督 +0.46 |
| + flow&depth | 28.03 | +运动和3D信息 +1.05 |
| + reg (full) | **29.16** | +正则化 +1.13 |

### 关键发现

- AD-GS 在自监督设置下显著超越所有自监督方法，在 Waymo 上甚至接近有标注的 StreetGS（33.91 vs 33.97 PSNR）
- B 样条曲线对动态物体的 PSNR 贡献最大（+2.37），证明了局部拟合的重要性
- 物理刚体正则化对防止混乱行为至关重要（+1.13 PSNR）
- 在极稀疏视角下（KITTI-25%），自监督方法仍与有标注方法有较大差距

## 亮点与洞察

- **B 样条 + 三角函数的互补组合**：B 样条提供局部精确拟合，三角函数提供全局稳定性，两者结合完美平衡了自监督场景下噪声鲁棒性和运动精度的矛盾。这种局部-全局组合思路可迁移到其他时间序列拟合任务。
- **简化分割的智慧**：不追求精细分割而用粗粒度二分类，在噪声伪标签环境下反而更鲁棒。这是一个"少即是多"的设计哲学。
- **LiDAR 时间戳作为可见性先验**：巧妙地利用已有信息（LiDAR 采集时刻）作为物体出现时间的锚点，避免了学习时间位置的不稳定性。

## 局限与展望

- 在极稀疏视角下性能仍然受限，自监督运动拟合在缺少足够约束时不够准确
- 行人等非刚体运动物体未被特别建模
- 场景分解为二分类可能在复杂交互场景中不够精细

## 相关工作与启发

- **vs PVG**：PVG 仅用三角函数建模运动，缺乏局部拟合能力；AD-GS 通过 B 样条补充局部细节
- **vs EmerNeRF**：EmerNeRF 用检测模型特征监督分解，AD-GS 用更简单的二分类分割但效果更好
- **vs 4DGF**：4DGF 依赖人工 3D 标注，AD-GS 在无标注下接近其性能

## 评分

- 新颖性: ⭐⭐⭐⭐ B样条+三角函数运动建模方案新颖且有效
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集 + 与有标注/无标注方法全面对比 + 详细消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，公式完整
- 价值: ⭐⭐⭐⭐⭐ 自监督方法接近有标注方法性能，对降低自动驾驶仿真成本有重大意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving](coda-4dgs_dynamic_gaussian_splatting_with_context_and_deformation_awareness_for_.md)
- [\[ICCV 2025\] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)

</div>

<!-- RELATED:END -->
