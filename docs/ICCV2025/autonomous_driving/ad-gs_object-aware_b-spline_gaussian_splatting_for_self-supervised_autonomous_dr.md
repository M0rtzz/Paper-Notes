---
title: >-
  [论文解读] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving
description: >-
  [ICCV 2025][自动驾驶][Gaussian Splatting] 本文提出 AD-GS，一种基于 3D Gaussian Splatting 的自监督自动驾驶场景渲染框架，核心创新是将可学习 B-spline 曲线与三角函数结合进行局部-全局运动建模，并通过简化的二值伪分割实现鲁棒的场景分解，在不依赖人工 3D 标注的条件下大幅超越现有自监督方法。
tags:
  - ICCV 2025
  - 自动驾驶
  - Gaussian Splatting
  - 自监督学习
  - B-spline运动建模
  - 动态场景渲染
---

# AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving

**会议**: ICCV 2025  
**arXiv**: [2507.12137](https://arxiv.org/abs/2507.12137)  
**代码**: [Project Page](https://jiaweixu8.github.io/AD-GS-web/)  
**领域**: 自动驾驶场景重建  
**关键词**: Gaussian Splatting, 自动驾驶, 自监督学习, B-spline运动建模, 动态场景渲染

## 一句话总结
本文提出 AD-GS，一种基于 3D Gaussian Splatting 的自监督自动驾驶场景渲染框架，核心创新是将可学习 B-spline 曲线与三角函数结合进行局部-全局运动建模，并通过简化的二值伪分割实现鲁棒的场景分解，在不依赖人工 3D 标注的条件下大幅超越现有自监督方法。

## 研究背景与动机
自动驾驶场景的高质量渲染和重建对于自驾仿真至关重要。现有高质量方法（如 4DGF、StreetGS、ML-NSG）依赖昂贵的人工 3D 标注（目标物体的边界框和位姿），虽然效果好但标注成本高，限制了大规模应用。

自监督方法试图仅从图像和 LiDAR 数据中重建动态场景，但面临两大技术挑战。第一，运动建模的局部-全局困境：使用神经网络（MLP）建模运动计算开销大且难以捕获局部运动细节；使用三角函数建模（如 PVG）速度快、全局拟合性好（因为所有参数对每帧都参与优化），但局部运动细节不足。第二，场景分解的噪声问题：使用实例分割的伪标签进行场景分解会引入大量噪声，导致重建伪影。

核心矛盾是：如何在自监督（有噪声伪标签）条件下，同时实现精确的运动建模和鲁棒的场景分解？本文的切入角度是引入 B-spline 曲线——它的局部控制性使得每个时刻的位置只受附近控制点影响，能精确拟合局部运动细节，同时与三角函数结合保持全局平滑性。场景分解方面，将分割简化为仅两类（物体 vs 背景），大幅降低伪标签噪声的影响。

核心 idea：B-spline 的局部拟合 + 三角函数的全局拟合 = 精确且鲁棒的运动建模。

## 方法详解

### 整体框架
AD-GS 将场景分解为物体高斯 $\boldsymbol{\Omega}_{obj}$ 和背景高斯 $\boldsymbol{\Omega}_{bkg}$。背景高斯保持静止，仅颜色随时间变化。物体高斯通过 B-spline + 三角函数进行位置和旋转变形，并附带双向时间可见性掩模处理物体的突然出现/消失。远景区域（天空等）用可学习球面环境图表示。整体通过图像重建、光流、深度、分割的多种伪标签与正则化自监督训练。

### 关键设计
1. **可学习 B-spline 运动曲线**:

    - 功能：用 B-spline 曲线和三角函数的组合对每个物体高斯的位置进行时间变形
    - 核心思路：给定 $n+1$ 个可学习控制点 $\mathbf{p}_i$，构建 $k$ 阶 B-spline 曲线 $\mathbf{p}(t) = \sum_{i=0}^{n} \mathbf{p}_i B_{i,k}(t)$。基函数 $B_{i,k}$ 仅在局部区间 $[t_i, t_{i+k}]$ 非零，因此每个时刻位置只受附近 $k$ 个控制点影响。为避免 de Boor-Cox 递推公式的低效性，使用矩阵形式 $\mathbf{p}(t) = [1, u, u^2, ..., u^{k-1}] M_k [\mathbf{p}_{i-k+1}, ..., \mathbf{p}_i]^T$ 加速计算。最终位置变形：$$\boldsymbol{\mu}' = \boldsymbol{\mu} + \mathbf{p}(t) + \sum_{l=1}^{L} \mathbf{a}_l \sin(t \cdot l\pi) + \mathbf{b}_l \cos(t \cdot l\pi)$$
    - 设计动机：单独使用 B-spline 容易在有噪声自监督下过拟合局部噪声，而三角函数通过全局优化所有参数来平滑噪声。两者互补：B-spline 捕捉局部细节（如车辆变道），三角函数保证全局趋势（如匀速前行）。

2. **B-spline 四元数旋转曲线**:

    - 功能：用 B-spline 四元数曲线建模物体高斯的旋转变形
    - 核心思路：标准 B-spline 无法处理单位四元数的不均匀插值问题。采用专门的四元数 B-spline 曲线：$\mathbf{q}(t) = \mathbf{q}_{i-k+1} \prod_{j=i-k+2}^{i} \exp(\mathbf{w}_j \tilde{B}_{j,k}(t))$，其中 $\mathbf{w}_i = \log(\mathbf{q}_{i-1}^{-1} \mathbf{q}_i)$
    - 设计动机：旋转参数在四元数空间中需要特殊处理以保持单位约束和均匀插值。直接用旋转的 B-spline 四元数曲线 $R = \mathbf{q}(t)$ 即足以覆盖自动驾驶中的刚体旋转需求。

3. **简化 2D 伪分割的场景分解**:

    - 功能：将场景简化分为两类——物体（可能运动的，如车辆）和背景（其他所有）
    - 核心思路：使用 SAM 生成二值分割掩模 $\mathcal{M}_{obj}$，根据 LiDAR 点在二值掩模上的投影位置将高斯分为 $\boldsymbol{\Omega}_{obj}$ 和 $\boldsymbol{\Omega}_{bkg}$。通过 α-blending 渲染物体掩模 $\hat{\mathcal{M}}_{obj}$ 并用 BCE 损失监督
    - 设计动机：前人使用实例分割（细粒度多类别）做场景分解，但实例级伪标签噪声极大。简化为二值分割大幅提高了鲁棒性——对自驾场景来说，区分"动态物体"和"静态背景"已经足够。

4. **双向时间可见性掩模**:

    - 功能：处理物体在时间序列中的突然出现/消失
    - 核心思路：对每个物体高斯的不透明度应用时间高斯窗口：$\sigma'(t) = \sigma \cdot e^{-\frac{(t - \mu_t)^2}{2s^2}}$，其中 $\mu_t$ 固定为 LiDAR 采集时间（不可学习），$s_0, s_1$ 分别为双向学习缩放因子。配合扩展正则化损失 $\mathcal{L}_s = \| \frac{2\Delta_f}{s_0 + s_1} \|_1$ 防止窗口过窄
    - 设计动机：前人将 $\mu_t$ 也设为可学习参数，但本文认为 LiDAR 采集时间本身就是物体可见时刻的有效提示。双向（前后方向不同 $s$）设计允许物体从不同方向进出视野。

### 损失函数 / 训练策略
总损失函数包含 8 项：
$$\mathcal{L} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c \mathcal{L}_{D-SSIM} + \lambda_d \mathcal{L}_d + \lambda_f \mathcal{L}_f + \lambda_{obj} \mathcal{L}_{obj} + \lambda_{sky} \mathcal{L}_{sky} + \lambda_r \mathcal{L}_r + \lambda_s \mathcal{L}_s$$

- 图像重建损失（$\mathcal{L}_1$, $\mathcal{L}_{D-SSIM}$）
- 逆深度损失（$\mathcal{L}_d$）：使用 DPTv2 生成单目深度伪标签，直接渲染逆深度期望值避免数值不稳定
- 光流损失（$\mathcal{L}_f$）：使用 CoTracker3 生成伪标签，只监督物体部分
- 物体/天空分割损失（$\mathcal{L}_{obj}$, $\mathcal{L}_{sky}$）：BCE 损失
- 物理刚性正则化（$\mathcal{L}_r$）：约束近邻 8 个高斯的变形参数方差最小化（KNN 每 10 步更新）
- 时间掩模扩展正则化（$\mathcal{L}_s$）
- B-spline 阶数 $k=6$，控制点数为帧数的 1/3，三角函数最大频率 $K=L=6$

## 实验关键数据

### 主实验

| 数据集/设置 | 指标 | AD-GS (自监督) | PVG (自监督) | 4DGF (有标注) |
|------------|------|---------------|-------------|-------------|
| KITTI-75% | PSNR↑ | **29.16** | 27.13 | 31.34 |
| KITTI-75% | SSIM↑ | **0.920** | 0.895 | 0.945 |
| KITTI-75% | LPIPS↓ | **0.033** | 0.049 | 0.026 |
| Waymo | PSNR↑ | **33.91** | 29.54 | 34.64 |
| Waymo | PSNR*(动态)↑ | **27.41** | 21.56 | 29.77 |
| nuScenes | PSNR↑ | **31.06** | 29.49 | - |
| nuScenes | LPIPS↓ | **0.164** | 0.211 | - |

AD-GS 在三个数据集上显著超越所有自监督方法，且接近依赖人工 3D 标注的方法。

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| Full AD-GS | **29.16** | **0.920** | **0.033** | 完整模型 |
| w/o reg | 28.03 | 0.910 | 0.042 | 缺少正则化，结构混乱 |
| w/o flow&depth | 26.98 | 0.902 | 0.048 | 缺少运动和3D信息 |
| w/o obj&sky | 26.52 | 0.896 | 0.053 | 缺少场景分解 |
| 仅 sin/cos | PSNR*=24.28 | - | - | 只有全局拟合 |
| 仅 B-spline | PSNR*=25.70 | - | - | 只有局部拟合 |
| B-spline+sin/cos | PSNR*=26.65 | - | - | 无时间掩模 |
| 完整(+t-mask) | PSNR*=**27.41** | - | - | 全部组件 |

### 关键发现
- B-spline + 三角函数的组合比单独使用任一方法效果更好，验证了局部-全局互补的核心假设
- 双向时间可见性掩模对处理物体出入场景的区域效果显著（PSNR* 从 26.65 提升到 27.41）
- 简化二值分割比细粒度实例分割更鲁棒，因为减少了伪标签噪声
- 训练视图极度稀疏时（KITTI-25%），自监督方法的运动估计困难加大，性能差距拉开
- 物理刚性正则化（$\mathcal{L}_r$）防止了近邻高斯变形不一致导致的伪影

## 亮点与洞察
- **B-spline 的引入**：在动态高斯 Splatting 中首次使用 B-spline 进行运动建模，局部控制性质恰好适合处理有噪声的自监督信号
- **矩阵表示加速**：用矩阵形式替代 de Boor-Cox 递推，实现高效的 B-spline 计算
- **逆深度渲染技巧**：直接渲染逆深度的期望值避免了像素射线上无高斯时的数值不稳定问题
- **"少即是多"的分割策略**：将多类别实例分割简化为二值分割，反而因鲁棒性提升获得了更好的效果

## 局限与展望
- 在极稀疏训练视图（25%）下仍然明显逊色于有标注方法，说明自监督运动估计的上限还有待突破
- 仅建模刚体运动，无法处理行人等非刚体动态物体
- B-spline 控制点数量固定为帧数的 1/3，对于不同速度和复杂度的场景可能不是最优
- 依赖多种外部伪标签（SAM 分割、DPTv2 深度、CoTracker3 光流），如果某个伪标签质量差可能级联影响

## 相关工作与启发
- **vs PVG**: PVG 使用三角函数和周期振动建模运动，全局平滑但缺乏局部细节。AD-GS 在 PVG 基础上加入 B-spline 实现了局部-全局互补
- **vs EmerNeRF**: EmerNeRF 用检测模型特征监督实现静-动分解，但基于 NeRF 速度慢。AD-GS 基于 Gaussian Splatting 效率更高
- **vs 4DGF/StreetGS**: 这些方法依赖人工 3D 标注，AD-GS 在完全自监督下接近其性能，展示了自监督方法的巨大潜力

## 评分
- 新颖性: ⭐⭐⭐⭐ B-spline + 三角函数运动建模的组合设计新颖，简化分割策略反直觉但有效
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、与有标注/无标注方法全面对比、详尽的消融实验
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，公式推导完整，定性/定量结果丰富
- 价值: ⭐⭐⭐⭐⭐ 显著缩小了自监督与有标注方法的差距，对自动驾驶仿真有直接工程价值

<!-- RELATED:START -->

## 相关论文

- [6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)

<!-- RELATED:END -->
