---
title: >-
  [论文解读] Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)
description: >-
  [CVPR 2025][3D视觉][高斯溅射] 本文提出 EOGS，首个基于 3D 高斯溅射的地球观测框架，通过仿射相机近似、阴影映射和三种正则化策略，在卫星图像三维重建任务上达到与 EO-NeRF 相当的精度，同时训练速度快 300 倍（3 分钟 vs 15 小时）。
tags:
  - CVPR 2025
  - 3D视觉
  - 高斯溅射
  - 卫星摄影测量
  - 数字高程模型
  - 阴影建模
  - 遥感
---

# Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)

**会议**: CVPR 2025  
**arXiv**: [2412.13047](https://arxiv.org/abs/2412.13047)  
**代码**: https://mezzelfo.github.io/EOGS/ (有)  
**领域**: 3D视觉  
**关键词**: 高斯溅射, 卫星摄影测量, 数字高程模型, 阴影建模, 遥感

## 一句话总结
本文提出 EOGS，首个基于 3D 高斯溅射的地球观测框架，通过仿射相机近似、阴影映射和三种正则化策略，在卫星图像三维重建任务上达到与 EO-NeRF 相当的精度，同时训练速度快 300 倍（3 分钟 vs 15 小时）。

## 研究背景与动机

**领域现状**：卫星遥感摄影测量旨在从多视角卫星图像恢复地表三维几何（数字表面模型 DSM）和外观。传统方法依赖双目或三目立体视觉，需要近乎同时采集且卫星位置特定的图像。近年来，基于 NeRF 的方法（如 EO-NeRF）通过可微体渲染实现了多日期、任意卫星位置的高精度重建，成为当前 SOTA。

**现有痛点**：EO-NeRF 虽然精度最高，但训练时间长达 15 小时以上，难以应对未来卫星数据量指数增长的需求。SAT-NGP 虽然将训练缩短到 25 分钟，但精度有明显下降。

**核心矛盾**：遥感场景具有特殊性——推扫式传感器（pushbroom）的相机模型、复杂的光照/阴影变化、多日期采集导致的辐射不一致——标准 3DGS 无法直接处理这些问题。同时 3DGS 缺乏 NeRF 的隐式正则化特性，导致在稀疏视角下优化不稳定。

**本文目标** (1) 如何将 3DGS 的高效框架适配到遥感场景的特殊相机模型和光照条件；(2) 如何在缺乏隐式正则化的 3DGS 中保证稀疏视角下的重建质量。

**切入角度**：推扫式卫星传感器在局部可以用仿射相机精确近似（误差仅 0.012 像素），这恰好兼容高斯溅射的公式推导；阴影可以借鉴图形学中的 Shadow Mapping 技术，通过额外的"太阳相机"渲染来判断遮挡。

**核心 idea**：用仿射相机近似卫星传感器 + Shadow Mapping 建模阴影 + 稀疏性/视图一致性/不透明度三重正则化，让 3DGS 在遥感场景中兼顾效率与精度。

## 方法详解

### 整体框架
EOGS 以标准 3DGS 为基础，输入 N 张非正射校正的卫星图像及其 RPC 相机参数，优化一组高斯原语来恢复场景的三维几何和外观。整体 pipeline 包括：(1) RPC 到仿射相机的近似转换；(2) 带阴影映射的可微渲染；(3) 三种正则化约束的联合优化。最终输出高程渲染图和反照率图。

### 关键设计

1. **仿射相机近似（Affine Camera Approximation）**:

    - 功能：将复杂的 RPC 卫星相机模型转换为兼容 3DGS 的线性投影
    - 核心思路：从世界坐标到 NDC 空间的完整变换链（世界→UTM→经纬度高度→RPC→行列→NDC）可以用逐场景的仿射变换 $\mathcal{A}(\mathbf{x}) = A\mathbf{x} + \mathbf{a}$ 来近似，引入的平均误差仅约 0.012 像素。仿射模型下，高斯投影公式简化为 $\boldsymbol{\mu}^{\mathcal{A}}_k = A\boldsymbol{\mu}_k + \mathbf{a}$，$\Sigma^{\mathcal{A}}_k = A\Sigma A'$，无需标准 3DGS 中的局部一阶近似
    - 设计动机：消除了透视投影的非线性近似误差，同时使卫星相机和太阳相机的处理完全统一

2. **阴影映射（Shadow Mapping）**:

    - 功能：在高斯溅射框架内物理准确地建模卫星图像中的建筑阴影
    - 核心思路：构建一个"太阳相机"$\mathcal{S}$（方向光源用仿射相机建模），从太阳视角和卫星视角分别渲染高程图。对于卫星视角中的每个像素 $\mathbf{u}$，计算其三维点在太阳视角下的同名点高程差 $\Delta h$。若 $\Delta h > 0$（太阳看到更高的点），则该点在阴影中。用指数衰减函数 $s = \min\{\exp(-\rho \Delta h), 1\}$ 计算暗化系数，再结合逐相机的环境光参数 $\psi^{\mathcal{A}}$ 得到最终光照系数
    - 设计动机：EO-NeRF 用光线追踪计算阴影，但高斯溅射没有光线追踪的概念。Shadow Mapping 只需要从不同视角渲染场景——这正是 3DGS 擅长的操作，完美兼容高斯溅射的局部性假设

3. **三重正则化策略**:

    - 功能：弥补 3DGS 缺乏 NeRF 隐式正则化的问题，提升稀疏视角下的重建质量
    - 核心思路：(a) **稀疏性正则化** $\mathcal{L}_o = \frac{1}{K}\sum\alpha_k$：对不透明度做 L1 惩罚，鼓励无用高斯自然消亡，配合阈值剪枝可加速 2 倍训练；(b) **视图一致性正则化**：随机扰动相机得到虚拟视角，要求同一三维点在真实和虚拟视角下的颜色和高程一致，通过遮挡掩码避免被遮部分干扰；(c) **不透明度正则化** $\mathcal{L}_s = \sum H(s)$：对阴影图做二元熵惩罚，迫使阴影非黑即白，防止半透明物体"错误利用"阴影来编码纹理
    - 设计动机：遥感场景视角少且稀疏，3DGS 原语几乎独立优化、缺乏约束。三种正则化从不同角度施加先验：减少原语数量、保证多视角几何一致性、确保物理合理的阴影

### 损失函数 / 训练策略
总损失为：$\min \sum_{i=1}^N \ell(\hat{I}_i, I_i) + 0.1\mathcal{L}_o + 0.1\mathcal{L}_{cc} + 0.01\mathcal{L}_{ac} + 0.01\mathcal{L}_s$，其中 $\ell$ 是标准 3DGS 的光度损失。正则化系数在单一场景上实验确定后取最近的 10 的幂次，直接应用到所有场景。训练仅需 5000 次迭代，阴影映射和正则化在第 1000 次迭代启用。高斯初始化为白色、低不透明度（1%），密度约 0.13 个/m³。

## 实验关键数据

### 主实验
数据集来自 DFC2019（JAX 4 场景）和 IARPA2016（3 场景），使用 LiDAR 扫描作为高程真值。

| 方法 | JAX MAE↓(m) | IARPA MAE↓(m) | 训练时间 |
|------|-------------|---------------|----------|
| EO-NeRF | 1.35 | 1.51 | 15 小时 |
| SAT-NGP | 1.72 | 1.78 | 25 分钟 |
| S2P | 1.53 | 1.78 | 20 分钟 |
| **EOGS** | **1.46** | **1.62** | **3 分钟** |

使用植被掩码后：

| 方法 | JAX MAE↓(m) | IARPA MAE↓(m) |
|------|-------------|---------------|
| EO-NeRF | 1.21 | 1.38 |
| **EOGS** | **1.19** | **1.37** |

### 消融实验

| 配置 | MAE↓(m) | 训练时间(min) |
|------|---------|--------------|
| Base (仿射 3DGS) | 5.03 | 4.18 |
| + Shadow Mapping | 1.86 | - |
| + Sparsity | 1.83 | - |
| + Consistency | 1.69 | - |
| + Opaqueness | 1.79 | - |
| Full EOGS | 1.54 | 2.85 |

线性回归分析各组件独立贡献：Shadow Mapping 3.16m > Consistency 0.20m > Opaqueness 0.09m > Sparsity 0.04m。

### 关键发现
- Shadow Mapping 是最关键的组件，贡献了绝大部分精度提升（3.16m），遥感场景中阴影建模是核心
- 稀疏性正则化虽然对精度贡献最小（0.04m），但显著加速训练（2 倍），是效率的关键
- 去掉植被区域后 EOGS 精度与 EO-NeRF 完全持平，说明 EOGS 对结构化物体（建筑等）的重建质量极高
- EOGS 在高覆盖区域（被多个相机观察到的区域）表现优于 EO-NeRF，但在低覆盖区域稍弱

## 亮点与洞察
- **仿射近似的精妙**：推扫式卫星传感器看似复杂，但由于卫星距离极远、场景尺度相对小，仿射近似误差仅 0.012 像素。这个观察消除了适配 3DGS 的最大障碍，且使太阳相机和卫星相机统一处理
- **Shadow Mapping 的巧妙迁移**：将图形学中经典的 Shadow Mapping 技术引入可微渲染框架，完美绕开了"3DGS 没有光线追踪"的限制。这种思路——从图形学借用成熟技术来解决可微渲染问题——值得推广
- **300 倍加速的实用价值**：3 分钟 vs 15 小时的差距使大规模卫星数据处理从不可行变为可行，这是真正的工程突破

## 局限与展望
- 在植被区域表现不佳：树木等非刚性、半透明物体难以用高斯原语精确建模
- 低覆盖区域（仅被少数相机观察到）的重建质量弱于 EO-NeRF，需要更强的先验或更好的初始化
- 目前仅处理光学卫星图像，未考虑 SAR 或多光谱数据
- 正则化系数虽然鲁棒，但仍是手动设置的（取最近的 10 的幂次）

## 相关工作与启发
- **vs EO-NeRF**: EO-NeRF 用光线追踪计算阴影且有隐式正则化，精度略高但训练极慢。EOGS 用 Shadow Mapping 和显式正则化取代，在效率上实现质的飞跃
- **vs SAT-NGP**: 基于 Instant-NGP 的加速方案，25 分钟但精度明显下降。EOGS 既更快又更准，是 Pareto 最优
- **vs 标准 3DGS**: EOGS 的核心贡献在于表明 3DGS 的"遥感适配"只需少量改动（仿射相机+阴影+正则化），而非重新设计框架

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个将 3DGS 应用于卫星摄影测量，仿射近似和 Shadow Mapping 的结合很自然但确实是新的
- 实验充分度: ⭐⭐⭐⭐ 7 个场景的完整评估+详细消融+参数敏感性分析+可见性分析
- 写作质量: ⭐⭐⭐⭐⭐ 公式推导清晰，图示直观，方法部分逻辑链条完整
- 价值: ⭐⭐⭐⭐⭐ 300 倍加速使大规模遥感成为可能，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting](rainygs_efficient_rain_synthesis_with_physically-based_gaussian_splatting.md)
- [\[CVPR 2025\] GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Matrix3D: Large Photogrammetry Model All-in-One](matrix3d_large_photogrammetry_model_all-in-one.md)
- [\[CVPR 2025\] AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)
- [\[CVPR 2026\] From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images](../../CVPR2026/3d_vision/from_orbit_to_ground_generative_city_photogrammetry_from_extreme_off-nadir_satel.md)

<!-- RELATED:END -->
