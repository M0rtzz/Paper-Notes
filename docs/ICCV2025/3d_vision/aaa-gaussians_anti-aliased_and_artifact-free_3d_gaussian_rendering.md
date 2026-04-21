---
title: >-
  [论文解读] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering
description: >-
  [ICCV 2025][3D视觉][3D高斯喷溅] AAA-Gaussians提出了一种统一的3D高斯光栅化框架，通过自适应3D平滑滤波器、视空间透视正确边界计算和基于视锥体的3D裁剪，在单一框架内同时解决了3DGS的锯齿、投影畸变和闪烁三大顽疾，在分布外视角评估中大幅领先其他方法，同时保持实时渲染性能。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯喷溅
  - 抗锯齿
  - 伪影消除
  - 视锥体裁剪
  - 实时渲染
---

# AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering

**会议**: ICCV 2025  
**arXiv**: [2504.12811](https://arxiv.org/abs/2504.12811)  
**代码**: [https://github.com/DerThomy/AAA-Gaussians](https://github.com/DerThomy/AAA-Gaussians)  
**领域**: 3D视觉 / 3D高斯喷溅  
**关键词**: 3D高斯喷溅, 抗锯齿, 伪影消除, 视锥体裁剪, 实时渲染

## 一句话总结
AAA-Gaussians提出了一种统一的3D高斯光栅化框架，通过自适应3D平滑滤波器、视空间透视正确边界计算和基于视锥体的3D裁剪，在单一框架内同时解决了3DGS的锯齿、投影畸变和闪烁三大顽疾，在分布外视角评估中大幅领先其他方法，同时保持实时渲染性能。

## 研究背景与动机
- **领域现状**：3D高斯喷溅（3DGS）彻底变革了逆渲染领域，通过高效的光栅化实现实时渲染。但原始3DGS将3D高斯投影为2D splat时采用仿射近似，引入了多种伪影
- **三类核心伪影**：
    - (1) **投影畸变**：将3D高斯近似为2D splat在大FOV和图像边缘产生云状扭曲
    - (2) **锯齿伪影**：缩放训练/测试分辨率差异导致的闪烁和采样不足
    - (3) **闪烁/弹出伪影**：全局深度排序的不精确导致相机旋转时混合顺序突变
- **现有解决方案的碎片化**：StopThePop解决排序、Mip-Splatting解决锯齿、3D评估方法解决畸变——但没有方法同时处理所有问题。部分方法在解决一个问题时引入新问题（如3D评估时无法使用2D抗锯齿滤波器）
- **核心矛盾**：3D评估（在射线上计算高斯贡献）可避免投影畸变，但现有3D抗锯齿和2D边界计算方案出现冲突、性能损失或引入新伪影
- **本文切入角度**：在整个3DGS渲染管线中全面考虑高斯的3D本质——从抗锯齿、边界、裁剪到排序全部在3D中完成

## 方法详解

### 整体框架
基于层级排序光栅化器（StopThePop）和屏幕空间平面3D评估（Hybrid Transparency），替换其中的边界计算、裁剪、深度评估、贡献估计和抗锯齿为全3D实现。使用MCMC进行点云致密化。

### 关键设计

1. **自适应3D抗锯齿滤波器**:

    - 功能：解决3D评估下的锯齿伪影，替代无法直接应用的2D Mip滤波器
    - 核心问题：朴素地在3D中重算平滑滤波器会导致高斯过度透明——因为振幅按体积变化缩放（公式(8)中三个轴的乘积），而3D评估只取射线上最大贡献点，不进行积分
    - 解决方案：只根据垂直于观察射线方向 $\mathbf{d}$ 的面积变化来调整振幅：
    $\hat{\mathcal{G}}_\perp(\mathbf{x}) = \sqrt{\frac{|\mathbf{\Sigma}_\perp|}{|\hat{\mathbf{\Sigma}}_\perp|}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \hat{\mathbf{\Sigma}}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$
    - 垂直缩放因子的封闭解：$\sqrt{\frac{|\mathbf{\Sigma}| \cdot \mathbf{d}^\top\mathbf{\Sigma}^{-1}\mathbf{d}}{|\hat{\mathbf{\Sigma}}| \cdot \mathbf{d}^\top\hat{\mathbf{\Sigma}}^{-1}\mathbf{d}}}$
    - 自适应机制：$\hat{v}' = \min(\hat{v}_{\text{train}}, \hat{v})$，其中 $\hat{v} = f/d$，确保靠近时不过度缩小、远离时有效滤波
    - 设计动机：避免沿射线方向的缩放变化被纳入振幅计算，防止高各向异性高斯变得过于透明

2. **视空间透视正确边界**:

    - 功能：解决高斯延伸到近平面后方时的弹出伪影
    - 核心问题：Hahlbohm等人在屏幕空间求解边界平面，当高斯的z范围超出近/远平面时直接丢弃该高斯，导致画面边缘弹出
    - 解决方案：改为在视空间中用角度参数化的平面进行拟合：
    $\theta_{1,2} = \tan^{-1}\left(\frac{s_{1,3} \pm \sqrt{s_{1,3}^2 - s_{1,1}s_{3,3}}}{s_{3,3}}\right)$
    - 角度范围裁剪到 $[-\pi/2+\epsilon, \pi/2-\epsilon]$，保证在远超视锥体时仍能正确处理
    - 退化处理：椭球与坐标轴相交时保守设为全屏边界；相机在椭球内部时丢弃该高斯
    - 设计动机：视空间角度参数化天然处理了高斯穿过近平面的情况，而屏幕空间方案在此场景下失效

3. **基于视锥体的3D裁剪**:

    - 功能：将2D基于tile的裁剪提升为3D视锥体裁剪，加速渲染并减少排序开销
    - 核心思路：为每个tile构造3D视锥体 $\mathcal{F}$，由4个屏幕空间平面定义，然后在高斯归一化空间中找视锥体内最大贡献点
    - 裁剪判据：$\min_{\mathbf{x}\in\mathcal{F}} \rho(\mathbf{x})^2 < \tau_\rho$，若最大贡献小于阈值则丢弃
    - 优化：只投影到最近的x/y平面（2个平面+3条边），而非全部4个平面+4条边
    - 额外用途：同样的视锥裁剪应用于全局预处理，丢弃与整个视锥体不相交的高斯
    - 设计动机：2D tile裁剪对非轴对齐的长条形高斯效果差；3D裁剪大幅减少无效的高斯-tile组合，显著降低层级排序的排序开销

### 损失函数 / 训练策略
- 遵循标准3DGS训练流程，使用MCMC致密化策略
- 3D抗锯齿滤波核大小 $k = 0.3$（与Mip-Splatting一致）
- 训练时启用所有组件以产生视图一致的表征

## 实验关键数据

### 主实验（标准分布内评估）

| 数据集 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 无伪影 |
|--------|------|--------|--------|---------|--------|
| Mip-NeRF 360 | 3DGS | 27.44 | 0.814 | 0.215 | ✗ |
| Mip-NeRF 360 | MCMC | 28.03 | 0.836 | 0.187 | ✗ |
| Mip-NeRF 360 | Mip-Splatting | 27.54 | 0.817 | 0.216 | 部分 |
| Mip-NeRF 360 | **AAA-Gaussians** | **27.84** | **0.836** | **0.188** | **✓** |
| Deep Blending | **AAA-Gaussians** | **30.49** | **0.913** | **0.222** | **✓** |

### 消融实验（分布外评估 - 大FOV 3×）

| 方法 | Mip-NeRF 360 PSNR | T&T PSNR | Deep Blending PSNR |
|------|-------------------|----------|-------------------|
| 3DGS | 26.82 | 17.11 | 26.19 |
| MCMC | 23.35 | 14.37 | 18.32 |
| Mip-Splatting | 26.05 | 17.31 | 25.59 |
| StopThePop | 27.04 | 20.24 | 27.55 |
| **AAA-Gaussians** | **27.84** | **23.58** | **30.49** |

### 关键发现
- 在分布外视角（大FOV 3×）下，MCMC和Taming 3DGS的质量崩塌严重（T&T从24.6降到14.4），而本文方法完全不受影响
- 多分辨率评估中，半分辨率下Bonsai场景从MCMC的28.98提升到32.12 PSNR，双分辨率下也有显著优势
- 3D视锥裁剪对质量无影响但显著提升性能（移除裁剪后渲染时间从7.72ms增至14.40ms）
- 本文方法仅比MCMC略慢（7.72ms vs 6.79ms），在某些场景甚至更快
- 3D评估的速度与2D splat近似相当甚至更快（7.03ms vs 7.52ms）

## 亮点与洞察
- **统一框架**：据作者所知，这是唯一一个在单框架内同时消除锯齿、畸变和闪烁三类伪影的光栅化方法
- **垂直缩放因子的推导**：只考虑垂直于观察方向的面积变化是一个精妙的洞察，避免了3D抗锯齿导致高斯过度透明的问题
- **视空间角度边界**：角度参数化使得高斯穿过近平面时仍能正确处理，这是实际VR渲染中的关键需求
- **工程质量**：开源代码，实测超100 FPS消费级硬件，具备实际部署价值
- **诚实的评估**：作者坦承标准分布内指标提升不显著，因为视图一致性反而阻止了模型"作弊"过拟合

## 局限与展望
- 标准分布内指标提升有限（因为强视图一致性限制了过拟合空间）
- 视空间边界仍依赖针孔相机模型，对鱼眼等其他相机模型适应性有限
- 更强的视图一致性意味着需要更具表达力的视角依赖编码来弥补
- 目前基于MCMC致密化，与其他致密化策略的组合有待探索

## 相关工作与启发
- **vs Mip-Splatting**：Mip-Splatting的2D Mip滤波器无法用于3D评估；本文的3D自适应滤波器完全替代了2D方案
- **vs StopThePop**：继承了其层级排序解决闪烁，但替换了2D裁剪为3D视锥裁剪
- **vs Hybrid Transparency**：继承了其屏幕空间平面3D评估，但修复了其边界计算导致的弹出和锯齿问题
- **vs 射线追踪方法**：射线追踪虽然天然3D但计算昂贵且需要加速结构；本文证明光栅化可以达到同等的无伪影效果

## 评分
- 新颖性: ⭐⭐⭐⭐ 3D抗锯齿垂直缩放因子和视空间边界计算具有技术新颖性
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖分布内/外评估、多分辨率、大FOV、性能计时、详细消融
- 写作质量: ⭐⭐⭐⭐ 技术推导清晰，问题分析到位
- 价值: ⭐⭐⭐⭐⭐ 实用性极强，是3DGS走向工程化的关键一步

<!-- RELATED:START -->

## 相关论文

- [Anti-Aliased 2D Gaussian Splatting](../../NeurIPS2025/3d_vision/anti-aliased_2d_gaussian_splatting.md)
- [Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs](compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)
- [A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting](7dgs_unified_spatial-temporal-angular_gaussian_splatting.md)
- [Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)

<!-- RELATED:END -->
