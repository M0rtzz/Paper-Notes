---
title: >-
  [论文解读] RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] RainyGS 将基于物理的雨滴仿真和浅水动力学与 3D 高斯溅射渲染框架结合，首次实现了开放世界场景中高保真、物理准确且实时（>30fps）的动态雨天效果合成，支持从小雨到暴雨的灵活控制。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯溅射
  - 雨天效果合成
  - 物理仿真
  - 浅水方程
  - 屏幕空间光线追踪
---

# RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2503.21442](https://arxiv.org/abs/2503.21442)  
**代码**: https://pku-vcl-geometry.github.io/RainyGS/  
**领域**: 3D视觉 / 场景编辑  
**关键词**: 3D高斯溅射, 雨天效果合成, 物理仿真, 浅水方程, 屏幕空间光线追踪

## 一句话总结

RainyGS 将基于物理的雨滴仿真和浅水动力学与 3D 高斯溅射渲染框架结合，首次实现了开放世界场景中高保真、物理准确且实时（>30fps）的动态雨天效果合成，支持从小雨到暴雨的灵活控制。

## 研究背景与动机

**领域现状**：3D场景建模近年取得重大进展，NeRF 和 3DGS 成为强大的场景表示工具。同时，基于物理的天气仿真能产生逼真效果但依赖手工场景搭建。两者各有优势但难以结合。

**现有痛点**：(1) 视频生成方法（如 Runway）能生成视觉上有吸引力的雨天风格，但缺乏3D一致性、物理准确性和动态演化；(2) ClimateNeRF 只支持静态天气效果；(3) Gaussian Splashing 使用低效的粒子仿真，无法实时运行；(4) 数据驱动的雨天合成方法（CycleGAN等）只处理单张图像，无法保证多视角一致性。

**核心矛盾**：雨天场景涉及极其复杂的物理和渲染现象——雨丝飞行、地面积水、波纹、反射、折射、水花等需同时出现并随时间演化，且需高效计算。现有方法要么物理不准确、要么计算太慢、要么缺乏3D一致性。

**本文目标**：在3DGS框架中实现完整的物理雨天效果系统，覆盖所有关键雨天现象并达到实时性能。

**切入角度**：作者注意到雨天场景中水体可以用高度场近似（浅水假设），这与3DGS的光栅化管线天然兼容。结合屏幕空间光线追踪实现反射渲染，避免了3DGS中高斯实体做光追的巨大开销。

**核心 idea**：用浅水方程在高度图上仿真水的动力学，用屏幕空间光线追踪在光栅化管线中实现反射/折射，将两者与3DGS无缝集成。

## 方法详解

### 整体框架

给定多视角图像输入，首先用 PGSR 重建场景的外观和几何。然后进行辅助图提取（高度图、深度图、法线图）。在高度图上运行浅水仿真生成积水动力学。最后通过反射感知的水面光栅化方法合成雨丝、反射、折射和水花效果，组合生成最终的动态雨天图像。

### 关键设计

1. **基于高度图的浅水仿真**:

    - 功能：模拟地面积水的动态行为（水波、涟漪、雨滴溅射）
    - 核心思路：采用浅水方程(SWE) $\partial h/\partial t + (\mathbf{u}\cdot\nabla)h = -h(\nabla\cdot\mathbf{u})$ 在2D高度图上仿真。雨滴随机生成，碰撞检测后体积加入高度场 $h_{i,j} \leftarrow h_{i,j} + 4\pi r_i^3/(3\Delta x^2)$。使用两层高度图（地面和遮挡物）和半拉格朗日对流方案。关键是维护水面高度场 $\eta = H + h$，其中 $H$ 是地面高度
    - 设计动机：浅水假设非常适合雨天——积水确实很浅，且高度场表示规避了3DGS内部几何缺失的问题（GS没有体内部结构）。相比粒子仿真大幅提高效率

2. **反射感知的水面光栅化**:

    - 功能：在光栅化管线中实现物理准确的反射、折射和高光效果
    - 核心思路：分两个渲染 pass。第一个 pass 渲染水面层 $I_0 = (1-F)I_{\text{refra}} + F(I_{\text{spec}} + I_{\text{highl}})$，其中 $F$ 是 Fresnel 系数。镜面反射 $I_{\text{spec}}$ 通过屏幕空间光线追踪（SSR）在已栅格化图像上 march 获得——只在2D图像上追踪反射方向，不需真正的3D光追。折射用图像扭曲近似 $I_{\text{refra}}(u,v) = I_{\text{src}}(u+n_uk, v+n_vk)$。高光用 Blinn-Phong 模型。第二个 pass 光栅化雨丝和水花
    - 设计动机：3D光追在GS场景中极其昂贵（需遍历大量高斯），屏幕空间光追只在2D层面操作，与光栅化管线完美兼容。反射效果对视觉真实感贡献最大，是最值得投入的渲染效果

3. **基于 PGSR 的精确几何重建**:

    - 功能：提供雨天仿真所需的高质量场景几何
    - 核心思路：使用 PGSR（带平面约束的高斯）替代普通3DGS进行场景建模，加入单目法线估计先验监督法线图。PCA 确定地面平面方向，从正交相机渲染深度图得到高度图。辅助图（深度、法线、外观）通过 alpha blending 获取
    - 设计动机：雨天仿真对几何精度要求高——地面平整度直接影响积水效果，法线精度影响反射方向。普通3DGS的几何重建质量不够

### 损失函数 / 训练策略

场景建模阶段使用 PGSR 的标准训练流程加上法线先验损失。雨天效果合成阶段无需训练，是纯推理时的物理仿真+渲染。用户可通过雨滴密度、速度、方向等参数控制雨量级别。

## 实验关键数据

### 主实验

在 MipNeRF360（Garden、Treehill、Bicycle）和 Tanks and Temples（Family、Truck）数据集上进行评估。

性能对比:

| 方法 | 帧渲染时间 | 显存峰值 |
|------|-----------|---------|
| PGSR（无雨效果） | 0.007s | 7.989 GB |
| PGSR + RT（3D光追） | 1.942s | 14.161 GB |
| Runway-V2V | ~0.4s | NA |
| RainyGS | **0.032s** | 8.561 GB |

### 消融实验

Fig.3 展示了逐步叠加各渲染模块的效果：

| 配置 | 效果描述 |
|------|---------|
| $I_{\text{src}}$ | 原始场景 |
| + $I_{\text{spec}}$ | 加入镜面反射，地面出现倒影 |
| + $I_{\text{refra}}$ | 加入折射，地面出现扭曲效果 |
| + $I_{\text{highl}}$ | 加入高光，水面出现太阳光斑 |
| + Fresnel 调色 | 场景整体色调变暗、更真实 |
| + 雨丝 | 完整雨天效果 |

### 关键发现

- RainyGS 仅比 PGSR 增加 0.025s/帧和 0.572 GB显存，而3D光追方案增加近 2s/帧——屏幕空间光追极为高效
- Runway-V2V 虽然视觉上有雨天风格但缺乏3D一致性（同一场景不同视角雨效果不连贯）、物理不准确（水波没有涟漪响应）
- Rain Motion 无法生成积水效果，只能叠加简单的雨丝
- Instruct-GS2GS 只能做静态编辑，无法生成动态雨天效果
- 在自动驾驶（Waymo）场景中也能工作，展示了框架在大规模场景的可扩展性

## 亮点与洞察

- **浅水方程的巧妙选择**：雨天积水天然满足"浅水"假设，这个物理建模的选择既准确又高效。高度场表示还恰好回避了3DGS没有内部几何的缺陷，一举两得
- **屏幕空间光追的工程智慧**：将反射计算从3D空间降维到2D屏幕空间，完美融入光栅化管线。虽然无法处理被遮挡物体的反射，但对可见区域的反射效果已经足够真实
- **完整的效果系统**：从雨丝、积水、水波到反射、折射、Fresnel 调色，覆盖了雨天场景的所有关键视觉元素，不是只做一个效果

## 局限与展望

- 浅水模型不适用于深水场景（如洪水）和复杂多层水体
- 屏幕空间渲染无法反射被遮挡物体
- 精确度依赖场景重建质量——输入视角不足时地面几何可能不平整导致积水异常
- 未来可扩展到雾、雪等其他天气效果，构建统一的天气仿真框架
- 若结合可微渲染，雨天效果可以作为数据增强提升自动驾驶系统的鲁棒性

## 相关工作与启发

- **vs ClimateNeRF**: ClimateNeRF 是该领域先驱但只支持静态天气（定格的洪水、雾、雪），RainyGS 实现了动态演化的雨天效果
- **vs Gaussian Splashing**: Gaussian Splashing 将每个高斯作为物理粒子进行流体仿真，计算极其昂贵。RainyGS 通过高度场抽象和屏幕空间渲染大幅降低成本
- **vs PhysGaussian**: PhysGaussian 关注固体物理（如弹性变形），RainyGS 专注流体相关效果，两者互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在3DGS中实现完整的物理雨天仿真，浅水+SSR的技术选型非常合适
- 实验充分度: ⭐⭐⭐⭐ 多场景验证、性能分析充分，但缺乏量化的雨天真实感度量（如用户研究）
- 写作质量: ⭐⭐⭐⭐ Pipeline 清晰，模块化描述便于理解，图示丰富
- 价值: ⭐⭐⭐⭐ 开放世界雨天合成有广泛应用前景（自动驾驶、游戏、AR），实时性能是重要优势

<!-- RELATED:START -->

## 相关论文

- [CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis](splatflow_multi-view_rectified_flow_model_for_3d_gaussian_splatting_synthesis.md)
- [Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)
- [GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)
- [Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](../../CVPR2026/3d_vision/physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)

<!-- RELATED:END -->
