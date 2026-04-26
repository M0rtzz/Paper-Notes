---
title: >-
  [论文解读] PGC: Physics-Based Gaussian Cloth from a Single Pose
description: >-
  [CVPR 2025][3D视觉][3D高斯] 提出 PGC 方法，仅从单帧多视角拍摄重建可模拟的逼真服装资产，通过网格嵌入 3D 高斯 + 基于物理的渲染（PBR）的混合策略，实现了新姿态下同时具备高频细节和正确光照效果的服装渲染。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯
  - 布料模拟
  - 物理渲染
  - 混合表示
  - 单帧重建
---

# PGC: Physics-Based Gaussian Cloth from a Single Pose

**会议**: CVPR 2025  
**arXiv**: [2503.20779](https://arxiv.org/abs/2503.20779)  
**代码**: 无（项目页面: phys-gaussian-cloth.github.io）  
**领域**: 3D视觉 / 服装重建  
**关键词**: 3D高斯, 布料模拟, 物理渲染, 混合表示, 单帧重建

## 一句话总结

提出 PGC 方法，仅从单帧多视角拍摄重建可模拟的逼真服装资产，通过网格嵌入 3D 高斯 + 基于物理的渲染（PBR）的混合策略，实现了新姿态下同时具备高频细节和正确光照效果的服装渲染。

## 研究背景与动机

**领域现状**：服装重建和动画化是虚拟数字人的关键技术。传统方法基于网格表示，天然适合物理模拟但难以表达高频几何细节（如毛绒、口袋、拉链）。近年来 3D 高斯泼溅（3DGS）方法在外观重建上表现出色，能捕捉精细的体积化细节，但高斯缺乏驱动和模拟能力。

**现有痛点**：(1) 网格方法受限于网格分辨率，无法表达毛绒、针织等高频表面细节；(2) 纯 3DGS 方法会"烘焙"训练帧的光照信息（如阴影、高光），新姿态下会出现错误的光照效果；(3) 将 3DGS 应用于服装通常需要多帧追踪（如 Gaussian Garments 需要视频序列追踪），追踪本身计算昂贵且不精确会导致模糊。

**核心矛盾**：高质量服装外观（需要 3DGS 的体积化细节）与新姿态泛化（需要正确的光照响应和物理模拟能力）之间存在固有矛盾，且现有方法要么需要大量多帧数据要么牺牲外观质量。

**本文目标**：仅从单帧多视角拍摄重建一个同时具备精细外观和物理可模拟性的服装资产。

**切入角度**：观察到图像信号可以分解为低频（姿态相关的光照、阴影等远场效应）和高频（织物纹理、毛绒等近场细节）两部分。低频部分可以用传统 PBR 方法在新姿态下正确计算，高频部分在姿态变化时大致不变可用 3DGS 固定捕捉。

**核心 idea**：构建网格嵌入 3DGS + PBR 的混合表示，在渲染时分别用高通滤波提取 3DGS 的高频细节和低通滤波提取 PBR 渲染的低频光照，两者合成得到兼具细节和正确光照的最终图像。

## 方法详解

### 整体框架

输入为一个静态姿态下的多视角拍摄（170 相机半球排列），输出为可模拟服装资产。Pipeline 包含三个核心组件：(1) 网格嵌入 3DGS 重建——在重建网格上锚定高斯并优化外观；(2) PBR 外观重建——估计反照率贴图和布料特有的 BRDF 反射率参数；(3) 混合渲染——在新姿态下用物理模拟驱动网格变形后，组合 3DGS 高频信息和 PBR 低频信息得到最终渲染。

### 关键设计

1. **网格嵌入 3DGS 重建**:

    - 功能：从多视角拍摄重建锚定在网格上的高斯表示，捕捉精细织物细节
    - 核心思路：先通过立体匹配和表面重建从多视角图像得到服装网格，重新网格化使其适合模拟。在网格表面采样 100 万个高斯点，每个高斯的位置、旋转定义在其所属三角面的局部坐标系中。优化两个损失：重建损失 $\mathcal{L}_{3DGS} = \lambda \|I_k - G_k\|_1 + (1-\lambda) \text{SSIM}(I_k, G_k)$ 和前景正则化 $\mathcal{L}_{fg}$（使高斯的累积不透明度与前景 mask 一致）。训练在完整图像（非分割后的服装图像）上进行，避免分割不准影响边缘模糊细节。
    - 设计动机：将高斯锚定在网格上使得网格变形时高斯能忠实跟随，同时保留了高斯捕捉体积化细节的优势。在完整图像上训练可以学到服装边缘的模糊细节（毛绒、飞丝）。

2. **布料特化的 PBR 外观模型**:

    - 功能：为服装提供可在新姿态/新光照下正确响应的远场着色能力
    - 核心思路：基于 Disney BRDF 但替换了光泽（sheen）模型。观察到织物在掠射角会有显著的前向和后向散射（由于边缘的飞丝纤维），Disney BRDF 自带的 sheen 分量无法正确匹配。作者采用了基于光学仿真并考虑了纤维多次散射的 sheen BRDF $f_s$，最终外观模型 $f = \sigma_s f_s + H_s(\mathbf{o}) \sigma_d(\mathbf{x}) f_d$，其中 $\sigma_d$ 是空间变化的反照率，$\sigma_s$ 是光泽颜色，$H_s$ 是保证能量守恒的光泽透射项。反照率通过预训练的内在图像分解网络分离光照后反投影到纹理空间，其余参数（粗糙度、光泽颜色、光泽粗糙度）通过 Mitsuba 可微渲染在训练帧上联合优化。
    - 设计动机：Lambertian 模型完全缺乏光泽效果，标准 Disney BRDF 会过度估计前向散射。布料特化的光泽模型能更准确地匹配真实面料的外观，PSNR 提升显著。

3. **高斯-PBR 混合渲染**:

    - 功能：组合高频细节和低频光照，实现新姿态下的高质量渲染
    - 核心思路：利用物理模拟器（XPBD）生成新姿态下的网格顶点位置 $V_t$。远场着色：在纹理空间用 PBR 模型渲染新姿态的贴图 $T_t$，将颜色传递到高斯的零阶球谐系数 $\phi_t^{\circ}$，用 3DGS 渲染器得到 $S_t$。近场着色：将原始优化的高斯以完整球谐系数在新姿态下渲染得到 $G_t$，用 alpha 加权的高斯模糊做高通 $h(G_t) = G_t - l(G_t)$。最终合成 $\hat{I}_t = h(G_t) + l(S_t)$，高通保留了织物纹理细节，低通提供了正确的褶皱阴影和间接照明。
    - 设计动机：姿态相关效应（阴影、间接照明）主要在低频信号中，可由 PBR 正确计算；高频信号（织物纹理、毛绒）在姿态变化时基本不变，可从原始 3DGS 中提取。这种频域分解策略优雅地解决了"烘焙光照"问题。

### 损失函数 / 训练策略

3DGS 优化使用标准 L1+SSIM 重建损失和前景正则化损失。PBR 参数优化使用 Mitsuba 可微渲染。模拟使用 XPBD 方法在 SMPL-like 参数化身体上进行。服装网格使用图像分割自动分离。整个流程不需要视频序列追踪，仅需单帧多视角拍摄。

## 实验关键数据

### 主实验

| 方法 | FSIM↑ | LPIPS (×10⁻²)↓ |
|------|-------|-----------------|
| SCARF | 0.764 | 5.00 |
| Animatable Gaussians | 0.827 | 3.39 |
| **PGC (Ours)** | **0.834** | **3.38** |

### 消融实验

| 配置 | FSIM↑ | LPIPS (×10⁻²)↓ |
|------|-------|-----------------|
| 3DGS-Only | 0.825 | 3.41 |
| PBR-Only | 0.809 | 4.67 |
| **Full (Hybrid)** | **0.834** | **3.38** |

### 关键发现

- 3DGS-Only 在训练帧重建质量好但新姿态下光照错误（烘焙阴影导致不真实的外观）
- PBR-Only 有正确的重新光照但缺乏高频织物细节（只能渲染平面 2D 纹理）
- 混合方法兼具两者优势，在 FSIM 和 LPIPS 上均优于单独使用
- 布料特化的 PBR 模型（含 sheen）在 PSNR 上显著优于 Lambertian 和标准 Disney BRDF
- 相比 Gaussian Garments 省去了 24.5 小时的多视角配准时间，且避免了追踪导致的细节模糊

## 亮点与洞察

- "只用一帧"的约束极具实用价值，大幅降低了数据采集成本
- 高频/低频分解的渲染思路非常优雅，充分发挥了两种表示的各自优势
- 布料特化的 sheen BRDF 切中了织物外观的关键视觉特征（飞丝散射）
- 重建的资产天然支持实时应用（实时 PBR + 实时 3DGS 渲染 + 实时 XPBD 模拟）

## 局限与展望

- 忽略了变形和新光照对高频外观的影响（如新褶皱处的微观纹理变化）
- 对 3DGS 在未充分观察区域（如腋下）的泛化能力有限
- 反照率提取依赖于外部模型，多视角不一致会导致远场渲染的伪影
- 仅支持薄壳几何假设，无法处理口袋内部、多层服装或服装开口
- 未来可通过可微模拟恢复更准确的休息形状和材料参数，减少"下垂"现象

## 相关工作与启发

- 将 3DGS 与传统 PBR 结合的混合策略可推广到其他需要"外观重建 + 物理驱动"的场景
- 频域分解的思路可应用于其他烘焙光照问题（如 NeRF 的 relighting）
- 布料 BRDF 模型对虚拟试穿、时尚电商等应用场景有直接价值

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4.5 |
| 实验充分度 | 3.5 |
| 写作质量 | 4 |
| 总体评价 | 4 |

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [\[CVPR 2025\] FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity](freegave_3d_physics_learning_from_dynamic_videos_by_gaussian_velocity.md)
- [\[CVPR 2025\] AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] PhysAnimator: Physics-Guided Generative Cartoon Animation](physanimator_physics-guided_generative_cartoon_animation.md)

<!-- RELATED:END -->
