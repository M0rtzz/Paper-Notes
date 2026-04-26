---
title: >-
  [论文解读] Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation
description: >-
  [NeurIPS 2025][3D视觉][4D Gaussian Splatting] 提出 4D Gaussian Ray Tracing (4D-GRT)，将 4D Gaussian Splatting 与物理光线追踪结合，从多视角视频重建动态场景后，以可控参数生成鱼眼畸变、景深模糊、卷帘快门等物理精确的相机效果视频数据。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 4D Gaussian Splatting
  - Ray Tracing
  - Camera Effects
  - Dynamic Scene Reconstruction
  - Data Generation
---

# Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation

**会议**: NeurIPS 2025  
**arXiv**: [2509.10759](https://arxiv.org/abs/2509.10759)  
**代码**: [项目主页](https://shigon255.github.io/4DGRT-project-page)  
**领域**: 3d_vision  
**关键词**: 4D Gaussian Splatting, Ray Tracing, Camera Effects, Dynamic Scene Reconstruction, Data Generation  

## 一句话总结

提出 4D Gaussian Ray Tracing (4D-GRT)，将 4D Gaussian Splatting 与物理光线追踪结合，从多视角视频重建动态场景后，以可控参数生成鱼眼畸变、景深模糊、卷帘快门等物理精确的相机效果视频数据。

## 研究背景与动机

**真实相机效果无处不在**：现实世界中鱼眼畸变 (fisheye)、景深 (depth of field)、卷帘快门 (rolling shutter) 等相机效果普遍存在，但主流视觉系统仅假设理想针孔相机模型，遇到这些效果时性能显著下降。

**训练数据严重匮乏**：缺乏带有精确相机效果参数标注的高质量动态场景数据，导致模型无法学习和应对各类相机效果。

**传统合成方法成本高且存在域差距**：使用 Blender 等渲染引擎合成数据虽然参数可控，但场景建模人力成本高，且合成-真实域差距 (sim-to-real gap) 限制了模型泛化。

**视频生成模型无法理解相机参数**：当前主流视频生成模型（世界模型）对相机效果参数缺乏物理理解，给出数值参数后生成的视频严重违背物理规律，产生伪影或错误效果。

**动态高斯泼溅缺乏光传输建模能力**：Dynamic Gaussian Splatting 方法擅长动态场景重建，但其基于光栅化的渲染管线无法模拟需要光线追踪的相机效果。

**动态 NeRF 速度太慢**：Dynamic NeRF 方法支持光线追踪但渲染速度极慢，且重建质量不如 Gaussian Splatting，难以用于大规模数据生成。

## 方法详解

### 整体框架

4D-GRT 采用两阶段流水线：**第一阶段**从同步多视角视频通过可微光线追踪优化 4D Gaussian Splatting (4D-GS) 表示，重建动态场景；**第二阶段**在重建的场景上，基于物理相机模型参数进行光线追踪，渲染出带有可控相机效果的视频。

### 关键设计 1：4D 高斯变形场动态场景表示

- **功能**：将动态场景表示为一组规范 3D 高斯 $G$ 加上一个变形场网络，通过时空编码器和多头解码器预测每帧的高斯属性残差 $(\Delta x, \Delta r, \Delta s)$。
- **核心思路**：采用 4D 体素平面 $R_l$ 提取时空特征，经轻量 MLP $\varphi$ 融合后，分别用 $\varphi_x, \varphi_r, \varphi_s$ 预测位置、旋转、缩放的残差，叠加到规范高斯上得到变形后的 $G_t$。
- **设计动机**：相比直接存储每帧高斯，变形场参数化更紧凑高效，且时空体素平面编码能捕获多分辨率运动信息，平衡表达能力与存储开销。

### 关键设计 2：可微光线追踪渲染

- **功能**：在训练阶段用可微光线追踪替代传统 3D-GS 光栅化，直接对 3D 高斯基元进行光线追踪。
- **核心思路**：采用 3DGRT 的 $k$-buffer hit-based marching 方案，借助 NVIDIA OptiX 硬件加速接口，光线直接穿过 3D 高斯基元计算交点和颜色，支持端到端可微优化。
- **设计动机**：光栅化渲染仅支持 pinhole 投影，无法模拟复杂透镜畸变和光传输相关的光学效果。光线追踪天然支持任意相机模型的射线生成方式，为后续相机效果模拟提供统一接口。

### 关键设计 3：物理相机效果渲染模块

- **功能**：在重建场景上模拟三种代表性相机效果——鱼眼畸变、景深模糊、卷帘快门。
- **核心思路**：
    - **鱼眼**：采用 4 阶多项式径向畸变模型 $\theta = k_0 + k_1 r + k_2 r^2 + k_3 r^3 + k_4 r^4$，将像素坐标转换为物理传感器坐标后计算极角和方位角，定义球面射线方向。
    - **景深**：给定对焦距离 $f_z$ 和光圈半径 $r_a$，在圆形光圈上均匀采样扰动射线起点，多次采样取平均实现物理准确的散焦模糊。
    - **卷帘快门**：每行像素按其感光时间 $t_r$ 对变形高斯 $G_{t_r}$ 追踪射线，并采用行分块近似策略（chunk size $N_c$）加速。
- **设计动机**：直接在光线追踪框架中嵌入相机光学模型，无需对 Gaussian Splatting 做复杂的 Jacobian 计算或额外模块改造，保持参数可控和物理精确。

### 关键设计 4：卷帘快门行分块加速策略

- **功能**：将图像行划分为大小为 $N_c$ 的块，块内所有行共享同一平均感光时间，从而实现并行光线追踪。
- **核心思路**：假设场景运动适中且快门时间较短，块内行的感光时间差异可忽略；块越大速度越快但可能引入块状伪影，块越小质量越高但速度越低。
- **设计动机**：精确的逐行变形方案无法并行化（每行需独立的变形高斯），分块近似在速度和质量之间取得实用平衡。

## 损失函数与训练

训练损失由两部分组成：渲染图像与真值图像的 L1 损失 $\mathcal{L}_1(C_{v,t}, \hat{C}_{v,t})$，以及 4D 体素平面上的全变分正则化 $\mathcal{L}_{TV}$。每次迭代随机采样视角 $v$ 和时刻 $t$，获取变形高斯后通过光线追踪渲染并优化。训练在 RTX 4090 上约需 3 小时。

## 实验

### 数据集与设置

作者使用 Blender 4.5 构建了 8 个动态室内场景（篮球场、仓库、客厅、浴室），每个场景 50 个相机视角、50 帧、512×512 分辨率，分别渲染 pinhole/fisheye/rolling shutter/DoF 四种相机效果的配对数据作为 benchmark。

### 表 1：Pinhole 渲染质量对比

| 方法 | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ |
|------|------------|--------|---------|-------|
| HexPlane | 23.11 | 0.7956 | 0.2942 | 0.20 |
| MSTH | 29.43 | 0.9023 | 0.1139 | 9.38 |
| **4D-GRT (Ours)** | **32.80** | 0.8898 | **0.1018** | **36.56** |

### 表 2：景深效果渲染对比

| 方法 | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ |
|------|------------|--------|---------|-------|
| HexPlane | 18.37 | 0.7343 | 0.5056 | 0.01 |
| MSTH | 28.47 | 0.9009 | 0.1540 | 0.57 |
| **4D-GRT (Ours)** | **31.25** | **0.9124** | **0.1210** | **3.44** |

### 表 3：卷帘快门效果渲染（不同 chunk size）

| 方法 | Chunk | PSNR (dB) ↑ | FPS ↑ |
|------|-------|-------------|-------|
| HexPlane | N/A | 21.35 | 0.21 |
| MSTH | N/A | 28.70 | 9.35 |
| **4D-GRT** | 1 row | **31.61** | 0.76 |
| **4D-GRT** | 4 rows | **31.61** | 4.99 |
| **4D-GRT** | 16 rows | **31.61** | 13.54 |

### 结果分析

- 4D-GRT 在 pinhole/DoF/rolling shutter 渲染的 PSNR 上均大幅超越两个基线，fisheye 在 masked 评估下 PSNR 也最高（28.89 vs 26.79）。
- Pinhole 渲染速度达 36.56 FPS，是 MSTH 的 ~4 倍，HexPlane 的 ~180 倍。
- 卷帘快门的分块策略有效：chunk=16 时 FPS 从 0.76 提升到 13.54，PSNR 几乎无损。
- 真实世界数据集 Neural 3D Video 上的定性结果也展示了良好的泛化能力。

## 亮点

1. **首个在动态场景高斯表示上执行光线追踪的工作**，打通了 4D-GS 重建与物理相机效果模拟之间的桥梁。
2. 两阶段设计优雅简洁——重建与效果渲染解耦，同一重建结果可渲染任意相机效果组合。
3. 揭示了当前视频生成模型（世界模型）无法理解相机效果参数的问题，有实际工程价值。
4. 提供了首个包含 4 种相机效果的多视角动态场景配对 benchmark，填补数据空白。
5. 卷帘快门行分块加速策略简单有效，在速度与精度之间灵活权衡。

## 局限性

1. **依赖多视角视频输入**：需要充足、同步的多视角视频才能实现高质量重建，无法处理稀疏视角或单目输入场景。
2. **仅在合成数据上定量评估**：缺乏真实世界场景的定量数据，真实场景仅有定性展示。
3. **训练时间较长**：3 小时训练时间虽远快于 HexPlane (12h)，但比 MSTH (8min) 慢得多，如需大规模数据生成可能成为瓶颈。
4. **当前仅支持静态相机的卷帘快门**：尚未实现运动相机下的卷帘快门模拟，限制了应用范围。
5. **鱼眼渲染未观测区域质量不可控**：超出训练视场的区域重建质量无保障，需掩码评估。

## 相关工作

- **3DGRT**：在静态 3D 高斯上做光线追踪，是本文方法在动态场景的直接扩展基础。
- **3DGUT**：通过 unscented transform 近似高斯投影实现不同相机模型支持，但同样限于静态场景。
- **4D-GS (Wu et al.)**：本文直接采用的动态高斯表示框架，用时空体素平面编码变形场。
- **HexPlane / MSTH**：基于平面/网格的 4D NeRF 方法，支持光线追踪但速度慢、质量差。
- **Curved Diffusion / AKiRa**：基于扩散模型的相机效果合成，缺乏物理约束和参数精确控制。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将光线追踪引入 4D 高斯表示用于动态场景相机效果生成
- 实验充分度: ⭐⭐⭐⭐ — 四种效果全面对比，有合成 benchmark 和真实数据定性验证，但缺少下游任务验证
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、方法完整、实验详尽
- 价值: ⭐⭐⭐⭐ — 为视觉模型的相机效果鲁棒性提供了实用的数据生成方案

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](../../ICCV2025/3d_vision/radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[CVPR 2026\] 4C4D: 4 Camera 4D Gaussian Splatting](../../CVPR2026/3d_vision/4c4d_4_camera_4d_gaussian_splatting.md)
- [\[ICCV 2025\] Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation](../../ICCV2025/3d_vision/simulating_dual-pixel_images_from_ray_tracing_for_depth_estimation.md)
- [\[NeurIPS 2025\] RGB-Only Supervised Camera Parameter Optimization in Dynamic Scenes](rgb-only_supervised_camera_parameter_optimization_in_dynamic_scenes.md)
- [\[NeurIPS 2025\] VisualSync: Multi-Camera Synchronization via Cross-View Object Motion](visual_sync_multi-camera_synchronization_via_cross-view_object_motion.md)

<!-- RELATED:END -->
