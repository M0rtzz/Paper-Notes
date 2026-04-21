---
title: >-
  [论文解读] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction
description: >-
  [AAAI 2026][3D视觉][3D高斯泼溅] 提出 OceanSplat，通过三目视图一致性约束、合成对极深度先验和深度感知透明度调整，实现了散射介质下的高保真水下 3D 高斯泼溅场景重建，显著减少了浮动伪影并超越现有方法。
tags:
  - AAAI 2026
  - 3D视觉
  - 3D高斯泼溅
  - 水下场景重建
  - 三目立体一致性
  - 深度正则化
  - 散射介质
---

# OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction

**会议**: AAAI 2026  
**arXiv**: [2601.04984](https://arxiv.org/abs/2601.04984)  
**代码**: [oceansplat.github.io](https://oceansplat.github.io)  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 水下场景重建, 三目立体一致性, 深度正则化, 散射介质

## 一句话总结

提出 OceanSplat，通过三目视图一致性约束、合成对极深度先验和深度感知透明度调整，实现了散射介质下的高保真水下 3D 高斯泼溅场景重建，显著减少了浮动伪影并超越现有方法。

## 研究背景与动机

水下场景重建对海底测绘、生态监测、水下基础设施检测等海洋机器人任务至关重要。然而水下环境的光学特性（波长相关衰减、散射、低照明）严重退化视觉线索，给基于视觉的场景重建带来巨大挑战。

**现有方法的局限**：

**NeRF 方法**（SeaThru-NeRF 等）：将水下物理模型嵌入体积渲染，但隐式表示阻碍精确几何理解，且渲染速度慢

**3DGS 方法**（SeaSplat、WaterSplatting 等）：虽然渲染快，但**介质强度常被吸收进 3D 高斯中**，导致大量浮动伪影（floating artifacts），3D 高斯与散射介质纠缠，重建质量下降

**核心问题**：在散射介质中，alpha-blending 的视点相关采样会导致多视图不一致，3D 高斯容易错误地表示水体本身（而非场景物体），产生浮动伪影。

**解决思路**：
- 借鉴多基线立体视觉优于单基线的经验，将双目一致性扩展为**三目一致性**（水平+垂直虚拟视点），提供正交约束
- 利用虚拟视点间的三角测量生成**自监督深度先验**
- 通过**深度感知透明度调整**，在训练早期抑制介质区域的 3D 高斯

## 方法详解

### 整体框架

OceanSplat 基于 3DGS 框架，使用 SfM 初始化 3D 高斯，通过 MLP 建模水下介质属性（衰减、后向散射、介质颜色）。在训练过程中引入四个关键模块：三目视图一致性、合成对极深度先验、深度残差损失和深度感知透明度调整。

水下图像形成模型将观测图像分解为衰减的物体颜色和后向散射：
$$C = C^{obj} \cdot e^{-\sigma^{attn} \cdot z} + C^{\infty} \cdot (1 - e^{-\sigma^{bs} \cdot z})$$

物体和介质的渲染分别按 alpha-blending 累积，支持物体-介质解耦。

### 关键设计

1. **三目视图一致性（Trinocular View Consistency）**

   **核心思路**：从原始相机位姿 $P_c$ 生成水平和垂直两个虚拟视点 $P_h$ 和 $P_v$，强制三视图间的一致性来正则化 3D 高斯的空间位置。

   虚拟视点通过平移构造：
   $$P_h = \begin{bmatrix} \mathbb{I} & \mathbf{t}_h \\ \mathbf{0}^\top & 1 \end{bmatrix} P_c, \quad P_v = \begin{bmatrix} \mathbb{I} & \mathbf{t}_v \\ \mathbf{0}^\top & 1 \end{bmatrix} P_c$$
   其中 $\mathbf{t}_h = (b_h, 0, 0)^\top$, $\mathbf{t}_v = (0, b_v, 0)^\top$。

   从虚拟视点渲染图像后，利用深度图计算视差进行反向变换，将虚拟视点图像对齐到中心视图：
   $$d_h(x,y) = \frac{f_h \cdot b_h}{D_c(x,y)}, \quad d_v(x,y) = \frac{f_v \cdot b_v}{D_c(x,y)}$$

   一致性损失包含三部分：
   - **物体立体一致性**：$L_{obj\text{-}stereo}$，反向变换后的物体图像与中心视图物体图像的 R-L1 损失
   - **完整立体一致性**：$L_{full\text{-}stereo}$，合成完整图像与 GT 的 R-L1 损失
   - **视差平滑性**：$L_{smooth}$，边缘感知的视差正则化

   **设计动机**：单基线立体仅提供一个方向的约束，水平+垂直的正交基线提供了更强的空间约束力，能更好地消除散射介质中的几何模糊性。$b_v$ 从 [-0.4, 0.4] 采样，$b_h = 1.5 b_v$，使用不等长基线增加约束多样性。

2. **合成对极深度先验（Synthetic Epipolar Depth Prior）**

   **核心思路**：利用虚拟视点间的三角测量推导自监督深度先验 $D_{epi}$，无需外部深度监督。

   具体步骤：
   - 选择三目视锥交集内、透明度 > $\tau_\alpha$ 的 3D 高斯
   - 将选中高斯投影到 $P_h$ 和 $P_v$ 的图像平面
   - 通过对极几何建立线性系统 $\mathbf{A}_i \tilde{\mathbf{X}}_i = \mathbf{0}$
   - 最小二乘求解三角测量点，转到中心相机坐标系取 z 分量作为深度先验

   应用边缘感知 Log-L1 损失：
   $$L_{epi} = \frac{1}{HW}\sum_{x,y}\sum_{k}\log(1 + |D_c' - D_{epi}|) \cdot e^{-|\nabla_k I_c|}$$

   **设计动机**：水下场景几何线索有限，外部深度模型可能不准确，利用自身虚拟视点的几何关系提供自洽的深度约束，避免了外部依赖。

3. **深度残差损失（Depth Residual Loss）**

   约束每个 3D 高斯的 z 分量与 alpha-blending 渲染深度一致：
   $$L_{res} = \frac{1}{N'}\sum_{i=1}^{N'}|D_c(\mathbf{x}_i) - z_i|$$

   防止 3D 高斯沿光线过度分散，减少浮动伪影。

4. **深度感知透明度调整（Depth-aware Alpha Adjustment）**

   在训练早期（$t < t_\alpha$），使用 MLP 根据深度和观察方向调整每个 3D 高斯的透明度：
   $$\alpha_i' = (1-w)\alpha_i + w \cdot \phi_\alpha(\alpha_i, z_i, \vec{\mathbf{v}}_i)$$

   过渡步 $t_\alpha$ 之后权重 $w$ 衰减为 0，消除推理开销。

   **设计动机**：散射介质中，被错误放置的 3D 高斯会获取介质颜色的贡献。通过在训练早期抑制这些高斯的透明度，鼓励它们被剪枝掉，从根源上防止介质诱导的伪影。

### 损失函数 / 训练策略

$$L_{total} = L_{photo} + \lambda_{tri} L_{tri} + \lambda_{epi} L_{epi} + \lambda_{res} L_{res}$$

- $L_{photo}$：加权 R-L1 + R-SSIM（$\lambda_s = 0.2$）
- $\lambda_{tri} = 0.1$，$\lambda_{res} = 0.01$
- $\lambda_{epi}$ 从 0.4 退火到 0.2
- 训练步数：SeaThru-NeRF 数据 7K/3K（致密化/微调），In-the-Wild 数据 10K/5K
- 分辨率渐进训练：1/4 → 1/2 → 全分辨率

## 实验关键数据

### 主实验

**真实水下场景（SeaThru-NeRF + In-the-Wild）**：

| 数据集 | 指标 | OceanSplat | WaterSplatting | SeaSplat | 提升 |
|--------|------|------------|----------------|----------|------|
| Curaçao | PSNR | **34.56** | 32.32 | 29.77 | +2.24 |
| Panama | PSNR | **32.74** | 31.71 | 28.65 | +1.03 |
| J.G-Redsea | PSNR | **25.35** | 24.77 | 23.07 | +0.58 |
| IUI3-Redsea | PSNR | **30.17** | 29.84 | 27.23 | +0.33 |
| Coral | PSNR | **29.15** | 28.19 | 28.41 | +0.96 |
| Composite | PSNR | 26.39 | 25.47 | 26.22 | +0.92 |

平均 PSNR 超过 WaterSplatting 1.05 dB，超过 SeaThru-NeRF-NS 2.88 dB。

**模拟散射场景（水下+雾）**：

| 场景 | 指标 | OceanSplat | WaterSplatting | SeaSplat |
|------|------|------------|----------------|----------|
| 水下-NVS | PSNR | **28.80** | 28.12 | 15.62 |
| 雾-NVS | PSNR | **29.12** | 28.45 | 27.52 |
| 水下-恢复 | SSIM | **0.768** | 0.748 | 0.719 |
| 雾-恢复 | SSIM | **0.791** | 0.770 | 0.744 |

### 消融实验

| 配置 | PSNR | SSIM | LPIPS | 说明 |
|------|------|------|-------|------|
| Full Model | **34.56** | **0.961** | **0.113** | 完整模型 |
| w/o $L_{res}$ | 34.30 | 0.960 | 0.115 | 深度残差损失有效 |
| w/o $L_{epi}$ | 33.82 | 0.959 | 0.120 | 对极深度先验贡献显著 |
| w/o $L_{tri}$ | 33.20 | 0.957 | 0.115 | 三目一致性贡献最大（-1.36dB） |
| w/o $\alpha^d$ | 33.90 | 0.960 | 0.116 | 深度感知透明度调整有效 |

效率对比：训练 19 分钟（vs SeaThru-NeRF 18h25m），推理 85.67 FPS，显存 7.6 GB。

### 关键发现

- 三目一致性是最重要的组件（移除后 PSNR 下降 1.36 dB）
- 对极深度先验的贡献排第二（-0.74 dB）
- 深度感知透明度调整在抑制介质伪影方面效果显著
- 所有以上组件为自监督设计，无需外部深度 GT 或标注

## 亮点与洞察

1. **三目扩展的几何动机清晰**：相比双目方法仅有水平约束，增加垂直虚拟视点提供了正交方向的约束，立体几何理论基础扎实
2. **完全自监督的深度正则化**：合成对极深度先验来自模型自身的虚拟视点三角测量，不依赖任何外部深度模型，实现了"自洽"的几何约束
3. **物体-介质解耦**：通过有效的几何约束促进 3D 高斯与散射介质的分离，既改善了重建质量，也支持了场景恢复（去水/去雾）
4. **早期透明度调整的"预防式"策略**：不是等伪影出现再修复，而是在训练初期就抑制可能产生问题的 3D 高斯

## 局限与展望

- 每次迭代需要额外的光栅化（虚拟视点渲染）和最小二乘求解，训练时间略长于 WaterSplatting（19min vs 10min）
- 虚拟视点基线长度 $b_h, b_v$ 为经验值，可能对不同场景尺度敏感
- 目前仅在静态水下场景验证，动态场景（水流、气泡）未涉及
- 散射模型仍为简化模型，未考虑波长相关的复杂散射效应

## 相关工作与启发

- **WaterSplatting (2024)**：隐式介质 + 显式物体的混合方法，本文的主要比较对象和基线
- **SeaSplat (2024)**：将水下物理融入 3D 高斯，但几何约束不足
- **StereoGS (2024, Han et al.)**：双目立体一致性正则化 3DGS，本文将其扩展为三目
- **启发**：虚拟视点构造约束的思路可推广到其他退化场景（雾、烟、粉尘）的 3D 重建

## 评分

- 新颖性: ⭐⭐⭐⭐ （三目扩展动机充分，自监督深度先验设计巧妙）
- 实验充分度: ⭐⭐⭐⭐⭐ （真实+模拟，NVS+恢复，消融完整，效率比较详尽）
- 写作质量: ⭐⭐⭐⭐⭐ （公式推导完整，图示清晰，物理动机解释充分）
- 价值: ⭐⭐⭐⭐ （水下场景重建的重要推进，自监督设计实用性强）

<!-- RELATED:START -->

## 相关论文

- [Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction](splat-sap_feed-forward_gaussian_splatting_for_human-centered_scene_with_scale-aw.md)
- [MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)

<!-- RELATED:END -->
