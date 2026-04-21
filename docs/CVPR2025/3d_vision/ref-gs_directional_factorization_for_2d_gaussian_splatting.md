---
title: >-
  [论文解读] Ref-GS: Directional Factorization for 2D Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][2D高斯泼溅] 提出 Ref-GS，在 2D 高斯泼溅 (2DGS) 中引入延迟渲染和方向分解，通过 Sph-Mip 球形特征网格建模远场光照和表面粗糙度变化，再经紧凑的张量分解实现空间变化的视角相关效果，在反射场景渲染和几何恢复上达到 SOTA 且保持 45+ FPS 实时性能。
tags:
  - CVPR 2025
  - 3D视觉
  - 2D高斯泼溅
  - 反射建模
  - 延迟渲染
  - 方向分解
  - 法线恢复
---

# Ref-GS: Directional Factorization for 2D Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2412.00905](https://arxiv.org/abs/2412.00905)  
**代码**: [项目主页](https://ref-gs.github.io/)  
**领域**: 3D视觉  
**关键词**: 2D高斯泼溅, 反射建模, 延迟渲染, 方向分解, 法线恢复

## 一句话总结

提出 Ref-GS，在 2D 高斯泼溅 (2DGS) 中引入延迟渲染和方向分解，通过 Sph-Mip 球形特征网格建模远场光照和表面粗糙度变化，再经紧凑的张量分解实现空间变化的视角相关效果，在反射场景渲染和几何恢复上达到 SOTA 且保持 45+ FPS 实时性能。

## 研究背景与动机

- **视角相关效果的重要性**：反射和折射是真实感渲染的关键元素，但 NeRF/3DGS 通过球谐函数 (SH) 建模视角依赖颜色，本质上假设每个点独立发射辐射，无法正确处理光线弹射。
- **3DGS 中方向查询的歧义**：Ref-NeRF 用反射方向替代视角方向作为颜色查询，但在 GS 中直接应用存在 SH 系数与基元朝向的歧义——反射方向的变换可被 SH 系数更新所抵消。
- **前向渲染的冗余**：3DGS 的前向渲染对每个基元独立计算光照后投影到屏幕，在高深度复杂度场景中存在大量冗余的着色计算。
- **近场光照建模不足**：现有方法要么用全局环境图假设所有光源无穷远（远场），要么仅建模直接光照，无法处理光源或物体靠近目标表面的近场情况。
- **建模目标**：在保持 2DGS 精确几何重建能力的同时，引入高保真的视角相关效果渲染（反射/折射/高光）。

## 方法详解

### 整体框架

Ref-GS 采用延迟渲染架构，分为几何通道和光照通道。几何通道：2DGS 基元通过 alpha blending 将属性（外观特征 $\mathbf{K}$、粗糙度 $\mathbf{M}$、法线 $\mathbf{N}$）混合到 G-buffer。光照通道：基于 G-buffer 计算反射方向 $\omega_r$，经 Sph-Mip 编码获得方向特征 $\mathbf{s}$。渲染通道：通过张量分解 $\mathbf{s} \circ \mathbf{k}$ 的外积连接空间和方向特征，经 MLP 解码得到最终颜色 $\mathbf{I} = \mathbf{I}_d + f_\Theta(\mathbf{S}, \mathbf{K} \otimes \mathbf{S})$。

### 关键设计

**设计一：延迟高斯着色 (Deferred Gaussian Shading)**
- **功能**：消除前向渲染中基元级方向查询的歧义
- **核心思路**：不在每个基元上独立计算视角依赖颜色，而是先将基元属性（漫反射颜色 $\mathbf{c}_d$、特征 $\mathbf{f}$、粗糙度 $\rho$）通过标准 alpha blending 混合到 G-buffer，得到每个像素的期望属性后再做光照计算。颜色分解为漫反射分量 $\mathbf{I}_d$ + 由 shader $f_\Theta$ 计算的镜面分量。
- **设计动机**：前向渲染中每个基元独立查询反射方向，导致 SH 系数和朝向之间的歧义；延迟渲染在混合后的表面上查询，消除了这种歧义（如 Fig.3(c) 所示）。

**设计二：Sph-Mip 球形多级特征网格**
- **功能**：建模远场高频光照并感知表面粗糙度
- **核心思路**：在球面上分布特征点，用经纬格展开为 2D 特征网格。反射方向 $\omega_r$ 转换为球面坐标 $(\theta, \phi)$，与粗糙度 $\rho$ 一起在 (θ, ϕ, ρ) 三维格上做三线性插值得到方向特征 $\mathbf{s} = \text{Sph-Mip}(\omega_r, \rho, \mathcal{M})$。多级 mipmap 结构中，基级 $\mathcal{M}^{L_0}$ 分辨率最高，后续层级分辨率逐级减半，粗糙度越高使用越低级别的特征。
- **设计动机**：SH 无法表达高频环境光照；mipmap 结构天然对应粗糙度的物理含义——光滑表面产生锐利反射（高分辨率），粗糙表面产生模糊反射（低分辨率）。

**设计三：张量分解的空间-方向因式分解**
- **功能**：高效表示空间变化的视角相关效果
- **核心思路**：空间特征 $\mathbf{k} \in \mathbb{R}^D$ 和方向特征 $\mathbf{s} \in \mathbb{R}^C$ 通过向量外积生成 $D \times C$ 的矩阵，展平后送入轻量 MLP 解码最终颜色。受 TensoRF 启发的低秩张量分解：$\mathbf{I} = \mathbf{I}_d + f_\Theta(\mathbf{S}, \mathbf{K} \otimes \mathbf{S})$。
- **设计动机**：外积分解将几何和光照解耦为独立的向量，避免了在每个基元上存储高维特征（降低体积渲染开销），同时保留了空间变化的材质属性与方向变化的光照的交互。

### 损失函数

总损失 $\mathcal{L} = \mathcal{L}_{\text{color}} + \lambda_n \mathcal{L}_{\text{normal}} + \lambda_d \mathcal{L}_{\text{depth}}$，其中包含 L1 + D-SSIM 的颜色重建损失、法线一致性正则化和深度正则化。

## 实验关键数据

### 主实验：Shiny Blender 数据集 PSNR↑

| 方法 | Car | Ball | Helmet | Toaster | Avg. |
|------|-----|------|--------|---------|------|
| Ref-NeRF | 30.41 | 29.14 | 29.92 | 25.29 | 32.32 |
| 3DGS | 27.24 | 27.69 | 28.32 | 20.99 | 30.37 |
| GaussianShader | 27.51 | 29.02 | 28.73 | 22.86 | 30.42 |
| 3DGS-DR | 30.43 | 33.44 | 31.49 | 26.69 | 33.94 |
| **Ref-GS** | **30.94** | **36.10** | **33.40** | **27.28** | **34.80** |

### Shiny Real 数据集

| 方法 | Garden | Sedan | Toycar | Avg. |
|------|--------|-------|--------|------|
| Ref-NeRF | 22.01 | 25.21 | 23.65 | 23.62 |
| 3DGS | 21.75 | 26.03 | 23.78 | 23.85 |
| 3DGS-DR | 21.52 | 26.32 | 23.57 | 23.80 |
| **Ref-GS** | **22.48** | **26.63** | **24.20** | **24.44** |

### 关键发现

1. Ref-GS 在 Shiny Blender 上平均 PSNR 达 34.80，超过 3DGS-DR 的 33.94，且超过隐式方法 ENVIDR (32.88) 和 Ref-NeRF (32.32)
2. 渲染速度 > 45 FPS @ 800×800 分辨率，保持实时性能
3. 延迟渲染有效消除了方向查询歧义——Garden 场景下成功重建了反射桌面的几何，现有方法失败
4. 法线恢复质量显著优于现有 GS 方法，尤其在高光物体（Toaster、Bell）上表现突出

## 亮点与洞察

- **从计算机图形学借鉴延迟渲染**思路解决了 GS 特有的方向歧义问题，简洁有效
- **Sph-Mip 网格**优雅地将粗糙度建模为 mipmap 级别选择，物理直觉清晰
- **外积分解**同时降低了每基元的特征维度和体积渲染开销，是一举两得的设计
- 在同时追求 SOTA 渲染质量和精确几何恢复方面取得了很好的平衡

## 局限与展望

- Sph-Mip 主要建模远场光照，对复杂近场光照（如物体间互反射、自遮挡造成的间接反射）仍有局限
- 延迟渲染假设每个像素对应单一表面，对半透明/多层折射材质处理有限
- 需要已知相机参数和 SfM 初始化，无法处理无位姿输入的场景
- 球面 Mip-grid 是全局共享的单一环境图，无法为场景不同区域学习独立的局部环境光照
- 在大规模户外场景中的泛化能力有待进一步验证

## 相关工作与启发

- **vs GaussianShader**: GaussianShader 分别建模视角相关效果，但仍在基元级别操作，受方向歧义困扰。Ref-GS 通过延迟到像素级别从根本上解决了这个问题
- **vs 3DGS-DR**: 3DGS-DR 也引入延迟渲染做反射建模，但 Ref-GS 加入了 Sph-Mip 编码和张量分解，在高频反射建模质量和近场光照处理上更优
- **vs Ref-NeRF**: Ref-NeRF 使用积分方向编码，在连续 NeRF 表示中效果好但训练和渲染慢。Ref-GS 将类似思想迁移到 GS 并通过延迟着色解决了离散表示的歧义问题，同时保持 >45 FPS 实时渲染
- Ref-GS 的延迟渲染 + G-buffer 设计思路可推广到其他需要材质分解的 GS 应用（如场景编辑、重光照）

## 评分

⭐⭐⭐⭐ — 优雅地将计算机图形学的延迟渲染引入 2DGS，同时实现了高质量反射渲染和精确几何恢复，Sph-Mip 和张量分解设计都有独立贡献价值。

<!-- RELATED:START -->

## 相关论文

- [Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)
- [IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)
- [Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)
- [HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)
- [POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality](pop-gs_next_best_view_in_3d-gaussian_splatting_with_p-optimality.md)

<!-- RELATED:END -->
