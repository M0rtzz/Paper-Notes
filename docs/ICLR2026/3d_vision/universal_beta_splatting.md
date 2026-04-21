---
title: >-
  [论文解读] Universal Beta Splatting
description: >-
  [3D视觉] 提出 Universal Beta Splatting (UBS)，将 3D 高斯 Splatting 推广为 N 维各向异性 Beta 核，通过逐维度形状控制在单一表示中统一建模空间几何、视角依赖外观和场景动态，实现了可解释的场景分解和 SOTA 渲染质量。
tags:
  - 3D视觉
---

# Universal Beta Splatting

- **会议**: ICLR 2026
- **arXiv**: [2510.03312](https://arxiv.org/abs/2510.03312)
- **代码**: [项目页面](https://rongliu-leo.github.io/universal-beta-splatting/)
- **领域**: 3D 视觉 / 神经渲染
- **关键词**: 3D Gaussian Splatting, Beta Kernel, N-Dimensional, View-Dependent, Dynamic Scene, Real-Time Rendering

## 一句话总结

提出 Universal Beta Splatting (UBS)，将 3D 高斯 Splatting 推广为 N 维各向异性 Beta 核，通过逐维度形状控制在单一表示中统一建模空间几何、视角依赖外观和场景动态，实现了可解释的场景分解和 SOTA 渲染质量。

## 研究背景与动机

3D 高斯 Splatting (3DGS) 通过显式基元实现了实时渲染，但高斯核固定的钟形轮廓存在根本限制：

**空间维度**：锐利边界需要大量小基元，效率低

**角度维度**：视角依赖效果需要额外的球谐编码（48参数），碎片化表示

**时间维度**：动态场景需要额外变形网络，增加复杂度

**核心洞察**：不同场景性质需要不同的核行为——空间几何需要自适应锐度，角度外观从漫反射到镜面反射不等，时间动态从静态到快速运动。高斯核在所有维度强制相同的对称轮廓，而 Beta 核可以提供逐维度的形状控制。

## 方法详解

### N 维 Beta 核

核心密度函数：

$$\sigma(\mathbf{x}, \mathbf{q}) = \mathcal{B}(\mathbf{x}, \mathbf{q}; \boldsymbol{\mu}, \boldsymbol{\Sigma}, \mathbf{b}) \cdot o$$

其中 $\mathbf{x} \in \mathbb{R}^3$ 为空间坐标，$\mathbf{q} \in \mathbb{R}^{N-3}$ 编码额外维度（视角/时间），$\mathbf{b} \in \mathbb{R}^{N-2}$ 控制各维度的 Beta 形状参数。每个维度的 Beta 指数 $\beta_i = 4\exp(b_i)$：
- **负 $b_i$**：平坦轮廓（适合光滑表面、静态元素、漫反射）
- **正 $b_i$**：尖锐峰值（适合精细纹理、快速运动、镜面反射）

### 空间正交 Cholesky 参数化

协方差矩阵分解：

$$\mathbf{L} = \begin{pmatrix} \mathbf{R}_x \text{diag}(\mathbf{s}_x) & \mathbf{0} \\ \mathbf{L}_{qx} & \mathbf{L}_q \end{pmatrix}$$

- $\mathbf{R}_x \in SO(3)$：保持空间正交结构（一阶 Taylor 近似）
- $\mathbf{L}_{qx}$：编码跨维度相关性
- 保证向后兼容 3DGS 的旋转-缩放参数化

### Beta 调制条件切片

**条件均值和协方差**：

$$\boldsymbol{\mu}_{x|q} = \boldsymbol{\mu}_x + \boldsymbol{\Sigma}_{xq} \boldsymbol{\Sigma}_q^{-1} \text{diag}(\tilde{\boldsymbol{\beta}}_q) (\mathbf{q} - \boldsymbol{\mu}_q)$$

$$\boldsymbol{\Sigma}_{x|q} = \boldsymbol{\Sigma}_x - \boldsymbol{\Sigma}_{xq} \boldsymbol{\Sigma}_q^{-1} \text{diag}(\tilde{\boldsymbol{\beta}}_q) \boldsymbol{\Sigma}_{qx}$$

其中 $\text{diag}(\tilde{\boldsymbol{\beta}}_q)$ 对非空间维度施加 Beta 调制。

**Beta 调制不透明度**：

$$o(\mathbf{q}) = o \prod_{i=1}^C (1 - d_i)^{4\beta_{q_i}}$$

$d_i = \tanh(d_i^{raw}) \in [0,1)$，逐维度的 Mahalanobis 距离映射到有界值。

### 通用兼容性

| $\mathbf{b}$ 设置 | 等价方法 |
|---------|---------|
| $N=3$, $b_x=0$ | ≈ 3DGS |
| $N=3$, $b_x \neq 0$ | ≈ DBS |
| $N=6$, $\mathbf{b}=\mathbf{0}$ | ≈ 6DGS |
| $N=7$, $\mathbf{b}=\mathbf{0}$ | ≈ 7DGS |

### 可解释场景分解

学习到的 Beta 参数自然提供无监督场景分解：
- **空间 $b_x$**：负 → 光滑表面；正 → 精细纹理
- **角度 $b_d$**：负 → 漫反射；正 → 镜面反射
- **时间 $b_t$**：负 → 静态元素；正 → 动态元素

### 损失函数

$$\mathcal{L} = (1-\lambda_{SSIM})\mathcal{L}_1 + \lambda_{SSIM}\mathcal{L}_{SSIM} + \lambda_o \sum_i |o_i| + \lambda_\Sigma \sum_i \|\mathbf{s}_i\|_1$$

不透明度正则化确保 MCMC 致密化有效，尺度惩罚促进基元重定位。

### 参数效率

- 静态场景：比 3DGS 减少 **41%** 参数（无需 48 参数球谐）
- 动态场景：比 4DGS 减少 **73%** 参数

## 实验

### 静态场景

**NeRF Synthetic**（UBS-6D vs 3DGS vs 6DGS）：

| 场景 | 3DGS PSNR | 6DGS PSNR | UBS-6D PSNR |
|------|-----------|-----------|-------------|
| chair | 35.60 | 35.55 | **36.72** |
| ficus | 35.49 | 34.62 | **36.90** |
| materials | 30.50 | 30.63 | **32.90** |
| lego | 36.06 | 35.22 | **36.95** |

PSNR 提升最高达 **+8.27 dB**（6DGS-PBR 数据集）。

### 动态场景

**D-NeRF 和 7DGS-PBR**（UBS-7D vs 4DGS vs 7DGS）：
- PSNR 提升最高达 **+2.78 dB**
- 在复杂时空角度关联场景（心脏运动、日光变化、半透明变形）上优势明显

### 关键发现

1. **Beta 参数初始化为零**保证从高斯极限开始收敛
2. 空间正交 Cholesky 的一阶近似与精确旋转精度相当，计算更快
3. MCMC 优化策略对 Beta 核同样有效
4. 实时渲染性能与 3DGS 相当

### 训练效率

- 30K 迭代
- 静态：单张 RTX 4090，约 8-10 分钟
- 动态：单张 V100，与 4DGS/7DGS 基线一致

## 亮点

1. **统一框架**：单一 Beta 核基元同时处理空间/角度/时间，取代多种专用编码
2. **向后兼容**：Beta=0 时退化为高斯，保证性能下界
3. **无监督分解**：学习到的 Beta 参数自然分离几何、外观和运动
4. **大幅减参**：41-73% 参数减少，同时性能提升
5. **实时渲染**：完整 CUDA 加速实现

## 局限性

1. N 维 Cholesky 参数化的参数数量随维度增长
2. Beta 核的有界支撑在极远场可能需要更多基元
3. 目前仅验证到 7 维，更高维度的效果待考
4. 训练时需要 batch size 4 处理动态场景（显存消耗较大）
5. 一阶旋转近似在极端旋转角度下可能不够精确

## 相关工作

- **核设计替代**：GES, TNT-GS, Half-GS, Disc-GS 修改 3D 高斯；DBS 引入 3D Beta 核
- **高维高斯**：6DGS（空间+视角）、7DGS（空间+视角+时间）通过条件分布建模
- **动态场景**：D-NeRF 用变形场；4DGS 直接扩展时间维度
- **替代基元**：3D Convex Splatting, Triangle Splatting 等

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — N 维 Beta 核统一框架是优雅的理论贡献
- **实用性**: ⭐⭐⭐⭐⭐ — 即插即用兼容性 + 实时渲染
- **清晰度**: ⭐⭐⭐⭐ — 数学推导清晰，但符号较多
- **意义**: ⭐⭐⭐⭐⭐ — 为辐射场渲染建立了通用基元框架

<!-- RELATED:START -->

## 相关论文

- [DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers](../../CVPR2025/3d_vision/dune_distilling_a_universal_encoder_from_heterogeneous_2d_and_3d_teachers.md)
- [SplatTalk: 3D VQA with Gaussian Splatting](../../ICCV2025/3d_vision/splattalk_3d_vqa_with_gaussian_splatting.md)
- [UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)
- [UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)
- [Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)

<!-- RELATED:END -->
