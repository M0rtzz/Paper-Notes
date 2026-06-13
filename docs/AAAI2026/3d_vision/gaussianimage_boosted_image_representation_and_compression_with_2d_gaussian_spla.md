---
title: >-
  [论文解读] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting
description: >-
  [AAAI 2026][3D视觉][2D Gaussian Splatting] 提出 GaussianImage++，通过失真驱动的密度化机制和内容感知高斯滤波器，在有限的2D高斯原语下实现高质量图像表示与压缩，同时保持实时解码速度。 领域现状 图像表示与压缩是视觉数据存储和传输的核心问题。当前主流方案包括： - 基于自编…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "2D Gaussian Splatting"
  - "图像压缩"
  - "隐式神经表示"
  - "密度化机制"
  - "量化感知训练"
---

# GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting

**会议**: AAAI 2026  
**arXiv**: [2512.19108](https://arxiv.org/abs/2512.19108)  
**代码**: [https://github.com/Sweethyh/GaussianImage_plus](https://github.com/Sweethyh/GaussianImage_plus)  
**领域**: 3D视觉 / 图像表示与压缩  
**关键词**: 2D Gaussian Splatting, 图像压缩, 隐式神经表示, 密度化机制, 量化感知训练

## 一句话总结

提出 GaussianImage++，通过失真驱动的密度化机制和内容感知高斯滤波器，在有限的2D高斯原语下实现高质量图像表示与压缩，同时保持实时解码速度。

## 研究背景与动机

### 领域现状

图像表示与压缩是视觉数据存储和传输的核心问题。当前主流方案包括：
- **基于自编码器的神经压缩**（如 Ballé18, ELIC）：率失真性能优秀，但解码延迟高
- **隐式神经表示（INR）**（如 SIREN, COIN）：用MLP拟合像素坐标到颜色的映射，但训练慢、内存大
- **2D高斯泼溅（GS）**：GaussianImage 首次将GS用于2D图像，显著降低了训练时间和内存

### 现有痛点

1. **GaussianImage 缺乏密度化机制**：无法根据图像内容自适应分配高斯原语，导致欠重建区域大量存在
2. **Mirage 使用3D GS的ADC**：容易导致高斯数量不可控增长，产生OOM错误
3. **LIG 没有压缩**：专注于拟合大图像但不探索属性压缩，存储开销大
4. **3D GS压缩方法不可直接迁移**：HAC、ContextGS基于neural Gaussian（Scaffold），架构上与显式2D GS不匹配

### 核心矛盾

如何在**有限数量的2D高斯原语**下同时实现高视觉保真度和高效压缩？

### 本文切入角度

从三个维度增强2D GS：(1) 渐进式失真驱动密度化控制高斯分布；(2) 内容感知滤波器优化高斯渲染质量；(3) 属性分离的可学习标量量化实现高效压缩。

## 方法详解

### 整体框架

GaussianImage++ 的流程分两大阶段：
1. **图像表示**：稀疏初始化 → 周期性失真驱动密度化 → 内容感知滤波 → 累积和光栅化
2. **图像压缩**：先过拟合高斯属性 → 量化感知训练微调 → 编码为紧凑比特流

每个2D高斯由位置 $\boldsymbol{\mu} \in \mathbb{R}^2$、协方差 $\boldsymbol{\Sigma} \in \mathbb{R}^{2 \times 2}$、颜色 $\mathbf{c} \in \mathbb{R}^3$ 参数化。渲染公式为：

$$G_i(\mathbf{x}) = \exp\left(-\frac{(\mathbf{x}-\boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}_i)}{2}\right)$$

$$\mathbf{C} = \sum_{i \in N} \mathbf{c}_i G_i(\mathbf{x})$$

### 关键设计

#### 1. 失真驱动密度化（D³）

**功能**：渐进式地将高斯原语分配到欠重建区域。

**核心思路**：三阶段机制：

- **稀疏初始化**：初始数量 $N_0 = M/2$（M为最大高斯数），位置在图像坐标内均匀随机采样，颜色初始化为零
- **高斯生长**：每5000次迭代，在重建失真最大的top-k像素位置添加新高斯，数量由调度器 $\tau(t, N_t, M) = (M - N_t)/2$ 决定
- **高斯修剪**：每100次迭代检查协方差矩阵的半正定性，剪除无效高斯

**设计动机**：3D GS的ADC依赖位置梯度，但在2D场景中梯度变化太小无法有效触发。本文直接用像素级失真（L1 loss）决定密度化位置，更直接且面向图像质量。新高斯的位置和颜色直接从原图高失真像素获取：

$$\boldsymbol{\mu}_\Psi = \xi(\text{Top}_k(D(X, \hat{X})))$$
$$\mathbf{c}_\Psi = X(\xi(\text{Top}_k(D(X, \hat{X}))))$$

#### 2. 内容感知高斯滤波器（CAF）

**功能**：为每个高斯原语施加自适应强度的低通滤波，减少渲染空洞和伪影。

**核心思路**：对原始高斯核施加零均值高斯低通滤波器 $h(x)$，方差向量 $\mathbf{s} \in \mathbb{R}^{N_t}$ 控制每个高斯的滤波强度：

$$G_i'(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T(\boldsymbol{\Sigma}_i + s_i I)^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)}$$

方差公式：
$$s_i = \frac{HW}{\alpha N_t} \quad (\text{新加入的高斯})$$

**设计动机**：训练早期高斯稀疏时（$N_t \ll HW$），大方差滤波器扩大覆盖面积、减少空洞，产生粗糙但可识别的图像引导优化。随密度化推进，新高斯的滤波强度逐渐减小，聚焦细节。关键是 $\mathbf{s}$ 不增加存储——直接存储滤波后的协方差 $\boldsymbol{\Sigma} + sI$。

#### 3. 压缩框架（属性分离量化）

**功能**：用可学习标量量化器（LSQ+）对不同属性施加不同比特深度的量化。

**核心思路**：
- 位置 $\boldsymbol{\mu}$：12-bit（几何敏感，需高精度）
- 协方差 $\boldsymbol{\Sigma}$：10-bit
- 颜色 $\mathbf{c}$：6-bit

量化公式：
$$\bar{\mathbf{v}} = \lfloor \text{clip}(\frac{\mathbf{v} - \beta}{s}, 0, 2^b - 1) \rfloor, \quad \hat{\mathbf{v}} = \bar{\mathbf{v}} \cdot s + \beta$$

**设计动机**：量化感知训练（QAT）使高斯能主动调整属性以适应量化误差。与FP16或RVQ相比，LSQ+的可学习offset和scale能实现更好的率失真平衡。

### 损失函数 / 训练策略

- 表示阶段：L2 loss，Adam优化器，50000次迭代，学习率0.18（20000次后减半）
- 压缩阶段：6000次warm-up后进行量化感知微调，量化器学习率0.001

## 实验关键数据

### 主实验

#### 图像表示（Kodak，10k高斯）

| 方法 | PSNR↑ | MS-SSIM↑ | 参数量(M) | GPU内存(MiB) | 渲染FPS |
|------|-------|----------|-----------|-------------|---------|
| Siren (INR) | 26.50 | 0.875 | 3.74 | 2044 | 977 |
| GaussianImage | 32.48 | 0.982 | 0.08 | 814 | 2009 |
| LIG | 31.00 | 0.975 | 0.08 | 832 | 1331 |
| **Ours** | **35.41** | **0.983** | 0.08 | 876 | **2216** |

#### 图像压缩（Kodak，低/高bpp）

| 方法 | Bpp | PSNR | 解码FPS |
|------|-----|------|---------|
| JPEG | 0.22/1.03 | 23.8/32.8 | 377/148 |
| COIN | 0.17/0.98 | 24.9/27.4 | 769/344 |
| GaussianImage | 0.15/1.00 | 25.0/29.7 | 1827/1822 |
| **Ours** | 0.15/1.08 | **25.3/31.1** | **1839/1666** |

### 消融实验

#### 组件消融（Kodak）

| 配置 | PSNR提升（vs GS Cholesky） | 说明 |
|------|--------------------------|------|
| + D³ alone | ~2dB | 密度化单独贡献最大 |
| + D³ + CAF | ~3dB | 两者协同进一步提升 |
| vs LIG | ~4dB | 综合提升显著 |

#### 量化策略消融

| 配置（位置/颜色） | BD-PSNR(dB) | BD-Rate(%) |
|------------------|-------------|------------|
| LSQ+/LSQ+（本文） | 0 | 0 |
| FP16/LSQ+ | -0.761 | +25.11% |
| FP16/RVQ | -2.471 | +138.88% |
| LSQ+/RVQ | -2.491 | +147.24% |

### 关键发现

1. D³密度化在高斯数较少时效果尤其显著，因为稀疏高斯更需要精准分配
2. CAF在训练早期的作用至关重要——在t=500时就能产生可识别的粗糙图像（而baseline有大量空洞）
3. 两个组件对三种不同的协方差参数化方式（Cholesky、RS、直接参数化）都有效，具有通用性
4. GS方法的解码速度远超传统和学习型编解码器（>1800 FPS vs JPEG的~150 FPS）

## 亮点与洞察

1. **失真驱动的密度化非常直觉**：直接在最"差"的像素位置放新高斯，简单有效
2. **CAF的渐进减弱策略精巧**：早期放大覆盖→后期精细化，与密度化形成天然协同
3. **通用增强技术**：D³和CAF可以作为即插即用模块应用于其他2D GS方法
4. **实时解码优势明显**：相比VAE和INR的解码延迟，GS的简单累积求和具有本质速度优势

## 局限与展望

1. **高比特率下仍落后于SOTA神经编解码器**：这是当前2D GS压缩的共性问题，需要更先进的熵模型
2. **编码时间远非实时**：训练和量化过程耗时长，制约实际部署
3. **缺乏自适应比特分配**：当前对所有图像用相同量化配置，未根据图像复杂度调整
4. 可探索将D³和CAF扩展到视频GS场景

## 相关工作与启发

- **GaussianImage** (Zhang et al., 2024)：首个2D GS图像表示，本文的直接基础
- **3D GS ADC** (Kerbl et al., 2023)：基于位置梯度的密度控制，启发了D³但机制不同
- **LSQ+** (Bhalgat et al., 2020)：带可学习offset/scale的低比特量化，本文压缩的核心工具
- **COOL-CHIC** (Ladune et al., 2023)：混合INR压缩方法，需要自回归熵模型增加解码开销

## 评分

- 新颖性: ⭐⭐⭐⭐ — D³和CAF的设计简洁有效，但核心思路（在高失真处加高斯）较为直觉
- 实验充分度: ⭐⭐⭐⭐⭐ — 双数据集、多baseline、跨方法消融、量化策略消融均覆盖
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机论述充分
- 价值: ⭐⭐⭐⭐ — 作为通用增强技术有实用价值，但与SOTA编解码器的差距限制了应用场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](../../ECCV2024/3d_vision/gaussianimage_1000_fps_image_representation_and_compression_by_2d_gaussian_splat.md)
- [\[AAAI 2026\] SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images](smartsplat_feature-smart_gaussians_for_scalable_compression_of_ultra-high-resolu.md)
- [\[AAAI 2026\] Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space](split-layer_enhancing_implicit_neural_representation_by_maximizing_the_dimension.md)
- [\[CVPR 2026\] SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation](../../CVPR2026/3d_vision/sgi_structured_2d_gaussians_for_efficient_and_compact_large_image_representation.md)
- [\[AAAI 2026\] Point-SRA: Self-Representation Alignment for 3D Representation Learning](point-sra_self-representation_alignment_for_3d_representation_learning.md)

</div>

<!-- RELATED:END -->
