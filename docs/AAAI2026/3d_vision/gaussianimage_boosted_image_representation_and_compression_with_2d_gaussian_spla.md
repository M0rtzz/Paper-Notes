---
title: >-
  [论文解读] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting
description: >-
  [AAAI 2026][3D视觉][2D Gaussian Splatting] 提出 GaussianImage++，通过失真驱动的密度化机制和内容感知高斯滤波器，在有限的2D高斯原语下实现高质量图像表示与压缩，同时保持实时解码速度。
tags:
  - AAAI 2026
  - 3D视觉
  - 2D Gaussian Splatting
  - 图像压缩
  - 隐式神经表示
  - 密度化机制
  - 量化感知训练
---

# GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting

**会议**: AAAI 2026  
**arXiv**: [2512.19108](https://arxiv.org/abs/2512.19108)  
**代码**: [GitHub](https://github.com/Sweethyh/GaussianImage_plus.git)  
**领域**: 3D视觉  
**关键词**: 2D Gaussian Splatting, 图像表示, 图像压缩, 密度控制, 量化感知训练

## 一句话总结

提出 GaussianImage++，通过失真驱动的密度化机制和内容感知高斯滤波器，用有限数量的 2D Gaussian 原语实现高质量图像表示和压缩，并结合属性分离的可学习标量量化器实现高效压缩。

## 研究背景与动机

### 领域现状

神经图像表示与压缩是视觉数据存储、传输和渲染的关键技术。隐式神经表示（INR）方法如 SIREN、COIN 等基于轻量 MLP 实现了不错的视觉保真度，但训练时间长、内存开销大。近年来，3D Gaussian Splatting（3DGS）因其显式原语表示的高效性而受到广泛关注，GaussianImage 首次将 GS 应用于 2D 图像表示与压缩，显著降低了内存和训练时间。

### 现有痛点

**GaussianImage 缺乏密度化机制**：不能根据图像内容自适应地控制 2D 高斯原语的数量，导致表示能力未被充分利用，拟合效果受限

**Mirage 使用 3D GS 的 ADC 进行密度控制**：容易导致高斯数量不可控增长，引发训练时的 OOM 错误

**LIG 关注大图像拟合**：使用大量 2D 高斯但没有探索紧凑压缩，导致存储开销大

**3D GS 压缩方法（HAC, ContextGS）** 基于 neural Gaussians（Scaffold），架构不匹配，无法直接适用于 2D GS

### 核心矛盾

如何在**有限数量的 2D 高斯原语**下实现**高质量图像表示**，同时实现**高效压缩**和**实时解码**？

### 切入角度

从两个互补方向入手：(1) 渐进式失真驱动密度化，让高斯原语**按需分配**到重建质量差的区域；(2) 内容感知高斯低通滤波器，在训练早期用强滤波器填补稀疏高斯之间的"空洞"，后期逐步减弱以保留细节。

## 方法详解

### 整体框架

GaussianImage++ 建立在 2D Gaussian Splatting 之上，每个高斯原语由位置 $\boldsymbol{\mu} \in \mathbb{R}^2$、协方差 $\boldsymbol{\Sigma} \in \mathbb{R}^{2 \times 2}$ 和颜色 $\mathbf{c} \in \mathbb{R}^3$ 参数化。渲染公式为累积求和而非 alpha 混合：

$$\mathbf{C} = \sum_{i \in N} \mathbf{c}_i G_i(\mathbf{x})$$

框架分为两个管线：**表示管线**（密度化 + 滤波 + 光栅化）和**压缩管线**（过拟合初始化 + 量化感知训练）。

### 关键设计

#### 1. 失真驱动密度化 (D³)

**功能**：渐进式地将高斯原语分配到图像中重建质量差的区域。

**核心思路**：分三个阶段——

- **稀疏初始化**：以 $N_0 = M/2$ 个高斯开始，在图像坐标内均匀随机采样位置，加速早期训练
- **高斯增长**：每隔 5000 次迭代，计算渲染图与原图的逐像素失真 $D(X, \hat{X})$，在 Top-k 失真最大的像素处插入新高斯原语：

$$\boldsymbol{\mu}_\Psi = \xi(\text{Top}_k(D(X, \hat{X})))$$

$$\mathbf{c}_\Psi = X(\xi(\text{Top}_k(D(X, \hat{X}))))$$

其中 $k = \tau(t, N_t, M) = (M - N_t)/2$，即每次分配剩余配额的一半

- **高斯剪枝**：每 100 次迭代检查协方差矩阵 $\Sigma$ 的半正定性，移除无效高斯

**设计动机**：3D GS 的 ADC 基于位置梯度，在 2D 场景中梯度变化通常太小无法触发。本方法直接基于像素级失真决定密度化位置，更直观且面向图像质量。

#### 2. 内容感知高斯滤波器 (CAF)

**功能**：为每个高斯原语分配自适应方差的低通滤波器，调节其覆盖范围。

**核心思路**：对高斯足迹函数施加零均值高斯低通滤波 $h(x)$：

$$G_i'(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T(\boldsymbol{\Sigma}_i + \mathbf{s}I)^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)}$$

每个高斯的滤波器方差 $s_i$ 按时间调度递减：

$$s_i = \frac{HW}{\alpha N_t}$$

早期高斯的 $s_i$ 大（强滤波，扩大覆盖面积减少"空洞"），后期新增高斯的 $s_i$ 小（弱滤波，关注细节）。

**设计动机**：训练早期 $N_t \ll HW$，很多像素没有任何高斯覆盖，出现明显伪影。大的 $s_i$ 使少量高斯也能覆盖整个图像，产生粗略但可辨识的重建，为后续优化提供良好的初始指导。存储时保存滤波后的 $\Sigma + sI$，不增加额外存储开销。

#### 3. 压缩框架：属性分离 LSQ+ 量化

**功能**：对过拟合的高斯属性进行不同精度的量化压缩。

**核心思路**：采用 LSQ+（可学习标量量化器），对不同属性用不同位深：

- 位置 $\mu$：12-bit（对几何敏感）
- 协方差 $\Sigma$：10-bit
- 颜色 $c$：6-bit

量化公式（带直通梯度）：

$$\bar{\mathbf{v}} = \lfloor \text{clip}(\frac{\mathbf{v}-\beta}{s}, 0, 2^b-1) \rfloor, \quad \hat{\mathbf{v}} = \bar{\mathbf{v}} \cdot s + \beta$$

先做 6000 步 warm-up 训练，再进行量化感知微调，使高斯属性适应量化误差。

### 损失函数 / 训练策略

- 表示任务：L2 loss，Adam 优化器，学习率 0.18（20k 迭代后减半），50k 迭代
- 压缩任务：先 warm-up 6000 步，再量化感知训练，量化器学习率 0.001
- 每 5000 步进行一次密度化，每 100 步检查协方差有效性并剪枝

## 实验关键数据

### 主实验（图像表示）

| 数据集 | 方法 | PSNR (5k GS) | PSNR (10k GS) | 渲染 FPS |
|--------|------|-------------|---------------|---------|
| Kodak | GaussianImage | 29.85 | 32.48 | 2009 |
| Kodak | GaussianImage++ | **31.83** | **35.41** | 2216 |
| DIV2K | GaussianImage | 26.54 | 31.45 | 662 |
| DIV2K | GaussianImage++ | **28.14** | **33.75** | 765 |
| Kodak | Siren (INR) | 26.50 | - | 977 |

关键发现：在 Kodak 上 10k 高斯下，GaussianImage++ 比 GaussianImage PSNR 高出约 3dB；比 LIG 高出约 4dB。渲染速度不降反升。

### 消融实验

| 配置 | BD-PSNR (dB)↑ | BD-Rate (%)↓ | 说明 |
|------|-------------|-------------|------|
| LSQ+/LSQ+ (Ours) | 0 | 0 | 最终策略 |
| FP16/LSQ+ | -0.761 | +25.11 | 位置用 FP16 |
| FP16/RVQ | -2.471 | +138.88 | 颜色用 RVQ |
| LSQ+/RVQ | -2.491 | +147.24 | 颜色用 RVQ 最差 |

D³ 组件单独可带来约 2dB PSNR 提升；D³ + CAF 组合可达 3dB（vs GaussianImage）和 4dB（vs LIG）。两个组件在三种不同的协方差参数化方法（Cholesky、RS、直接参数化）上均有效。

### 关键发现

1. D³ 是最关键的组件，尤其在高斯数量较少时收益最大
2. CAF 配合 D³ 实现进一步提升，特别是在训练早期帮助稳定优化
3. RVQ 用于颜色量化效果很差，说明 codebook 表达能力不足
4. 压缩性能在低比特率（0.1-0.7bpp）超过 JPEG，但在高比特率落后于学习式编解码器
5. 解码速度（>1600 FPS）远超所有基于 VAE 和 INR 的方法

## 亮点与洞察

1. **"稀疏到密集"的渐进训练思路**：不是一次性初始化所有高斯，而是从 M/2 开始逐步增长，既降低了早期训练开销，又通过失真引导将资源分配到最需要的区域
2. **滤波器方差的自适应调度**：巧妙利用低通滤波器的"扩大覆盖"效应来弥补早期稀疏高斯的不足，且不增加存储开销
3. **两个组件的通用性**：D³ 和 CAF 可作为即插即用模块提升其他 2D GS 方法的性能
4. **实时解码**：2D GS 天然支持并行渲染，解码速度远优于需要自回归熵解码的传统方法

## 局限与展望

1. 在高比特率下仍落后于 SOTA 学习式编解码器（如 Ballé18），需要更先进的属性编码和熵模型
2. 编码时间远未达到实时，训练和量化过程亟需优化
3. 未探索与更先进的熵编码（如算术编码）的结合
4. 仅在 Kodak 和 DIV2K 上评估，缺乏对更多分辨率和内容类型的验证

## 相关工作与启发

- **GaussianImage**：2D GS 图像表示的开创性工作，本文的直接改进对象
- **LIG**：大图像拟合但无压缩，本文的 D³ 可作为替代其分层方案的更优选择
- **3DGS ADC**：3D 场景中的自适应密度控制，本文分析了其在 2D 场景不适用的原因并提出替代方案
- **启发**：2D GS 在图像压缩领域还有很大发展空间，尤其在结合更好的熵模型和多分辨率表示方面

## 评分

- 新颖性: ⭐⭐⭐⭐ — D³ 和 CAF 思路清晰但不算非常新颖
- 实验充分度: ⭐⭐⭐⭐ — 多数据集、多配置消融较充分
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ — 为 2D GS 图像表示/压缩提供了通用增强模块
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

## 相关论文

- [GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](../../ECCV2024/3d_vision/gaussianimage_1000_fps_image_representation_and_compression_by_2d_gaussian_splat.md)
- [SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images](smartsplat_feature-smart_gaussians_for_scalable_compression_of_ultra-high-resolu.md)
- [Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space](split-layer_enhancing_implicit_neural_representation_by_maximizing_the_dimension.md)
- [SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation](../../CVPR2026/3d_vision/sgi_structured_2d_gaussians_for_efficient_and_compact_large_image_representation.md)
- [Point-SRA: Self-Representation Alignment for 3D Representation Learning](point-sra_self-representation_alignment_for_3d_representation_learning.md)

<!-- RELATED:END -->
