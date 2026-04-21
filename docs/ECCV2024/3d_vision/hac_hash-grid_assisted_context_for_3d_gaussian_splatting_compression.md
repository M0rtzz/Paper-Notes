---
title: >-
  [论文解读] HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression
description: >-
  [ECCV 2024][3D视觉][3D Gaussian Splatting] 利用结构化二值哈希网格为无序的3DGS锚点建立空间上下文关系，通过条件概率建模和自适应量化实现高效熵编码，达到相比vanilla 3DGS **75×** 的压缩率，同时保持甚至提升渲染质量。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D Gaussian Splatting
  - 模型压缩
  - 上下文建模
  - 熵编码
  - 哈希网格
---

# HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression

**会议**: ECCV 2024  
**arXiv**: [2403.14530](https://arxiv.org/abs/2403.14530)  
**代码**: https://github.com/YihangChen-ee/HAC (有)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 模型压缩, 上下文建模, 熵编码, 哈希网格

## 一句话总结

利用结构化二值哈希网格为无序的3DGS锚点建立空间上下文关系，通过条件概率建模和自适应量化实现高效熵编码，达到相比vanilla 3DGS **75×** 的压缩率，同时保持甚至提升渲染质量。

## 研究背景与动机

**领域现状**: 3D Gaussian Splatting (3DGS) 凭借高保真度和实时渲染速度成为新视角合成的主流方法，但需要大量高斯基元（百万级别）来表示场景，存储开销可达数GB。

**现有痛点**: 现有3DGS压缩方法主要聚焦于参数"值"本身（如剪枝、向量量化），忽略了高斯基元之间的**结构关系冗余**。Scaffold-GS引入锚点聚类高斯但仍将每个锚点独立处理。

**核心矛盾**: 3DGS的点云本质使得高斯基元稀疏且无组织，难以像NeRF特征网格那样直接利用空间结构关系进行压缩。

**本文目标**: 如何挖掘无序锚点之间的空间一致性，建立有效的上下文模型来大幅压缩3DGS表示。

**切入角度**: 受NeRF系列使用特征网格表示3D空间的启发，探索无序锚点属性与结构化哈希网格之间的互信息关系。

**核心 idea**: 联合学习二值化哈希网格作为锚点属性的上下文，通过条件概率估计实现高效熵编码压缩。

## 方法详解

### 整体框架

HAC基于Scaffold-GS构建，整体包含三个层次：
- **底层**: Scaffold-GS的锚点-高斯结构，锚点属性 $\mathcal{A} = \{\mathbf{f}^a, \mathbf{l}, \mathbf{o}\}$ 通过MLP预测高斯属性
- **中层**: 联合学习的二值哈希网格 $\mathcal{H}$，对任意锚点位置 $\mathbf{x}^a$ 进行插值得到哈希特征 $\mathbf{f}^h$
- **上层**: 上下文模型(MLP)以 $\mathbf{f}^h$ 为输入，预测锚点属性的值分布参数，用于算术编码

核心公式是条件概率分解：

$$p(\mathcal{A}, \mathbf{x}^a, \mathcal{H}) = p(\mathcal{A}|\mathbf{x}^a, \mathcal{H}) \times p(\mathbf{x}^a, \mathcal{H}) \sim p(\mathcal{A}|\mathbf{f}^h) \times p(\mathcal{H})$$

关键洞察：不直接用哈希特征替代锚点特征（会导致质量下降），而是将其作为**上下文**来估计锚点属性的概率分布。

### 关键设计

1. **自适应量化模块 (AQM)**: 锚点属性需量化为有限集合以进行熵编码。不同属性（特征 $\mathbf{f}^a$、缩放 $\mathbf{l}$、偏移 $\mathbf{o}$）的数值范围差异大，固定步长不适用。AQM通过上下文MLP从 $\mathbf{f}^h$ 预测量化步长调整因子 $\mathbf{r}$：

$$\mathbf{q}_i = Q_0 \times (1 + \text{Tanh}(\mathbf{r}_i)), \quad \mathbf{r}_i = \text{MLP}_q(\mathbf{f}^h_i)$$

量化步长被约束在 $(0, 2Q_0)$ 范围内，$Q_0$ 对 $\mathbf{f}^a$、$\mathbf{l}$、$\mathbf{o}$ 分别设为1、0.001、0.2。训练时使用加噪近似，测试时使用舍入。

2. **高斯分布概率建模**: 统计发现锚点属性近似服从高斯分布。上下文MLP从 $\mathbf{f}^h$ 为每个锚点独立预测 $\boldsymbol{\mu}_i$ 和 $\boldsymbol{\sigma}_i$，计算量化属性 $\hat{\mathbf{f}}_i$ 落在量化区间的概率：

$$p(\hat{\mathbf{f}}_i) = \Phi_{\boldsymbol{\mu}_i, \boldsymbol{\sigma}_i}\left(\hat{\mathbf{f}}_i + \frac{1}{2}\mathbf{q}_i\right) - \Phi_{\boldsymbol{\mu}_i, \boldsymbol{\sigma}_i}\left(\hat{\mathbf{f}}_i - \frac{1}{2}\mathbf{q}_i\right)$$

其中 $\Phi$ 为高斯CDF。高概率意味着低熵，即更少的编码比特数。

3. **自适应偏移掩码**: 统计发现偏移 $\mathbf{o}$ 在零点处有脉冲分布，意味着大量冗余高斯。使用直通估计(STE)的二值掩码剪枝无效偏移，若锚点的所有偏移均被剪枝则整体移除该锚点。

4. **哈希网格压缩**: 哈希表参数二值化为 $\{-1, +1\}$，通过统计"+1"出现频率 $h_f$ 用AE编码。采用混合3D-2D结构：12层3D嵌入(分辨率16~512) + 4层2D嵌入(分辨率128~1024)，特征维度 $D^h=4$。

### 损失函数 / 训练策略

总损失为三部分加权：

$$\mathcal{L} = L_{\text{Scaffold}} + \lambda_e \frac{1}{N(D^a+6+3K)}(L_{\text{entropy}} + L_{\text{hash}}) + \lambda_m L_m$$

- $L_{\text{Scaffold}}$: Scaffold-GS原始渲染损失
- $L_{\text{entropy}}$: 锚点属性的熵损失 $\sum -\log_2 p(\hat{f}_{i,j})$
- $L_{\text{hash}}$: 哈希网格的比特消耗估计
- $L_m$: 偏移掩码正则项

**分阶段训练**：0-3K迭代为Scaffold-GS原始训练；3K-10K引入加噪量化适应；10K后完整接入哈希网格和熵约束。$\lambda_e$ 从 $5\times10^{-4}$ 到 $4\times10^{-3}$ 调节压缩率。

## 实验关键数据

### 主实验

| 方法 | Synthetic-NeRF PSNR↑ | 大小(MB)↓ | Mip-NeRF360 PSNR↑ | 大小(MB)↓ | BungeeNeRF PSNR↑ | 大小(MB)↓ |
|------|------|------|------|------|------|------|
| 3DGS | 33.80 | 68.46 | 27.49 | 744.7 | 24.87 | 1616 |
| Scaffold-GS | 33.41 | 19.36 | 27.50 | 253.9 | 26.62 | 183.0 |
| Lee et al. | 33.33 | 5.54 | 27.08 | 48.80 | 23.36 | 82.60 |
| Compressed3D | 32.94 | 3.68 | 26.98 | 28.80 | 24.13 | 55.79 |
| **HAC-low** | **33.24** | **1.18** | **27.53** | **15.26** | **26.48** | **18.49** |
| **HAC-high** | **33.71** | **1.86** | **27.77** | **21.87** | **27.08** | **29.72** |

### 消融实验

| 组件 | BungeeNeRF效果 | Synthetic-NeRF效果 |
|------|------|------|
| 完整HAC | 最优RD曲线 | 最优RD曲线 |
| 移除哈希网格互信息(置零) | 比特消耗剧增，概率退化为无条件 $p(\mathcal{A})$ | 同左 |
| 移除AQM | 高码率/复杂场景质量显著下降 | 保真度损失 |
| 移除偏移掩码 | 简单场景/低码率时码率节省减少 | 移除大量位置冗余的空间 |

### 关键发现

- HAC的高保真模式PSNR甚至超过Scaffold-GS基线，归因于熵损失的正则化效果和增大的锚点特征维度
- 推理时哈希网格可完全移除，不影响渲染FPS（HAC在BungeeNeRF上283 FPS vs Scaffold-GS 232 FPS）
- 比特分配可视化显示复杂纹理区域分配更多总比特，但每个锚点的平均比特消耗反而平滑，验证了空间一致性假设

## 亮点与洞察

- **首创上下文建模用于3DGS压缩**：将图像/视频压缩中成熟的上下文建模思路迁移到3D高斯压缩
- **不修改原始结构的压缩**：上下文建模仅用于编解码阶段的概率估计，渲染时完全移除，保证速度和质量上界不受影响
- **二值哈希网格的双重作用**：既作为上下文信号源，自身又极易压缩（仅需存储00/1频率）

## 局限与展望

- 训练时间比Scaffold-GS增加约0.9×（BungeeNeRF: 15.1min → 27.6min）
- 编解码过程中AE的CPU单线程执行是瓶颈（BungeeNeRF需26.7秒）
- 锚点位置 $\mathbf{x}^a$ 直接以32位存储，未纳入熵约束
- 可探索更强的概率模型（如混合高斯）替代单高斯分布

## 相关工作与启发

- **Scaffold-GS**: 提供锚点-高斯的层次化结构基础，使上下文建模成为可能
- **Instant-NGP / CNC**: 哈希网格的压缩方案可直接复用
- **图像压缩中的上下文模型**: 条件概率建模的核心思路直接来源于learned image compression

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将上下文编码引入3DGS压缩，哈希网格桥接无序锚点的idea巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 5个数据集、完整消融、比特分配可视化、RD曲线
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，从互信息验证到条件概率建模层层递进
- **实用价值**: ⭐⭐⭐⭐⭐ — 75×压缩率使3DGS大规模部署成为可能

<!-- RELATED:START -->

## 相关论文

- [GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](gaussianimage_1000_fps_image_representation_and_compression_by_2d_gaussian_splat.md)
- [Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)
- [SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)
- [3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting](3igs_factorised_tensorial_illumination_for_3d_gaussian_splatting.md)
- [G2fR: Frequency Regularization in Grid-Based Feature Encoding Neural Radiance Fields](g2fr_frequency_regularization_in_grid-based_feature_encoding_neural_radiance_fie.md)

<!-- RELATED:END -->
