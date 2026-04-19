---
title: >-
  [论文解读] SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation
description: >-
  [CVPR 2026][3D视觉][2D Gaussian Splatting] SGI 提出基于种子点(seed)的结构化 2D 高斯表示框架，通过将无结构高斯原语组织为种子驱动的神经高斯、结合上下文引导的熵编码和多尺度拟合策略，在高分辨率图像表示中实现最高 7.5× 压缩比和 6.5× 优化加速，同时保持甚至提升重建保真度。
tags:
  - CVPR 2026
  - 3D视觉
  - 2D Gaussian Splatting
  - image representation
  - neural compression
  - entropy coding
  - multi-scale optimization
---

# SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation

**会议**: CVPR 2026  
**arXiv**: [2603.07789](https://arxiv.org/abs/2603.07789)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 2D Gaussian Splatting, image representation, neural compression, entropy coding, multi-scale optimization  

## 一句话总结

SGI 提出基于种子点(seed)的结构化 2D 高斯表示框架，通过将无结构高斯原语组织为种子驱动的神经高斯、结合上下文引导的熵编码和多尺度拟合策略，在高分辨率图像表示中实现最高 7.5× 压缩比和 6.5× 优化加速，同时保持甚至提升重建保真度。

## 背景与动机

**问题背景**：2D Gaussian Splatting 作为一种新兴的图像表示技术，能在低端设备上实现高效渲染。然而，将其扩展到高分辨率图像时需要独立优化和存储数百万个无结构高斯原语，导致两大核心瓶颈：

1. **参数冗余**：现有方法（如 GaussianImage、LIG）将每个高斯独立优化，未利用空间局部性（spatial locality）——相邻像素往往共享相似颜色、纹理和结构，导致相邻原语之间存在大量冗余参数
2. **优化开销**：高分辨率下百万级高斯原语的独立优化收敛缓慢，尤其当引入量化感知微调进行压缩时，开销更加显著

**已有尝试的局限**：
- INR 方法（SIREN、I-NGP）需大量 MLP 前向传播，计算量大
- 3D anchor-based 方法（Scaffold-GS、HAC）直接应用于 2D 图像建模无法获得同等压缩收益，因为 2D 高斯已去掉了 opacity 等参数，存储节省空间有限
- LIG 的 Level-of-Gaussian 仅用部分高斯做残差拟合，非全局优化

## 方法详解

### 整体框架

SGI 包含三个核心组件：

1. **Seed-based 2D Neural Gaussians**（种子驱动的 2D 神经高斯）：将图像分解为多尺度局部空间，每个种子点通过轻量 MLP 预测一组结构化高斯原语
2. **Neural Entropy Coding**（神经熵编码）：利用二值哈希网格和上下文模型估计种子属性的概率分布，实现自适应比特分配
3. **Multi-scale Fitting**（多尺度拟合）：采用高斯金字塔从粗到细逐级优化，加速收敛

### 关键设计一：种子驱动的 2D 神经高斯

给定预定义的 $N$ 个种子点，均匀初始化覆盖图像。每个种子位于位置 $\boldsymbol{x_a} \in \mathbb{R}^2$，关联一组属性：

$$\mathcal{A} = \{\boldsymbol{f_a} \in \mathbb{R}^D, \boldsymbol{s_o} \in \mathbb{R}^2, \boldsymbol{s_a} \in \mathbb{R}^2, \boldsymbol{\delta} \in \mathbb{R}^{K \times 2}\}$$

其中各属性含义：

| 符号 | 维度 | 含义 |
|------|------|------|
| $\boldsymbol{f_a}$ | $\mathbb{R}^D$ | 种子特征向量，编码局部区域信息 |
| $\boldsymbol{s_o}$ | $\mathbb{R}^2$ | 偏移缩放因子，控制关联高斯的空间分布范围 |
| $\boldsymbol{s_a}$ | $\mathbb{R}^2$ | 尺度缩放因子，调整高斯的最终尺度 |
| $\boldsymbol{\delta}$ | $\mathbb{R}^{K \times 2}$ | $K$ 个关联高斯的学习偏移量 |

**高斯位置计算**：每个种子关联的 $K$ 个高斯位置由种子位置加上缩放后的偏移得到：

$$\{\boldsymbol{\mu}^{(k)}\}_{k=0}^{K-1} = \boldsymbol{x_a} + \{\boldsymbol{\delta}^{(k)}\}_{k=0}^{K-1} \cdot \boldsymbol{s_o}$$

**属性解码**：通过两个共享的轻量 MLP 从种子特征 $\boldsymbol{f_a}$ 解码高斯属性：
- $\text{MLP}_c$：输出 opacity 加权的颜色系数 $\mathbf{c'} \in \mathbb{R}^3$
- $\text{MLP}_\Sigma$：预测基础尺度 $\boldsymbol{s_{\text{base}}}$ 和旋转角度 $\theta$

最终尺度 $\boldsymbol{s} = \boldsymbol{s_{\text{base}}} \cdot \boldsymbol{s_a}$，协方差矩阵通过正定分解构造：

$$\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top$$

其中 $\mathbf{R}(\theta)$ 为 2D 旋转矩阵，$\mathbf{S}$ 为对角尺度矩阵。

**渲染公式**：像素颜色通过所有贡献高斯的累积求和计算：

$$\boldsymbol{C} = \sum_{i \in I} \mathbf{c'}_i G_i(\mathbf{x}), \quad G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

**设计精妙之处**：每个高斯仅需 8 个参数（位置 2 + 协方差 3 + 加权颜色 3），而大量空间冗余信息被种子特征和共享 MLP 隐式编码，显著减少独立参数量。

### 关键设计二：基于上下文模型的熵编码

单纯使用种子结构仅能减少约 3% 的存储（对比纯 2D 高斯如 LIG），增益有限。SGI 的核心创新是利用种子引入的结构正则性进一步做熵编码压缩。

**量化策略**：训练时用噪声注入模拟量化，推理时用取整量化：

$$\hat{\boldsymbol{f}}_j^{(i)} = \begin{cases} \boldsymbol{f}_j^{(i)} + \mathcal{U}(-\frac{1}{2}, \frac{1}{2}) \cdot q_j^{(i)} & \text{训练} \\ \text{Round}(\boldsymbol{f}_j^{(i)} / q_j^{(i)}) \cdot q_j^{(i)} & \text{推理} \end{cases}$$

量化步长 $q_j^{(i)} = Q_j \times (1 + \tanh(r_j^{(i)}))$，其中 $r_j^{(i)}$ 由上下文模型自适应预测。

**概率建模**：引入可学习的二值哈希网格 $\mathcal{H}$ 捕获种子的空间一致性，上下文模型 $\text{MLP}_p$ 估计每个属性分量的高斯分布参数：

$$\{\mu_j^{(i)}, \sigma_j^{(i)}, r_j^{(i)}\}_{j=0}^{3} = \text{MLP}_p(\mathcal{H}(\boldsymbol{x}_a^{(i)}))$$

属性概率通过对高斯分布在量化区间上积分得到，驱动算术编码实现自适应比特分配。

### 关键设计三：多尺度拟合策略

直接在高分辨率图像上优化种子参数计算代价高且收敛慢。SGI 构建 $M$ 层高斯金字塔，从粗到细逐级优化：

1. 构建金字塔 $\{I_0=I, I_1, \ldots, I_{M-1}\}$，每层下采样 2 倍
2. 从最粗层 $l=M-1$ 开始优化种子和 MLP 参数
3. 将优化结果传递到下一层：位置和尺度乘以 2 适配分辨率加倍
4. 逐层迭代直至最细层 $l=0$

### 损失函数

总损失结合重建保真度和比特消耗正则化：

$$L = L_{\text{img}} + \frac{\lambda}{N \cdot d_\mathcal{A}}(L_{\text{entropy}} + L_{\text{hash}})$$

- $L_{\text{img}}$：渲染图像与目标图像的 L1 损失
- $L_{\text{entropy}}$：种子属性的信息熵损失，驱动概率模型学习紧凑分布
- $L_{\text{hash}}$：二值哈希网格的比特消耗上界
- $\lambda = 0.001$：率失真权衡超参数
- $d_\mathcal{A} = D + 4 + 2K$：每个种子的属性总维度

## 实验

### 实验设置

- **数据集**：FGF2（4 张卫星图，~51MP）、ICB（2 张自然图，27.7/39.1MP）、STimage（3 张病理图，~76MP）
- **评估指标**：PSNR、SSIM、LPIPS、优化时间（分钟）、模型大小（MB）
- **SGI 设置**：低码率（3.5M 高斯）和高码率（10M 高斯）

### 主实验结果

| 方法 | FGF2 PSNR↑ | FGF2 Size(MB)↓ | FGF2 Time(min)↓ | ICB PSNR↑ | ICB Size(MB)↓ | ICB Time(min)↓ |
|------|-----------|----------------|-----------------|----------|--------------|----------------|
| SIREN | 22.05 | 15.79 | 649.71 | 27.62 | 15.79 | 363.34 |
| I-NGP | 28.55 | 21.07 | 72.32 | 33.09 | 21.07 | 48.11 |
| HAC | 25.15 | 16.78 | 261.69 | 34.47 | 13.52 | 270.57 |
| GaussianImage | 27.30 | 23.37 | 322.17 | 31.09 | 23.37 | 282.61 |
| **SGI (low)** | **31.24** | **16.33** | **48.43** | **35.27** | **12.30** | **44.75** |
| 3DGS | 34.93 | 787.73 | 642.85 | 37.52 | 787.73 | 515.99 |
| Scaffold-GS | 28.25 | 112.61 | 248.83 | 35.76 | 105.81 | 162.11 |
| LIG | 32.10 | 106.81 | 87.56 | 36.40 | 106.81 | 68.73 |
| **SGI (high)** | **36.27** | **41.74** | **97.75** | **39.09** | **32.15** | **86.11** |

### 消融实验

**每个种子的高斯数量 $K$ 消融**：

| $K$ | FGF2 PSNR↑ | FGF2 Size(MB)↓ | ICB PSNR↑ | ICB Size(MB)↓ |
|-----|-----------|----------------|----------|--------------|
| 5 | 31.29 | 18.48 | 35.03 | 13.64 |
| **10** | **31.24** | **16.33** | **35.27** | **12.30** |
| 15 | 30.61 | 15.32 | 34.88 | 11.48 |
| 20 | 30.62 | 14.83 | 34.57 | 10.87 |

$K$ 越大模型越紧凑但保真度略降，$K=10$ 为最佳折中。

**关键发现**：

1. **熵编码是压缩核心**：$\lambda=0$ 时（无熵编码），FGF2 存储为 104.08MB；$\lambda=0.001$ 时降至 16.33MB，压缩 6.4 倍。种子结构本身仅减 3%，熵编码贡献了绝大部分压缩增益
2. **多尺度拟合显著加速**：金字塔层数 $M=3$ 时优化时间从 $M=1$ 的 71.59 分钟降至 48.43 分钟（加速 32%），同时 PSNR 从 30.58 提升至 31.24 dB
3. **存储分解**：种子特征 $\boldsymbol{f_a}$ 占总存储的 48%，偏移 $\boldsymbol{\delta}$ 占 35%，哈希网格和 MLP 仅占 ~1%，开销极小
4. **低码率优势**：在低 bpp 区域，SGI 显著超越 JPEG（ICB 上 0.245 bpp 达 26.09 dB vs JPEG 0.296 bpp 仅 22.77 dB）

## 亮点

- **首个结构化 2D 高斯表示**：将 3D 领域的 anchor-based 思想适配到 2D 图像，并通过熵编码弥补了 2D 场景压缩增益有限的问题
- **压缩效率突出**：相比 LIG 实现 7.5× 压缩，相比量化后的 GaussianImage 实现 1.6× 压缩
- **优化效率高**：多尺度拟合策略实现 1.6~6.5× 加速，且不牺牲重建质量
- **低端设备友好**：基于可微光栅化的快速渲染，24GB A10 GPU 即可处理 76MP 图像

## 局限性

- 大型图像仍需数十分钟优化（per-image optimization 范式），不适合实时编码场景
- 每张图像需独立优化一个模型，缺乏跨图像的泛化能力
- 当前仅在静态图像上验证，未扩展到视频或动态场景
- 高码率设置下存储仍在 30~40MB 级别，与学习型图像编解码器（如 VVC/AV1）相比并无绝对优势

## 评分 ⭐⭐⭐⭐

扎实的系统性工作，将种子结构+熵编码+多尺度优化三者有机结合，在大规模图像表示任务上取得了存储、速度、质量三方面的显著改进。实验全面，消融充分。但 per-image 优化范式和泛化能力是固有局限。

<!-- RELATED:START -->

## 相关论文

- [GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting](../../AAAI2026/3d_vision/gaussianimage_boosted_image_representation_and_compression_with_2d_gaussian_spla.md)
- [SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation](swifttailor_efficient_3d_garment_generation_with_geometry_image_representation.md)
- [GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](../../ECCV2024/3d_vision/gaussianimage_1000_fps_image_representation_and_compression_by_2d_gaussian_splat.md)
- [MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping](magician_efficient_long-term_planning_with_imagined_gaussians_for_active_mapping.md)
- [CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)

<!-- RELATED:END -->
