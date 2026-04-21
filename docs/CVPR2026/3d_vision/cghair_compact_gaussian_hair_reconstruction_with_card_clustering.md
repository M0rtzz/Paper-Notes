---
title: >-
  [论文解读] CGHair: Compact Gaussian Hair Reconstruction with Card Clustering
description: >-
  [CVPR 2026][3D视觉][3D高斯溅射] 提出 CGHair，通过发片（hair card）引导的分层聚类和共享高斯外观码本，在保持可比视觉质量的同时实现 200 倍以上的外观参数压缩和 4 倍发丝重建加速。
tags:
  - CVPR 2026
  - 3D视觉
  - 3D高斯溅射
  - 头发重建
  - 发片聚类
  - 紧凑表示
  - 外观压缩
---

# CGHair: Compact Gaussian Hair Reconstruction with Card Clustering

**会议**: CVPR 2026  
**arXiv**: [2604.03716](https://arxiv.org/abs/2604.03716)  
**代码**: [项目页](https://humansensinglab.github.io/CGHair/) (有)  
**领域**: 3D Vision  
**关键词**: 3D高斯溅射, 头发重建, 发片聚类, 紧凑表示, 外观压缩

## 一句话总结

提出 CGHair，通过发片（hair card）引导的分层聚类和共享高斯外观码本，在保持可比视觉质量的同时实现 200 倍以上的外观参数压缩和 4 倍发丝重建加速。

## 研究背景与动机

**领域现状**：基于 3DGS 的头发重建方法（如 GaussianHair）通过圆柱高斯链建模发丝，实现了高保真实时渲染。但密集头发建模需要数百万个高斯基元，导致存储和渲染成本巨大。

**现有痛点**：GaussianHair 等方法为每根发丝独立分配球谐系数，忽略了头发在同一发型中具有强结构和外观冗余的特性，导致严重的参数冗余。

**核心矛盾**：高保真度要求大量独立参数 vs 实际部署需要紧凑表示。

**本文目标**：利用头发的内在结构相似性，设计紧凑的高斯头发表示。

**切入角度**：借鉴游戏和影视工业中广泛使用的"发片（hair card）"概念——将相似发丝聚类为发片并共享纹理。

**核心 idea**：分层聚类 + 共享外观码本，实现发丝级高保真与极致压缩的兼得。

## 方法详解

### 整体框架

四阶段管线：(1) 基于生成先验的高效发丝重建；(2) 发片生成（发丝聚类 + 发片几何 + 纹理映射）；(3) 基于发片聚类的紧凑高斯表示（共享外观码本）；(4) 多视图外观优化。

### 关键设计

1. **生成先验加速的发丝重建**：采用 PERM 参数化头发几何模型，在头皮 UV 空间上将头发表示为引导图 $\mathbf{G} = \mathcal{D}_{guide}(\alpha)$ 和风格图 $\mathbf{S} = \mathcal{D}_{style}(\beta)$，通过预训练的 StyleGAN2 和 VAE 解码器生成。联合优化发型编码 $\alpha, \beta$ 和解码器，结合沿发丝的圆柱高斯进行光度监督。
   设计动机：直接优化潜在纹理效率低下且需要昂贵的扩散正则化。利用 PERM 的强几何先验实现 4 倍加速（vs GaussianHaircut）且全自动（vs GaussianHair 需手动操作）。

2. **分层发片生成**：

    - **发丝聚类**：将每根发丝的 3D 点串联成向量，用 k-means 聚类为 $N_c$ 组（真实头发 800 组，合成 400 组）
    - **发片几何构建**：以聚类中心发丝为轴线，通过约束优化求解法向量 $\{n_k^*\} = \arg\min \sum_k \sum_{p_i} \|(p_i - \bar{p}_k) \cdot n_k\|$，确保发片平面包含所有发丝
    - **纹理生成**：通过优化 UV 坐标和法向位移 $\delta_i$ 将 3D 发丝投影到发片表面，生成抗锯齿纹理图
   设计动机：发片是工业标准的头发简化表示，直接编码了几何和外观冗余。

3. **共享高斯外观码本**：将 $N_c$ 个发片按纹理特征再聚类为 $N_T=64$ 组，每组共享一个含 $K=10$ 个特征的外观码本（$D=64$ 维），每根发丝通过 Gumbel-Softmax 软索引选择码本条目：
    $\mathbf{F}_{strand} = \sum_{k=1}^K \pi_k \cdot \mathbf{F}_k$
   然后用全局共享 MLP 解码器 $\phi_{dec}$ 将低维特征解码为每个高斯的球谐系数：
    $[\mathbf{SH}_1, ..., \mathbf{SH}_{|\mathcal{S}|}] = \phi_{dec}(\mathbf{F}_{strand})$
   设计动机：直接将码本条目映射为所有高斯的 SH 系数需要 4k+ 维度。通过低维潜在空间 + 共享解码器实现极致压缩。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_p + \mathcal{L}_a + \mathcal{L}_o$$
- $\mathcal{L}_p$：L1 + D-SSIM 光度损失
- $\mathcal{L}_a$：alpha 掩码监督损失
- $\mathcal{L}_o$：不透明度平滑正则化
- 前 7k 迭代冻结几何参数，之后以较小学习率联合优化

## 实验关键数据

### 主实验 — 发丝重建质量

| 指标 | CGHair | 仅 $\mathcal{D}_{PCA}$ | 冻结 $\alpha,\beta$ | 仅 $\mathcal{D}_{style}$ |
|------|--------|-------------------|-------------------|---------------------|
| Pos. Error ↓ | **0.148** | 0.162 | 0.159 | 0.182 |
| Cur. Error ↓ | **6.73** | 9.12 | 6.98 | 7.48 |

### 消融实验 — 紧凑表示

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Size(MB)↓ | 压缩比↑ |
|------|-------|-------|--------|----------|---------|
| Unique (每发丝独立SH) | 33.99 | 0.982 | 0.019 | 163.7 | 1.0 |
| 无发片聚类 | 28.00 | 0.938 | 0.042 | 22.42 | 7.30 |
| Single-SH | 30.48 | 0.958 | 0.037 | 0.46 | 355.9 |
| **Latents (完整)** | **32.40** | **0.970** | **0.026** | **0.71** | **230.6** |

### 关键发现

1. 完整 CGHair 仅用 0.71MB 外观参数即达到 PSNR 32.40，相比独立 SH（163.7MB）压缩 230 倍，PSNR 仅降 1.59。
2. 发片聚类是关键：无聚类时全局共享PSNR 仅 28.00，证明发片级结构化聚类的必要性。
3. 码本大小 $K=10$ 已足够，增至 90 仅提升微小 PSNR 但存储急增。
4. 潜在维度 $D=64$ 是最优平衡点。

## 亮点与洞察

- 将工业中成熟的"发片"概念优雅地引入学术头发重建，实现了结构先验与数据驱动重建的统一。
- 分层压缩策略：发丝→发片→发片组→共享码本，每层利用不同粒度的冗余。
- 可插拔设计：CGHair 模块可直接叠加到 GaussianHair 的重建发丝上，作为后处理压缩模块。
- Gumbel-Softmax 软索引设计允许端到端训练，避免了 VQ 中常见的 codebook collapse。

## 局限与展望

- 聚类数量 $N_c, N_T$ 为手动设定，可考虑自适应确定。
- 当前仅支持静态头发，未处理动态/运动场景。
- 压缩后 PSNR 仍比无压缩低 ~1.6dB，对细节要求极高的场景可能不足。
- 依赖 PERM 预训练先验，对非常规发型的泛化未充分验证。

## 相关工作与启发

- CompGS 等通用 3DGS 压缩方法未利用头发特有的结构冗余，CGHair 是领域特化压缩的范例。
- VQVAE 的量化编码思路被成功迁移到 3D 头发表示。
- 对其他具有强结构冗余的 3D 重建任务（如织物、植被）有启发意义。

## 评分

- 新颖性: ⭐⭐⭐⭐ 发片聚类+共享码本的组合新颖，但各技术组件相对成熟
- 实验充分度: ⭐⭐⭐⭐⭐ 多粒度消融非常充分
- 写作质量: ⭐⭐⭐⭐ 管线清晰，图示丰富
- 价值: ⭐⭐⭐⭐ 200倍压缩对实际部署有重大意义

<!-- RELATED:START -->

## 相关论文

- [Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives](prune_wisely_reconstruct_sharply_compact_3d_gaussian_splatting_via_adaptive_prun.md)
- [Human Hair Reconstruction with Strand-Aligned 3D Gaussians](../../ECCV2024/3d_vision/human_hair_reconstruction_with_strand-aligned_3d_gaussians.md)
- [DGH: Dynamic Gaussian Hair](../../NeurIPS2025/3d_vision/dgh_dynamic_gaussian_hair.md)
- [E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation](sgi_structured_2d_gaussians_for_efficient_and_compact_large_image_representation.md)

<!-- RELATED:END -->
