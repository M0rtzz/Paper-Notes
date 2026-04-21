---
title: >-
  [论文解读] Sparfels: Fast Reconstruction from Sparse Unposed Imagery
description: >-
  [ICCV 2025][3D视觉][稀疏视图重建] 提出Sparfels方法，将3D基础模型（MASt3R）与高效的测试时优化（2DGS）相结合，通过MASt3R提供初始化点云/相机和对应关系引导优化，并创新性地引入泼溅色彩方差损失，在3分钟内从稀疏无位姿图像实现SOTA几何重建。
tags:
  - ICCV 2025
  - 3D视觉
  - 稀疏视图重建
  - 无位姿重建
  - 2D高斯泼溅
  - MASt3R
  - 色彩方差正则化
---

# Sparfels: Fast Reconstruction from Sparse Unposed Imagery

**会议**: ICCV 2025  
**arXiv**: [2505.02178](https://arxiv.org/abs/2505.02178)  
**代码**: [有](https://shubhendu-jena.github.io/Sparfels-web/)  
**领域**: 3D视觉  
**关键词**: 稀疏视图重建, 无位姿重建, 2D高斯泼溅, MASt3R, 色彩方差正则化

## 一句话总结

提出Sparfels方法，将3D基础模型（MASt3R）与高效的测试时优化（2DGS）相结合，通过MASt3R提供初始化点云/相机和对应关系引导优化，并创新性地引入泼溅色彩方差损失，在3分钟内从稀疏无位姿图像实现SOTA几何重建。

## 研究背景与动机

从稀疏无标定图像进行3D几何重建是一个难度很高但极具实际价值的问题：

**经典方法的困境**：SfM/MVS管线需要大量重叠图像和精确标定；3DGS/2DGS等方法依赖COLMAP提供的精确位姿和初始点云，在稀疏无标定场景下完全失效

**基础模型的机遇与局限**：DUSt3R/MASt3R等大模型可以从少量图像推理相机位姿和粗糙3D结构，但精度和细节不足，无法独立用于新视图合成

**现有稀疏视图方法的不足**：多数方法依赖多种外部先验（单目深度、法线、预训练等），管线复杂且计算成本高

**评估方法的问题**：在无位姿设定下，传统的基于相机对齐的Chamfer距离评估不够可靠

Sparfels旨在"两个世界的最佳结合"：利用MASt3R提供强初始化，用2DGS进行高效精化。

## 方法详解

### 整体框架

给定少量无位姿彩色图像：
1. **初始化阶段**：MASt3R生成全局对齐的点云、初始相机位姿和密集对应关系
2. **优化阶段**：初始化bunale-adjusting 2DGS → 联合优化高斯参数和相机外参
3. **输出**：通过TSDF算法从收敛的2DGS深度图中提取三角网格

### 关键设计

**1. 场景初始化**

- **全局几何对齐**：构建图像连接图，通过MASt3R获得成对点图和对应，优化全局一致性的点图和相机参数
- **表面元初始化**：对全局点云中每个点，PCA计算局部协方差矩阵，最小特征向量作为法线方向，构建局部坐标系$[\mathbf{u}, \mathbf{v}, \mathbf{n}]$初始化2D高斯的朝向。这种法线感知的初始化对表面质量至关重要

**2. 对应关系损失**

利用MASt3R提供的密集像素对应关系，构建跨视图几何一致性约束：

$$\mathcal{L}_{corr} = w_{p_n,p_m} \rho(p_m - \pi(P_m^{-1}P_n\pi^{-1}(p_n, d_n)))$$

其中$d_n$为2DGS泼溅深度，$\rho$为Huber损失。该约束通过深度重投影误差引导相机优化，与InstantSplat等方法不同的是利用了MASt3R的对应而非仅靠光度损失。

**3. 色彩方差正则化损失（核心创新）**

从统计矩角度分析泼溅渲染：渲染颜色是沿射线的颜色期望$C = \mathbb{E}_{t \sim p(t)}[c(t)]$。为提高鲁棒性，最小化射线上的颜色方差：

$$\mathcal{L}_{var} = \mathbb{V}ar_{t \sim p(t)}[c(t)] = \mathbb{E}[c(t)^2] - C^2$$

实现上通过修改2DGS的CUDA核函数，在渲染颜色的同时渲染颜色平方，高效计算方差。减少方差可降低渲染不确定性，产生更清晰和多视图一致的重建。

### 损失函数 / 训练策略

总目标：$\mathcal{L} = \lambda_{photo}\mathcal{L}_{photo} + \lambda_{corr}\mathcal{L}_{corr} + \lambda_{var}\mathcal{L}_{var}$

- $\mathcal{L}_{photo}$：标准L1 + SSIM + 2DGS几何正则（深度法线一致性+深度畸变）
- $\lambda_{photo}=1.0$，$\lambda_{corr}=5 \times 10^{-5}$
- $\lambda_{var}$：余弦退火调度从1.0衰减到0.0
- 优化迭代：DTU 1k次，其他2k-4k次；单阶段联合优化，无需多阶段策略
- 新视图测试：固定高斯参数，仅优化测试相机1k次

## 实验关键数据

### 主实验

**DTU 3视图重建（Rel↓ / NC↑）：**

| 方法 | Rel↓ (均值) | NC↑ (均值) |
|------|------------|------------|
| MASt3R | 7.34 | 0.830 |
| UFORecon | 42.77 | 0.371 |
| SparseCraft | 6.50 | — |
| InstantSplat2DGS | 5.73 | — |
| **Sparfels** | **4.82** | — |

Sparfels在15个场景中的Rel均值为4.82，大幅优于InstantSplat2DGS（5.73）和SparseCraft（6.50）。

**新视图合成（Tanks&Temples + MVImgNet + MipNeRF360，3/6/12视图）：**
Sparfels在多个数据集上的NVS指标（PSNR/SSIM/LPIPS）以及相机位姿估计精度（ATE）均达到或超过SOTA。

### 消融实验

**损失组件消融（DTU 3视图，Rel↓）：**

| 配置 | Photo | Corr | Var | Rel↓ |
|------|-------|------|-----|------|
| 基线 | ✓ | | | 5.73 |
| +对应损失 | ✓ | ✓ | | 5.21 |
| +方差损失 | ✓ | | ✓ | 5.35 |
| **完整** | ✓ | ✓ | ✓ | **4.82** |

对应关系损失和方差正则化分别贡献了约0.5和0.4的Rel改善，组合使用效果最佳。

### 关键发现

- **速度极快**：平均不到3分钟完成重建（消费级GPU），远快于NeRF类方法（数小时）
- **单模型依赖**：仅需MASt3R一个外部模型，无需加载额外深度/法线网络
- MASt3R对应关系引导的相机优化比纯光度优化更准确（验证在ATE指标上）
- 色彩方差损失不仅改善定量指标，定性上也产生更清晰的网格细节
- 方差权重的余弦退火策略很关键：前期强正则化稳定训练，后期放松以恢复细节
- 法线感知的初始化对表面质量影响显著

## 亮点与洞察

1. **色彩方差损失的统计视角**：从分布鲁棒优化的角度推导方差正则化，理论上界是L1损失加方差项，提供了超越经验设计的理论支撑
2. **高效的"嫁接"策略**：不训练新模型，而是将MASt3R的多种能力（点云、相机、对应）一一"嫁接"到2DGS优化中，物尽其用
3. **评估方法的改进**：针对无位姿设定提出屏幕空间depth/normal评估，避免了不可靠的相机对齐
4. **高效CUDA实现**：方差计算通过修改泼溅核函数实现，增加的开销极小

## 局限与展望

1. 极度稀疏（如2视图）时MASt3R初始化质量下降，重建精度受限
2. 无纹理区域和重复纹理区域的对应关系可能不可靠
3. 当前仅支持静态场景，未处理动态物体
4. TSDF网格提取步骤的分辨率和阈值需要手动调节
5. 可进一步探索与更强基础模型（如更大的DUSt3R变体）的结合

## 相关工作与启发

- 与InstantSplat使用3DGS不同，Sparfels选择2DGS获得更好的表面一致性
- 对应损失借鉴SPARF的跨视图约束思想，但使用MASt3R而非SfM对应
- 色彩方差损失与NeRF中的深度畸变正则化异曲同工，但从渲染颜色角度创新

## 评分

- 新颖性: ⭐⭐⭐⭐ （色彩方差损失是有理论支撑的新颖设计，整体管线也有创新）
- 实验充分度: ⭐⭐⭐⭐⭐ （5个数据集，3项任务，详细消融）
- 写作质量: ⭐⭐⭐⭐ （方法描述清晰，理论推导完整）
- 价值: ⭐⭐⭐⭐ （3分钟重建的效率优势明显，实用性强）

<!-- RELATED:START -->

## 相关论文

- [RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration](reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)
- [SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)
- [Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](../../CVPR2025/3d_vision/speedy-splat_fast_3d_gaussian_splatting_with_sparse_pixels_and_sparse_primitives.md)
- [Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)
- [CF³: Compact and Fast 3D Feature Fields](cf3_compact_and_fast_3d_feature_fields.md)

<!-- RELATED:END -->
