---
title: >-
  [论文解读] SAGS: Structure-Aware 3D Gaussian Splatting
description: >-
  [ECCV 2024][3D视觉][3D高斯喷溅] 提出 SAGS，通过局部-全局图表示和图神经网络隐式编码场景几何结构，在保持实时渲染的同时提升3DGS的渲染质量、减少存储需求（最高24×压缩），并显著抑制浮点伪影。 解决思路 本文目标：领域现状：3D-GS以几何无关的方式独立优化每个高斯核，忽视场景内在3D结构…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "3D高斯喷溅"
  - "图神经网络"
  - "结构感知"
  - "新视角合成"
  - "模型压缩"
---

# SAGS: Structure-Aware 3D Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2404.19149](https://arxiv.org/abs/2404.19149)  
**代码**: [有](https://eververas.github.io/SAGS/)  
**领域**: 3D视觉  
**关键词**: 3D高斯喷溅, 图神经网络, 结构感知, 新视角合成, 模型压缩

## 一句话总结

提出 SAGS，通过局部-全局图表示和图神经网络隐式编码场景几何结构，在保持实时渲染的同时提升3DGS的渲染质量、减少存储需求（最高24×压缩），并显著抑制浮点伪影。

## 研究背景与动机

### 解决思路

**本文目标**：**领域现状**：3D-GS以几何无关的方式独立优化每个高斯核，忽视场景内在3D结构，导致高斯核大幅偏离初始位置，产生浮点伪影，深度图质量差。现有压缩方法（codebook量化等）也未利用结构信息。本文引入点云分析中的图网络思想，让邻近高斯核共享信息并学习保持拓扑的位移。

## 方法详解

### 整体框架

1. **曲率感知致密化**：估计点云高斯曲率，在低曲率区域通过中点插值增加点密度
2. **结构感知编码器**：构建k-NN图，用GNN聚合局部+全局特征
3. **精炼网络**：4个独立MLP分别解码颜色、不透明度、协方差和位移

### 关键设计

**曲率感知致密化**：COLMAP在纹理缺乏的平面区域采点不足，利用局部PCA估计曲率，在低曲率区域插入中点补充点云密度。

**结构感知GNN编码器**：
$$\Phi(\mathbf{p}_i, \mathbf{f}_i) = \phi\left(\sum_{j \in \mathcal{N}(i)} w_{ij} h_\Theta(\gamma(\mathbf{p}_j), \mathbf{f}_j - \mathbf{f}_i, \mathbf{g})\right)$$
使用相对特征 f_j - f_i、位置编码 γ(p)、全局特征 g=max(f)，通过反距离权重聚合邻域信息。

**位移预测**：高斯位置建模为初始COLMAP位置加上MLP预测的位移 Δp，强制小位移保持场景拓扑。

**SAGS-Lite**：仅在关键点上训练网络，中点属性通过插值得到，无需任何压缩技术即可实现极致轻量化。

### 损失函数

$$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{SSIM}, \quad \lambda=0.2$$

## 实验关键数据

### 主实验

**Mip-NeRF360 / Tanks&Temples / Deep Blending 渲染质量**：

| 方法 | MipNeRF360 PSNR/SSIM/LPIPS | T&T PSNR/SSIM/LPIPS | DB PSNR/SSIM/LPIPS |
|------|---------------------------|---------------------|---------------------|
| 3D-GS | 28.69/0.870/0.182 | 23.14/0.841/0.183 | 29.41/0.903/0.243 |
| Scaffold-GS | 28.84/0.848/0.220 | 23.96/0.853/0.177 | 30.21/0.906/0.254 |
| **SAGS** | **29.65/0.874/0.179** | **24.88/0.866/0.166** | **30.47/0.913/0.241** |

**存储压缩比（对比3D-GS）**：

| 方法 | MipNeRF360 (MB) | T&T (MB) | Deep Blending (MB) |
|------|-----------------|----------|---------------------|
| 3D-GS | 693 | 411 | 676 |
| Scaffold-GS | 252 (2.8×↓) | 87 (4.7×↓) | 66 (10.2×↓) |
| **SAGS** | **135 (5.1×↓)** | **75 (5.5×↓)** | **58 (11.7×↓)** |
| **SAGS-Lite** | **76 (9.1×↓)** | **35 (12×↓)** | **28 (24×↓)** |

### 消融实验

| 消融项 | DB PSNR | T&T PSNR |
|--------|---------|----------|
| w/o 曲率致密化 | 29.87% | 23.97% |
| w/o GNN | 29.94% | 24.19% |
| w/o 位置编码 | 30.21% | 24.31% |
| w/o 全局特征 | 30.17% | 24.42% |
| w/o 视角相关位置 | 30.07% | 24.37% |
| **完整SAGS** | **30.47%** | **24.88%** |

### 关键发现

- 结构感知使高斯核位移集中在小范围内，抑制浮点伪影
- SAGS的深度图显著优于3D-GS和Scaffold-GS，能捕捉尖锐边缘和平坦表面
- SAGS-Lite在无压缩技术下实现24×存储缩减，仍保持接近3D-GS的质量

## 亮点与洞察

1. **首个结构感知3DGS方法**：桥接点云分析和3DGS两个领域
2. **位移预测范式**：约束高斯核保持在初始几何附近，隐式保护场景拓扑
3. **SAGS-Lite极致轻量**：中点插值方案简单有效，24×压缩无需量化
4. 同时获得更好的渲染质量和更小的模型尺寸

## 局限与展望

- GNN推理增加一定计算开销
- 依赖COLMAP初始点云质量
- 对高度动态场景不适用

## 相关工作与启发

- Scaffold-GS引入层级结构但仍用无结构优化
- 点云分析中GNN（DGCNN、PointNet++）的思想可迁移到3DGS
- 启发：场景结构先验是减少冗余高斯核、提升质量的关键

## 评分

- 创新性：★★★★★ 首次将GNN引入3DGS，结构感知思想新颖
- 实用性：★★★★☆ 实时渲染、大幅压缩对VR/AR应用价值大
- 实验质量：★★★★★ 13个场景3个数据集完全评估，消融详尽

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting](talkinggaussian_structure-persistent_3d_talking_head_synthesis_via_gaussian_spla.md)
- [\[ECCV 2024\] Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting](per-gaussian_embedding-based_deformation_for_deformable_3d_gaussian_splatting.md)
- [\[ECCV 2024\] HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression](hac_hash-grid_assisted_context_for_3d_gaussian_splatting_compression.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)

</div>

<!-- RELATED:END -->
