---
title: >-
  [论文解读] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering
description: >-
  [CVPR 2025][3D视觉][稀疏体素] 本文提出 SVRaster，一种无需神经网络或 3D 高斯的高效辐射场渲染方法，通过自适应多层次稀疏体素表示和基于方向相关 Morton 排序的定制光栅化器，实现了无伪影的实时高保真渲染。 领域现状：新视角合成领域有两大主流路线——基于光线追踪的 NeRF 方法使用密集采样进行…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "稀疏体素"
  - "光栅化"
  - "辐射场渲染"
  - "无神经网络"
  - "新视角合成"
---

# Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering

**会议**: CVPR 2025  
**arXiv**: [2412.04459](https://arxiv.org/abs/2412.04459)  
**代码**: [https://github.com/NVlabs/svraster](https://github.com/NVlabs/svraster)  
**领域**: 3D视觉  
**关键词**: 稀疏体素, 光栅化, 辐射场渲染, 无神经网络, 新视角合成

## 一句话总结

本文提出 SVRaster，一种无需神经网络或 3D 高斯的高效辐射场渲染方法，通过自适应多层次稀疏体素表示和基于方向相关 Morton 排序的定制光栅化器，实现了无伪影的实时高保真渲染。

## 研究背景与动机

**领域现状**：新视角合成领域有两大主流路线——基于光线追踪的 NeRF 方法使用密集采样进行体积渲染，物理精确但速度慢；3DGS 使用高斯基元的光栅化渲染，速度快但存在排序不正确导致的 popping 伪影。

**现有痛点**：（1）3DGS 基于基元中心排序，无法保证正确的深度顺序，视角变化时出现突然的颜色跳变（popping artifact）；（2）3D 高斯覆盖重叠时体积密度定义不明确，表面重建困难；（3）现有体素方法（如 Plenoxels）使用均匀分辨率或依赖密集 3D 数据结构，效率和质量有限。

**核心矛盾**：光栅化的高效性和体积渲染的物理正确性看似只能二选一——但体素作为网格基元天然具有明确的排序和体积定义，是连接两者的理想桥梁。

**本文目标**：用稀疏体素作为场景表示，设计高效的光栅化算法，同时获得 3DGS 的速度和体积渲染的正确性。

**切入角度**：Morton 编码（Z 序曲线）为八叉树布局的体素提供了天然的空间排序，只需根据光线方向选择正确的 Morton 排列即可保证渲染顺序。

**核心 idea**：（1）用自适应多层次稀疏体素（Octree 布局，最高 $65536^3$ 分辨率）无神经网络地显式存储密度和球谐系数；（2）设计方向相关的 Morton 序排序，在光栅化框架中保证正确的体素渲染顺序，消除 popping 伪影。

## 方法详解

### 整体框架

场景用稀疏体素集合表示，每个体素存储 8 个角点密度值（三线性插值得到连续密度场）和球谐系数（视角相关颜色）。渲染时将体素投影到图像空间分配到 tile，按方向相关 Morton 序排序，然后按标准 alpha blending 合成像素颜色。

### 关键设计

1. **自适应多层次稀疏体素表示**:

    - 功能：以不同分辨率忠实地表达场景的不同细节层次
    - 核心思路：在 Octree 布局下分配体素，最大层级 $L=16$（分辨率 $(2^{16})^3 = 65536^3$）。每个体素由索引 $v=\{i,j,k\}$ 和层级 $l$ 定义，大小 $\mathbf{v}_s = \mathbf{w}_s \cdot 2^{-l}$。密度场用 8 个角点参数 $\mathbf{v}_{\text{geo}} \in \mathbb{R}^{2 \times 2 \times 2}$ 表示，相邻体素共享角点保证密度场连续。Alpha 值通过在光线-体素交段均匀采样 $K$ 个点数值积分得到 $\alpha = 1 - \exp(-\frac{l}{K}\sum_k \text{explin}(\text{interp}(\mathbf{v}_{\text{geo}}, \mathbf{q}_k)))$。不使用任何密集 3D 数据结构，体素存储为 1D 数组。
    - 设计动机：均匀分辨率体素方法（如 Plenoxels）在细节不一的场景中要么浪费内存要么丢失细节。自适应层级允许粗糙区域用大体素、精细区域用小体素，大幅提高表示效率。

2. **方向相关 Morton 序排序光栅化器**:

    - 功能：保证任意大小体素的正确渲染顺序，消除 popping 伪影
    - 核心思路：在 Octree 布局下，按 Morton 编码（位交织操作）对体素排序可以保证正确的深度顺序，但前提是 Morton 排列方式要与光线方向一致。3D 空间有 8 种 Morton 排列（由光线方向的 3 个分量正负号决定），算法根据当前光线方向选择对应的 Morton 排列进行排序。整个光栅化流程：（a）投影体素 8 个角点到图像空间；（b）分配体素到覆盖的 tile；（c）按方向相关 Morton 序排序；（d）前向/后向 alpha 合成。
    - 设计动机：3DGS 按基元中心排序只是近似，混合大小的高斯导致顺序不对（图 4a 示例）。Octree 布局的体素天然支持 Morton 序排序，完全避免了排序近似的问题。

3. **渐进场景优化策略**:

    - 功能：从粗到细自适应地增加体素分辨率
    - 核心思路：训练开始时在粗层级初始化体素，每隔固定步数进行"成长"操作——将渲染梯度大的体素细分为 8 个子体素（Octree 分裂）。同时裁剪低密度体素减少冗余。密度激活函数使用 $\text{explin}(x)$ 代替 softplus，在大值区域为线性更高效。不需要 COLMAP 稀疏点作为初始化。
    - 设计动机：从粗到细的策略避免了一开始就需要大量体素，渐进增长让优化更稳定。无需稀疏点初始化使方法更通用。

### 损失函数 / 训练策略

使用 L1 + SSIM 颜色损失，加上密度蒸馏损失和法线平滑正则化。SH 阶数从 0 阶渐进增加到 3 阶。训练和渲染都利用 CUDA 定制实现。体素属性（颜色、法线）在预处理阶段计算一次，所有像素共享。

## 实验关键数据

### 主实验

MipNeRF-360 数据集新视角合成：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|---|---|---|---|---|
| iNGP | 25.59 | 0.699 | 0.331 | 9.43 |
| Plenoxels | 23.08 | 0.626 | 0.463 | 6.79 |
| 3DGS | 27.49 | 0.815 | 0.214 | 134 |
| 2DGS | 26.76 | 0.805 | 0.230 | 117 |
| **SVRaster** | **27.30** | **0.813** | **0.218** | **142** |

### 消融实验

| 配置 | PSNR↑ | FPS↑ | 说明 |
|---|---|---|---|
| 均匀体素 | ~23-24 | ~10 | Plenoxels 级别 |
| 自适应体素 + naive 排序 | — | — | popping 伪影 |
| 自适应体素 + Morton 排序 | **27.30** | **142** | 完整方法 |

与 Plenoxels 对比：

| 方法 | PSNR↑ | FPS↑ |
|---|---|---|
| Plenoxels | 23.08 | 6.79 |
| **SVRaster** | **27.30** | **142** |

提升：+4.22 dB PSNR，20× FPS 加速。

### 关键发现

- SVRaster 在 PSNR/SSIM/LPIPS 上与 3DGS 相当（27.30 vs 27.49），FPS 更高（142 vs 134）
- 相比同类无神经网络体素方法 Plenoxels，提升超过 4 dB PSNR 和 20× FPS
- 方向相关 Morton 排序完全消除了 popping 伪影，这是 3DGS 至今未完全解决的问题
- 稀疏体素与 Volume Fusion、Marching Cubes 等经典 3D 算法天然兼容——可以直接提取网格、融合语义特征场

## 亮点与洞察

1. **"回归体素"的逆向思维**：在高斯 splatting 风头正劲时重新审视体素，发现体素在排序正确性和体积定义上的天然优势
2. **Morton 序的巧妙应用**：利用 Octree 结构和 Morton 编码的数学性质解决渲染排序问题，这个方案简洁、高效、正确
3. **与经典 3D 处理的无缝兼容**：稀疏体素可以直接进行 Volume Fusion、Voxel Pooling 和 Marching Cubes，为辐射场的下游应用（语义理解、网格提取）提供了便利

## 局限与展望

- 体素表示本质上是离散的，在极平滑表面上可能不如连续表示
- $K$ 点采样在体素内部是近似的，大体素内的积分精度有限
- 未在极大规模场景（城市级别）上验证可扩展性
- 未来可以结合 SDF 替代密度场来获得更好的表面重建

## 相关工作与启发

- **vs 3DGS**：3DGS 用高斯基元 + 光栅化，速度快但有 popping 伪影和体积歧义；SVRaster 用体素 + 定制光栅化，速度相当且无伪影
- **vs Plenoxels**：同为无神经网络体素方法，但 SVRaster 的自适应层级和光栅化比 Plenoxels 的均匀体素和光线投射高效得多
- **vs iNGP**：iNGP 使用哈希网格 + 小 MLP；SVRaster 完全无神经网络，渲染速度更快
- 启发：场景表示不一定要创新——经典体素 + 好的渲染算法就能达到 SOTA

## 评分

- 新颖性: 8/10 — "体素+光栅化"方向的开创性，Morton 排序方案优雅
- 实验充分度: 8/10 — 主流数据集全覆盖，与 SOTA 对比充分，有下游应用展示
- 写作质量: 9/10 — 方法描述清晰，图示直观，排序问题的分析很有教育意义
- 价值: 9/10 — 为辐射场渲染提供了兼具速度、质量和物理正确性的新方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)
- [\[CVPR 2025\] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction](depth-guided_bundle_sampling_for_efficient_generalizable_neural_radiance_field_r.md)
- [\[CVPR 2025\] Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians](sparse_point_cloud_patches_rendering_via_splitting_2d_gaussians.md)
- [\[CVPR 2025\] Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization](evolving_high-quality_rendering_and_reconstruction_in_a_unified_framework_with_c.md)
- [\[ICCV 2025\] SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing](../../ICCV2025/3d_vision/svg-head_hybrid_surface-volumetric_gaussians_for_high-fidelity_head_reconstructi.md)

</div>

<!-- RELATED:END -->
