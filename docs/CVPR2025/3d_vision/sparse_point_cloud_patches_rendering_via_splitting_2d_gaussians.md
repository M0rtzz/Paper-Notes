---
title: >-
  [论文解读] Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians
description: >-
  [CVPR 2025][3D视觉][点云渲染] 提出直接从点云预测 2D Gaussians 进行照片级真实渲染的方法，通过 entire-patch 架构实现跨类别泛化，通过 splitting decoder 将稀疏点云上采样为更密集的高斯原语，在仅用2K-100K点的条件下实现了 SOTA 渲染质量和 142 FPS 的实时渲染速度。
tags:
  - CVPR 2025
  - 3D视觉
  - 点云渲染
  - 2D高斯溅射
  - 稀疏点云
  - 跨类泛化
  - 新视角合成
---

# Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians

**会议**: CVPR 2025  
**arXiv**: [2505.09413](https://arxiv.org/abs/2505.09413)  
**代码**: https://github.com/murcherful/GauPCRender  
**领域**: 3D视觉  
**关键词**: 点云渲染, 2D高斯溅射, 稀疏点云, 跨类泛化, 新视角合成

## 一句话总结

提出直接从点云预测 2D Gaussians 进行照片级真实渲染的方法，通过 entire-patch 架构实现跨类别泛化，通过 splitting decoder 将稀疏点云上采样为更密集的高斯原语，在仅用2K-100K点的条件下实现了 SOTA 渲染质量和 142 FPS 的实时渲染速度。

## 研究背景与动机

**领域现状**：点云渲染是3D视觉的基础任务。传统方法直接将点投影为平面/球体，结果充满孔洞且不真实。近年来深度学习方法（NPBG、NPBG++）需要多视角图像作为额外输入。最新方法（TriVol、Point2Pix）从点云预测 NeRF，PFGS 预测3D Gaussians 但需要后续对渲染图像的精修。

**现有痛点**：(1) TriVol 和 Point2Pix 依赖 NeRF 表示，渲染速度慢（1.62 FPS）；(2) PFGS 虽用了3D高斯但需要额外的 recurrent decoder 精修渲染结果，不是真正的端到端；(3) 所有方法都依赖密集点云（80K-100K）且只能在单一类别上泛化；(4) 3D Gaussians 的法线模糊导致几何不精确。

**核心矛盾**：高质量点云渲染需要密集表示，但真实场景中的点云往往稀疏且分布不均匀；现有方法处理整个点云，导致对类别先验的依赖和泛化能力差。

**本文目标**：设计一个方法能够(1)从稀疏点云直接预测高质量高斯原语用于渲染，(2)无需精修即可产生照片级质量，(3)跨类别和跨数据集泛化。

**切入角度**：2D Gaussian Splatting (2DGS) 相比3DGS 有显式法线，而点云的法线可以轻松估计——这意味着可以利用点云法线初始化2D高斯，为网络提供良好的起点。处理点云 patches 而非整体可以摆脱类别先验的约束。

**核心 idea**：用 entire-patch 架构在 patch 级别预测 2D Gaussians（用整体预测作为"背景"实现完整图像监督），并通过 splitting decoder 将每个点分裂为 K 个高斯原语来处理稀疏点云。

## 方法详解

### 整体框架

输入为点云（坐标+颜色），输出为渲染图像。网络包含两个相同的 2D Gaussian Prediction Module——一个处理完整点云（$\mathcal{N}_e$），一个处理点云 patch（$\mathcal{N}_p$）。训练时先预训练 $\mathcal{N}_e$，再冻结它训练 $\mathcal{N}_p$；$\mathcal{N}_e$ 预测的非 patch 部分的高斯作为"背景"，与 $\mathcal{N}_p$ 预测的 patch 高斯合并后渲染完整图像用于损失计算。推理时可直接用 $\mathcal{N}_p$ 分 patch 处理任意点云。

### 关键设计

1. **基于法线的 2D 高斯初始化**:

    - 功能：利用点云几何信息为网络提供良好的初始化
    - 核心思路：对每个输入点估计法线 $\hat{n}$，用法线初始化 2D 高斯的朝向（法线 $n$ + 绕法线旋转角 $\alpha=0$），用点坐标初始化位置，用点颜色初始化球谐系数，用最近邻距离初始化尺度，不透明度设为1。与3DGS/2DGS的随机四元数初始化不同，法线初始化确保每个高斯都朝向正确方向且在任意视角可见，避免了网络预测四元数旋转的困难。
    - 设计动机：消融实验表明没有法线初始化时 PSNR 从 27.88 暴跌到 9.62，因为随机初始化的高斯在很多视角不可见，网络无法收敛。

2. **Splitting Decoder（分裂解码器）**:

    - 功能：将每个初始化的高斯"分裂"为 K 个更精细的高斯
    - 核心思路：首先用 PointMLP 编码器提取局部特征 $F_l$。然后六个分裂解码器分别预测位置、尺度、颜色、法线、旋转角、不透明度六个参数的 K 个偏移量。以位置为例，$\Delta_1^x, ..., \Delta_K^x = D_x(F_l, X^i, C^i, N^i, S^i)$，最终位置 $X^p = \bigcup_{j=1}^{K}(\Delta_j^x + X^i)$。每个解码器包含共享权重的 MLP。K=4 时每个点变为4个高斯，有效增加了表示密度。
    - 设计动机：稀疏点云无法覆盖物体表面的所有细节，通过分裂上采样可以用更少的输入点生成更密集的高斯来捕捉纹理和几何细节。分裂的偏移量基于学到的局部特征预测，比简单的上采样更智能。

3. **Entire-Patch 架构**:

    - 功能：让网络在 patch 级别工作以实现跨类别泛化
    - 核心思路：完整点云和 patch 分别由两个结构相同的预测模块处理。训练时随机选一个中心点，取最近的 $N_p=2048$ 个点作为 patch。$\mathcal{N}_e$ 预测完整点云的高斯中去掉 patch 部分作为"背景"，与 $\mathcal{N}_p$ 预测的 patch 高斯合并渲染完整图像。关键洞察：直接用 patch 渲染的图像不完整，无法计算有效的损失函数；但有了完整点云预测的"背景填充"，就保证了监督信号的完整性。
    - 设计动机：处理局部 patch 使网络不再依赖全局点分布模式，从而泛化到不同类别和不同密度的点云。之前方法必须对每个类别单独训练，而 patch-based 方法天然具有跨类能力。

### 损失函数 / 训练策略

使用 MSE + SSIM 的组合损失：$\mathcal{L} = \frac{1}{N_c}\sum_{i=1}^{N_c}(\beta \mathcal{L}_{MSE} + (1-\beta)\mathcal{L}_{SSIM})$，$\beta=0.8$。每个样本渲染 $N_c=8$ 个视角的图像用于监督。图像分辨率：物体 256×256、场景 640×512、人体 512×512。分裂数 K=4，patch 点数 2048。Adam 优化器，lr=1e-4，batch=8，最多 480 epoch。先预训练 $\mathcal{N}_e$ 再训练 $\mathcal{N}_p$。RTX 3090 训练约 30 小时。

## 实验关键数据

### 主实验

| 数据集 | 点数 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|------|-------|-------|--------|
| Car (ShapeNet) | 100K | TriVol | 27.22 | 0.927 | 0.084 |
| Car (ShapeNet) | 100K | PFGS | 27.34 | 0.942 | 0.077 |
| Car (ShapeNet) | 100K | **Ours** | **28.73** | **0.960** | **0.060** |
| Car (ShapeNet) | 20K | **Ours** | **27.88** | **0.949** | **0.068** |
| THuman2.0 | 80K | PFGS | 34.74 | 0.983 | 0.009 |
| THuman2.0 | 80K | **Ours** | **35.43** | **0.987** | **0.009** |
| ScanNet | 100K | PFGS | 19.86 | 0.758 | 0.452 |
| ScanNet | 100K | **Ours** | **20.24** | **0.759** | **0.490** |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| No Initialization | 9.62 | 0.685 | 0.655 | 无法线初始化→训练崩溃 |
| No $\mathcal{N}_e$ | 12.94 | 0.676 | 0.441 | 无背景补全→patch无法有效训练 |
| $\mathcal{N}_p$ only | 23.23 | 0.829 | 0.126 | 仅patch处理整体→资源消耗大 |
| **Full model** | **27.88** | **0.949** | **0.068** | 完整模型 |

| 渲染速度 | 方法 | FPS |
|----------|------|-----|
| | NPBG++ | 37.45 |
| | TriVol | 1.62 |
| | PFGS | 3.80 |
| | **Ours** | **142.86** |

### 关键发现

- **法线初始化是方法成立的前提**：去掉后 PSNR 从 27.88 坠落到 9.62，训练直接发散。随机初始化的高斯在很多视角不可见，梯度无法传播。
- **仅用 20% 点即超越 100K 基线**：在 Car 类别上 20K 点的本方法（27.88 PSNR）已超过 TriVol-100K（27.22）和 PFGS-100K（27.34），稀疏点云处理能力极强
- **泛化能力突出**：在 Car-20K 上训练的模型直接测试 DTU 场景数据集，20K 点获得 PSNR 16.71，而 PFGS 用同样设置只有 8.07
- **渲染速度碾压**：142.86 FPS vs PFGS 3.80 FPS（37倍加速），因为直接用 2DGS 光栅化，无需后处理精修
- 分裂数 K 的增加持续改善稀疏点云性能，K=4 在性能与计算开销之间取得最佳平衡

## 亮点与洞察

- **2D Gaussians + 法线初始化的巧妙组合**：2DGS 相比 3DGS 的优势在于有显式法线，而点云法线可以轻松估计。这个搭配让初始化从"随机四元数"变为"几何对齐的法线方向"，是方法成立的关键。这种"找到表示方式与数据特性之间的自然匹配"的思路非常值得学习。
- **Splitting Decoder 作为隐式上采样**：不同于传统的点云上采样方法，splitting decoder 在高斯参数空间操作，一步完成上采样和属性预测，非常高效。这个设计可以推广到其他需要从稀疏表示生成密集表示的场景。
- **Entire-Patch 架构解决了 patch 训练的监督问题**：用完整预测做"背景"使 patch 可训练，是一个工程上非常实用的trick。

## 局限与展望

- 当点云某部分完全缺失时（而非稀疏），方法无法"脑补"缺失区域，因为没有先验支持
- ScanNet 室内场景上的提升不如物体类别明显（20.24 vs 19.86 PSNR），可能因为室内场景的全局上下文更重要
- Entire-patch 架构需要两阶段训练（先 $\mathcal{N}_e$ 再 $\mathcal{N}_p$），增加了训练复杂度
- 未来可结合点云补全技术处理大面积缺失的情况
- 可以尝试处理从实际扫描仪获得的带噪声野外点云

## 相关工作与启发

- **vs PFGS**: PFGS 也预测高斯+渲染，但需要两阶段（预测+精修），速度只有 3.8 FPS 且泛化差。本方法直接预测即可渲染，速度快 37 倍，且支持跨类泛化
- **vs TriVol**: TriVol 用三平面表示预测 NeRF，速度极慢（1.62 FPS）且只在单类别内泛化。本方法在 20K 稀疏点下就能超越其 100K 的结果
- **vs 2DGS/3DGS**: 原始 2DGS/3DGS 需要对每个场景单独优化数千次迭代。本方法是前馈预测，训练一次后可泛化到新点云，无需单独优化

## 评分

- 新颖性: ⭐⭐⭐⭐ 法线初始化+splitting decoder+entire-patch架构的组合很精巧，但各组件单独看并不全新
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集（场景/物体/人体）、多种点云密度、泛化实验、速度对比、详细消融
- 写作质量: ⭐⭐⭐⭐ 清晰易读，图表直观，Table 1 对比维度全面
- 价值: ⭐⭐⭐⭐⭐ 实时渲染+稀疏点云+跨类泛化，实用性极强，代码已开源

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)
- [\[CVPR 2025\] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)
- [\[CVPR 2025\] Toward Robust Neural Reconstruction from Sparse Point Sets](toward_robust_neural_reconstruction_from_sparse_point_sets.md)
- [\[CVPR 2025\] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [\[CVPR 2025\] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter](pma_towards_parameter-efficient_point_cloud_understanding_via_point_mamba_adapte.md)

<!-- RELATED:END -->
