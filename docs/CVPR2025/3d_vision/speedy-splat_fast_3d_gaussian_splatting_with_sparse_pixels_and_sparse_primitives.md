---
title: >-
  [论文解读] Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives
description: >-
  [CVPR 2025][3D视觉][3D高斯泼溅] 提出 Speedy-Splat，通过两条互补路线加速 3DGS 渲染：(1) SnugBox/AccuTile 精确定位高斯在图像平面的范围减少无效像素处理，(2) 高效剪枝（Soft+Hard Pruning）将高斯数量减少 90% 以上，两者结合实现平均 6.71× 渲染加速，同时减少 10.6× 模型大小和 1.4× 训练时间。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯泼溅
  - 渲染加速
  - 剪枝
  - 精确瓦片相交
  - 实时渲染
---

# Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives

**会议**: CVPR 2025  
**arXiv**: [2412.00578](https://arxiv.org/abs/2412.00578)  
**代码**: [项目主页](https://speedysplat.github.io)  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 渲染加速, 剪枝, 精确瓦片相交, 实时渲染

## 一句话总结

提出 Speedy-Splat，通过两条互补路线加速 3DGS 渲染：(1) SnugBox/AccuTile 精确定位高斯在图像平面的范围减少无效像素处理，(2) 高效剪枝（Soft+Hard Pruning）将高斯数量减少 90% 以上，两者结合实现平均 6.71× 渲染加速，同时减少 10.6× 模型大小和 1.4× 训练时间。

## 研究背景与动机

- **3DGS 的速度瓶颈**：虽然 3DGS 实现了桌面 GPU 实时渲染，但在边缘设备（手机等）上仍无法实时，且模型体积大。
- **渲染成本的两个因素**：3DGS 渲染成本正比于高斯数量 × 每个高斯处理的像素数，Speedy-Splat 同时优化这两个因素。
- **瓦片分配的保守性**：原始 3DGS 用最大特征值计算外接圆再外接正方形确定瓦片相交区域，严重高估了高斯的实际范围——忽略了不透明度 $\sigma_i$ 和椭圆形状信息。
- **高斯的冗余性**：大量研究表明 3DGS 模型中约 90% 的高斯是冗余的，可在保持视觉质量的前提下剪除。
- **已有剪枝的实用性限制**：PUP 3D-GS 提出了基于 Hessian 的有原则剪枝方法，效果好但存储需求为 $N \times 36$，无法集成到训练过程中。

## 方法详解

### 整体框架

Speedy-Splat 由两部分组成：精确瓦片相交（SnugBox + AccuTile，即插即用不改变渲染结果）和高效剪枝（Soft Pruning 集成到稠密化阶段 + Hard Pruning 在稠密化后应用），两者集成到 3DGS 训练管线中。

### 关键设计

**设计一：SnugBox — 精确轴对齐包围盒**
- **功能**：为每个高斯计算考虑不透明度的紧凑包围盒，替代原始 3DGS 的保守圆形估计
- **核心思路**：高斯 $\mathcal{G}_i$ 的实际像素范围由 $\alpha_i \geq \frac{1}{255}$ 阈值决定，联合 $\sigma_i$ 和 2D 协方差 $\Sigma_{i_{2D}}$ 可得椭圆方程 $t = ax_d^2 + 2bx_dy_d + cy_d^2$，其中 $t = 2\log(255\sigma_i)$。对椭圆方程求极值 $\partial y_d / \partial x_d = 0$ 得 $y_{\min}, y_{\max}$，交换 $x, y$ 和 $a, c$ 得 $x_{\min}, x_{\max}$，构成紧凑包围盒。
- **设计动机**：原始方法的外接正方形在高斯不透明度低或高度各向异性时严重高估范围；SnugBox 操作开销恒定，仅需在 preprocess 和 duplicateWithKeys 中各调用一次。

**设计二：AccuTile — 精确瓦片-椭圆相交**
- **功能**：进一步从 SnugBox 的矩形区域中找出真正与椭圆相交的瓦片
- **核心思路**：沿 SnugBox 矩形的短边逐行/列处理，每次迭代仅需计算椭圆与边界线的两个交点（利用相邻行共享边界），时间复杂度正比于矩形短边长度。关键洞察：SnugBox 极值点是唯一的拐点，非拐点行/列上椭圆单调，只需计算边界交点即可确定最小/最大瓦片范围。
- **设计动机**：SnugBox 的矩形中仍有不与椭圆相交的角部瓦片，AccuTile 进一步消除这些无效分配，且增加的计算开销极小。

**设计三：高效 Soft/Hard Pruning — 集成式剪枝训练**
- **功能**：在训练过程中逐步剪除冗余高斯，减少 90%+ 的高斯数量
- **核心思路**：将 PUP 3D-GS 的 Hessian 重参数化：改为对每个高斯的 2D 投影值 $g_i(p)$ 计算灵敏度 $\tilde{U}_i = \log|\nabla_{g_i} I_\mathcal{G} \nabla_{g_i} I_\mathcal{G}^T|$，将存储需求从 $N \times 36$ 降至 $N \times 1$（36× 压缩），且能直接利用渲染核的像素并行梯度。Soft Pruning 在稠密化阶段每隔固定迭代剪除 80% 低灵敏度高斯；Hard Pruning 在稠密化结束后进一步剪除 30%。
- **设计动机**：PUP 原始方法需要对 3D 参数 $\mu, s$ 的像素级梯度（破坏 3DGS 高效梯度流），重参数化为 2D 投影值的梯度完美契合 3DGS 的渲染-反传架构。

### 损失函数

与标准 3DGS 相同：$L = \|I_\mathcal{G}(\phi) - I_{\text{gt}}\|_1 + L_{\text{D-SSIM}}(I_\mathcal{G}(\phi), I_{\text{gt}})$。剪枝不改变损失函数，仅在训练过程中周期性执行。

## 实验关键数据

### 主实验：渲染管线各环节加速（平均）

| 配置 | Preprocess | Radix Sort | Render | Overall | 加速比 |
|------|-----------|-----------|--------|---------|-------|
| Baseline | 0.665ms | 1.551ms | 4.483ms | 7.478ms | 1× |
| +SnugBox | 0.656ms | 0.729ms | 2.344ms | 4.102ms | 1.82× |
| +AccuTile | 0.668ms | 0.612ms | 2.062ms | 3.748ms | 2.00× |
| +Soft Pruning | 0.370ms | 0.404ms | 1.337ms | 2.381ms | 3.14× |
| **+Hard Pruning** | **0.091ms** | **0.215ms** | **0.619ms** | **1.114ms** | **6.71×** |

### Tanks & Temples Truck 场景示例

| 指标 | 3DGS原始 | Speedy-Splat |
|------|---------|-------------|
| 高斯数量 | ~100% | <10% (减少90%+) |
| PSNR下降 | - | 仅微小下降 |
| 渲染加速 | 1× | 6.2× |
| 训练加速 | 1× | 1.38× |

### 关键发现

1. SnugBox + AccuTile 作为即插即用模块，不改变任何渲染结果（零精度损失），即可获得 2× 加速
2. 累积使用所有优化后，平均渲染加速 6.71×，模型减小 10.6×，训练加速 1.4×
3. 高效剪枝分数的重参数化将存储从 $N \times 36$ 压缩至 $N \times 1$，使剪枝可在训练中实时进行
4. Soft + Hard 的两阶段剪枝在稠密化前后分别应用，比单一后处理剪枝更有效

## 亮点与洞察

- **精确性即效率**：SnugBox/AccuTile 通过更精确的几何计算减少无效工作而非近似简化——精确反而更快
- **36× 存储压缩的 Hessian 重参数化**：优雅地将 3D 参数的灵敏度转化为 2D 投影值的灵敏度，完美契合 3DGS 架构
- **工程创新出色**：深入 CUDA 渲染管线的每个环节分析瓶颈，提供了详尽的逐函数计时分析
- 所有方法与现有 3DGS 变体正交，可组合使用

## 局限与展望

- AccuTile 的逐行/列遍历在极大高斯（覆盖很多瓦片）时可能引入额外开销
- 当前仅在静态场景上验证，动态 3DGS 的适用性未知
- 剪枝比例（80%/30%）为经验设定，不同场景可能需要调整
- 对具有大量半透明高斯的场景，SnugBox 的加速效果可能受限

## 相关工作与启发

- SnugBox 的椭圆包围盒计算可推广到其他需要精确 2D 高斯范围估计的场景（如碰撞检测）
- Hessian 重参数化思路可扩展到其他需要训练中计算灵敏度的场景（如 NAS、结构化剪枝）
- 与 StopThePop 等方法的精确瓦片相交虽思路相似，但 Speedy-Splat 更快且计算开销更低

## 评分

⭐⭐⭐⭐ — 工程深度出色，从理论分析到 CUDA 实现都很扎实。6.71× 的加速在不显著牺牲质量的前提下非常实用。SnugBox 的即插即用零精度损失加速和 Hessian 重参数化都是有独立价值的贡献。

<!-- RELATED:START -->

## 相关论文

- [DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)
- [S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction](spars3r_semantic_prior_alignment_and_regularization_for_sparse_3d_reconstruction.md)
- [DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)

<!-- RELATED:END -->
