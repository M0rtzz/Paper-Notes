---
title: >-
  [论文解读] DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering
description: >-
  [CVPR 2025][3D视觉][稀疏视角渲染] DropoutGS 通过随机 Dropout 正则化（RDR）缓解稀疏视角 3DGS 的过拟合问题，再用边缘引导分裂策略（ESS）补偿低复杂度模型丢失的高频细节，作为即插即用模块可与多种 3DGS 方法结合，在 LLFF、DTU、Blender 上达到 SOTA。
tags:
  - CVPR 2025
  - 3D视觉
  - 稀疏视角渲染
  - 3D高斯泼溅
  - Dropout正则化
  - 边缘引导分裂
  - 过拟合缓解
---

# DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering

**会议**: CVPR 2025  
**arXiv**: [2504.09491](https://arxiv.org/abs/2504.09491)  
**代码**: https://xuyx55.github.io/DropoutGS/ (有)  
**领域**: 3D视觉  
**关键词**: 稀疏视角渲染, 3D高斯泼溅, Dropout正则化, 边缘引导分裂, 过拟合缓解

## 一句话总结
DropoutGS 通过随机 Dropout 正则化（RDR）缓解稀疏视角 3DGS 的过拟合问题，再用边缘引导分裂策略（ESS）补偿低复杂度模型丢失的高频细节，作为即插即用模块可与多种 3DGS 方法结合，在 LLFF、DTU、Blender 上达到 SOTA。

## 研究背景与动机

1. **领域现状**：3DGS 通过可微光栅化实现实时渲染，在密集视角下表现优异。然而获取大量训练视角在实际场景中成本高昂，稀疏视角下的 3DGS 仍是重要挑战。
2. **现有痛点**：（a）基于深度先验的方法（DRGS、DNGaussian）对深度精度敏感，误差会传播并放大产生伪影；（b）深度估计需要额外的计算模块；（c）3DGS 在稀疏输入时容易过参数化，导致训练损失持续下降但测试损失上升（经典过拟合）。
3. **核心矛盾**：模型复杂度（高斯数量）与训练数据量（视角数量）之间的不平衡是过拟合的根本原因。10k 高斯在 3-view 下严重过拟合，1k 高斯虽过拟合减轻但丢失高频细节。
4. **本文目标** 在缓解过拟合的同时保持模型建模高频细节的能力——这是一个"既要又要"的矛盾。
5. **切入角度**：作者通过实验观察到两个关键现象：（a）高斯数越少，过拟合越轻但高频细节越差；（b）少高斯模型的高斯尺度更大，倾向于覆盖更大区域来理解 3D 结构。这启发了"先用 dropout 获得低复杂度模型的泛化优势，再用边缘引导分裂补充高频细节"的策略。
6. **核心 idea**：用 Dropout 在训练中随机失活高斯以获得多个低复杂度子模型的集成效果来缓解过拟合，再用边缘引导分裂策略精细化大尺度高斯来恢复高频细节。

## 方法详解

### 整体框架
DropoutGS 由两个互补模块组成：（1）Random Dropout Regularization（RDR）在训练中随机失活高斯，以全模型渲染结果为 target 监督子模型，实现类似集成学习的效果；（2）Edge-guided Splitting Strategy（ESS）在训练后期识别高边缘分数的大尺度高斯并分裂，恢复边缘处的高频细节。整个方法可无缝集成到各种 3DGS 框架中。

### 关键设计

1. **随机 Dropout 正则化（Random Dropout Regularization, RDR）**:

    - 功能：通过随机失活高斯降低模型有效复杂度，缓解稀疏视角下的过拟合。
    - 核心思路：以概率 $p$ 随机失活高斯，渲染结果 $\hat{C} = \sum_{i \in \mathcal{N}} r_i \cdot \alpha_i \prod_{j=1}^{i-1}(1 - r_j \cdot \alpha_j) c_i$，其中 $r_i \sim \text{Bernoulli}(1-p)$。关键设计是 RDR 的监督信号不是 GT 图像而是**全模型渲染结果** $C$：$\mathcal{L}_{RDR} = \| C - \hat{C} \|_1 + \text{SSIM}(C, \hat{C})$。这样梯度只作用于被 dropout 影响的局部区域，让邻近高斯学会补偿被失活的高斯。
    - 设计动机：如果用 GT 图像监督 dropout 后的子模型，所有高斯的梯度会相互抵消（全局优化），效果不佳。用全模型渲染结果监督则将梯度限制在 dropout 影响的局部区域，更有效。从集成学习视角看，训练多个低复杂度子模型并在推理时集成（使用全部高斯），可获得一致的性能提升。

2. **边缘引导分裂策略（Edge-guided Splitting Strategy, ESS）**:

    - 功能：识别边缘区域的大尺度高斯并将其分裂为小高斯，恢复 RDR 导致的高频细节损失。
    - 核心思路：（a）用边缘检测器获取每个输入视角的像素级边缘概率 $E(I)$；（b）将高斯投影到边缘图上，计算单视角边缘分数 $\mathcal{E}'_i = \alpha_i \prod_j^{i-1}(1-\alpha_j) \cdot \sum_p E(I) \mathcal{M}^i(p)$；（c）跨视角累积归一化得到最终边缘分数 $\mathcal{E}_i$；（d）对同时满足大尺度 $S_i \geq \mathcal{S}_{thr}$ 和高边缘分数 $\mathcal{E}_i \geq \mathcal{E}_{thr}$ 的高斯执行分裂。
    - 设计动机：RDR 鼓励高斯增大尺度以覆盖更大区域、学习 3D 结构，但大高斯无法捕捉边缘等高频细节。ESS 有针对性地只在边缘区域分裂大高斯，在不增加过拟合风险的前提下恢复细节。

3. **集成学习视角的理论解释**:

    - 功能：从理论上解释为什么 Dropout 式的正则化对 3DGS 有效。
    - 核心思路：每次 dropout 后的渲染等价于一个低复杂度子模型的输出；训练过程中对多个子模型的优化等价于隐式训练了一个子模型集成；推理时使用全部高斯等价于对所有子模型的几何平均进行集成。从图 6 的可视化可以看到，单个子模型的渲染有伪影，但随着集成子模型数量增加，结果显著改善。
    - 设计动机：Dropout 在深度学习中被证明是通过近似指数数量的子网络的几何平均来缓解过拟合。将此理论迁移到 3DGS，为方法的有效性提供了解释。

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{gs} + \lambda_{depth} \mathcal{L}_{depth} + \lambda_{RDR} \mathcal{L}_{RDR}$
- $\mathcal{L}_{gs}$ 为标准 3DGS 损失（L1 + D-SSIM）
- $\mathcal{L}_{depth}$ 为可选的深度正则化（来自 DNGaussian）
- 不使用 Dropout 的补偿策略（实验表明补偿不显著提升且可能引入负面影响）
- 训练 6k 迭代，使用随机初始化点云

## 实验关键数据

### 主实验

| 数据集 | 设置 | 指标 | DropoutGS | DNGaussian | FreeNeRF | 3DGS |
|--------|------|------|-----------|------------|----------|------|
| LLFF | 3-view | PSNR↑ | **19.35** | 19.12 | 19.63 | 16.46 |
| LLFF | 3-view | LPIPS↓ | **0.282** | 0.294 | 0.308 | 0.401 |
| DTU | 3-view | PSNR↑ | **20.22** | 18.91 | 19.92 | 14.74 |
| DTU | 3-view | SSIM↑ | **0.830** | 0.790 | 0.787 | 0.672 |
| Blender | 8-view | PSNR↑ | **24.476** | 24.305 | 24.259 | 22.226 |
| Blender | 8-view | LPIPS↓ | **0.085** | 0.088 | 0.098 | 0.114 |

### 消融实验 / 兼容性

| 方法 | w/ DropoutGS | PSNR | LPIPS | SSIM | 初始化 |
|------|-------------|------|-------|------|--------|
| 3DGS† | ✗ | 16.46 | 0.401 | 0.440 | 随机 |
| 3DGS† | ✓ | **18.05** | 0.326 | 0.545 | 随机 |
| FSGS | ✗ | 19.86 | 0.222 | 0.670 | MVS |
| FSGS | ✓ | **20.53** | - | - | MVS |
| CoR-GS | ✗ | 20.45 | 0.196 | 0.712 | MVS |
| CoR-GS | ✓ | - | - | - | MVS |

### 关键发现
- **DTU 上提升最为显著**：DropoutGS 在 DTU 3-view 上 PSNR 达到 20.22，比 DNGaussian 高 1.31 dB，说明在物体中心场景中防止过拟合效果更明显。
- **RDR 监督信号的选择至关重要**：用全模型渲染结果而非 GT 图像作为监督目标，梯度只作用于局部（被 dropout 影响的区域），更有效地引导邻近高斯学习。
- **兼容多种 3DGS 方法**：与 3DGS、FSGS、CoR-GS 结合均有一致提升，证明了作为即插即用模块的通用性。
- **深度图质量提升**：DropoutGS 生成的深度图比 DNGaussian 更平滑、更准确，表明过拟合缓解也改善了几何质量。
- **不使用补偿策略**：与 DropGaussian 不同，DropoutGS 实验发现补偿策略效果不显著且可能影响未被 dropout 影响的像素。

## 亮点与洞察
- **"先粗后细"的解题思路**：先用 RDR 获得低复杂度模型的泛化优势（平滑但缺细节），再用 ESS 有针对性地在边缘区域恢复高频细节——这种 coarse-to-fine 框架可以迁移到其他过拟合场景。
- **监督信号的精妙选择**：用全模型渲染结果（而非 GT）监督 dropout 子模型是关键创新。这使得梯度只出现在 dropout 影响的局部区域，避免了全局梯度的相互抵消。梯度图的可视化（Fig 5）直观展示了这一差异。
- **模型复杂度与数据量的 pilot study 很有说服力**：Fig 3 展示了不同高斯数量与不同视角数量的最优匹配关系，Fig 4 展示了高复杂度模型的高斯尺度分布偏小（过拟合训练视角细节），为方法设计提供了坚实的实验基础。

## 局限与展望
- **LLFF 上 FreeNeRF 的 PSNR 仍更高**：在 LLFF 3-view 上，FreeNeRF（19.63）的 PSNR 高于 DropoutGS（19.35），但 DropoutGS 的 LPIPS 更优。说明频率正则化在前向场景上仍有优势。
- **边缘检测器的选择未深入讨论**：ESS 依赖边缘检测器，不同检测器的影响未分析。
- **dropout 概率 $p$ 的调节**：$p$ 作为超参数可能需要数据集特定的调整，缺乏自适应策略。
- **极端稀疏场景（2-view）未测试**：仅在 3-view 及以上进行了实验，在极端稀疏下的表现未知。

## 相关工作与启发
- **vs DropGaussian**: 两篇同期工作独立提出了 Dropout 用于稀疏视角 3DGS 的想法。DropoutGS 的创新在于（a）使用全模型渲染结果而非 GT 作为监督，（b）增加了 ESS 补偿高频细节。DropGaussian 使用不透明度补偿因子和渐进丢弃率。两者互补。
- **vs DNGaussian**: DNGaussian 通过深度先验正则化，DropoutGS 从过拟合角度出发。两者可组合——DropoutGS 在 DNGaussian 上进一步提升。
- **vs FreeNeRF**: FreeNeRF 用频率正则化解决 NeRF 的稀疏视角问题，DropoutGS 将类似的正则化思路迁移到 3DGS 的离散表示上。
- **集成学习启发**：将 Dropout 解释为隐式子模型集成为 3DGS 的正则化提供了理论框架，启发了更多正则化技术的探索。

## 评分
- 新颖性: ⭐⭐⭐⭐ RDR 的监督信号设计和 ESS 的补偿策略是关键创新，pilot study 分析扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集 + 兼容性验证 + 深度图可视化 + 梯度图分析 + 点云可视化
- 写作质量: ⭐⭐⭐⭐ 动机分析透彻，pilot study 有说服力，方法阐述清晰
- 价值: ⭐⭐⭐⭐ 即插即用的通用正则化模块，在 DTU 上提升显著，实用性强

<!-- RELATED:START -->

## 相关论文

- [DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)
- [DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](../../CVPR2026/3d_vision/dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [iSegMan: Interactive Segment-and-Manipulate 3D Gaussians](isegman_interactive_segment-and-manipulate_3d_gaussians.md)
- [CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)

<!-- RELATED:END -->
