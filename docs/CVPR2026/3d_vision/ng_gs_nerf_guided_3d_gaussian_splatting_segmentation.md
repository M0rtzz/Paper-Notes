---
title: >-
  [论文解读] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation
description: >-
  [CVPR 2026][3D视觉][3D Gaussian Splatting] 提出 NG-GS 框架，利用 NeRF 的连续建模能力解决 3DGS 分割中的边界离散化问题，通过 RBF 插值构建连续特征场结合多分辨率哈希编码和 NeRF-GS 联合优化实现高质量对象分割。
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "图像分割"
  - "NeRF"
  - "boundary refinement"
  - "hash encoding"
---

# NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation

**会议**: CVPR 2026  
**arXiv**: [2604.14706](https://arxiv.org/abs/2604.14706)  
**代码**: [github.com/BJTU-KD3D/NG-GS](https://github.com/BJTU-KD3D/NG-GS)  
**领域**: 三维视觉  
**关键词**: 3D Gaussian Splatting, segmentation, NeRF, boundary refinement, hash encoding

## 一句话总结

提出 NG-GS 框架，利用 NeRF 的连续建模能力解决 3DGS 分割中的边界离散化问题，通过 RBF 插值构建连续特征场结合多分辨率哈希编码和 NeRF-GS 联合优化实现高质量对象分割。

## 研究背景与动机

3DGS 已实现高效逼真的新视角合成，但其离散高斯表示导致对象边界处的分割具有锯齿和伪影。现有 3DGS 分割方法（特征蒸馏、前馈推理、掩码提升）大多忽略了高斯元素在边界的离散性问题。直接移除边界突变的高斯分布虽可改善分割但会干扰视觉质量。核心思路是利用 NeRF 的连续表示能力来调整 3DGS 在边界处的坐标和属性。

## 方法详解

### 整体框架

两阶段流程：(1) 边缘高斯连续化——通过掩码方差分析识别模糊边界高斯，RBF 插值和多分辨率哈希编码生成连续特征场；(2) NeRF-GS 联合优化——对齐损失和空间连续性损失协调两个模型的输出，确保分割边界的平滑过渡和跨视角一致性。

### 关键设计

1. **掩码方差边界高斯检测**: 对每个高斯点通过多视角 SAM 生成掩码信号集，计算掩码值的方差 $\sigma_i^2$。方差大于阈值 $\tau$ 的点视为边界高斯，构成边界集合 $\mathcal{B}$。在图像平面上扩展边界框采样查询点。

2. **RBF 插值 + 多分辨率哈希编码**: 对查询点通过 K-NN 找到邻居高斯，RBF 核加权插值生成连续特征 $\mathbf{f}^{inter}$。同时使用多分辨率哈希编码提取从粗到细的空间特征 $\mathbf{f}^{hash}$。两者组合送入轻量 NeRF 模块，插值特征作为条件向量通过 FiLM 调制 NeRF 隐层。

3. **NeRF-GS 联合优化**: 对齐损失约束边界区域 3DGS 和 NeRF 的 RGB 颜色和透明度一致；连续性损失约束相邻边界高斯点颜色的一致性；梯度平滑损失惩罚突变；掩码损失以 NeRF 密度加权进行监督学习。

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{total} = \mathcal{L}_{align} + \lambda_m \mathcal{L}_{mask} + \lambda_c \mathcal{L}_{cont} + \lambda_s \mathcal{L}_{smth}$。NeRF 和 3DGS 均使用 Adam 优化器联合训练。边界区域的局部 7×7 方差项用于促进对齐损失中的空间平滑。

## 实验关键数据

### 主实验

| 数据集 | 指标 | COB-GS | NG-GS | 提升 |
|--------|------|--------|-------|------|
| NVOS | B-mIoU | 79.1% | **84.7%** | +5.6pp |
| NVOS | mIoU | 92.1% | **92.6%** | +0.5pp |
| LERF-OVS | B-mIoU | 基线前 | **+4.4pp** | 显著 |
| ScanNet | B-mIoU | 基线前 | **+6.8pp** | 显著 |

在所有三个基准的所有指标上一致超越所有基线，边界 mIoU 提升最显著。

### 消融实验

- RBF 插值和哈希编码各自贡献互补：前者提供连续性，后者提供多尺度表达
- NeRF-GS 联合优化比单独优化任何一方效果更好
- 掩码方差阈值 $\tau=0.6$ 在精度和召回间达到最佳平衡

### 关键发现

- 边界 mIoU 的大幅提升（5-7pp）证实了连续化处理的有效性
- NeRF 作为连续精修网络而非替代方案的定位正确
- 方法可直接扩展到多对象分割场景

## 亮点与洞察

- NeRF 的连续性与 3DGS 的高效性互补的思路新颖
- 掩码方差自动检测边界高斯无需人工标注
- FiLM 调制将 RBF 插值特征融入 NeRF 的设计简洁有效

## 局限与展望

- NeRF 模块带来额外训练和推理开销
- 边界高斯检测依赖 2D 分割模型（SAM）的质量
- 在大规模场景中 K-NN 查询和 RBF 插值的计算效率需优化

## 相关工作与启发

- NeRF-GS 互补的思路可推广到编辑、生成等其他 3DGS 任务
- 多分辨率哈希编码在边界精修中的应用可借鉴到其他精细化任务
- 掩码方差分析为自动边界检测提供了简单有效的基线

## 评分

7/10 — 方法设计巧妙，边界 mIoU 提升显著，但额外计算开销需权衡。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2026\] Action-guided Generation of 3D Functionality Segmentation Data](action-guided_generation_of_3d_functionality_segmentation_data.md)
- [\[CVPR 2026\] Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment](cross-instance_gaussian_splatting_registration_via_geometry-aware_feature-guided.md)
- [\[CVPR 2026\] RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing](rap_fast_feedforward_rendering-free_attribute-guided_primitive_importance_score_.md)
- [\[ICLR 2026\] PD²GS: Part-Level Decoupling and Continuous Deformation of Articulated Objects via Gaussian Splatting](../../ICLR2026/3d_vision/pd2gs_part-level_decoupling_and_continuous_deformation_of_articulated_objects_vi.md)

</div>

<!-- RELATED:END -->
