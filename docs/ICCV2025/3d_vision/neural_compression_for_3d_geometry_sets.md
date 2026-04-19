---
title: >-
  [论文解读] Neural Compression for 3D Geometry Sets
description: >-
  [ICCV 2025][3D视觉][3D几何压缩] 提出NeCGS,首个能将包含数千个多样3D网格模型的几何集合压缩高达900倍的神经压缩范式,通过TSDF-Def隐式表示和量化感知自解码器实现高精度保持。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D几何压缩
  - 神经压缩
  - TSDF
  - 自解码器
  - 几何集合
---

# Neural Compression for 3D Geometry Sets

**会议**: ICCV 2025  
**arXiv**: [2405.15034](https://arxiv.org/abs/2405.15034)  
**代码**: [GitHub](https://github.com/rsy6318/NeCGS)  
**领域**: 3D视觉  
**关键词**: 3D几何压缩, 神经压缩, TSDF, 自解码器, 几何集合

## 一句话总结

提出NeCGS,首个能将包含数千个多样3D网格模型的几何集合压缩高达900倍的神经压缩范式,通过TSDF-Def隐式表示和量化感知自解码器实现高精度保持。

## 研究背景与动机

3D网格模型在计算机图形学、VR、机器人等领域广泛应用。随着几何数据日益复杂,高效压缩技术至关重要。

现有方法的局限:

**基于体素点云的方法**(GPCC/VPCC)需高分辨率(≥$2^{10}$)才能准确表示,引入冗余

**SDF/TSDF表示**面临需要不同尺寸张量的问题,复杂模型需极大张量

**神经隐式方法**(DeepSDF)在处理大量不同类别模型时能力有限
4. 大多数方法只处理单个模型或时序相关序列,无法处理多样化不相关的几何集合

## 方法详解

### 两阶段流水线

**阶段一:规则几何表示(RGR)** — 将不规则3D网格模型转化为统一尺寸的规则4D张量
**阶段二:紧凑神经表示(CNR)** — 用量化感知自解码器探索内部和跨模型的几何相似性

### TSDF-Def表示

扩展传统TSDF,为每个网格点引入额外变形:

$$\mathbf{V}(u,v,w) := [\texttt{TSDF}(u,v,w), \Delta u, \Delta v, \Delta w]$$

其中 $\mathbf{V} \in \mathbb{R}^{K \times K \times K \times 4}$,变形量通过**可微变形Marching Cubes (DMC)**在表面提取时使用。

优化目标:
$$\min_{\mathbf{V}} \mathcal{E}_{Rec}(\texttt{DMC}(\mathbf{V}), \mathbf{S}) + \lambda_{Reg}\|\mathbf{V}[...,1:3]\|_1$$

L1正则化减少不必要的变形,因为大多数区域用TSDF即可精确表示。

### 量化感知自解码器

每个模型 $\mathbf{V}_i$ 对应一个嵌入特征 $\mathbf{F}_i \in \mathbb{R}^{K' \times K' \times K' \times C}$,$K' \ll K$:

$$\widehat{\mathbf{V}}_i = \mathcal{D}_{\mathcal{Q}(\boldsymbol{\Theta})}(\mathcal{Q}(\mathbf{F}_i))$$

集成可微量化 $\mathcal{Q}(\cdot)$ 到训练中减少量化误差。

### 损失函数

$$\mathcal{L}(\widehat{\mathbf{V}}_i, \mathbf{V}_i) = \|\widehat{\mathbf{V}}_i - \mathbf{V}_i\|_1 + \lambda_1\|\mathbf{M}_i \odot (\widehat{\mathbf{V}}_i - \mathbf{V}_i)\|_1 + \lambda_2(1 - \texttt{SSIM}(\widehat{\mathbf{V}}_i, \mathbf{V}_i))$$

其中 $\mathbf{M}_i$ 为表面附近网格的掩码,给予更高权重。

### 熵编码

嵌入特征和网络参数经量化后通过Huffman编码压缩为比特流。

## 实验

### 不同数据集上的压缩效率

| 方法 | 压缩时间(h) | 解压时间(ms) |
|------|-----------|------------|
| GPCC | 0.62 | 562.56 |
| VPCC | 39.34 | 762.87 |
| PCGCv2 | 1.76 | 100.32 |
| Draco | 0.06 | 365.18 |
| **NeCGS** | 10.01 | **98.95** |

### TSDF vs TSDF-Def消融

| 表示 | CD↓ | NC↑ | F1-0.005↑ |
|------|-----|-----|-----------|
| TSDF K=64 | 较高 | 较低 | 较低 |
| TSDF K=128 | 中 | 中 | 中 |
| **TSDF-Def K=64** | 低 | 高 | 高 |
| **TSDF-Def K=128** | **最低** | **最高** | **最高** |

### 关键发现

1. NeCGS在DT4D数据集上压缩比达到近**900倍**,仍保持细节
2. TSDF-Def在低分辨率(K=64)下就能保持薄结构的细节,而传统TSDF在K=128时仍丢失
3. 解压速度最快(98.95ms),这对下游应用至关重要
4. 支持动态场景:新模型可不断加入已压缩集合

## 亮点与洞察

1. **TSDF-Def的精妙设计** — 引入网格点变形使低分辨率张量也能表示精细结构,统一了不同复杂度模型的表示尺寸
2. **集合级压缩** — 利用跨模型的几何相似性实现远超单模型压缩的效率
3. **量化感知训练** — 将量化集成到训练中减少量化误差
4. **增量能力** — 支持动态添加新模型,实用性强

## 局限性

- 压缩时间较长(10小时),属于离线压缩
- 在类别差异大的混合数据集上性能有所下降
- 解码器架构固定,不同压缩比通过调整解码器大小实现

## 相关工作

- **单模型压缩**: GPCC, VPCC, Draco, PCGCv2
- **序列压缩**: SLRMA, SMPL/SMAL驱动
- **神经隐式表示**: DeepSDF, 各种SDF/UDF方法

## 评分

- 新颖性: ⭐⭐⭐⭐ (TSDF-Def + 集合级神经压缩)
- 技术深度: ⭐⭐⭐⭐ (两阶段设计完整,量化感知训练)
- 实验充分度: ⭐⭐⭐⭐⭐ (四个数据集,全面消融)
- 实用价值: ⭐⭐⭐⭐ (900x压缩比,支持动态添加)

<!-- RELATED:START -->

## 相关论文

- [LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression](linr-pcgc_lossless_implicit_neural_representations_for_point_cloud_geometry_comp.md)
- [Geometry Distributions](geometry_distributions.md)
- [Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs](compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)
- [Learning 3D Scene Analogies with Neural Contextual Scene Maps](learning_3d_scene_analogies_with_neural_contextual_scene_maps.md)
- [GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)

<!-- RELATED:END -->
