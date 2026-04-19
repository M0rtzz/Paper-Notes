---
title: >-
  [论文解读] Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing
description: >-
  [ICCV 2025][3D视觉][3D分割] 提出Gaussian Instance Tracing (GIT)机制，通过反向光栅化为每个高斯核维护跨视角的实例权重矩阵，统一解决2D分割多视角不一致和边界高斯模糊两大问题，在离线对比学习和在线自提示两种设定下均显著提升3D分割质量。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D分割
  - Gaussian Splatting
  - 2D-to-3D提升
  - 多视角一致性
  - 实例追踪
  - 密度控制
---

# Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing

**会议**: ICCV 2025  
**arXiv**: [2508.03227](https://arxiv.org/abs/2508.03227)  
**领域**: 3D视觉  
**关键词**: 3D分割, Gaussian Splatting, 2D-to-3D提升, 多视角一致性, 实例追踪, 密度控制

## 一句话总结

提出Gaussian Instance Tracing (GIT)机制，通过反向光栅化为每个高斯核维护跨视角的实例权重矩阵，统一解决2D分割多视角不一致和边界高斯模糊两大问题，在离线对比学习和在线自提示两种设定下均显著提升3D分割质量。

## 研究背景与动机

将2D基础模型（如SAM）的分割能力提升到3D是当前热门范式，但面临两个核心挑战：

**多视角分割不一致**：SAM等模型在不同视角对同一物体产生不同层级/粒度的mask，导致3D特征监督自相矛盾——同一3D区域在某些视角应分为一组，在另一些视角又应分开

**模糊边界高斯**：标准3DGS重建时不考虑物体语义，导致大量高斯核同时跨越多个物体（边界模糊高斯），提取3D资产时产生严重伪影

现有解决方案各有缺陷：
- GaussianGrouping/CoSSegGaussians用视频跟踪器关联mask，但存在丢失轨迹问题
- FlashSplat/SAGS直接过滤模糊高斯，导致细节丢失
- EgoLifter忽略模糊高斯不做处理
- SAGD事后分解边界高斯，缺乏全局一致性

**核心思路**：高斯核本身在3D中是天然一致的（同一个高斯在所有视角都是同一个3D点），利用这一特性反向追踪每个高斯的实例归属，即可解决2D mask不一致问题。

## 方法详解

### 整体框架

给定输入图像集和SAM生成的mask，方法分三步：(1) GIT计算全局实例权重矩阵；(2) 基于权重矩阵合并不一致的patch；(3) GIT引导的自适应密度控制处理模糊高斯。最后在这些改进基础上做3D分割提升（对比学习或自提示）。

### 关键设计1：Gaussian Instance Tracing (GIT)

**实例patch化**：将每个视角的SAM mask重叠形成不相交的实例patch。

**反向光栅化**：对每个视角的每个像素，追踪哪些高斯核贡献了该像素的渲染，将像素的实例标签按贡献比例反向分配给这些高斯核。

**权重矩阵**：每个高斯 $G_i$ 维护一个 $T \times L$ 的权重矩阵 $\mathbf{W}_i$，其中 $T$ 是最大patch数，$L$ 是视角数。$\mathbf{W}_{i,j}^\nu$ 表示第 $i$ 个高斯在视角 $\nu$ 下属于第 $j$ 个实例的概率。

**高效性**：GIT的计算与前向渲染并行，效率等同于一次正向渲染。

### 关键设计2：一致性实例图生成

对于同一视角下的两个patch，单看当前视角无法判断是否应合并。GIT的解决方案：

1. 追踪两个patch对应的高斯集
2. 找到在其他视角共同可见的高斯对
3. 计算多视角相似度（内积衡量实例概率分布的相似性）：
$$\text{sim}(P_a, P_b) = \frac{1}{|\Omega|} \sum_{(G_a, G_b, \nu) \in \Omega} \langle W_{G_a}^\nu, W_{G_b}^\nu \rangle$$
4. 相似度超过阈值 $\theta=0.5$ 的patch自动合并

这实质上是在3D空间中进行**多数投票**：利用3D高斯的跨视角一致性来纠正2D分割的不一致。

### 关键设计3：GIT引导的自适应密度控制

**模糊高斯检测**：对每个高斯计算模糊度分数：
$$As_i = \frac{1}{|\mathcal{V}_i|} \sum_{\nu \in \mathcal{V}_i} \mathbb{I}(\max_j(W_{i,j}^\nu) < \gamma)$$
若在超过 $\theta_{As}=0.5$ 比例的可见视角中，最大实例概率都低于 $\gamma=0.8$，则该高斯被判定为模糊。

**自适应处理**：
- **分裂**：将模糊高斯一分为二（尺度除以2），按原高斯的PDF采样新位置
- **剪枝**：分裂后仍模糊的高斯被移除
- 每1000迭代重复一次，同时重新训练整个高斯场景

与暴力删除的关键区别：分裂操作给了高斯核"重新选择归属"的机会，避免直接删除导致的表面伪影。

### 3D分割提升

**离线对比学习**：为每个高斯添加16维特征向量，通过对比损失训练：
$$L_{contr} = -\frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \log \frac{\sum_{u' \in \mathcal{U}^+} \exp(\text{sim}(F[u], F[u']; \tau))}{\sum_{u' \in \mathcal{U}} \exp(\text{sim}(F[u], F[u']; \tau))}$$
使用一致性实例图和优化后的高斯集，有效消除训练信号中的矛盾。

**在线自提示**：用户在参考视角给点提示 → SAM生成初始mask → GIT在新视角提取点提示 → 迭代扩展 → 最终从权重矩阵获取完整3D高斯集。

### 损失函数

标准2DGS重建损失 + 对比损失 $L_{contr}$（离线设定时）。

## 实验关键数据

### 3D物体提取（Table 2，Replica数据集）

| 方法 | 平均 mIoU↑ | 平均 PSNR↑ |
|------|:-:|:-:|
| Gaussian Grouping | 29.6 | 13.4 |
| FlashSplat | 39.3 | 16.9 |
| EgoLifter | 55.6 | 20.1 |
| **Ours** | **72.1** | **22.6** |

大幅领先第二名EgoLifter 16.5 mIoU，且渲染质量也更优。

### 新视角2D分割（Table 3，Replica数据集）

| 方法 | 平均 mIoU↑ |
|------|:-:|
| SA3D (NeRF) | 83.0 |
| OmniSeg3D (NeRF) | 84.4 |
| SA3D-GS | 79.1 |
| EgoLifter | 82.1 |
| **Ours** | **85.5** |

首次使GS-based方法超越所有NeRF-based方法。

### NVOS基准（Table 4）

| 方法 | mIoU↑ | mAcc↑ |
|------|:-:|:-:|
| FlashSplat | 91.8 | 98.6 |
| GaussianCut | 92.5 | 98.4 |
| **Ours** | **92.5** | **98.6** |

达到SOTA水平，两项指标均为最优或并列最优。

### 消融实验（Table 5）

| 设置 | NVS mIoU | Object mIoU | PSNR |
|------|:-:|:-:|:-:|
| 2DGS + 无GIT + 无密度控制 | 87.0 | 62.5 | 21.0 |
| 2DGS + 仅密度控制 | 87.3 | 70.1 | 22.4 |
| 2DGS + 仅一致mask | 89.2 | 63.6 | 21.2 |
| **2DGS + 两者结合** | **89.1** | **72.1** | **22.6** |

- 密度控制对3D提取贡献最大（+13.4% object mIoU）
- 一致mask对NVS分割贡献最大（+2.2 mIoU）
- 两者互补效果最优

## 亮点与洞察

1. **反向光栅化的巧妙利用**：正向渲染是像素→高斯的聚合，GIT反向追踪实现高斯→实例的映射，计算代价几乎为零
2. **3D一致性约束2D不一致性**：利用高斯核在3D空间中的不变性反向纠正2D预测，将"数据问题"转化为"几何优势"
3. **分裂优于删除**：对模糊高斯做分裂而非删除，给予重新分配的机会，避免表面破损
4. **统一在线/离线**：GIT机制同时适用于对比学习和自提示两种范式，展现了其通用性
5. **层次化分割能力**：支持不同粒度的物体提取，甚至可以提取物体部件（如Figure 1中美国队长的锤子）

## 局限性

- GIT依赖初始SAM mask的质量，极小或纹理状物体仍有困难
- 一致mesh重建需要合理的视角覆盖，稀疏输入可能导致权重矩阵不可靠
- 对比学习和自提示各有优劣，尚未找到最优统一策略
- 计算开销方面，维护全局权重矩阵对大规模场景的内存需求值得关注

## 相关工作

- **SA3D / SA3D-GS**：在线自提示方法，跨视角迭代SAM但缺乏一致性保障
- **OmniSeg3D**：NeRF上的层次化对比学习，但不处理mask不一致
- **EgoLifter**：对比学习+3DGS但忽略模糊高斯
- **FlashSplat**：通过过滤模糊高斯做二值分割，细节丢失严重
- **GaussianGrouping**：用视频跟踪器关联mask，但轨迹丢失问题突出
- **GaussianEditor**：也用反向渲染做语义编辑，但只用于单视角编辑而非全局一致性

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — GIT机制新颖且通用，密度控制策略巧妙
- **技术深度**: ⭐⭐⭐⭐ — 方法设计清晰，各模块逻辑自洽
- **实验充分性**: ⭐⭐⭐⭐⭐ — Replica+NVOS+in-the-wild，完整消融，公平对比
- **实用价值**: ⭐⭐⭐⭐⭐ — 3D资产提取和场景编辑的直接应用价值
- **总体推荐**: ⭐⭐⭐⭐⭐ — 3D分割提升领域的重要突破，解决了长期痛点

<!-- RELATED:START -->

## 相关论文

- [PanSt3R: Multi-view Consistent Panoptic Segmentation](panst3r_multi-view_consistent_panoptic_segmentation.md)
- [CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](../../CVPR2025/3d_vision/layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)
- [Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)
- [Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](../../ECCV2024/3d_vision/part2object_hierarchical_unsupervised_3d_instance_segmentation.md)

<!-- RELATED:END -->
