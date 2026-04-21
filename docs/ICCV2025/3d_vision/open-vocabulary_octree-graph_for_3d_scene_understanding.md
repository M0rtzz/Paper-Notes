---
title: >-
  [论文解读] Open-Vocabulary Octree-Graph for 3D Scene Understanding
description: >-
  [ICCV 2025][3D视觉][开放词汇] 提出 Octree-Graph，一种将自适应八叉树与图结构结合的新颖场景表示，通过时序分组式段合并(CGSM)和实例特征聚合(IFA)获取准确的语义对象，实现高效的开放词汇3D场景理解。
tags:
  - ICCV 2025
  - 3D视觉
  - 开放词汇
  - 3D场景理解
  - 八叉树
  - 场景图
  - 语义分割
---

# Open-Vocabulary Octree-Graph for 3D Scene Understanding

**会议**: ICCV 2025  
**arXiv**: [2411.16253](https://arxiv.org/abs/2411.16253)  
**代码**: [GitHub](https://github.com/wangzg1/Octree-Graph)  
**领域**: 3D视觉  
**关键词**: 开放词汇, 3D场景理解, 八叉树, 场景图, 语义分割

## 一句话总结

提出 Octree-Graph，一种将自适应八叉树与图结构结合的新颖场景表示，通过时序分组式段合并(CGSM)和实例特征聚合(IFA)获取准确的语义对象，实现高效的开放词汇3D场景理解。

## 研究背景与动机

开放词汇3D场景理解对于具身智能体至关重要。现有方法利用预训练VLM进行目标分割并投射到点云上构建3D地图，但存在两个核心问题：

**空间表示效率低**：主流方法基于点云构建3D地图，点云是无序离散坐标，需要大量存储空间，且缺乏显式的占据信息和空间连通性表达，不利于路径规划和文本检索等下游任务。

**语义分割不准确**：现有方法忽略了基础模型在分割和特征提取时的不精确性，导致3D对象分割和语义质量下降。

## 方法详解

### 整体框架

给定RGB-D序列和相机位姿，流程分为四步：
1. 利用VLM提取2D分割提案和语义特征
2. 通过CGSM将segments合并为实例
3. 通过IFA为每个实例聚合特征
4. 构建Octree-Graph进行下游应用

### 时序分组式段合并 (CGSM)

现有段合并策略分为帧级和图级，前者易受噪声影响，后者引入冗余计算。CGSM的核心设计：

- **时序分组**：按时间顺序将帧分为若干组，间隔为 $I$，保留相邻帧的时空细节同时避免全局干扰。
- **语义引导的欠分割过滤**：对于可能包含不同物体的欠分割段 $\mathcal{S}_m$，计算其内部段的语义特征方差，超过阈值 $\tau_u$ 时过滤。
- **动态阈值衰减**：整体相似度 $\phi = \phi_{\text{geo}}^{\text{iou}} + \phi_{\text{geo}}^{\text{ior}} + \phi_{\text{sem}}^{v} + \phi_{\text{sem}}^{c}$，阈值 $\theta_i$ 线性衰减以合并低空间重叠的部分观察段。

### 实例特征聚合 (IFA)

为每个实例融合语义特征，同时考虑代表性和区分性：

$$a_{i,j}^{v} = \cos(\mathbf{f}_{i,j}^{v}, \bar{\mathbf{f}}_{i}^{v}) - \sum_{\mathcal{O}_k \in \mathcal{N}_i} \cos(\mathbf{f}_{i,j}^{v}, \bar{\mathbf{f}}_{k}^{v})$$

其中权重通过softmax归一化。直觉是：距离自身聚类中心越近、距离邻居实例越远的特征，获得更大的融合权重。

### 自适应八叉树

传统八叉树使用立方体体素，对大长宽比物体（如墙壁）需要很深的树才能逼近形状。自适应八叉树根据物体形状调整每个节点的大小：

$$\mathbf{d}_l = (\mathbf{b}_{\max} - \mathbf{b}_{\min}) / 2^l$$

其中 $\mathbf{b}_{\max}$, $\mathbf{b}_{\min}$ 是物体包围盒角点坐标，$l$ 是八叉树深度。

### Octree-Graph 构建

- **节点** $\mathbf{N}_i$：包含语义 $n_i^s$、中心 $n_i^c$ 和自适应八叉树 $n_i^o$
- **边** $\mathbf{E}_{i,j}$：包含语义关系 $e_{i,j}^s$、空间距离 $e_{i,j}^d$ 和3D向量 $\mathbf{e}_{i,j}^v$

## 实验

### 主实验 - 3D语义分割

| 方法 | Replica mIoU↑ | ScanNet mIoU↑ | ScanNet mAcc↑ |
|------|--------------|---------------|---------------|
| ConceptFusion | 0.10 | 0.08 | 0.15 |
| ConceptGraph | 0.18 | 0.16 | 0.28 |
| HOV-SG | 0.231 | 0.222 | 0.431 |
| **Ours** | **0.320** | **0.393** | **0.601** |

在Replica和ScanNet上显著超越所有方法，尤其在ScanNet上比HOV-SG提升+17.1% mIoU和+17.0% mAcc。

### 3D实例分割 (ScanNet200)

| 方法 | AP↑ | AP50↑ | AP25↑ |
|------|-----|-------|-------|
| Mask-Clustering | 12.0 | 23.3 | 30.1 |
| **Ours (z.s.)** | **14.3** | **25.8** | **33.6** |

### 消融实验 - 合并策略

| 合并策略 | mIoU↑ | mAcc↑ |
|---------|-------|-------|
| 帧级 | 0.323 | 0.519 |
| 全局级 | 0.286 | 0.476 |
| **CGSM (I=200)** | **0.356** | **0.574** |

### 路径规划

| 方法 | SR(1.0m) | SR(0.5m) | SR(0.25m) |
|------|----------|----------|-----------|
| HOV-SG | 55.25 | 46.75 | 32.16 |
| **Ours** | **97.88** | **96.88** | **96.38** |

路径规划成功率显著优于HOV-SG，因为 Octree-Graph 支持导航到任意空区域。

## 亮点与洞察

1. 自适应八叉树巧妙解决了传统八叉树对大长宽比物体表示效率低的问题
2. CGSM通过分组策略在局部细节利用和全局冗余避免之间取得平衡
3. IFA通过考虑实例间区分性进行加权聚合，比简单平均更鲁棒
4. 存储空间极大节约：所有自适应八叉树总共仅42KB vs 点云6.8M

## 局限性

- 依赖2D基础模型的分割质量，对小物体和遮挡场景仍存在挑战
- 八叉树深度 $L_{\max}=4$ 可能不足以表示极精细的几何细节
- 路径规划仍可能因八叉树的离散化误占某些空区域

## 相关工作

- ConceptGraph, HOV-SG: 3D场景图方法
- OpenScene, ConceptFusion: 点/网格级3D地图
- OVIR-3D, MaskClustering: 实例级3D地图
- PlenOctrees, OctreeOcc: 八叉树结构用于渲染/语义

## 评分

- 新颖性: ⭐⭐⭐⭐ (自适应八叉树+图结构的混合表示很新颖)
- 技术深度: ⭐⭐⭐⭐ (CGSM和IFA设计精巧)
- 实验充分度: ⭐⭐⭐⭐⭐ (4个任务4个数据集)
- 实用价值: ⭐⭐⭐⭐⭐ (对具身智能体极有价值)

<!-- RELATED:START -->

## 相关论文

- [3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding](3dgraphllm_combining_semantic_graphs_and_large_language_models_for_3d_scene_unde.md)
- [Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding](../../CVPR2025/3d_vision/masked_point-entity_contrast_for_open-vocabulary_3d_scene_understanding.md)
- [Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)
- [ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail](excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)
- [OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](../../AAAI2026/3d_vision/openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)

<!-- RELATED:END -->
