---
title: >-
  [论文解读] ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation
description: >-
  [ECCV 2024][自动驾驶][半监督学习] 提出IT2框架，通过利用LiDAR数据的对等表示（range image + voxel grid）之间的一致性学习作为新型扰动形式，并引入基于高斯混合模型的跨分布对比学习，大幅提升半监督LiDAR语义分割性能。
tags:
  - ECCV 2024
  - 自动驾驶
  - 半监督学习
  - LiDAR语义分割
  - 对等表示一致性
  - 对比学习
  - 高斯混合模型
---

# ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation

**会议**: ECCV 2024  
**arXiv**: [2407.07171](https://arxiv.org/abs/2407.07171)  
**代码**: https://github.com/yyliu01/IT2 (有)  
**领域**: 自动驾驶 / 3D语义分割  
**关键词**: 半监督学习, LiDAR语义分割, 对等表示一致性, 对比学习, 高斯混合模型

## 一句话总结

提出IT2框架，通过利用LiDAR数据的对等表示（range image + voxel grid）之间的一致性学习作为新型扰动形式，并引入基于高斯混合模型的跨分布对比学习，大幅提升半监督LiDAR语义分割性能。

## 研究背景与动机

**领域现状**：户外LiDAR语义分割是自动驾驶的基础任务。当前方法通过将点云转换为不同表示（range image、voxel grid、BEV等）来提取特征，但高度依赖大规模标注数据。LiDAR点云标注极其耗时昂贵，推动了半监督学习（SSL）方法的发展。

**现有痛点**：

**一致性学习局限**：现有SSL方法（如LaserMix、Mean Teacher）仅在**单一LiDAR表示**上进行一致性学习，扰动形式有限（数据增强、网络扰动），泛化能力受限。不同表示各有弱点——range view在投影过程中丢失信息（如Pole类别mIoU仅52.02），voxel grid在远距离密集区域精度差（如Sidewalk类别mIoU仅69.50）。

**对比学习瓶颈**：现有对比学习方法从有限的mini-batch中随机采样正负embedding对，无法全面理解整个嵌入空间的分布。多表示场景下，一个表示的噪声预测还会通过对比学习传导到另一个表示，造成确认偏差。

**核心矛盾**：同一3D点云在不同表示下应有相同的语义——range image和voxel grid中同一点的"terrain"标签一定相同——但它们的预测置信度和错误模式截然不同。这种互补性被现有单表示SSL方法完全忽视。

**本文切入角度**：将不同LiDAR表示之间的差异视为一种天然的"扰动"——不同于传统的数据增强扰动，对等表示扰动本质上是对同一语义信息的不同观察方式，提供了更有效的一致性学习信号。

**核心idea**：通过对等表示之间的交叉伪标签监督实现一致性学习（range预测监督voxel，反之亦然），同时用GMM建模跨表示的嵌入分布来采样高信息量的对比学习样本。

## 方法详解

### 整体框架

IT2同时训练两个网络：range网络（FIDNet/ResNet34）处理range image，voxel网络（Cylinder3D）处理voxel grid。同一点云被同时转换为两种表示送入各自网络。在无标签数据上，一个表示的预测通过表示转换映射到另一个表示空间，作为伪标签进行交叉监督。同时，两个网络的嵌入空间通过GMM建模后进行跨分布对比学习。

### 关键设计

1. **对等表示一致性学习 (Peer-Representation Consistency)**：核心思路是将voxel网络的预测投影到range图像空间（或反向），作为对方的伪标签进行交叉监督。具体通过表示间的双向投影函数实现：

   $$\tilde{\mathbf{y}}^r(\omega^r) = \text{argmax}\ \Psi_{v \to r}^{(\omega^r)}(\hat{\mathbf{y}}^v), \quad \tilde{\mathbf{y}}^v(\omega^v) = \text{argmax}\ \Psi_{r \to v}^{(\omega^v)}(\hat{\mathbf{y}}^r)$$

   其中 $\Psi_{v \to r} = \psi_{p \to r} \circ \psi_{v \to p}$ 通过3D点空间中转完成表示间的投影。总损失为两个表示的有标签+无标签损失之和：

   $$\ell_{\text{IT2}} = \ell_{\text{range}} + \ell_{\text{voxel}}$$

   设计动机：不同表示对同一场景的错误模式互补——range view对细长物体（Pole）弱，voxel对远距离密集物体（Sidewalk）弱。交叉监督利用这种互补性，减少单表示伪标签的确认偏差。

2. **跨分布对比学习 (Cross-Distribution Contrastive Learning)**：不同于传统从mini-batch采样嵌入的做法，IT2使用**类别级高斯混合模型（GMM）**建模两个表示的联合嵌入空间。GMM参数通过EM算法优化，其中**伪标签置信度**作为权重因子降低低质量标签的影响：

   $$\mathbf{P}_{\Gamma^y}(\mathbf{z}|y) = \sum_{m=1}^{M} \boldsymbol{\pi}_m^y \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_m^y, \boldsymbol{\Sigma}_m^y)$$

   每个类别使用 $M=5$ 个高斯分量。训练时，从GMM分布中采样虚拟的正负原型样本，而非从实际训练样本中随机选取：

   $$\ell_{\text{cross}} = \sum_{(\mathbf{z},y) \in \mathcal{A}} \sum_{\mathbf{s} \in \mathcal{S}^y} -\log \frac{\exp(\mathbf{z} \cdot \mathbf{s} / \tau)}{\exp(\mathbf{z} \cdot \mathbf{s} / \tau) + \sum_{\mathbf{s}^- \in \bar{\mathcal{S}}^y} \exp(\mathbf{z} \cdot \mathbf{s}^- / \tau)}$$

   设计动机：GMM能学习到嵌入空间的完整分布特征（多模态、协方差结构），采样出的虚拟原型比随机采样更具代表性和信息量，且不参与反向传播，避免了跨表示噪声传导。

3. **表示特定数据增强 (Representation-Specific Augmentation)**：不同表示适合不同的增强策略——range image使用multi-boxes CutMix，voxel使用single-inclination LaserMix。不同于以往对所有表示使用相同增强的做法。

   设计动机：range image是2D投影，适合2D空间切割增强；voxel是3D结构，适合基于激光束角度的3D增强。适配的增强策略能带来更好的泛化。

### 损失函数 / 训练策略

- 总训练损失：$\mathcal{L} = \ell_{\text{IT2}} + \ell_{\text{cross}}$，端到端优化
- 分割损失 $\ell$ 由交叉熵 + Lovász-Softmax组成
- 对比学习温度系数 $\tau = 0.1$
- 嵌入投影器为3层MLP，输出64维
- GMM组件数 $M=5$，均匀权重 $\pi_m^y = 1/M$

## 实验关键数据

### 主实验（Uniform Sampling）

**nuScenes数据集 (mIoU%)**：

| 方法 | 表示 | 1% | 10% | 20% | 50% |
|------|------|-----|------|------|------|
| LaserMix | Range | 49.5 | 68.2 | 70.6 | 73.0 |
| **IT2 (Ours)** | **Range** | **56.5** | **71.3** | **73.4** | **74.0** |
| LaserMix | Voxel | 55.3 | 69.9 | 71.8 | 73.2 |
| **IT2 (Ours)** | **Voxel** | **57.5** | **72.1** | **73.6** | **74.1** |

**SemanticKITTI + ScribbleKITTI (mIoU%)**：

| 方法 | 表示 | KITTI 1% | KITTI 10% | Scribble 1% | Scribble 10% |
|------|------|---------|----------|------------|-------------|
| LaserMix | Range | 43.4 | 58.8 | 38.3 | 54.4 |
| **IT2** | **Range** | **51.9** | **60.3** | **46.6** | **57.1** |
| LaserMix | Voxel | 50.6 | 60.0 | 44.2 | 53.7 |
| **IT2** | **Voxel** | **52.0** | **61.4** | **47.9** | **56.7** |

在nuScenes 1%标签条件下，Range提升+7.0%，Voxel提升+2.2%。ScribbleKITTI 1%标签下Range提升+8.3%。

### 消融实验

**各组件贡献（nuScenes + ScribbleKITTI, Range表示）**：

| IT2架构 | 对比学习 | 增强 | nuScenes 10% | ScribbleKITTI 10% | 说明 |
|---------|---------|------|-------------|-------------------|------|
| ✗ | ✗ | ✗ | 60.8 | 50.0 | CPS基线 |
| ✓ | ✗ | ✗ | 67.3 (+6.5) | 53.2 (+3.2) | 对等表示一致性 |
| ✓ | ✓ | ✗ | 70.3 (+9.5) | 54.9 (+4.9) | +跨分布对比学习 |
| ✓ | ✓ | ✓ | **71.3 (+10.5)** | **57.1 (+7.1)** | +表示特定增强 |

三个组件均有明显贡献，其中IT2架构本身贡献最大（+6.5%），对比学习在此基础上进一步提升约3%，增强策略再贡献约1-2%。

### 关键发现

- **对比学习对比**：与ContrasSeg（像素级对比学习SOTA）相比，GMM-based方法在nuScenes 10%上提升约1% mIoU（range: 71.3 vs 70.3, voxel: 72.1 vs 71.2），证明分布采样优于随机采样
- **温度系数**：$\tau=0.10$ 为最优值（在nuScenes 10%上），过大或过小均导致性能退化
- **多表示扩展性**：加入BEV（PolarNet）作为第三表示，能进一步提升性能（与range搭配+6.3%，与voxel搭配+7.8%），证明方法可扩展到任意表示组合
- **Partial/Significant Sampling**：在SemanticKITTI partial 5%上超GPC +4.5%（43.8 vs 40.2），在significant 10%上超lim3D +2.2%
- **增强策略**：range用multi CutMix + voxel用1-inc LaserMix的组合最优，比统一使用LaserMix在range上+1.3%

## 亮点与洞察

- 核心洞察出色：不同LiDAR表示是同一3D场景的不同"观察"，天然满足聚类假设——这是一种比数据增强更本质的扰动形式
- 利用GMM建模嵌入分布再采样是非常优雅的方案——既能覆盖完整分布，又通过置信度加权避免噪声伪标签的影响
- 跨表示的交叉伪标签监督巧妙地利用了表示间的互补性，减少了单表示SSL固有的确认偏差
- 方法框架的通用性好：可扩展到任意LiDAR表示组合（range+voxel+BEV），且均带来显著提升

## 局限与展望

- 使用两个表示意味着两个独立网络，训练计算开销翻倍
- 表示间的投影映射依赖于精确的点云坐标，传感器噪声可能影响跨表示标签质量
- GMM假设嵌入分布为高斯混合，对于某些复杂类别可能不够灵活
- 可探索：将Camera模态引入作为第三种表示，实现LiDAR-Camera SSL融合
- 可探索：将对等表示一致性思路推广到其他SSL领域（如医学影像的多模态SSL）

## 相关工作与启发

- LaserMix在单表示上提出了激光束混合增强，IT2在此基础上将"表示间差异"本身作为一种更高层次的扰动
- GPC和lim3D的对比学习受限于实际样本，IT2用GMM虚拟原型突破了这一瓶颈
- CPS（Cross Pseudo Supervision）在2D图像SSL中用两个相同网络交叉监督，IT2将其扩展为两个**不同表示**的交叉监督，更自然地实现了扰动多样性
- 启发：多视角/多模态数据的天然互补性是半监督学习的宝贵资源，比人工增强更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ 对等表示一致性学习和GMM-based对比学习均为有意义的创新
- 实验充分度: ⭐⭐⭐⭐⭐ 3个数据集×3种采样策略×多种消融+扩展表示+对比学习对比，极其充分
- 写作质量: ⭐⭐⭐⭐ 动机分析透彻（Fig.1的互补性示意），数学公式清晰完整
- 价值: ⭐⭐⭐⭐ 对LiDAR SSL领域有显著推动，在nuScenes 1%设置下+7% mIoU的提升具有实际意义

<!-- RELATED:START -->

## 相关论文

- [CSOT: Cross-Scan Object Transfer for Semi-Supervised LiDAR Object Detection](csot_cross-scan_object_transfer_for_semi-supervised_lidar_object_detection.md)
- [Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](../../CVPR2025/autonomous_driving/exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)
- [SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds](sfpnet_sparse_focal_point_network_for_semantic_segmentation_on_general_lidar_poi.md)
- [Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)
- [RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D LiDAR Segmentation](rapid-seg_range-aware_pointwise_distance_distribution_networks_for_3d_lidar_segm.md)

<!-- RELATED:END -->
