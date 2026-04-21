---
title: >-
  [论文解读] PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion
description: >-
  [ECCV 2024][3D视觉][全景分割提升] 提出 PCF-Lift，通过概率特征嵌入（多元高斯分布）替代确定性特征，结合概率乘积核（PP Kernel）的对比损失和跨视图约束，有效应对2D分割中的不一致分割和不一致ID问题，在 ScanNet 和 Messy Room 数据集上显著超越前沿方法。
tags:
  - ECCV 2024
  - 3D视觉
  - 全景分割提升
  - 概率特征嵌入
  - 对比学习
  - 多视图融合
  - NeRF
---

# PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion

**会议**: ECCV 2024  
**arXiv**: [2410.10659](https://arxiv.org/abs/2410.10659)  
**代码**: [GitHub](https://github.com/Runsong123/PCF-Lift)  
**领域**: 3D视觉  
**关键词**: 全景分割提升, 概率特征嵌入, 对比学习, 多视图融合, NeRF  

## 一句话总结

提出 PCF-Lift，通过概率特征嵌入（多元高斯分布）替代确定性特征，结合概率乘积核（PP Kernel）的对比损失和跨视图约束，有效应对2D分割中的不一致分割和不一致ID问题，在 ScanNet 和 Messy Room 数据集上显著超越前沿方法。

## 研究背景与动机

1. 3D全景分割需要同时预测语义标签和实例标签，是实现场景完整理解的关键任务
2. 3D标注数据稀缺，近期方法转向利用2D基础模型的全景分割结果进行"全景提升"（Panoptic Lifting）

**不一致ID问题**：同一3D物体在不同视图中被2D分割器分配了不同的实例ID

**不一致分割问题**：同一物体在不同视图中被分割为不同的部分（如椅子在视图1被切为两半，在视图2是完整的）
5. 现有方法（Panoptic Lifting、Contrastive Lift）使用确定性特征嵌入，对噪声不够鲁棒
6. 确定性特征无法建模不确定性，面对不一致分割时训练不稳定，在复杂场景中性能显著下降

## 方法详解

### 整体框架

PCF-Lift 基于 TensoRF 架构构建3D全景场，包含语义场、实例场、密度场和颜色场。核心创新集中在实例场的设计：

1. **训练阶段**：从两个视图采样射线，通过体渲染获得概率特征图，使用概率对比损失和跨视图约束优化实例场
2. **推理阶段**：通过多视图物体关联（MVOA）算法提取原型特征集，生成一致的全景分割结果

### 关键设计

#### 模块一：概率特征嵌入

将实例场中每个3D点映射为多元高斯分布随机变量：

$$\mathcal{F} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad \boldsymbol{\Sigma} = \text{diag}(\sigma^{(1)^2}, \sigma^{(2)^2}, \cdots, \sigma^{(N)^2})$$

其中 $\boldsymbol{\mu} \in \mathbb{R}^N$ 为均值向量（中心特征值），$\boldsymbol{\Sigma}$ 为对角协方差矩阵（不确定性）。实例场对每个查询点 $\mathbf{x} \in \mathbb{R}^3$ 预测 $(\boldsymbol{\mu}, \sigma^2) \in \mathbb{R}^{2N}$（实验中 $N=3$）。

两个高斯分布之间的相似度通过 **概率乘积核（PP Kernel）** 度量：

$$K_\rho(\mathcal{F}_i, \mathcal{F}_j) = \left(\prod_{d=1}^{N} \frac{\sigma_i^{(d)^2}/\sigma_j^{(d)^2} + \sigma_j^{(d)^2}/\sigma_i^{(d)^2}}{2}\right)^{-\frac{1}{2}} \exp\left(-\sum_{d=1}^{N} \frac{(\mu_i^{(d)} - \mu_j^{(d)})^2}{4(\sigma_i^{(d)^2} + \sigma_j^{(d)^2})}\right)$$

PP Kernel 输出范围为 $[0, 1]$，与确定性方法使用的 RBF 核相同范围但具有更强的表达能力。

**理论性质（Corollary 1）**：当所有高斯分布的协方差为各向同性且固定（$\Sigma_i = \Sigma_j = \sigma \mathbf{I}$）时，PP Kernel 退化为 RBF 核。因此确定性方法是概率方法的特例。

#### 模块二：概率对比损失与跨视图约束

**像素级对比损失**使用 PP Kernel 替代 RBF 核：

$$\mathcal{L}_{\text{pixel-contra}} = -\frac{1}{|\Omega|} \sum_{u \in \Omega} \log \frac{\sum_{u' \in \Omega} \mathbf{1}_{(u,u')} \exp(K_\rho(\mathcal{F}_u, \mathcal{F}_{u'}))}{\sum_{u' \in \Omega} \exp(K_\rho(\mathcal{F}_u, \mathcal{F}_{u'}))}$$

**集中损失**鼓励同一实例的特征聚集：

$$\mathcal{L}_{\text{concen}} = -\frac{1}{|\Omega|} \sum_{u \in \Omega} \log K_\rho\left(\mathcal{F}_u, \frac{\sum_{u'} \mathbf{1}_{(u,u')} \mathcal{F}_{u'}}{\sum_{u'} \mathbf{1}_{(u,u')}}\right)$$

**跨视图约束**增强不同视图间同一物体的特征一致性：

$$\mathcal{L}_{\text{cross}} = -\frac{1}{|\mathcal{P}|} \sum_{(\mathcal{F}_r, \mathcal{F}_s) \in \mathcal{P}} \log K_\rho(\mathcal{F}_r, \mathcal{F}_s)$$

其中正样本对 $\mathcal{P} = \{(\mathcal{F}_r, \mathcal{F}_s) \mid K_\rho(\mathcal{F}_r, \mathcal{F}_s) > \tau\}$，阈值 $\tau = 0.9$。

#### 模块三：多视图物体关联（MVOA）算法

推理时通过类似 NMS 的贪心算法提取原型特征集 $\mathcal{D}$：

1. **实例分组**：对每个视图，将同一实例ID的像素特征平均为分组特征 $\mathcal{C}_l^p$，并计算特征集中度评分 $\mathcal{S}_l^p = \Phi(\mathcal{C}_l^p)$
2. **多视图匹配**：构建无向相似度图 $G = (\mathcal{C}, E)$，以贪心方式选择评分最高的节点加入原型集 $\mathcal{D}$，并抑制与其相似度超过阈值 $\mathcal{T}$ 的节点
3. **掩码生成**：对任意视图，前景像素根据与 $\mathcal{D}$ 中最相似原型的匹配结果分配实例标签

### 损失函数 / 训练策略

总损失函数：

$$\mathcal{L} = \mathcal{L}_{\text{contra}} + w_{\text{cross}} \mathcal{L}_{\text{cross}} + w_{\text{reg}} \mathcal{L}_{\text{reg}}$$

- $w_{\text{cross}} = 0.05$（仅在最后几个 epoch 生效），前期设为0
- $w_{\text{reg}} = 0.001$，协方差正则化 $\mathcal{L}_{\text{reg}} = \log(\prod_{d=1}^{N} \sigma^{(d)^2})$
- 实例场使用 slow-fast 架构的 5 层浅层 MLP
- 概率特征维度 $N = 3$

## 实验关键数据

### 主实验

**ScanNet 数据集（12个场景）**：

| 方法 | 会议 | 类型 | $\text{SQ}^{\text{scene}}$ | $\text{RQ}^{\text{scene}}$ | $\text{PQ}^{\text{scene}}$ |
|------|------|------|:---:|:---:|:---:|
| DM-NeRF | ICLR'23 | 3D全景分割 | 53.3% | 46.1% | 41.7% |
| PNF | CVPR'22 | 3D全景分割 | 63.0% | 50.7% | 48.3% |
| Panoptic Lifting | CVPR'23 | 2D全景提升 | 73.5% | 65.0% | 58.9% |
| Contrastive Lift | NeurIPS'23 | 2D全景提升 | 75.7% | 63.6% | 62.0% |
| **PCF-Lift (Ours)** | - | 2D全景提升 | **78.5%** | **65.4%** | **63.5%** |

**Messy Room 数据集（平均 PQ^scene）**：

| 方法 | 25物体 | 50物体 | 100物体 | 500物体 | 均值 |
|------|:---:|:---:|:---:|:---:|:---:|
| Panoptic Lifting | 69.4% | 70.5% | 63.1% | 50.0% | 63.2% |
| Contrastive Lift | 77.7% | 75.7% | 68.9% | 53.8% | 69.0% |
| **PCF-Lift** | **81.0%** | **78.9%** | **74.4%** | **59.6%** | **73.4%** |

### 消融实验

| 模型 | 特征空间 | 聚类方法 | $\text{PQ}^{\text{scene}}$ |
|------|---------|---------|:---:|
| (a) Contrastive Lift | 确定性 | HDBSCAN | 69.0% |
| (b) | 确定性 | MVOA | 70.4% |
| (d) | 概率高斯 | MVOA | 72.3% |
| (f) **PCF-Lift** | 概率高斯 + 跨视图约束 | MVOA | **73.4%** |

### 关键发现

1. **概率 vs 确定性**：概率特征嵌入将 PQ 从 70.4% 提升到 72.3%（+1.9%），证明了高斯分布建模不确定性的有效性
2. **MVOA 算法的通用性**：即使用于确定性方法也能带来 +1.4% 的提升（69.0% → 70.4%）
3. **跨视图约束**：进一步提升 +1.1%（72.3% → 73.4%），增强了多视图特征一致性
4. **不确定性分析**：学到的高协方差区域主要分布在实例边界附近，符合直觉
5. **鲁棒性**：在不同2D分割模型和不同噪声水平下均一致优于确定性方法

## 亮点与洞察

- 将概率建模引入全景提升是非常自然且有效的设计，因为2D分割本身就包含大量不确定性
- PP Kernel 的理论分析优雅地证明了概率方法是确定性方法的推广（RBF 核 ⊂ PP 核）
- 跨视图约束在训练后期启用的策略设计合理，避免了早期不可靠特征对引入的噪声
- MVOA 算法作为通用聚类方法，可即插即用地提升其他方法的性能

## 局限性

- 依赖 TensoRF 重建质量，在几何重建失败的区域全景分割也会失效
- 概率特征维度仅为3维，在更复杂场景中可能不够
- 跨视图约束需要额外的双视图采样，增加训练成本
- 仅在室内场景评估，对于室外大规模场景的适用性未知

## 相关工作与启发

- **Contrastive Lift**：使用确定性特征嵌入的对比学习基线，PCF-Lift 的直接改进对象
- **Panoptic Lifting**：通过ID排列拟合学习实例表示，可扩展性受限
- **PP Kernel**：概率乘积核在机器学习中已有研究，本文将其首次应用于全景提升场景
- **启发**：概率特征嵌入的思路可推广到其他需要多视图融合的任务（如3D语义分割、场景编辑）

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion](prodepth_boosting_self-supervised_multi-frame_monocular_depth_with_probabilistic.md)
- [UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning](../../AAAI2026/3d_vision/unic-lift_unified_3d_instance_segmentation_via_contrastive_learning.md)
- [SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields](slotlifter_slot-guided_feature_lifting_for_learning_object-centric_radiance_fiel.md)
- [Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](../../CVPR2025/3d_vision/layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)
- [TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)

<!-- RELATED:END -->
