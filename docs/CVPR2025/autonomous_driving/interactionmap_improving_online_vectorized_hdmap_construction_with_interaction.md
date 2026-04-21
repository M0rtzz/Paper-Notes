---
title: >-
  [论文解读] InteractionMap: Improving Online Vectorized HDMap Construction with Interaction
description: >-
  [CVPR 2025][自动驾驶][HD地图构建] 本文提出InteractionMap，通过点级和实例级关系嵌入、关键帧分层时序融合和几何感知分类-定位对齐三个模块，全面增强在线矢量化HD地图构建中的信息交互，在nuScenes (mAP 71.8) 和Argoverse2 (mAP 74.7) 上均取得SOTA。
tags:
  - CVPR 2025
  - 自动驾驶
  - HD地图构建
  - 在线矢量化
  - 时序融合
  - 关系嵌入
  - 几何感知对齐
  - DETR
---

# InteractionMap: Improving Online Vectorized HDMap Construction with Interaction

**会议**: CVPR 2025  
**arXiv**: [2503.21659](https://arxiv.org/abs/2503.21659)  
**代码**: 暂无  
**领域**: 自动驾驶  
**关键词**: HD地图构建, 在线矢量化, 时序融合, 关系嵌入, 几何感知对齐, DETR

## 一句话总结

本文提出InteractionMap，通过点级和实例级关系嵌入、关键帧分层时序融合和几何感知分类-定位对齐三个模块，全面增强在线矢量化HD地图构建中的信息交互，在nuScenes (mAP 71.8) 和Argoverse2 (mAP 74.7) 上均取得SOTA。

## 研究背景与动机

高精度矢量化HD地图是自动驾驶系统的核心组件，包含车道线、道路边界、人行横道等实例级信息。传统HD地图依赖LiDAR SLAM离线构建，维护成本高且更新困难。近年来，基于DETR的端到端矢量化方法（如MapTR、MapTRv2）成为主流，但仍存在三个关键问题：

1. **点集表示的局限性**：DETR-like检测器中的点集表示对实例级信息建模能力有限，缺乏显式的几何先验利用
2. **时序一致性不足**：单帧预测在复杂场景（如车辆遮挡）下表现不稳定，现有streaming策略受限于GRU容量导致长时记忆遗忘
3. **分类-定位不对齐**：分类分支和回归分支独立优化，导致高置信度预测可能对应低定位质量的输出

## 方法详解

### 整体框架

InteractionMap在DETR-like地图矢量化框架上引入三个核心模块：
- **Relation Embedding Module (REM)**：在decoder的self-attention中注入点级和实例级的显式几何关系先验
- **Temporal Fusion Module (TFM)**：基于关键帧的分层时序融合，从局部到全局整合时序信息
- **Geometry-aware Alignment Module (GAM)**：通过几何感知分类损失和匹配代价解决分类-定位不对齐问题

### 关键设计

**1. 关键帧分层时序融合 (KFS)**

借鉴机器人导航中的关键帧策略，将时序融合分为两级：
- **局部BEV融合**：使用GRU循环融合相邻帧BEV特征，保持局部时序一致性。通过ego位姿变换对齐前一帧的BEV特征：$\mathcal{F}_{submap}^t = ResBlock(LN(GRU(\tilde{\mathcal{F}}_{submap}^{t-1}, \mathcal{F}_{local}^t)))$
- **全局BEV融合**：基于距离步长 $d_{stride}$ 而非固定时间间隔选取关键帧，提供两种策略：KFS-streaming（递归GRU融合）和KFS-stacking（拼接+ResBlock融合）

**2. 关系嵌入模块 (REM)**

在decoder的解耦self-attention中加入显式几何关系先验：
- **点级关系嵌入 (PRE)**：基于归一化坐标差和边方向余弦相似度差，编码点与点之间的空间和方向关系
- **实例级关系嵌入 (IRE)**：基于分类分数排名关系和实例间Chamfer距离，建立实例之间的语义-几何联合关系

两者均具有无偏性（$i=j$ 时关系值为0），通过正弦位置编码+线性变换+ReLU提升到高维空间。

**3. 几何感知对齐 (GAM)**

受2D检测中IoU-aware Focal Loss启发，提出三种几何感知分类分数 (GCS)：
- $s_{p2p}$：归一化的点到点L1距离分数
- $s_{dir}$：边方向余弦相似度分数  
- $s_{giou}$：归一化的GIoU分数

在Focal Loss中，前景目标使用GCS作为soft target，取代硬标签1：

$$\mathcal{L}_{GFL} = \sum_{i=1}^{N_{pos}} s_{geo_i} BCE(s_{geo_i}, p_i) + \sum_{j=1}^{N_{neg}} \alpha p_j^{\gamma} BCE(p_j, 0)$$

同时在匹配代价中引入GFC (Geometry-aware Focal Cost)，抑制定位不准确的候选。

### 损失函数

总损失由检测损失、分割损失和辅助损失组成：
- **检测损失**：$\mathcal{L}_{det} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{p2p}\mathcal{L}_{p2p} + \lambda_{dir}\mathcal{L}_{dir}$
- **分割损失**：基于query的实例分割（MGFL + Dice Loss），提供BEV级supervision
- **辅助损失**：深度预测 + 透视图语义分割

## 实验关键数据

### nuScenes验证集 (24 epochs, R50)

| 方法 | 时序 | AP_ped | AP_div | AP_bou | mAP |
|------|------|--------|--------|--------|-----|
| MapTRv2 | ✗ | 59.8 | 62.4 | 62.4 | 61.5 |
| HIMap | ✗ | 62.6 | 68.4 | 69.1 | 66.7 |
| HRMapNet | ✓ | 65.8 | 67.4 | 68.5 | 67.2 |
| **InteractionMap-R** | **✓** | **71.3** | **70.8** | **72.8** | **71.6** |
| **InteractionMap-C** | **✓** | **69.7** | **72.7** | **73.0** | **71.8** |

相比MapTRv2 (24ep) 提升 **+10.3 mAP**，相比HRMapNet提升 **+4.6 mAP**。

### Argoverse2验证集 (110 epochs, R50)

| 方法 | AP_ped | AP_div | AP_bou | mAP |
|------|--------|--------|--------|-----|
| MapTRv2 | 68.1 | 68.3 | 69.7 | 68.7 |
| HIMap | 71.3 | 75.0 | 74.7 | 73.7 |
| **InteractionMap-C** | **73.8** | **75.5** | **74.9** | **74.7** |

## 亮点与洞察

1. **系统性的信息交互设计**：从点级→实例级的渐进式关系建模，以及从局部→全局的分层时序融合，体现了对HD地图任务特性的深入理解
2. **几何感知对齐**：将2D检测中成熟的分类-定位对齐方法迁移到地图矢量化任务，首次解决该领域长期忽视的misalignment问题
3. **关键帧策略的引入**：基于距离步长选取关键帧比固定时间间隔更符合自动驾驶场景的实际需求（低速时不需要频繁更新，高速时需要更多帧）
4. **显著的性能提升**：在nuScenes 24 epochs条件下比MapTRv2提升超过10个mAP点，证明信息交互的重要性

## 局限性

1. 仅使用纯视觉输入（6相机），未探索LiDAR融合场景
2. KFS策略引入额外的BEV特征存储开销，实际部署中需要权衡内存限制
3. 消融实验中KFS-streaming和KFS-stacking在不同数据集上各有优劣，缺乏自适应选择机制
4. 关系嵌入模块增加了decoder的计算复杂度，可能影响实时性

## 相关工作

- **在线HD地图构建**：MapTR → MapTRv2 → BeMapNet → PivotNet → MapQR → MGMap → HIMap
- **时序融合策略**：StreamMapNet (streaming)、SQD-MapNet (query denoising)、MapTracker (tracking策略)
- **地图元素交互**：ADMap (级联交互)、InsightMapper (内部实例聚合)、GeMap (解耦自注意力)、HoMap (高阶建模)
- **分类-定位对齐**：VFL (IoU-aware focal loss)、LD (localization distillation)

## 评分

- **新颖性**：3/5 — 各模块均有一定的已有工作基础，但组合和适配到HD地图构建场景的设计较为系统
- **有效性**：5/5 — 在两个主流数据集上均取得显著的SOTA提升
- **清晰度**：4/5 — 方法描述清晰，数学推导完整
- **意义**：4/5 — 对HD地图构建领域有实质性推动

<!-- RELATED:START -->

## 相关论文

- [MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction](uncertainty-instructed_structure_injection_for_generalizable_hd_map_construction.md)
- [Stream Query Denoising for Vectorized HD-Map Construction](../../ECCV2024/autonomous_driving/stream_query_denoising_for_vectorized_hd-map_construction.md)
- [Online Video Understanding: OVBench and VideoChat-Online](online_video_understanding_ovbench_and_videochat-online.md)
- [DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction](decoupledgaussian_object-scene_decoupling_for_physics-based_interaction.md)

<!-- RELATED:END -->
