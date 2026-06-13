---
title: >-
  [论文解读] Hierarchical Temporal Context Learning for Camera-based Semantic Scene Completion
description: >-
  [ECCV 2024][自动驾驶][语义场景补全] 针对相机语义场景补全（SSC）中时序信息利用粗糙的问题，提出层级式时序上下文学习（HTCL）范式：先通过跨帧模式亲和度（CPA）度量当前帧与历史帧的细粒度对应关系，再通过基于亲和度的动态精炼（ADR）自适应采样补偿不完整观测，在SemanticKITTI上排名第1，甚至在OpenOccupancy上mIoU超过LiDAR方法。
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "语义场景补全"
  - "时序上下文学习"
  - "跨帧亲和度"
  - "可变形卷积"
  - "占据预测"
---

# Hierarchical Temporal Context Learning for Camera-based Semantic Scene Completion

**会议**: ECCV 2024  
**arXiv**: [2407.02077](https://arxiv.org/abs/2407.02077)  
**代码**: [https://github.com/Arlo0o/HTCL](https://github.com/Arlo0o/HTCL)  
**领域**: 自动驾驶  
**关键词**: 语义场景补全, 时序上下文学习, 跨帧亲和度, 可变形卷积, 占据预测

## 一句话总结
针对相机语义场景补全（SSC）中时序信息利用粗糙的问题，提出层级式时序上下文学习（HTCL）范式：先通过跨帧模式亲和度（CPA）度量当前帧与历史帧的细粒度对应关系，再通过基于亲和度的动态精炼（ADR）自适应采样补偿不完整观测，在SemanticKITTI上排名第1，甚至在OpenOccupancy上mIoU超过LiDAR方法。

## 研究背景与动机

**领域现状**：3D语义场景补全（SSC）是自动驾驶中的关键密集感知任务，要求从有限的传感器观测中联合推断场景的几何结构和语义信息。出于成本优势，基于相机的SSC方案受到广泛关注，代表方法有MonoScene、TPVFormer、OccFormer和VoxFormer等。

**现有痛点**：早期方法仅使用单帧信息，观测信息极为有限。VoxFormer-T首次引入时序建模，但方法非常粗糙——直接将多帧图像堆叠（stack）后聚合。这种朴素的时序建模存在两个核心问题：

**语义位置漂移**：不同视角下相同语义内容经历了不确定的位置变化，直接聚合会导致预测信息模糊

**冗余信息干扰**：堆叠方法假设不同帧在像素级天然对应，未区分有效信息与冗余信息

**核心矛盾**：SSC是一个细粒度的密集感知任务，需要对每个体素进行语义预测。粗粒度的时序聚合不仅无法提供有效补充，反而增加了学习难度。时序3D检测（区域级粗粒度）和视频深度估计（匹配任务）的策略不适用于SSC场景。

**切入角度**：将时序上下文建模分解为两个层级步骤——先度量"哪些时序信息是相关的"（亲和度测量），再"如何有效利用相关信息"（动态精炼采样）。

**核心idea**：构建层级式时序上下文学习，通过尺度感知隔离和多组独立学习器进行细粒度跨帧对应建模，再基于亲和度权重引导可变形3D卷积自适应精炼采样位置。

## 方法详解

### 整体框架
HTCL框架由三个主要部分组成：
- **对齐时序体积构建（上支路）**：通过PoseNet估计相对位姿，利用单应性变换（Homography Warping）对齐历史帧特征
- **体素特征体积构建（下支路）**：EfficientNetB7 UNet提取当前帧特征，扩展LSS范式构建体素特征体积
- **可靠时序聚合**：通过CPA和ADR提取可靠时序信息，加权体素交叉注意力（WVA）聚合到体素特征中

### 关键设计

#### 1. 时序内容对齐（TCA）：特征体积 vs 代价体积

**功能**：构建时序对齐的特征表示，为后续亲和度计算提供基础。

**核心思路**：不同于视频深度估计中的代价体积（cost volume），SSC是密集感知重建问题而非匹配任务，因此保留细粒度特征上下文而非计算匹配代价。

**具体实现**：
1. 当前帧和历史帧通过轻量PoseNet估计相对位姿
2. 历史帧特征通过单应性变换（Homography Warping）对齐到当前帧坐标系：
$$\text{Warp}(\mathbf{p}) = \mathbf{K}_i \cdot (\mathbf{R}_{0,i} \cdot (\mathbf{K}_0^{-1} \cdot \mathbf{p} \cdot d_j) + \mathbf{t}_{0,i})$$
3. 当前特征沿深度维度lift，与warped历史特征拼接构成时序特征体积 $V_{tem}$

**设计动机**：保留特征上下文使后续模块能充分利用细粒度语义信息，消融实验证明特征体积比代价体积提升1.11 mIoU。

#### 2. 跨帧模式亲和度（CPA）：细粒度对应建模

**功能**：度量历史帧与当前帧之间区域级的语义对应关系，筛选出最相关的时序上下文。

**核心思路**：改进余弦相似度，引入尺度感知隔离（Scale-aware Isolation）和多组独立学习（Multi-group Context），克服传统余弦相似度在密集分布上的局限。

**具体实现**：
1. **多组上下文生成**：对历史和当前特征体积分别使用不同膨胀率（1, 2, 4）的3D空洞卷积生成多组上下文特征：
$$\mathbf{H}_i = \text{GN}(\delta(\text{Atrous}_i(V_{tem}^{his})))$$
2. **尺度感知隔离的相似度计算**：在每个组尺度内减去各自均值后计算余弦相似度：
$$\mathbf{A}_i = \frac{\sum_{j=0}^{C}(\mathbf{C}_i^j - \bar{\mathbf{C}}_i)(\mathbf{H}_i^j - \bar{\mathbf{H}}_i)}{\sqrt{\sum_{j=0}^{C}(\mathbf{C}_i^j - \bar{\mathbf{C}}_i)^2} \sqrt{\sum_{j=0}^{C}(\mathbf{H}_i^j - \bar{\mathbf{H}}_i)^2}}$$
3. **聚合**：沿通道维度拼接多组亲和度矩阵得到跨帧模式亲和度 $\hat{\mathbf{A}}$

**设计动机**：
- 尺度感知隔离：消除规模偏差，避免原始余弦相似度对不相似向量产生高分的问题
- 多组独立学习器：类似集成学习，不同膨胀率关注不同空间尺度的模式，增强密集分布下的区分能力

#### 3. 基于亲和度的动态精炼（ADR）

**功能**：利用亲和度权重自适应精炼采样位置，动态补偿不完整观测。

**核心思路**：在高亲和度位置及其邻域进行可变形采样，结合亲和度权重加权特征。

**关键公式**：
$$V_{def} = \sum_{k=1}^{K_w} w_k \cdot V_{tem}(\mathbf{p} + \mathbf{p}_k + \Delta \mathbf{p}_k) \cdot a_k$$
其中 $\Delta \mathbf{p}_k$ 是可变形偏移量，$w_k$ 是空间权重，$a_k$ 是亲和度权重。

通过三层级联3D可变形卷积构建多层级精炼块，输出聚合为可靠时序体积：
$$\widetilde{V}_{tem} = W(\text{Concat}\{V_{def}^1, V_{def}^2, V_{def}^3\})$$

#### 4. 加权体素注意力（WVA）

**功能**：将可靠时序信息聚合到体素特征中。

整合公式：$V_{ret} = \alpha \cdot \text{CrossAtt}(Q, K, V) + V_{vox}$

其中可学习系数 $\alpha$ 初始化为0，训练中逐渐增大，避免训练初期不成熟的时序信息干扰体素特征学习。

### 损失函数 / 训练策略

使用标准SSC损失（交叉熵+Scene-Class Affinity损失），AdamW优化器，学习率 $1 \times 10^{-4}$，权重衰减0.01，训练24个epoch，批大小4。默认使用当前帧+前3个历史帧。

提供两种配置：
- **HTCL-S**：基于立体深度估计的stereo pipeline
- **HTCL-M**：基于单目深度估计的mono pipeline

## 实验关键数据

### 主实验

**SemanticKITTI测试集**：

| 方法 | 输入 | IoU↑ | mIoU↑ | car | road | building | vegetation |
|------|------|--------|--------|------|-------|----------|-----------|
| MonoScene | M | 34.16 | 11.08 | 18.8 | 54.7 | 14.4 | 14.9 |
| TPVFormer | M | 34.25 | 11.26 | 19.2 | 55.1 | 14.8 | 13.9 |
| SurroundOcc | M | 34.72 | 11.86 | 20.6 | 56.9 | 15.2 | 14.9 |
| OccFormer | M | 34.53 | 12.32 | 21.6 | 55.9 | 15.7 | 16.8 |
| VoxFormer-S | S | 42.95 | 12.20 | 20.8 | 53.9 | 19.8 | 22.4 |
| VoxFormer-T | S-T | 43.21 | 13.41 | 21.7 | 54.1 | 23.5 | 24.4 |
| **HTCL-S** | **S-T** | **44.23** | **17.09** | **27.3** | **64.4** | **25.9** | **25.3** |

相比VoxFormer-T提升：IoU +1.02, **mIoU +3.68**（相对提升27.4%）

**OpenOccupancy验证集**（超越LiDAR方法）：

| 方法 | 输入 | IoU↑ | mIoU↑ |
|------|------|--------|--------|
| LMSCNet (LiDAR) | L | 27.3 | 11.5 |
| JS3C-Net (LiDAR) | L | 30.2 | 12.5 |
| MonoScene (Camera) | M | 18.4 | 6.9 |
| TPVFormer (Camera) | M | 15.3 | 7.8 |
| **HTCL-M (Camera)** | **M-T** | **21.4** | **14.1** |

HTCL-M的mIoU(14.1)超过LiDAR方法JS3C-Net(12.5)和LMSCNet(11.5)！

### 消融实验

**架构组件消融（SemanticKITTI验证集）**：

| 组件变更 | IoU(%) | mIoU(%) | 说明 |
|---------|--------|---------|------|
| 完整HTCL-S | **45.51** | **17.13** | 全部组件 |
| Cost Volume替换Feature Volume | 43.07 | 15.18 | TCA: 特征体积优于代价体积(+1.95) |
| 无Scale-aware Isolation | 42.79 | 14.65 | CPA: 尺度隔离贡献+1.95 mIoU |
| 无Multi-group Context | 42.96 | 15.14 | CPA: 多组上下文贡献+1.87 mIoU |
| 无Affinity权重 | 43.97 | 15.85 | ADR: 亲和度权重贡献+1.28 mIoU |
| 普通卷积替换Deformable | 44.13 | 15.98 | ADR: 可变形卷积贡献+1.15 mIoU |

**时序帧数消融**：

| 历史帧数 | mIoU(%) | 推理时间(s) |
|---------|---------|-----------|
| 1帧 | 15.08 | 0.268 |
| 2帧 | 16.43 | 0.283 |
| **3帧（默认）** | **17.13** | 0.297 |
| 4帧 | 17.31 | 0.311 |
| 5帧 | 17.42 | 0.324 |

### 关键发现

1. **特征体积远优于代价体积**：SSC需要细粒度语义，保留特征上下文比计算匹配代价更重要
2. **CPA的两个组件各贡献约2 mIoU**：尺度隔离和多组独立学习缺一不可
3. **3帧是最佳效率-效果平衡点**：超过3帧的增益递减但延迟线性增加
4. **相机方法在语义上可以超越LiDAR**：LiDAR几何测量准确（IoU优势），但丰富的视觉语义使相机方法在mIoU上反超
5. **WVA的渐进学习策略**（$\alpha$ 从0开始）确保模型训练稳定性

## 亮点与洞察

1. **层级分解的设计哲学**：将"时序信息利用"这个大问题分解为"测量相关性→利用相关信息"两步，逻辑清晰
2. **相机超越LiDAR的突破意义**：在OpenOccupancy上mIoU超过LiDAR方法，证明时序建模可以弥补几何精度的不足
3. **对SSC任务特性的深刻理解**：指出SSC不是匹配任务，区别于视频深度估计，所以用特征体积而非代价体积
4. **从相似度度量角度创新**：改进余弦相似度的dense场景适应性，类似Pearson相关系数的思路
5. **$\alpha$ 渐进学习**的设计避免时序信息在早期干扰核心表征学习，类似于课程学习的思想

## 局限与展望

1. 推理速度~0.3s/帧，距实时部署仍有差距，可探索更轻量的时序聚合方案
2. PoseNet依赖位姿估计的准确性，ego-motion估计误差会影响时序对齐质量
3. 可变形卷积的offset学习可能在极端遮挡场景下效果有限
4. 可尝试将HTCL扩展到多相机（环视）SSC场景（如nuScenes的6相机设置）
5. 长时序建模（>5帧）的信息利用策略值得进一步研究

## 相关工作与启发

- **VoxFormer/VoxFormer-T**：SSC的重要baseline，HTCL直接改进其时序建模
- **MonoScene / TPVFormer / OccFormer**：camera-based SSC的代表方法
- **视频深度估计方法**：HTCL借鉴了homography warping构建时序体积的思路
- **可变形卷积/可变形注意力**：ADR模块的核心机制
- 启发：在密集感知任务中，时序信息的利用应该是"选择性"的而非"全盘接收"的

## 评分
- 新颖性: ⭐⭐⭐⭐ 层级式时序建模范式新颖，CPA和ADR的组合设计有创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 两个benchmark、全面消融、Stereo/Mono两个变体、时序帧数分析，非常完整
- 写作质量: ⭐⭐⭐⭐ 方法动机和设计逻辑清晰，图表质量高
- 价值: ⭐⭐⭐⭐⭐ 相机超越LiDAR的mIoU是重要里程碑，时序建模思路对SSC/占据预测领域有广泛启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](../../NeurIPS2025/autonomous_driving/learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)
- [\[AAAI 2026\] Towards 3D Object-Centric Feature Learning for Semantic Scene Completion](../../AAAI2026/autonomous_driving/towards_3d_object-centric_feature_learning_for_semantic_scene_completion.md)
- [\[ECCV 2024\] GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)
- [\[CVPR 2026\] OccuFly: A 3D Vision Benchmark for Semantic Scene Completion from the Aerial Perspective](../../CVPR2026/autonomous_driving/occufly_a_3d_vision_benchmark_for_semantic_scene_completion_from_the_aerial_pers.md)
- [\[CVPR 2026\] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](../../CVPR2026/autonomous_driving/sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)

</div>

<!-- RELATED:END -->
