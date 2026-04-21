---
title: >-
  [论文解读] ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images
description: >-
  [CVPR 2025][自动驾驶][位置识别] 本文提出ForestLPR，通过将点云在不同高度切片生成多张BEV密度图，利用ViT提取局部特征后经multi-BEV交互模块自适应关注不同高度的判别性特征，实现森林环境下鲁棒的LiDAR位置识别，在多个数据集上大幅超越SOTA。
tags:
  - CVPR 2025
  - 自动驾驶
  - 位置识别
  - LiDAR
  - BEV密度图
  - 森林环境
  - 多高度交互
  - 旋转不变性
---

# ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images

**会议**: CVPR 2025  
**arXiv**: [2503.04475](https://arxiv.org/abs/2503.04475)  
**代码**: [shenyanqing1105/ForestLPR-CVPR2025](https://github.com/shenyanqing1105/ForestLPR-CVPR2025)  
**领域**: 自动驾驶  
**关键词**: 位置识别, LiDAR, BEV密度图, 森林环境, 多高度交互, 旋转不变性

## 一句话总结

本文提出ForestLPR，通过将点云在不同高度切片生成多张BEV密度图，利用ViT提取局部特征后经multi-BEV交互模块自适应关注不同高度的判别性特征，实现森林环境下鲁棒的LiDAR位置识别，在多个数据集上大幅超越SOTA。

## 研究背景与动机

### 领域现状
位置识别（Place Recognition）是大规模定位系统中保持全局一致性的核心技术，可为SLAM系统提供回环检测约束。基于LiDAR的位置识别在城市环境中已取得显著进展，但在自然森林环境中的应用仍严重不足。

### 现有痛点
1. **森林环境的高自相似性**：树木之间外观极为相似，缺乏城市环境中建筑、标志等显著特征，传统特征提取方法难以区分不同位置
2. **植被随时间变化剧烈**：季节变化导致树冠、灌木等外观差异巨大，跨时间的重定位极具挑战
3. **高度信息丢失**：现有BEV方法通常生成单张BEV图像，将所有高度信息压缩到一张图中，丢失了沿z轴的判别性信息
4. **3D方法的点分布敏感性**：直接基于3D点云的方法（如MinkLoc3D）对点分布变化敏感，在不同传感器和树种条件下泛化性差

### 核心矛盾
如何在高自相似且随时间变化的森林环境中，提取稳定、判别性强、且跨场景泛化的位置描述子？

### 切入角度
基于两个关键假设：(1) 森林中不同树木的独特空间分布可用于位置区分；(2) 不同树木的判别性特征可能出现在不同高度。因此，将点云在不同高度切片生成多张BEV密度图，并设计交互模块自适应关注各位置最具判别性的高度。

### 核心idea
用多层BEV密度图替代单层BEV或原始3D点云，通过patch级别的高度自适应注意力机制，隐式地识别稳定的判别性特征，同时通过预处理去除地面和高处树冠来增强鲁棒性。

## 方法详解

### 整体框架
ForestLPR的pipeline分为四个阶段：(1) 预处理——地面分割、高度偏移去除、地面和树冠裁剪；(2) 多BEV密度图生成——在不同高度切片后投影生成BEV密度图；(3) 特征提取——共享ViT backbone提取多层局部特征；(4) 多BEV交互与全局聚合——patch级高度注意力加权后通过GeM池化生成旋转不变的全局描述子。

### 关键设计

#### 1. 点云预处理与多BEV密度图生成

- **功能**：去除森林环境中的高变异性元素（地面、高处树冠），在稳定的高度范围内生成多层BEV密度图
- **核心思路**：
    - 地面分割后，对每个非地面点减去其邻近地面点的加权平均高度，消除地形起伏影响
    - 去除1m以下（草地、落叶、积雪等）和6m以上（季节性树冠变化）的点
    - 将剩余点云在1-6m范围内以1m间隔切割为S=5个水平切片
    - 每个切片投影到地面并离散化，用$\log(V'(u,v)+1)$计算密度并归一化，生成BEV密度图
- **设计动机**：地面和高处树冠是森林中最易受天气和季节影响的部分，去除它们可显著提升跨时间的一致性。多层BEV保留了沿高度的结构分布信息。log变换避免密集区域主导学习过程

#### 2. 多BEV交互模块（Multi-BEV Interaction Module）

- **功能**：在patch级别实现不同高度BEV特征的自适应加权融合，让每个位置关注最具判别性的高度层
- **核心思路**：
    - S张BEV密度图共享同一个DeiT backbone分别提取局部特征，得到 $\mathbf{P'} \in \mathbb{R}^{(N+2) \times S \times 3C}$
    - 对每个patch计算各高度特征与均值的相对差值 $\Delta \mathbf{P'}_i$，再通过可学习权重 $\mathbf{W}_a$ 和Softmax生成高度注意力权重 $\mathbf{w}_i \in \mathbb{R}^{S \times 1}$
    - 用权重对各高度特征加权求和得到融合特征 $\mathbf{P}^w_i$
- **设计动机**：森林中即使去除了地面和高处树冠，中间层仍包含灌木残留等噪声。不同patch位置的有用信息可能来自不同高度，需要自适应地选择。使用相对特征值而非绝对值可以捕捉高度间的相对关系，且输入维度固定不受S变化影响
- **核心特性**：每个patch位置独立计算高度权重，实现了真正的空间位置自适应的高度注意力

#### 3. 描述子提取与全局聚合

- **功能**：生成旋转不变的全局位置描述子
- **核心思路**：
    - DeiT backbone提取低、中、高三层的输出token，沿通道拼接得到多层特征
    - 经multi-BEV交互后，使用GeM池化从patch级特征提取旋转不变的全局特征
    - 拼接[class]和[distillation] token与GeM输出，经L2归一化和线性降维得到最终全局描述子 $\mathbf{G} \in \mathbb{R}^D$（D=1024）
- **设计动机**：结合多层Transformer输出可以整合不同尺度的结构信息。GeM池化天然具有旋转不变性，适合森林中的反向重访场景
- **损失函数**：Triplet loss $\mathcal{L} = \max(d(\mathbf{G}^q, \mathbf{G}^p) - d(\mathbf{G}^q, \mathbf{G}^n) + m, 0)$，使用Cosine距离，正样本通过体积重叠（Octree计算，IoU > 0.9）而非距离阈值确定

## 实验关键数据

### 主实验 — 环内回环检测（Intra-sequence）

| 方法 | 7个评估集平均 F1 | 7个评估集平均 R@1 |
|------|-----------------|-----------------|
| LoGG3D-Net | 72.29 | 71.07 |
| MinkLoc3Dv2 | 56.75 | 55.41 |
| BEVPlace | 50.18 | 60.56 |
| MapClosures | 57.37 | 43.64 |
| **ForestLPR** | **75.46** | **78.45** |

R@1相比最强竞争者LoGG3D-Net平均提升**7.38%**。

### 跨序列重定位（Inter-sequence）
在Wild-Places inter-sequence评估中，ForestLPR R@1平均提升**9.11%**。

### 零样本泛化
- 仅在Wild-Places训练集上训练，直接在ANYmal和Botanic数据集上测试**无需微调**
- ANYmal数据集：F1 81.45（超越Scan-Context 8.01%），R@1 71.87（超越4.06%）
- 不同传感器（VLP-16 vs 手持）、不同树种和密度的场景下均表现良好

### 关键发现
1. 在具有大量反向重访的挑战性场景（V-03, K-03）中，ForestLPR优势最为显著
2. 多BEV密度图是性能提升的关键，单BEV→多BEV可带来显著增益
3. 基于体积重叠的正样本挖掘比距离阈值更准确，尤其在存在遮挡和盲区时

## 亮点与洞察

1. **问题选择精准**：森林环境的位置识别是一个被忽视但重要的问题，具有独特的挑战（高自相似性、季节变化），本文首次系统性地解决
2. **假设驱动的设计**：从"不同高度包含不同判别信息"这一直觉出发，逐步构建多BEV密度图→高度自适应注意力的完整方案，逻辑清晰
3. **工程价值高**：预处理流程（地面分割+高度归一化+裁剪）简单有效，可直接迁移到其他森林场景感知任务
4. **跨域泛化性强**：在不同传感器、不同树种、不同地形条件下均保持优异性能，表明方法的设计具有良好的通用性

## 局限性

1. 预处理参数（1m下限、6m上限、切片间隔1m）可能不适用于所有类型的森林（如热带雨林中树木可高达数十米）
2. 目前仅支持位置识别，未涉及6-DoF位姿估计
3. BEV密度图的分辨率（0.5m）在近距离场景可能不够精细
4. 训练仅使用Wild-Places单一数据集，更大规模更多样化的训练数据可能进一步提升泛化能力

## 相关工作与启发

- **与BEVPlace的对比**：BEVPlace使用单层BEV密度图+分组卷积，在部分均匀树林场景中表现好但计算量最大。ForestLPR通过多层BEV和自适应注意力更鲁棒
- **与LoGG3D-Net的对比**：LoGG3D-Net基于稀疏卷积捕获局部一致性，在3D点云上直接操作。ForestLPR将3D问题转化为多层2D问题，利用了预训练ViT的强大表示能力
- **BEV密度图的通用性**：BEV密度图作为LiDAR的2D表示，不依赖特定传感器配置，比range image更适合多帧拼接的submap场景
- **对机器人导航的启发**：森林环境的位置识别对搜救机器人、农林巡检等应用场景有直接价值

## 评分

⭐⭐⭐⭐ (4/5)

问题选择独特且有实际应用价值，方法设计逻辑清晰、实验全面。多BEV交互机制新颖且有效。不过核心技术创新点相对集中在多高度BEV的引入，后续可考虑结合语义信息进一步提升判别力。

<!-- RELATED:START -->

## 相关论文

- [L2RSI: Cross-View LiDAR-Based Place Recognition for Large-Scale Urban Scenes via Remote Sensing Imagery](../../NeurIPS2025/autonomous_driving/l2rsi_cross-view_lidar-based_place_recognition_for_large-scale_urban_scenes_via_.md)
- [Rethinking LiDAR Domain Generalization: Single Source as Multiple Density Domains](../../ECCV2024/autonomous_driving/rethinking_lidar_domain_generalization_single_source_as_multiple_density_domains.md)
- [Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning](tra-moe_learning_trajectory_prediction_model_from_multiple_domains_for_adaptive_.md)
- [RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting](toward_real-world_bev_perception_depth_uncertainty_estimation_via_gaussian_splat.md)

<!-- RELATED:END -->
