---
title: >-
  [论文解读] Occupancy Learning with Spatiotemporal Memory
description: >-
  [ICCV 2025][自动驾驶][3D occupancy prediction] 提出 ST-Occ，一个场景级时空占用表示学习框架，通过统一时序建模（Unified Temporal Modeling）范式，使用场景坐标系下的时空记忆库和具有不确定性/动态感知的记忆注意力机制，在 Occ3D 基准上比 SOTA 提升 3 mIoU，同时将时序不一致性降低 29%。
tags:
  - ICCV 2025
  - 自动驾驶
  - 3D occupancy prediction
  - temporal fusion
  - spatiotemporal memory
  - uncertainty awareness
---

# Occupancy Learning with Spatiotemporal Memory

**会议**: ICCV 2025  
**arXiv**: [2508.04705](https://arxiv.org/abs/2508.04705)  
**代码**: https://github.com/matthew-leng/ST-Occ  
**领域**: autonomous_driving  
**关键词**: 3D occupancy prediction, temporal fusion, spatiotemporal memory, autonomous driving, uncertainty awareness

## 一句话总结

提出 ST-Occ，一个场景级时空占用表示学习框架，通过统一时序建模（Unified Temporal Modeling）范式，使用场景坐标系下的时空记忆库和具有不确定性/动态感知的记忆注意力机制，在 Occ3D 基准上比 SOTA 提升 3 mIoU，同时将时序不一致性降低 29%。

## 研究背景与动机

3D 占用表示（occupancy representation）已成为自动驾驶中建模周围环境的一种重要细粒度感知范式。然而，跨多帧高效聚合 3D 占用信息仍然面临三大挑战：

**效率问题**：占用表示是密集的体素特征，额外的高度维度使得存储和处理多帧历史特征非常消耗资源

**不确定性**：遮挡和光照变化导致体素级别的不确定性在帧间累积，影响预测鲁棒性

**动态性**：场景中的动态物体会导致体素偏移，若不准确建模将导致历史特征错位

现有方法主要基于循环式（recurrent）或堆叠式（stacking）的逐帧队列时序融合，从 BEV 扩展到 3D 占用，但计算和内存开销大，利用时序信息的效率低。本文提出在**场景坐标系**（而非自车坐标系）下构建统一的时空记忆，突破上述瓶颈。

## 方法详解

### 整体框架

ST-Occ 的核心思想是**统一时序建模（Unified Temporal Modeling）**：用一个场景坐标系下的统一记忆 $\mathbf{M}$ 替代传统的帧级队列。给定当前帧多视角图像 $I_t$，占用编码器提取自车坐标系下的占用表示 $\mathbf{V}_t$，然后记忆注意力模块将 $\mathbf{V}_t$ 与时空记忆中的历史信息 $\mathbf{H}_t$ 融合得到 $\tilde{\mathbf{V}}_t$，最后更新记忆。

### 关键设计

1. **时空记忆库（Spatiotemporal Memory）**：在场景坐标系下维护一个全局表示 $\mathbf{M} \in \mathbb{R}^{H_G \times W_G \times Z_G \times C_G}$。当使用 $k$ 帧时序信息时，传统方法需存储 $k$ 个完整表示，而统一时序建模只需一个，内存效率大幅提升。记忆中存储三类时序属性 $\mu = \{\mathbf{c}, \delta, \mathbf{f}\}$：

    - **历史类别激活 $\mathbf{c}$**：经 softmax 的类别预测向量，以指数衰减 $\alpha=0.5$ 递进更新
    - **平均对数方差 $\delta$**：用于不确定性估计
    - **占用流 $\mathbf{f}$**：俯视图中的 2D 运动向量，用于动态实例的运动补偿

2. **记忆注意力（Memory Attention）**：将当前占用表示 $\mathbf{V}_t$ 条件化于时空记忆中的历史信息。核心公式为：
    $(1-u) \cdot \text{DA}(V_{t_p}, p+f, V_t) + u \cdot \text{DA}(V_{t_p}, p+f, \chi[\mathbf{M}_t, T_t])$
   其中 $u$ 是通过 MLP 编码时序属性得到的不确定性权重，$f$ 是占用流用于动态补偿。不确定性 $u$ 的输入包括历史类别激活 $\mathbf{c}$、对数方差 $\delta$ 和当前-历史特征余弦相似度 $\varepsilon$。

3. **时序一致性评估指标（mSTCV）**：提出 mean Spatiotemporal Classification Variability，衡量同一真实世界位置的体素在不同帧间的分类变化率，用于量化时序预测的稳定性。公式为每帧非空体素中分类发生改变的比例，再对所有帧取平均。

### 损失函数 / 训练策略

总损失由三部分组成：
$$\mathcal{L} = \mathcal{L}_{occ} + \mathcal{L}_{nll} + \mathcal{L}_{of}$$

- $\mathcal{L}_{occ}$：占用预测损失，包含 Focal loss、Lovász softmax loss、affinity loss 和 depth loss
- $\mathcal{L}_{nll}$：高斯负对数似然损失，用于对数方差预测
- $\mathcal{L}_{of}$：L1 损失，用于占用流预测

训练策略：学习率 $2 \times 10^{-4}$，训练 26 epochs，前 3 个 epoch 不使用时序建模以稳定训练。占用流的真实标签从 nuScenes 标注中实时计算实例边界框的时序偏移得到。

## 实验关键数据

### 主实验 (表格)

Occ3D 基准上的 3D 占用预测结果（ResNet50 backbone）：

| 方法 | mIoU | barrier | car | driv. surf. | sidewalk | terrain | manmade | vegetation |
|------|------|---------|-----|-------------|----------|---------|---------|------------|
| FB-OCC† (无时序) | 37.39 | 44.83 | 47.97 | 78.83 | 49.06 | 52.22 | 39.07 | 34.61 |
| FB-OCC (有时序) | 39.11 | 44.74 | 49.10 | 80.07 | 51.18 | 55.13 | 42.19 | 37.53 |
| **ST-Occ (ours)** | **42.13** | **49.62** | **52.55** | **84.26** | **56.09** | **59.85** | **45.27** | **40.11** |
| ViewFormer† | 37.80 | 44.89 | 48.90 | 81.93 | 53.72 | 55.50 | 42.18 | 36.29 |
| ViewFormer | 41.44 | 50.16 | 53.36 | 84.67 | 57.43 | 59.64 | 47.57 | 40.38 |
| ViewFormer† + ST-Occ | **42.30** | **50.61** | **53.24** | **85.28** | **58.39** | **60.39** | **48.02** | **41.42** |

时序一致性评估（mSTCV ↓）：

| 方法 | mSTCV (%) | mSTCV† (%) |
|------|-----------|------------|
| FB-OCC | 12.18 | 8.57 |
| ST-Occ | **8.68** | **6.48** |

### 消融实验 (表格)

不同设计组件的贡献（Occ3D）：

| 设置 | mIoU |
|------|------|
| No Temporal (FB-OCC基线) | 37.39 |
| Mem. Attn. | 41.17 |
| Mem. Attn. + Dynamics | 41.73 |
| Mem. Attn. + Uncertainty | 41.85 |
| **ST-Occ (完整)** | **42.13** |

时序融合效率对比（相同融合操作和帧数）：

| 时序建模方式 | 训练显存 (GB) ↓ | 融合时间 (ms) ↓ | 推理显存 (GB) ↓ | FPS ↑ |
|-------------|----------------|----------------|----------------|-------|
| Recurrent | 12.89 | 705 | 10.08 | 5.95 |
| Stacked | 19.02 | 84 | 11.29 | 5.42 |
| **Unified (ours)** | **10.90** | **24** | **5.57** | **8.65** |

子组件消融（时序属性的贡献）：

| c | ε | δ | f | mIoU |
|---|---|---|---|------|
| - | - | - | - | 41.17 |
| ✓ | - | - | - | 41.45 |
| ✓ | ✓ | - | - | 41.73 |
| ✓ | ✓ | ✓ | - | 41.85 |
| - | - | - | ✓ | 41.73 |
| ✓ | ✓ | ✓ | ✓ | **42.13** |

### 关键发现

- ST-Occ 比 FB-OCC 提升 3 mIoU，时序信息利用效率是 FB-OCC 的 2.8 倍
- 统一时序建模的融合时间仅 24ms（vs. 循环式 705ms），推理显存仅 5.57GB（vs. 堆叠式 11.29GB）
- 动态感知主要提升动态类别（如 car +1.2 IoU），不确定性感知主要提升静态类别
- 随着融合帧数增加，性能持续提升且时序一致性增强，且计算成本增长远低于传统方法

## 亮点与洞察

- **场景坐标系统一记忆** 是核心创新：将传统帧级队列替换为单一全局表示，从根本上解决了多帧 3D 占用特征的存储和计算瓶颈
- 不确定性和动态感知的解耦设计很优雅：通过 MLP 编码的 $u$ 自动平衡当前帧和历史帧的贡献，占用流 $f$ 提供动态补偿
- mSTCV 是一个有价值的新指标，填补了占用预测中时序一致性评估的空白
- 方法具有很好的即插即用性：可以替换 FB-OCC 和 ViewFormer 的时序模块

## 局限与展望

- 动态建模依赖 nuScenes 标注计算占用流的真实标签，未来可将动态信息直接从时序特征中推导
- 可扩展到基于稀疏查询的感知方法
- 场景记忆的大小受限于 GPU 显存，对超大规模场景的可扩展性需要进一步验证

## 相关工作与启发

- 从 BEVFormer 的 temporal self-attention 出发，但跳出了逐帧对齐的范式
- PasCo 提出了占用预测的不确定性意识，本文将其融入时序建模
- 场景级表示的思路类似于 SLAM 中的全局地图构建，但以可微学习的方式实现

## 评分

- **新颖性**: ⭐⭐⭐⭐ 统一时序建模范式是一个简洁而有效的创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 消融全面，效率对比、时序一致性评估都很完善
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，公式推导连贯
- **价值**: ⭐⭐⭐⭐ 对占用预测的时序建模提供了新的高效范式

<!-- RELATED:START -->

## 相关论文

- [GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting](frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)
- [DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)

<!-- RELATED:END -->
