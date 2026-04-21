---
title: >-
  [论文解读] MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception
description: >-
  [ICCV 2025][自动驾驶][多任务学习] 提出 MAESTRO 框架，通过类别原型生成（CPG）、任务特定特征生成（TSFG）和场景原型聚合（SPA）三个模块，在多任务3D感知中生成任务特定特征并抑制任务间干扰，在3D目标检测、BEV地图分割和3D占用预测三个任务上同时超越单任务模型。
tags:
  - ICCV 2025
  - 自动驾驶
  - 多任务学习
  - 3D感知
  - BEV分割
  - 3D目标检测
  - 占用预测
---

# MAESTRO: Task-Relevant Optimization via Adaptive Feature Enhancement and Suppression for Multi-task 3D Perception

**会议**: ICCV 2025  
**arXiv**: [2509.17462](https://arxiv.org/abs/2509.17462)  
**代码**: 将公开  
**领域**: autonomous_driving  
**关键词**: 多任务学习, 3D感知, BEV分割, 3D目标检测, 占用预测

## 一句话总结

提出 MAESTRO 框架，通过类别原型生成（CPG）、任务特定特征生成（TSFG）和场景原型聚合（SPA）三个模块，在多任务3D感知中生成任务特定特征并抑制任务间干扰，在3D目标检测、BEV地图分割和3D占用预测三个任务上同时超越单任务模型。

## 研究背景与动机

自动驾驶感知系统需要同时执行多个任务：3D目标检测关注可移动的前景物体，BEV地图分割关注静态背景结构，3D占用预测则需要同时关注前景和背景。多任务学习（MTL）可以通过共享骨干网络提升计算效率，但不同任务的梯度信号方向不一致会导致**任务冲突**，使各任务性能下降。

现有MTL方法虽然缓解了部分冲突，但仍然无法生成真正的任务特定特征表示，性能落后于独立训练的单任务模型。作者发现不同任务关注的语义线索和空间区域存在根本差异，因此需要一种能够从共享特征中**增强任务相关信息并抑制无关信息**的机制。

## 方法详解

### 整体框架

MAESTRO 首先从多视角图像中通过共享骨干网络提取2D特征，再通过 Lift-Splat-Shoot（LSS）方法提升到3D体素表示 $F_s \in \mathbb{R}^{C \times X \times Y \times Z}$。然后依次经过 CPG、TSFG 和 SPA 三个模块，生成各任务的特定特征，最后送入各任务头进行预测。

### 关键设计

1. **Class-wise Prototype Generator (CPG)**: 将语义类别分为前景组（车、行人等）和背景组（道路、人行道等），通过轻量级掩码分类器对共享体素特征计算逐体素语义置信度 $S_v$，根据最高置信类别生成二值掩码 $B_k$，再对掩码区域的特征进行平均池化得到类别原型 $P_k$。前景原型分配给检测任务，背景原型分配给BEV分割任务，两者联合分配给占用预测任务。核心公式：$P_k = \text{AvgPool}(F_s \otimes B_k)$。设计动机：通过语义分组为各任务提供有针对性的先验信息。

2. **Task-Specific Feature Generator (TSFG)**: 包含三个子模块。(a) **任务依赖特征变换**：将共享体素特征分别变换为BEV域特征（检测/分割）或体素域特征（占用预测）。(b) **自适应特征增强**：利用原型组与变换后的特征做点积生成原型级特征（激活与原型语义对齐的空间区域），同时通过通道注意力生成原型感知特征，拼接后经卷积得到增强特征 $\tilde{F}_t$。(c) **特征抑制**：通过轻量CNN从原型感知特征生成抑制分数图 $S^{supp}_t$，与增强特征相乘实现门控操作：$F^{TS}_t = \tilde{F}_t \otimes S^{supp}_t$。设计动机：先增强再抑制，彻底消除任务无关信息。

3. **Scene Prototype Aggregator (SPA)**: 利用检测头的预测边界框通过RoIAlign生成检测原型 $P_{Det}$，利用分割头的预测掩码通过掩码平均池化生成地图原型 $P_{Map}$。然后通过语义对齐规则将这些任务导向原型聚合到占用预测的原型组中，形成场景原型作为占用解码器的初始查询。设计动机：利用检测和分割的互补语义信息增强占用预测性能而不降低其他任务。

### 损失函数 / 训练策略

总损失函数为六部分之和：
$$L_{total} = L_{depth} + L_{CPG} + L_{Sup} + L_{det} + L_{map} + L_{occ}$$

- $L_{CPG}$ 使用 Dice Loss + Lovász Loss 监督类别掩码分类
- $L_{Sup}$ 使用 Focal Loss 监督各任务的抑制分数图
- 各任务头分别使用 CenterPoint（检测）、BEVFusion（分割）、OccFormer（占用）的损失
- 使用 ResNet-50 骨干，AdamW 优化器（lr=1e-4），24个epoch，不使用CBGS

## 实验关键数据

### 主实验

| 方法 | mAP | NDS | mIoU (Map) | mIoU (Occ) | 延迟 (ms) |
|------|-----|-----|-----------|-----------|----------|
| Baseline-STL | 33.8 | 41.7 | 47.5 | 36.5 | 405.9 |
| Baseline-MTL | 32.7 | 38.2 | 43.5 | 36.0 | 219.6 |
| BEVFusion MTL | 33.6 | 39.2 | 44.0 | - | - |
| DualBEV (STL) | 35.2 | 42.5 | - | - | 65.1 |
| DifFUSER (STL) | - | - | 48.3 | - | 92.2 |
| FB-Occ (STL) | - | - | - | 37.4 | 129.7 |
| **MAESTRO-MTL** | **36.4** | **43.2** | **51.3** | **38.6** | 250.3 |

MAESTRO 相比 Baseline-STL 在三个任务上分别提升 2.6% mAP、3.8% mIoU、2.1% mIoU，同时延迟降低155.6ms；相比 Baseline-MTL 提升更为显著。

### 消融实验

| CPG | TSFG (Det) | TSFG (Map) | TSFG (Occ) | SPA | mAP | NDS | mIoU (Map) | mIoU (Occ) |
|-----|-----------|-----------|-----------|-----|-----|-----|-----------|-----------|
| ✗ | ✗ | ✗ | ✗ | ✗ | 31.3 | 32.3 | 40.4 | 33.6 |
| ✓ | ✗ | ✗ | ✗ | ✗ | 31.3 | 32.4 | 40.3 | 34.6 |
| ✓ | ✓ | ✓ | ✓ | ✗ | 32.6 | 34.3 | 44.2 | 36.9 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **32.6** | **34.3** | **44.2** | **36.9** |

TSFG 子模块消融：原型级特征贡献 +1.8% mIoU (Map)，原型感知特征 +1.3%，特征抑制 +0.7%。SPA 中移除地图原型降低0.6%占用mIoU，移除检测原型再降0.4%。

### 关键发现

- MTL 框架首次在三个3D感知任务上全面超越独立训练的STL模型
- 前景/背景语义分组原型是缓解任务冲突的有效先验
- 特征抑制模块在BEV地图分割上贡献最大（需要抑制前景干扰）
- 检测和分割的输出信息对占用预测有显著互补作用

## 亮点与洞察

- 核心洞察独到：不同3D感知任务对前景/背景的关注天然不同，利用这种语义差异设计原型分组是一个自然而有效的思路
- "增强+抑制"的双阶段特征精炼机制设计精巧——先用原型组增强相关信息，再用可学习的抑制分数图过滤无关成分
- SPA 利用已训练任务的输出作为辅助信息是一种巧妙的任务依赖性建模方式
- 整体框架模块化程度高，各组件可独立验证贡献

## 局限与展望

- 消融实验在 1/4 训练集上进行，消融结论的适用性有待验证
- 前景/背景的手动分组规则可能无法泛化到更多任务（如车道线检测）
- 仅在 nuScenes 验证集上评估，未在其他大规模数据集（如 Waymo）验证
- 抑制分数图需要额外的GT监督（RoI掩码），增加了标注依赖
- 未探讨时序信息融合和更大骨干网络的影响

## 相关工作与启发

- 与 HENet、SOGDet 等现有 MTL 方法相比，MAESTRO 不仅缓解了任务冲突，还通过原型机制实现了真正的任务特定特征生成
- TaskExpert 的 MoE 思路与本文的原型分组思想异曲同工，均在探索如何为不同任务分配不同的特征处理路径
- 原型学习的思想来自 Prototypical Networks，在此被创新性地应用于多任务特征解耦

## 评分

- **新颖性**: ⭐⭐⭐⭐ 前景/背景原型分组+增强抑制的设计有新意
- **实验充分度**: ⭐⭐⭐⭐ 主实验对比全面，消融详尽
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图示到位
- **价值**: ⭐⭐⭐⭐ 多任务3D感知MTL超越STL是重要进步

<!-- RELATED:START -->

## 相关论文

- [Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](../../AAAI2026/autonomous_driving/task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)
- [DuET: Dual Incremental Object Detection via Exemplar-Free Task Arithmetic](duet_dual_incremental_object_detection_via_exemplar-free_task_arithmetic.md)
- [Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d.md)
- [Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Perception](unleashing_the_temporal_potential_of_stereo_event_cameras_for_continuous-time_3d.md)
- [AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)

<!-- RELATED:END -->
