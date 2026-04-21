---
title: >-
  [论文解读] Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation
description: >-
  [ECCV 2024][目标检测][旋转目标检测] 提出点-轴（Point-Axis）表示方法，将旋转目标的位置（点集）和方向（轴编码）解耦，配合 Max-Projection Loss 和 Cross-Axis Loss 实现无需额外标注的优化，并基于此设计 Oriented DETR 模型，解决传统旋转框表示的损失不连续问题。
tags:
  - ECCV 2024
  - 目标检测
  - 旋转目标检测
  - 点-轴表示
  - DETR
  - 航空图像
  - 损失不连续
---

# Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation

**会议**: ECCV 2024  
**arXiv**: [2407.08489](https://arxiv.org/abs/2407.08489)  
**代码**: [https://PointAxis.github.io/](https://PointAxis.github.io/)  
**领域**: 目标检测  
**关键词**: 旋转目标检测, 点-轴表示, DETR, 航空图像, 损失不连续

## 一句话总结

提出点-轴（Point-Axis）表示方法，将旋转目标的位置（点集）和方向（轴编码）解耦，配合 Max-Projection Loss 和 Cross-Axis Loss 实现无需额外标注的优化，并基于此设计 Oriented DETR 模型，解决传统旋转框表示的损失不连续问题。

## 研究背景与动机

旋转目标检测是计算机视觉中的重要任务，广泛应用于航空图像分析。现有方法主要使用旋转框（rotated bounding box）表示目标，但面临**损失不连续**问题：

**角度方法** $(x, y, w, h, \theta)$：当长短边长度接近时，角度 $\theta$ 会在 $\theta$ 和 $\theta \pm 90°$ 之间跳变，导致 loss 不连续

**四边形方法** $(x, y, w, h, l_1, l_2, l_3, l_4)$：目标接近水平时偏移量突变，同样产生不连续

**其他变体**（bbox boundary vectors、middle lines、Gaussian distributions）：均面临类似的边界问题（如 square problem）

近期基于**点集**的方法（如 Oriented Reppoints）虽避免了定义边界的跳变，但缺乏方向信息描述能力——当点集呈近圆形分布时，计算得到的最小外接矩形可能无法准确包围目标。

**核心动机**：能否设计一种同时描述位置和方向、且避免损失不连续的表示方法？

## 方法详解

### 整体框架

Point-Axis 表示将每个旋转目标 $i$ 定义为：
- **点集** $\mathcal{P}_i = \{p_i^j\}_{j=1,...,K}$：$K$ 个点描述目标的空间范围和轮廓，第 $K$ 个点为中心点
- **轴表示** $\mathcal{A}_i$：将方向离散化为 bins 并用高斯平滑生成四峰标签编码，表示目标的主方向

这种设计的核心优势：
- **位置与旋转解耦**：避免了旋转框定义中耦合导致的边界跳变
- **轴序不变性（axis-order invariant）**：不区分长轴/短轴，天然解决 square/circular 目标的边界问题
- **循环标签**：0° 和 360° 的标签一致，确保角度周期性处的连续性

### 关键设计

1. **Max-Projection Loss（最大投影损失）**

   用于监督点集学习，无需显式关键点标注。对于预测点集 $\hat{\mathcal{P}}_i$，先转为相对中心的向量 $\hat{\mathcal{V}}_i$，然后将每个向量投影到 GT 边界向量 $\mathcal{V}_i$ 上，选择最大投影值进行优化：

   $$\text{minimize} \sum_{j=1}^{4} \left| \max_{m=1,...,K-1} \frac{(\hat{v}_i^m - v_i^j) \cdot v_i^j}{\|v_i^j\|} \right| + \|\hat{v}_i^K\|$$

   设计动机：仅约束每个方向的最大投影值而不约束非最大值，减少优化方向歧义，增强点集描述的灵活性。实验表明添加额外惩罚项（penalty 或 top-k）反而降低精度。

2. **Cross-Axis Loss（交叉轴损失）**

   用于轴表示学习。将方向离散为 $N_{bins} = 360$ 个 bins，用交叉熵损失监督：

   $$\text{minimize} \frac{1}{N_{bins}} \sum_{j=1}^{N_{bins}} [\mathcal{A}_i^j \log \hat{\mathcal{A}}_i^j + (1-\mathcal{A}_i^j) \log(1-\hat{\mathcal{A}}_i^j)]$$

   推理时取 argmax 获得主方向，再间隔 90° 扩展为四个方向。对于缺乏明确方向定义的目标（如某些游泳池），模型仍能学到覆盖所有可能方向的分布。

3. **Oriented DETR 架构**

   基于 DETR 框架的端到端检测模型，包含三个核心模块：

   - **Object-to-Point Query Conversion**：将每个物体查询 $Q_o^i$ 转换为 $K$ 个点查询。中心点查询通过 MLP 预测相对参考点的偏移；边界点查询通过极坐标系预测各方向的距离
   - **Points Detection Decoder**：包含 Point-to-Point Attention（组内自注意力，同一实例的 $K$ 个点交互）和 Object-to-Object Attention（提取各实例中心点进行跨实例交互），避免不同实例点查询间的歧义交互
   - **Prediction Head**：每个点查询映射为 2D 坐标，类别和轴表示从所有条件化点查询中预测

### 损失函数 / 训练策略

总体损失函数为 Max-Projection Loss 和 Cross-Axis Loss 的加权组合：

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\lambda_1 \mathcal{L}_{proj} + \lambda_2 \mathcal{L}_{ca})$$

- 使用 AdamW 优化器，初始学习率 1e-4，weight decay 1e-4
- 训练 36 epochs（DOTA/DIOR/COCO），50 epochs（HRSC2016）
- 4 × RTX 4090 GPU，batch size = 8

## 实验关键数据

### 主实验

**DOTA 数据集（单尺度训练和测试）对比：**

| 方法 | 类型 | Backbone | mAP50 |
|------|------|----------|-------|
| Orient-Rep | 点集表示 | Swin-T | 77.6 |
| AO2-DETR | DETR | R-50 | 77.7 |
| ARS-DETR | DETR | Swin-T | 75.5 |
| EMO2-DETR | DETR | Swin-T | 72.3 |
| **Oriented DETR** | **DETR** | **R-50** | **79.1** |
| **Oriented DETR** | **DETR** | **Swin-T** | **79.8** |

**DOTA 数据集（多尺度）对比：**

| 方法 | Backbone | mAP50 |
|------|----------|-------|
| LSKNet | - | 81.85 |
| AO2-DETR | - | 79.22 |
| **Oriented DETR** | **Swin-L** | **82.26** |

**DIOR-R 数据集：**

| 方法 | Backbone | mAP50 |
|------|----------|-------|
| 此前最佳 | R-50 | 63.91 |
| **Oriented DETR** | **R-50** | **66.80** (+2.89) |
| 此前最佳 | Swin-T | 71.05 |
| **Oriented DETR** | **Swin-T** | **74.26** (+3.21) |

### 消融实验

**Points Detection Decoder 各组件贡献（DOTA val）：**

| 配置 | mAP50 | mAP75 | 说明 |
|------|-------|-------|------|
| Baseline（无点查询） | 72.80 | 45.25 | 两阶段 Deformable DETR |
| + Point Queries | 70.98 | 44.06 | 点查询+普通自注意力（有跨实例歧义） |
| + Group Self-Attention | 74.21 | 48.30 | 组内自注意力消除歧义 |
| + Decouple Cross-Attention | **75.35** (+2.55) | **50.14** (+4.89) | 全部组件 |

**点约束损失对比：**

| Loss 设计 | mAP50 | mAP75 | 说明 |
|-----------|-------|-------|------|
| Max-Projection | **75.35** | **50.14** | 仅最大投影值 |
| + penalty | 75.20 | 50.02 | 加外部点惩罚 |
| top-2 | 74.77 | 49.36 | 约束前2投影值 |
| top-3 | 73.20 | 47.88 | 约束前3投影值，灵活性下降 |

### 关键发现

- 增加点数从 K=5 到 K=13 仅提升 0.49% mAP50，说明轴表示有效弥补了点数不足的方向信息
- 组自注意力（Group Self-Attention）是 decoder 最关键的组件——普通自注意力让不同实例的点查询互相干扰，反而降低性能
- 对于方向不明确的目标（如某些游泳池），模型能学到覆盖所有方向的分布，展示鲁棒性
- mAP75 提升（+4.89）远大于 mAP50 提升（+2.55），说明点-轴表示显著提高定位精度

## 亮点与洞察

1. **表示方法创新**：首次在旋转目标检测中提出位置-方向解耦表示，从表示层面根治损失不连续问题
2. **Max-Projection Loss 设计精妙**：仅约束最大投影值，给予点集自由学习空间，避免过度约束导致的优化歧义
3. **轴序不变性**：不区分长轴短轴，自然处理 square/circular 目标——传统方法在此场景下频繁失败
4. **与 DETR 的自然融合**：点查询机制与 DETR 的 query 架构契合度高，Group Self-Attention 设计优雅
5. **实验覆盖广泛**：DOTA、DIOR-R、HRSC2016、COCO 四个数据集，单/多尺度评测

## 局限与展望

1. **点数增加收益递减**：K=5 到 K=13 仅提升 0.49%，大量点的计算开销是否值得需进一步分析
2. **HRSC2016 上优势不大**：该数据集目标长宽比大、形状简单，传统方法本身不存在边界问题
3. **Cross-Axis Loss 的 360 bins 固定设计**：更粗或更细的粒度是否影响不同场景下的性能未探索
4. **未考虑实例分割**：点-轴表示描述的是方向框，对密集像素级任务的扩展性待验证
5. **推理速度未详细对比**：作为 DETR 模型，端到端推理效率是否满足实时需求

## 相关工作与启发

- **vs Oriented Reppoints**：同为点集方法但本文增加了轴表示弥补方向信息缺失，R-50 下超越 2.2% mAP50
- **vs CSL（角度分类）**：CSL 将角度回归转为分类解决周期性，但未解决定义边界问题；本文的轴编码更彻底
- **vs AO2-DETR/ARS-DETR**：同为 DETR 系列的旋转检测器，但仍使用旋转框查询迭代更新，未考虑水平框对齐假设不适用旋转场景的问题
- **启发**：解耦思想可推广到 3D 目标检测（位置与姿态解耦），以及其他需要方向表示的任务如文字检测

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 点-轴解耦表示和 Max-Projection Loss 设计新颖，是旋转检测表示方法的重要进步
- **实验充分度**: ⭐⭐⭐⭐ — 四个数据集 + 详细消融，但缺少推理速度对比
- **写作质量**: ⭐⭐⭐⭐ — 图示清晰（尤其 Figure 2-3），概念解释直观
- **实用价值**: ⭐⭐⭐⭐ — 在航空遥感目标检测上取得明确 SOTA，DETR 端到端设计利于部署

<!-- RELATED:START -->

## 相关论文

- [TAPTR: Tracking Any Point with Transformers as Detection](taptr_tracking_any_point_with_transformers_as_detection.md)
- [MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection](mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection.md)
- [Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](../../CVPR2026/object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](../../ICLR2026/object_detection/spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)

<!-- RELATED:END -->
