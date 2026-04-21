---
title: >-
  [论文解读] PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds
description: >-
  [CVPR 2025][自动驾驶][自监督学习] 提出 PSA-SSL，通过在对比学习中增加自监督边界框回归预训练任务保留物体姿态和尺寸信息，并结合 LiDAR 光束模式增强实现跨传感器泛化，在 3D 语义分割和目标检测上显著超越 SOTA 自监督方法。
tags:
  - CVPR 2025
  - 自动驾驶
  - 自监督学习
  - 点云表示
  - LiDAR跨传感器
  - 边界框回归
  - 对比学习
---

# PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds

**会议**: CVPR 2025  
**arXiv**: [2503.13914](https://arxiv.org/abs/2503.13914)  
**代码**: [GitHub](https://github.com/TRAILab/PSA-SSL)  
**领域**: 自动驾驶  
**关键词**: 自监督学习, 点云表示, LiDAR跨传感器, 边界框回归, 对比学习

## 一句话总结

提出 PSA-SSL，通过在对比学习中增加自监督边界框回归预训练任务保留物体姿态和尺寸信息，并结合 LiDAR 光束模式增强实现跨传感器泛化，在 3D 语义分割和目标检测上显著超越 SOTA 自监督方法。

## 研究背景与动机

自监督学习（SSL）在 LiDAR 点云上有望学到可迁移到多种下游任务和传感器的特征表示。然而，现有基于对比学习的 SSL 方法存在根本性限制：对比损失最大化同一实例在不同几何变换（旋转、平移、缩放）下的特征相似性，导致学到的特征对这些变换不变——即丢失了物体的姿态和尺寸信息。

这对下游局部化和几何敏感的 3D 场景理解任务（如语义分割、目标检测）是有害的。此外，现有方法通常只在同一数据集上预训练和微调，跨 LiDAR 传感器的泛化能力有限。

核心发现：对比学习使得特征对几何变换不变，而下游任务恰恰需要几何感知的特征。作者通过边界框回归预训练任务来显式编码姿态和尺寸信息，与对比损失互补——对比损失学类别判别特征，回归头保留几何信息。

## 方法详解

### 整体框架

PSA-SSL 包含三个阶段：(1) **预处理**——用 Patchwork++ 分离地面点、HDBSCAN 聚类非地面点、拟合边界框作为回归目标；(2) **预训练**——联合优化对比损失和边界框回归损失；(3) **微调**——在下游分割或检测任务上微调。方法是模型无关的，可应用于任何基于对比学习的 SSL 方法。

### 关键设计1：自监督边界框回归预训练任务

**功能**：在对比学习的基础上保留物体姿态和尺寸信息。

**核心思路**：在预处理阶段，对非地面点进行 HDBSCAN 聚类得到物体簇，使用 off-the-shelf 算法拟合朝上的边界框。在预训练阶段，对 query 和 momentum 编码器输出的拼接特征，添加全连接回归头（2 层 256 维），预测每个聚类点到固定大小锚框的偏移量。使用 smooth L1 损失仅对聚类点计算回归损失。总损失：$\mathcal{L} = \beta_1 \mathcal{L}_{con} + \beta_2 \mathcal{L}_{reg}$。

**设计动机**：对比损失使特征对旋转/平移/缩放不变，回归任务则强制特征保留这些几何信息。两者互补——对比学习提供类别判别性，回归提供几何感知性。回归目标由无监督聚类+框拟合自动获取，真正的自监督。

### 关键设计2：LiDAR 光束模式增强（LPA）

**功能**：学习跨不同 LiDAR 传感器的泛化特征。

**核心思路**：在数据增强阶段，将输入点云通过球面投影变换为不同 LiDAR 配置（Velodyne-32/64, Ouster-64 等）的光束模式，再逆投影回点云。当对比损失最大化不同光束模式下的特征相似性时，模型学到对 LiDAR 稀疏模式不变的表示。两种变体：Single Pattern（变换为单一随机配置）和 PolarMix（混合多种配置的方位角裁剪）。

**设计动机**：现有 SSL 方法使用随机丢点和立方体裁剪作为增强，不能模拟真实的 LiDAR 光束模式差异。通过真实的传感器参数变换（FOV、通道数、分辨率），使预训练的单一模型可迁移到不同 LiDAR。

### 关键设计3：框架的通用性设计

**功能**：即插即用地提升任何对比学习 SSL 方法。

**核心思路**：PSA 扩展可应用于 DepthContrast（场景级）和 SegContrast（区域级）等不同粒度的对比学习方法。回归头作为并行分支不增加预训练时间（实际上因更快收敛减少了 33%）。使用 MoCo 框架中的 query/momentum 编码器。

**设计动机**：避免设计新的 SSL 框架，而是作为通用插件提升现有方法。对比学习的粒度（场景/区域/点级）各有利弊，PSA 扩展在所有粒度上都有效。

### 损失函数

$\mathcal{L} = \beta_1 \mathcal{L}_{con} + \beta_2 \mathcal{L}_{reg}$，其中 $\mathcal{L}_{con}$ 为 InfoNCE 对比损失，$\mathcal{L}_{reg}$ 为 smooth L1 边界框回归损失。$\beta_1 = 1.0$，$\beta_2 = 0.5$。

## 实验关键数据

### 主实验：预训练Waymo → 微调不同数据集（1% 标签，mIoU）

| 方法 | Waymo | nuScenes | SemanticKITTI |
|------|-------|----------|---------------|
| No pretraining | 49.34 | 35.71 | 41.95 |
| DepthContrast | 50.36 | 35.32 | 48.77 |
| **PSA-DC (Ours)** | **53.02** (+2.66) | **37.75** (+2.43) | **49.92** (+1.15) |
| SegContrast | 53.50 | 36.01 | 49.72 |
| **PSA-SC (Ours)** | **54.36** (+0.86) | **37.89** (+1.87) | **52.11** (+2.39) |

### 与 SOTA 对比（SemanticKITTI 预训练+微调）

| 方法 | 1% | 10% | 50% | 100% |
|------|-----|-----|-----|------|
| BEVContrast | 44.1 | 56.3 | 62.2 | 63.1 |
| MAELi | 41.1 | 56.3 | - | 63.5 |
| **PSA-SC (Ours)** | **45.9** | **58.5** | **63.3** | **64.2** |

### 跨 LiDAR 泛化（SemanticKITTI → nuScenes, 1% 标签）

| 方法 | mIoU |
|------|------|
| TARL | 25.9 |
| **PSA-SC (Ours)** | **29.5** (+3.56) |

### 关键发现

- PSA 扩展在所有基线方法和数据集上都带来显著提升，特别是在极低标签比例(1%)下改善最明显。
- PSA-DC 使场景级方法性能接近区域级方法，说明边界框回归帮助场景级方法学到了更细粒度的特征。
- 仅用 10% SemanticKITTI 标签即可达到 BEVContrast 用 100% 标签的性能（节省 10 倍标注）。
- 单一模型预训练在 Waymo 上，直接迁移到 nuScenes/SemanticKITTI 均显著优于其他方法，证明跨传感器泛化。
- 预训练时间不增加甚至减少 33%，因为边界框回归加速了有用特征的学习。

## 亮点与洞察

1. **识别对比学习的根本局限**：指出对比损失导致几何不变性这一被忽视的问题，并提供了简洁有效的解决方案。
2. **自监督边界框回归**：首次将边界框回归从监督学习引入 SSL 预训练，利用无监督聚类生成回归目标。
3. **跨传感器泛化**：通过真实 LiDAR 参数变换（而非简单随机丢点）实现对不同传感器的鲁棒迁移。

## 局限与展望

- 预处理依赖 HDBSCAN 聚类质量，可能存在过/欠分割问题。
- 边界框拟合假设物体是朝上的，对倾斜物体不适用。
- LiDAR 增强仅支持从密到稀的变换，预训练在稀疏 LiDAR（如 nuScenes）时不适用。
- 回归目标基于聚类簇，地面点和超大簇被排除，可能遗漏部分场景信息。

## 相关工作与启发

- **DepthContrast / SegContrast**：两个主要基线方法，PSA 扩展以即插即用方式显著提升两者性能。
- **BEVContrast**：此前 SOTA，使用 BEV 特征对比，PSA-SC 在所有标签比例上超越。
- **LiDomAug**：LiDAR 光束增强方法，本文将其从域适应引入 SSL 并验证了有效性。

## 评分

⭐⭐⭐⭐ — 识别了对比学习的根本性局限，解决方案简洁而有效（仅添加一个轻量回归头）。10 倍标签节省的结果非常实用。跨传感器迁移能力是自动驾驶领域的实际刚需。

<!-- RELATED:START -->

## 相关论文

- [WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion](weathergen_a_unified_diverse_weather_generator_for_lidar_point_clouds_via_spider.md)
- [RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow](voteflow_enforcing_local_rigidity_in_self-supervised_scene_flow.md)
- [Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting](point-to-region_loss_for_semi-supervised_point-based_crowd_counting.md)
- [How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning](../../NeurIPS2025/autonomous_driving/how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)

<!-- RELATED:END -->
