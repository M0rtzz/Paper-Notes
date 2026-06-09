---
title: >-
  [论文解读] Rectify the Regression Bias in Long-Tailed Object Detection
description: >-
  [ECCV 2024][目标检测][长尾目标检测] 首次揭示并系统解决长尾目标检测中被忽视的**回归偏差**问题：稀有类别的类别专属(class-specific)回归头参数因样本不足导致泛化能力差，通过添加额外的类别不可知(class-agnostic)回归分支进行权衡…
tags:
  - "ECCV 2024"
  - "目标检测"
  - "长尾目标检测"
  - "回归偏差"
  - "类别不可知"
  - "LVIS"
  - "边界框回归"
---

# Rectify the Regression Bias in Long-Tailed Object Detection

**会议**: ECCV 2024  
**arXiv**: [2401.15885](https://arxiv.org/abs/2401.15885)  
**代码**: 无  
**领域**: 目标检测  
**关键词**: 长尾目标检测, 回归偏差, 类别不可知, LVIS, 边界框回归

## 一句话总结

首次揭示并系统解决长尾目标检测中被忽视的**回归偏差**问题：稀有类别的类别专属(class-specific)回归头参数因样本不足导致泛化能力差，通过添加额外的类别不可知(class-agnostic)回归分支进行权衡，在LVIS等数据集上取得了SOTA性能。

## 研究背景与动机

长尾目标检测面临极端的类别分布不平衡问题。现有方法（如EQLv2、SeeSaw、ECM等）几乎全部集中于解决**分类偏差**，通过重加权、过采样、均衡分组等策略防止尾部类别被压制。然而，检测流程中除了分类分支，还有同样重要的**回归分支**——没有人关注过它。

作者通过实验发现了一个关键现象：
- **RCNN回归分支**的loss存在严重的类别差异——稀有类别的回归loss显著高于频繁类别，表明回归质量很差
- 而**RPN回归分支**的loss则几乎是平衡的（各类loss接近）

两者的关键差异在于：RPN的回归头是**类别不可知**的（所有类共享同一组参数），而RCNN的回归头是**类别专属**的（每个类有自己的参数）。稀有类别在使用类别专属回归头时，由于训练样本极少且面临尺度偏移等问题，无法学到好的回归参数。

作者还进一步发现，稀有类别在训练集和验证集之间的**目标尺度偏移**远大于频繁类别，加剧了回归的难度。

一组关键验证实验：将分类器替换为GT标签后，用class-agnostic回归头的稀有类APr从0.7暴增到54.6，甚至超过了频繁类的40.0！这有力证明了稀有类确实需要**类别不可知的回归**。

## 方法详解

### 整体框架

在标准的Faster-RCNN两阶段检测框架中，RCNN头的回归分支为每个类别i维护一个独立的线性层Wi来预测bbox偏移量。作者提出三种方案来缓解回归偏差，核心思路都是在不同程度上引入类别间的参数共享。

### 关键设计

1. **添加类别不可知分支（CAB）**：最简洁有效的方案。为所有类别额外添加一个共享的回归头W0，最终每个类的回归头变为两者的加权组合：

    $W_i' = \alpha W_0 + (1-\alpha) W_i$
   
   其中α是权衡超参数。稀有类从共享头W0获得更好的泛化能力，频繁类仍可从类别专属头Wi中获益。实验发现α=0.5即可取得最优效果。此方案不需要任何数据集统计信息，且同时利用了类别不可知的先验知识和类别专属的细粒度知识。

2. **聚类回归头（Clustering Heads）**：基于图1(c)中不同类别的目标尺度统计具有相似性的观察。将C个类别按实例数量或平均尺度排序后分成K组，每组共享一个回归头：

    - 排序：按实例数量或尺度将W1,...,WC排序
    - 分组：相邻类别分入同组，每组N=C/K个类别
    - 替换：组内类别共享同一个回归矩阵Wgi
   
   使用尺度统计聚类（k=100）时效果较好，APr从13.4提升到16.7。

3. **合并回归头（Merging Heads）**：最直接的方案，按LVIS预定义的稀有/常见/频繁分区合并回归头。例如让所有稀有类共用一个Wrare。有趣的是，仅合并common类时APr提升最大（从14.2到17.7），作者推测这与训练集和验证集之间的类别划分偏移有关。

### 损失函数 / 训练策略

- 采用与各baseline相同的默认训练策略，仅修改回归头结构
- 使用FP16混合精度训练和warmup策略
- 在MMDetection框架下实现
- 所有实验在8张RTX3090上运行3次取平均以减少方差

## 实验关键数据

### 主实验

**三种方法的对比（Baseline: CE + Mask-RCNN R50-FPN, LVIS1.0）**

| 方法 | AP | APr | APb | APrb |
|------|-----|------|------|------|
| Baseline (class-specific) | 23.7 | 14.2 | 24.7 | 13.4 |
| + CAB (α=0.5) | **25.1** | **17.5** | **27.0** | **18.0** |
| Merging (c) | 25.5 | 17.7 | 27.2 | 17.2 |
| Clustering (k=100, scale) | 25.2 | 16.7 | 26.9 | 16.7 |

**CAB与现有长尾方法结合（LVIS1.0, Mask-RCNN R50-FPN）**

| 方法 | +CAB | APb | APrb | AP | APr |
|------|------|------|------|-----|------|
| RFS  | ✗ | 24.7 | 13.4 | 23.7 | 14.2 |
| RFS  | ✓ | 27.0 (+2.3) | 18.0 (+4.6) | 25.1 | 17.5 |
| EQLv2 | ✗ | 26.0 | 16.1 | 25.2 | 17.4 |
| EQLv2 | ✓ | 28.1 (+2.1) | 20.4 (+4.3) | 26.0 | 19.5 |
| SeeSaw | ✗ | 27.3 | 18.2 | 26.9 | 19.6 |
| SeeSaw | ✓ | 28.9 (+1.6) | 19.9 (+1.7) | 27.7 | 20.2 |
| ECM | ✗ | 27.7 | 17.7 | 27.2 | 19.6 |
| ECM | ✓ | 29.1 (+1.4) | 18.4 (+0.7) | 27.8 | 19.1 |

**与SOTA对比（SeeSaw + CAB = "Our"）**

| 架构 | Backbone | 方法 | AP | APb |
|------|----------|------|-----|------|
| Mask-RCNN | R50 | ECM | 27.2 | 27.7 |
| Mask-RCNN | R50 | **Our** | **27.7** | **28.9** |
| Mask-RCNN | R101 | ECM | 28.6 | 29.3 |
| Mask-RCNN | R101 | **Our** | **29.0** | **30.7** |
| Cascade R-CNN | Swin-T | **Our** | **34.6** | **38.2** |
| Cascade R-CNN | Swin-B | **Our** | **39.9** | **44.2** |

### 消融实验

| α值 | AP | APr | APb | APrb |
|-----|-----|------|------|------|
| 0.0 (baseline) | 23.7 | 14.2 | 24.7 | 13.4 |
| 0.2 | 24.1 | 15.8 | 25.4 | 15.1 |
| **0.5** | **25.1** | **17.5** | **27.0** | **18.0** |
| 0.8 | 24.4 | 17.0 | 25.9 | 16.4 |
| 1.0 (纯agnostic) | 24.7 | 16.7 | 26.7 | 18.3 |

### 关键发现

- CAB对所有现有长尾检测方法都能带来一致且显著的提升，尤其是稀有类
- "RFS+CAB"（使用CE loss）的检测精度几乎赶上SeeSaw方法，说明回归改进是分类改进的有力补充
- 方法对COCO-LT（人工长尾）和COCO2017（相对均衡）数据集同样有效
- 在更严格的评估指标（APboundary、APfixed_bbox）下优势更明显
- 可直接泛化到mask分支设计

## 亮点与洞察

- **问题定义新颖**：首次发现并系统研究长尾检测中的回归偏差，此前所有方法都聚焦于分类偏差
- **思路清晰直觉好**：通过RPN（agnostic）vs RCNN（specific）回归loss的对比自然引出假设
- **GT实验**：Table 1 用GT标签替换分类器的实验设计很精妙，干净地解耦了分类和回归的影响
- **方法极其简单**：核心修改仅一行公式（加权组合），即插即用
- **泛化性强**：跨数据集、跨指标、跨分支（mask）均有效

## 局限与展望

- α的最优值可能因数据集/类别分布而异，目前固定为0.5
- 合并方案（Table 2c）在某些指标下甚至优于CAB，但作者选择了更简单的CAB，未进一步探索自适应合并
- 聚类方案的分组策略较简单（等分），更精细的聚类可能带来进一步提升
- 未讨论在anchor-free检测器（如FCOS）中回归偏差是否同样存在

## 相关工作与启发

- 与BaGS、EQL系列的思路互补：前者解决分类偏差，本文解决回归偏差，两者可叠加
- 类似思想可扩展到其他涉及长尾分布的任务（如姿态估计、关键点检测）
- 受[37]的启发但得出相反结论：[37]认为性能下降主要由分类引起，忽视了回归的影响

## 评分

- 新颖性: ⭐⭐⭐⭐☆（问题发现新颖，方法比较简单）
- 实验充分度: ⭐⭐⭐⭐⭐（多方法、多架构、多backbone、多数据集、多指标）
- 写作质量: ⭐⭐⭐⭐☆（表述清晰、图表丰富）
- 价值: ⭐⭐⭐⭐☆（即插即用的提升手段，对领域有启发性）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection](../../CVPR2025/object_detection/simltd_simple_supervised_and_semi-supervised_long-tailed_object_detection.md)
- [\[ICCV 2025\] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts](../../ICCV2025/object_detection/toward_long-tailed_online_anomaly_detection_through_class-agnostic_concepts.md)
- [\[ECCV 2024\] Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](adaptive_bounding_box_uncertainties_via_twostep_conformal_pr.md)
- [\[ECCV 2024\] Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)
- [\[ECCV 2024\] MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection](mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection.md)

</div>

<!-- RELATED:END -->
