---
title: >-
  [论文解读] CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools
description: >-
  [CVPR 2025][医学图像][手术器械跟踪] 本文提出CholecTrack20数据集，首次为腹腔镜手术器械跟踪引入三种视角的轨迹定义（术中/腹腔内/可见性），包含20个完整手术视频、35K+帧、65K+标注器械实例，基准测试表明当前SOTA方法（<45% HOTA）远不能满足临床需求。
tags:
  - CVPR 2025
  - 医学图像
  - 手术器械跟踪
  - 多视角跟踪
  - 腹腔镜
  - 数据集
  - 多类多目标
---

# CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools

**会议**: CVPR 2025  
**arXiv**: [2312.07352](https://arxiv.org/abs/2312.07352)  
**代码**: [GitHub](https://github.com/camma-public/cholectrack20)  
**领域**: 医学影像 / 手术器械跟踪  
**关键词**: 手术器械跟踪, 多视角跟踪, 腹腔镜, 数据集, 多类多目标

## 一句话总结
本文提出CholecTrack20数据集，首次为腹腔镜手术器械跟踪引入三种视角的轨迹定义（术中/腹腔内/可见性），包含20个完整手术视频、35K+帧、65K+标注器械实例，基准测试表明当前SOTA方法（<45% HOTA）远不能满足临床需求。

## 研究背景与动机

**领域现状**：手术视频中的器械跟踪是计算机辅助手术的关键任务，支撑技能评估、安全区估计和人机协作。现有方法多在通用跟踪数据集上训练，在手术场景（出血、烟雾、反光、器械进出视野）下性能大幅下降。

**现有痛点**：现有手术跟踪数据集使用过于通用的跟踪定义——当器械离开相机视野或退出腹腔时，如何处理轨迹ID缺乏明确规范。这导致不同临床应用无法获得所需的轨迹类型，限制了AI在手术中的实际应用。

**核心矛盾**：手术中器械的"轨迹"在不同应用场景下含义不同——技能评估需要全手术跟踪（术中），工作流分析需要腹腔内跟踪（器械在体内的轨迹），实时反馈需要可见性跟踪（相机视野内的轨迹）。单一跟踪定义无法满足所有需求。

**本文目标**：定义手术器械的三种视角跟踪问题，构建高质量标注数据集，填补手术AI训练数据的空白。

**切入角度**：从临床应用需求出发，反向定义跟踪的形式化——不同的临床任务需要不同粒度的轨迹。

**核心 idea**：三视角跟踪定义（术中/腹腔内/可见性）+ 包含空间位置、器械类别、身份、操作者、手术阶段、视觉挑战等丰富标注的数据集。

## 方法详解

### 整体框架
基于Cholec80和CholecT50原始视频，选取20个完整手术视频，以1fps采样。四名经训练的标注员标注边界框、器械类别（7类）、操作者（4类）、手术阶段（7种）、视觉挑战（8类），以及三种视角下的轨迹ID。标注经严格质量控制。

### 关键设计

1. **三种视角的轨迹形式化**:

    - 功能：为不同临床应用提供适配的跟踪定义
    - 核心思路：(a) 术中轨迹——器械在患者体内首次出现到最后出现的终生跟踪，需跨遮挡、出视野、重新插入的重识别；(b) 腹腔内轨迹——器械从进入腹腔到退出腹腔为一条轨迹，退出后再进入则新开轨迹；(c) 可见性轨迹——器械在相机视野中可见的连续片段为一条轨迹
    - 设计动机：技能评估需要术中（器械全程使用分析），风险预测需要腹腔内（腹腔操作分析），实时辅助需要可见性（当前可见状态）

2. **多维度丰富标注**:

    - 功能：支持多种手术AI任务的训练和评估
    - 核心思路：每个器械实例标注：空间坐标（bbox）、类别（7种器械）、三种视角的轨迹ID、操作者（主刀/助手+左右手）、手术阶段、当前帧的视觉挑战类型
    - 设计动机：器械的身份判断不仅依赖外观，还需结合操作者和trocar口位置等临床知识

3. **严格的标注质量控制**:

    - 功能：确保标注一致性和准确性
    - 核心思路：内评者一致性（Jaccard 99.4%，Cohen's Kappa 94.6%）、评者间一致性（Jaccard 91.8%，Kappa 95.2%）、外科专家仲裁歧义情况（758个不确定样本中133个需修正）
    - 设计动机：手术数据标注需要专业知识，质量控制是数据集可信度的基础

### 损失函数 / 训练策略
数据集论文，不涉及特定模型训练。基准实验使用DeepSORT、ByteTrack等现有跟踪方法。

## 实验关键数据

### 基准实验

| 方法 | HOTA(三视角平均) | 说明 |
|------|----------------|------|
| 当前最优方法 | <45% | 远未达到临床要求 |
| 可见性视角最好 | ~40% | 最简单的跟踪定义 |
| 术中视角较差 | ~30% | 需要跨长时遮挡的重识别 |

### 数据统计

| 指标 | 数值 |
|------|------|
| 视频数 | 20个完整手术 |
| 总帧数 | 35,000+ |
| 标注器械实例 | 65,000+ |
| 器械类别 | 7类 |
| 手术阶段 | 7种 |
| 视觉挑战类型 | 8类 |

### 关键发现
- 所有现有跟踪方法在手术场景中表现不佳（<45% HOTA），说明通用跟踪技术需要手术专用的适配
- 出血和烟雾是性能下降最显著的视觉挑战
- 腹腔内视角跟踪最具挑战性，因为需要推断器械在相机视野外的状态
- 器械更换和重新插入是导致ID switch的主要原因

## 亮点与洞察
- 三种视角的跟踪定义是高度原创的形式化贡献，来源于对临床需求的深入理解
- 标注方案融合视觉线索和临床知识（如trocar口推断操作者），反映了手术AI的领域特殊性
- <45% HOTA的基准结果清楚地展示了当前方法与临床需求之间的巨大差距

## 局限与展望
- 仅包含腹腔镜胆囊切除术一种手术类型
- 1fps标注可能漏掉快速运动
- 7类器械可能不足以覆盖更复杂的手术
- 可拓展到其他手术类型和更高时间分辨率

## 相关工作与启发
- **vs ATLAS Dione**: 仅有检测标注，无多视角跟踪
- **vs CholecT50**: 提供工具-组织交互标注但无跟踪ID
- **vs MOTChallenge/DanceTrack**: 通用视频跟踪数据集，不考虑手术特有的进出腹腔和视觉挑战

## 评分
- 新颖性: ⭐⭐⭐⭐ 三视角跟踪定义是重要的形式化贡献
- 实验充分度: ⭐⭐⭐⭐ 完整的基准测试+质量控制分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，标注流程详实
- 价值: ⭐⭐⭐⭐⭐ 填补手术AI领域数据集空白，有直接临床应用意义

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark](../../AAAI2026/medical_imaging/bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)
- [\[NeurIPS 2025\] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](../../NeurIPS2025/medical_imaging/ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)
- [\[CVPR 2025\] Surg-R1: A Hierarchical Reasoning Foundation Model for Scalable and Interpretable Surgical Decision Support](surg-r1_a_hierarchical_reasoning_foundation_model_for_scalable_and_interpretable.md)
- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](../../NeurIPS2025/medical_imaging/starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)

<!-- RELATED:END -->
