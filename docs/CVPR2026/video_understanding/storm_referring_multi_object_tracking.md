---
title: >-
  [论文解读] STORM: End-to-End Referring Multi-Object Tracking in Videos
description: >-
  [CVPR 2026][视频理解][指代多目标跟踪] STORM 是首个端到端的多模态大语言模型框架用于指代多目标跟踪（RMOT），通过任务组合学习策略大幅减少对 RMOT 标注数据的依赖，并构建了高质量 STORM-Bench 数据集。
tags:
  - CVPR 2026
  - 视频理解
  - 指代多目标跟踪
  - 多模态大语言模型
  - 任务组合学习
  - 数据集
---

# STORM: End-to-End Referring Multi-Object Tracking in Videos

**会议**: CVPR 2026  
**arXiv**: [2604.10527](https://arxiv.org/abs/2604.10527)  
**代码**: [https://github.com/amazon-science/storm-referring-multi-object-grounding](https://github.com/amazon-science/storm-referring-multi-object-grounding)  
**领域**: 视频理解  
**关键词**: 指代多目标跟踪, 多模态大语言模型, 任务组合学习, 视频理解, 数据集

## 一句话总结
STORM 是首个端到端的多模态大语言模型框架用于指代多目标跟踪（RMOT），通过任务组合学习策略大幅减少对 RMOT 标注数据的依赖，并构建了高质量 STORM-Bench 数据集。

## 研究背景与动机

**领域现状**：指代多目标跟踪要求模型根据文本描述在视频中跟踪所有匹配的目标。现有 RMOT 方法将目标定位和跟踪拆分为独立模块，依赖外部检测器。

**现有痛点**：(1) RMOT 训练视频极度稀缺；(2) 现有数据集标注模糊且领域受限；(3) 模块化方法难以理解复杂的指代表达式和推理因果/关系依赖。

**核心矛盾**：RMOT 是一个需要联合视觉-语言理解和时序跟踪的复杂任务，但标注成本极高导致无法获得足够的训练数据。

**本文目标**：统一定位和跟踪，消除外部模块依赖，解决数据稀缺问题。

**切入角度**：借鉴 LLM 预训练中"先学基础能力再微调"的思路，将 RMOT 分解为图像定位和单目标跟踪两个基础子任务。

**核心 idea**：用任务组合学习将 RMOT 分解为数据丰富的子任务，先学定位和跟踪基础能力，再用少量 RMOT 数据微调。

## 方法详解

### 整体框架
STORM 采用 LLaVA 风格的 MLLM 架构：ViT 视觉编码器提取帧级视觉特征 → MLP 投影器映射到文本空间 → LLaMA-based LLM 自回归生成目标边界框序列。输出格式为结构化文本：`Object 1: Frame 1: [x1,y1,x2,y2], ...`。

### 关键设计

1. **任务组合学习（TCL）**:

    - 功能：通过分解 RMOT 为子任务来减少对大规模 RMOT 数据的依赖
    - 核心思路：Stage 1 在大规模图像定位和单目标跟踪数据上预训练，学习跨模态对齐和时序一致性；Stage 2 在 STORM-Bench 上微调，用 Chain-of-Thought 训练策略引导模型先在首帧定位再跨帧跟踪
    - 设计动机：RMOT 标注极其昂贵，但图像定位和 SOT 数据丰富，通过渐进式学习可以有效组合这些能力

2. **端到端统一架构**:

    - 功能：在单一 MLLM 框架内完成定位和跟踪
    - 核心思路：所有目标的边界框直接以纯文本形式输出，利用预训练语言模型的推理能力处理复杂指代表达式。对长视频切分为短片段，用前一片段的预测框作为下一片段的提示来拼接轨迹
    - 设计动机：消除外部检测器/跟踪器避免了模块间的信息损失，让模型能学习统一的时空表示

3. **STORM-Bench 数据集（自底向上标注）**:

    - 功能：提供高质量 RMOT 训练和评估数据
    - 核心思路：先定位目标生成描述（利用 MLLM + 三种不同视觉输入验证），再用 LLM 组合多目标指代表达式并二次验证。包含 15.7K 视频、251K 图像、200K 指代表达式
    - 设计动机：现有 RMOT 数据集标注模糊且规模小，自底向上的标注比自顶向下更鲁棒

### 损失函数 / 训练策略
标准的 next-token prediction 交叉熵损失。

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | STORM | 之前SOTA | 提升 |
|-------------|------|-------|----------|------|
| RefCOCO val | Acc@0.5 | 89.1 | 88.7 (M-GPT2) | +0.4 |
| Elysium RSOT | AUC | 84.1 | 83.3 (Elysium) | +0.8 |
| STORM-Bench RMOT | HOTA | 42.9 | 37.9 (Qwen2.5-VL) | +5.0 |

### 消融实验

| 配置 | HOTA | 说明 |
|------|------|------|
| Full STORM | 42.9 | 完整模型 |
| w/o Stage 1 预训练 | 35.2 | 子任务预训练贡献显著 |
| w/o CoT 推理 | 39.6 | Chain-of-Thought 提升跟踪一致性 |

### 关键发现
- TCL 策略显著减少了对 RMOT 数据的需求，图像定位子任务的训练也反过来提升了 RMOT 性能
- 使用更长更全面的提示词可以进一步提升跟踪性能（87.4→87.5 AUC）
- 端到端方法在复杂指代表达式上的优势明显，Grounding DINO + 跟踪器的管线方法 HOTA 仅 31.7

## 亮点与洞察
- **任务分解学习的实用性**：将复杂任务分解为数据丰富的子任务是解决标注瓶颈的通用策略，可迁移到其他需要复杂标注的视频任务
- **自底向上标注管线**：比自顶向下更鲁棒的标注方式，利用"描述比定位简单"的不对称性

## 局限与展望
- 基于 8B 参数模型，推理效率仍是部署瓶颈
- 长视频分段处理可能丢失跨片段的跟踪一致性
- 自由形式文本输出有时会产生格式错误的边界框

## 相关工作与启发
- **vs ReferGPT**: ReferGPT 在 MLLM 上增加匹配模块，STORM 完全端到端
- **vs Elysium**: Elysium 用自顶向下标注有噪声，STORM 的自底向上标注更可靠

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个端到端 MLLM RMOT 框架，TCL 策略巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 三个层级的评估（图像/SOT/RMOT）非常完整
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详尽
- 价值: ⭐⭐⭐⭐ 数据集和方法都有较高价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT](rethinking_two-stage_referring-by-tracking_in_referring_multi-object_tracking_ma.md)
- [\[ECCV 2024\] OneTrack: Demystifying the Conflict Between Detection and Tracking in End-to-End 3D Trackers](../../ECCV2024/video_understanding/onetrack_demystifying_the_conflict_between_detection_and_tracking_in_end-to-end_.md)
- [\[CVPR 2025\] WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](../../CVPR2025/video_understanding/wilor_end-to-end_3d_hand_localization_and_reconstruction_in-the-wild.md)
- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_test_time_calibration_experience_intuition_mot.md)

</div>

<!-- RELATED:END -->
