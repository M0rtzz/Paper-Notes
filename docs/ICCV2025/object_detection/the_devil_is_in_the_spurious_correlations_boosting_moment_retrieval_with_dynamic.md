---
title: >-
  [论文解读] The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning
description: >-
  [ICCV 2025][目标检测][Moment Retrieval] 首次揭示文本查询与视频背景帧之间的虚假相关性是时刻检索性能瓶颈的根本原因，提出 TD-DETR 框架通过动态上下文视频合成和文本-动态交互增强两个策略来缓解该问题，在 QVHighlights 和 Charades-STA 上达到 SOTA。
tags:
  - "ICCV 2025"
  - "目标检测"
  - "Moment Retrieval"
  - "Spurious Correlation"
  - "Video Synthesis"
  - "Temporal Dynamics"
  - "DETR"
---

# The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning

**会议**: ICCV 2025  
**arXiv**: [2501.07305](https://arxiv.org/abs/2501.07305)  
**代码**: 即将公开  
**领域**: 目标检测  
**关键词**: Moment Retrieval, Spurious Correlation, Video Synthesis, Temporal Dynamics, DETR

## 一句话总结

首次揭示文本查询与视频背景帧之间的虚假相关性是时刻检索性能瓶颈的根本原因，提出 TD-DETR 框架通过动态上下文视频合成和文本-动态交互增强两个策略来缓解该问题，在 QVHighlights 和 Charades-STA 上达到 SOTA。

## 研究背景与动机

时刻检索（Moment Retrieval）旨在根据文本查询定位视频中的对应片段。现有 DETR 架构的方法在文本-视频对齐上表现良好，但预测准确的时间跨度仍是主要挑战。

本文揭示了一个被忽视的根本原因：**虚假相关性**（Spurious Correlation）——模型过度将文本查询与背景帧关联，而非区分目标片段。例如，SOTA 方法 BAM-DETR 在原始视频和将目标片段遮挡后的视频上预测出几乎相同的时间跨度，说明模型依赖的是背景帧而非目标片段本身。

这在时刻检索领域是首次被系统性研究，与图像领域的空间偏差不同，视频任务中的虚假相关性更为复杂。

## 方法详解

### 整体框架

TD-DETR（Temporal Dynamics DETR）包含三个核心模块：
1. **Video Synthesizer**：通过视频合成构建动态上下文
2. **Dynamics Enhancement**：通过时序动态与文本交互增强表征
3. **Transformer Encoder-Decoder + Prediction Heads**：标准 DETR 检测头

### 关键设计

1. **视频合成器（Video Synthesizer for Dynamic Context）**：

    - **虚假对选择**：在训练 batch 中，为每个视频 $V_i$ 选择余弦相似度最高的视频 $V_k$ 构成虚假对 $p_i = \{V_i, V_k\}$，确保合成视频足够具有挑战性
    - **动态上下文合成**：保留目标片段的完整性（选择概率设为 1），以采样比例 $\alpha$ 从 $V_i$ 的非目标帧中采样，以 $1-\alpha$ 从 $V_k$ 中采样，拼接成新视频 $\tilde{V}_i$，同步更新 ground truth 标注
    - 设计动机：让模型在不同的背景上下文中学习识别目标片段，减少对特定背景的依赖

2. **时序动态增强（Dynamics Enhancement）**：

    - **Dynamic Tokenizer**：在视频序列前拼接可学习起始 token $st$，通过逐帧差分 $T = \{\tilde{v}_1 - st, \tilde{v}_2 - \tilde{v}_1, \ldots\}$ 提取时序动态表征，几乎无额外计算开销
    - **Text-Dynamics Interaction**：使用交叉注意力分别计算视频-文本和动态-文本的交互表征，通过加权融合 $\tilde{V}_i' = \beta \cdot \tilde{V}_i + (1-\beta) \cdot T'$ 将动态信息注入视频表征
    - 设计动机：让模型不仅关注静态的视觉帧语义，也关注背景无关的时序变化信息，建立文本与目标片段之间的稳健关联

3. **匈牙利匹配与预测**：

    - 对合成对中的两个视频分别进行预测，与对应 ground truth 匹配
    - 显著性得分也同步预测，用于 Highlight Detection

### 损失函数 / 训练策略

$$\mathcal{L}_{total} = \mathcal{L}_{hl} + \mathcal{L}_{moment}$$

其中：
- $\mathcal{L}_{moment} = \lambda_{L_1}\mathcal{L}_{L_1} + \lambda_{iou}\mathcal{L}_{gIoU} + \lambda_{cls}\mathcal{L}_{cls}$（L1损失 + gIoU损失 + 分类交叉熵）
- $\mathcal{L}_{hl} = \lambda_{margin}\mathcal{L}_{margin} + \lambda_{cont}\mathcal{L}_{cont} + \lambda_{neg}\mathcal{L}_{neg}$（边际排序 + 排序对比 + 负样本损失）

## 实验关键数据

### 主实验（QVHighlights test split）

| 方法 | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|------|--------|--------|---------|----------|---------|
| Moment-DETR | 52.89 | 33.02 | 54.82 | 29.40 | 30.73 |
| QD-DETR | 62.40 | 44.98 | 62.52 | 39.88 | 39.86 |
| CG-DETR | 65.40 | 48.40 | 64.50 | 42.80 | 42.90 |
| BAM-DETR | 64.53 | 48.64 | 64.57 | 46.33 | 45.36 |
| **TD-DETR (Ours)** | **64.53** | **50.37** | **66.21** | **47.32** | **46.69** |
| SnAG w/ TD-DETR | 66.48 | 52.93 | 63.71 | 49.11 | 46.75 |

**Charades-STA test split**：

| 方法 | R1@0.5 | R1@0.7 |
|------|--------|--------|
| QD-DETR | 57.31 | 32.55 |
| BAM-DETR | 59.95 | 39.38 |
| **TD-DETR (Ours)** | **60.89** | **40.35** |

### 消融实验

**采样比例 $\alpha$ 消融（QVHighlights val）**：

| $\alpha$ | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|----------|--------|--------|---------|----------|---------|
| 0.0 | 11.61 | 3.35 | 23.93 | 7.50 | 10.09 |
| 0.3 | 65.10 | 51.94 | 65.77 | 48.13 | 47.55 |
| 0.7 | **65.88** | **53.67** | 66.43 | **49.86** | **49.05** |
| 0.9 | 64.19 | 51.23 | 66.29 | 48.88 | 47.94 |

**动态增强 $\beta$ 消融**：$\beta=0.7$ 性能最优，纯动态（$\beta=0$）或不注入动态（$\beta=1$）都导致性能下降。

**采样策略对比**：

| 策略 | R1@0.7 | mAP@0.75 | mAP |
|------|--------|----------|-----|
| baseline (QD-DETR) | 46.66 | 41.82 | 41.22 |
| w/ random | 51.29 | 47.82 | 47.56 |
| w/ similarity | **53.67** | **49.86** | **49.05** |

### 关键发现

- 相似度引导的 pair 选择优于随机选择，但随机选择也已显著优于 baseline
- 使用 InternVideo2 特征时，TD-DETR 在 Charades-STA 上 R1@0.5 达到 73.49，R1@0.7 达到 53.01，大幅领先
- 动态上下文验证集上 TD-DETR 同样保持 SOTA，证实虚假相关性确实被缓解
- 可视化表明 baseline 在目标被遮挡后仍预测相同位置，而 TD-DETR 能正确调整

## 亮点与洞察

- **问题定义精准**：首次在时刻检索中系统性识别和分析虚假相关性问题，通过遮挡实验提供了直观有力的证据
- **方法简洁有效**：视频合成不需要任何外部数据或生成模型，仅利用 batch 内视频重组；时序动态 tokenizer 通过简单差分实现，几乎零额外开销
- **泛化性强**：可适配 SnAG、QD-DETR 等不同架构，且在 InternVideo2 特征上同样有效
- 虚假相关性的分析范式可推广到其他视频理解任务

## 局限与展望

- 合成策略依赖 batch 内视频对，batch 较小时可能无法找到高质量配对
- 时序差分是线性操作，可能无法捕获复杂的非线性时序模式
- 仅在 QVHighlights 和 Charades-STA 两个数据集上验证
- 未探讨在更长视频（如电影级）场景中虚假相关性的表现

## 相关工作与启发

- **BAM-DETR / CG-DETR**：当前 DETR-based 时刻检索 SOTA，聚焦架构设计但忽略了虚假相关性
- **SBI (Self-Blended Images)**：视频合成策略的灵感来源之一
- **Spurious Correlation in Vision**：图像领域已有大量研究，但视频领域特别是时刻检索中首次被系统探讨

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次揭示时刻检索中的虚假相关性问题，视角新颖
- **实验充分度**: ⭐⭐⭐⭐ 消融全面，含跨架构泛化和虚假相关性专项评估
- **写作质量**: ⭐⭐⭐⭐ 问题阐述清晰，图示直观
- **价值**: ⭐⭐⭐⭐ 提出的框架简洁通用，虚假相关性的分析视角有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning](augmenting_moment_retrieval_zero-dependency_two-stage_learning.md)
- [\[CVPR 2026\] Beyond Caption-Based Queries for Video Moment Retrieval](../../CVPR2026/object_detection/beyond_caption-based_queries_for_video_moment_retrieval.md)
- [\[CVPR 2025\] Boosting Domain Incremental Learning: Selecting the Optimal Parameters Is All You Need](../../CVPR2025/object_detection/boosting_domain_incremental_learning_selecting_the_optimal_parameters_is_all_you.md)
- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[CVPR 2026\] PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection](../../CVPR2026/object_detection/paq-detr_learning_pattern_and_quality-aware_dynamic_queries_for_object_detection.md)

</div>

<!-- RELATED:END -->
