---
title: >-
  [论文解读] A Distractor-Aware Memory for Visual Object Tracking with SAM2
description: >-
  [CVPR 2025][图像分割][SAM2] 提出SAM2.1++的干扰物感知记忆模型（DAM），将SAM2的记忆拆分为近期外观记忆（RAM，确保分割精度）和干扰物解析记忆（DRM，确保跟踪鲁棒性），通过内省式更新策略检测干扰物并自动存储锚帧，在7个基准上设立新SOTA。
tags:
  - CVPR 2025
  - 图像分割
  - SAM2
  - 干扰物感知记忆
  - 视觉目标跟踪
  - 记忆管理
  - 鲁棒性
---

# A Distractor-Aware Memory for Visual Object Tracking with SAM2

**会议**: CVPR 2025  
**arXiv**: [2411.17576](https://arxiv.org/abs/2411.17576)  
**代码**: https://github.com/jovanavidenovic/DAM4SAM  
**领域**: 分割 / 视觉目标跟踪  
**关键词**: SAM2, 干扰物感知记忆, 视觉目标跟踪, 记忆管理, 鲁棒性

## 一句话总结

提出SAM2.1++的干扰物感知记忆模型（DAM），将SAM2的记忆拆分为近期外观记忆（RAM，确保分割精度）和干扰物解析记忆（DRM，确保跟踪鲁棒性），通过内省式更新策略检测干扰物并自动存储锚帧，在7个基准上设立新SOTA。

## 研究背景与动机

**领域现状**：基于记忆的跟踪器（如SAM2）通过将近期帧存入记忆缓冲区并用cross-attention定位目标，已在多个基准上达到SOTA。SAM2使用初始帧+最近6帧的FIFO记忆。

**现有痛点**：干扰物（与目标视觉相似的区域）是跟踪失败的主要原因。现有记忆管理策略仅存储近期帧，无法有效区分目标和干扰物。当目标短暂离开视野时，记忆会被空mask帧填满，导致重检测失败。

**核心矛盾**：准确分割需要最新的目标外观（时序相关），但鲁棒的干扰物处理需要包含干扰物的锚帧（时序无关）。这两种需求本质上不同，不应用同一种记忆策略处理。

**核心idea**：按功能划分记忆为RAM（近期外观，带时间编码，FIFO更新）和DRM（干扰物解析，不带时间编码，仅在检测到干扰物且跟踪可靠时更新）。

## 方法详解

### 整体框架

在SAM2.1基础上重新设计记忆模型，不需要任何额外训练。将总共6个记忆槽一分为二：3个给RAM（最近帧），3个给DRM（锚帧+初始帧）。提出基于SAM2多假设输出的干扰物检测机制和可靠性守卫策略。

### 关键设计

1. **近期外观记忆（RAM）管理**：

    - 功能：存储最近目标外观，确保分割精度
    - 核心思路：每$\Delta=5$帧更新一次（避免高度相关帧的冗余），始终包含最新一帧。关键改进——目标不在场时不更新（predicted mask为空时跳过），避免空mask帧污染记忆
    - 设计动机：[51]证明高频更新导致视觉冗余降低定位能力，减少更新频率+跳过目标缺失帧可保持外观多样性

2. **干扰物解析记忆（DRM）管理**：

    - 功能：存储包含关键干扰物的锚帧，确保跟踪鲁棒性和重检测能力
    - 核心思路：利用SAM2的多假设输出——SAM2预测3个mask，选择IoU最高的。关键发现：在跟踪失败前，干扰物信息实际已出现在备选mask中。通过计算输出mask和备选mask联合区域的bounding box面积比，当比率低于$\theta_{anc}=0.7$时检测到干扰物。仅在跟踪可靠时更新（IoU > $\theta_{IoU}=0.8$且面积在中位数±20%内），防止错误分割帧污染记忆
    - 设计动机：DRM不使用时间编码——因为干扰物解析信息不应被时间远近偏置，应作为"无时间先验"

3. **DiDi数据集**：

    - 功能：从多个标准benchmark中蒸馏出干扰物密集的序列子集
    - 核心思路：用DINO2特征计算每帧的干扰物分数——目标区域外与目标区域内特征相似度高的像素比例超过0.5则视为含干扰物帧。选取≥1/3帧含干扰物的序列
    - 最终获得180个序列共274K帧，专注暴露干扰物处理能力

### 训练策略

完全无需训练，直接利用预训练SAM2.1的已有组件实现DAM结构。这是因为SAM2已支持可变记忆长度和时间编码/无时间编码的灵活配置。

## 实验关键数据

### 主实验：DiDi数据集SOTA对比

| 方法 | Quality | Accuracy | Robustness |
|------|---------|----------|------------|
| TransT | 0.465 | 0.669 | 0.678 |
| SeqTrack | 0.529 | 0.714 | 0.718 |
| KeepTrack | 0.502 | 0.646 | 0.748 |
| ODTrack | 0.608 | 0.740 | 0.809 |
| Cutie | 0.575 | 0.704 | 0.776 |
| SAM2.1 | 0.649 | 0.720 | 0.887 |
| SAMURAI | 0.680 | 0.722 | 0.930 |
| SAM2.1Long | 0.646 | 0.719 | 0.883 |
| **SAM2.1++** | **0.694** | **0.727** | **0.944** |

### 消融实验：记忆设计逐步验证

| 配置 | Quality | Accuracy | Robustness |
|------|---------|----------|------------|
| SAM2.1 (基线) | 0.649 | 0.720 | 0.887 |
| +仅目标在场时更新 | 0.665 | 0.723 | 0.903 |
| +降低更新频率Δ=5 | 0.667 | 0.718 | 0.914 |
| +DRM仅可靠时更新 | 0.672 | 0.710 | 0.932 |
| +DRM仅检测干扰物时更新 | 0.644 | 0.691 | 0.913 |
| **完整DAM (两条件同时)** | **0.694** | **0.727** | **0.944** |
| DAM+DRM时间编码 | 0.669 | 0.711 | 0.925 |

### 关键发现

- **DRM两个更新条件缺一不可**：仅检测干扰物更新（无可靠性守卫）反而降低性能（0.644 < 0.667），因为不可靠时的错误分割会污染DRM
- **不使用时间编码对DRM很关键**：加时间编码后Quality下降3.6%，证明干扰物解析信息确实不应有时间偏置
- **鲁棒性提升是核心**：相比SAM2.1基线，Quality +7%，主要来自Robustness +6%（0.887→0.944），说明DAM确实减少了跟踪丢失
- VOT2022上EAO达0.753（比赛冠军MS_AOT 0.673提升12%），VOT2020上EAO达0.729

## 亮点与洞察

- **功能性记忆拆分**：首次按功能（精度vs鲁棒性）划分跟踪记忆，概念清晰、设计优雅。RAM用时间编码因为近期帧更相关，DRM不用因为干扰物信息应无时间偏向
- **利用已有的多假设输出**：SAM2的3个输出mask中，备选mask包含了干扰物信息，这是被之前研究完全忽视的宝贵信号。这种"挖掘已有输出中的隐藏信息"的思路很有启发
- **零训练成本**：完全利用预训练SAM2的已有组件，无需任何额外训练就能获得显著性能提升。极其实用

## 局限与展望

- 干扰物检测基于bounding box面积比的简单规则，可能遗漏无法用面积变化捕捉的干扰物类型
- DRM固定使用3个槽，可能不足以应对长序列中多个不同干扰物的情况
- 仅在单目标跟踪场景验证，多目标场景的干扰物处理更复杂

## 相关工作与启发

- **vs SAMURAI**：SAMURAI也改进SAM2的记忆管理，但使用运动线索进行记忆选择。SAM2.1++通过功能性拆分+内省式更新，在DiDi上Quality高2%且设计更简洁
- **vs SAM2Long**：SAM2Long用约束树搜索优化长序列，但在干扰物场景下与基线SAM2.1性能持平，说明长期记忆不等于干扰物处理能力
- **vs KeepTrack**：KeepTrack显式建模多目标检测关联网络，但架构复杂。DAM更轻量且效果更好

## 评分

- 新颖性: ⭐⭐⭐⭐ 功能性记忆拆分概念新颖，利用多假设输出检测干扰物思路巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 7个benchmark全面评估，消融实验逐步验证每个设计决策
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑严谨，每个设计选择都有实验验证，DiDi数据集贡献也很有价值
- 价值: ⭐⭐⭐⭐⭐ 零训练改进SAM2跟踪性能显著，在多个benchmark设立新SOTA，代码开源

<!-- RELATED:START -->

## 相关论文

- [Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](../../ICLR2026/segmentation/efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)
- [SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)
- [SAMWise: Infusing Wisdom in SAM2 for Text-Driven Video Segmentation](samwise_infusing_wisdom_in_sam2_for_text-driven_video_segmentation.md)
- [Visual Consensus Prompting for Co-Salient Object Detection](visual_consensus_prompting_for_co-salient_object_detection.md)
- [MatAnyone: Stable Video Matting with Consistent Memory Propagation](matanyone_stable_video_matting_with_consistent_memory_propagation.md)

<!-- RELATED:END -->
