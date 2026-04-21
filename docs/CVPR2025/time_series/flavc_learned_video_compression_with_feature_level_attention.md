---
title: >-
  [论文解读] FLAVC: Learned Video Compression with Feature Level Attention
description: >-
  [CVPR 2025][时间序列][学习型视频压缩] 提出 FLAVC，在学习型视频压缩框架中引入特征级注意力机制，在特征域（而非像素域）中更高效地利用运动估计和时空上下文信息。
tags:
  - CVPR 2025
  - 时间序列
  - 学习型视频压缩
  - 特征级注意力
  - 运动估计
  - 上下文编码
---

# FLAVC: Learned Video Compression with Feature Level Attention

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 待确认  
**领域**: 视频压缩  
**关键词**: 学习型视频压缩, 特征级注意力, 运动估计, 上下文编码

## 一句话总结
提出 FLAVC，在学习型视频压缩框架中引入特征级注意力机制，在特征域（而非像素域）中更高效地利用运动估计和时空上下文信息。

## 研究背景与动机
**领域现状**：学习型视频压缩（LVC）通过深度学习方法减少时序数据冗余，近期方法将压缩操作从像素域转移到特征域，显著提升了编码性能。

**现有痛点**：现有特征域方法在运动估计和运动补偿后直接融合特征，未能充分利用特征通道和空间维度的冗余分布差异。

**本文目标** 如何在特征域中更精细地利用时空冗余。

**核心 idea**：在特征域的运动补偿和上下文预测环节引入注意力机制，自适应地调整不同通道和空间位置的信息保留/压缩力度。

## 方法详解

### 关键设计
1. **通道注意力增强的运动补偿**：自适应地为不同特征通道分配运动补偿权重。
2. **空间注意力指导的上下文预测**：根据空间位置的运动复杂度调整上下文建模粒度。
3. **多尺度特征融合**：在不同分辨率上聚合注意力加权的时间特征。

## 实验关键数据

### 关键发现
- BD-rate 相比基线方法降低 5-10%
- 在高运动场景中注意力机制的增益更显著
- 计算开销增加有限，适合实际部署

## 亮点与洞察
- 特征域注意力是一个自然且有效的改进方向
- 与像素域注意力相比，特征域操作计算效率更高

## 局限与展望
- 注意力模块增加的参数量对低延迟场景的影响需评估
- 可以探索与传统编码标准（如 H.266/VVC）的混合框架

<!-- RELATED:START -->

## 相关论文

- [DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification](dejavid_encoder-agnostic_learned_temporal_matching_for_video_classification.md)
- [AttentionPredictor: Temporal Patterns Matter for KV Cache Compression](../../NeurIPS2025/time_series/attentionpredictor_temporal_patterns_matter_for_kv_cache_com.md)
- [Learning Time-Scale Invariant Population-Level Neural Representations](../../NeurIPS2025/time_series/learning_time-scale_invariant_population-level_neural_representations.md)
- [Customizing the Inductive Biases of Softmax Attention using Structured Matrices](../../ICML2025/time_series/customizing_the_inductive_biases_of_softmax_attention_using_structured_matrices.md)
- [Relational Feature Caching for Accelerating Diffusion Transformers](../../ICLR2026/time_series/relational_feature_caching_for_accelerating_diffusion_transformers.md)

<!-- RELATED:END -->
