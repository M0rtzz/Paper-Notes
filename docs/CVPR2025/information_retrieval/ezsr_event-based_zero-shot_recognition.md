---
title: >-
  [论文解读] EZSR: Event-based Zero-Shot Recognition
description: >-
  [CVPR 2025][事件相机][零样本识别] 提出基于事件相机的零样本物体识别方法，利用CLIP跨模态对齐实现事件数据的零样本泛化
tags:
  - CVPR 2025
  - 事件相机
  - 零样本识别
  - CLIP
  - 跨模态对齐
---

# EZSR: Event-based Zero-Shot Recognition

**会议**: CVPR 2025  
**arXiv**: [2407.21616](https://arxiv.org/abs/2407.21616)  
**代码**: 待确认  
**领域**: 事件相机 / 零样本识别  
**关键词**: 事件相机, 零样本识别, CLIP, 事件编码器, 跨模态

## 一句话总结
提出 EZSR，通过将事件相机数据与 CLIP 的 RGB 嵌入空间对齐，实现基于事件数据的零样本物体识别。

## 研究背景与动机
**领域现状**：事件相机具有低延迟、高动态范围的独特优势，但缺乏大规模标注数据限制了其在识别任务上的应用。

**现有痛点**：现有方法利用 CLIP 指导事件识别，但简单地优化事件-RGB 嵌入相似度忽略了两种模态之间的本质差异（事件=异步时序变化 vs RGB=同步帧）。

**本文目标** 在没有事件数据标注的情况下实现零样本事件物体识别。

**核心 idea**：设计事件专用编码器，通过精细的跨模态对齐将事件表示映射到 CLIP 空间。

## 方法详解

### 关键设计
1. **事件专用编码器**：针对事件数据的时序稀疏特性设计专门的编码器架构。
2. **渐进式跨模态对齐**：从浅层到深层逐步对齐事件特征和 RGB 特征。
3. **时序聚合策略**：有效聚合事件流的时间信息为紧凑表示。

## 实验关键数据

### 关键发现
- 在 N-Caltech101、N-ImageNet 等事件数据集上实现了有竞争力的零样本性能
- 事件专用编码器显著优于将事件简单转为帧后使用 CLIP 视觉编码器
- 跨模态对齐的渐进策略比端对端对齐更稳定

## 亮点与洞察
- 将 CLIP 的零样本能力迁移到事件相机模态
- 事件专用编码器的设计考虑了模态特性

## 局限与展望
- 事件-RGB 域差距仍然存在，极端场景（快速运动、极低光照）下性能下降
- 可以探索利用事件数据的独特优势（如运动信息）来超越 RGB-based CLIP

<!-- RELATED:START -->

## 相关论文

- [Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning](few-shot_recognition_via_stage-wise_retrieval-augmented_finetuning.md)
- [Explaining CLIP Zero-shot Predictions Through Concepts](../../CVPR2026/information_retrieval/explaining_clip_zero-shot_predictions_through_concepts.md)
- [Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](../../NeurIPS2025/information_retrieval/worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)
- [OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description](../../AAAI2026/information_retrieval/oad-promoter_enhancing_zero-shot_vqa_using_large_language_models_with_object_att.md)
- [BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](../../ICLR2026/information_retrieval/btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)

<!-- RELATED:END -->
