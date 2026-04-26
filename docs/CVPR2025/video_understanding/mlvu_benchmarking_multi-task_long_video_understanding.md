---
title: >-
  [论文解读] MLVU: Benchmarking Multi-task Long Video Understanding
description: >-
  [CVPR 2025][视频理解][长视频理解] 提出 MLVU 基准，通过9种多样化评测任务、多种视频类型和灵活的时长设置，系统评估多模态大模型在长视频理解上的能力，揭示现有模型在处理长视频时的显著不足。
tags:
  - CVPR 2025
  - 视频理解
  - 长视频理解
  - 视频基准测试
  - 多模态大模型
  - 多任务评估
  - 视频问答
---

# MLVU: Benchmarking Multi-task Long Video Understanding

**会议**: CVPR 2025  
**arXiv**: [2406.04264](https://arxiv.org/abs/2406.04264)  
**代码**: https://github.com/JUNJIE99/MLVU (有)  
**领域**: 视频理解  
**关键词**: 长视频理解, 视频基准测试, 多模态大模型, 多任务评估, 视频问答

## 一句话总结

提出 MLVU 基准，通过9种多样化评测任务、多种视频类型和灵活的时长设置，系统评估多模态大模型在长视频理解上的能力，揭示现有模型在处理长视频时的显著不足。

## 研究背景与动机

### 领域现状

**领域现状**：现有视频理解基准存在三大问题：(1) 视频长度不足，大部分基准的视频仅几秒到几十秒，无法反映真正的长视频理解能力；(2) 视频类型和评测任务缺乏多样性，通常聚焦于单一类型（如第一人称视频）或单一任务（如字幕生成）；(3) 评测设计不当——很多问题可以不看视频直接回答，例如基于知名电影常识或仅关注单帧信息。

MLVU 的核心目标是构建一个 **长度充分、类型丰富、任务多样** 的长视频理解评测基准，确保评测任务必须基于对长视频的深入理解才能完成。

### 解决思路

**本文目标**：### 整体框架

MLVU 是一个包含 3,102 个问题、9 类任务的多任务长视频理解基准，基于 1,730 个视频构建。


## 方法详解

### 整体框架

MLVU 是一个包含 3,102 个问题、9 类任务的多任务长视频理解基准，基于 1,730 个视频构建。视频长度从 3 分钟到 2 小时不等，平均约 15 分钟。视频被进一步分割为递增片段（如前3分钟、前6分钟、全长），使得不同时长下的评估成为可能。

### 关键设计

1. **三层次评测任务体系**: 将长视频理解分为三个层次——(a) 整体理解 (Holistic LVU)：包括主题推理 (TR)、异常识别 (AR)、视频摘要 (VS)，需利用全局信息；(b) 单细节理解 (Single-Detail LVU)：包括 Needle QA (NQA)、自我推理 (ER)、剧情问答 (PQA)、子场景字幕 (SSC)，需定位并理解特定片段；(c) 多细节理解 (Multi-Detail LVU)：包括动作排序 (AO)、动作计数 (AC)，需联合利用多处信息。

2. **Needle QA 创新设计**: 借鉴文本领域的 Needle-In-the-Haystack-Search，将短视频（needle）随机插入长背景视频中，模型需根据问题推断 needle 位置并回答问题。这有效测试了模型在长视频中定位和利用局部信息的能力。

3. **MLVU Time-ladder 衍生数据集**: 为同一类任务在不同时长（180s、360s、600s）下创建评测，系统研究视频长度对模型性能的影响，实现灵活的长度维度分析。

### 损失函数 / 训练策略

本文是基准测试论文，不涉及模型训练。评估策略包括：多选题用准确率衡量；生成任务（VS 和 SSC）使用 GPT-4 评分比对生成内容与人工标注。所有模型以零样本方式评测。

## 实验关键数据

### 主实验

| 模型 | M-Avg (多选) | G-Avg (生成) | TR | NQA | AO | AC |
|------|-------------|-------------|-----|-----|-----|-----|
| GPT-4o | 54.5% | 5.87% | 83.7% | 42.9% | 46.2% | 35.0% |
| LLaVA-OneVision | 51.7% | 4.42% | 83.5% | 46.7% | 35.7% | 23.3% |
| Video-XL | 46.3% | 4.21% | 78.0% | 50.0% | 48.6% | 31.7% |
| VideoLLaMA2 | 48.4% | 3.95% | 80.2% | 36.7% | 42.9% | 16.7% |
| InternVL-2 | 47.5% | 3.90% | 85.7% | 48.3% | 32.9% | 15.0% |
| 随机基线 | 16.7% | - | 16.7% | 16.7% | 16.7% | 16.7% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 上下文长度: GPT-4o 16帧→256帧 | M-Avg: 45.8→54.5 (+8.7) | 更长输入显著提升LVU性能 |
| 上下文长度: MGV 16帧→90帧 | M-Avg: 24.2→31.7 (+7.5) | 开源模型同样获益 |
| LLM骨干: Vicuna-7B→13B | M-Avg: 13.3→18.8 (+5.5) | 更大骨干网络有帮助 |
| LLM骨干: LLaMA-7B→Mistral-7B | M-Avg: 20.6→31.7 (+11.1) | 更强骨干网络提升显著 |
| 图像理解 (MMMU): GPT-4V→4o | 58.1→63.8 (MMMU) / 43.3→45.8 (M-Avg) | 图像理解能力与LVU高度相关 |

### 关键发现

1. 长视频理解对现有MLLM仍是巨大挑战：GPT-4o 的 M-Avg 仅 54.5%，多数模型在 NQA、AO、AC 等任务上接近随机水平
2. 随视频长度增加，所有模型性能持续下降：短视频模型在10分钟视频上接近随机表现
3. 多细节任务（AO、AC）比单细节任务困难很多，probe 数量增加时性能急剧下降
4. 先进长视频模型（LongVA、Video-XL）对引用片段在视频中的位置不敏感，表现更稳定
5. 上下文长度、图像理解能力、LLM骨干是影响LVU性能的三个关键因素

## 亮点与洞察

- 评测任务设计精巧：Needle QA 巧妙借鉴了文本领域经验，能有效测试长视频中的信息检索能力
- Time-ladder 设计使得视频长度对性能的影响可被量化分析，这在以往基准中罕见
- 涵盖了电影、监控、第一人称、卡通、游戏等多种视频类型，贴近真实应用场景
- 开放式和多选题双轨评测，全面考察模型的不同能力维度

## 局限与展望

- 生成任务（VS、SSC）依赖 GPT-4 评分，可能引入评估偏差
- 视频来源的多样性仍有拓展空间（如医学、教育等专业领域视频较少）
- 仅评估了帧采样方式输入，未探索连续视频流处理
- 缺乏对音频信息的考量，而音频在很多视频理解任务中至关重要

## 相关工作与启发

- 与 Video-MME、LongVideoBench 等并行工作互补，MLVU 在视频长度和任务多样性上更有优势
- Needle QA 的设计思路可扩展到其他模态的长上下文评测
- Time-ladder 思路可应用于评估模型的上下文窗口利用效率

## 评分

- 新颖性: ⭐⭐⭐⭐ 三层次任务体系和 Needle QA 设计有创新，但基准类工作新颖性有限
- 实验充分度: ⭐⭐⭐⭐⭐ 23个模型的全面评测，多维度消融分析非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分析有深度
- 价值: ⭐⭐⭐⭐⭐ 填补了长视频理解系统性评测的空白，对社区有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding](hierarq_task-aware_hierarchical_q-former_for_enhanced_video_understanding.md)
- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](rewind_understanding_long_videos_with_instructed_learnable_memory.md)
- [\[CVPR 2025\] DrVideo: Document Retrieval Based Long Video Understanding](drvideo_document_retrieval_based_long_video_understanding.md)
- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](seal_semantic_attention_learning_for_long_video_representation.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](../../ICCV2025/video_understanding/4d_bench_benchmarking_multimodal_llms_for_4d_object_understanding.md)

<!-- RELATED:END -->
