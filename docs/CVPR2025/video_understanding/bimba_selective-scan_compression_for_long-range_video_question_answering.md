---
title: "Bimba: Selective-Scan Compression for Long-Range Video Question Answering"
description: "提出Bimba，利用Mamba选择性状态空间模型压缩长视频token序列，实现高效的长视频问答"
tags:
  - CVPR2025
  - 视频理解
  - Video QA
  - Mamba
  - State Space Model
  - Token Compression
---

# Bimba: Selective-Scan Compression for Long-Range Video Question Answering

**会议**: CVPR 2025  
**arXiv**: 2503.09590  
**主题**: 长视频问答 / Token压缩  

## 研究背景与动机

长视频问答（Long-Range Video Question Answering）要求模型理解数分钟甚至数小时的视频内容并回答相关问题。这对现有的视频理解模型提出了严峻的挑战：长视频被编码为大量的视觉token，直接输入大语言模型会导致计算成本爆炸式增长。

**核心挑战分析：**

**Token序列过长**：一段10分钟的视频，按常规采样和编码方式可能产生数万个视觉token。现有LLM的上下文窗口难以容纳如此长的序列

**计算复杂度**：Transformer的自注意力机制具有 $O(n^2)$ 的计算复杂度，处理长序列时计算成本急剧升高

**信息冗余**：视频中存在大量的时间冗余，相邻帧之间的变化通常很小，但简单的均匀采样可能遗漏关键信息

**问题相关性**：不同问题关注视频的不同部分，理想的压缩策略应该是问题感知的，但这增加了模型设计的复杂性

**现有压缩方法的局限**：简单的池化或均匀采样丢失了细粒度的时序信息；基于注意力的选择方法计算成本高

Mamba（选择性状态空间模型）的出现为长序列建模提供了新的可能性——它具有线性的计算复杂度，同时通过选择性机制能够动态关注序列中的重要部分。本文的核心思想是利用Mamba的选择性扫描机制来高效压缩长视频的token序列。

## 方法详解

### 整体框架

Bimba的整体流程包含三个阶段：视觉编码、选择性扫描压缩和语言模型推理。

### 视频编码

输入视频首先通过预训练的视觉编码器（如CLIP-ViT）提取每帧的视觉特征。对于一个包含 $T$ 帧、每帧产生 $N$ 个patch token的视频，生成的token序列长度为 $T \times N$。

### 选择性扫描压缩模块

这是Bimba的核心创新。该模块基于Mamba的选择性状态空间模型（Selective State Space Model, S6），对长视频token序列进行高效压缩：

**状态空间模型基础：**

连续状态空间模型定义为：

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

其中 $A$ 是状态矩阵，$B$ 和 $C$ 是投影矩阵。

**选择性机制：**

与传统SSM使用固定参数不同，Mamba使得参数 $B$、$C$ 和步长 $\Delta$ 成为输入依赖的：

$$B_t = \text{Linear}_B(x_t), \quad C_t = \text{Linear}_C(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}_\Delta(x_t))$$

这使得模型能够根据输入内容选择性地保留或遗忘信息。

**双向扫描设计：**

| 扫描方向 | 作用 | 优势 |
|----------|------|------|
| 正向扫描 | 从视频起始到结束 | 捕捉因果时序关系 |
| 反向扫描 | 从视频结束到起始 | 利用未来上下文信息 |
| 双向融合 | 合并两个方向的特征 | 获得更完整的时序理解 |

### 压缩策略

选择性扫描后，模型通过以下策略将长序列压缩为固定长度的紧凑表示：

- **状态聚合**：利用SSM的隐状态作为压缩后的表示，隐状态维度远小于原始token数量
- **重要性加权**：根据选择性门控值对不同时刻的信息进行加权，确保关键帧的信息被保留
- **层次化压缩**：先在帧内进行空间压缩，再在帧间进行时序压缩

### 问答推理

压缩后的视频表示与文本问题一起输入大语言模型进行推理。由于视频token已被大幅压缩，LLM可以高效地处理并生成答案。

## 实验结果

### 效率分析

Bimba的核心优势在于效率：

| 方面 | 效果 |
|------|------|
| 计算复杂度 | 线性 $O(n)$，远优于Transformer的 $O(n^2)$ |
| 内存消耗 | 大幅降低，支持更长的视频输入 |
| 推理速度 | 显著提升，适合实际部署 |
| 信息保留 | 选择性机制保留关键信息 |

### 主要结果

在多个长视频问答基准上，Bimba在保持竞争力的问答准确率的同时，显著降低了计算成本。特别是在处理超长视频时，效率优势更加明显。

### 消融实验

- 双向扫描优于单向扫描，验证了双向上下文的重要性
- 选择性机制是关键，去掉后性能明显下降
- 层次化压缩策略优于简单的全局压缩

## 总结与展望

Bimba将Mamba的选择性扫描机制引入长视频理解领域，提供了一种高效的长视频token压缩方案。线性复杂度使其能够扩展到更长的视频输入，而选择性机制确保了关键信息的保留。未来工作可以探索将该方法扩展到视频生成任务，以及与更先进的视觉编码器结合。

<!-- RELATED:START -->

## 相关论文

- [EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering](question-aware_gaussian_experts_for_audio-visual_question_answering.md)
- [Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](../../CVPR2026/video_understanding/question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](../../ICLR2026/video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [EgoPointVQA: Gesture-Based Egocentric Video Question Answering](../../CVPR2026/video_understanding/egopointvqa_gesture_based_egocentric_video_qa.md)

<!-- RELATED:END -->
