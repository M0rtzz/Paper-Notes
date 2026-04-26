---
title: >-
  [论文解读] BIMBA: Selective-Scan Compression for Long-Range Video Question Answering
description: >-
  [CVPR 2025][视频理解][状态空间模型] 本文提出 BIMBA，基于 Mamba selective scan 的时空 token 选择器，将长视频的 10万+ token 序列压缩 16 倍至 6400 个包含关键信息的 token，在 7 个长视频 VQA 基准上取得 SOTA。
tags:
  - CVPR 2025
  - 视频理解
  - 状态空间模型
  - Mamba
  - 视频压缩
  - 长视频QA
  - 多模态大模型
---

# BIMBA: Selective-Scan Compression for Long-Range Video Question Answering

**会议**: CVPR 2025  
**arXiv**: [2503.09590](https://arxiv.org/abs/2503.09590)  
**代码**: [https://sites.google.com/view/bimba-mllm](https://sites.google.com/view/bimba-mllm)  
**领域**: 视频理解 / 长视频问答  
**关键词**: 状态空间模型、Mamba、视频压缩、长视频QA、多模态大模型

## 一句话总结
本文提出 BIMBA，基于 Mamba selective scan 的时空 token 选择器，将长视频的 10万+ token 序列压缩 16 倍至 6400 个包含关键信息的 token，在 7 个长视频 VQA 基准上取得 SOTA。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）在视频理解上取得重要进展，但处理长视频（分钟/小时级）是核心挑战。以 LLaMA-3.2 为例，图像编码器每帧输出 1600~6400 个 token，128 帧会产生 20-80万 token，远超 LLM 处理能力。

**现有痛点**：(1) 空间/时间池化丢弃重要时空信息；(2) 卷积压缩缺乏长程依赖建模能力；(3) 自注意力计算量随序列长度平方增长，对长序列不可行；(4) Perceiver/Q-former 压缩虽高效但缺乏跨帧分析。

**核心矛盾**：长视频中大量帧高度冗余，但关键事件可能瞬间发生。需要选择性保留重要信息同时大幅压缩序列长度——这需要既能长程建模又计算高效的机制。

**本文目标**：设计高效的长视频 token 压缩模块，将 10万级 token 压缩到千级同时保留关键时空依赖。

**切入角度**：Mamba (S6) 的 selective scan 机制正好满足需求——线性计算复杂度、输入依赖的选择性保留、天然的长程建模能力。

**核心 idea**：用 Mamba 的 selective scan 作为时空 token 选择器，引入交错查询分布和双向扫描来适配视频的时空结构，实现 16 倍压缩且精度不降反升。

## 方法详解

### 整体框架
视频帧→预训练图像编码器→时空 token 序列（64帧 × 40×40 = 102,400 token）→BIMBA 时空 token 选择器→压缩 token（16×20×20 = 6,400）→LLM 解码器生成答案。

### 关键设计

1. **时空 Token 选择器**:

    - 功能：将大量冗余视频 token 压缩为少量信息密集的查询 token
    - 核心思路：(a) 用3D自适应平均池化初始化少量视觉查询 $Q$（从 $L$ 个输入 token 到 $N$ 个查询，$N \ll L$）；(b) 将查询与时空 token 拼接成序列 $Z' = [Z; Q]$；(c) 应用 Mamba selective scan 层，让查询通过选择性状态空间模型从海量 token 中"吸收"关键信息；(d) 提取更新后的查询 $Q'$ 传给 LLM。
    - 设计动机：与池化不同，selective scan 可以根据输入内容动态决定保留什么丢弃什么，对冗余视频内容特别有效。与自注意力不同，计算复杂度线性增长。

2. **交错查询分布（Interleaved Queries）**:

    - 功能：避免查询位置偏差
    - 核心思路：传统做法把查询放在序列末尾，导致查询偏向视频后段的 token。本文将查询均匀交错分布在时空 token 之间，使查询能均匀地与视频各部分交互。
    - 设计动机：有效消除位置偏差，使时空信息更均衡地传递给查询 token。

3. **双向 Selective Scan**:

    - 功能：增强对2D/3D时空结构的捕获能力
    - 核心思路：原始 Mamba 为1D NLP 序列设计，对视觉任务的空间结构不敏感。本文采用前向+后向双向扫描，使模型能从两个方向建模时空依赖。
    - 设计动机：双向扫描在视觉 Mamba 中已被验证有效，可以更好捕获空间结构。

### 损失函数 / 训练策略
标准语言模型自回归损失。冻结图像编码器，训练 token 选择器 + LLM（LoRA）。可选：问题条件token选择（将问题 token 前置，让选择器参考问题上下文）。

## 实验关键数据

### 主实验
7 个长视频 VQA 基准上取得 SOTA：

| 基准 | BIMBA-LLaMA | 之前 SOTA |
|------|-------------|----------|
| PerceptionTest | SOTA | - |
| NExT-QA | 76.61 | 低于本文 |
| EgoSchema | 62.20+ | 低于本文 |
| VNBench | SOTA | - |
| LongVideoBench | SOTA | - |
| Video-MME | SOTA | - |
| MLVU | SOTA | - |

### 消融实验
NExT-QA 数据集上消融（LLaMA 变体）：

| 配置 | 准确率 |
|------|--------|
| 平均池化初始化 + LN + 双向扫描 + 交错查询 (完整) | 75.57 |
| 学习型初始化（无池化） | 68.91 (-6.66) |
| 去掉双向→单向 | 71.16 (-4.41) |
| 末端拼接查询（不交错） | 73.23 (-2.34) |
| 去掉 LayerNorm | 70.56 (-5.01) |

### 关键发现
- BIMBA 在所有序列长度上单调提升，pool 在 16 帧后饱和甚至下降
- 自注意力在 9K-13K token 后 OOM，BIMBA 可以处理 102K token
- 计算成本接近池化（最低），但精度显著更高
- 问题条件选择额外提升 1-2%，说明根据问题上下文做选择是有效的
- 交错查询 (+2.34%) 和双向扫描 (+4.41%) 贡献最大

## 亮点与洞察
- **Mamba 用于视频 token 压缩**：selective scan 的内容感知选择性非常适合视频的高冗余特征——这比池化或固定压缩策略优雅得多。
- **交错查询**：简单但有效的设计，消除了序列位置偏差，对长序列建模有普适价值。
- **精度-效率最优平衡**：接近池化的计算成本，接近自注意力的精度（甚至更好），是长视频 MLLM 的理想压缩模块。

## 局限与展望
- 压缩比固定（16×），自适应压缩比可能更好
- 当前帧级独立编码后再压缩，缺少帧间早期交互
- 对超长视频（小时级）的可扩展性还需验证

## 相关工作与启发
- **vs 池化方法 (Video-ChatGPT等)**: 池化对长序列精度饱和，BIMBA 持续提升
- **vs Perceiver/Q-former**: 固定交叉注意力缺乏长程选择性，BIMBA 精度更高
- **vs VideoMamba**: 用 Mamba 替代自注意力，但未做 token 压缩；本文专注压缩

## 评分
- 新颖性: ⭐⭐⭐⭐ Mamba 用于视频压缩+交错查询+双向扫描的组合很有效
- 实验充分度: ⭐⭐⭐⭐⭐ 7个基准+详细消融+计算成本分析+多变体
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验设计合理
- 价值: ⭐⭐⭐⭐⭐ 对长视频理解有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [\[CVPR 2025\] QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering](question-aware_gaussian_experts_for_audio-visual_question_answering.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](../../NeurIPS2025/video_understanding/egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[CVPR 2026\] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](../../CVPR2026/video_understanding/question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [\[CVPR 2025\] MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)

<!-- RELATED:END -->
