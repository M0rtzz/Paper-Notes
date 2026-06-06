---
title: >-
  [论文解读] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark
description: >-
  [NeurIPS 2025][视频理解][第一人称视频] 提出 EgoGazeVQA，首个融合用户眼动注视数据的第一人称视频问答基准，通过注视引导的提示策略（文本/视觉/显著性图）显著提升 MLLM 对用户意图的理解能力…
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "第一人称视频"
  - "注视引导"
  - "视频问答"
  - "用户意图理解"
  - "多模态大模型"
---

# EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark

**会议**: NeurIPS 2025  
**arXiv**: [2509.07447](https://arxiv.org/abs/2509.07447)  
**代码**: [https://taiyi98.github.io/projects/EgoGazeVQA](https://taiyi98.github.io/projects/EgoGazeVQA)  
**领域**: 视频理解  
**关键词**: 第一人称视频, 注视引导, 视频问答, 用户意图理解, 多模态大模型

## 一句话总结
提出 EgoGazeVQA，首个融合用户眼动注视数据的第一人称视频问答基准，通过注视引导的提示策略（文本/视觉/显著性图）显著提升 MLLM 对用户意图的理解能力，Gaze Salience Map 策略最高可将 MiniCPM-o 的准确率从35.9%提升至53.7%。

## 研究背景与动机

**领域现状**：多模态大模型在视频理解方面取得显著进展，但现有基准主要基于第三人称视角，无法直接捕捉用户的注意力焦点和行为意图。

**现有痛点**：现有第一人称视频QA基准（如QaEgo4D、EgoSchema）忽略了一个关键的第一人称信号——注视(Gaze)。注视直接反映用户的注意力和意图，而大部分用户问题本质上取决于用户正在看什么。

**核心矛盾**：MLLM使用全局图像帧构建视觉token，提供了广泛上下文但无法捕捉相机佩戴者的显式意图信号，导致模型难以准确推断用户在看什么、想做什么。

**本文目标**：构建首个整合注视数据的第一人称VQA基准，评估MLLM能否利用注视信息增强对用户意图的理解。

**切入角度**：将注视坐标信息以三种不同方式（文本、视觉标记、显著性图）融入MLLM提示中。

**核心 idea**：注视信号是理解第一人称视频中用户意图的关键缺失模态，通过注视引导提示可以显著弥补 MLLM 的意图理解能力不足。

## 方法详解

### 整体框架
从 Ego4D、EgoExo4D 和 EGTEA Gaze+ 三个带眼动追踪数据的第一人称视频数据集中，提取视频片段和注视坐标，使用 Qwen2.5-VL 生成空间/时间感知的意图相关QA对，经人工审核后形成包含913个视频、1757个QA对的基准数据集。

### 关键设计

1. **数据构建流水线**:

    - 功能：生成高质量的注视引导QA对
    - 核心思路：每9帧为一组关键帧段，配以归一化注视坐标，输入 Qwen2.5-VL 生成3组QA对（每组5选1），通过相关性、可回答性、流畅度、准确性、简洁性、难度6个维度的人工审核
    - 设计动机：干扰项设计包含反因果选项、空间邻近陷阱、高显著性干扰等，确保需要利用注视信息才能正确推理

2. **三种注视引导提示策略**:

    - 功能：以不同方式将注视信息编码到MLLM输入中
    - 核心思路：(1) 文本提示(T)：将注视坐标作为文本直接输入；(2) 视觉提示(V)：在视频帧上标注注视点；(3) 显著性图(S)：将注视轨迹生成热力图作为额外视觉上下文
    - 设计动机：不同MLLM可能对不同形式的注视编码敏感度不同，需要系统比较

3. **LoRA微调实验**:

    - 功能：评估微调能否弥合MLLM对注视信号的理解不足
    - 核心思路：由于MLLM训练数据中很少包含注视信号，通过LoRA微调帮助模型学习利用注视线索
    - 设计动机：验证注视理解能力是否可以通过少量微调获得

### 损失函数 / 训练策略
基准评测不涉及训练策略。LoRA微调实验使用标准的指令跟随损失。

## 实验关键数据

### 主实验

| 模型 | 无注视 | +文本(T) | +视觉(V) | +显著性图(S) |
|------|--------|---------|---------|-------------|
| InternVL2.5-8B | 58.3% | 60.1% | **60.6%** | 59.9% |
| GPT-4o mini | 57.0% | 58.8% | 58.5% | 58.7% |
| MiniCPM-o 2.6 | 35.9% | 50.0% | 50.2% | **53.7%** |
| 人类基线 | - | - | - | 83.8% |

### 消融实验

| 维度 | 无注视 | 最佳注视策略 | 提升 |
|------|--------|-------------|------|
| 空间 | 36.1-50.4% | 49.4-55.0% | +3-13% |
| 时间 | 34.5-51.1% | 41.6-52.7% | +2-7% |
| 因果 | 32.5-75.6% | 64.4-80.3% | +5-32% |

### 关键发现
- MiniCPM-o 获益最大（+17.8%），说明较弱模型从注视信号中获益更多
- 显著性图策略整体最优，因为它同时编码了注视轨迹的空间和时间信息
- 因果推理维度受益最明显，因为注视直接指向了用户行为的原因和意图
- 人类基线(83.8%)与最佳模型(60.6%)差距巨大，说明注视理解仍有很大提升空间

## 亮点与洞察
- 首次将注视数据引入VQA基准，填补了第一人称视频理解的重要空白。注视作为用户意图的直接证据，其价值被长期低估
- 显著性图策略的成功启示：将稀疏的点坐标转化为密集的空间先验（热力图），可以更有效地被视觉模型利用

## 局限与展望
- 数据集规模较小（1757个QA对），可能限制统计结论的可靠性
- 注视数据需要专用设备采集，限制了实际应用场景
- 未来可探索从普通视频中估计注视方向，再用于意图理解的自动化流程

## 相关工作与启发
- **vs EgoSchema**: EgoSchema 专注于长视频推理但不利用注视数据，EgoGazeVQA 则将注视作为核心信号
- **vs GazeGPT**: GazeGPT 展示了注视对MLLM UI的帮助，但缺乏标准化基准和评估，本文填补了这一空白

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个注视引导VQA基准，切入点新颖
- 实验充分度: ⭐⭐⭐⭐ 多模型、多策略、多维度对比充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机有说服力
- 价值: ⭐⭐⭐⭐ 对第一人称AI助手发展有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](../../CVPR2025/video_understanding/egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](../../CVPR2026/video_understanding/movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)
- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](../../CVPR2026/video_understanding/do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[CVPR 2026\] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](../../CVPR2026/video_understanding/herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)
- [\[NeurIPS 2025\] DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering](dsas_a_universal_plug-and-play_framework_for_attention_optimization_in_multi-doc.md)

</div>

<!-- RELATED:END -->
