---
title: >-
  [论文解读] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?
description: >-
  [CVPR 2025][视频理解][在线视频理解] OVO-Bench 是首个强调时间戳在视频理解中重要性的在线视频基准，将在线视频理解分为"回溯追踪"、"实时感知"和"前瞻主动响应"三种模式，通过 12 个任务、644 个视频和 2800+ 精细标注评估 Video-LLM 的在线理解能力。
tags:
  - CVPR 2025
  - 视频理解
  - 在线视频理解
  - 视频大语言模型
  - 时序感知
  - 基准测试
  - 实时感知
---

# OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?

**会议**: CVPR 2025  
**arXiv**: [2501.05510](https://arxiv.org/abs/2501.05510)  
**代码**: https://github.com/JoeLeelyf/OVO-Bench  
**领域**: 视频理解  
**关键词**: 在线视频理解, 视频大语言模型, 时序感知, 基准测试, 实时感知

## 一句话总结
OVO-Bench 是首个强调时间戳在视频理解中重要性的在线视频基准，将在线视频理解分为"回溯追踪"、"实时感知"和"前瞻主动响应"三种模式，通过 12 个任务、644 个视频和 2800+ 精细标注评估 Video-LLM 的在线理解能力。

## 研究背景与动机

**领域现状**：Video-LLM（如 GPT-4o、Gemini-1.5-Pro、Qwen2-VL）在离线视频理解基准上取得了令人印象深刻的成绩。然而，现有基准都假设模型能够访问完整视频后再回答问题（离线设置），与真实世界的在线视频助手需求存在巨大鸿沟。

**现有痛点**：(1) 现有离线基准无法评估模型的"时序感知"能力——即模型在视频流的不同时间点提问时应给出不同答案的能力；(2) 少数在线基准（如 VStream-QA、StreamingBench）主要关注利用已有视觉输入立即响应，缺少"等待未来信息再回答"的评估维度；(3) 缺乏对在线视频理解中三种根本不同的推理模式（回溯、实时、前瞻）的系统性评估。

**核心矛盾**：离线和在线视频理解的本质区别在于"时序感知"——同一个问题在视频的不同时间点提出，答案可能不同。现有基准完全忽略了这一关键维度。

**本文目标**：构建一个评估 Video-LLM 在线视频理解能力的全面基准，覆盖回溯追踪、实时感知和前瞻主动响应三种模式。

**切入角度**：受人类视频理解过程的启发，提出"视频思维链时间"（Video Chain-of-Time）概念——面对流式视频中的查询，模型需要决定是利用过去信息立即回答、关注当前帧实时回答，还是等待足够的未来信息再回答。

**核心 idea**：将在线视频理解系统化地分为三种模式（Backward Tracing / Real-Time Perception / Forward Active Responding），设计 12 个任务覆盖这三种模式，通过沿时间轴密集查询的评估 pipeline 模拟连续信息处理。

## 方法详解

### 整体框架
OVO-Bench 评估框架包含三个层次：(1) 数据构建：从多个数据集和网络爬取收集 644 个视频，通过半自动+人工策划生成 2814 个高质量元标注；(2) 提示生成：自动化问答生成 + 视觉相关选项生成 + 人工审核；(3) 评估 pipeline：沿时间轴密集查询 Video-LLM，模拟在线理解场景。

### 关键设计

1. **三种在线理解模式的形式化定义**:

    - 功能：系统化地定义在线视频理解应具备的三种核心能力，指导基准设计。
    - 核心思路：给定时间 $t_0$ 的查询 $Q_{t_0}$ 和流式视频 $X_{(-\infty,+\infty)}$：(a) 回溯追踪 $R_{t_0} = P(Q_{t_0}, X_{(-\infty,-T]})$——利用远期历史信息回答；(b) 实时感知 $R_{t_0} = P(Q_{t_0}, X_{(-T,t_0]})$——理解当前发生的事件；(c) 前瞻主动响应 $R_{(t_0,+\infty)} = P(Q_{t_0}, X_{(t_0,+\infty)})$——延迟响应直到足够信息可用。
    - 设计动机：这三种模式对应了人类在在线视频理解中的三种认知策略：回忆过去、感知当下、预判未来。它们需要的能力完全不同，不应混为一谈。

2. **12 个细分任务设计**:

    - 功能：全面覆盖在线视频理解的各个维度，提供细粒度的能力诊断。
    - 核心思路：回溯追踪包含 3 个任务（情景记忆 EPM、动作序列识别 ASI、幻觉检测 HLD）；实时感知包含 6 个任务（空间理解 STU、物体识别 OJR、属性识别 ATR、动作识别 ACR、OCR、未来预测 FPD）；前瞻主动响应包含 3 个任务（重复事件计数 REC、序列步骤识别 SSR、线索揭示响应 CRR）。每个任务都有精确到帧的时间戳标注。
    - 设计动机：前瞻主动响应（FAR）是 OVO-Bench 的独特贡献——这是首个要求模型"判断何时回答比回答什么更重要"的评估维度。传统基准假设模型应当立即回答，但真正的在线助手需要知道何时该等待。

3. **密集时间轴查询评估 Pipeline**:

    - 功能：让离线 Video-LLM 也能在模拟在线场景中被评估，实现公平比较。
    - 核心思路：对于回溯追踪和实时感知任务，将视频截断到查询时间点，转为多项选择题评估。对于前瞻主动响应任务，设计"多触发密集查询"pipeline——在查询提出后的多个时间点反复询问模型是否已有足够信息回答，模拟模型持续适应新视觉输入的过程。选项生成使用基于规则+视觉驱动的变换，引入来自原始视频的迷惑信息提高难度。
    - 设计动机：目前主流强大的 Video-LLM 都是离线模型，直接排除它们会遗漏大量有价值的比较。模拟在线设置让我们能评估"是否有效利用 SOTA 离线模型进行在线理解"这个重要问题。

### 损失函数 / 训练策略
OVO-Bench 是评估基准，不涉及训练。标注通过 GPT-4o 和 Gemini-1.5 Pro 半自动生成后经人工精调。视频来源多样化：Ego4D、QA-Ego4D、OpenEQA、COIN、CrossTask、MovieNet 等公开数据集 + YouTube 爬取。

## 实验关键数据

### 主实验
11 个 Video-LLM 在 OVO-Bench 上的整体表现：

| 模型 | 实时感知 Avg | 回溯追踪 Avg | 前瞻响应 Avg | 总体 Avg |
|------|-------------|-------------|-------------|---------|
| Human Agents | 93.20 | 92.33 | 92.90 | **92.81** |
| Gemini 1.5 Pro | 69.32 | 62.54 | 57.15 | 63.00 |
| GPT-4o | 64.46 | 60.75 | 53.40 | 59.54 |
| Qwen2-VL | 中等 | 中等 | 低 | 中等 |
| Flash-VStream（在线模型）| 更低 | 更低 | 更低 | 更低 |

### 消融实验
| 分析维度 | 发现 |
|---------|------|
| 离线 vs 在线模型 | 离线 SOTA 显著优于在线模型（反直觉） |
| 前瞻主动响应 | 所有模型表现最差，与人类差距最大 |
| GPT-4 (blind) | 纯文本 LLM 在回溯追踪上 53.82，说明存在文本偏置 |
| 幻觉检测 (HLD) | 模型普遍低于50%，存在严重应答偏差 |
| OCR 任务 | Gemini 最强（85.91），说明视觉编码能力差异大 |

### 关键发现
- 人类与最强模型之间存在约 30% 的巨大差距（92.81 vs 63.00），表明在线视频理解远未被解决
- 前瞻主动响应（FAR）是最具挑战性的模式——所有模型在 REC 任务上仅约 30%，说明模型缺乏"何时回答"的判断能力
- 反直觉地，在线模型（如 Flash-VStream）表现不如离线模型被直接用于在线场景，暴露了当前在线模型架构的不足
- 幻觉检测任务上模型普遍不到 50%，说明模型倾向于对不存在的事件编造答案
- 纯文本 LLM（GPT-4-turbo blind）在回溯追踪上达到 53.82，提示存在文本捷径

## 亮点与洞察
- **前瞻主动响应的开创性提出**：首次系统化评估模型"判断何时有足够信息回答"的能力，这是真正在线助手的核心需求但被所有先前基准忽视——这一贡献定义了新的研究方向
- **在线/离线模型对比的反直觉发现**：强大的离线模型在在线场景中表现优于专门设计的在线模型，这一发现对在线视频理解的研究方向有重要指导意义
- **Video Chain-of-Time 概念**：类比 Chain-of-Thought，提出面对流式查询时的时间推理范式——应当先判断信息是来自过去、当下还是未来，再决定回答策略

## 局限与展望
- 评估仍以多选题为主，无法完全捕获开放式在线交互的复杂性
- 前瞻主动响应的评估 pipeline（密集多触发查询）计算开销较大
- 644 个视频和 2814 个 QA 对的数据规模仍有限
- 仅评估了英文场景，跨语言在线视频理解未涉及

## 相关工作与启发
- **vs StreamingBench**: StreamingBench 主要关注实时感知，缺少前瞻主动响应评估。OVO-Bench 的三种模式划分更为完整
- **vs Video-MME/LongVideoBench**: 这些基准评估离线长视频理解，不涉及时间戳感知。OVO-Bench 强调"同一问题在不同时间点答案不同"
- **vs E.T.Bench**: E.T.Bench 探索时序事件检测但仍在离线设置下。OVO-Bench 进一步推进到在线设置

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 前瞻主动响应模式和三模式框架是该领域的开创性贡献
- 实验充分度: ⭐⭐⭐⭐ 11个模型的全面评估，但数据规模可更大
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，分类体系严谨，图示丰富
- 价值: ⭐⭐⭐⭐⭐ 定义了在线视频理解的评估范式，揭示了当前模型的巨大差距，将推动该方向的研究

<!-- RELATED:START -->

## 相关论文

- [LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](../../NeurIPS2025/video_understanding/livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)
- [Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs](q-bench-video_benchmark_the_video_quality_understanding_of_lmms.md)
- [DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)
- [Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence](reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)
- [LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant](lion-fs_fast_slow_video-language_thinker_as_online_video_assistant.md)

<!-- RELATED:END -->
