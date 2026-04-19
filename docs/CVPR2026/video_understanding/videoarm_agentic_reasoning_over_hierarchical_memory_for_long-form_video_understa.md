---
title: >-
  [论文解读] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding
description: >-
  [CVPR 2026][视频理解][长视频理解] VideoARM 提出了一种基于分层多模态记忆（HM3）的 Agent 推理范式，通过"观察-思考-行动-记忆"的自适应循环和粗到细的工具调用策略，在长视频理解基准上超越 SOTA 的同时将 token 消耗降低到 DVD 的 1/34。
tags:
  - CVPR 2026
  - 视频理解
  - 长视频理解
  - Agent推理
  - 分层记忆
  - 粗到细推理
  - Token效率
---

# VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding

**会议**: CVPR 2026  
**arXiv**: [2512.12360](https://arxiv.org/abs/2512.12360)  
**代码**: https://milvlg.github.io/videoarm/  
**领域**: 视频理解 / LLM Agent  
**关键词**: 长视频理解, Agent推理, 分层记忆, 粗到细推理, Token效率

## 一句话总结
VideoARM 提出了一种基于分层多模态记忆（HM3）的 Agent 推理范式，通过"观察-思考-行动-记忆"的自适应循环和粗到细的工具调用策略，在长视频理解基准上超越 SOTA 的同时将 token 消耗降低到 DVD 的 1/34。

## 研究背景与动机

1. **领域现状**：长视频理解需要在数十分钟到数小时的视频中捕捉细粒度时空细节并推理长程依赖。近年来 MLLM 的长上下文能力和跨模态对齐为此提供了基础。现有 LLM 驱动方法分两类：手工推理流程（如 LLoVi、VideoTree）和 Agent 自主推理（如 DVD）。

2. **现有痛点**：(a) 手工方法（VideoTree）将视频分段→聚类→打分→建树→推理，流程固定限制了自主性，无法充分利用更强基座模型的推理能力。(b) Agent 方法（DVD）先对所有 10 秒片段做详尽预处理建数据库，token 消耗极高（30 分钟视频 ~400 万 token），且数据库是静态的无法在推理中更新。

3. **核心矛盾**：穷举式预处理既浪费 token 又引入与 query 无关的冗余信息；而手工流程限制了模型自主推理的潜力。如何在保持推理质量的同时大幅降低 token 消耗？

4. **本文目标** 设计一种自适应的、按需的 Agent 推理范式，替代静态穷举预处理，实现高效且灵活的长视频理解。

5. **切入角度**：用分层记忆（感知→结果→工作）替代预构建数据库，让 Agent 按需动态构建记忆；用粗到细的工具集替代检索范式，让 Agent 通过时序聚焦和局部分析逐步缩小搜索范围。

6. **核心 idea**：用动态构建的三层记忆（HM3）替代静态数据库，让 MLLM Agent 在"观察-思考-行动-记忆"循环中按需探索视频，实现 token 高效的长视频推理。

## 方法详解

### 整体框架
VideoARM 由两个核心组件构成：(1) 分层多模态记忆（HM3）——三层结构（感知记忆、结果记忆、工作记忆）动态记录 Agent 的观察和推理状态；(2) 粗到细视频推理 Agent——由 Controller（OpenAI o3）驱动，配合时序聚焦工具集和多模态理解工具集，在 observe-think-act-memorize 循环中自主推理。最大推理步数 $N=10$。

### 关键设计

1. **分层多模态记忆 HM3（Hierarchical Multimodal Memory）**:

    - 功能：作为 Agent 的上下文知识库，动态构建并在执行过程中持续更新
    - 核心思路：三层设计——**感知记忆**（Sensory Memory）包含长期感知池 $P_l$（当前处理时间段的帧，用 3×2 网格压缩）和短期感知池 $P_s$（局部探索的帧和音频，分析完即清除）；**结果记忆**（Result Memory）记录每轮工具输出和对应时间区间，形成时序有序的证据历史；**工作记忆**（Working Memory）记录 Controller 每次工具调用前的推理轨迹和意图，外化推理链以释放上下文压力。
    - 设计动机：感知记忆提供当前视觉上下文，结果记忆让 Agent 反思历史避免重复动作，工作记忆解决上下文溢出问题。三层从感知→语义→认知逐级抽象，形成完整的推理支撑。

2. **时序聚焦工具集（Temporal Scoping Tools）**:

    - 功能：自适应缩小 Agent 的关注范围到 query 相关区域
    - 核心思路：**Interval Localizer** 根据 HM3 中的上下文信号定位与 query 最相关的帧区间 $T_{long}$，自适应决定采样帧数 $N_1$（30-150 帧），将帧合成为紧凑的 3×2 网格图并更新长期感知池。**Clip Explorer** 在长期焦点的局部区间 $T_{local}$ 进行短暂的细粒度探测（不改变全局焦点），用固定帧数 $N_2$ 采样并存入短期感知池，同时存储音频片段。
    - 设计动机：Interval Localizer 实现粗粒度的时序漏斗效应——缩小关注范围，Clip Explorer 实现细粒度的假设验证——在局部快速收集证据。二者配合实现粗到细的探索策略。

3. **多模态理解工具集（Multimodal Understanding Tools）**:

    - 功能：从不同角度提取和验证 query 相关的证据
    - 核心思路：三个互补工具——**Scene Snapper** 对长期感知池中的帧进行总结，生成场景描述 $V_C$，提供全局语义抽象（GPT-4.1/4o 实现）。**Audio Transcriber** 用 whisper-1 转录短期感知池中的音频，补充视觉线索不足时的语义信息。**Clip Analyzer** 对短期感知池中的帧针对子问题 $Q_{sub}$ 进行分析，返回答案 $A_{sub}$ 和置信度 $S_{sub}$，提供细粒度局部语义证据。使用后，结果写入结果记忆，短期感知池清空。
    - 设计动机：三个工具分别覆盖全局概览、听觉补充和局部细节三个维度，Agent 可根据需要灵活组合，平衡广度与深度。

### Controller 和推理循环

Controller 使用 OpenAI o3 实现，遵循精简的 observe-think-act-memorize 循环（类似 ReAct 但有 HM3 支撑）。不预设刚性工作流和工具使用规则，最大化利用 MLLM 的内在推理能力。每轮迭代中：观察 HM3 中的全局-局部上下文 → 思考并生成推理计划 $R_t$ → 选择工具及参数执行 → 将结果写入 HM3。达到步数预算 $N$ 或选择 Answer 动作时终止，生成最终回答。

## 实验关键数据

### 主实验

| 方法 | Video-MME Overall | Video-MME Long | LongVideoBench | EgoSchema |
|------|-------------------|----------------|----------------|-----------|
| GPT-4o | 71.9 | 65.3 | 66.7 | 72.2 |
| OpenAI o3 | - | 63.2 | 67.5 | 63.2 |
| DVD | - | 67.3 | 71.6 | 76.6 |
| VideoLucy | 72.5 | 66.8 | - | - |
| **VideoARM (o3+GPT-4.1)** | **80.1** | **75.3** | **73.7** | **78.2** |
| **VideoARM (o3+GPT-4o)** | **82.8** | **81.2** | **78.0** | 76.2 |

### Token 效率对比

| 方法 | 理论估算 (30min/1query) | 实测 (10 videos/30 queries) |
|------|------------------------|---------------------------|
| DVD | 3.98M tokens | 64.21M tokens |
| **VideoARM** | **0.08M (1/50 of DVD)** | **1.89M (1/34 of DVD)** |

### 消融实验

| 配置 | Video-MME Long |
|------|----------------|
| Full (o3 + GPT-4.1) | 76.5 |
| w/o 短期感知池 | 72.5 (-4.0) |
| w/o 长期感知池 | 67.0 (-9.5) |
| w/o 结果记忆 | 无效（重复循环） |
| w/o 工作记忆 | 75.5 (-1.0) |
| 仅用 Controller 上下文 | 74.5 (-2.0) |
| Controller: GPT-4o | 40.5 |
| Controller: Qwen3-VL | 54.9 |

### 关键发现
- VideoARM 在 Video-MME Long 上以 81.2% 大幅超越 DVD 的 67.3%（+13.9pp），同时 token 消耗仅为 1/34
- 长期感知池是最关键组件（去掉后下降 9.5%），说明时序聚焦大幅减少了搜索空间
- Controller 的推理能力至关重要——GPT-4o 作为 Controller 仅 40.5%，说明复杂多步推理需要强推理模型（o3/GPT-5）
- 自适应帧采样策略优于固定采样（76.5 vs 74.0），平均只用 49.8 帧
- 步数预算 $N=10$ 在长视频上最佳，短视频不需要太多步

## 亮点与洞察
- **动态记忆 vs 静态数据库是核心创新**：DVD 先花大量 token 预建数据库再检索，VideoARM 按需构建记忆只处理 query 相关内容。这个思路类似于数据库中的 lazy evaluation vs eager evaluation
- **三层记忆设计有认知科学基础**：感知→工作→长期记忆的层级与人类认知模型一致，Working Memory 的外化设计巧妙地解决了 LLM 上下文长度限制
- **Controller 的"自由度"设计理念值得学习**：不预设工具调用顺序而让强推理模型自主决策，充分释放了 o3 的推理潜力

## 局限与展望
- 完全依赖 API 调用（o3 + GPT-4.1/4o + whisper-1），成本不可忽略，且受 API 可用性限制
- 10 步推理预算可能对超长视频（>1h）不够，但增加步数会增加 API 成本
- 帧采样和网格拼接策略可能丢失空间细节
- 未考虑开源模型部署的场景（Qwen3-VL 作 Controller 效果很差说明该方案对模型能力要求高）
- 无法处理实时流式视频

## 相关工作与启发
- **vs DVD**: VideoARM 的改进本质上是将"穷举预处理+检索"替换为"按需推理+记忆"。在 token 效率上 34x 提升，性能上 +13.9pp
- **vs VideoTree**: VideoTree 用固定分层聚类，VideoARM 用自适应工具调用，后者更灵活且不受预定义策略限制
- **vs VideoLucy**: VideoLucy 用固定文本摘要和回溯机制，VideoARM 维持层级化的多模态证据缓冲，信息粒度更丰富
- 思路可推广到长文档理解、多模态 RAG 等需要高效探索大规模信息的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ HM3 分层记忆和按需推理范式有较好创新性，但 observe-think-act 循环并非首创
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 5 个基准（Video-MME/LongVideoBench/EgoSchema/MLVU/LVBench），消融非常详细
- 写作质量: ⭐⭐⭐⭐ 结构清晰，工具设计描述详尽，但部分内容稍有冗余
- 价值: ⭐⭐⭐⭐ Token 效率的大幅提升有实际应用价值，但依赖高端 API 限制了可部署性

<!-- RELATED:START -->

## 相关论文

- [FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)
- [LensWalk: Agentic Video Understanding by Planning How You See in Videos](lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)
- [Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](../../AAAI2026/video_understanding/tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)

<!-- RELATED:END -->
