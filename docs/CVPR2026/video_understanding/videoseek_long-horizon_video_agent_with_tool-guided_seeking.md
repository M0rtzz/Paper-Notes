---
title: >-
  [论文解读] VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking
description: >-
  [CVPR 2026][视频理解][视频Agent] VideoSeek 提出一种长程视频 Agent，利用视频逻辑流主动"寻找"关键证据而非穷举解析所有帧，通过 think-act-observe 循环和多粒度工具包（overview/skim/focus），在 LVBench 上比基座模型 GPT-5 提升 10.2 个点的同时减少 93% 的帧使用量。
tags:
  - CVPR 2026
  - 视频理解
  - 视频Agent
  - 长视频理解
  - 工具调用
  - 逻辑流
  - think-act-observe
---

# VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking

**会议**: CVPR 2026  
**arXiv**: [2603.20185](https://arxiv.org/abs/2603.20185)  
**代码**: https://github.com/jylins/videoseek  
**领域**: 视频理解 / Agent  
**关键词**: 视频Agent, 长视频理解, 工具调用, 逻辑流, think-act-observe

## 一句话总结
VideoSeek 提出一种长程视频 Agent，利用视频逻辑流主动"寻找"关键证据而非穷举解析所有帧，通过 think-act-observe 循环和多粒度工具包（overview/skim/focus），在 LVBench 上比基座模型 GPT-5 提升 10.2 个点的同时减少 93% 的帧使用量。

## 研究背景与动机
1. **领域现状**：视频语言理解近年因 LMM 进步而迅速发展。但主流方法（包括 Qwen2.5-VL、GPT-4o 等）仍采用单次推理范式——输入固定帧数后直接生成答案，在长视频和复杂推理场景下力不从心。视频 Agent 方法（如 DrVideo、DVD）虽引入迭代推理，但严重依赖密集视频预处理。
2. **现有痛点**：现有视频 Agent 以 0.2-2 FPS 密集采样并逐帧生成详细文本描述或结构化记忆，计算开销随视频长度线性增长。例如 DVD 在 LVBench 上需处理 8074 帧，MR.Video 也用了 8074 帧。然而在 LVBench 上超过 80% 的问题只需查看不到 5% 的视频即可回答——穷举解析极其浪费。
3. **核心矛盾**：详尽的视频预处理虽能提升准确率但代价极高且不可扩展。如何在极稀疏的视觉预算下实现甚至超越密集解析的效果？
4. **本文目标** 设计一种高效的视频 Agent，通过主动寻找关键证据而非暴力穷举来回答视频问题。
5. **切入角度**：人类不会逐帧观看视频来回答问题——他们利用时间和因果结构推断有用证据可能出现的位置，快速建立粗略故事线，检查有望的时间段，只在需要细节时才仔细观看。
6. **核心 idea**：用 ReAct 式 think-act-observe 循环和三层粒度工具（全局概览→粗略浏览→精细聚焦），基于视频逻辑流主动导航到答案关键帧。

## 方法详解

### 整体框架
VideoSeek 将视频语言任务建模为长程问题而非单步推理。给定查询 $\mathbf{Q}$ 和视频 $\mathbf{X}$，Agent 在每个推理步 $t$ 产生 think-act-observe 三元组 $\langle z_t, a_t, o_t \rangle$，形成轨迹 $\tau$。最终答案由 $p(\mathbf{Y}|\mathbf{X}, \mathbf{Q}, \tau)$ 生成。核心思路是让轨迹探索过程 $p(\tau|\mathbf{X}, \mathbf{Q})$ 聚焦于少量高信息量的观测，而非穷举覆盖。Agent 使用 GPT-5 作为默认思考 LLM，最大推理轮次 $N=20$。

### 关键设计

1. **多粒度工具包（Toolkit）**:
    - 功能：模拟人类观看视频的"先总览、再浏览、后聚焦"行为
    - 核心思路：三个工具各操作在不同时间粒度上——
        - `<overview>`：从全视频均匀采样 $16\alpha$ 帧（LVBench $\alpha=4$ 即64帧），生成粗略故事线和关键区间标记。主要在开头使用，建立全局认知地图
        - `<skim>`：对候选长片段均匀采样 $4\alpha$ 帧（最短 $4\alpha$ 秒），快速定位查询相关时刻。可多次调用在不同片段上渐进缩小搜索空间
        - `<focus>`：以 1 FPS 高帧率密检短片段（最长 $4\alpha$ 秒），获取精细细节如文字阅读、人脸识别、物体计数等。作为最终"特写"步骤确保答案准确性
    - 设计动机：不同层级的信息需要不同粒度的观察。全局概览发现故事结构和逻辑流，浏览定位相关区间，聚焦获取决定性证据。三者互补，缺一不可

2. **Think-Act-Observe 循环**:
    - 功能：实现自适应的长程推理和证据收集
    - 核心思路：采用 ReAct 架构——每步中 thinking LLM 阅读完整轨迹（包括所有历史思考、行动和观测），输出推理过程 $z_t$ 和工具调用计划 $\alpha_t$。如果 $\alpha_t$ 是 `<answer>`，解析答案并停止；否则执行工具获取新观测 $o_t$，追加到轨迹中继续下一轮。关键创新在于Agent基于不断累积的观测动态调整工具调用策略，而非按预定义的"粗到细"规则执行
    - 设计动机：相比预建视频数据库（DrVideo/DVD）或维护固定记忆缓冲区（VCA），直接在完整对话历史上推理更灵活——Agent 可以回溯之前的观测、修正判断、改变搜索方向

3. **视频逻辑流利用**:
    - 功能：使 Agent 能通过视频的时间-因果结构推断证据位置
    - 核心思路：视频具有内在的逻辑流（场景因果、时间顺序、角色关系等）。当有字幕时，逻辑流通过文本故事线直接暴露，Agent 可以更快定位关键片段。没有字幕时，`<overview>` 通过视觉摘要构建粗略逻辑流。Agent 根据逻辑流推断答案可能出现的位置，而非盲目搜索
    - 设计动机：LVBench 实验显示加入字幕后帧使用量从 92.3 降到 27.2 而准确率从 68.4 升到 76.7，充分证明逻辑流的暴露使导航更精准高效

### 训练策略
VideoSeek 是一个无需训练的 Agent 框架（model-agnostic），直接利用 GPT-5 的推理和工具使用能力。工具返回的观测由 GPT-5 解读视觉内容。

## 实验关键数据

### 主实验

| 方法 | 类型 | LVBench (无字幕) | 帧数 | LVBench (有字幕) | 帧数 |
|------|------|-----------------|------|-----------------|------|
| GPT-5 (Base) | LMM | 60.1 | 384 | 66.5 | 384 |
| Gemini 1.5 Pro | LMM | 33.1 | 3600 | - | - |
| DVD | Agent | 74.2 | 8074 | 76.0 | 8074 |
| MR. Video | Agent | 60.8 | 8074 | - | - |
| **VideoSeek** | **Agent** | **68.4** | **92.3** | **76.7** | **27.2** |

VideoMME Long (有字幕): VideoSeek 81.2% / 15.9帧 vs GPT-5 78.1% / 384帧
LongVideoBench Long: VideoSeek 73.5% / 29.6帧 vs GPT-5 64.5% / 384帧

### 消融实验

| 配置 | LVBench (无字幕) | 说明 |
|------|-----------------|------|
| Full toolkit | **68.4** | 完整模型 |
| w/o overview | 55.1 (-13.3) | 失去全局视角 |
| w/o skim | 62.4 (-6.0) | 失去中间粒度浏览 |
| w/o focus | 63.7 (-4.7) | 失去精细检查 |

| Thinking LLM | 帧数 | 轮次 | LVBench |
|--------------|------|------|---------|
| GPT-5 | 92.3 | 4.42 | **68.4** |
| o4-mini | 112.6 | 5.08 | 58.5 (-9.9) |
| GPT-4.1 | 74.2 | 2.99 | 53.0 (-15.4) |

### 关键发现
- **overview 是最关键的工具**（去掉后降 13.3 个点），因为它提供全局故事线和逻辑流——这是后续所有导航的基础
- **推理能力决定上限**：GPT-4.1（非 thinking model）倾向于过早自信地停止推理（仅 2.99 轮），导致证据不足；o4-mini 虽然多探索但推理质量差，额外计算不能转化为更好性能
- **字幕 = 显式逻辑流**：加入字幕后帧使用量降 70%（92.3→27.2）但准确率升 8.3 个点，证明文本故事线极大地简化了证据搜索
- **vs DVD**：VideoSeek 有字幕时超过 DVD（76.7 vs 76.0），仅使用 0.3% 的帧数（27.2 vs 8074）
- 在 Video-Holmes 复杂推理基准上，VideoSeek 47.3% 超过 Gemini 2.5 Pro 的 45.0%，帧数仅为其 1/4

## 亮点与洞察
- **"主动寻找"vs"穷举解析"**的范式转变：这是本文最深刻的贡献。VideoSeek 证明了在长视频理解中，聪明的导航比暴力的密集采样有效得多——用 1% 的帧就能达到甚至超过密集方法。这符合人类认知的经济性原则
- **工具包设计的层次美感**：overview/skim/focus 三层粒度恰好对应人类的"扫一眼→翻阅→细看"行为，简单直觉却效果显著。特别是 overview 的全局概览价值远超预期（贡献 13.3 个点）
- **thinking model 是核心引擎**：非 thinking 模型无法有效使用这个框架——需要真正的推理能力来判断"证据是否充分"、"下一步应该看哪里"。这暗示了 Agent 系统对底层推理模型的高依赖

## 局限与展望
- 完全依赖闭源模型 GPT-5，无法开源复现和在成本敏感场景中部署
- 对突发或高度局部化的关键时刻（如异常检测）效果可能较差——逻辑流驱动的导航难以预见意外事件
- 每次工具调用都需要 LMM 解读视觉内容，API 调用成本可能很高
- 未探索如何将这种 Agent 框架蒸馏到更小的开源模型中
- 工具的超参数（$\alpha$、最大帧数等）需要针对不同基准调整

## 相关工作与启发
- **vs DVD Agent**: DVD 构建多粒度视频数据库再检索，需要 8074 帧预处理。VideoSeek 按需探索，帧数降两个数量级但性能相当（无字幕）或更好（有字幕）
- **vs DrVideo**: DrVideo 以 0.2 FPS 将视频转为长文档，属于穷举范式。VideoSeek 证明穷举不必要
- **vs 单次推理 LMM**: GPT-5 单次 384 帧得 60.1%，VideoSeek 同一模型通过 Agent 框架提升到 68.4%/76.7%，说明 Agent 范式能释放基座模型的潜力

## 评分
- 新颖性: ⭐⭐⭐⭐ "主动寻找而非穷举"的理念有价值，但 ReAct + 工具调用的框架并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 四个基准、有无字幕对比、Thinking LLM 消融、工具消融，分析非常深入
- 写作质量: ⭐⭐⭐⭐⭐ 叙事流畅，Figure 1 的效率-性能对比一目了然，case study 清晰
- 价值: ⭐⭐⭐⭐ 对视频 Agent 效率优化有重要启示，但闭源依赖限制了社区影响力

<!-- RELATED:START -->

## 相关论文

- [A4VL: A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a4vl_multiagent_long_video_reasoning.md)
- [A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)
- [Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](virtuebench_evaluating_trustworthiness_under_uncertainty_in_long_video_understan.md)
- [VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)

<!-- RELATED:END -->
