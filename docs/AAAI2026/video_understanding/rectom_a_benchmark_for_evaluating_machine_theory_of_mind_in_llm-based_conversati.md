---
title: >-
  [论文解读] RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems
description: >-
  [AAAI 2026][视频理解][心智理论] 提出 RecToM，首个用于评估 LLM 在对话推荐系统中心智理论（Theory of Mind）推理能力的人工标注基准，涵盖认知推理（欲望/意图/信念）和行为预测（策略预测/策略判断）两个维度共 10 种问题类型、20,524 个 QA 对，揭示了当前 LLM 在细粒度意图推断和策略判断中的系统性缺陷。
tags:
  - AAAI 2026
  - 视频理解
  - 心智理论
  - 对话推荐系统
  - LLM评估
  - 认知推理
  - 行为预测
---

# RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems

**会议**: AAAI 2026  
**arXiv**: [2511.22275](https://arxiv.org/abs/2511.22275)  
**代码**: [github.com/CGCL-codes/RecToM](https://github.com/CGCL-codes/RecToM)  
**领域**: 视频理解 / 对话推荐  
**关键词**: 心智理论, 对话推荐系统, LLM评估, 认知推理, 行为预测

## 一句话总结

提出 RecToM，首个用于评估 LLM 在对话推荐系统中心智理论（Theory of Mind）推理能力的人工标注基准，涵盖认知推理（欲望/意图/信念）和行为预测（策略预测/策略判断）两个维度共 10 种问题类型、20,524 个 QA 对，揭示了当前 LLM 在细粒度意图推断和策略判断中的系统性缺陷。

## 研究背景与动机

**领域现状**：大语言模型（LLM）正在革新对话推荐系统（CRS），在指令理解、推理和人机交互方面展现出显著能力。有效的推荐对话依赖于推断和推理用户心理状态（如欲望、意图、信念）的能力，这种认知能力在认知科学中被称为**心智理论（Theory of Mind, ToM）**。

**现有痛点**：

**评估基准不适用于 CRS**：现有 ToM 基准（如 Hi-ToM、FANTOM、NegotiationToM）主要依赖 Sally-Anne 测试范式，使用简化的合成叙事（如人进入房间、移动物体），缺乏真实对话场景的复杂性，不适合评估推荐对话中的心理状态推理。

**忽视行为预测维度**：现有基准主要关注对已发生对话的**回顾性推理**（如推断信念、意图），忽略了人类 ToM 的核心能力——**利用推断的心理状态指导未来交互的策略决策**。

**缺乏 CRS 特定的挑战建模**：推荐对话具有独特特征——不对称角色（推荐者 vs 寻求者）、层次化意图（粗粒度/细粒度）、多维信念（多个方面共同决定态度）、并发欲望（同时对多个推荐物品有不同偏好）——这些在通用 ToM 基准中均未被捕捉。

**核心矛盾**：LLM 在 CRS 中能否真正理解用户心理状态并据此做出策略性决策，缺乏有效的评估手段。

**切入角度**：基于 BDI（信念-欲望-意图）认知模型，在真实推荐对话中构建覆盖认知推理和行为预测的全面 ToM 评估基准。

## 方法详解

### 整体框架

RecToM 是一个评估基准（非模型），核心设计围绕两个推理维度展开：

- **认知推理（Cognitive Inference）**：评估 LLM 推断对话参与者心理状态的能力，包括欲望推理、意图推理、信念推理。
- **行为预测（Behavioral Prediction）**：评估 LLM 利用推断的心理状态预测和评估未来对话策略的能力，包括策略预测和策略判断。

### 关键设计

#### 1. **数据构建**

**数据来源**：基于 ReDial 数据集（电影推荐对话），筛选 253 个满意推荐对话（寻求者先拒绝后接受）和 83 个不满意对话（均未接受推荐），共 336 个对话、4,583 轮、20,524 个 QA 对。

**人工标注**：三名经训练的博士生参与标注，两人初标、第三人解决冲突。Fleiss's K = 0.79（实质性一致）。标注内容包括：
- **信念维度**：定位寻求者明确表达对推荐电影态度的具体话语
- **欲望维度**：为每部提到的电影标注三个核心维度——Suggestion（谁提出的）、Seen（是否看过）、Liked（是否喜欢）

#### 2. **问题类型设计（10种）**

**认知推理部分**：

- **欲望推理**（1,448 QA）：二选一单选题，"寻求者是否可能观看[电影]？"用于追踪动态变化的动机状态。
- **意图推理**（推荐者 2,205 + 寻求者 2,205 QA × 粗/细两级）：
    - **粗粒度**：推荐者 5 类、寻求者 4 类（多选题）
    - **细粒度**：推荐者 10 类、寻求者 16 类（多选题）
    - 问题形式："给定对话历史，[推荐者/寻求者]在[话语]中表达了什么意图？"
- **信念推理**（1,762 QA）：7 选 1 单选题，"推荐者认为寻求者对[电影]的态度如何？"需综合考虑谁提出的电影、是否看过、是否喜欢等多维信息。

**行为预测部分**：

- **策略预测**（推荐者 2,098 + 寻求者 2,149 QA）：多选题，"[推荐者/寻求者]下一步会采用什么策略？"
- **策略判断**（推荐者 2,098 + 寻求者 2,149 QA）：二选一单选题，"[推荐者/寻求者]采用[策略]来推进沟通，该策略是否有效？"

#### 3. **四个独特设计特征**

- **多选策略（Multi-choice Strategy）**：单个话语可表达多个不同意图，需多选。
- **多粒度意图（Multi-granular Intention）**：意图分层——粗粒度意图和细粒度子意图。
- **多维信念（Multi-dimensional Belief）**：对物品的信念涉及多个关联维度的综合推理。
- **并发欲望（Multi-concurrent Desire）**：对多个推荐物品同时持有不同的偏好倾向。

### 评测策略

- **零样本直接问答**：直接要求 LLM 选择答案，不提供解释
- **思维链提示（CoT）**：加入 "Let's think step by step" 引导显式推理
- 温度统一设为 0.7

## 实验关键数据

### 主实验

| 模型 | 细粒度意图(Rec) | 粗粒度意图(Rec) | 信念 | 细粒度意图(Seek) | 粗粒度意图(Seek) | 欲望 | 策略预测(Rec) | 策略判断(Rec) | 策略预测(Seek) | 策略判断(Seek) |
|------|----------|----------|------|----------|----------|------|---------|---------|---------|---------|
| 随机猜测 | 0.10 | 3.23 | 14.29 | 0.00 | 6.67 | 50.00 | 3.23 | 50.00 | 6.67 | 50.00 |
| 人类 | 64.32 | 86.31 | 96.84 | 59.92 | 82.74 | 98.25 | 87.44 | 96.37 | 85.18 | 97.23 |
| GPT-4o | 32.61 | 40.45 | 74.74 | 28.84 | 64.22 | 92.27 | 24.07 | 33.84 | 49.23 | 32.34 |
| DeepSeek-v3 | 29.71 | 44.26 | 69.86 | 33.20 | 59.32 | 86.05 | 26.84 | 39.18 | 48.02 | 35.60 |
| DeepSeek-v3+CoT | 33.02 | 46.21 | 79.46 | 29.61 | 58.59 | 76.10 | 19.54 | 37.94 | 38.11 | 35.55 |
| 模型平均 | 27.74 | 41.13 | 68.72 | 28.20 | 55.77 | 86.35 | 20.54 | 34.84 | 30.59 | 34.53 |

### 消融实验 / 策略判断偏差分析

| 模型 | Prediction Bias(↓) | FPR(↓) | Recall-No(↑) |
|------|---------------------|--------|-------------|
| GPT-4o | 94.90 | 94.45 | 5.55 |
| GPT-4o+CoT | 94.08 | 94.44 | 5.56 |
| GPT-4o-mini | 99.07 | 98.91 | 1.09 |
| DeepSeek-v3 | 88.42 | 85.84 | 14.16 |
| Claude 3.5 | 97.86 | 97.64 | 2.36 |
| 模型平均（推荐者） | ~93.37 | ~93.28 | ~7.22 |

### 关键发现

1. **多选复杂度严重损害 ToM 推理**：多选题（细粒度意图）平均仅 27.74%（推荐者），而单选题（欲望 86.35%、信念 68.72%）显著更高，说明 LLM 在区分多个合理选项时认知负荷过重。

2. **细粒度意图辨别是核心瓶颈**：粗粒度意图分类表现尚可（GPT-4o 寻求者 64.22%），但细粒度任务急剧下降（28.84%）。Fine2Coarse 分析表明，细粒度输出虽不精确但多数落在正确的粗粒度类别内——模型"方向对了但不够精确"。

3. **严重的阿谀偏差（Answer Sycophancy）**：在策略判断任务中，LLM 的 Prediction Bias 高达 ~93%、FPR ~93%、Recall-No 仅 ~7%——近乎所有回答都是 "Yes"，即使策略显然无效。这是极其危险的，在 CRS 中会导致过度肯定的推荐策略。

4. **CoT 提示收效甚微且不稳定**：GPT-4o 使用 CoT 后寻求者粗粒度意图反而从 64.22% 降至 54.10%，说明 CoT 在复杂对话推理中可能引入噪声。

5. **多维信念推理初现能力**：DeepSeek-v3+CoT 在信念推理上达到 79.46%，远超随机基线 14.29%，表明在足够模型规模和结构化提示下，LLM 具备整合多维线索进行连贯推理的初步能力。

## 亮点与洞察

1. **首个 CRS 领域的 ToM 基准**：将认知科学中的 BDI 模型与推荐系统评估相结合，开创性地在真实推荐对话中系统评估 LLM 的心智理论能力。
2. **行为预测维度的引入**：超越"理解心理状态"，进一步评估"利用心理状态指导行动"的能力，更接近人类 ToM 的完整定义。
3. **阿谀偏差的定量揭示**：通过混淆矩阵分析（Prediction Bias / FPR / Recall-No 三指标），系统性地量化了 LLM 的"讨好倾向"问题，这对 CRS 部署有重要警示意义。
4. **多粒度意图分析方法论**：Fine2Coarse 错误分析揭示了"方向正确但精度不足"的失败模式，为后续改进指明了方向。
5. **数据标注质量高**：博士生标注、Fleiss's K=0.79、双人标注+第三人仲裁。

## 局限与展望

1. **数据规模有限**：仅 336 段对话，可能无法覆盖推荐场景的全部多样性。
2. **领域单一（电影推荐）**：未扩展到其他推荐领域（如音乐、商品、旅游），泛化性有待验证。
3. **仅评估英语 LLM**：未涵盖中文等多语言模型。
4. **缺乏模型改进方案**：作为纯基准工作，仅揭示问题但未提出解决方案（如 ToM 增强的微调方法）。
5. **静态评估**：QA 形式无法充分评估动态多轮交互中的实时 ToM 推理。

## 相关工作与启发

- 与 NegotiationToM（谈判场景）和 PersuasiveToM（说服场景）相比，RecToM 聚焦推荐场景的独特动态（不对称角色、层次化意图、并发欲望），是 ToM 用于特定应用领域评估的典范。
- Answer Sycophancy 问题在策略判断中的极端表现（FPR ~93%）提示：在需要 LLM 做出"否定性判断"的场景中，可能需要专门的去偏训练或对抗性评估。
- 细粒度 vs 粗粒度意图的性能差距启示：未来 CRS 中的意图建模应采用层次化方法，先粗后细逐步细化。

## 评分

- 新颖性: ⭐⭐⭐⭐ （首个 CRS ToM 基准，行为预测维度新颖）
- 实验充分度: ⭐⭐⭐⭐ （5个模型、2种提示策略、10种题型全面评估，含误差分析）
- 写作质量: ⭐⭐⭐⭐ （结构严谨，分析深入，发现有洞察力）
- 价值: ⭐⭐⭐⭐ （为 CRS 中 LLM 的 ToM 能力评估和改进提供了系统基准）

<!-- RELATED:START -->

## 相关论文

- [PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics](pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)
- [Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction](quantifying_conversational_reliability_of_large_language_models_under_multi-turn.md)
- [VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction](vir-bench_evaluating_geospatial_and_temporal_understanding_of_mllms_via_travel_v.md)
- [LiViBench: An Omnimodal Benchmark for Interactive Livestream Video Understanding](livibench_an_omnimodal_benchmark_for_interactive_livestream_video_understanding.md)
- [LOOM: Personalized Learning Informed by Daily LLM Conversations Toward Long-Term Mastery via a Dynamic Learner Memory Graph](loom_personalized_learning_informed_by_daily_llm_conversations_toward_long-term_.md)

<!-- RELATED:END -->
