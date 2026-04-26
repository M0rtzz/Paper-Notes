---
title: >-
  [论文解读] CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration
description: >-
  [ACL 2025][模型压缩][动机式访谈] 本文提出CAMI（Counselor Agent for Motivational Interviewing），一个基于动机式访谈（MI）原则的咨询Agent，通过STAR框架（状态推断、话题探索、回复生成）来引导来访者产生改变谈话（change talk），在自动化和人工评估中均优于现有方法。
tags:
  - ACL 2025
  - 模型压缩
  - 动机式访谈
  - 咨询Agent
  - 状态推断
  - 话题探索
  - 行为改变
---

# CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration

**会议**: ACL 2025  
**arXiv**: [2502.02807](https://arxiv.org/abs/2502.02807)  
**代码**: 无  
**领域**: LLM Agent / 对话系统 / 心理咨询  
**关键词**: 动机式访谈, 咨询Agent, 状态推断, 话题探索, 行为改变

## 一句话总结

本文提出CAMI（Counselor Agent for Motivational Interviewing），一个基于动机式访谈（MI）原则的咨询Agent，通过STAR框架（状态推断、话题探索、回复生成）来引导来访者产生改变谈话（change talk），在自动化和人工评估中均优于现有方法。

## 研究背景与动机

**领域现状**：随着心理健康服务需求的急剧增长，自动化对话咨询Agent成为扩展心理健康支持可及性的重要手段。现有的心理咨询对话系统大多基于简单的对话策略，缺乏系统性的咨询理论支撑。

**现有痛点**：(1) 现有LLM驱动的咨询Agent缺乏对来访者心理状态的动态追踪能力，往往只对当前话语做反应而非基于整体状态规划；(2) 大多数方法未遵循成熟的咨询理论框架，导致对话缺乏方向性和治疗效果；(3) 在动机式访谈中，引导来访者从"矛盾心理"过渡到"改变谈话"需要精细的策略，简单的LLM生成无法胜任。

**核心矛盾**：动机式访谈要求咨询师同时完成三个认知任务——理解来访者当前的改变准备阶段、选择合适的话题进行探索、生成符合MI原则的回复。将这三个任务混为一个端到端的生成问题会导致每个子任务都做不好。

**本文目标**：设计一个遵循MI原则的结构化咨询Agent，将咨询过程解耦为状态推断、话题探索和回复生成三个显式模块。

**切入角度**：MI理论中的"改变阶段模型"（stages of change）和"引发改变谈话"（evoking change talk）为Agent的决策提供了明确的理论指引。

**核心 idea**：通过显式的状态推断模块追踪来访者的改变准备度，利用话题探索模块寻找能引发改变谈话的话题，使Agent的咨询行为具有理论指导的方向性。

## 方法详解

### 整体框架

CAMI的STAR框架由三个模块组成：**S**tate inference（状态推断）→ **T**opic exploration（话题探索）→ **A**ction & **R**esponse generation（动作与回复生成）。输入是多轮对话历史，输出是符合MI原则的咨询师回复。

### 关键设计

1. **来访者状态推断模块（State Inference）**:

    - 功能：基于对话历史推断来访者当前的改变准备阶段
    - 核心思路：根据MI理论的跨理论模型（Transtheoretical Model），来访者的改变过程分为前沉思期、沉思期、准备期、行动期和维持期。该模块使用LLM分析对话中的语言线索（如抗拒性语言 vs 改变性语言），推断来访者当前处于哪个阶段。状态推断的结果直接影响后续的话题选择策略——对处于前沉思期的来访者应更多使用开放性提问，对处于准备期的来访者可以更直接地讨论行动计划
    - 设计动机：没有状态推断的咨询Agent无法根据来访者的准备度调整策略，容易出现"过早干预"或"过于保守"的问题

2. **话题探索模块（Topic Exploration）**:

    - 功能：根据来访者的当前状态选择最有可能引发改变谈话的话题
    - 核心思路：维护一个话题候选集，包括来访者提到的各种主题（如健康、家庭、工作等）。基于当前推断的状态和对话历史，使用LLM评估每个候选话题的"改变潜力"——即在该话题上继续探索是否有助于引发来访者的改变谈话。选择得分最高的话题进行深入探索。这实现了MI中的"选择性强化"策略
    - 设计动机：MI的核心是"引发"而非"指导"——咨询师通过选择正确的话题让来访者自己发现改变的需要，而非直接告诉来访者该怎么做

3. **回复生成模块（Action & Response Generation）**:

    - 功能：基于推断的状态和选定的话题生成MI兼容的咨询回复
    - 核心思路：将状态推断结果和话题选择作为条件信息，结合MI的回复策略库（如开放性提问、反映式倾听、肯定、总结等），使用LLM生成最终的咨询师回复。回复的MI策略类型也被显式规划——对抗拒性来访者更多使用反映式倾听，对改变性来访者更多使用肯定和引导
    - 设计动机：给LLM生成回复时提供结构化的条件信息（状态+话题+策略类型），比直接生成能产生更符合MI原则的回复

### 训练策略

使用模拟来访者（simulated clients）进行大规模对话生成和评估。模拟来访者由LLM扮演，被赋予特定的问题场景和抗拒程度，与CAMI进行多轮对话。

## 实验关键数据

### 主实验

| 评估维度 | CAMI | ChatGPT | MI-specific baseline | 人类咨询师 |
|----------|------|---------|---------------------|-----------|
| MI兼容性得分 | 最优 | 较低 | 中等 | 参考标准 |
| 改变谈话引发率 | 显著优于基线 | 低 | 中等 | - |
| 回复多样性 | 高 | 高 | 低 | 高 |
| 平均对话轮次到改变谈话 | 较少 | 较多 | 中等 | - |
| 状态推断准确率 | ~75% | - | - | - |

### 消融实验

| 配置 | MI兼容性 | 改变谈话引发率 | 说明 |
|------|---------|-------------|------|
| Full CAMI (STAR) | 最优 | 最优 | 完整模型 |
| w/o State Inference | 明显下降 | 大幅下降 | 无法根据来访者状态调整策略 |
| w/o Topic Exploration | 中等下降 | 明显下降 | 话题选择变为随机 |
| w/o 策略规划 | 轻微下降 | 轻微下降 | 回复风格受影响 |
| 直接端到端生成 | 最差 | 最差 | 无结构化指引 |

### 关键发现
- 状态推断模块是CAMI最关键的组件——去掉后MI兼容性和改变谈话引发率均大幅下降
- 话题探索模块的贡献在长对话中更明显——短对话（<5轮）影响不大，但10+轮对话中选对话题至关重要
- CAMI生成的回复在MI技能专家评估中得到较高分数，特别是在反映式倾听和开放性提问方面
- 模拟来访者评估与少量真人评估结果一致性较好，验证了模拟评估的可靠性

## 亮点与洞察
- **将心理咨询理论（MI的跨理论模型）显式融入Agent架构**是本文最大的亮点——不是把LLM当作黑箱来用，而是用心理学理论指导模块设计。这种"理论驱动的Agent设计"思路可迁移到其他需要专业知识的对话场景
- **状态推断→话题选择→回复生成的级联式决策**比端到端生成更有可解释性，便于人类咨询师审核和干预
- 使用模拟来访者进行大规模评估是一个务实的解决方案——真人咨询评估成本极高且涉及伦理问题

## 局限与展望
- 模拟来访者虽然与少量真人评估一致，但无法完全替代真实临床场景
- 状态推断的准确率约75%，误判可能导致不当的咨询策略
- 仅关注MI一种咨询流派，未探索认知行为疗法（CBT）等其他主流方法
- 安全性考量不足——在面对有自伤风险的来访者时，Agent应如何安全地转介到人类咨询师

## 相关工作与启发
- **vs SMILE**: SMILE是基于CBT的心理咨询对话系统，使用预定义的对话策略；CAMI的状态推断机制更灵活
- **vs AugESC**: AugESC通过增强数据来训练情感支持对话模型，但缺乏显式的咨询策略规划
- **vs ChatCounselor**: ChatCounselor主要做情感识别和共情回复，不涉及行为改变引导

## 评分
- 新颖性: ⭐⭐⭐⭐ STAR框架将MI理论与LLM Agent优雅结合，状态推断模块设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 多维度评估（MI兼容性、改变谈话率、消融），但缺少大规模真人评估
- 写作质量: ⭐⭐⭐⭐ 将心理咨询理论解释清楚的同时保持了技术深度
- 价值: ⭐⭐⭐⭐ 为心理健康AI领域提供了理论驱动的方法论

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ECCV 2024\] ELSE: Efficient Deep Neural Network Inference through Line-based Sparsity Exploration](../../ECCV2024/model_compression/else_efficient_deep_neural_network_inference_through_line-based_sparsity_explora.md)

<!-- RELATED:END -->
