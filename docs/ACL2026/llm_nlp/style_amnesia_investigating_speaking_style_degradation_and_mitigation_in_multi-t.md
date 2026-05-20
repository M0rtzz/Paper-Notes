---
title: >-
  [论文解读] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models
description: >-
  [ACL 2026][LLM/NLP][口语语言模型] 发现口语语言模型（SLMs）在多轮对话中无法维持初始指定的说话风格（情感、口音、音量、语速），称之为"风格遗忘"现象，并通过注意力分析揭示其成因（注意力衰减），提出显式回忆过程作为缓解手段。
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "口语语言模型"
  - "风格遗忘"
  - "多轮对话"
  - "说话风格"
  - "指令遵循"
---

# Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models

**会议**: ACL 2026  
**arXiv**: [2512.23578](https://arxiv.org/abs/2512.23578)  
**代码**: [GitHub](https://github.com/YuXiangLin1234/SLM-Style-Amnesia)  
**领域**: 语音语言模型  
**关键词**: 口语语言模型, 风格遗忘, 多轮对话, 说话风格, 指令遵循

## 一句话总结

发现口语语言模型（SLMs）在多轮对话中无法维持初始指定的说话风格（情感、口音、音量、语速），称之为"风格遗忘"现象，并通过注意力分析揭示其成因（注意力衰减），提出显式回忆过程作为缓解手段。

## 研究背景与动机

**领域现状**：口语语言模型（如GPT-4o、Gemini Live、Qwen2.5-Omni等）已能在单轮交互中遵循用户指定的说话风格（情感、口音、语速等），展现出令人印象深刻的表达能力。

**现有痛点**：现有研究几乎全部聚焦于单轮评估，对多轮对话中风格一致性的维持能力一无所知。然而在实际应用中，用户在对话开始时设定风格后，期望SLM在整个会话过程中始终保持该风格，不可能每轮都重复指令。

**核心矛盾**：SLMs在第一轮能较好地遵循风格指令，但随着对话轮次增加，风格遵循率急剧下降——模型并非"忘记"了指令（回忆测试表明模型能准确复述指令），而是"无法执行"已记住的指令。

**本文目标**：系统性地评估和分析SLMs在多轮对话中的风格维持能力，找出成因并探索缓解方法。

**切入角度**：构建端到端评估框架，使用用户模拟器进行真实交互式多轮对话，逐轮测量风格遵循率。

**核心idea**：风格遗忘的根本原因是注意力稀释——随着对话轮次增加，模型对风格指令token的注意力权重从~8%衰减到<0.6%，而非真正的记忆丢失。

## 方法详解

### 整体框架

评估框架由三个核心组件构成：（1）风格指令——在对话开始时给定情感（悲伤/快乐/愤怒/中性）、口音（北美/印度英语）、音量（高/低）、语速（快/慢）共10种风格指令；（2）对话主题——从Soda数据集选取100个多样化对话开场白；（3）多轮交互——使用级联SLM（ASR + GPT-5 mini + TTS）作为用户模拟器，与被评估SLM进行4轮真实对话交互。

### 关键设计

1. **逐轮风格遵循率度量（Turn-Level IF Rate）**：

    - 功能：量化风格遵循在多轮对话中的变化趋势
    - 核心思路：定义首轮遵循率 $IF_1$ 和退化率 $D = \sum_{j=2}^{K} \frac{\max(IF_1(s) - IF_j(s), 0)}{K-1}$ 来分别捕获基线能力和退化程度。使用4种专用自动评判器分别评估情感（Emotion2vec-Large）、口音（Voxlect）、音量（LUFS）和语速（WPM）
    - 设计动机：与聚合全局分数的方法不同，逐轮分析能精确揭示退化从何时开始和如何发展

2. **注意力动态分析**：

    - 功能：揭示风格遗忘的内在机制
    - 核心思路：提取开源模型（Step-Audio 2 mini）在生成响应时对风格指令token的平均注意力权重。结果显示：第1轮~8.3%，第2轮~1.6%，第3轮~0.87%，第4轮~0.58%，严重的注意力稀释与IF率退化高度吻合
    - 设计动机：区分"忘记指令"和"无法执行"——如果是记忆问题可通过提示工程解决，如果是注意力稀释则需要架构改进

3. **回忆过程（Recall Process）**：

    - 功能：探索风格遗忘的缓解方法
    - 核心思路：在第2轮起的每一轮开始前，先提示SLM回忆初始风格指令，然后再处理用户输入。实验表明大部分模型能准确回忆（闭源模型近100%回忆率），且回忆过程能显著降低退化率（GPT-4o mini 悲伤风格从65.3%降至30.3%）
    - 设计动机：测试"模型是否还记得指令"以及"显式回忆能否改善执行"

### 文本-声学协同分析

对情感风格，语义和声学特征同时遭受风格遗忘——文本内容和声音表达同步退化。对语速风格，不同模型采取不同策略：Gemini Live通过减少字数实现"说得快"，GPT-4o通过声学加速而非内容压缩。但随轮次推进，快/慢条件的WPM差异持续缩小。

## 实验关键数据

### 主实验

| 模型 | 风格 | IF₁(首轮) | 退化率D |
|------|------|----------|---------|
| GPT-4o mini | 悲伤 | ~85% | 65.3% |
| GPT-4o mini | 印度口音 | ~75% | 49.7% |
| GPT-4o | 悲伤 | ~95% | 26.7% |
| Gemini Live | 悲伤 | ~85% | 21.3% |
| Step-Audio 2 mini | 悲伤 | ~70% | 14.0% |
| 级联基线(TTS) | 所有情感 | ~95% | <3.0% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 指令在系统消息 | IF₁下降30-80% | 系统消息反而更难遵循 |
| 指令在用户消息 | IF₁较高 | 默认设置效果更好 |
| +回忆过程 | D降低3-35% | GPT-4o mini 获益最大 |
| 注意力权重 | 8.3%→0.58% | 4轮内衰减14倍 |

### 关键发现
- 所有5个评估模型（3个闭源+2个开源）均出现风格遗忘，无一例外
- 模型"记得"指令但"做不到"——回忆率近100%但IF率持续下降
- 系统消息悖论：系统消息设计用于全局持久指令，但SLMs对系统消息中的风格指令遵循更差
- 默认风格偏差：模型倾向于回退到"快乐/中性"情感和"北美"口音等默认风格
- 级联基线（每轮给TTS提供风格指令）几乎不退化，证明问题出在端到端SLM的架构上

## 亮点与洞察
- **发现了一个重要且此前未被注意的问题**：风格遗忘是SLMs实用化的关键障碍
- **区分"记忆"和"执行"**：通过回忆测试精确定位问题不在记忆而在注意力分配，为解决方案指明方向
- **评估框架完善**：使用模拟器进行真实交互+4种专用评判器+人工验证，评估可靠性高
- **系统消息悖论的发现很有价值**：揭示了SLMs架构设计中的深层问题

## 局限与展望
- **风格种类有限**：仅覆盖4类副语言属性，未涉及语调变化、角色扮演等更复杂风格
- **未组合多种风格**：现有模型连单一风格都维持不了，多风格组合留待未来
- **开源模型注意力分析受限**：仅分析了Step-Audio 2 mini一个开源模型
- 未来方向：风格锚定注意力机制、风格嵌入的持久化表示、多风格组合遵循

## 相关工作与启发
- **vs Multi-Bench**：同样评估多轮SLM，但仅聚合全局分数；本文提供逐轮分析
- **vs VocalBench/VoxDialogue**：使用预定义对话而非真实交互，无法进行逐轮分析
- **vs 文本LLM多轮退化研究**：文本领域已发现类似的多轮性能退化，本文扩展到语音域的副语言特征

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性揭示SLM风格遗忘现象，发现"记得但做不到"的关键洞察
- 实验充分度: ⭐⭐⭐⭐ 覆盖5个模型、10种风格、1000组对话，有注意力分析和缓解实验
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，实验层层递进，图表直观
- 价值: ⭐⭐⭐⭐⭐ 指出SLMs实用化的关键障碍，对模型设计和训练有明确指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style](../../ACL2025/llm_nlp/investigating_context-faithfulness_in_large_language_models_the_roles_of_memory_.md)
- [\[ACL 2026\] MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)
- [\[ICLR 2026\] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](../../ICLR2026/llm_nlp/unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/llm_nlp/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](../../ICLR2026/llm_nlp/toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)

</div>

<!-- RELATED:END -->
