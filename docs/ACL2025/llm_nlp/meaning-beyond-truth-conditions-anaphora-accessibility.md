---
title: >-
  [论文解读] Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility
description: >-
  [ACL 2025][LLM 其他][篇章语义] 本文提出自然语言理解能力的三层次层级体系（词汇/句子/篇章），以回指可及性（anaphora accessibility）作为篇章级理解的诊断任务，通过动态语义学启发的评估数据集系统考察了 LLM 在全称量词、否定和析取三种语言结构下的篇章理解能力。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "篇章语义"
  - "回指可及性"
  - "动态语义学"
  - "量化词作用域"
  - "LLM评估"
---

# Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility

**会议**: ACL 2025  
**arXiv**: [2502.14119](https://arxiv.org/abs/2502.14119)  
**代码**: 无  
**领域**: NLP理解  
**关键词**: 篇章语义, 回指可及性, 动态语义学, 量化词作用域, LLM评估

## 一句话总结

本文提出自然语言理解能力的三层次层级体系（词汇/句子/篇章），以回指可及性（anaphora accessibility）作为篇章级理解的诊断任务，通过动态语义学启发的评估数据集系统考察了 LLM 在全称量词、否定和析取三种语言结构下的篇章理解能力。

## 研究背景与动机

1. **领域现状**：现有 NLU 评估主要聚焦词汇级（词义消歧、类比推理）和句子级（NLI、蕴含判断），对篇章级语义理解的评估远不充分。
2. **现有痛点**：少数篇章级评估工作要么针对简单场景（如实体追踪），要么仅考虑否定的作用域，缺乏对多种逻辑连接词（全称量词、条件句、析取等）与篇章实体交互的系统考察。
3. **核心矛盾**：篇章理解不仅需要句子级的真值条件语义，还需要动态更新话语状态的能力——但我们不知道 LLM 是否真正具备这种基于结构的状态更新能力。
4. **本文目标**：系统评估 LLM 是否理解各种语义算子的作用域如何影响篇章实体的可引用性。
5. **切入角度**：利用形式语义学（动态语义学）中关于回指可及性的精细预测，设计最小对立的评估刺激。
6. **核心 idea**：如果 LLM 具备篇章级理解，它应该对合适的续句分配更高概率——即在存在量词后可以回指，在全称量词后不可回指。

## 方法详解

### 整体框架

设计三组实验分别测试全称量词（every/if/whenever）、否定（单否定/双否定）和析取（either...or/and）对篇章实体引用可及性的影响。使用 surprisal（负对数概率）度量 LLM 对续句的接受程度，与人类被试的强制选择实验进行比较。

### 关键设计

1. **全称量词实验**:

    - 功能：测试 LLM 是否区分存在量词和全称量词的篇章实体可及性
    - 核心思路：对比 "A farmer worked..." 后的代词回指（应可及）vs "Every farmer worked..." 后的代词回指（应不可及）。使用差之差度量（difference-of-difference）：比较跨句 vs 句内的概率差在存在/全称条件下是否不同，以排除简单概率偏见。
    - 设计动机：仅比较绝对概率过于宽松——即使 LLM 更偏好存在条件，也可能是因为词汇偏好而非真正的篇章理解。

2. **否定/双否定实验**:

    - 功能：测试 LLM 是否理解否定阻止回指但双否定恢复回指
    - 核心思路：比较 "The farmer owned a cow" (可引用) vs "didn't own a cow" (不可引用) vs "It was not the case that...didn't own" (双否定=可引用) 三种条件下续句的概率。进一步通过添加 "in fact" 来探测词汇线索的影响。
    - 设计动机：双否定消除是一个精细的语义推理——两个否定相互抵消应恢复回指可及性。这是真正的结构理解 vs 词汇模式匹配的试金石。

3. **析取实验**:

    - 功能：测试 Evans观察——否定量词在析取中的特殊行为
    - 核心思路：对比 "Either there was no manuscript, or it was hidden" (合格) vs "Either there was a manuscript, or it was hidden" (不合格)。使用 SLOR 分数标准化句子长度和词频的影响。
    - 设计动机：析取中的回指可及性是形式语义学中最微妙的预测之一，能深度测试 LLM 的篇章理解。

### 损失函数 / 训练策略

本文是评估研究，无训练。使用 Llama3 系列（1B/3B/8B/8B-Instruct）和 GPT babbage-002/davinci-002。人类实验招募 104 名参与者，在 Prolific 平台上进行强制选择实验。

## 实验关键数据

### 主实验

| 实验 | LLM 表现 | 人类表现 | 说明 |
|------|---------|---------|------|
| exi > every | ~75% | 接近100% | LLM 成功但低于人类 |
| exi > if | ~100% | ~70% | LLM 反而高于人类（telescoping效应）|
| exi > neg | 高于chance | 高于chance | 两者都成功 |
| DN > neg | 部分模型反转 | 高于chance | LLM 在双否定上挣扎 |
| EitherOr > And | ~100% | 高于chance | LLM 成功 |
| or > EitherPosOr | 反转！ | 无明显偏好 | LLM 过度依赖 "either" 词汇线索 |

### 消融实验（添加 "in fact"）

| 配置 | 效果 | 说明 |
|------|------|------|
| DN>Neg + "in fact" | 准确率提升 | "in fact"与双否定共现模式帮助了LLM |
| Exi>Neg + "in fact" | 方向反转！ | "in fact"不常出现在存在句后，降低了概率 |
| 人类 + "in fact" | 无变化 | 人类不受词汇线索干扰 |

### 关键发现

- LLM 在简单的存在/全称对比上表现较好，但其成功部分依赖于词汇线索而非结构理解
- 双否定是 LLM 的弱项——大多数模型无法正确处理否定消除
- 析取中的关键发现：LLM 过度依赖 "either" 这个词——有 "either" 就偏好，没有就不偏好，即使语义等价
- 人类与 LLM 在 he-续句（vs it-续句）的条件句中表现相反，可能因为人类有 telescoping 倾向

## 亮点与洞察

- **三层次语义理解层级体系**（词汇→句子→篇章）是一个有价值的概念框架，为评估 NLU 提供了系统化的指导。
- 实验设计精巧：利用形式语义学的精细预测设计最小对立，并结合人类被试实验，实现了 LLM 与人类的直接对比。
- **LLM 依赖词汇线索而非结构抽象**这一发现意义重大——说明当前 LLM 的篇章"理解"可能更多是表面模式匹配。

## 局限与展望

- 仅测试了 Llama3 和 GPT 系列的较小模型，未能测试 GPT-4o 等 SOTA 模型（API 不支持 logprobs）
- 评估构造较简单，未覆盖更复杂的语言结构（如模态下属）
- 行为层面的评估无法揭示模型内部的机制，需要机制可解释性方法补充

## 相关工作与启发

- **vs Schuster & Linzen (2022)**: 仅测试否定对篇章实体的影响，本文扩展到全称量词和析取
- **vs Kim & Schuster (2023)**: 使用简单场景（如移动盒子中的物品）测试状态追踪，本文关注自然语言的语义结构
- **vs Li et al. (2021)**: 探测 transformer 内部状态的实体追踪表示，本文从行为层面补充评估

## 评分

- 新颖性: ⭐⭐⭐⭐ 将形式语义学预测系统地应用于LLM评估，视角独特
- 实验充分度: ⭐⭐⭐⭐ 人类对比实验+多种LLM+多种语言结构，但模型范围有限
- 写作质量: ⭐⭐⭐⭐⭐ 理论背景铺垫充分，概念层级体系清晰
- 价值: ⭐⭐⭐⭐ 揭示LLM篇章理解的局限性，对NLU研究有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)
- [\[ACL 2025\] Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs](beyond_profile_from_surface-level_facts_to_deep_persona_simulation_in_llms.md)
- [\[ACL 2025\] ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent](eclm_entity_level_language_model_spoken_language_understanding.md)
- [\[ACL 2025\] Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning](character_level_understanding.md)
- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)

</div>

<!-- RELATED:END -->
