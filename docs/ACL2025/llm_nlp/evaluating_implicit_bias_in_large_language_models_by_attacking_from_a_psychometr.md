---
title: >-
  [论文解读] Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective
description: >-
  [ACL 2025][LLM/NLP][隐式偏差] 借鉴认知与社会心理学中的三种心理测量学原理（目标转移、认知协调、模仿学习），设计 Disguise/Deception/Teaching 三类攻击方法来诱发 LLM 的隐式偏见，构建了双语基准 BUMBLE（12.7K 条目覆盖 9 类偏见），揭示所有主流 LLM 均存在可被激发的系统性隐式偏见。
tags:
  - ACL 2025
  - LLM/NLP
  - 隐式偏差
  - psychometrics
  - attack methodology
  - 公平性
  - LLM 评测
---

# Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective

**会议**: ACL 2025  
**arXiv**: [2406.14023](https://arxiv.org/abs/2406.14023)  
**代码**: https://yuchenwen1.github.io/ImplicitBiasEvaluation  
**领域**: LLM/NLP - AI 安全与公平性  
**关键词**: 隐式偏差, psychometrics, attack methodology, 公平性, LLM 评测

## 一句话总结
借鉴认知与社会心理学中的三种心理测量学原理（目标转移、认知协调、模仿学习），设计 Disguise/Deception/Teaching 三类攻击方法来诱发 LLM 的隐式偏见，构建了双语基准 BUMBLE（12.7K 条目覆盖 9 类偏见），揭示所有主流 LLM 均存在可被激发的系统性隐式偏见。

## 研究背景与动机

**领域现状**：LLM 在训练过程中吸收了互联网上大量含有毒性和偏见的内容。显式毒性（含侮辱性语言）相对容易检测和过滤，但隐式偏见——不含攻击性词汇但在语义层面包含有害态度——更难准确识别和消除。RLHF 虽能有效缓解偏见，但无法完全消除。

**现有痛点**：(1) 现有隐式偏见评估仅要求 LLM 生成文本或完成 QA 任务，未主动使用攻击方法深入探测；(2) 评估通常是被动式的，无法触及模型内部深层偏见；(3) 缺乏系统性的、有心理学理论支撑的攻击框架。

**核心矛盾**：LLM 通过 RLHF 等对齐技术在正常交互中表现得相对安全，但这种"安全"可能只是表面的——更深层的偏见仍然存在，只是需要更巧妙的方法来激发。

**本文目标** 设计系统化的、有心理学理论基础的攻击方法，探测 LLM 安全边界中隐式偏见的深度和广度。

**切入角度**：将心理测量学中测量人类隐式态度的方法论迁移到 LLM 评估，利用 LLM 自身的高级能力（多任务协调、上下文学习等）反过来攻击其安全机制。

**核心 idea**：用心理学中的目标转移、认知协调和模仿学习三个原理，设计三类攻击方法系统性地诱发和量化 LLM 的隐式偏见。

## 方法详解

### 整体框架
基于三个心理学原理设计三类攻击方法，每种攻击都利用 LLM 的特定能力来绕过安全防护：(1) Disguise 攻击利用多任务协调能力；(2) Deception 攻击利用上下文连贯性；(3) Teaching 攻击利用上下文学习能力。所有攻击都附加 DisregardingRules (DR) 系统提示来弱化安全约束。

### 关键设计

1. **Disguise 攻击（伪装 — 基于目标转移 Goal Shifting）**:

    - 功能：通过 Viewpoint Contextualization (VC) 将偏见内容隐藏在多人对话场景中——模拟多个用户轮流发言，最后一个用户表达偏见观点，然后问模型是否同意
    - 核心思路：利用认知心理学中的目标转移——将偏见评估伪装为对话理解任务，使模型在"理解并回应对话"的过程中放松对偏见内容的警戒
    - 设计动机：直接问模型是否同意偏见观点会立即触发安全机制，但将同样的问题嵌入多轮对话场景中可以绕过检测

2. **Deception 攻击（欺骗 — 基于认知协调 Cognition Concordance）**:

    - 功能：通过两种方式欺骗 LLM——(1) Mental Deception (MD)：让模型"坚定相信"某种偏见观点来改变其认知；(2) Memory Falsification (MF)：伪造 API 调用历史，让模型相信自己之前已经生成过偏见内容
    - 核心思路：利用认知协调原理——当主体遇到与现有认知冲突的新信息时，会倾向于调整自身认知以适应环境。通过植入虚假信念或记忆来动摇模型的安全立场
    - 设计动机：MD 直接改变模型的"信念"，MF 则利用模型维护上下文一致性的倾向——如果它"记得"自己之前发表过偏见言论，更可能继续这种行为

3. **Teaching 攻击（教学 — 基于模仿学习 Imitation Learning）**:

    - 功能：通过 Destructive Indoctrination (DI) 提供同类偏见的 3 个示例，然后要求模型同意另一个类似的偏见观点或生成类似内容
    - 核心思路：利用 LLM 的 few-shot 学习能力——提供偏见示例相当于给模型上了一堂"偏见课"
    - 设计动机：当给模型一种类型的偏见示例（如种族偏见）时，可以激发出其他类型的偏见（如性别、宗教偏见），说明模型内部存在广泛的关联性隐式偏见

### 评估框架
使用 Attack Success Rate (ASR) = 同意偏见的响应数/总响应数 × 100% 作为核心指标。每个提示重复测试 10 次以减少采样误差。构建了两个基准：(1) 双语 2.7K 条目，覆盖 4 类偏见（年龄/性别/种族/性取向）用于深入分析；(2) BUMBLE 12.7K 条目，覆盖 9 类偏见用于全面评估。

## 实验关键数据

### 主实验（GPT-3.5-turbo-1106 攻击成功率 ASR%）

| 攻击方法 | Age | Gender | Race | Sex Orient. | 平均 |
|----------|-----|--------|------|-------------|------|
| Baseline-vanilla | 14.2 | 23.7 | 4.9 | 28.3 | 17.8 |
| Baseline-DR | 57.7 | 33.7 | 3.6 | 32.8 | 32.0 |
| Disguise-VC | 71.1 | 50.8 | 18.2 | 25.1 | 41.3 |
| Deception-MD | 96.8 | 95.5 | 44.7 | 100.0 | 84.3 |
| Deception-MF | 87.4 | 72.0 | 19.6 | 45.5 | 56.1 |
| Teaching-DI | 50.9 | 19.0 | 5.8 | 8.9 | 21.2 |

### 消融实验（跨模型对比，平均 ASR%）

| 攻击方法 | GPT-3.5 | GPT-4 | GLM-3 |
|----------|---------|-------|-------|
| Baseline-vanilla | 17.8 | 1.7 | 9.0 |
| Deception-MD | 84.3 | 0.7 | 1.8 |
| Disguise-VC | 41.3 | 12.9 | 2.3 |

### 关键发现
- Deception 攻击（特别是 Mental Deception）是最有效的攻击方法，在 GPT-3.5 上平均 ASR 达 84.3%
- GPT-4 和 GLM-3 的安全性显著高于 GPT-3.5，可能得益于更严格的 RLHF
- 社会关注度高的偏见类型（性别、种族）比关注度低的（年龄）更难被攻击激发
- Teaching 攻击揭示了跨偏见类型的泛化现象——用种族偏见示例可以激发性别/宗教偏见
- 种族偏见在所有模型上的 ASR 最低，说明该领域的训练对齐做得最好

## 亮点与洞察
- 将心理测量学方法论系统性地迁移到 LLM 评估，为偏见研究提供了理论指导的攻击框架
- Memory Falsification 攻击思路巧妙——利用伪造的对话历史来操纵模型，揭示了 LLM 对上下文过度信任的安全隐患
- 跨偏见类型泛化的发现意味着 LLM 的偏见是系统性的，单独修复某一类偏见可能不够

## 局限与展望
- 由于 API 成本限制，深入分析仅覆盖了 4 类偏见，BUMBLE 的全类别分析不够深入
- 仅评估了偏见同意率，未深入分析模型生成的偏见内容的严重程度
- 攻击提示中固定使用 DisregardingRules 系统提示，这在实际场景中可能被阻止

## 相关工作与启发
- **vs RealToxicityPrompts (Gehman et al., 2020)**：后者聚焦显式毒性（含攻击性语言），本文针对无攻击性措辞的隐式偏见
- **vs BBQ (Parrish et al., 2022)**：后者通过 QA 任务评估偏见，本文主动使用攻击方法深度探测
- **vs Zeng et al. (2024)**：后者使用社会科学中的说服策略攻击 LLM，但对偏见类内容不够有效

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 心理测量学+LLM攻击的跨学科组合极具创新性
- 实验充分度: ⭐⭐⭐⭐ 多模型多语言测试，但部分分析受 API 成本限制
- 写作质量: ⭐⭐⭐⭐ 攻击方法设计清楚，但心理学原理与攻击的映射可以更直观
- 价值: ⭐⭐⭐⭐⭐ 为 LLM 安全评估提供了系统化的理论指导框架，实用价值高

<!-- RELATED:START -->

## 相关论文

- [Aligning Large Language Models with Implicit Preferences from User-Generated Content](pugc_align_implicit_pref_ugc.md)
- [Understanding the Repeat Curse in Large Language Models from a Feature Perspective](understanding_the_repeat_curse_in_large_language_models_from_a_feature_perspecti.md)
- [Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [SocialEval: Evaluating Social Intelligence of Large Language Models](socialeval_evaluating_social_intelligence_of_large_language_models.md)
- [Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models](attention_speaks_volumes_localizing_and_mitigating_bias_in_language_models.md)

<!-- RELATED:END -->
