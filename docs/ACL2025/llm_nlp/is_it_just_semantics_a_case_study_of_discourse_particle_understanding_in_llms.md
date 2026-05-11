---
title: >-
  [论文解读] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs
description: >-
  [ACL 2025][LLM/NLP][discourse particles] 以英语多义语气词 "just" 为案例，通过专家构造数据集和电影字幕标注数据，用两种元语言实验（few-shot 语义标注和成对比较）系统评估 LLM 对语气词细粒度语义的理解能力，发现模型能区分大类（形容词、时间义）但无法充分捕捉语气词的微妙语义差异（排他、轻描淡写、无因、强调义）。
tags:
  - ACL 2025
  - LLM/NLP
  - discourse particles
  - pragmatics
  - linguistic evaluation
  - sense disambiguation
  - LLM
---

# Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs

**会议**: ACL 2025  
**arXiv**: [2506.04534](https://arxiv.org/abs/2506.04534)  
**代码**: [GitHub](https://github.com/sheffwb/IsItJUSTSemantics)  
**领域**: LLM/NLP  
**关键词**: discourse particles, pragmatics, linguistic evaluation, sense disambiguation, LLM

## 一句话总结

以英语多义语气词 "just" 为案例，通过专家构造数据集和电影字幕标注数据，用两种元语言实验（few-shot 语义标注和成对比较）系统评估 LLM 对语气词细粒度语义的理解能力，发现模型能区分大类（形容词、时间义）但无法充分捕捉语气词的微妙语义差异（排他、轻描淡写、无因、强调义）。

## 研究背景与动机

**领域现状**：语气词（discourse particles）是语言理解的关键因素，一个词承载多种语义功能（如 "just" 有排他、时间、强调、轻描淡写、无因等至少 12 种义项）。现有研究已大量探讨 LLM 对功能词和篇章连接词的处理能力。

**现有痛点**：已有工作（Chan et al. 2024; Yung et al. 2024）发现 LLM 在篇章关系理解上存在不足，但均停留在较粗粒度。对于同一词的多义微妙差异——特别是形式语义学中深入研究的语气词——LLM 的表现完全未知。

**核心矛盾**：语气词在日常对话中极为高频且对语篇理解至关重要，但其语义区分高度依赖语用上下文，且不同义项之间的边界模糊（如 "Fido is just a dog" 的轻描淡写义 vs "The lights just turn on and off" 的无因义），这对 LLM 构成独特挑战。

**本文目标** 定量评估 LLM 是否能区分语气词 "just" 的细粒度语义。

**切入角度**：从形式语义学理论出发（Deo and Thomas 2025 的 "just" 统一分析），由语言学专家构造高质量数据，设计两种互补的元语言评估范式。

**核心 idea**：用专家标注数据 + 两种评估范式（few-shot 标注 + 成对比较），证明 LLM 对多义语气词的细粒度语义理解存在显著缺陷。

## 方法详解

### 整体框架

本文属于 benchmark/评估类研究。围绕英语语气词 "just" 设计了两组实验：（1）Few-shot 语义标注实验——给模型提供六种义项的定义和示例，让模型判断句子中 "just" 的语义；（2）成对比较实验——给模型两个句子，判断其中 "just" 的用法是否相同。两组实验分别使用手工构造数据（90 句，每种义项 15 句）和电影字幕标注数据（149 句）。

### 关键设计

1. **数据构造与标注**:

    - 功能：构建两个高质量数据集用于评估
    - 核心思路：（a）手工构造语料——由语言学研究生精心创建 90 句无歧义句子（每种义项 15 句），确保每句只有唯一的 "just" 义项读解；（b）标注语料——从 OpenSubtitles 提取 149 句包含 "just" 的电影字幕，由 2 名资深语义学家和 8 名研究生标注，不一致时由额外 2 名高级标注者裁决
    - 设计动机：手工数据确保评估精确性（排除歧义），字幕数据检验自然场景下的真实表现

2. **目标义项选取**:

    - 功能：从 Deo and Thomas 2025 的 12 种义项中选取 4 种目标义项 + 2 种控制义项
    - 核心思路：4 种目标义项为排他义（exclusive: "Betsy just eats chicken nuggets"）、轻描淡写义（unelaboratory: "Fido is just a dog"）、无因义（unexplanatory: "The lights just turn on and off"）、强调义（emphatic: "This pumpkin bisque is just delicious!"）；2 种控制义项为时间义（temporal）和形容词义（adjective）
    - 设计动机：4 种目标义项在语义上相对可区分但仍有微妙差别，2 种控制义项提供清晰的难度梯度（形容词义最易区分，时间义中等）

3. **实验一：Few-shot 语义标注**:

    - 功能：评估模型在给定义项定义和示例后，能否正确预测 "just" 的语义
    - 核心思路：使用条件对数概率 $\arg\max_{l \in L} P_M(l|S)$ 选取最高概率的标签，而非解析模型的自由文本输出。使用 minicons 库计算条件概率
    - 设计动机：避免解析冗长生成文本的困难，直接比较各标签的注意力概率

4. **实验二：成对比较**:

    - 功能：不假设模型了解义项标签，直接探测模型是否感知两个句子中 "just" 用法的异同
    - 核心思路：给模型两个句子 $s_i, s_j$，计算 $H^M_{ij} = \log P_M(\text{Yes}|Z_{ij}) - \log P_M(\text{No}|Z_{ij})$ 并归一化到 $[0,1]$，生成热力图。理想情况下，同义项句对的值应高于不同义项句对
    - 设计动机：消除对义项标签名称的依赖，允许语义区分的梯度性，提供比标注实验更自由的评估视角

### 评估模型

测试了 10 个指令微调模型：Llama-3-8b、Llama-3.2-1b/3b、Llama-3.3-70b、Mistral-7b-v0.3、OLMo-7b、OLMo2-7b/13b、Gemma2-2b/9b。

## 实验关键数据

### 主实验（Few-shot 语义标注准确率）

| 模型 | 手工数据（4目标义项） | 字幕数据（无上下文） | 字幕数据（有上下文） |
|------|---------------------|--------------------|--------------------|
| Llama-3.2-1b | ≈ chance (0.167) | ≈ chance (0.403) | - |
| Gemma-2-2b | +0.28 vs 1b | 下降 | - |
| Mistral-7b-v0.3 | 表现最佳之一 | -0.24 avg | - |
| Gemma-2-9b | 表现最佳之一 | 下降 | 进一步 -0.05 |
| Llama-3.3-70b | 与7B持平 | 下降 | +0.10（仍近chance） |

### 成对比较实验

| 模型 | 同义/异义分离显著性 | Cohen's d |
|------|-------------------|-----------|
| Llama-3.2-1b | p=.11（不显著） | - |
| Llama-3.3-70b | p<.005 | 2.32 |
| Gemma-2-9b | p<.005 | 1.91 |
| Mistral-7b-v0.3 | p<.005 | 1.66 |

### 关键发现
- 存在关键模型规模阈值：2B 参数以上才展现超越随机的能力（1B→2B 准确率 +0.28）
- 最大模型（70B）并未比最好的 7B 模型（Mistral、Gemma-9b）表现明显更好
- 从手工数据到自然字幕，所有模型准确率平均下降 0.24，说明模型在真实场景中的理解更差
- 上下文信息（前 2 句对白）并未帮助模型消歧，反而平均降低 0.05（70B 除外）
- 成对比较中：形容词义和时间义能被较好区分，但 4 种目标语气词义之间的区分极弱
- 对照实验（bat、bank）证明该方法对明确多义词有效，问题确实在于语气词语义的微妙性

## 亮点与洞察
- 研究设计精巧，两种互补的评估范式（标注 vs 成对比较）共同构成有力证据——前者测试显式元语言能力，后者测试隐式语义感知，两者结论一致
- 形式语义学理论与 NLP 实证研究的优秀结合，数据由语义学专家专业构建，质量远超自动构造的 benchmark

## 局限与展望
- 仅研究英语 "just" 一个语气词，结论向其他语气词的推广需要验证
- 元语言 prompting 方法可能低估 LLM 的语言能力（Hu and Levy 2023），但基于对数概率的标准方法不适用于语气词语义区分
- 仅使用文本，无法利用语调信息（语调在口语中是消歧的关键线索）

## 相关工作与启发
- **vs Chan et al. 2024**: 研究 ChatGPT 在篇章关系（时间、因果）上的表现，但粒度较粗；本文聚焦单词内部的多义微妙差异
- **vs Pandia et al. 2021**: 研究 LM 对篇章连接词的敏感性，本文进一步深入到语气词的特定义项区分
- **vs Coppock and Beaver 2014**: 形式语义学中排他义的经典分析，本文将其语义理论用于 LLM 评估

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统评估 LLM 对语气词细粒度语义的理解，填补了一个重要空白
- 实验充分度: ⭐⭐⭐⭐ 两种互补实验 + 10 个模型 + 控制实验，覆盖面广；但无闭源大模型（GPT-4等）的对比
- 写作质量: ⭐⭐⭐⭐ 语言学和 NLP 两个视角的叙述衔接流畅
- 价值: ⭐⭐⭐⭐ 揭示了 LLM 在语用理解上的重要盲区，对 NLP 语义评估和语言学计算实验均有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs](how_llms_comprehend_temporal_meaning_in_narratives_a_case_study_in_cognitive_eva.md)
- [\[ACL 2025\] Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility](meaning-beyond-truth-conditions-anaphora-accessibility.md)
- [\[ACL 2025\] Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study](algorithmic_fidelity_german_opinion.md)
- [\[ACL 2025\] Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study](enhancing_text_editing_for_grammatical_error_correction_arabic_as_a_case_study.md)

</div>

<!-- RELATED:END -->
