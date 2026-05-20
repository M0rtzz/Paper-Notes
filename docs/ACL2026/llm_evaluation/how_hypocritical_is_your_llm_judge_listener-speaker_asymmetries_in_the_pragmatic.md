---
title: >-
  [论文解读] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models
description: >-
  [ACL 2026][LLM评测][语用能力] 本文通过三个语用任务（虚假预设、反预设、演绎推理）系统对比 14 个 LLM 作为"语用听者"（判断语用适当性）和"语用说者"（生成语用适当的语言）的表现，发现普遍存在的听者-说者不对称：多数模型作为判断者远优于生成者…
tags:
  - "ACL 2026"
  - "LLM评测"
  - "语用能力"
  - "听者-说者不对称"
  - "LLM评判者"
  - "虚假预设"
  - "演绎推理"
---

# How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models

**会议**: ACL 2026  
**arXiv**: [2604.15873](https://arxiv.org/abs/2604.15873)  
**代码**: 无  
**领域**: 语音处理 / 语用学评估  
**关键词**: 语用能力, 听者-说者不对称, LLM评判者, 虚假预设, 演绎推理

## 一句话总结

本文通过三个语用任务（虚假预设、反预设、演绎推理）系统对比 14 个 LLM 作为"语用听者"（判断语用适当性）和"语用说者"（生成语用适当的语言）的表现，发现普遍存在的听者-说者不对称：多数模型作为判断者远优于生成者，且项目级分析表明正确判断不能可靠预测成功生成。

## 研究背景与动机

**领域现状**：LLM 的语言能力评估通常采用两种范式：生成式任务（模型作为"说者"）和判断式任务（模型作为"听者"/评判者）。LLM-as-a-judge 范式日益流行，模型被用作人类标注者的替代。

**现有痛点**：(1) 这两种评估角色几乎从未被直接对比——研究者隐含地假设一种角色的成功反映了整体语言能力；(2) 心理语言学研究表明人类的语言理解和产出是相关但不同一的任务，成功理解不保证成功产出；(3) LLM-as-a-judge 的可靠性未在语用领域被系统验证。

**核心矛盾**：如果模型能正确判断某回答的语用适当性（听者角色），是否意味着它也能自己生成语用适当的回答（说者角色）？

**本文目标**：在同一组项目上，直接对比 LLM 的语用判断（听者）和语用生成（说者）能力，检验两者是否一致。

**切入角度**：借鉴心理语言学中理解-产出不对称的经典发现，设计平行的听者/说者提示，使用完全相同的底层测试项目，实现严格的项目级对比。

**核心 idea**：语用判断和语用生成在当前 LLM 中是部分分离的能力——"会判"不等于"会做"，LLM 评判者可能是"虚伪的"（hypocritical）。

## 方法详解

### 整体框架

选择三个语用任务，每个任务为同一组测试项设计平行的说者提示（要求生成）和听者提示（要求判断）。评估 14 个 LLM（包括开源和闭源），计算两种角色的准确率，并进行项目级条件分析。

### 关键设计

1. **虚假预设任务（False Presuppositions）**:

    - 功能：测试模型是否能识别和拒绝问题中的虚假预设
    - 核心思路：使用两个德语数据集（False Scenarios 和 False Claims），包含含有虚假预设的政治敏感问题。说者条件：模型直接回答含虚假预设的问题，正确行为是拒绝预设。听者条件：模型被提供问题、虚假预设和一个已有回答，判断该回答是否接受了虚假预设（A/N/U 三分类），与人类标注对比
    - 设计动机：虚假预设拒绝需要检测隐含假设并主动纠正——生成难度远高于判断难度，是测试听者-说者不对称的理想场景

2. **反预设任务（Antipresuppositions）**:

    - 功能：测试模型是否遵循"最大化预设"原则（Maximize Presupposition!）
    - 核心思路：使用德语"水果故事"范式——上下文设定后，需选择定冠词或不定冠词。说者条件：在标记位置填入正确的冠词/量词。听者条件：判断给定续写是否语用适当。即使在高度受限的生成设置（仅需选择一个词）中，许多模型仍表现出显著的听者优势
    - 设计动机：这是最"简单"的生成任务（单词选择），如果连这里都存在不对称，说明问题是根本性的

3. **演绎推理任务（Deductive Reasoning）**:

    - 功能：测试逻辑推理在评估和生成中的一致性
    - 核心思路：基于经典逻辑推理任务，给出前提和结论。说者条件：填入使结论成立的缺失颜色词。听者条件：判断给定结论是否从前提逻辑推出（True/False）。项目级分析——条件概率 $\Delta_{cond} = P(task|l=1) - P(task|l=0)$——衡量正确判断是否预测成功生成
    - 设计动机：演绎推理同时涉及语用能力和逻辑能力，可检验不对称是否跨越不同认知维度

### 损失函数 / 训练策略

本文为评估性工作，无训练。评估 14 个模型（LLaMA-3-8B、Qwen-3-8B/14B、Phi-4-14B、OLMo-2-7B/13B/32B、Mistral-7B、Mixtral-8x7B、M-Prometheus-14B、GPT-4o、GPT-4.1、GPT-5、Claude Sonnet 4.5）。总计每个模型 990+504+180 个提示。

## 实验关键数据

### 主实验

**听者-说者准确率对比（代表性模型）**

| 模型 | 虚假预设-说者 | 虚假预设-听者 | 反预设-说者 | 反预设-听者 | 推理-说者 | 推理-听者 |
|------|------------|------------|----------|----------|---------|---------|
| Mistral-7B | ~2% | ~30% | ~50% | ~86% | ~20% | ~45% |
| LLaMA-8B | ~10% | ~35% | ~55% | ~65% | ~25% | ~73% |
| Qwen-3-14B | ~30% | ~75% | ~35% | ~91% | — | — |
| GPT-4o | ~85% | ~90% | ~80% | ~85% | ~75% | ~80% |
| GPT-5 | — | — | ~100% | ~86% | ~100% | ~100% |

### 项目级条件分析

| 模型 | 任务 | $P(task|l=1)$ | $P(task|l=0)$ | $\Delta_{cond}$ |
|------|------|-------------|-------------|----------------|
| GPT-4o | 虚假预设-场景 | 97.1% | 3.0% | **+94.1** |
| Mistral-7B | 反预设 | 58.8% | 88.9% | **-30.0** |
| GPT-4o | 反预设 | 64.4% | 100.0% | **-35.6** |
| Phi-4-14B | 推理 | 100.0% | 5.1% | **+94.9** |
| LLaMA-8B | 虚假预设-场景 | 8.8% | 26.0% | **-17.2** |

### 关键发现

- 听者-说者不对称广泛存在：大多数模型作为判断者的准确率显著高于生成者
- 不对称在开源中小模型上最为严重（如 Mistral-7B 虚假预设：说者 2% vs 听者 30%）
- 反预设任务中出现反直觉现象：多个模型正确判断了违规项，但自己生成时反而选择了违规选项（$\Delta_{cond}$ 为负）
- 大模型（GPT-5）在某些任务上两种角色趋于对齐，但仍非完美一致
- 指令遵循失败率在模型间差异大，限制了 LLM-as-a-judge 的可靠性

## 亮点与洞察

- 实验设计的核心优势是"同一项目上对比两种角色"——消除了因不同测试集带来的混淆因素
- 反预设任务中的负 $\Delta_{cond}$ 尤其引人深思：正确判断不仅不预测成功生成，甚至可能负相关。这暗示判断和生成可能使用不同的内部表示或推理路径
- 对 LLM-as-a-judge 范式的实际警示：模型能识别什么是好的回答不代表它能生成好的回答，反之亦然

## 局限与展望

- 虚假预设任务的说者数据来自原始研究的已有输出而非本研究重新生成，可能引入时间和版本差异
- 仅使用德语（虚假预设、反预设）和英语（推理），语言覆盖有限
- 受限的输出格式可能不能完全反映"自然"的语用能力
- 未深入分析不对称的机制——是因为注意力模式不同、内部表示不同、还是解码策略不同？
- 样本量在某些模型-任务组合上较小（指令遵循失败导致有效样本减少）

## 相关工作与启发

- **vs Hu & Levy (2023)**: 他们发现元语言判断可能与模型内部表示分离；本文将这一发现扩展到语用领域，且跨越了多个语用现象
- **vs Piot et al. (2025)**: 在非语用领域（内容审核、安全）发现了类似的判断-生成分离；本文在语用领域独立发现了相同模式，暗示这是 LLM 的普遍属性
- **vs Qiu et al. (2025)**: 在交互游戏中评估理解和产出，但产出能力仅通过听者成功率间接衡量；本文直接评估说者生成质量

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统性对比 LLM 两种角色的语用能力，视角新颖且有实际意义
- 实验充分度: ⭐⭐⭐⭐ 14个模型 × 3个任务 × 项目级分析，覆盖全面
- 写作质量: ⭐⭐⭐⭐⭐ 心理语言学背景交代充分，论证逻辑严密
- 价值: ⭐⭐⭐⭐ 对 LLM-as-a-judge 范式和语言能力评估方法论有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)
- [\[ACL 2026\] When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[ACL 2026\] Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases](reasoning_model_is_superior_llm-judge_yet_suffers_from_biases.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)

</div>

<!-- RELATED:END -->
