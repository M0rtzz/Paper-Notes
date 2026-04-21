---
title: >-
  [论文解读] ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events
description: >-
  [ACL2025][LLM/NLP][时间推理] 提出 ChronoSense 基准，首次完整覆盖 Allen 区间代数全部 13 种时间关系并加入 3 类时间算术任务，通过对 7 个 LLM 在 0-shot / few-shot / CoT 下的系统评估，揭示模型时间理解能力普遍薄弱且严重依赖预训练记忆。
tags:
  - ACL2025
  - LLM/NLP
  - 时间推理
  - Allen区间关系
  - benchmark
  - 时间算术
  - LLM评估
---

# ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events

**会议**: ACL2025  
**arXiv**: [2501.03040](https://arxiv.org/abs/2501.03040)  
**代码**: [duyguislakoglu/chronosense](https://github.com/duyguislakoglu/chronosense)  
**领域**: llm_nlp  
**关键词**: 时间推理, Allen区间关系, benchmark, 时间算术, LLM评估

## 一句话总结

提出 ChronoSense 基准，首次完整覆盖 Allen 区间代数全部 13 种时间关系并加入 3 类时间算术任务，通过对 7 个 LLM 在 0-shot / few-shot / CoT 下的系统评估，揭示模型时间理解能力普遍薄弱且严重依赖预训练记忆。

## 研究背景与动机

1. **LLM 时间推理能力薄弱**：尽管 LLM 在通用 NLP 任务上表现优异，但在推理、算术和数值处理方面仍存在重大短板，直接制约了其时间推理能力。
2. **Allen 区间代数作为基础框架长期被忽视**：Allen 区间代数定义了两个时间区间之间 13 种互斥且穷尽的关系（Before, After, During, Contains, Equals, Overlaps, Overlapped-by, Meets, Met-by, Starts, Started-by, Finishes, Finished-by），是时间推理的经典形式化工具，已使用超 30 年，但现有 LLM 时间推理基准**没有任何一个**完整覆盖全部 13 种关系。
3. **已有基准各有缺陷**：TimeBench 聚焦抽象时间表达和常识推理但未覆盖所有 Allen 关系；TGQA 仅覆盖 3 种简单事件关系；TRAM 缺少显式事件的起止时间信息；TORQUE 缺少明确时间戳——亟需一个系统性基准填补空白。
4. **记忆 vs. 推理无法区分**：当模型看到预训练语料中出现过的真实历史事件时，可能直接"背答案"而非真正进行时间推理，需要实验机制来剥离这一混淆因素。
5. **实际应用要求高可靠性**：历史分析、法律 AI、医学时间线管理等场景对时间关系判断的正确率要求远高于 50% 的随机水平，需要诊断工具定位 LLM 的具体弱点。

## 方法详解

### 数据集整体架构

ChronoSense 包含两大类 True/False 判别题，时间粒度统一为「年」：

- **Allen 区间关系题**（类型 1）：给定两个事件及其时间区间，判断某个 Allen 关系是否成立
- **时间算术题**（类型 2）：给定单个事件的时间属性，判断某个算术推导结论是否正确

每种 Allen 关系和每种算术任务分别包含 **4,000 训练 / 500 验证 / 500 测试**样本，正负样本严格 50:50 均衡。

### Allen 区间关系题的构建

数据构建流程：

1. **事件对抽取**：通过 SPARQL 从 Wikidata 抽取具有明确起止年份的真实历史事件对
2. **关系自动标注**：比较两个事件的时间区间，确定唯一正确的 Allen 关系
3. **自然语言化**：将 Allen 关系转化为自然语言假设（Hypothesis），与事件描述（Context）组合
4. **负样本生成**：将正确的 Allen 关系替换为另一种关系并标记为 False。由于年粒度下某些关系可能存在歧义（如 Equals 不能用 Contains 作为负样本，因具体日期未知），论文精心设计了排除规则来避免模糊案例
5. **抽象版本**：将真实事件名替换为"Event A""Event B"，用于对比检测记忆效应
6. **多提示模板**：每个问题设计 3 种不同的自然语言表述，以测试提示敏感性

示例：Context 给出"第四次霍乱大流行 1863-1875"和"二战 1939-1945"，Hypothesis 问"第四次霍乱大流行是否发生在二战之前且无重叠？"，Correctness = True。

### 时间算术题的构建

三类算术任务：

- **End Timepoint**：给定起始时间和持续时间，判断结束时间计算是否正确（如 1948 年开始 + 39 年 = 1987 年？）
- **Next Occurrence**：给定事件首次发生时间和频率周期，判断下次发生时间是否正确（如首次 1773 年 + 每 5 年 → 1779 年发生？）
- **Intermediate Timepoint**（本文首创）：判断事件在起止时间区间内的某个年份是否正在进行

由于 Wikidata 中频率事件有限，算术题均使用合成"Event A"命名，天然处于抽象设置。

### 评估配置

- **7 个模型**：Gemma2-9B-it, GPT-4o, GPT-4o-mini, Llama3.1-8B, Mistral-7B, Mixtral-8x7B, Phi-3-mini
- **4 种提示策略**：0-shot, 1-shot, 3-shot, Chain-of-Thought（在提示末尾加"Let's think step by step."）
- **评估指标**：准确率（随机基线 50%）
- **生成限制**：普通设置最大 64 tokens，CoT 设置最大 512 tokens

## 实验结果

### 表 1：Allen 关系与算术任务的平均准确率

| 任务类型 | 设置 | Gemma2-9B | GPT-4o | GPT-4o-mini | Llama3.1-8B | Mistral-7B | Mixtral-8x7B | Phi-3-mini |
|---------|------|-----------|--------|-------------|-------------|------------|-------------|-----------|
| Allen | 0-shot | 0.09\* | **0.87** | 0.72 | 0.13\* | 0.50 | 0.54 | 0.56 |
| Allen | 1-shot | 0.75 | **0.93** | 0.75 | 0.01\* | 0.47 | 0.56 | 0.59 |
| Allen | 3-shot | 0.26\* | **0.95** | 0.78 | 0.01\* | 0.49 | 0.58 | 0.66 |
| Allen | CoT | 0.75 | 0.65 | 0.69 | **0.75** | 0.51 | 0.57 | **0.75** |
| Allen(抽象) | 0-shot | 0.15\* | **0.78** | 0.64 | 0.14\* | 0.23\* | 0.35 | 0.61 |
| 算术 | 0-shot | **0.76** | 0.55 | 0.60 | 0.48 | 0.36\* | 0.35 | 0.67 |
| 算术 | CoT | 0.94 | **0.99** | **0.99** | 0.92 | 0.70 | 0.75 | 0.98 |

> \* 表示模型因产生大量不清晰答案（≥250 条未回答 True/False）导致准确率失真。

### 表 2：GPT-4o 在 13 种 Allen 关系上的 0-shot 细分表现

| Allen 关系 | GPT-4o | Mixtral-8x7B | Phi-3-mini | 7 模型平均 |
|-----------|--------|-------------|-----------|----------|
| Before | 0.914 | 0.902 | 0.758 | 0.70 |
| After | 0.956 | 0.780 | 0.566 | 0.64 |
| Contains | 0.884 | 0.472 | 0.652 | 0.47 |
| During | 0.878 | 0.512 | 0.490 | 0.46 |
| Overlaps | 0.884 | 0.430 | 0.648 | 0.47 |
| Overlapped-By | 0.842 | 0.476 | 0.786 | 0.49 |
| Meets | 0.910 | 0.740 | 0.488 | 0.52 |
| Met-By | 0.864 | 0.594 | 0.494 | 0.48 |
| Starts | 0.846 | 0.442 | 0.492 | 0.45 |
| Started-By | 0.896 | 0.578 | 0.474 | 0.46 |
| Finishes | 0.908 | 0.430 | 0.492 | 0.43 |
| Finished-By | 0.926 | 0.398 | 0.486 | 0.45 |
| **Equals** | **0.690** | **0.336** | 0.540 | **0.33** |

### 表 3：时间算术三类子任务在 0-shot 与 CoT 下的对比

| 子任务 | 设置 | GPT-4o | GPT-4o-mini | Phi-3-mini | Gemma2-9B | 7 模型平均 |
|--------|------|--------|-------------|-----------|-----------|----------|
| End Timepoint | 0-shot | 0.552 | 0.652 | 0.604 | 0.670 | 0.56 |
| End Timepoint | CoT | 0.978 | 0.978 | 0.996 | 0.992 | 0.95 |
| Intermediate | 0-shot | 1.000 | 0.996 | 0.994 | 0.938 | 0.81 |
| Intermediate | CoT | 0.998 | 0.998 | 0.984 | 0.978 | 0.86 |
| Next Occurrence | 0-shot | 0.126\* | 0.158\* | 0.432 | 0.678 | 0.24 |
| Next Occurrence | CoT | **1.000** | **1.000** | 0.962 | 0.874 | 0.88 |

> Next Occurrence 在 0-shot 下极为困难（多数模型远低于随机基线），但 CoT 后 GPT-4o 达到 1.000，充分证明逐步推理的价值。

## 关键发现

1. **整体表现低下**：多数模型在 Allen 关系任务上接近或低于随机水平（0.50），即使最强的 GPT-4o 在 0-shot 下也仅 0.87，且存在显著的关系间差异
2. **对称关系处理不对称**：尽管 Before/After、Contains/During 等是对称关系对，模型表现差异显著（Before 平均 0.70 vs. After 0.64，Contains 0.47 vs. During 0.46），揭示模型并未真正理解时间对称性
3. **Equals 是最难的关系**：在 0-shot 和 CoT 下平均准确率仅 0.33 和 0.49，因为需要精确比对两个端点是否完全相同
4. **记忆效应显著**：Allen 关系题从真实事件名切换到抽象名后，GPT-4o 准确率从 0.87 降至 0.78，Mixtral 从 0.54 降至 0.35，证实模型部分依赖预训练记忆而非真正推理
5. **CoT 对算术题效果惊人**：算术题从 0-shot 到 CoT 的提升幅度远超 Allen 关系题（GPT-4o: 0.55→0.99），因为算术题本质是多步计算，适合逐步推理
6. **典型错误模式多样**：混淆起止年份、推理逻辑错误、额外无关计算、解释正确但结论错误、时间粒度引起的混乱等

## 亮点与创新

1. **首个完整覆盖 Allen 全部 13 种区间关系的 LLM 时间推理基准**，填补了该领域长期空白
2. **抽象版本设计**精准剥离了记忆与推理的混淆因素，实验设计严谨
3. **Intermediate Timepoint 任务为本文首创**，测试模型判断事件在时间区间内某点是否正在进行，具有原创性
4. **负样本生成策略考虑周到**：在年粒度下精心排除可能产生歧义的 Allen 关系对作为负样本，确保标签无歧义
5. **实验覆盖全面**：4 种提示策略 × 7 个模型 × 16 类任务 × 3 个提示变体，实验矩阵完整

## 局限性与改进方向

1. **时间粒度单一**：仅以"年"为粒度，无法评估模型在日/月/时分秒级别的表现，而细粒度推理在实际场景中更为关键
2. **模型覆盖有限**：仅 7 个模型（含 2 个闭源），缺少 Claude、Qwen、DeepSeek 等重要模型系列
3. **任务形式简单**：仅 True/False 二分类，未测试模型主动识别或生成 Allen 关系的能力
4. **缺少传递推理**：仅测试两两事件的关系判断，未利用 Allen 代数的传递性组合进行多事件链式推理
5. **部分事件名称有歧义**：Wikidata 中存在以人名命名的展览等，可能对模型造成额外困惑
6. **输出截断影响结果**：超过最大 token 限制的回答被直接截断，可能遗漏正确答案

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次完整覆盖 Allen 13 种区间关系，抽象版本设计和 Intermediate Timepoint 任务具有原创价值
- 实验充分度: ⭐⭐⭐⭐ 7 个模型 × 4 种策略 × 16 任务 × 3 提示变体，实验矩阵完整；但模型覆盖可更广
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验设计环环相扣，分析有深度
- 实用价值: ⭐⭐⭐⭐ 为 LLM 时间推理能力诊断提供了标准化工具，直接指导模型改进方向

<!-- RELATED:START -->

## 相关论文

- [Does Time Have Its Place? Temporal Heads Where Language Models Recall Time-specific Information](does_time_have_its_place_temporal_heads_where_language_models_recall_time-specif.md)
- [SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms](synapticrag_enhancing_temporal_memory_retrieval_in_large_language_models_through.md)
- [DeAL: Decoding-time Alignment for Large Language Models](deal_decoding_time_alignment.md)
- [Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events](synergizing_unsupervised_episode_detection_with_llms_for_large-scale_news_events.md)
- [Understanding the Repeat Curse in Large Language Models from a Feature Perspective](understanding_the_repeat_curse_in_large_language_models_from_a_feature_perspecti.md)

<!-- RELATED:END -->
