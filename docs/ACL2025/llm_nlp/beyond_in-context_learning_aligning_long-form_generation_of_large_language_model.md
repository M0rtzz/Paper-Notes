---
title: >-
  [论文解读] Beyond In-Context Learning: Aligning Long-form Generation of LLMs via Task-Inherent Attribute Guidelines
description: >-
  [ACL 2025][LLM/NLP][in-context learning] 从理论和实验两方面证明 ICL 示例无法充分传递任务的语言和格式属性，提出 LongGuide 算法从少量训练数据中自动学习 Metric Guideline (MG) 和 Output Constraint Guideline (OCG) 两类指导规则，在 7 个长文本生成任务上平均提升超过 5% ROUGE-L。
tags:
  - ACL 2025
  - LLM/NLP
  - in-context learning
  - Long-form Generation
  - LongGuide
  - Metric Guidelines
  - Output Constraints
---

# Beyond In-Context Learning: Aligning Long-form Generation of LLMs via Task-Inherent Attribute Guidelines

**会议**: ACL 2025  
**arXiv**: [2506.01265](https://arxiv.org/abs/2506.01265)  
**机构**: National University of Singapore, NTU, A*STAR, Salesforce AI Research
**领域**: LLM NLP / 长文本生成对齐  
**关键词**: in-context learning, Long-form Generation, LongGuide, Metric Guidelines, Output Constraints

## 一句话总结

从理论和实验两方面证明 ICL 示例无法充分传递任务的语言和格式属性，提出 LongGuide 算法从少量训练数据中自动学习 Metric Guideline (MG) 和 Output Constraint Guideline (OCG) 两类指导规则，在 7 个长文本生成任务上平均提升超过 5% ROUGE-L。

## 研究背景与动机

**领域现状**：In-context learning (ICL) 是 LLM 最重要的能力之一，通过在 prompt 中提供少量示例即可校准模型行为。ICL 在分类任务上非常有效，但在长文本生成任务（摘要、翻译、对话生成）中效果有限，且已有理论分析均假设模型完全捕获了任务的语言分布 $P_\mathcal{M}(X) = P_T(X)$，这一假设在现实中并不成立。

**核心痛点**：即使提供 5 个属性得分完美（5/5）的示例，ICL 模型生成的输出中仅 4%–44% 能保持相同的属性得分；增加示例数量（3→5→10）也无法解决这一问题。模型无法从示例中隐式学会任务的语言特征（如简洁性、信息量）和格式特征（句子数、token 数），作者将此称为**文本属性传递 (Property Transfer, PT) 问题**。

**理论洞察**：作者证明当 $P_\mathcal{M} \neq P_T$ 时，无论提供多少示例，ICL 都无法在极限情况下恢复真实的任务分布。这个存在性结论意味着示例中展示的某些属性无法可靠地传递到模型的生成输出中。

**切入角度**：既然隐式学习行不通，就用显式的文本指导规则 (guidelines) 来弥补。实验表明，即使是简单的 "The output must maintain {property}" 指令也能显著改善模型的属性保持能力，尤其是格式属性（句子数、token 数）的改善最为明显。

**核心 idea**：自动从少量训练数据中学习两类 guidelines——质量指标指南 (MG) 捕获语言属性，输出约束指南 (OCG) 捕获格式属性——作为补充指令增强 LLM 长文本生成。

## 方法详解

### 整体框架：LongGuide 五步流程

LongGuide 是一个高效的 guideline 生成算法，仅需 ≤50 个训练样本，通过 5 个步骤并行生成两类 guidelines：

1. **Step 1 — 指标收集与选择**：从 27 个预定义指标池中，用 CoT prompting 让 LLM 选出对当前任务最重要的 top-5 指标，重复 K 轮取并集
2. **Step 2 — 指标自评分**：用 LLM + Self-Consistency 在训练集上对 ground-truth 答案逐样本打分（1-5 分），取平均得到各指标的期望得分
3. **Step 3 — 生成 Metric Guideline (MG)**：将指标得分转换为自然语言描述（如 "Informativeness 4/5" → "应包含适量信息"），拼接为 MG
4. **Step 4 — 生成 Output Constraint Guideline (OCG)**：用 NLTK 计算训练集输出的句子数和 token 数的 min/max/avg 统计量，转化为格式约束
5. **Step 5 — MG-OCG 自动选择**：在训练集上比较 4 种配置（无 guideline, MG, OCG, MG+OCG）的 ROUGE-L，选最优组合

### 指标池设计 (Step 1)

指标池 $S$ 包含 27 个无参考评估指标，来源如下：

| 来源 | 指标 | 数量 |
|------|------|------|
| ABC's of Communication (Wagner, 1963) | Accuracy, Brevity, Clarity | 3 |
| BARTScore (Yuan et al., 2021) | Relevance, Coherence | 2 |
| GPTScore (Fu et al., 2023) | Semantic Coverage, Factuality, Fluency, Informativeness, Consistency, Engagement, Specificity, Correctness, Understandability, Diversity | 10 |
| 作者新增 | Completeness, Conciseness, Neutrality, Naturalness, Readability, Creativity, Rationalness, Truthfulness, Respect of Chronology, Non-repetitiveness, Indicativeness, Resolution | 12 |

设计关键：不收集 LM-based 指标（如 FactScore），因为 LLM 难以定义和自评此类指标；不预设指标定义，因为不同任务对同一指标的理解不同。

### Metric Guideline 生成机制 (Step 2-3)

MG 的核心思路是让 LLM 既当裁判又当选手：

- **自评分**：对每个训练样本，LLM 使用 Self-Consistency 对 ground-truth 答案在所有选中指标上打 1-5 分，取平均。这一步与 Step 1 分离，确保评估的数据和选择指标的数据独立
- **转自然语言**：将数值分数转化为自然语言定义，因为 LLM 对上下文描述的理解优于数值分数。例如 Informativeness 得分 4/5 被描述为 "good amount of informative content"
- **拼接成 MG**：所有指标的定义按字母序拼接为完整的 MG guideline

### Output Constraint Guideline (Step 4)

OCG 关注 6 个格式统计量：ground-truth 答案的句子数和 token 数各自的 min、max、avg。输出模板为："The response must have from {min_s} to {max_s} sentences and from {min_t} to {max_t} words with an average of {avg_t} words and {avg_s} sentences."

### 自动组合选择 (Step 5)

关键发现：不同模型对不同任务有不同的内在知识，单一配置不是万能的。例如 SWiPE 任务因 ground-truth 长度方差极大，OCG 反而有害，而 MG 效果显著；翻译任务中 OCG 比 MG 更有效。因此需要在训练集上自动选择，仅需评估 4 个变体。

## 实验关键数据

### 主实验：7 个长文本生成任务 (ROUGE-L / GPT-4o-Judge)

| 任务 | 模型 | Zero-shot | + LongGuide | 提升 | Few-shot | + LongGuide | 提升 |
|------|------|-----------|-------------|------|----------|-------------|------|
| SAMSum | Mistral | 22.20 / 7.43 | **28.35** / 7.73 | +6.15 / +0.30 | 27.13 / 7.66 | **30.65** / 7.72 | +3.52 / +0.06 |
| CNN/DM | Mistral | 19.23 / 7.38 | **22.46** / 7.45 | +3.23 / +0.07 | 17.56 / 5.84 | **19.19** / 5.99 | +1.63 / +0.15 |
| XL-Sum | Mistral | 9.19 / 5.96 | **14.38** / 6.29 | +5.19 / +0.33 | 9.79 / 4.46 | **15.23** / 5.06 | +5.44 / +0.40 |
| SWiPE | Mistral | 36.60 / 7.21 | **38.21** / 7.32 | +1.61 / +0.11 | 39.47 / 7.12 | **41.36** / 7.24 | +1.89 / +0.12 |
| CommGen | Mistral | 10.12 / 5.14 | **25.20** / 6.81 | +15.08 / +1.67 | 3.98 / 1.34 | **25.05** / 6.65 | +21.07 / +5.31 |
| SAMSum | ChatGPT | 23.83 / 7.43 | **30.47** / 7.59 | +6.64 / +0.16 | 22.21 / 7.32 | **31.46** / 7.72 | +9.25 / +0.40 |
| CommGen | ChatGPT | 24.21 / 6.53 | **34.41** / 7.23 | +10.20 / +0.70 | 22.08 / 4.19 | **38.21** / 7.21 | +16.13 / +3.02 |

平均提升：Mistral +5.39% ROUGE-L，ChatGPT +6.58% ROUGE-L。LongGuide 在所有配置中均超过 APO prompt optimization 基线。

### ICL 属性传递实验：5-shot 示例下的属性保持率

| 模型 | COV | FAC | CON | INF | COH | REL | NT mean / std |
|------|-----|-----|-----|-----|-----|-----|---------------|
| 期望值 | 100% | 100% | 100% | 100% | 100% | 100% | 17.00 / 0.00 |
| Mistral-7B-it | 38% | 80% | 78% | 17% | 75% | 88% | 50.25 / 55.54 |
| Llama-3.1-8B-it | 44% | 86% | 82% | 26% | 81% | 87% | 34.72 / 45.29 |
| Qwen2.5-7B | 43% | 90% | 85% | 40% | 78% | 96% | 281.38 / 264.59 |

关键发现：所有示例均包含 17 个 output tokens，但模型输出平均 50-281 个 tokens，标准差极大。Semantic Coverage (COV) 和 Informativeness (INF) 等关键指标的传递率仅 17%–44%。

### 组件消融：MG vs OCG vs MG+OCG 最优选择次数

在 28 组实验（2 模型 × 7 任务 × 2 设置）中：MG+OCG 胜出 **15** 次，OCG 胜出 **10** 次，MG 胜出 **2** 次，无 guideline 胜出 **1** 次。OCG 在摘要、翻译、表格转文本任务上尤为有效；MG 在 SWiPE 文本简化任务上更具优势。

## 亮点与洞察

- **理论扎实**：通过反证法严格证明了当 $P_\mathcal{M} \neq P_T$ 时 ICL 无法恢复真实任务分布，为 PT 问题提供了理论基础
- **MG 的自评范式巧妙**：利用 LLM 的自评能力（LLM-as-Judge）来发现任务需要优化的属性维度，再反过来用这些维度指导同一模型生成——"让模型告诉自己该怎么做"
- **极低成本**：仅需 ≤50 个训练样本和 4 个 prompt 变体验证，成本约为 APO 的 1/3.75
- **跨模型可迁移**：弱模型 (Mistral) 学到的 MG 可提升强模型 (ChatGPT) 的性能，反之不成立——强模型有更好的理解力来利用弱 guideline
- **与 prompt optimization 互补**：LongGuide 的 guideline 可进一步被 APO、adv-ICL 等算法优化，两者协同效果更好
- **人类评估验证**：标注者 92% 偏好 LongGuide 输出的生成质量，OCG 的人类评估胜率高达 95%

## 局限与展望

- **任务级统计而非样本级**：MG 和 OCG 基于训练集的平均统计量，无法为每个样本提供针对性指导；对于输出长度方差极大的任务（如 Code2Text、StoryGeneration）可能无效
- **依赖指令遵循能力**：非 instruct 模型直接使用效果有限，需要借用 instruct 模型学到的 guideline
- **指标池人工构建**：27 个指标虽覆盖面广，但可能遗漏特定领域的重要属性
- **OCG 约束较粗粒度**：仅约束句子数和 token 数，未涉及段落结构、关键词、语调等更精细的控制维度
- **对已训练任务效果有限**：对模型训练数据中已包含的任务（如 WebNLG、E2E NLG），guidelines 可能引入 OOD 上下文反而降低性能

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐：首次从理论角度揭示 ICL 在长文本生成中的根本局限性，提出的 PT 问题定义清晰；MG+OCG 双流 guideline 框架设计合理且实用
- **实验充分度** ⭐⭐⭐⭐⭐：7 个生成任务 + 1 个真实对话基准，涵盖摘要/简化/翻译/对话/表格转文本，开源和闭源模型均有实验，消融实验和人类评估完善
- **实用价值** ⭐⭐⭐⭐：方法简单易用、成本低（仅需 API 调用），可与现有 prompt optimization 方法协同使用，具有良好的工程落地潜力
- **表达清晰度** ⭐⭐⭐⭐：论文结构完整，理论部分虽然严谨但假设条件的实际意义需要仔细思考
