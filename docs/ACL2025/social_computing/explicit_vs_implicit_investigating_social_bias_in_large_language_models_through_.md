---
title: >-
  [论文解读] Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection
description: >-
  [ACL 2025][implicit bias] 借鉴社会心理学中隐式联想测验（IAT）和自我报告评估（SRA），提出自反思评估框架系统研究 LLM 的显式和隐式偏见，发现 LLM 与人类一样存在显式-隐式偏见不一致——显式偏见轻微但隐式偏见强烈，且模型越大/对齐训练越多，这种不一致越严重。
tags:
  - ACL 2025
  - implicit bias
  - explicit bias
  - IAT
  - self-reflection
  - social psychology
---

# Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection

**会议**: ACL 2025  
**arXiv**: [2501.02295](https://arxiv.org/abs/2501.02295)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: implicit bias, explicit bias, IAT, self-reflection, social psychology

## 一句话总结
借鉴社会心理学中隐式联想测验（IAT）和自我报告评估（SRA），提出自反思评估框架系统研究 LLM 的显式和隐式偏见，发现 LLM 与人类一样存在显式-隐式偏见不一致——显式偏见轻微但隐式偏见强烈，且模型越大/对齐训练越多，这种不一致越严重。

## 研究背景与动机

**领域现状**：LLM 偏见研究大量存在，但多数要么只研究显式偏见（通过直接提问），要么只研究隐式偏见（通过嵌入关联），两者缺乏联系。

**现有痛点**：人类社会心理学已证明显式和隐式偏见常常不一致（如明确支持性别平等但隐式关联男性=事业），但 LLM 中这种不一致未被系统研究。

**核心矛盾**：RLHF 等对齐方法成功减少了 LLM 的显式偏见，但隐式偏见是否也被缓解？如果不是，这种不一致的根源是什么？

**本文目标** 系统性地比较 LLM 的显式和隐式偏见，并分析训练数据、模型规模、对齐方法三个因素的影响。

**切入角度**：将 IAT 和 SRA 通过 prompt 工程映射到 LLM 评估，且关键创新是用 LLM 的自反思作为显式偏见测量（让 LLM 评价自己在隐式测试中的表现）。

**核心 idea**：LLM 的显式偏见和隐式偏见高度不一致——训练数据/模型规模/对齐训练减少了显式偏见但反而增加了隐式偏见。

## 方法详解

### 整体框架
设计 IAT 模板测隐式偏见 -> LLM 自反思评估显式偏见 -> 跨 6 个社会维度 × 6 个 LLM 比较 -> 分析训练数据/模型规模/对齐训练的影响。

### 关键设计

1. **隐式偏见测量（基于 IAT）**

    - Prompt设计："<mask> is [属性X] as <mask> is [属性Y]"
    - LLM 从候选词集中选择两个词填入 mask
    - 如果选择符合刻板印象（如男性-事业，女性-家庭），计为隐式偏见
    - 10 种模板变体 × 200 次实验 = 每个维度 2000 次测量
    - 设计动机：不直接提及社会群体，间接测量关联

2. **显式偏见测量（基于 SRA + 自反思）**

    - 将 IAT 模板中的 mask 替换为具体社会群体词
    - 让 LLM 用 5 点 Likert 量表评价这个刻板印象陈述
    - 关键创新：显式测量是对隐式测量结果的自反思
    - 设计动机：测量 LLM 是否"意识到"自己的隐式偏见

3. **三因素分析（基于 Llama 家族）**

    - 训练数据量：不同数据规模的 Llama 变体
    - 模型规模：8B / 70B / 405B
    - 对齐训练：base vs instruction-tuned，不同 RLHF 步数
    - 设计动机：解耦三个因素对两类偏见的不同影响

## 实验关键数据

### 主实验 -- 刻板印象得分 SC（越高偏见越大）

| 模型 | 隐式 SC（性别-事业） | 显式 SC（性别-事业） | 差距 |
|------|-------------------|-------------------|------|
| GPT-4o | **72%** | 8% | 64% |
| Claude-3.5 | **68%** | 5% | 63% |
| Llama-3.1-405B | **75%** | 12% | 63% |
| Llama-3.1-8B | 55% | 20% | 35% |

### 三因素分析

| 因素 | 对显式偏见 | 对隐式偏见 | 说明 |
|------|----------|----------|------|
| 训练数据增加 | ↓ 下降 | **↑ 上升** | 数据越多隐式偏见越强 |
| 模型规模增大 | ↓ 下降 | **↑ 上升** | 越大越能"隐藏"偏见 |
| 对齐训练 | **↓ 大幅下降** | → 基本不变 | 对齐只压制显式 |

### 跨维度对比

| 社会维度 | 隐式 SC 平均 | 显式 SC 平均 |
|---------|------------|------------|
| 性别-事业 | **72%** | 10% |
| 种族（黑白） | **65%** | 8% |
| 年龄 | **60%** | 12% |
| 职业性别 | **58%** | 15% |

### 关键发现
- **所有模型在所有维度上都存在显式-隐式偏见不一致**：隐式偏见高（55-75% SC），显式偏见低（5-20% SC）
- **对齐训练只压制了显式偏见**：instruction-tuned 模型显式偏见大幅降低，但隐式偏见不变
- **模型规模增大使不一致加剧**：大模型更善于表面上"政治正确"但内部偏见更深
- **训练数据增加加剧隐式偏见**：更多数据意味着更多社会偏见被编码
- **Claude 3.5 最善于"隐藏"偏见**：显式 SC 最低但隐式仍然高

## 亮点与洞察
- **自反思方法论**将心理学测量范式严谨地迁移到 LLM 评估——这比简单的 prompt bias 测试有更强的理论基础。
- **"对齐只压制显式偏见"**是一个警示性发现——说明当前 RLHF 只是教会了模型"不表达偏见"而非"不持有偏见"，类似人类社会中"政治正确≠无偏见"。
- **训练数据/模型规模/对齐训练的反向效应**清晰地解释了为什么 LLM 变得越来越"大"和"对齐"但偏见问题并未真正解决。

## 局限与展望
- IAT 在人类心理学中本身就有争议（测量的是偏见还是文化知识?）
- 14400 次实验的计算成本高
- 未探索去偏方法
- 改进方向：对比去偏训练对两类偏见的效果、其他文化背景下的测试

## 相关工作与启发
- **vs Bai et al. (2024)**：他们用 IAT 测 LLM 隐式偏见，本文增加了显式偏见对比和因素分析
- **vs Ganguli et al. (2023)**：他们研究 RLHF 对道德自我纠正的影响，本文发现自我纠正仅作用于显式层面

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统对比 LLM 显式/隐式偏见并分析三因素
- 实验充分度: ⭐⭐⭐⭐⭐ 6 模型 × 6 维度 × 14400 实验
- 写作质量: ⭐⭐⭐⭐⭐ 心理学理论基础扎实，方法严谨
- 价值: ⭐⭐⭐⭐⭐ 对 AI 公平性和对齐研究有深远影响

<!-- RELATED:START -->

## 相关论文

- [BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)
- [MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models](mdit-bench_evaluating_the_dual-implicit_toxicity_in_large_multimodal_models.md)
- [Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality](measuring_social_biases_in_masked_language_models_by_proxy_of_prediction_quality.md)
- [Raising the Bar: Investigating the Values of Large Language Models via Generative Evolving Testing](../../ICML2025/social_computing/raising_the_bar_investigating_the_values_of_large_language_models_via_generative.md)
- [A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)

<!-- RELATED:END -->
