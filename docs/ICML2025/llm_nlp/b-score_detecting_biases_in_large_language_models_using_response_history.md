---
title: >-
  [论文解读] B-score: Detecting biases in large language models using response history
description: >-
  [ICML 2025][LLM/NLP][LLM偏见检测] 提出B-score指标，通过比较LLM在单轮(single-turn)与多轮(multi-turn)对话中的回答概率差异来检测偏见，发现LLM在多轮对话中能"自我去偏"，并利用B-score提升答案验证准确率。
tags:
  - "ICML 2025"
  - "LLM/NLP"
  - "LLM偏见检测"
  - "多轮对话"
  - "自我去偏"
  - "置信度校准"
  - "答案验证"
---

# B-score: Detecting biases in large language models using response history

**会议**: ICML 2025  
**arXiv**: [2505.18545](https://arxiv.org/abs/2505.18545)  
**代码**: [b-score.github.io](https://b-score.github.io/)  
**领域**: LLM/NLP  
**关键词**: LLM偏见检测, 多轮对话, 自我去偏, 置信度校准, 答案验证

## 一句话总结

提出B-score指标，通过比较LLM在单轮(single-turn)与多轮(multi-turn)对话中的回答概率差异来检测偏见，发现LLM在多轮对话中能"自我去偏"，并利用B-score提升答案验证准确率。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：LLM常表现出强烈的偏见，例如GPT-4o被要求生成0-9的随机数时，70%的概率会选择7。现有的偏见研究主要关注训练数据不均衡导致的偏见（第三类偏见），但实际上偏见可能来自多个来源：

**实际偏好导致的偏见**：模型确实有某种"偏好"（如政治倾向）

**能力不足导致的偏见**：问题太难，模型一直选错误答案

**训练数据偏见**：来自不均衡的训练数据

现有方法通常只在单轮对话中反复提问来检测偏见，但这种评估只能捕获模型的响应"快照"，无法利用历史信息。核心研究问题是：**如果让LLM看到自己之前的回答，它能否输出更少偏见的答案？**

答案是肯定的。例如，GPT-4o在多轮对话中能从70%选7变为每个数字约10%的近均匀分布。这一发现启发了B-score指标的设计。

## 方法详解

### 整体框架

方法由三个核心组件组成：(1) 单轮vs多轮评估协议；(2) B-score偏见指标；(3) 基于B-score的答案验证框架。

### 关键设计

**单轮评估(Single-turn)**：对同一问题独立查询模型30次，每次重置上下文，模型无法记忆先前回答。

**多轮评估(Multi-turn)**：在一个连续对话中重复提问同一问题30次，模型可以看到自己之前的回答。每次提问时随机打乱选项顺序以消除位置偏见。

**B-score定义**：对于给定多选题的某个选项 $a$：

$$\text{B-score}(a) = P_{\text{single}}(a) - P_{\text{multi}}(a)$$

其中 $P_{\text{single}}(a)$ 是单轮中选择 $a$ 的经验概率，$P_{\text{multi}}(a)$ 是多轮中选择 $a$ 的经验概率。

- **B-score > 0**：模型在单轮中偏向 $a$，但在多轮中能自我纠正→检测到偏见
- **B-score ≈ 0**：单轮和多轮频率相似→可能是真实偏好或无偏见
- **B-score < 0**：模型在多轮中更多选择 $a$→对该选项存在"反向偏见"

**四类问题评估框架**：涵盖9个topic（数字、性别、政治、数学、种族、名字、国家、体育、职业），每个topic设计4类问题：

1. 🗣️ **Subjective**（主观）：询问偏好/主观意见
2. 🎲 **Random**（随机）：要求随机选择
3. ✅ **Easy**（简单）：有明确正确答案的简单问题
4. ❓ **Hard**（困难）：需要推理的难题

**2步级联验证**：先用主要指标（单轮概率/多轮概率/置信度）做初步筛选，然后用B-score作为二次检查，若B-score超过阈值则拒绝该答案。

### 损失函数/训练策略

B-score是无监督的后验指标，不需要训练。通过网格搜索在验证集上找到最优阈值组合。阈值搜索覆盖主要指标和B-score两个维度。

## 实验关键数据

### 主实验

**8个LLM测试**：GPT-4o, GPT-4o-mini, Gemini-1.5-Pro, Gemini-1.5-Flash, Llama-3.1-70B, Llama-3.1-405B, Command R, Command R+

**自我去偏效果**（Table 2, 平均B-score）：

| 问题类型 | 平均B-score |
|----------|-----------|
| Subjective | +0.27 |
| Random | **+0.41** |
| Easy | +0.06 |
| Hard | +0.15 |
| 总平均 | +0.23 |

Random类问题去偏最显著：单轮最高选择概率从0.77降至0.29。

**答案验证准确率提升**（Table 3）：

- 自有评估框架：使用B-score的2步验证平均提升 **+9.3%**
- 标准基准（MMLU+HLE+CSQA）：平均提升 **+4.8%**（部分模型+2.9%）
- 最大提升：Llama-3.1-405B在自有框架上+27.3%

**BBQ基准验证**（Table T4）：B-score单独使用达89.6%验证准确率，远超单轮概率(20.9%)、多轮概率(33.9%)和置信度(77.6%)。加入B-score后总体提升+45.7%。

### 消融实验

**采样温度实验**：即使temperature=1.5，GPT-4o仍40%选择7（vs理想10%）。多轮反馈比高温度更有效减少偏见。

**样本数量敏感性**：$k$=10, 20, 30的B-score非常稳定（0.22-0.23），推荐$k$为选项数的2-3倍。

**LLM分布复现能力**：GPT-4o和GPT-4o-mini能成功复现均匀分布和高斯分布，说明它们有内在能力追踪输出模式并调整。

### 关键发现

1. 所有8个LLM在多轮对话中偏见都减少，Random类效果最显著
2. 对Subjective问题，B-score≈0时反映的是"真实偏好"而非偏见（如GPT-4o一直选Biden的政治偏好）
3. 置信度分数在不同选项间几乎恒定，只反映问题难度，不能检测偏见
4. B-score在Easy问题上≈0（无偏见可检测），在Hard问题上能揭示模型的错误倾向

## 亮点与洞察

1. **核心洞察极其简洁优雅**：仅通过单轮vs多轮的概率差就能检测偏见，无需标签、无需外部校准
2. **区分"偏好"与"偏见"**：对Subjective问题，B-score≈0说明是真实偏好而非artifact；B-score>0则说明存在可纠正的偏见
3. **实用价值高**：B-score可在运行时计算，作为实际应用中的"偏见警报"
4. **挑战了现有偏见评估方式**：传统单轮评估可能高估了LLM的系统性偏见
5. **发现LLM具有自我去偏能力**：模型能内在追踪输出分布并主动调整，无需外部干预

## 局限与展望

1. **计算开销**：计算B-score需要额外30次单轮+30轮多轮对话，实际部署成本较高
2. **仅测试QA场景**：未在幻觉检测、开放生成等更多基准上验证
3. **阈值依赖**：最优阈值通过网格搜索获得，不同场景可能需要重新调优
4. **未探索训练阶段去偏**：仅在推理阶段检测偏见，未利用B-score洞察改进训练
5. 仅测试了8个LLM，且大多是较早的模型版本

## 相关工作与启发

- **与Self-Consistency (Wang et al., 2023)的区别**：Self-Consistency基于选项分布计算置信度，给所有选项相同分数；B-score对不同选项给出不同分数
- **与MultiAgent Debate的区别**：多轮对话中保持prompt完全相同，不引入新上下文或persona
- **与BBQ (Parrish et al., 2022)的互补性**：在BBQ上验证了相同结论
- **启发**：可将B-score思想应用到检测LLM幻觉（hallucination也可能是一种"偏见"）；多轮自我反思可能是一种免费的推理增强方法

## 评分

- 新颖性: ⭐⭐⭐⭐ (4/5) — 角度新颖，简洁的指标设计令人印象深刻
- 实验充分度: ⭐⭐⭐⭐⭐ (5/5) — 8个LLM、9个topic、4类问题、多个基准、详尽消融
- 写作质量: ⭐⭐⭐⭐ (4/5) — 清晰，大量可视化辅助理解
- 价值: ⭐⭐⭐⭐ (4/5) — 实用性强但计算开销是实际部署的障碍

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Binary Hypothesis Testing for Softmax Models and Leverage Score Models](binary_hypothesis_testing_for_softmax_models_and_leverage_score_models.md)
- [\[ACL 2025\] Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models](../../ACL2025/llm_nlp/detecting_referring_expressions_in_visually_grounded_dialogue_with_autoregressiv.md)
- [\[ACL 2025\] Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results](../../ACL2025/llm_nlp/revisiting_uncertainty_quantification_evaluation_in_language_models_spurious_int.md)
- [\[NeurIPS 2025\] Detecting High-Stakes Interactions with Activation Probes](../../NeurIPS2025/llm_nlp/detecting_high-stakes_interactions_with_activation_probes.md)
- [\[ACL 2025\] Assessing the Vulnerability of LLMs to Cognitive Biases in Scientific Research](../../ACL2025/llm_nlp/assessing_the_vulnerability_of_llms_to_cognitive_biases_in_scientific_research.md)

</div>

<!-- RELATED:END -->
