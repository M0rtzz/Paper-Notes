---
title: >-
  [论文解读] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?
description: >-
  [ACL 2025][LLM/NLP][认知标记] 本文定义了"标记置信度"（marker confidence）概念来衡量 LLM 使用认知标记（如"fairly certain"）时的实际准确率，通过 7 个模型和 7 个数据集的系统实验发现：认知标记在分布内场景表现稳定，但在分布外场景下极不可靠。
tags:
  - ACL 2025
  - LLM/NLP
  - 认知标记
  - 置信度估计
  - 不确定性
  - LLM 校准
  - 分布外泛化
---

# Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?

**会议**: ACL 2025  
**arXiv**: [2505.24778](https://arxiv.org/abs/2505.24778)  
**代码**: [github.com/HKUST-KnowComp/MarCon](https://github.com/HKUST-KnowComp/MarCon)  
**领域**: LLM/NLP  
**关键词**: 认知标记, 置信度估计, 不确定性, LLM 校准, 分布外泛化

## 一句话总结

本文定义了"标记置信度"（marker confidence）概念来衡量 LLM 使用认知标记（如"fairly certain"）时的实际准确率，通过 7 个模型和 7 个数据集的系统实验发现：认知标记在分布内场景表现稳定，但在分布外场景下极不可靠。

## 研究背景与动机

随着 LLM 在高风险领域（医疗、法律等）的应用增多，准确评估模型输出的置信度变得至关重要。人类通常通过认知标记（epistemic markers）来表达置信度，例如"I am fairly confident"或"it is unlikely that"，而不是给出数值。自然语言是人类与 LLM 交互的主要界面，因此 LLM 能否通过认知标记可靠地反映其内在置信度是一个关键问题。

已有研究主要关注人类与 LLM 对认知标记的理解差异，得出结论是模型总是无法准确用语言表达置信度。但本文指出，即使标记与人类理解不完全一致，只要模型能保持**内部一致性**（即同一标记在不同场景下对应相似的准确率），这些标记仍然有用。因此，之前的研究可能不够充分——未检验 LLM 是否能一致地应用其自身的置信度框架。

## 方法详解

### 整体框架

本文提出了一套系统的认知标记置信度评估框架：（1）定义标记置信度为模型使用特定认知标记时的实际准确率；（2）在多个 QA 数据集上计算所有标记的置信度；（3）通过 7 个评估指标从多维度评估标记的稳定性和一致性。

### 关键设计

1. **标记置信度定义**：给定认知标记 $W$、数据集 $D$、模型 $M$，标记置信度定义为：$\text{Conf}(W, D, M) = \frac{1}{|Q_W|}\sum_{q \in Q_W} \mathbb{I}(M(q))$，即模型生成的回答中包含标记 $W$ 的问题子集上的准确率。这个定义偏离了传统的语义不确定性解释，转而关注标记与实际准确率的对应关系。

2. **七维评估指标体系**：

    - **I-AvgECE**（分布内平均期望校准误差）：衡量同分布下标记置信度与实际准确率的对齐程度，越低越好。
    - **C-AvgECE**（跨分布平均期望校准误差）：评估分布外场景下标记置信度的鲁棒性，越低越好。
    - **NumECE**（数值 ECE）：衡量模型数值置信度输出的整体校准性能，作为基线对比。
    - **MAC**（标记-准确率相关性）：基于 Pearson 系数衡量标记置信度与模型在不同数据集上准确率的相关性。0 表示无相关，1 表示完全正相关。
    - **MRC**（标记排序相关性）：基于 Spearman 系数衡量标记置信度排序在不同数据集间的一致性。
    - **I-AvgCV**（分布内平均变异系数）：捕捉数据集内标记置信度的离散程度，越高说明区分能力越强。
    - **C-AvgCV**（跨分布平均变异系数）：衡量标记置信度在不同数据集间的一致性，越低越稳定。

3. **标记过滤策略**：仅分析在训练集中出现不少于 10 次的标记，以消除随机性影响。过滤阈值的选择是数据完整性与可靠性之间的权衡。

### 损失函数 / 训练策略

本文是分析性研究，不涉及模型训练。核心方法是通过提示（prompt）引导模型在回答中使用认知标记表达不确定性，然后统计分析标记与准确率的关系。同时设计了数值置信度基线作为对比。

## 实验关键数据

### 主实验

| 模型 | I-AvgECE↓ | C-AvgECE↓ | NumECE↓ | C-AvgCV↓ | MAC | MRC↑ | I-AvgCV |
|------|-----------|-----------|---------|----------|-----|------|---------|
| Llama-3.1-8B | 10.09 | 15.95 | 22.70 | 20.80 | 60.91 | 11.37 | 20.48 |
| Qwen2.5-7B | 7.85 | 23.60 | 21.84 | 31.29 | 68.06 | 11.85 | 22.39 |
| Qwen2.5-32B | 4.78 | 10.40 | 8.86 | 19.24 | 78.20 | 36.97 | 16.26 |
| Mistral-7B | 10.58 | 24.81 | 24.46 | 28.52 | 84.57 | 10.54 | 21.01 |
| GPT-4o | 8.55 | 11.84 | 7.56 | 15.72 | 76.44 | 27.54 | 14.30 |
| GPT-4o-mini | 7.65 | 17.15 | 12.79 | 21.98 | 87.68 | 16.48 | 20.61 |
| 平均 | 8.17 | 17.73 | 16.60 | 23.43 | 75.69 | 21.34 | 19.84 |

### 消融实验

| 过滤阈值 | C-AvgCV↓ | MAC | MRC↑ | I-AvgCV |
|---------|----------|-----|------|---------|
| 10（主表） | 23.43 | 75.69 | 21.34 | 19.84 |
| 50 | 23.84 | 86.62 | 23.02 | 13.75 |
| 100 | 23.90 | 82.71 | 20.91 | 12.24 |

模型能力与标记一致性相关性分析：

| 指标 | 与模型准确率的相关系数 |
|------|-------------------|
| C-AvgCV | -0.88（强负相关，能力越强标记越稳定） |
| MRC | 0.75（强正相关，能力越强排序越一致） |

### 关键发现

1. **分布内稳定但跨分布不可靠**：I-AvgECE 始终低于 C-AvgECE（6/7 模型），说明标记在同分布下校准较好，但跨分布泛化能力差。平均 C-AvgCV 达到 23.43%，标记置信度对分布偏移高度敏感。

2. **标记排序不一致**：MRC 整体偏低（平均 21.34%），模型无法在不同数据集间保持稳定的认知标记排序，即"fairly certain"在某数据集上可能比"very likely"置信度低，但在另一数据集上反转。

3. **标记置信度分布集中**：I-AvgCV 仅在 14%-24% 之间，标记之间的区分度有限。49 个（数据集, 模型）设置中仅 4 个包含置信度低于 10% 的标记，说明模型在表达不确定性方面严重不足。

4. **更强的模型理解更好**：C-AvgCV 与模型准确率呈强负相关（-0.88），能力更强的模型（GPT-4o、Qwen2.5-32B）展现出更稳定的标记使用。

5. **标记置信度跟随模型准确率变化**：5/7 模型的 MAC 超过 0.7，标记置信度与模型在不同数据集上的准确率强正相关，本质上反映的是数据集难度而非真正的置信度校准。

## 亮点与洞察

- 提出了标记置信度这一新定义，将关注点从"标记是否与人类理解一致"转向"模型是否能自洽地使用标记"，视角更实用也更深入。
- 七维评估指标体系设计全面，从校准、排序、区分度、跨域稳定性等多个维度提供了系统性洞察。
- 发现更强的模型对认知标记有更好的理解，暗示标记校准可能随模型能力提升而自然改善。
- 揭示了一个深层问题：LLM 虽然在 QA 任务上表现良好，但并不真正"理解"认知标记的含义。

## 局限与展望

- 仅在英语闭源 QA 任务上评估，未考虑其他语言或开放式生成任务。
- 认知标记的定义相对简化，未考虑句法结构、上下文等对置信度表达的影响。
- 仅评估了短回答场景，长文本中的置信度表达更加复杂。
- 未提出改进方案，仅做了诊断性分析。后续可探索通过训练或提示策略提升标记一致性。

## 相关工作与启发

- **Kadavath et al. (2022)** 最早研究了 LLM 在回答中表达不确定性的能力。
- **Yona et al. (2024)** 发现模型无法准确用语言表达置信度，但本文指出其标准（与人类理解对齐）可能过于严格。
- **Xiong et al. (2024)** 探索了基于一致性的黑盒置信度估计方法，是本文数值基线的重要参考。
- 本文的发现对 LLM 部署在高风险场景中的可信度评估有重要启示。

## 评分

- 新颖性: ⭐⭐⭐⭐ 提出了标记置信度的新定义和全面的评估框架，但核心发现（标记不可靠）并不令人意外
- 实验充分度: ⭐⭐⭐⭐ 7 个模型 × 7 个数据集的全面矩阵实验，补充了过滤阈值鲁棒性验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，指标定义明确，但公式符号较密集
- 价值: ⭐⭐⭐⭐ 对 LLM 置信度估计领域有重要的方法论贡献，但缺少改进方案降低了实践价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](refining_salience-aware_sparse_fine-tuning_strategies_for_language_models.md)
- [\[ACL 2025\] Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](direct_confidence_alignment_aligning_verbalized_confidence_with_internal_confide.md)
- [\[ACL 2025\] Reconsidering LLM Uncertainty Estimation Methods in the Wild](reconsidering_llm_uncertainty_estimation_methods_in_the_wild.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)

</div>

<!-- RELATED:END -->
