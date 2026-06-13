---
title: >-
  [论文解读] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs
description: >-
  [ACL 2025][LLM 其他] 本文系统评估了 LLM 利用抽象语义表示（AMR）进行下游任务的能力，发现 AMR 增强的 prompt 在长上下文任务（如对话摘要）中显著提升 Llama 3.1 零样本性能（余弦相似度从 66% 提升至 76%），但在短上下文任务中通常会降低性能。 1. LLM 在 NLP 任务中表…
tags:
  - "ACL 2025"
  - "LLM 其他"
---

# Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs

**会议**: ACL 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: LLM / NLP  

## 一句话总结

本文系统评估了 LLM 利用抽象语义表示（AMR）进行下游任务的能力，发现 AMR 增强的 prompt 在长上下文任务（如对话摘要）中显著提升 Llama 3.1 零样本性能（余弦相似度从 66% 提升至 76%），但在短上下文任务中通常会降低性能。

## 背景与动机

1. **LLM 在 NLP 任务中表现突出但理解深度存疑**：LLM 在翻译、摘要等任务上表现优异，但其是否能从结构化语义表示中提取和利用信息仍不明确。
2. **AMR 在传统方法中的有效性已被验证**：抽象语义表示（AMR）在结构感知 NLP 任务中已被证明能有效增强推理能力，尤其在长上下文场景中。
3. **现有方法依赖架构修改**：先前利用 AMR 的工作大多通过修改模型架构（如 text-graph attention、图 Transformer），增加了复杂度且难以泛化。
4. **直接评估 LLM 理解 AMR 的研究缺失**：尚无工作系统地评估通用 LLM 直接解读线性化 AMR 的能力，以及这种能力在不同任务类型中的变化。
5. **Prompt 工程的新方向**：将结构化语义信息融入 prompt 是一种低成本、无需修改模型的增强策略，但其效果边界尚未被系统探索。
6. **长短上下文任务的差异化需求**：不同长度上下文的任务可能从结构化表示中获益不同，需要细粒度的实验分析。

## 方法详解

### AMR 构建与线性化

- 使用 IBM 的 transition-based neural parser（AMR3-structbart-L 和 doc-sen-conll-amr-seed42 模型）将文本解析为文档级 AMR 结构。
- AMR 被线性化为扁平文本表示后送入 LLM。

### 三种 Prompting 策略

1. **Context-only（基线）**：仅提供原始文本上下文。
2. **AMR-augmented**：同时提供原始文本和其对应的线性化 AMR，测试 AMR 能否辅助上下文理解。
3. **AMR-only**：仅提供线性化 AMR 而不提供原始文本，测试 LLM 从 AMR 中直接推理的能力。

### 任务设置

覆盖 6 类任务：上下文再生（AMR-to-text）、单跳问答（SQuAD 2.0）、双跳推理（HotpotQA）、对话摘要（SAMSum）、句子级 NLI（SNLI）、文档级 NLI（DocNLI）。每个任务均进行零样本、3-shot 和 5-shot 实验。

### 模型

使用 8-bit 量化的指令微调模型：Llama 3.1 (8B)、Phi-3、Mistral 7B。对 SAMSum 还进行了 Llama 3.1 的 rank-32 LoRA 微调。

## 实验结果

### AMR-to-text 再生能力（LDC2020T02）

| 模型 | 样本数 | 余弦相似度 |
|------|--------|-----------|
| Llama 3.1 | 0-shot | 73% |
| Llama 3.1 | 3-shot | 80% |
| Llama 3.1 | 5-shot | **81%** |
| Phi-3 | 0-shot | 74% |
| Phi-3 | 5-shot | 76% |
| Mistral | 5-shot | 76% |

LLM 能有效从线性化 AMR 重建原始文本，Llama 3.1 五样本达到 81% 余弦相似度。

### SAMSum 对话摘要（Llama 3.1 余弦相似度）

| Prompting 策略 | 0-shot | 3-shot | 5-shot |
|---------------|--------|--------|--------|
| Context-only | 66% | ~74% | ~74% |
| AMR-augmented | **76%** | ~75% | ~75% |
| AMR-only | ~60% | ~70% | ~68% |

- AMR 增强在零样本场景下带来 10 个百分点的显著提升。
- 少样本设置下 AMR 增强的优势缩小但仍存在。

### 短上下文任务（SQuAD 2.0，Llama 3.1 F1）

AMR 增强在单跳 QA 中反而降低性能：3-shot 从 59% 降至 52%。AMR-only 在 3-shot 下达到 48% F1，但 5-shot 时急剧下降至 26%，表明过多 AMR 示例会干扰推理。

### NLI 任务

Phi-3 在 SNLI 上表现最佳，AMR 增强在零样本下显著提升 macro F1（27%→39%），但少样本时 context-only 更优（82%）。

## 亮点

- **系统且全面的评估框架**：覆盖 6 种任务 × 3 种 prompt 策略 × 3 种模型 × 3 种 shot 设置，实验矩阵完整。
- **关键发现具有指导意义**：AMR 对长上下文有帮助、对短上下文有害的结论清晰实用，可直接指导 prompt 设计。
- **LLM 确实能理解 AMR**：81% 的文本重建相似度证明 LLM 对结构化语义表示有较强的解读能力。
- **可扩展到其他结构化表示**：方法论框架可推广至知识图谱、话语表示结构等其他结构化形式。

## 局限性

- **未进行全量微调实验**：仅做了 LoRA 微调，且效果不如少样本 prompt，缺乏全量微调的系统对比。
- **长上下文有利的解释不够深入**：AMR 在长上下文中有效的根本原因（信息压缩？关键信息保留？）未做深入分析。
- **模型规模受限**：所用模型均为 7-8B 级别，未探索更大模型（如 70B）上的表现。
- **HotpotQA 未使用 CoT prompting**：这限制了双跳推理实验的公平性和说服力。
- **DocNLI 仅在部分测试集上评估**：需要完整测试集验证才能得出可靠结论。

## 相关工作

- **AMR-to-text 生成**：Zhu et al. (2019) 利用图结构改进 Transformer 的 AMR 生成质量；Koncel-Kedziorski et al. (2022) 使用图 Transformer 从知识图谱生成文本。
- **AMR 增强的 NLP 任务**：Hua et al. (2023) 通过 text-graph attention 融合 AMR 改进长对话摘要；Yang et al. (2024) 在对话评估中通过门控机制融入 AMR。
- **LLM prompting 技术**：Wei et al. (2023) 的 Chain-of-Thought prompting；Lester et al. (2021) 的 soft prompt tuning 为未来方向。
- **参数高效微调**：Hu et al. (2021) 的 LoRA 和 Houlsby et al. (2019) 的 Adapter 为结构化表示与 LLM 集成提供了可能路径。

## 评分

- ⭐⭐⭐ 新颖性：评估框架系统但不涉及新模型或新方法，更偏实证分析
- ⭐⭐⭐⭐ 实用性：结论直接可用于指导 prompt 设计和结构化信息利用策略
- ⭐⭐⭐⭐ 实验充分度：6 种任务 × 多变量的全面实验矩阵，置信区间完整
- ⭐⭐⭐⭐ 写作清晰度：结构规范，可视化丰富，方法描述明确

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)

</div>

<!-- RELATED:END -->
