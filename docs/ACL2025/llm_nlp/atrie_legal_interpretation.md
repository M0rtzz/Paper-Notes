---
title: >-
  [论文解读] ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation
description: >-
  [ACL 2025][LLM/NLP][Legal Interpretation] 提出 ATRIE 框架，模拟法学专家的教义法学研究流程，利用 LLM 自动从案例库中检索相关信息、生成法律概念解释并评估解释质量，消除对人类法律专家的依赖。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "Legal Interpretation"
  - "RAG"
  - "Legal Concept Entailment"
  - "Doctrinal Legal Research"
  - "LLM"
---

# ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation

**会议**: ACL 2025  
**arXiv**: [2501.01743](https://arxiv.org/abs/2501.01743)  
**代码**: [GitHub](https://github.com/lkc233/ATRIE)  
**领域**: 法律NLP / 检索增强生成  
**关键词**: Legal Interpretation, RAG, Legal Concept Entailment, Doctrinal Legal Research, LLM  

## 一句话总结

提出 ATRIE 框架，模拟法学专家的教义法学研究流程，利用 LLM 自动从案例库中检索相关信息、生成法律概念解释并评估解释质量，消除对人类法律专家的依赖。

## 研究背景与动机

**研究问题：** 法律解释——特别是模糊法律概念的解释——对法律实践至关重要，但现有方法严重依赖人类法律专家，存在耗时长、更新不及时和主观不完整的问题。

**现有方法的不足：** 先前研究（如 Savelka et al. 2023）利用 GPT-4 结合人工标注的案例关键句来解释法律概念，但仍然依赖法律专家 (1) 人工标注案例中与概念相关的有价值句子，以及 (2) 评估生成解释的质量，无法实现自动化扩展。

**核心动机：** 受教义法学研究 (Doctrinal Legal Research) 方法启发，模拟法律专家从大量历史案例中阅读、提取和总结的工作流程，构建一个完全自动化的法律概念解释框架。

## 方法详解

### 整体框架

ATRIE 包含两个核心模块：**法律概念解释器 (Legal Concept Interpreter)** 和 **法律概念解释评估器 (Legal Concept Interpretation Evaluator)**。解释器使用 RAG 框架从案例数据库检索信息并生成解释；评估器基于下游任务 Legal Concept Entailment (LCE) 的性能变化来自动衡量解释质量。

### 关键设计

1. **三阶段案例检索与提取：** (1) 精确字符串匹配从中国裁判文书网数据库中检索提及目标概念的案例集 $\mathcal{D}_0$；(2) LLM 过滤出法院观点中对概念进行详细讨论的相关案例 $\mathcal{D}_1$，并提取适用/不适用的原因；(3) 正负样本平衡采样构建最终数据集 $\mathcal{D}$ 和原因集 $\mathcal{R}$。

2. **结构化解释生成：** 输入法律条文、模糊概念、原因集和示例解释，要求 LLM 输出包含三部分的解释：分析 (Analysis，基本含义与适用条件)、案例举例 (Case Examples，正面与反面案例) 和裁判裁量 (Judicial Discretion，法官灵活适用的判断标准)。

3. **Legal Concept Entailment (LCE) 评估任务：** 给定案件事实描述，判断模糊概念是否适用并给出原因。将不同解释作为参考输入 LLM 进行 LCE 任务，分类准确率的变化作为解释质量的代理指标。评估涵盖分类任务（Accuracy, Macro-F1）和理由生成任务（GPT-4o 一致性评分 1-10）。

### 损失函数/优化目标

评估器不涉及训练损失，而是通过固定 LLM 在 LCE 下游任务上的表现变化来衡量解释质量；解释器使用 Qwen2.5-72B，温度设置为 0.9 以鼓励多样化输出。

## 实验

### 主实验结果

| 方法 | Acc (72B) | Ma-F (72B) | CS (72B) | Acc (14B) | Ma-F (14B) | CS (14B) |
|------|-----------|------------|----------|-----------|------------|----------|
| Random | 51.66 | 50.32 | / | 51.66 | 50.32 | / |
| Zero-Shot | 71.38 | 61.42 | 5.658 | 70.92 | 59.88 | 5.525 |
| Chain-of-Thought | 71.95 | 63.46 | 5.717 | 71.52 | 61.01 | 5.666 |
| Judicial Interp. | 72.10 | 66.54 | 5.573 | 70.92 | 65.23 | 5.347 |
| Expert Interp. | 72.13 | 65.30 | 5.630 | 71.95 | 66.01 | 5.581 |
| Direct Interp. | 72.35 | 67.18 | 5.642 | 72.72 | 66.90 | 5.677 |
| **ATRIE** | **75.03** | **70.87** | **5.946** | **74.50** | **70.39** | **5.840** |

### 消融实验

| 检索方式 | Ma-F (14B) | CS |
|----------|------------|------|
| No Retrieval | 66.90 | 5.677 |
| String Match | 69.04 | 5.772 |
| + Filter | 69.60 | 5.817 |
| + Filter + Balance (ATRIE) | 70.39 | 5.840 |

| 解释组件消融 | Macro-F1 |
|-------------|----------|
| w/o Example Cases | 67.41 |
| w/o Analysis | 70.43 |
| w/o Judicial Discretion | 70.69 |
| ATRIE (完整) | 70.87 |

### 关键发现

- ATRIE 在几乎所有指标上显著优于人类法律专家编写的解释（Expert/Judicial Interpretation），体现了 LLM 在大规模案例分析上的优势。
- 案例举例 (Example Cases) 是解释中最关键的组成部分，移除后 Macro-F1 下降 3.46 个点。
- 通用 LLM (Qwen2.5-72B) 在法律解释任务上显著优于专用法律 LLM (Farui-plus)，归因于其更强的长文本理解和生成能力。
- 输入更多案例可持续提升解释质量，符合法律专家的实际工作经验。

## 亮点

- 完整模拟教义法学研究的自动化流程，从检索到生成到评估形成闭环。
- 提出 Legal Concept Entailment 作为客观可复现的法律概念解释质量评估方法，替代主观人工评估。
- 人工评估表明生成解释在全面性和可读性上优于专家解释，仅在精确性上略有差距。

## 局限性

- 仅在中国法律体系和中文案例上验证，跨法律体系的泛化性未知。
- 选取的 16 个法律概念虽具代表性，但覆盖范围有限。
- 精确字符串匹配检索策略在法律术语有同义表达时可能遗漏相关案例。

## 相关工作

- **法律解释：** Savelka et al. (2023) 利用 GPT-4 基于人工标注句子解释法律概念；Coan & Surden (2024) 直接用 GPT 生成宪法解释。
- **教义法学研究自动化：** Yung-chin Su (2024) 建议 Legal AI 可替代人工进行案例阅读和理论提取。
- **RAG 用于法律：** Lewis et al. (2020) 的 RAG 框架被应用于法律领域，增强 LLM 对具体案例的理解。

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 7 |
| 实用性 | 8 |
| 实验充分度 | 8 |
| 写作质量 | 8 |
| 总体评分 | 7.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery](learning_from_litigation_graphs_and_llms_for_retrieval_and_reasoning_in_ediscove.md)
- [\[ACL 2025\] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA](hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)
- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)
- [\[ICML 2026\] Stop Automating Peer Review Without Rigorous Evaluation](../../ICML2026/llm_nlp/stop_automating_peer_review_without_rigorous_evaluation.md)
- [\[ACL 2025\] Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility](leveraging_human_production-interpretation_asymmetries_to_test_llm_cognitive_pla.md)

</div>

<!-- RELATED:END -->
