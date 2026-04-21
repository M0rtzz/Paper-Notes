---
title: >-
  [论文解读] Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge
description: >-
  [ACL 2025][LLM/NLP][signal temporal logic] 提出 STL-DivEn 数据集（16K样本）和 KGST（知识引导的 STL 转换）框架，通过"生成-精炼"两阶段流程将自然语言转换为信号时序逻辑（STL），在 STL Formula Accuracy 上达到 0.5587，显著超过 GPT-4（0.4733）和 DeepSeek（0.4790）。
tags:
  - ACL 2025
  - LLM/NLP
  - signal temporal logic
  - NL-to-STL
  - formal specification
  - knowledge-guided refinement
  - dataset construction
---

# Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge

**会议**: ACL 2025  
**arXiv**: [2505.20658](https://arxiv.org/abs/2505.20658)  
**代码**: https://github.com/YueFang0618/STL-DivEn  
**领域**: LLM/NLP - 形式化规约  
**关键词**: signal temporal logic, NL-to-STL, formal specification, knowledge-guided refinement, dataset construction

## 一句话总结
提出 STL-DivEn 数据集（16K样本）和 KGST（知识引导的 STL 转换）框架，通过"生成-精炼"两阶段流程将自然语言转换为信号时序逻辑（STL），在 STL Formula Accuracy 上达到 0.5587，显著超过 GPT-4（0.4733）和 DeepSeek（0.4790）。

## 研究背景与动机

**领域现状**：信号时序逻辑（STL）是信息物理系统（如自动驾驶、机器人控制）中广泛使用的形式化规约语言，能够描述实时和实值约束。手动编写 STL 公式耗时且容易出错，因此自动将自然语言转换为 STL 是一个具有重要实用价值的研究方向。

**现有痛点**：(1) NL-STL 数据集极度匮乏，DeepSTL 通过模板随机采样生成，多样性严重不足；(2) 基于 Transformer 的模型在处理复杂嵌套时序约束时表现不佳；(3) 即使是 GPT-4 和 DeepSeek 等先进模型，在 NL-to-STL 转换中的准确率也很低。

**核心矛盾**：NL-to-STL 转换需要精确理解时序语义和数值约束，但现有模型既缺乏高质量训练数据，也缺乏有效利用外部知识的机制来提升转换精度。

**本文目标** (1) 构建高质量、多样化的 NL-STL 数据集；(2) 设计有效的转换框架，利用外部知识提升 STL 生成的准确性。

**切入角度**：基于聚类引导的数据增强构建多样化数据集，配合"微调生成+检索增强精炼"两阶段转换框架。

**核心 idea**：用聚类引导 GPT-4 生成多样化 NL-STL 数据集，再通过微调 LLM 生成初步 STL + 检索相似样例由 GPT-4 精炼的 generate-then-refine 流程提升转换准确度。

## 方法详解

### 整体框架
分为两大部分：(1) STL-DivEn 数据集构建——从人工种子集出发，通过聚类选取代表样本引导 GPT-4 生成新 NL-STL 对，经规则过滤和人工验证后扩展数据集；(2) KGST 转换框架——先在数据集上微调 LLaMA 3-8B 生成初步 STL，再从数据集中检索 Top-K 相似样本作为外部知识，由 GPT-4 评估并精炼初步 STL。

### 关键设计

1. **聚类引导的多样化数据增强（Diversity-Guided Augmentation）**:

    - 功能：从 120 个人工种子对出发，通过 K-means 聚类选出 5 个代表性种子，用这些种子引导 GPT-4 生成新的 NL-STL 对
    - 核心思路：使用 Sentence-Transformers 将 NL-STL 对映射到高维向量空间进行聚类，选取聚类中心作为 GPT-4 的 in-context 示例。生成的新对通过语法检查、Rouge 评分过滤（< 0.5 视为足够多样）和人工验证后加入种子集和数据集
    - 设计动机：直接让 GPT-4 生成的数据会严重模仿提供的示例，聚类选取代表性种子可最大化输出多样性

2. **Generate-then-Refine 转换流程**:

    - 功能：第一步在 STL-DivEn 上微调 LLaMA 3-8B 生成初步 STL 公式；第二步从外部知识库中检索 Top-5 最相似的 NL-STL 对，与原始自然语言和初步 STL 一起送入 GPT-4 进行精炼
    - 核心思路：微调模型擅长捕捉数据分布但在复杂嵌套逻辑上不够精确；GPT-4 有强大的推理能力但缺乏领域知识。两者结合取长补短
    - 设计动机：Self-Refine（GPT-4 自我精炼）反而降低了性能，说明精炼需要外部知识参考而非仅靠模型内部能力

3. **多层质量保证机制**:

    - 功能：对生成的 NL-STL 对执行两阶段过滤——(1) 基于 STL 语法规则的自动检查；(2) 与种子集计算 Rouge 分数确保多样性；(3) 7 名标注员进行 2 个月的人工语义一致性验证
    - 核心思路：语法检查确保 STL 公式格式正确，Rouge 过滤防止重复，人工验证保证 NL 和 STL 之间的语义一致性
    - 设计动机：LLM 生成的 NL-STL 对可能存在语法错误、与种子集重复、或语义不一致等问题

## 实验关键数据

### 主实验（STL-DivEn 数据集）

| 模型 | STL Formula Acc. | Template Acc. | BLEU |
|------|-----------------|---------------|------|
| DeepSTL | 0.1986 | 0.1883 | 0.0293 |
| GPT-3.5 | 0.3018 | 0.3034 | 0.0424 |
| GPT-4 | 0.4733 | 0.4741 | 0.0831 |
| DeepSeek | 0.4790 | 0.4825 | 0.0791 |
| GPT-4 + Self-Refine | 0.4422 | 0.4466 | 0.0521 |
| **KGST** | **0.5587** | **0.5627** | **0.2142** |

### 消融实验（DeepSTL 数据集 — 跨数据集泛化）

| 模型 | STL Formula Acc. | Template Acc. | BLEU |
|------|-----------------|---------------|------|
| DeepSTL | 0.2002 | 0.2916 | 0.3332 |
| GPT-4 | 0.2262 | 0.3048 | 0.2881 |
| DeepSeek | 0.2537 | 0.3254 | 0.3982 |
| **KGST** | **0.4538** | **0.4939** | **0.5686** |

### 关键发现
- KGST 在两个数据集上一致取得最佳性能，证明框架的鲁棒性和跨数据集泛化能力
- Self-Refine 精炼后性能反而下降（0.4733 → 0.4422），说明 STL 精炼必须依赖外部知识参考
- STL-DivEn 的 N-gram 多样性（2.386）远高于 DeepSTL（1.474），子公式数量（14.66 vs 6.98）也显著更多
- 人工评估中 KGST 正确率 62.4%（STL-DivEn）/ 54.6%（DeepSTL），均为最高
- STL-DivEn 的自然语言描述词汇量（4,954 unique words）远超 DeepSTL（265）

## 亮点与洞察
- 数据集构建方法可推广到其他形式化语言的数据生成——聚类引导增强是保证多样性的通用策略
- Self-Refine 的失败揭示了一个重要洞察：形式化语言转换需要精确的参考知识，模型自身"直觉"不够可靠

## 局限与展望
- 依赖 GPT-4 进行精炼，成本较高且限制了部署的独立性
- 种子集仅覆盖自动驾驶、机器人和电子三个领域，可能不够全面
- 生成阶段使用较小的 LLaMA 3-8B，更大模型可能进一步提升初步 STL 质量

## 相关工作与启发
- **vs DeepSTL (He et al., 2022)**：后者用模板随机采样生成数据，多样性严重不足；本文用聚类引导+人工验证确保质量和多样性
- **vs NL2TL (Chen et al., 2023)**：后者用 LLM 创建 NL-TL 数据集微调 T5，但未使用外部知识增强精炼
- **vs DialogueSTL (Mohammadinejad et al., 2024)**：后者通过用户交互和强化学习进行 STL 转换，但依赖用户反馈增加了使用复杂度

## 评分
- 新颖性: ⭐⭐⭐⭐ Generate-then-Refine 框架和聚类引导数据增强有创新性
- 实验充分度: ⭐⭐⭐⭐ 两个数据集+自动评估+人工评估，但消融实验可以更细致
- 写作质量: ⭐⭐⭐⭐ 结构完整，STL 形式化定义清晰
- 价值: ⭐⭐⭐⭐ 数据集和框架对形式化规约自动生成领域有实际推动作用

<!-- RELATED:START -->

## 相关论文

- [Internal and External Impacts of Natural Language Processing Papers](internal_and_external_impacts_of_natural_language_processing_papers.md)
- [What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)
- [Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs](palm_a_culturally_inclusive_and_linguistically_diverse_dataset_for_arabic_llms.md)
- [How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs](how_llms_comprehend_temporal_meaning_in_narratives_a_case_study_in_cognitive_eva.md)

<!-- RELATED:END -->
