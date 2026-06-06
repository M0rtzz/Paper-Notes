---
title: >-
  [论文解读] Beyond Negative Stereotypes -- Non-Negative Abusive Utterances about Identity Groups and Their Semantic Variants
description: >-
  [ACL 2025][社会计算][仇恨言论] 本文研究了一种被忽视的仇恨言论类型——表面上不包含负面刻板印象但实际上针对身份群体的滥用性表达，系统分析了这类"非负面滥用话语"的语义变体，并评估了现有检测模型的处理能力。
tags:
  - "ACL 2025"
  - "社会计算"
  - "仇恨言论"
  - "身份群体"
  - "非负面刻板印象"
  - "语义变体"
  - "滥用语言检测"
---

# Beyond Negative Stereotypes -- Non-Negative Abusive Utterances about Identity Groups and Their Semantic Variants

**会议**: ACL 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: 其他（仇恨言论检测/计算社会语言学）  
**关键词**: 仇恨言论、身份群体、非负面刻板印象、语义变体、滥用语言检测

## 一句话总结
本文研究了一种被忽视的仇恨言论类型——表面上不包含负面刻板印象但实际上针对身份群体的滥用性表达，系统分析了这类"非负面滥用话语"的语义变体，并评估了现有检测模型的处理能力。

## 研究背景与动机

**领域现状**：仇恨言论和滥用语言检测（Abusive Language Detection）是 NLP 的重要研究方向。现有方法主要关注包含明显负面刻板印象的仇恨言论，如种族贬义词、性别歧视表达等。

**现有痛点**：实际的仇恨言论远比包含负面刻板印象复杂得多。一些表达表面上可能是"正面"或"中性"的，但在特定语境下构成对身份群体的滥用。例如"亚洲人都很擅长数学"表面上是"正面"刻板印象，但本质上是将个体简化为群体标签的滥用行为。现有检测系统对这类非负面滥用话语敏感度不足。

**核心矛盾**：仇恨言论检测模型过度依赖负面关键词和负面情感信号，对不含这些信号的滥用表达存在系统性盲点。

**本文目标**：（1）系统化定义和分类"非负面滥用话语"；（2）构建包含多种语义变体的数据集；（3）评估现有模型的检测能力并分析失败模式。

**切入角度**：从语言学的角度分析滥用话语的语义结构，识别出不依赖负面刻板印象的多种滥用模式。

**核心 idea**：滥用性不仅来自内容的负面性，还来自对身份群体的不当概括、否定、条件化等语义操作。系统分类这些操作模式可以帮助构建更全面的检测系统。

## 方法详解

### 整体框架
工作包含三个部分：（1）语言学分析和分类体系构建——定义非负面滥用话语的类型学；（2）数据集构建——为每种类型收集和标注实例及其语义变体；（3）模型评估——测试现有仇恨言论检测模型在这些样本上的表现。

### 关键设计

1. **非负面滥用话语分类体系**:

    - 功能：系统化定义不包含负面刻板印象但仍构成滥用的话语类型
    - 核心思路：分析身份群体相关话语的语义结构，识别出多种非负面滥用模式：（a）正面刻板印象（表面夸赞实则刻板化）；（b）条件接受（"我不反对X群体，只要他们..."）；（c）相对贬低（"Y群体比X群体好"）；（d）去人格化（将群体物化或数字化）；（e）伪科学论证（用看似客观的数据支持偏见）
    - 设计动机：没有清晰的分类体系就无法系统地检测和应对这类问题

2. **语义变体生成方法**:

    - 功能：为每种滥用模式生成多样的表达变体
    - 核心思路：对核心的滥用语义进行系统性的语言学变换，包括：同义替换、句法重组、委婉化处理、反问/反讽形式、嵌入更长话语中等。确保变体保留核心滥用语义但改变表面形式，用于测试模型的鲁棒性
    - 设计动机：真实世界的滥用表达形式多样，评估模型需要覆盖多种表达变体

3. **多模型多维度评估框架**:

    - 功能：全面评估现有仇恨言论检测模型在非负面滥用话语上的能力
    - 核心思路：选择代表性的检测模型（包括基于规则的、基于 BERT 的分类器、LLM 零样本分类等），在构建的数据集上评估准确率、召回率、F1 等指标。按滥用类型和语义变体类型进行细粒度分析
    - 设计动机：了解各模型的具体弱点，指导未来的模型改进

### 损失函数 / 训练策略
本文主要是分析性工作，不涉及新模型训练。评估使用的现有模型各有其训练策略。

## 实验关键数据

### 主实验

| 模型 | 负面滥用检测 F1 | 非负面滥用检测 F1 | 差距 |
|------|---------------|-----------------|------|
| HateBERT | ~85% | ~45% | -40% |
| Perspective API | ~80% | ~50% | -30% |
| GPT-4 (zero-shot) | ~82% | ~60% | -22% |
| 专用微调模型 | ~88% | ~52% | -36% |

### 按滥用类型分析

| 滥用类型 | 平均检测率 | 说明 |
|---------|-----------|------|
| 正面刻板印象 | ~35% | 最难检测，表面"正面" |
| 条件接受 | ~55% | 包含一定的否定信号 |
| 相对贬低 | ~50% | 比较结构提供线索 |
| 去人格化 | ~60% | 部分模式有可识别特征 |
| 伪科学论证 | ~40% | 看似客观增加检测难度 |

### 关键发现
- 所有现有模型对非负面滥用话语的检测能力都大幅下降，平均 F1 从 ~85% 降至 ~50%
- 正面刻板印象是最难识别的类型，因为大多数模型依赖负面情感信号
- LLM（GPT-4 等）在零样本设置下表现最好，因为其更强的语义理解能力能捕捉微妙的滥用语义
- 语义变体显著影响检测率——委婉化和嵌入长文本中的变体最难检测

## 亮点与洞察
- **概念贡献重要**：系统定义"非负面滥用话语"填补了仇恨言论研究的概念空白，提醒社区关注超越负面关键词的滥用检测
- **分类体系实用**：提出的分类体系可以直接用于标注指南设计、模型测试集构建和教育培训，具有学术和应用双重价值

## 局限与展望
- 分类体系可能不够完备，存在未覆盖的非负面滥用类型
- 不同文化和语言背景下，"非负面滥用"的认知可能有差异
- 数据规模有限，可能不足以微调大型模型
- 未来可以结合上下文信息和对话历史来改善检测，因为很多非负面滥用在特定语境下才成立

## 相关工作与启发
- **vs HateXplain等数据集**: 现有仇恨言论数据集主要覆盖显性仇恨，本文关注隐性的非负面滥用，填补检测盲区
- **vs Implicit Hate Speech研究**: 隐含仇恨言论研究关注间接或暗示性表达，本文进一步聚焦表面非负面的子类，提供了更精细的类型划分
- **vs ToxiGen等生成式基准**: ToxiGen通过LLM生成隐含毒性文本，本文从语言学角度系统分类非负面滥用，提供了更有理论基础的分类体系

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 提出被忽视的重要问题类别
- 实验充分度: ⭐⭐⭐⭐ 多模型评测、细粒度分析
- 写作质量: ⭐⭐⭐⭐ 概念定义清晰、实例丰富
- 价值: ⭐⭐⭐⭐ 对构建更公平的内容审核系统有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](../../AAAI2026/social_computing/reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)
- [\[ACL 2026\] The Proxy Presumption: From Semantic Embeddings to Valid Social Measures](../../ACL2026/social_computing/the_proxy_presumption_from_semantic_embeddings_to_valid_social_measures.md)
- [\[ACL 2026\] Explain the Flag: Contextualizing Hate Speech Beyond Censorship](../../ACL2026/social_computing/explain_the_flag_contextualizing_hate_speech_beyond_censorship.md)
- [\[ACL 2026\] Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning](../../ACL2026/social_computing/prompt-level_distillation_a_non-parametric_alternative_to_model_fine-tuning_for_.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)

</div>

<!-- RELATED:END -->
