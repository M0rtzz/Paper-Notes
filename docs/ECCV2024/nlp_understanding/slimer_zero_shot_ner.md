---
title: >-
  [论文解读] SLIMER: Show Less, Instruct More - Enriching Prompts with Definitions and Guidelines for Zero-Shot NER
description: >-
  [ECCV 2024][NLP理解][零样本NER] SLIMER 通过在提示中注入实体定义和标注指南来增强 LLM 的零样本命名实体识别能力，仅用 391 个实体类别训练即可在从未见过的实体标签上达到与使用 13000+ 实体类别训练的 SOTA 方法相当的性能。 领域现状：指令微调 LLM 用于 NER 已成为主流方向…
tags:
  - "ECCV 2024"
  - "NLP理解"
  - "零样本NER"
  - "指令微调"
  - "LLM"
  - "实体定义"
  - "提示工程"
---

# SLIMER: Show Less, Instruct More - Enriching Prompts with Definitions and Guidelines for Zero-Shot NER

**会议**: ECCV 2024  
**arXiv**: [2407.01272](https://arxiv.org/abs/2407.01272)  
**代码**: [HuggingFace](https://huggingface.co/expertai/SLIMER)  
**领域**: NLP理解 / 命名实体识别  
**关键词**: 零样本NER, 指令微调, LLM, 实体定义, 提示工程

## 一句话总结

SLIMER 通过在提示中注入实体定义和标注指南来增强 LLM 的零样本命名实体识别能力，仅用 391 个实体类别训练即可在从未见过的实体标签上达到与使用 13000+ 实体类别训练的 SOTA 方法相当的性能。

## 研究背景与动机

**领域现状**：指令微调 LLM 用于 NER 已成为主流方向，代表工作包括 InstructUIE、UniNER、GoLLIE 和 GNER，它们在 OOD（域外）零样本 NER 上表现出色。

**现有痛点**：(1) 现有方法在大量实体类别（如 UniNER 用 13020 个）上微调，训练集和测试集的实体标签高度重叠，"零样本"其实是"域外"而非"真正未见实体"；(2) 对真正从未见过的实体标签（unseen NE），大多数方法表现不佳；(3) 只有 GoLLIE 尝试用 Python 类定义来指导未见实体的识别，但需要大量人工编写定义。

**核心矛盾**：现有方法依赖大量重叠的训练实体标签来"记忆"实体类型，而非真正理解实体概念。当遇到全新实体类型时，缺乏泛化能力。

**本文目标**：用更少的训练样本和更少的实体类别，通过在提示中注入定义和指南来实现真正的零样本 NER。

**切入角度**：与其让模型"记住"更多实体类型，不如"教会"模型如何根据定义理解新实体类型——"Show Less, Instruct More"。

**核心 idea**：用 GPT 生成的实体定义和标注指南丰富提示内容，让模型学会根据语义描述识别实体，而非记忆实体标签。

## 方法详解

### 整体框架

SLIMER 基于 decoder-only LLM 架构进行指令微调。输入提示包含三部分：(1) 任务指令和待标注文本；(2) 目标实体类型的定义（entity definition）；(3) 标注指南（annotation guidelines），指导模型如何区分相似实体。每次推理针对一个实体类型生成该类型的所有实体提及。

### 关键设计

1. **定义增强提示（Definition-enriched Prompt）**:

    - 功能：为每个实体类型提供语义定义，使模型理解"什么是这种实体"
    - 核心思路：对每个命名实体标签（如"Algorithm"），用 GPT 生成一段简短定义（如"An algorithm is a step-by-step procedure for solving a problem..."），嵌入到提示模板中。模型学会将定义与文本中的 span 进行语义匹配
    - 设计动机：传统方法仅用实体标签名（如"PER"、"ORG"），语义信息不足以泛化到全新标签；定义提供了类型级别的语义锚点

2. **标注指南（Annotation Guidelines）**:

    - 功能：指导模型如何标注边界情况和区分相似实体
    - 核心思路：为每个实体类型额外提供标注规则（如"Include the full name including titles"、"Do not confuse product names with organization names"），这些指南同样由 GPT 生成
    - 设计动机：仅有定义不足以处理歧义情况，指南提供操作层面的指导，类似于人类标注员使用的标注手册

3. **少类别训练策略**:

    - 功能：用更少的实体类别训练实现更强的泛化
    - 核心思路：训练集仅包含 391 个实体类别（vs UniNER 的 13020 个），且刻意选择与测试集标签重叠最少的训练数据。还使用合成数据增强（GPT 生成的 NER 样本）
    - 设计动机：减少标签重叠迫使模型依赖定义和指南来理解实体，而非记忆训练中见过的标签

### 损失函数 / 训练策略

标准的 autoregressive language modeling loss。推理时每个实体类型独立调用（推理成本 $|\mathcal{X}| \times |\mathcal{Y}|$），支持文档级 NER 和嵌套实体。

## 实验关键数据

### 主实验

| 模型 | MIT (avg) | CrossNER (avg) | BUSTER (unseen NE) |
|------|----------|---------------|-------------------|
| UniNER | 52.3 | 61.2 | 18.5 |
| GoLLIE | 48.7 | 58.4 | 35.2 |
| GNER | 50.1 | 59.8 | 15.3 |
| **SLIMER** | **49.8** | **57.6** | **38.7** |

训练实体类别：SLIMER 391 vs UniNER/GNER 13020。

### 消融实验

| 配置 | CrossNER F1 | BUSTER F1 | 说明 |
|------|-----------|----------|------|
| SLIMER (定义+指南) | 57.6 | 38.7 | 完整模型 |
| w/o 定义和指南 (baseline) | 51.2 | 25.3 | 去掉定义指南大幅下降 |
| 仅定义 | 55.1 | 33.4 | 定义贡献最大 |
| 仅指南 | 53.8 | 30.1 | 指南也有显著贡献 |

### 关键发现

- 定义和指南在**未见实体**上的提升最为显著（BUSTER 上 +13.4 F1），证明模型学会了根据语义描述识别新实体
- 在 OOD 数据上（MIT、CrossNER），SLIMER 用 1/33 的训练实体类别达到了 SOTA 的 95%+ 性能
- 学习曲线显示，有定义的版本收敛更快更稳定
- 合成数据对于弥补训练实体类别少的劣势有帮助

## 亮点与洞察

- **"以少胜多"的训练哲学**：颠覆了"更多训练类别=更强泛化"的常识。用 391 vs 13020 个实体类别，靠定义弥补信息差，思路清晰
- **低成本指南生成**：用 GPT 自动生成定义和指南，几乎零人工成本（vs GoLLIE 需要为每个类别手写 Python 类）
- **真正的零样本评估协议**：首次系统性地区分 OOD 零样本和 unseen NE 零样本，为 NER 评估建立了更严格的标准

## 局限与展望

- 推理成本与实体类型数量线性增长（每个类型需独立推理一次），大量实体类型场景下效率不佳
- GPT 生成的定义和指南质量不可控，可能引入偏差
- 仅在英文 NER 上评估，多语言场景待验证
- 可探索用检索增强（RAG）动态获取实体定义，进一步降低训练依赖

## 相关工作与启发

- **vs UniNER/GNER**: 依赖大量实体类别暴力覆盖测试标签，训练-测试重叠高；SLIMER 证明了少类别+定义的路径可行性
- **vs GoLLIE**: 同样用定义引导，但需要高成本的人工 Python 类编写；SLIMER 用 GPT prompt 自动化了这一过程
- **vs GLiNER**: 编码器架构，不能处理嵌套实体；SLIMER 基于生成范式支持嵌套和文档级

## 评分

- 新颖性: ⭐⭐⭐⭐ "少类别+定义"的思路新颖，但定义增强提示的idea不算全新
- 实验充分度: ⭐⭐⭐⭐ OOD 和 unseen NE 双评估，消融清晰
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，对比公平详细
- 价值: ⭐⭐⭐⭐ 为低资源 NER 和真正零样本场景提供实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SAM-NER: Semantic Archetype Mediation for Zero-Shot Named Entity Recognition](../../ACL2026/nlp_understanding/sam-ner_semantic_archetype_mediation_for_zero-shot_named_entity_recognition.md)
- [\[ACL 2026\] DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition](../../ACL2026/nlp_understanding/diziner_disagreement-guided_instruction_refinement_via_pilot_annotation_simulati.md)
- [\[ACL 2026\] Refining and Reusing Annotation Guidelines for LLM Annotation](../../ACL2026/nlp_understanding/refining_and_reusing_annotation_guidelines_for_llm_annotation.md)
- [\[ACL 2025\] Towards a More Generalized Approach in Open Relation Extraction](../../ACL2025/nlp_understanding/generalized_open_relation_extract.md)
- [\[ACL 2025\] In the LLM Era, Word Sense Induction Remains Unsolved](../../ACL2025/nlp_understanding/in_the_llm_era_word_sense_induction_remains_unsolved.md)

</div>

<!-- RELATED:END -->
