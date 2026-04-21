---
title: >-
  [论文解读] MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance
description: >-
  [ACL 2025][多文档推理] 提出 MDBench，一个通过「结构化知识→LLM 辅助增强→自然文本生成」管线合成的多文档推理 QA 基准，可控地注入跨文档依赖，对前沿 LLM 构成显著挑战（最佳模型 EM 仅~60%）。
tags:
  - ACL 2025
  - 多文档推理
  - 合成数据
  - 知识引导生成
  - LLM 评测
  - QA 基准
---

# MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance

**会议**: ACL 2025  
**arXiv**: [2506.14927](https://arxiv.org/abs/2506.14927)  
**代码**: [GitHub](https://github.com/jpeper/MDBench)  
**领域**: NLP / 多文档推理 / 评估基准  
**关键词**: 多文档推理, 合成数据, 知识引导生成, LLM 评测, QA 基准

## 一句话总结

提出 MDBench，一个通过「结构化知识→LLM 辅助增强→自然文本生成」管线合成的多文档推理 QA 基准，可控地注入跨文档依赖，对前沿 LLM 构成显著挑战（最佳模型 EM 仅~60%）。

## 研究背景与动机

多文档推理是 LLM 应用中一个核心且日益重要的能力——模型需要综合多个文本来源的信息进行推理和回答。然而，当前 LLM 评估存在三个关键问题：

**基准稀缺**：尽管 LLM 处理长上下文的能力在快速提升，专门测试多文档推理的基准非常少。现有数据集如 HotPotQA、MuSiQue 主要面向多跳 QA，但往往依赖公开数据且标注成本极高。

**数据污染风险**：静态、手工标注的基准容易被 LLM 在训练中"见过"，导致评估失效。

**标注成本**：多文档场景的标注涉及长文本理解，人工标注成本极高，难以规模化。

作者提出的核心思路是：不直接从自然文本出发，而是以**压缩的结构化知识**（表格数据）为种子，通过 LLM 辅助编辑注入推理依赖，再将结构化知识转化为自然文本文档集。这种"先结构化后生成"的方式同时解决了可控性、新颖性和可扩展性问题。

## 方法详解

### 整体框架

MDBench 的生成管线包含四个步骤：

1. **获取种子知识** → 2. **知识增强**（注入推理依赖）→ 3. **生成自然文本** → 4. **自动化质量验证**

### 关键设计

1. **种子知识来源（Step 1）**：使用 TabFact 数据集中的 Wikipedia 表格。选择 5-17 行、3-9 列的表格。每一行最终对应生成文档集中的一篇文档，表格的列提供结构化属性，行间关系自然对应跨文档依赖。

2. **知识增强（Step 2）**：这是管线的核心创新。使用 GPT-4o 对表格进行编辑，注入五类多文档推理挑战：

    - **多跳推理**：需要跨多行（文档）的链式推演
    - **数值推理**：涉及数值计算和比较
    - **时序推理**：处理时间依赖和顺序关系
    - **知识聚合**：需要对齐、比较或对比多源信息
    - **软推理**：需要在不确定条件下做推断（如跨文档实体链接）
   
   增强过程包括两类 demonstrations：(a) 推理技能演示（每种技能展示简单和困难的例子），(b) 知识编辑演示（展示如何将简单表格编辑为复杂 QA 例子）。同时在此步骤生成对应的问答对。

3. **文档集生成（Step 3）**：将增强后的表格逐行独立转化为自然语言文档。每个文档的生成以增强表格、列名和对应行内容为条件，确保生成文档保持逻辑一致性。结构化表格到自然文本的长度约增加 9 倍（256 → 2397 tokens）。

4. **自动质量验证（Step 4）**：两层验证机制：

    - 定向一致性检查：验证每个组件（种子表、编辑计划、编辑后表、文档集）是否符合指令
    - **Oracle 自一致性检查**：在三种"已知答案"的变体条件下（原始表+编辑计划 / 生成表 / 生成文档集）让 GPT-4o 回答问题，只有三个变体答案一致时才保留该样本。该过滤保留了约 32% 的生成样本。

### 数据集统计

最终基准包含 1000 个多文档 QA 样本（300 个人工验证，700 个机器验证）。人工验证的有效率达到 87%。每个文档集平均包含 8.31 个文档，平均文档长度 268 tokens，总上下文平均 2397 tokens。

## 实验关键数据

### 主实验：多文档推理（Exact Match）

| 模型 | Zero-shot | ZS CoT | 1-shot | 1-shot CoT | 总体 |
|------|-----------|--------|--------|------------|------|
| LLaMA-3-8B | 42.7 | 43.1 | 39.3 | 38.9 | 41.0 |
| LLaMA-3-70B | 51.2 | 50.2 | 45.0 | 49.3 | 48.9 |
| GPT-3.5-Turbo | 49.8 | 37.4 | 44.5 | 36.5 | 42.1 |
| Claude-3.5-Sonnet | 59.2 | 56.4 | 58.3 | 55.9 | 57.5 |
| Gemini-2.5-Flash | 57.8 | 58.3 | 58.8 | 60.2 | 58.8 |
| **GPT-4o** | **59.7** | **62.1** | 59.7 | 58.3 | **60.0** |
| GPT-o1 | 57.3 | 60.7 | 59.7 | 59.2 | 59.2 |

### 消融实验：文档 vs 表格推理

| 模型 | 文档 EM | 表格 EM | 差值 |
|------|---------|---------|------|
| GPT-4o | 60.0 | 71.2 | -11.2 |
| GPT-o1 | 59.2 | 67.9 | -8.7 |
| Claude-3.5-Sonnet | 57.5 | 66.9 | -9.4 |
| LLaMA-3-70B | 48.9 | 52.4 | -3.5 |
| LLaMA-3-8B | 41.0 | 35.8 | +5.2 |

文档排序消融（GPT-4o Zero-shot EM）：

| 条件 | GPT-4o | GPT-3.5 |
|------|--------|---------|
| 原始 | 59.7 | 49.8 |
| 去除分隔符 | 55.5 | 45.0 |
| 打乱顺序 | 53.1 | 39.3 |
| 去除分隔符+打乱 | 50.2 | 41.7 |

### 关键发现

1. **MDBench 对前沿模型构成显著挑战**：最佳 EM 仅 60%（GPT-4o），最佳 Accuracy 82.2%（GPT-o1），说明多文档推理远未被解决。
2. **表面形式影响推理**：从结构化表格到自然文本文档，所有模型性能都显著下降（GPT-4o 降低 11.2% EM），说明多文档推理不仅受底层逻辑复杂度影响，还受表面形式的干扰。
3. **文档排序敏感**：去除分隔符和打乱文档顺序都会导致性能下降，证实 MDBench 中存在真实的跨文档时序依赖。
4. **CoT 对强模型有帮助，对弱模型效果有限**：GPT-4o 和 Gemini-2.5-Flash 从 CoT 中获益，但 LLaMA-3-8B 和 GPT-3.5 几乎没有提升。
5. **LLaMA 在表格推理上异常弱**：LLaMA-8B 在表格推理上甚至比文档推理更差，可能与其对结构化格式的处理能力有关。

## 亮点与洞察

1. **"先结构化后生成"的管线设计极为巧妙**：以表格为中间表示，天然地控制跨文档依赖，解决了多文档基准创建中的核心难题。
2. **Oracle 自一致性验证**：利用构建过程本身的信息来验证一致性，是一种低成本高质量的自动验证思路。
3. **提供了表格-文档对比视角**：同一推理问题在两种表面形式下的表现差异，揭示了 LLM 推理中"理解"和"格式适应"的纠缠。
4. **推理类型可控**：通过 demonstrations 指定推理技能类型，未来可方便地添加新的推理挑战。

## 局限与展望

1. **数据域偏窄**：种子数据来自 Wikipedia 表格，覆盖体育、政治、科技等通用领域，未涉及法律、医学等专业领域。
2. **只用 GPT-4o 做生成和验证**：可能引入该模型的特定偏好和局限。
3. **非平行多语言**：虽然使用英语，但多语言扩展需要重新考虑文化差异和语言特性。
4. **反事实可能不够极端**：目前的"轻微反事实调整"可能不足以完全防止数据污染。
5. **1000 个样本规模较小**：可信的模型排名可能需要更大规模的基准。

## 相关工作与启发

- **HotPotQA / MuSiQue / FanOutQA**：经典多文档 QA 基准，但依赖大量人工标注
- **MuSR**：使用 neurosymbolic 方法生成多步推理基准，思路类似但面向单文档
- **TabFact / Chain-of-Tables**：表格推理领域的工作，本文将表格作为中间表示的灵感来源
- **合成基准趋势**：随着数据污染问题日益严重，LLM 驱动的合成基准创建成为重要方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 知识引导的合成管线设计新颖，Oracle 自一致性验证很有创意
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖多个模型族、多种提示策略、表格 vs 文档对比、文档排序消融
- **写作质量**: ⭐⭐⭐⭐⭐ — 管线描述清晰，图示直观，实验分析有深度
- **价值**: ⭐⭐⭐⭐ — 为多文档推理评估提供了可控、可扩展的新方法，对社区有实际贡献

<!-- RELATED:START -->

## 相关论文

- [KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)
- [READoc: A Unified Benchmark for Realistic Document Structured Extraction](readoc_a_unified_benchmark_for_realistic_document_structured_extraction.md)
- [PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)
- [MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture](sanskriti_a_comprehensive_benchmark_for_evaluating_language_models_knowledge_of_.md)

<!-- RELATED:END -->
