---
title: >-
  [论文解读] Large Language Models as Medical Codes Selectors: A Benchmark Using the International Classification of Primary Care
description: >-
  [NeurIPS 2025 (GenAI for Health Workshop)][医学图像][医学编码] 构建了一个 extract-retrieve-select 框架的医学编码基准，在 33 个 LLM 上评估 ICPC-2 编码选择能力，发现 28 个模型 F1>0.8，证明 LLM 无需微调即可有效自动化初级保健编码。
tags:
  - NeurIPS 2025 (GenAI for Health Workshop)
  - 医学图像
  - 医学编码
  - ICPC-2
  - LLM基准测试
  - 极端多标签分类
  - 语义检索
---

# Large Language Models as Medical Codes Selectors: A Benchmark Using the International Classification of Primary Care

**会议**: NeurIPS 2025 (GenAI for Health Workshop)  
**arXiv**: [2507.14681](https://arxiv.org/abs/2507.14681)  
**代码**: 有  
**领域**: 医学图像/医学编码  
**关键词**: 医学编码, ICPC-2, LLM基准测试, 极端多标签分类, 语义检索

## 一句话总结

构建了一个 extract-retrieve-select 框架的医学编码基准，在 33 个 LLM 上评估 ICPC-2 编码选择能力，发现 28 个模型 F1>0.8，证明 LLM 无需微调即可有效自动化初级保健编码。

## 研究背景与动机

**领域现状**：医学编码是将临床表达映射到标准化分类系统（如 ICD-10、ICPC-2）的任务，属于极端多标签分类（XMC），传统方法费时且容易出错。

**现有痛点**：ICPC-2 有约 1300 个类别，数据不平衡严重；现有自动编码方法多针对 ICD，少有针对初级保健的 ICPC-2 研究。

**核心矛盾**：直接用 LLM 编码容易产生幻觉（编出不存在的代码），但结合检索可以约束输出。

**切入角度**：将问题分解为 extract-retrieve-select 三步，本文专注评估 select 步骤——给定检索结果，LLM 能否选出正确的 ICPC-2 代码。

## 方法详解

### 整体框架

(1) 语义搜索引擎（OpenAI text-embedding-3-large）从 73,563 个标注概念中检索候选；(2) LLM 接收查询和检索结果，选择最匹配的 ICPC-2 代码。

### 关键设计

1. **语义检索引擎**

    - 功能：将临床表达与 ICPC-2 概念库匹配
    - 核心思路：Chroma DB + HNSW 算法做向量相似性搜索
    - 设计动机：利用预训练嵌入捕捉语义相似性，克服精确匹配的局限

2. **LLM 选择评估**

    - 功能：33 个 LLM 在相同检索结果上进行代码选择
    - 核心思路：统一提示模板，提供查询+检索结果列表，要求 LLM 选择最佳 ICPC-2 代码
    - 评估维度：F1-score、token 用量、成本、响应时间、格式遵从度

### 损失函数 / 训练策略

无需训练。纯推理评估，所有 LLM 使用相同的 zero-shot 提示。

## 实验关键数据

### 主实验（437 条巴西葡语临床表达）

| 模型 | F1-score | Token 用量 | 格式遵从 |
|------|---------|-----------|---------|
| gpt-4.5-preview | **0.89** | 中等 | 99% |
| o3 | **0.88** | 高 | 99% |
| gemini-2.5-pro | **0.87** | 中等 | 98% |
| gpt-4o | 0.85 | 中等 | 99% |
| Baseline (top-1 检索) | 0.81 | 无 | 100% |
| gpt-4o (无检索) | 0.72 | 中等 | 95% |
| 小模型 (<3B) | <0.60 | 低 | <80% |

### 消融实验（检索优化影响）

| 检索配置 | 最佳模型 F1 | 提升 |
|---------|-----------|------|
| 默认检索 (k=10) | 0.85 | 基准 |
| 优化检索 (k=20) | **0.89** | +4pp |
| 无检索 | 0.72 | -13pp |

### 关键发现

- 28/33 模型 F1>0.8，证明 LLM 在约束条件下编码能力强
- 检索优化可提升最多 4pp，说明 retrieve 步骤的质量直接影响 select 质量
- 小模型 (<3B) 主要在格式遵从和长输入处理上失败
- 无检索直接编码 F1 显著下降（-13pp），验证了 retrieve-then-select 方案的必要性

## 亮点与洞察

- **模块化框架**：extract-retrieve-select 各步独立可优化，便于系统升级。
- **幻觉抑制**：通过检索约束输出空间，LLM 几乎不会生成不存在的代码。
- **多语言潜力**：在巴西葡语上测试，大多数 LLM 跨语言表现良好。

## 局限与展望

- 评估数据集仅 437 条，规模有限
- 仅评估 select 步骤，端到端（从临床笔记到编码）的评估缺失
- 单语言（葡语），多语言泛化需验证
- 未考虑编码者间一致性（inter-annotator agreement）

## 相关工作与启发

- **vs Infer-Retrieve-Rank**：D'Oosterlinck 等的通用 XMC 方案在非医疗基准上验证；本文首次在 ICPC-2 上系统评估
- **启发**：可以将此思路扩展到 ICD-10 编码（14,000+ 类别），规模更大但框架通用

## 评分
- 新颖性: ⭐⭐⭐ 框架思路并非全新，但首次在 ICPC-2 上系统评估
- 实验充分度: ⭐⭐⭐⭐ 33 个模型对比全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 对初级保健自动化有实际意义

<!-- RELATED:START -->

## 相关论文

- [FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](../../ACL2025/medical_imaging/medbiorag_semantic_search_and_retrieval-augmented_generation_for_biomedical_lite.md)
- [Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)
- [EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis](endobench_a_comprehensive_evaluation_of_multi-modal_large_language_models_for_en.md)

<!-- RELATED:END -->
