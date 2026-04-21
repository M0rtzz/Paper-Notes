---
title: >-
  [论文解读] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models
description: >-
  [benchmark] 提出首个面向极端天气影响理解的LLM评估基准WXImpactBench，包含四阶段数据构建流水线和两个评估任务（多标签分类与排序问答），系统性评估了多个LLM在气候适应领域的能力。
tags:
  - benchmark
  - weather impact
  - LLM evaluation
  - multi-label classification
  - question answering
---

# WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models

**会议/期刊**: ACL 2025  
**arXiv**: [2505.20249](https://arxiv.org/abs/2505.20249)  
**代码**: [GitHub](https://github.com/Michaelyya/WXImpactBench)  
**领域**: LLM评估 / 气候文本理解  
**关键词**: benchmark, weather impact, LLM evaluation, multi-label classification, question answering  

## 一句话总结

提出首个面向极端天气影响理解的LLM评估基准WXImpactBench，包含四阶段数据构建流水线和两个评估任务（多标签分类与排序问答），系统性评估了多个LLM在气候适应领域的能力。

## 研究背景与动机

- **问题定义**：气候变化适应需要理解极端天气对社会的影响，LLM在此领域的有效性尚未被探索。
- **现有不足**：现有气候相关数据多来自结构化气象记录，存在日常缺失问题；且这些数据可能已包含在LLM预训练中，导致评估偏差。
- **关键挑战**：历史报纸中存在气候术语的多义性（如"blizzard"既指暴风雪也指运动队名），OCR数字化后的文本噪声严重影响下游任务。
- **本文方案**：从历史报纸中构建高质量极端天气影响数据集，设计WXImpactBench基准，通过多标签分类和排序问答两个任务评估LLM。

## 方法详解

### 整体框架

四阶段数据构建流水线 + 两任务评估框架：
1. **语料收集**：从专有档案机构获取两个时期的数字化报纸文本
2. **Post-OCR纠错**：使用GPT-4o进行OCR文本纠错，达到与人工标注高度一致的BLEU/ROUGE分数
3. **主题感知文章选择**：通过LDA主题建模从53,521篇文章中筛选，经三位领域专家人工审核得到350篇高质量样本
4. **人工标签标注**：定义六类脆弱性相关影响（基础设施、政治、金融、生态、农业、人类健康），三位标注员进行多标签二值标注

### 关键设计

- **多标签分类任务**：测试LLM区分六类天气影响的能力，使用row-wise accuracy作为严格指标（要求同时正确分类六个标签）
- **排序问答任务**：为每篇文章生成伪问题，构建100篇候选文章池（1正例+99负例），评估LLM的检索排序能力，为RAG系统开发奠基
- **混合上下文版本**：将长文本切分为约250 token的片段并独立标注，形成1,386个样本用于评估长上下文影响

### 损失函数/评估指标

- 分类任务：$\mathcal{L}(\hat{\mathcal{Y}}_t, \mathcal{Y}_t) = -\sum_{i=1}^{6} y_i \log \hat{y}_i$
- 分类指标：F1-score、Accuracy、Row-wise Accuracy
- 排序任务指标：Hit@1、nDCG@5、Recall@5、MRR

## 实验

### 主实验结果

| 模型 | Infrastructure | Political | Financial | Ecological | Agricultural | Human Health | Average |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GPT-4o | **80.94** | **58.46** | **65.82** | 46.81 | 70.33 | 73.23 | **65.93** |
| DeepSeek-V3-671B | 81.87 | 44.44 | 60.91 | 36.00 | 61.74 | 65.20 | 58.03 |
| Mistral-24B-IT | 79.12 | 47.18 | 59.64 | **44.90** | **67.74** | 66.88 | 60.91 |
| Gemma-2-9b-IT | 77.42 | 43.33 | 54.60 | 42.16 | 55.60 | 61.82 | 55.82 |

> Zero-shot F1-score（混合上下文版本），↑表示相比长上下文版本的提升。

### 消融实验

| 设置 | 影响 |
|------|------|
| 长上下文 vs 混合上下文 | 混合上下文平均提升2.38 F1，说明LLM在短文本上表现更好 |
| Zero-shot vs One-shot | One-shot整体提升，但部分模型（如Mixtral）出现不稳定 |
| 历史 vs 现代文本 | 现代文本普遍表现更好，历史叙事风格增加理解难度 |

### 关键发现

- GPT-4o在多数类别中表现最佳，但所有模型在生态和政治影响识别上均较弱
- 模型规模并非决定性因素：DeepSeek-V3（671B）在某些类别不如Mistral-24B
- 混合上下文版本普遍优于长上下文版本，表明当前LLM在长文本理解上仍有提升空间
- Row-wise accuracy极低（最高仅~30%），说明同时准确分类六类影响极具挑战性

## 亮点

- 首个极端天气影响理解的LLM评估基准，填补了气候NLP领域的空白
- 四阶段数据构建流水线设计精巧，结合了OCR纠错、LDA主题建模和领域专家标注
- 评估任务设计兼顾分类和检索两大应用场景，为气候RAG系统开发提供基础

## 局限性

- 数据集规模较小（350篇文章），可能限制评估的统计显著性
- 仅覆盖英文报纸，缺乏多语言评估
- 排序问答任务的伪问题由LLM生成，可能引入偏差
- 六类影响分类体系可能无法覆盖所有天气影响类型

## 相关工作

- **气候文本处理**：Mallick et al. (2024), Xie et al. (2024) 关注极端天气事件提取
- **气候基准**：CLLMate (Li et al., 2024) 关注天气预测而非影响理解
- **OCR纠正**：Drobac & Lindén (2020) 的神经OCR纠正模型
- **灾害NLP**：Purohit et al. (2013), Imran et al. (2016) 的灾害文本分类

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models](exposing_numeracy_gaps_a_benchmark_to_evaluate_fundamental_numerical_abilities_i.md)
- [AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research](abgen_evaluating_large_language_models_in.md)

<!-- RELATED:END -->
