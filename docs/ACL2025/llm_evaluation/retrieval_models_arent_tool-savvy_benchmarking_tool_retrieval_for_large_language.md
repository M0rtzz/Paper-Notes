---
title: >-
  [论文解读] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models
description: >-
   提出ToolRet——首个大规模工具检索基准（7.6k检索任务、43k工具），揭示现有强IR模型在工具检索任务上表现不佳（最强模型nDCG@10仅33.83），并贡献超20万训练实例的ToolRet-train数据集，显著提升IR模型的工具检索能力和端到端工具使用任务通过率。
tags:

---

# Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models

## 论文信息

- **会议**: ACL 2025
- **arXiv**: [2503.01763](https://arxiv.org/abs/2503.01763)
- **代码**: [https://github.com/shizhl/Tool-Retrieval-Benchmark](https://github.com/shizhl/Tool-Retrieval-Benchmark)
- **领域**: LLM评估 / 工具检索
- **关键词**: Tool Retrieval, Tool Learning, Information Retrieval, Benchmark, LLM Agent

## 一句话总结

提出ToolRet——首个大规模工具检索基准（7.6k检索任务、43k工具），揭示现有强IR模型在工具检索任务上表现不佳（最强模型nDCG@10仅33.83），并贡献超20万训练实例的ToolRet-train数据集，显著提升IR模型的工具检索能力和端到端工具使用任务通过率。

## 研究背景与动机

- **领域现状**: 工具学习（Tool Learning）旨在为LLM配备外部工具以解决实际任务，在大规模工具集场景下，使用IR模型检索有用工具是关键的第一步。
- **已有基准局限**: 现有工具使用基准（如ToolBench、ToolACE）通过人工预标注每个任务仅10-20个相关工具来简化检索步骤，远离真实应用场景中面对数万工具的挑战。
- **关键差距**: 先导实验表明，用检索工具替代官方标注工具集后，Agent任务通过率显著下降；即使是ColBERTv2等强检索器也难以有效检索目标工具。
- **核心动机**: 需要(1)系统性评估IR模型在多样化工具检索场景中的表现；(2)分析检索质量对端到端工具使用任务通过率的影响。

## 方法详解

### 整体框架

ToolRet基准的构建包含三个阶段：(1) 数据收集——从AI顶会论文、会议资源和开源社区收集超30个工具使用数据集；(2) 数据采样——通过K-means聚类去冗余并合并工具集；(3) 指令构建——使用GPT-4o自动生成检索指令以支持指令性检索评估。

### 关键设计

1. **异构工具语料库**: 43k工具涵盖三种类型——Web API（36,978个）、代码函数（3,794个）和自定义应用（2,443个），覆盖多样化的工具文档类型和领域。
2. **基于聚类的任务采样**: 使用NV-embed-v1编码任务并执行K-means聚类，将聚类数设为工具集大小与查询数的最小值，从每个簇随机采样一个任务，确保多样性同时减少冗余。
3. **目标感知指令生成**: 邀请3位专家手写100条种子指令，再用GPT-4o通过上下文学习为每个任务自动生成指令，使指令能桥接查询意图与目标工具功能。

### 训练数据集 ToolRet-train

扩展数据收集至ToolACE、APIGen和ToolBench的训练集，构建超20万检索任务的训练数据。每个训练样本包含查询、生成指令、目标工具和10个由NV-embed-v1检索的负样本工具。训练采用对比学习框架，使用hard negative mining增强模型对工具相似性的辨别力。

### 基准统计

- **检索任务总量**: 7,615个（Web API检索: 4,916; 代码函数检索: 950; 自定义应用检索: 1,749）
- **工具总量**: 43,215个
- **平均查询/指令长度**: 46.87 / 43.43 tokens
- **平均工具文档长度**: 174.56 tokens

## 实验

### 主实验结果

| 模型类型 | 代表模型 | nDCG@10 |
|---------|---------|---------|
| 稀疏检索 | BM25 | 18.72 |
| 密集检索 | NV-embed-v1 | 33.83 |
| 密集检索 | E5-Mistral | 24.46 |
| ColBERT | ColBERTv2 | 19.82 |
| 交叉编码器 | MiniLM-L6 | 28.60 |
| LLM重排 | RankGPT | 30.56 |

- 即使最强模型NV-embed-v1在传统IR基准上表现优异，其在ToolRet上nDCG@10仅33.83，表明工具检索任务远比传统检索困难。

### 消融与深度分析

| 分析维度 | 发现 |
|---------|------|
| 词汇重叠率 | 工具检索任务中查询与目标工具的词汇重叠率远低于传统检索任务，要求IR模型具备更强的语义表示能力 |
| 任务迁移 | 从信息搜索型任务到工具检索的任务偏移导致IR模型性能下降 |
| 训练效果 | 在ToolRet-train上微调后，IR模型检索性能显著提升，端到端任务通过率也随之提高 |
| 检索对Agent影响 | 检索Recall@10与Agent任务通过率呈强正相关，验证了工具检索质量对下游的关键影响 |

### 关键发现

1. 现有IR模型在工具检索上表现远不如其在传统检索基准上的成绩，存在显著的任务差距。
2. 低质量的工具检索直接导致LLM Agent端到端任务通过率下降。
3. 使用ToolRet-train微调后的IR模型不仅检索性能大幅提升，还能有效提高工具使用LLM的整体任务表现。

## 亮点

- 首个系统性工具检索基准，覆盖7.6k任务和43k异构工具，填补了该领域的评估空白。
- 定量揭示了检索质量与Agent端到端性能之间的强相关性。
- 贡献的大规模训练数据集（20万+实例）为社区改进工具检索模型提供了实用资源。
- 基准设计遵循MTEB/BEIR格式标准，便于社区快速集成和复现。
- 覆盖6类IR模型的系统评估，为实践者选择工具检索方案提供了直接参考。

## 局限性

- 工具文档质量参差不齐，部分来源的文档描述过于简短或缺乏结构化信息。
- 评估仅覆盖英文工具检索任务，未考虑多语言工具检索场景。
- 指令由GPT-4o自动生成，可能存在与人工指令的分布偏差。
- 工具集合并策略可能丢失部分数据集特有的工具特征。
- 未评估检索延迟和在线部署场景下的效率表现。

## 相关工作

- **工具学习**: ToolBench (Qin et al., 2023)、ToolACE、APIGen等通过合成数据训练LLM使用工具，但面对大规模工具集时受限于上下文长度。
- **IR基准**: 传统IR基准如MS-MARCO和BEIR主要面向信息搜索任务，缺乏工具检索场景的评估。
- **相关系统**: ToolGen (Wang et al.)、COLT (Qu et al.)等使用语义检索辅助工具选择，但方案为临时性的，缺乏跨场景的系统评估。

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首次系统性定义和评估工具检索任务，填补明确空白。
- **实用性**: ⭐⭐⭐⭐⭐ — 基准和训练数据集对工具使用Agent生态有直接推动作用。
- **实验充分性**: ⭐⭐⭐⭐ — 评估模型类型全面，包含端到端任务通过率分析。
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，实验设计合理。

<!-- RELATED:START -->

## 相关论文

- [CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [AD-LLM: Benchmarking Large Language Models for Anomaly Detection](ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)
- [Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling](mis-prompt_benchmarking_large_language_models_for_proactive_error_handling.md)
- [Correlated Errors in Large Language Models](../../ICML2025/llm_evaluation/correlated_errors_in_large_language_models.md)
- [Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)

<!-- RELATED:END -->
