---
title: >-
  [论文解读] Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints
description: >-
  [ACL 2025][NLP理解][语义解析] 针对"语义解析结果直接输入LLM反而降低性能"这一反直觉现象，提出SENSE——一种在prompt中嵌入语义提示（而非显式解析结果）的零样本方法，在GLUE理解任务、机器翻译、复述和简化等生成任务上一致性地提升了LLM表现。
tags:
  - ACL 2025
  - NLP理解
  - 语义解析
  - 提示工程
  - SENSE
  - LLM增强
  - 零样本
---

# Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints

**会议**: ACL 2025  
**arXiv**: [2409.14469](https://arxiv.org/abs/2409.14469)  
**代码**: 无  
**领域**: NLP Understanding / Prompt Engineering  
**关键词**: 语义解析, 提示工程, SENSE, LLM增强, 零样本

## 一句话总结
针对"语义解析结果直接输入LLM反而降低性能"这一反直觉现象，提出SENSE——一种在prompt中嵌入语义提示（而非显式解析结果）的零样本方法，在GLUE理解任务、机器翻译、复述和简化等生成任务上一致性地提升了LLM表现。

## 研究背景与动机
语义解析（Semantic Parsing）是NLP的基础任务，将自然语言转换为结构化语义表示（如SRL、AMR等）。在BERT时代，将语义解析结果注入模型输入已被证明能有效提升下游任务性能。

**进入LLM时代后的悖论**：
- Jin et al. (2024) 发现将AMR解析结果直接加入LLM prompt反而**损害**性能
- 原因分析：AMR等形式化表示对LLM来说是"陌生的符号系统"，引入了不必要的复杂性和潜在的解析错误
- 核心矛盾：语义信息对语言理解有帮助，但LLM不擅长处理显式的语义解析结构

**本文的切入角度**：既然LLM本身具备强大的语义理解能力，何不通过**提示**让LLM自行调用内部的语义解析能力，而非从外部注入解析结果？这就是SENSE的核心idea——用"语义提示"替代"语义结果"。

## 方法详解

### 整体框架
SENSE是一种零样本提示方法，在任务prompt中加入简短的语义提示语句（如"请利用语义解析结果来增强对句子结构和语义的理解"），引导LLM在内部激活语义解析能力，而不向其提供任何显式的解析输出。

### 关键设计

1. **语义提示注入（SENSE的核心）**：

    - 在标准任务prompt中插入一句提示，类似："please use semantic parsing result which can enhance comprehension of the sentence's structure and semantics"
    - 不同任务的具体措辞略有调整，但核心信息一致：引导模型关注语义结构
    - 设计动机：LLM经过大规模预训练已经内化了语义解析能力，SENSE通过提示"唤醒"这种能力，避免了外部解析的噪声和格式不兼容问题
    - 与CoT的区别：CoT说"let's think step by step"引导推理链，SENSE说"use semantic parsing"引导语义理解——前者适合推理任务，后者适合理解和生成任务

2. **与其他范式的对比**：

    - **SP-Input**（图1b）：先用外部工具生成解析结果，拼接到输入中 → 性能下降（引入噪声和陌生符号）
    - **SP-Output**（图1c）：让LLM先输出解析结果再做任务 → 性能不稳定（解析质量不一致）
    - **SENSE**（图1d）：仅给语义提示，不产生/消费任何解析结果 → 稳定提升
    - 关键洞察：LLM处理不了显式语义结构，但可以被提示去"内在地"利用语义信息

3. **注意力机制分析**：

    - 通过可视化LLaMA3-70B的注意力分数分布，发现SENSE使模型在复述任务中更集中地关注**关键语义元素**（核心词汇单元和句子主要成分）
    - 相比vanilla prompt，SENSE的注意力分布更加聚焦，而非均匀散布
    - 这从机制层面验证了SENSE确实引导模型进行了更有针对性的语义处理

### 损失函数 / 训练策略
SENSE是纯推理时的prompt方法，不涉及任何训练或微调。在GPT-3.5-turbo、GPT-4o-mini和LLaMA3-70B上均以零样本方式使用，温度设为0。

## 实验关键数据

### 主实验（GLUE理解任务）
| 模型配置 | SST-2 | MRPC | QQP | MNLI | QNLI | RTE | CoLA | 平均 |
|---------|-------|------|-----|------|------|-----|------|------|
| BERT_LARGE | 93.20 | 88.00 | 91.30 | 86.60 | 92.30 | 70.40 | 60.60 | 83.20 |
| GPT-4o-mini | 91.63 | 72.30 | 73.00 | 73.90 | 92.30 | 87.36 | 65.49 | 79.43 |
| GPT-4o-mini + SENSE | **92.08** | **76.47** | 73.00 | **78.20** | **93.30** | **88.45** | **67.22** | **81.25** |
| GPT-3.5 | 91.86 | 73.28 | 73.40 | 61.80 | 82.40 | 81.81 | 63.50 | 75.44 |
| GPT-3.5 + SENSE | 92.20 | 75.49 | 77.20 | 64.60 | 83.20 | 84.12 | 64.57 | 77.34 |

GPT-4o-mini + SENSE平均从79.43提升到81.25，缩小了与BERT的差距。MRPC和MNLI提升最为显著。

### 消融实验（不同范式对比，GPT-3.5基线）
| 方法 | SST-2 | MRPC | QQP | MNLI | QNLI | RTE | CoLA |
|------|-------|------|-----|------|------|-----|------|
| 基线 | 91.86 | 73.28 | 73.40 | 61.80 | 82.40 | 81.81 | 63.50 |
| + CoT | 89.11(-2.75) | 73.28 | 77.00(+3.60) | 56.20(-5.60) | 82.70 | 82.54 | 64.32 |
| + SP-Input | 87.50(-4.36) | 74.26 | 74.30 | **50.50(-11.30)** | 78.40(-4.00) | 84.11 | 58.37(-5.13) |
| + SP-Output | 89.11(-2.75) | 73.52 | 71.90 | 62.00 | 78.40(-4.00) | 81.59 | 64.44 |
| + SENSE | **92.20(+0.34)** | **75.49(+2.21)** | **77.20(+3.80)** | **64.60(+2.80)** | **83.20(+0.80)** | **84.12(+2.31)** | **64.57(+1.07)** |

SENSE在全部7个任务上均正向提升，而SP-Input、SP-Output和CoT均存在显著负面影响。

### 生成任务
| 任务 | 指标 | GPT-4o-mini | + SENSE | 提升 |
|------|------|-------------|---------|------|
| 复述 | 语义相似度↑ | 89.71 | 90.26 | +0.55 |
| 复述 | 词汇重叠↓ | 39.00 | 34.00 | -5.00 |
| 复述 | 句法多样性↑ | 7.25 | 8.08 | +0.83 |
| 简化(TurkCorpus) | BLEU↑ | 58.16 | 63.42 | +5.26 |
| 简化(TurkCorpus) | SAMSA↑ | 31.42 | 37.03 | +5.61 |
| 机器翻译(DE→EN) | COMET22↑ | 85.71 | 86.44 | +0.73 |
| 机器翻译(EN→DE) | BLEU↑ | 33.42 | 34.18 | +0.76 |

### 关键发现
- 直接注入语义解析结果对LLM有害（MNLI下降11.30个点），而SENSE保持一致正向提升
- CoT在理解任务上效果不佳（因CoT更适合推理任务），SENSE更适合需要语义理解的任务
- 在生成任务中，SENSE促进了更丰富的句法变换和词汇多样性，同时保持语义忠实度
- 注意力可视化证实了SENSE使模型聚焦于关键语义成分

## 亮点与洞察
- **极简但有效**：只需在prompt中加一句话就能获得稳定提升，零额外计算开销，零训练需求
- **深层认知启发**：SENSE的成功暗示LLM内部已编码了语义解析能力，关键是如何"唤醒"——这对理解LLM的内部表征有重要启示
- **与CoT的互补性**：CoT适合推理链，SENSE适合语义理解，两者可能在不同任务类型上各有所长
- **注意力分析**提供了可解释性证据，不是黑箱提升

## 局限与展望
- 仅在LLaMA和GPT系列上验证，需要在更多模型架构上确认通用性
- 任务覆盖面主要集中在已知语义解析有益的任务，更多元化的任务测试是必要的
- SENSE如何影响LLM内部决策过程的机制仍不清楚（LLM是黑箱）
- 未探索SENSE与CoT的组合使用
- "semantic parsing result"这个表述对不同模型可能有不同理解，提示措辞的鲁棒性未充分分析

## 相关工作与启发
- **Jin et al. (2024)** 提出AMR-COT，发现AMR直接注入LLM有负面影响——本文在此基础上提出了解决方案
- **语义解析的两面性**：对小模型（BERT）有益、对LLM有害，这种差异本身就是LLM能力特性的有趣反映
- **更广泛的prompt工程启发**：类似的"提示模型使用内在能力"的策略是否可以推广到其他领域？如"请使用你的知识图谱推理能力"或"请使用你的因果推理能力"？

## 评分
- 新颖性: ⭐⭐⭐⭐ 视角新颖——让LLM"自调用"语义能力而非外部注入
- 实验充分度: ⭐⭐⭐⭐ 涵盖理解和生成两大类任务，消融分析完整
- 写作质量: ⭐⭐⭐⭐ 论述逻辑清晰，图表信息量大
- 价值: ⭐⭐⭐⭐ 提供了一种即插即用的LLM增强策略，启发了对LLM内在能力激活的思考

<!-- RELATED:START -->

## 相关论文

- [Disambiguate First, Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing](disambiguate_first_parse_later_generating_interpretations_for_ambiguity_resoluti.md)
- [Generating Diverse Training Samples for Relation Extraction with Large Language Models](generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)
- [SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis](syngraph_a_dynamic_graph-llm_synthesis_framework_for_sparse_streaming_user_senti.md)
- [Language Models and Logic Programs for Trustworthy Tax Reasoning](../../AAAI2026/nlp_understanding/language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)
- [End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems](end-to-end_dialog_neural_coreference_resolution_balancing_efficiency_and_accurac.md)

<!-- RELATED:END -->
