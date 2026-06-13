---
title: >-
  [论文解读] Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
description: >-
  [ACL 2025 (TACL 2025)][LLM评测][不确定性量化] 构建了 LM-Polygraph 不确定性量化（UQ）基准，实现了30+种SOTA方法，在11个文本生成任务上系统评估了UQ和置信度归一化技术的效果，为LLM幻觉检测提供了统一的评测框架。 领域现状：大语言模型的快速发展推动了其在各种场景中的应用…
tags:
  - "ACL 2025 (TACL 2025)"
  - "LLM评测"
  - "不确定性量化"
  - "幻觉检测"
  - "基准测试"
  - "置信度校准"
  - "文本生成"
---

# Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph

**会议**: ACL 2025 (TACL 2025)  
**arXiv**: [2406.15627](https://arxiv.org/abs/2406.15627)  
**代码**: [https://github.com/IINemo/lm-polygraph](https://github.com/IINemo/lm-polygraph)  
**领域**: LLM评测  
**关键词**: 不确定性量化, 幻觉检测, 基准测试, 置信度校准, 文本生成

## 一句话总结

构建了 LM-Polygraph 不确定性量化（UQ）基准，实现了30+种SOTA方法，在11个文本生成任务上系统评估了UQ和置信度归一化技术的效果，为LLM幻觉检测提供了统一的评测框架。

## 研究背景与动机

**领域现状**：大语言模型的快速发展推动了其在各种场景中的应用，但"幻觉"问题——模型生成看似合理但实际错误的内容——始终是一个关键挑战。不确定性量化（Uncertainty Quantification, UQ）是应对这一问题的重要技术手段：如果能准确估计模型对其输出的"信心"，就能在信心低时提醒用户或触发更可靠的回退机制。

**现有痛点**：当前UQ研究存在严重的碎片化问题：（1）不同论文使用不同的UQ方法实现，缺乏统一的代码库；（2）评测数据集和指标不一致，导致方法之间难以公平比较；（3）许多方法仅在特定任务上评测，缺乏跨任务的泛化性分析；（4）置信度归一化——将原始UQ分数转化为可解释的概率——这一重要问题被大多数工作忽略。

**核心矛盾**：UQ方法众多，但没有一个统一的平台来公平、可控地比较它们的效果。不同论文的实验设置差异太大，使得研究者无法判断哪种方法在什么场景下最有效。

**本文目标**：构建一个全面的UQ基准，（1）统一实现SOTA方法，（2）提供可控的评测环境，（3）覆盖多种文本生成任务，（4）支持置信度归一化方法的评估。

**切入角度**：作者在之前的 LM-Polygraph（EMNLP 2023 Demo）框架基础上，大幅扩展了方法库和评测范围，从一个工具升级为一个完整的基准平台。

**核心 idea**：统一框架下的大规模UQ方法对比，涵盖白盒和黑盒方法、序列级和claim级粒度、以及置信度归一化评估。

## 方法详解

### 整体框架

LM-Polygraph 基准包含三个层次：（1）方法库：实现了30+种UQ方法，涵盖信息论方法、语义多样性方法、密度估计方法、集成方法、反思方法等多个类别；（2）评测平台：支持11个任务的统一评测，包括QA、摘要、翻译、事实验证等；（3）归一化评测：额外评估将原始UQ分数转化为可解释置信度的归一化方法。

### 关键设计

1. **多类别UQ方法实现 (Multi-Category UQ Methods)**:

    - 功能：提供涵盖所有主流UQ范式的统一实现
    - 核心思路：将UQ方法分为以下几大类别：（a）**信息论方法**（白盒）：基于token概率的方法如最大序列概率、困惑度、平均/最大token熵、蒙特卡洛序列熵、点互信息等；（b）**语义多样性方法**：通过多次采样检测输出的语义一致性，如语义熵（Semantic Entropy）、TokenSAR、EigenScore 等；（c）**密度估计方法**：利用隐藏层表示的密度来估计是否为分布外输入，如马氏距离、鲁棒密度估计等；（d）**反思方法**：让模型自己评估其输出的可靠性，如 p(True)、Verbalized Uncertainty；（e）**黑盒方法**：不需要访问模型内部的方法，如基于图拉普拉斯特征值的 EigV、词汇相似度 LexSim 等
    - 设计动机：只有实现所有主流方法才能进行公平、全面的比较，避免因实现差异导致的评测偏差

2. **统一评测环境 (Unified Evaluation Environment)**:

    - 功能：提供可控、可复现的评测框架
    - 核心思路：对每个任务定义标准的数据集划分、预处理流程、和评测指标。主要评测指标包括：AUROC（区分正确和错误输出的能力）、AUPR（精确率-召回率曲线下面积）、以及校准误差（ECE）。支持在序列级（整个输出是否可靠）和claim级（单个声明是否可靠）两个粒度上评估
    - 设计动机：缺乏统一评测环境是当前UQ研究碎片化的根本原因，必须从平台层面解决

3. **置信度归一化评测 (Confidence Normalization Assessment)**:

    - 功能：评估将原始UQ分数转化为可解释概率值的归一化方法
    - 核心思路：原始的UQ分数（如熵、概率等）通常不能直接解释为"模型有X%的把握"。本文评估了多种归一化方法，包括Platt Scaling、温度缩放（Temperature Scaling）、等保序回归（Isotonic Regression）等，测量它们在将UQ分数映射为校准概率后的ECE
    - 设计动机：对于实际应用，一个可解释的置信度分数（如"这个回答有85%的可能是正确的"）比一个原始的熵值更有用

### 评测任务覆盖

涵盖11个任务：开放域问答（TriviaQA、CoQA、Natural Questions）、阅读理解、文本摘要（CNN/DM、XSum）、机器翻译（WMT）、常识推理、事实验证、数学推理等。

## 实验关键数据

### 主实验

UQ方法在不同任务上的AUROC表现（检测错误输出的能力）：

| UQ方法类别 | 代表方法 | QA任务平均AUROC | 摘要任务平均AUROC | 翻译任务平均AUROC | 总体排名 |
|-----------|---------|----------------|-----------------|-----------------|---------|
| 信息论（白盒） | Mean Token Entropy | 0.72 | 0.68 | 0.71 | 中等 |
| 信息论（白盒） | Perplexity | 0.70 | 0.66 | 0.69 | 中等 |
| 语义多样性 | Semantic Entropy | 0.78 | 0.73 | 0.74 | 最佳 |
| 语义多样性 | EigenScore | 0.76 | 0.71 | 0.73 | 优秀 |
| 密度估计 | Mahalanobis Distance | 0.65 | 0.62 | 0.63 | 较弱 |
| 反思方法 | p(True) | 0.74 | 0.70 | 0.68 | 良好 |
| 黑盒 | EigV (Graph Laplacian) | 0.75 | 0.72 | 0.71 | 良好 |
| 黑盒 | Verbalized UQ | 0.71 | 0.67 | 0.65 | 中等 |

### 消融实验——归一化方法效果

| 归一化方法 | 平均ECE↓ | 平均AUROC | 说明 |
|-----------|---------|-----------|------|
| 无归一化 | 0.32 | 0.74 | 原始分数校准差 |
| Platt Scaling | 0.12 | 0.74 | AUROC不变，校准显著改善 |
| Temperature Scaling | 0.14 | 0.74 | 效果接近Platt |
| Isotonic Regression | 0.09 | 0.74 | 最佳校准效果 |
| Histogram Binning | 0.15 | 0.73 | 简单但有效 |

### 关键发现

- **语义多样性方法总体最优**：Semantic Entropy 和 EigenScore 在大多数任务上表现最好，因为它们能够区分"相同意思的不同表述"和"真正不同的回答"
- **白盒方法优于黑盒方法**：能访问token概率的方法普遍优于只能看到最终输出的方法，但差距随模型变大而缩小
- **密度估计方法表现不佳**：基于隐藏层表示的方法在NLG任务上效果有限，可能因为这些方法最初为分类任务设计
- **置信度归一化至关重要**：原始UQ分数的校准误差很高（ECE~0.32），归一化后可降至0.09，使得置信度分数真正可解释
- **没有"一种方法适用所有场景"**：不同任务和模型下最优方法可能不同，但语义多样性方法是最稳健的选择
- **Claim级评估比序列级更有挑战**：在claim级别上识别不可靠的具体声明比判断整个输出是否可靠困难得多

## 亮点与洞察

- 这是目前最全面的LLM不确定性量化基准，实现了30+种方法并在11个任务上统一评测
- 首次系统评测了置信度归一化方法，这对实际部署非常重要
- 开源的代码库（468 stars）已成为UQ研究的事实标准工具
- 基准设计支持新方法的轻松集成，降低了后续研究的门槛

## 局限与展望

- 当前评测主要针对英语模型，多语言场景下的UQ行为可能不同
- 某些UQ方法（如语义多样性方法）需要多次采样，计算开销大，文中对效率的比较不够充分
- 未涵盖多模态LLM的不确定性量化
- 未来可以探索结合多种UQ方法的集成策略、以及UQ方法在实际部署中的用户体验影响

## 相关工作与启发

- 与 EMNLP 2023 的 LM-Polygraph Demo 论文是同一系列工作的升级版，从Demo工具升级为完整的Benchmark论文
- 语义熵（Semantic Entropy）作为表现最佳的方法之一，其核心思想——通过语义等价性聚类来衡量输出多样性——对后续UQ研究有重要启发
- 对于LLM应用开发者，本文提供了清晰的UQ方法选型指南

## 评分

- **新颖性**: ⭐⭐⭐ — 方法层面新颖性有限（主要是benchmark贡献），但系统性和全面性弥补了这一点
- **实验充分度**: ⭐⭐⭐⭐⭐ — 30+种方法、11个任务、多个模型的大规模评测，是目前最全面的UQ基准
- **写作质量**: ⭐⭐⭐⭐ — 框架描述清晰，结论有实用价值
- **价值**: ⭐⭐⭐⭐⭐ — TACL发表+83次引用，已成为UQ领域的标准参考，代码库广泛使用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](../../NeurIPS2025/llm_evaluation/efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [\[ACL 2025\] AD-LLM: Benchmarking Large Language Models for Anomaly Detection](ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)
- [\[ACL 2025\] Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling](mis-prompt_benchmarking_large_language_models_for_proactive_error_handling.md)

</div>

<!-- RELATED:END -->
