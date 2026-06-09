---
title: >-
  [论文解读] EvoWiki: Evaluating LLMs on Evolving Knowledge
description: >-
  [ACL 2025][LLM评测][知识演化] 提出 EvoWiki，一个可自动更新的动态评估基准，将知识分为稳定 (stable)、演化 (evolved) 和未知 (uncharted) 三级，系统评估 LLM 对演化知识的利用能力，发现 RAG 和持续学习 (CL) 结合使用具有协同效应。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "知识演化"
  - "动态基准"
  - "RAG"
  - "持续学习"
  - "Wikidata"
  - "多跳推理"
---

# EvoWiki: Evaluating LLMs on Evolving Knowledge

**会议**: ACL 2025  
**arXiv**: [2412.13582](https://arxiv.org/abs/2412.13582)  
**代码**: [GitHub](https://github.com/wtangdev/EvoWiki)  
**领域**: LLM Evaluation / Knowledge Utilization  
**关键词**: 知识演化, 动态基准, RAG, 持续学习, Wikidata, 多跳推理  

## 一句话总结

提出 EvoWiki，一个可自动更新的动态评估基准，将知识分为稳定 (stable)、演化 (evolved) 和未知 (uncharted) 三级，系统评估 LLM 对演化知识的利用能力，发现 RAG 和持续学习 (CL) 结合使用具有协同效应。

## 研究背景与动机

- **问题定义**: LLM 的知识存在时间截止点 (cut-off date)，世界知识持续演化（事实更新、新信息涌现），如何评估 LLM 对动态变化知识的利用能力？
- **现有方法局限**: (1) 传统基准（NaturalQA、HotpotQA）是静态的，不敏感于时间变化；(2) 动态基准（LiveBench、RealtimeQA）关注数据新鲜度但未系统区分知识的不同演化级别；(3) 静态 golden answer 可能随知识更新变得过时，导致误判 (false negative)。
- **核心动机**: 需要一个既能自动更新、又能区分不同知识演化状态（已知不变 vs 已知但变了 vs 全新出现）的基准，以精确评估 LLM 的知识利用能力。
- **核心挑战**: (1) 新发布 LLM 存在测试集污染风险；(2) 不同演化级别的知识利用难度不同，需要细粒度属性支持分析；(3) 基准需要可自动更新以跟随知识和模型的持续演化。

## 方法详解

### 整体框架

EvoWiki 基于 Wikidata 知识图谱和 Wikipedia 文本源构建，通过比较不同时间戳的知识图谱快照来识别知识演化状态。整个数据集围绕三个时间点：init-time (2021.09，LLM 知识已充分覆盖)、cutoff-time (2024.01，LLM 知识截止)、current-time (2024.05，评估时间)。

### 关键设计

1. **三级知识演化分类**:
    - **Stable (稳定)**: 从 init-time 到 current-time 一直未变的事实 → 测试 LLM 对已有知识的基线表现
    - **Evolved (演化)**: init-time 前已存在但在 cutoff-time 到 current-time 之间发生变化的事实 → 测试 LLM 能否识别知识更新（如某人的配偶更换了）
    - **Uncharted (未知)**: cutoff-time 之后才出现的全新事实 → 测试 LLM 获取新知识的能力，同时天然免受污染

2. **多维属性标注**:
    - **Referenced Context (参考上下文)**: 将事实三元组通过远程监督锚定到 Wikipedia 页面，确保每个问题都有可验证的文档支持
    - **Multi-hop Reasoning (多跳推理)**: 从 1-hop 扩展到 3-hop 推理问题，测试知识整合与推理能力
    - **Popularity (流行度)**: 用 Wikipedia 页面浏览量衡量知识的热门程度，分析流行度对模型表现的影响

3. **可自动更新机制**: 构建流程基于 Wikidata/Wikipedia 的持续更新，时间戳可灵活调整以适配不同 LLM 的知识截止日期，支持无人工干预的自动化数据更新。

### 损失函数

无模型训练，属于 benchmark + 评估工作。对 RAG 和 CL 使用标准设置评估。

## 实验

### 主实验：RAG vs CL vs 组合方法 (Llama-3.1-8B)

| 方法 | Stable 单跳 | Stable 多跳 | Evolved 单跳 | Evolved 多跳 | Uncharted 单跳 | Uncharted 多跳 |
|------|------------|------------|-------------|-------------|---------------|---------------|
| Open-book (上界) | 86.87 | 56.40 | 75.24 | 60.30 | 83.52 | 51.32 |
| Closed-book | 31.61 | 22.17 | 6.96 | 13.99 | 10.84 | 17.90 |
| BM25 检索 | 59.41 | 14.42 | 36.13 | 13.85 | 44.93 | 15.47 |
| Contriever 检索 | 77.90 | 19.37 | 48.99 | 17.85 | 72.69 | 21.42 |
| SFT + Closed-book | 36.97 | 24.41 | 8.53 | 17.34 | 15.15 | 20.59 |
| **SFT + Contriever** | **82.85** | **24.02** | **57.22** | **20.22** | **78.85** | **24.84** |
| **SFT + Open-book** | **92.10** | **60.22** | **80.78** | **62.90** | **89.34** | **55.07** |

RAG 在单跳上表现强劲但多跳仍差；CL 提供一致但有限的提升；RAG+CL 组合实现最佳性能。

### 消融实验：数据集质量人工评估

| 指标 | Stable (单跳/全部) | Evolved (单跳/全部) | Uncharted (单跳/全部) |
|------|-------------------|--------------------|-----------------------|
| 流畅度 | 99.17 / 95.69 | 94.58 / 95.56 | 95.00 / 95.42 |
| 可回答性 | 96.67 / 94.44 | 94.17 / 95.69 | 92.92 / 92.64 |
| 准确性 | 97.92 / 93.19 | 93.33 / 94.58 | 91.67 / 90.97 |

人工评估验证了数据集在流畅度、可回答性和准确性三方面均达到 90%+ 的高质量。

### 关键发现

- **Evolved 知识最具挑战性**: Closed-book 下 Evolved 知识的单跳准确率仅 6.96%（Llama-3.1-8B），说明 LLM 严重倾向于输出过时答案。
- **RAG 在单跳上有效但多跳依然差**: Contriever 在 Stable 单跳上接近 Open-book (77.90 vs 86.87)，但多跳仅 19.37（Open-book 56.40），原因是多跳需要检索和整合多个文档片段。
- **CL 提供稳定但温和的提升**: SFT 在 Closed-book 上仅将 Evolved 从 6.96 提升到 8.53，Uncharted 从 10.84 到 15.15。
- **RAG + CL 的协同效应**: SFT + Contriever 在多数指标上超越单独 RAG 或 CL，如 Stable 单跳从 77.90 (Contriever) 提升到 82.85 (SFT + Contriever)。
- **大检索语料库反而降低性能**: 扩大检索语料库后 (BM25_large_corpus)，性能反而下降（干扰信息增多）。

## 亮点

- 知识三级分类（stable/evolved/uncharted）设计精巧，既有基线对照又有前沿挑战，是评估 LLM 知识利用的全面框架。
- 全自动可更新机制使基准不会过时——只需调整时间戳即可适配新发布的 LLM。
- 多维属性（上下文、多跳、流行度）支持细粒度分析，而非仅报告一个总分。
- 发现 RAG + CL 的协同效应，为未来结合两种方法适应知识演化提供了方向。

## 局限性

- 数据集主要基于英语 Wikidata/Wikipedia，缺乏多语言版本。
- 知识领域限制在人物实体（entity type = human），其他领域（科技、地理等）的演化特征可能不同。
- 多跳推理最多 3 跳，且自动化生成的 3-hop 问题可能退化为 2-hop 的浅层推理。
- GPT-4o-mini 用于问题润色可能引入系统性偏差。
- Evolved 和 Uncharted 分类依赖 Wikidata 编辑时间戳，而 Wikidata 本身存在更新延迟和噪声。

## 相关工作

- **时序 QA 基准**: TimeQA (Chen et al., 2021)、TEMPLAMA (Dhingra et al., 2021) 等评估时间感知推理；EvoWiki 进一步区分了知识的演化级别。
- **动态基准**: RealtimeQA (Kasai et al., 2023)、LiveBench (White et al., 2024) 关注数据新鲜度和污染缓解；EvoWiki 在此基础上加入了知识分级和多维属性。
- **知识冲突**: Ying et al. (2024) 发现 LLM 倾向于使用内部知识即使外部知识正确；EvoWiki 的 Evolved 类别正是测试这一冲突的理想场景。
- **RAG 与 CL**: Lewis et al. (2020) 提出 RAG；Tang et al. (2024) 提出生成器-阅读器框架；本文系统比较了两者在知识演化场景下的表现。

## 评分

| 维度 | 分数 (1-10) |
|------|-----------|
| 创新性 | 7 |
| 实验充分性 | 8 |
| 论文清晰度 | 8 |
| 实用性 | 8 |
| **总分** | **7.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning](guessarena_guess_who_i_am_a.md)
- [\[ACL 2025\] SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture](sanskriti_a_comprehensive_benchmark_for_evaluating_language_models_knowledge_of_.md)
- [\[ACL 2025\] Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration?](towards_objective_fine-tuning_how_llms_prior_knowledge_causes_potential_poor_cal.md)
- [\[ACL 2025\] AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge](antileakbench_preventing_data_contamination_by_automatically_constructing_benchm.md)
- [\[ACL 2025\] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)

</div>

<!-- RELATED:END -->
