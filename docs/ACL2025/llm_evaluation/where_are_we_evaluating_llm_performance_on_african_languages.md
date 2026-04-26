---
title: >-
  [论文解读] Where Are We? Evaluating LLM Performance on African Languages
description: >-
  [ACL 2025][非洲语言] 构建了覆盖517种非洲语言、30个数据集、16类任务的 Sahara 基准，系统评估24个LLM在非洲语言上的表现，揭示语言政策驱动的数据不平等如何直接影响模型效果。
tags:
  - ACL 2025
  - 非洲语言
  - 多语言评估
  - 语言政策
  - 低资源NLP
  - 基准测试
---

# Where Are We? Evaluating LLM Performance on African Languages

**会议**: ACL 2025  
**arXiv**: [2502.19582](https://arxiv.org/abs/2502.19582)  
**代码**: [GitHub](https://github.com/UBC-NLP/sahara)  
**领域**: LLM NLP  
**关键词**: 非洲语言, 多语言评估, 语言政策, 低资源NLP, 基准测试

## 一句话总结

构建了覆盖517种非洲语言、30个数据集、16类任务的 Sahara 基准，系统评估24个LLM在非洲语言上的表现，揭示语言政策驱动的数据不平等如何直接影响模型效果。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：非洲拥有约2000种语言，是全球语言最多元的大陆，但在NLP研究中严重代表不足：

1. **历史语言政策的影响**：大多数非洲国家以殖民时期引入的外语（英语、法语、葡萄牙语）为官方语言。例如，尼日利亚512种本土语言中仅3种获得区域认可。即使获得官方认可的本土语言，其角色往往是象征性的而非功能性的。
2. **数据极度不均衡**：在517种非洲语言中，仅45种拥有超过1个数据集，绝大多数语言仅有语言识别数据。Amharic以11个数据集领先，而大部分语言几乎没有可用资源。
3. **现有评估不全面**：之前的工作如 IROKOBench 仅覆盖有限的非洲语言，缺乏全面的跨语言、跨任务评估基准来追踪整体进展。
4. **数据可用性≠说话人数**：拥有1.53亿使用者的尼日利亚皮钦语（Naija）被归为"被遗忘语言"，而仅500万使用者的加泰罗尼亚语却是高资源语言，说明语言声望、政策和数字化程度才是关键因素。

## 方法详解

### 整体框架

Sahara 基准采用模块化设计，从现有公开数据集中收集整合，覆盖分类、生成、多选/推理（MCCR）、token级四大任务簇，支持517种语言、30个数据集。同时提供了 HuggingFace 上的动态排行榜用于持续追踪模型表现。

### 关键设计

1. **广泛且多样的覆盖**：覆盖54个非洲国家中的50个，包含5种文字系统（阿拉伯文、科普特文、埃塞俄比亚文、拉丁文、Vai文），5个语系。每个任务从数据集中随机采样1000个样本用于few-shot测试。

2. **任务簇组织**：
    - 分类簇：跨语言NLI、语言识别（517种语言）、新闻分类、情感分析、主题分类
    - 生成簇：机器翻译（29种语言）、释义、摘要、标题生成
    - MCCR簇：通用知识（MMLU）、数学文字问题（MGSM）、阅读理解、问答
    - Token级簇：NER（27种语言）、短语分块、词性标注

3. **政策-数据-性能链分析**：不仅评估模型表现，还系统分析语言政策（教育政策、国家政策、区域政策）如何通过影响数据可用性，最终决定模型在特定语言上的效果，形成"政策→数据→性能"的因果链。

### 损失函数 / 训练策略

本文是评估工作，不涉及模型训练。评估设置：
- 统一使用 few-shot 设置（不同任务3-10 shots）
- 评估指标包括 Exact Match、F1、Accuracy、spBLEU1K、RougeL
- 评估24个模型：含 SLM（<8B）和 LLM（≥8B）两类

## 实验关键数据

### 主实验

24个模型在四大任务簇的平均表现（总体平均分）：

| 模型 | 分类Avg | 生成Avg | MCCR Avg | Token Avg | 总体Avg |
|------|---------|---------|----------|-----------|---------|
| Claude-4-Sonnet (闭源) | 47.28 | 10.59 | 60.53 | 44.86 | **40.82** |
| GPT-4.1 (闭源) | 48.07 | 11.06 | 50.98 | 34.05 | 36.04 |
| Command-A (111B) | 38.64 | 10.36 | 45.55 | 25.16 | 29.93 |
| Gemma3 (27B) | 44.44 | 8.19 | 43.20 | 16.45 | 28.07 |
| Llama3.1 (70B) | 35.96 | 11.15 | 43.67 | 15.51 | 26.57 |
| Phi-4 (3.8B) | 16.50 | 5.10 | 33.73 | 11.78 | 16.78 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 英语→非洲语言翻译 | spBLEU ~12 | 翻译为非洲语言困难 |
| 非洲语言→英语翻译 | spBLEU ~19 | 从非洲语言翻译更容易 |
| 法语→非洲语言翻译 | spBLEU ~3.4 | 法语翻译更困难，反映训练数据差异 |
| 语言识别任务 | F1 <5% | 大多数模型几乎无法识别非洲语言 |
| MGSM数学推理 | ExactM <10% | 开源模型在非洲语言数学推理上极差 |

### 关键发现

1. **闭源模型显著领先**：Claude-4-Sonnet以40.82的总体平均分大幅领先所有开源模型，GPT-4.1紧随其后。
2. **理解易于生成**：模型在分类任务上表现相对较好（部分语言>80%准确率），但在生成任务上表现糟糕（大部分BLEU/ROUGE<15），说明模型理解非洲语言的能力远高于生成。
3. **少数语言受益**：模型在Hausa、Swahili、Yorùbá、Afrikaans等少数资源丰富的语言上表现最好，这些语言都具有官方地位和充足的训练数据。
4. **数据可用性驱动性能**：性能差异与语言的数据量强相关，而非语言的内在复杂性。Swahili因其标准化拼写、规则形态和丰富的双语语料库而表现突出。
5. **小模型在特定任务可竞争**：Phi-4（3.8B）在MCCR任务中的SLM中表现最佳，某些场景下不必使用超大模型。

## 亮点与洞察

- **"政策→数据→性能"因果链的实证论证**：首次用大规模实证数据证明语言政策如何通过数据中介影响AI模型性能。
- **全面的覆盖**：517种语言、30个数据集、24个模型的综合评估，是非洲NLP最全面的基准。
- **动态排行榜**：在HuggingFace上公开的排行榜支持持续评估和追踪，推动社区发展。
- **可操作的政策建议**：不只是揭示问题，还提出了数据收集、政策改革、社区驱动标注的具体建议。
- **翻译方向的不对称性**：揭示了"翻译为非洲语言"比"从非洲语言翻译"困难得多，指出模型在目标端生成的薄弱。

## 局限与展望

1. **数据集多为翻译**：许多数据集（如AfriXLNI、AfriMMLU）是从英语翻译而来，不完全反映非洲语言的真实使用场景，可能引入标签不对齐和借词偏差。
2. **大多数语言仅有语言识别数据**：517种语言中的绝大多数只支持语言识别任务，无法评估更复杂的能力。
3. **评估方法限制**：采样1000个样本进行few-shot评估，may not capture the full variability of each language.
4. **缺乏生成质量的人类评估**：仅使用自动指标，未进行人类评估来验证生成文本的真实质量。
5. **方言变体未充分覆盖**：同一语言的方言差异可能很大，但基准中未区分方言。

## 相关工作与启发

- **IROKOBench**（Adelani et al., 2024b）首先评估了LLM在非洲语言上的表现，但覆盖范围有限。
- **AfroLID**（Adebara et al., 2022）覆盖517种非洲语言的语言识别，是Sahara的重要组成部分。
- **NLLB**（Meta）尝试改善低资源语言翻译，但性能仍不一致。
- **Masakhane**社区在推动非洲NLP方面做出了关键贡献，推动了多个数据集的创建。
- 本文的核心启示：技术进步必须与政策改革并行，单纯的模型优化无法解决数据稀缺的根本问题。

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 3 |
| 数据贡献 | 5 |
| 实验充分性 | 4 |
| 社会影响力 | 5 |
| 写作质量 | 4 |
| 总分 | 4.2 |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance](language_complexity_measurement_as_a_noisy_zero-shot_proxy_for_evaluating_llm_pe.md)
- [\[ACL 2025\] La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America](la_leaderboard_spanish.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[AAAI 2026\] Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](../../AAAI2026/llm_evaluation/where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)

<!-- RELATED:END -->
