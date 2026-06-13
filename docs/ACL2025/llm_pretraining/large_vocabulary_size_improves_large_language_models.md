---
title: >-
  [论文解读] Large Vocabulary Size Improves Large Language Models
description: >-
  [ACL 2025][预训练][词汇表大小] 实验证明更大的 subword 词汇表大小 (vocabulary size) 能持续提升 LLM 在下游任务上的性能，并提出了一种简洁的词汇表替换方法 (Swap & Insert) 用于持续训练场景下切换到更合适的词汇表。 - 问题定义： 构建 LLM 时…
tags:
  - "ACL 2025"
  - "预训练"
  - "词汇表大小"
  - "subword"
  - "持续训练"
  - "词嵌入替换"
  - "单语LLM"
---

# Large Vocabulary Size Improves Large Language Models

**会议**: ACL 2025  
**arXiv**: [2406.16508](https://arxiv.org/abs/2406.16508)  
**代码**: 基于 Megatron-LM / SentencePiece  
**领域**: LLM Pretraining / Tokenization  
**关键词**: 词汇表大小, subword, 预训练, 持续训练, 词嵌入替换, 单语LLM  

## 一句话总结

实验证明更大的 subword 词汇表大小 (vocabulary size) 能持续提升 LLM 在下游任务上的性能，并提出了一种简洁的词汇表替换方法 (Swap & Insert) 用于持续训练场景下切换到更合适的词汇表。

## 研究背景与动机

- **问题定义**: 构建 LLM 时，subword 词汇表大小如何影响模型性能？目前主流做法（30k-60k）是否真的最优？
- **现有方法局限**: 从 BERT 到 GPT 再到 Llama，词汇表大小 30k-60k 已成为"魔法数字"，几乎没有论文给出选择依据。虽然有研究讨论大词汇表对推理效率的好处（如 Falcon），但对下游任务质量的影响一直缺乏系统研究。
- **核心动机**: (1) 大词汇表意味着每个 token 承载更多信息，同样的文本用更少的 token 表示，是否能让模型学到更好的表示？(2) 在持续训练（如将英语 LLM 适配日语）场景下，能否通过替换词汇表来提升性能？
- **核心挑战**: 大词汇表导致嵌入层和输出层参数量增大，需要在固定 token 数和固定 epoch 数两种设置下进行公平比较；持续训练中替换词汇表需要处理嵌入矩阵的初始化问题。

## 方法详解

### 整体框架

分为两部分研究：(1) 从头训练实验——在英语和日语上分别使用 5k/10k/50k/100k/500k 词汇表训练 680M 参数的 GPT-3 Large 模型；(2) 持续训练实验——在 Llama2-7B 基础上切换词汇表进行日语持续训练。

### 关键设计

1. **公平实验设计**: 由于不同词汇表大小会产生不同的 token 数量（如英语 5k 词汇表产生 830B tokens，500k 产生 640B），设计了两种训练配置——固定 1T tokens 和固定 1 epoch——确保既不偏向大词汇也不偏向小词汇。
2. **Swap & Insert 词汇表替换**: 对于持续训练场景，提出全新独立构建目标语言词汇表 $V_{new}$，从原始嵌入矩阵 $E_{orig}$ 构建新嵌入矩阵：$E_{new} = \frac{W \cdot E_{orig}}{\sqrt{|V_{orig}|}}$，其中 $W$ 是标准正态随机矩阵。Insert 策略进一步保留原始词汇表和新词汇表交集中 subword 的预训练嵌入，继承已有知识。
3. **Scaled Embed 稳定训练**: 使用 scaled embed 技术 (Takase et al., 2024) 在大词汇表训练中保持训练稳定性。

### 损失函数

标准自回归语言模型负对数似然损失 (next-token prediction)。

## 实验

### 主实验一：英语从头训练

| 词汇表大小 | PIQA | OBQA | HellaSwag | WinoGrande | ARC-e | ARC-c | 平均 |
|-----------|------|------|-----------|------------|-------|-------|------|
| 5k (1T tokens) | 69.9 | 33.2 | 51.0 | 55.2 | 49.6 | 27.7 | 47.8 |
| 10k | 71.2 | 33.4 | 51.5 | 55.2 | 50.6 | 27.1 | 48.2 (+0.4) |
| 50k | 71.7 | 32.8 | 53.9 | 54.5 | 50.8 | 27.7 | 48.6 (+0.8) |
| 100k | 70.9 | 33.4 | 53.9 | 54.8 | 54.3 | 27.7 | 49.2 (+1.4) |
| **500k** | **71.4** | **34.0** | **55.3** | **57.5** | **55.1** | **28.3** | **50.3 (+2.5)** |

词汇表从 5k 扩大到 500k，英语常识推理平均分提升 +2.5 个点。

### 主实验二：日语从头训练

| 词汇表大小 | JSQuAD | JCQA | XWinograd | JAQKET | 平均 |
|-----------|--------|------|-----------|--------|------|
| 5k (1T tokens) | 58.1 | 68.1 | 58.9 | 12.5 | 49.4 |
| 100k | 62.1 | 71.9 | 59.6 | 34.9 | 57.1 (+7.7) |
| **500k** | **64.5** | **71.6** | **59.3** | **38.9** | **58.6 (+9.2)** |

日语的提升更加显著（+9.2），尤其在需要生成答案的 JAQKET 任务上从 12.5 提升到 38.9。

### 消融实验：持续训练词汇表替换 (Llama2 → 日语)

| 方法 | JSQuAD | JCQA | XWinograd | JAQKET | 平均 |
|------|--------|------|-----------|--------|------|
| Llama2 原始词汇表 (32k) | 80.7 | 79.4 | 72.6 | 47.7 | 70.1 (+17.7) |
| Swap (100k, 随机初始化) | 79.2 | 80.2 | 67.5 | 56.3 | 70.8 (+18.4) |
| **Swap & Insert (100k)** | **81.9** | **80.2** | **69.2** | **61.2** | **73.1 (+20.7)** |
| Fujii et al. (2024) 方法 | 81.6 | 77.6 | 69.1 | 61.1 | 72.4 (+20.0) |

即使随机初始化的新嵌入 (Swap) 也能超越使用原始词汇表；加上 Insert 策略后进一步提升，且超越了先前的专门方法。

### 关键发现

- **大词汇表一致性提升**: 无论英语还是日语，无论固定 token 数还是固定 epoch 数，大词汇表都带来更好的性能。
- **生成任务受益最大**: 日语 JAQKET（无候选答案的问答）从 12.5 到 38.9 提升最大，表明大词汇表对生成能力特别有利。
- **训练效率提升**: 大词汇表下相同文本只需更少的 token，GPU 计算时间可减少约 30%（日语 100k vs 5k）。
- **持续训练中替换词汇表可行**: 即使完全新建词汇表并随机初始化嵌入，持续训练后仍能收敛并超越原词汇表。

## 亮点

- 首次系统性地研究了词汇表大小对 LLM 下游任务性能的影响，填补了长期以来的经验性空白。
- 实验设计严谨：通过同时控制 token 数和 epoch 数两种配置确保比较公平。
- 提出的 Swap & Insert 方法极其简单但有效，为跨语言持续训练提供了实用的词汇表替换方案。
- 发现大词汇表不仅提升性能还提升训练效率，这一 "一箭双雕" 的结论具有广泛实际意义。

## 局限性

- 仅在英语和日语两种语言上验证，虽然方法不依赖语言特性，但缺乏多语种泛化实验。
- 模型规模有限（从头训练仅 680M，持续训练仅 7B），更大模型（10B+）上的结论可能不同。
- 词汇表最大到 500k，无法观察更极端大小（如 1M+）的趋势和上界。
- 大词汇表的输出层计算开销增大，实际应用需要 adaptive softmax 等优化技术。

## 相关工作

- **词汇表构建**: BPE (Sennrich et al., 2016) 和 Unigram (Kudo, 2018) 是两大主流分词算法。BERT 用 30k、GPT 用 40k 几乎从未给过理由。
- **词汇表大小研究**: Kiyono et al. (2020) 研究过词汇大小与性能的关系但最大仅 32k；Ali et al. (2024) 做了更广泛分析但结论是"影响可忽略"，与本文结论相反。
- **多语言词汇表扩展**: Fujii et al. (2024)、Kim et al. (2024) 等研究在持续训练中扩展词表，但初始化策略更复杂；本文的随机投影初始化更简洁。

## 评分

| 维度 | 分数 (1-10) |
|------|-----------|
| 创新性 | 6 |
| 实验充分性 | 8 |
| 论文清晰度 | 9 |
| 实用性 | 8 |
| **总分** | **7.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](fr_spec_speculative_sampling.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)

</div>

<!-- RELATED:END -->
