---
title: >-
  [论文解读] Enhancing Multilingual LLM Pretraining with Model-Based Data Selection
description: >-
  [NeurIPS 2025][多语言/翻译][multilingual pretraining] 提出一套透明、简洁、高效的多语言模型驱动数据筛选框架，利用 FastText 和 Transformer（XLM-RoBERTa）嵌入分类器识别结构化且知识丰富的样本…
tags:
  - "NeurIPS 2025"
  - "多语言/翻译"
  - "multilingual pretraining"
  - "data selection"
  - "model-based filtering"
  - "FineWeb-2"
  - "data curation"
---

# Enhancing Multilingual LLM Pretraining with Model-Based Data Selection

**会议**: NeurIPS 2025  
**arXiv**: [2502.10361](https://arxiv.org/abs/2502.10361)  
**代码**: [github.com/epfml/fineweb2-hq](https://github.com/epfml/fineweb2-hq)  
**领域**: 多语言翻译  
**关键词**: multilingual pretraining, data selection, model-based filtering, FineWeb-2, data curation

## 一句话总结

提出一套透明、简洁、高效的多语言模型驱动数据筛选框架，利用 FastText 和 Transformer（XLM-RoBERTa）嵌入分类器识别结构化且知识丰富的样本，在 FineWeb-2 数据集上仅用 15% 的 token 即可匹配基线 MMLU 分数，并将该框架扩展至 20 种语言并公开发布了精炼的预训练数据集。

## 研究背景与动机

数据集策展已成为 LLM 性能的基石。虽然 FineWeb-Edu 和 DCLM 等工作证明了基于模型的过滤在英语数据上的巨大潜力（仅用 10% token 即可匹配全量训练），但这些进展**几乎全部集中在英语**。这种偏差有可能进一步拉大不同语言之间的 LLM 性能差距。

当前多语言数据策展的现状：
- **FineWeb-2**（SOTA）：主要依赖启发式过滤器
- **CCNet**：使用语言模型困惑度过滤，但不是最优
- 缺乏系统性的多语言模型驱动过滤方法

本文的核心目标：将模型驱动过滤从英语扩展到多语言，质量定义聚焦于**结构化数据**和**知识丰富的样本**。选取中文、德语、法语、阿拉伯语、丹麦语作为代表语言，覆盖不同语系、文字系统和资源可用性。

## 方法详解

### 整体框架

框架分两步：(1) 选择合适的分类器训练数据集；(2) 训练分类器进行文档评分与过滤。

### 关键设计

#### 1. 分类器训练数据集构建

定义两个质量标准：样本必须信息丰富且结构良好，且数据集须多语言可用。

选取 5 个代表性数据集作为正样本来源：
- **Aya Collection**：约 514M 样本，101 种语言，AI 生成，无质量保证但规模最大
- **Aya Dataset**：约 202K 样本，65 种语言，人工标注
- **MMLU**：约 14K 多选题，英语原版 + 14 种语言专业翻译
- **OpenAssistant-2**：约 14K 对话，28 种语言
- **Include-Base-44**：约 23K 样本，44 种语言，学术和职业考试

组合为两种配置（权衡质量与规模）：
- **MKC**：Include-Base-44 + OpenAssistant-2 + MMLU + Aya Dataset
- **MKC+**：MKC + Aya Collection

构建二分类数据集：正样本最多 80K（从小数据集取全部 + Aya Collection 随机采样），负样本等量随机采自 FineWeb-2。每种语言独立构建，避免语言偏差泄露。

#### 2. FastText 过滤（FT）

使用二分类 FastText 分类器，CPU 上高效运行：
- 训练特征：2-gram（中文用 4-gram）
- 对所有文档评分，按目标保留比例设定阈值

#### 3. Transformer 嵌入过滤

使用预训练 XLM-RoBERTa base（279M 参数，支持 100 种语言），不做微调以保持通用嵌入。

对每个文档取前 512 tokens，通过 mean pooling 得到 768 维嵌入，然后使用两种方法：

**MLP 分类器**：单隐藏层（256 维），ReLU + 20% Dropout + Sigmoid，AdamW 优化器（lr=0.0003），6 epoch 训练，BCE 损失。

**余弦相似度（CS）**：计算文档嵌入与 $K=8192$ 个随机采样正样本嵌入的最大余弦相似度。

### 损失函数 / 训练策略

- LLM 评估：1B 参数 Llama 模型，70B 或 119B tokens 训练
- 优化器：AdamW，WSD 学习率调度
- Batch size：1.6M tokens，学习率 0.0008，2000 warmup steps
- Tokenizer：多语言 Mistral v3 (Tekken)
- 模型选择：跨语言的 FineTasks 全局排名

## 实验关键数据

### 主实验

**方法排名**（跨中德法阿丹 5 种语言，70B + 119B tokens）：

| 方法 | 平均排名 |
|------|---------|
| **MLP MKC+** | **4.35** |
| MLP MKC | 6.11 |
| FT MKC+ | 7.17 |
| FT MKC | 8.04 |
| CS MKC | 8.10 |
| **Baseline (FineWeb-2)** | **8.72** |
| CS MKC+ | 8.79 |

MLP MKC+ 一骑绝尘，大幅领先基线。

**英语验证**（119B tokens，对比 DCLM 和 FineWeb-Edu）：

| 数据集 | 平均排名 |
|--------|---------|
| **Ours (MLP MKC+)** | **1.83** |
| DCLM | 2.39 |
| FineWeb-Edu | 2.44 |
| FineWeb | 3.33 |

在英语上也取得了最优的综合排名。

**Token 效率**：在高资源语言上，仅用 10% 数据保留即可在约 20B tokens（总量的 16.7%）时达到基线水平。用 15% token 即可匹配基线 MMLU 分数。

### 消融实验

**阈值选择**（中德法，MLP/FT，10%/15%/20%）：
- MLP MKC+ 在 10% 保留率下排名最优（8.85）
- MKC 训练集在更高保留率（15-20%）下表现更好

**训练数据来源的重要性**（30B tokens, MLP 10%）：
- MKC+（组合所有数据）排名最优（2.52）
- Aya Collection 单独也很强（2.91），尽管无质量保证
- Include-Base-44 和 OpenAssistant-2 单独使用反而弱于基线

**数据污染分析**（13-gram 去污）：
- 去污后结果仅轻微下降，方法仍大幅优于未去污的基线
- 排除了性能提升主要来自数据污染的可能性

### 关键发现

1. **多语言诅咒的缓解**：在质量过滤后的数据上，多语言模型（5 语言 × 119B tokens = 595B tokens）反而**优于**对应的单语模型。但在未过滤数据上，多语言训练仍然受到诅咒影响。这一发现意义重大。

2. **FastText 是资源受限时的良好替代**：虽然 MLP MKC+ 最强，但 FT 分类器在 CPU 上运行，无需 GPU 计算嵌入，性价比高。

3. **框架的跨语言通用性**：从高资源语言（中德法）到低资源语言（阿拉伯语、丹麦语）均有效，支持不同文字系统和语系。

## 亮点与洞察

1. **极高的数据效率**：10% 数据保留即可匹配甚至超越 100% 数据训练的模型，资源节约巨大
2. **缓解多语言诅咒**：质量过滤使多语言训练从劣势变为优势，这一发现具有重要理论和实践意义
3. **框架的透明度和可复现性**：公开数据集（20 种语言）、代码、XLM-RoBERTa 嵌入，促进了开放科学
4. **正样本选择的巧妙设计**：利用多语言基准数据集作为"高质量"的代理信号，避免了人工标注
5. **系统性的消融研究**：对分类器类型、训练数据、保留阈值做了全面且跨语言的控制实验

## 局限与展望

1. **质量定义偏向知识/结构**：以基准数据集作为正样本可能偏向学术/考试风格文本，不一定覆盖所有"高质量"维度（如创意写作、对话等）
2. **低资源语言提升有限**：阿拉伯语和丹麦语保留率较高（56%/65%），过滤效果不如高资源语言显著
3. **XLM-RoBERTa 嵌入计算成本**：计算 20 种语言全量嵌入耗费约 4K H100 GPU 小时（虽然已公开发布以摊销）
4. **仅评估了 1B 参数模型**：能否在更大规模模型上保持同样的数据效率优势待验证
5. **仅覆盖 20 种语言**：互联网上有远超 20 种常用语言，框架在极低资源语言上的表现未知

## 相关工作与启发

- **DCLM**：本文的 FastText 方法直接受 DCLM 启发，但将其扩展到多语言场景
- **FineWeb-Edu**：使用 LLM 评估教育价值进行过滤，但仅限英语且计算更贵
- **FineWeb-2**：本文的基线数据集，主要依赖启发式过滤
- **启发**：(1) 模型驱动过滤是多语言 LLM 预训练的关键杠杆；(2) 数据质量提升可"治愈"多语言诅咒，暗示该问题的本质是数据噪声而非语言干扰

## 评分

- 新颖性: ⭐⭐⭐ — 方法本身（FastText/MLP 分类器）较简单，创新在于系统性地扩展到多语言
- 实验充分度: ⭐⭐⭐⭐⭐ — 5 种语言、多种分类器、多种阈值、去污染分析、多语言训练，极其全面
- 写作质量: ⭐⭐⭐⭐ — 清晰系统，实验描述详尽
- 价值: ⭐⭐⭐⭐⭐ — 公开 20 语言精炼数据集，实用价值极高，缓解多语言诅咒的发现意义深远

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LangSAMP: Language-Script Aware Multilingual Pretraining](../../ACL2025/multilingual_mt/langsamp_multilingual_pretraining.md)
- [\[NeurIPS 2025\] ParallelPrompt: Extracting Parallelism from Large Language Model Queries](parallelprompt_extracting_parallelism_from_large_language_model_queries.md)
- [\[AAAI 2026\] MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis](../../AAAI2026/multilingual_mt/midb_multilingual_instruction_data_booster_for_enhancing_cultural_equality_in_mu.md)
- [\[ACL 2025\] CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis](../../ACL2025/multilingual_mt/culfit_a_fine-grained_cultural-aware_llm_training_paradigm_via_multilingual_crit.md)
- [\[ACL 2025\] LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation](../../ACL2025/multilingual_mt/laca_crosslingual_absa.md)

</div>

<!-- RELATED:END -->
