---
title: >-
  [论文解读] Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models
description: >-
  [ACL 2025 Main][信息检索/RAG][文本嵌入] 本文系统研究了文本嵌入模型中的"粘性token"现象——某些异常token被反复插入句子后会将句子相似度拉向固定值，提出了高效检测方法 STD 并在 14 个模型家族的 40 个检查点中发现 868 个粘性token，揭示了高达 50% 的下游任务性能下降。
tags:
  - "ACL 2025 Main"
  - "信息检索/RAG"
  - "文本嵌入"
  - "粘性token"
  - "词表异常"
  - "tokenization鲁棒性"
  - "嵌入模型分析"
---

# Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models

**会议**: ACL 2025 Main  
**arXiv**: [2507.18171](https://arxiv.org/abs/2507.18171)  
**代码**: [GitHub](https://github.com/March-7/StickyToken)  
**领域**: 信息检索  
**关键词**: 文本嵌入、粘性token、词表异常、tokenization鲁棒性、嵌入模型分析

## 一句话总结

本文系统研究了文本嵌入模型中的"粘性token"现象——某些异常token被反复插入句子后会将句子相似度拉向固定值，提出了高效检测方法 STD 并在 14 个模型家族的 40 个检查点中发现 868 个粘性token，揭示了高达 50% 的下游任务性能下降。

## 研究背景与动机

**领域现状**：基于 Transformer 的文本嵌入模型在 NLP 任务中被广泛使用，包括语义检索、文本聚类、句子相似度计算等。这些模型通常采用均值池化（mean pooling）或 [CLS] token 来生成句子级嵌入表示。

**现有痛点**：研究者发现某些"异常"token 在被重复插入句子时，会劫持模型的内部表示，将句子嵌入之间的余弦相似度拉向某个固定值，严重破坏了嵌入空间的正常分布。这种现象此前未被系统研究，也缺乏检测工具。

**核心矛盾**：文本嵌入模型依赖 tokenizer 将文本转为 token 序列，但 tokenizer 的词表中包含大量特殊符号、未使用 token 和多语言碎片子词，这些 token 在训练中未获得充分学习，导致它们在推理时产生异常行为。均值池化机制使得这些异常 token 的影响被放大——即使只插入少量粘性 token，也能主导整个句子的嵌入表示。

**本文目标**：(1) 形式化定义粘性 token 概念；(2) 提出高效检测方法；(3) 大规模分析粘性 token 的来源和特征；(4) 评估其对下游任务的影响。

**切入角度**：作者观察到如果一个 token 被多次插入不同句子，句子之间的余弦相似度会收敛到一个固定值，而非反映真实语义关系。这种"粘性"行为破坏了嵌入模型的基本假设——语义相近的句子应该有更高的相似度。

**核心 idea**：通过句子级过滤和 token 级过滤的两阶段检测方法（STD），高效识别词表中的粘性 token，并系统分析其成因和影响。

## 方法详解

### 整体框架

STD（Sticky Token Detector）采用两阶段过滤流程：首先在句子级别筛选出可能包含粘性 token 的异常嵌入行为，然后在 token 级别精确定位具体的粘性 token。输入是文本嵌入模型和其词表，输出是被标记为粘性的 token 列表。

### 关键设计

1. **粘性 token 的形式化定义**:

    - 功能：给出粘性 token 的数学定义，为检测提供理论基础
    - 核心思路：定义一个 token $t$ 为粘性 token，当且仅当将 $t$ 重复 $k$ 次插入任意句子 $s$ 后，不同句子对 $(s_i, s_j)$ 之间的余弦相似度趋近于某个固定常数 $\mu_t$，即 $\text{cos}(\text{emb}(s_i \oplus t^k), \text{emb}(s_j \oplus t^k)) \to \mu_t$，其中 $\mu_t$ 与原始句子内容无关
    - 设计动机：此前"异常 token"的概念模糊，缺乏统一定义，形式化有助于后续量化分析和自动化检测

2. **句子级过滤（Sentence Filtering）**:

    - 功能：快速排除大部分正常 token，只保留可疑 token 进入精细检测
    - 核心思路：对词表中每个 token，将其重复若干次插入一组种子句子中，计算插入后所有句子对之间的余弦相似度标准差。如果标准差极低（低于阈值），说明该 token 将不同句子"拉"到了同一个位置，被标记为可疑粘性 token
    - 设计动机：正常 token 插入后不会显著改变句子间的相对距离，标准差应保持在正常水平；而粘性 token 会主导嵌入，使标准差趋近于零

3. **Token 级过滤（Token Filtering）**:

    - 功能：对句子级筛选出的可疑 token 进行精细验证
    - 核心思路：使用更多的句子对和不同的插入次数来验证可疑 token。观察随着重复次数增加，相似度是否稳定收敛到固定值。只有通过多轮验证仍然表现出粘性行为的 token 才被最终确认
    - 设计动机：句子级过滤可能引入假阳性（例如某些低频但正常的 token），token 级过滤通过更严格的验证减少误报

### 损失函数 / 训练策略

本文是分析型工作，不涉及模型训练。

## 实验关键数据

### 主实验

在 14 个模型家族的 40 个检查点上应用 STD 进行大规模检测：

| 模型家族 | 检查点数 | 发现粘性token数 | 典型来源 |
|----------|---------|----------------|---------|
| BERT 系列 | 8 | 120+ | 未使用 [unused] token |
| RoBERTa 系列 | 5 | 80+ | 特殊符号、多语言碎片 |
| Sentence-BERT | 6 | 90+ | 继承自基座模型 |
| E5 系列 | 4 | 70+ | 多语言子词碎片 |
| GTE 系列 | 4 | 60+ | 词表中未训练条目 |
| 其他模型 | 13 | 448+ | 混合来源 |
| **总计** | **40** | **868** | — |

### 消融实验

粘性 token 对下游任务的影响评估：

| 任务 | 指标 | 无粘性token | 插入1个 | 插入5个 | 性能下降 |
|------|------|-----------|--------|--------|---------|
| 文本聚类 | NMI | 正常 | -15% | -35% | 最高50% |
| 语义检索 | MRR@10 | 正常 | -10% | -30% | 显著 |
| STS 相似度 | Spearman | 正常 | -8% | -25% | 线性增长 |

### 关键发现

- **粘性 token 来源分析**：主要来自三类——(1) 词表中的特殊/未使用条目（如 BERT 的 `[unused]` token），(2) 多语言语料中的碎片子词（如日文、阿拉伯文等低频文字片段），(3) 控制字符和特殊符号
- **模型规模与粘性 token 数量不相关**：大模型不一定有更少的粘性 token，说明问题根源在 tokenizer 设计而非模型容量
- **注意力分析**：粘性 token 在注意力层中获得了不成比例的权重，"吸引"了其他 token 的注意力，从而主导了最终的均值池化表示

## 亮点与洞察

- **STD 检测方法简洁高效**：仅需少量种子句子和标准差计算即可完成检测，无需访问模型内部权重。这种黑盒检测思路可直接应用于任何嵌入模型的质量审计
- **揭示了 tokenizer 设计的系统性缺陷**：粘性 token 的广泛存在说明当前主流 tokenizer（BPE/WordPiece/SentencePiece）在处理未充分训练的子词时存在根本性问题，对嵌入模型的可靠性构成威胁
- **注意力机制的脆弱性**：粘性 token 主导注意力权重的现象揭示了 mean pooling + self-attention 组合在面对异常输入时的脆弱性，这个洞察可迁移到对抗攻击和模型鲁棒性研究

## 局限与展望

- **缺少修复方案**：论文侧重于检测和分析，未提出有效的缓解策略（如 token 过滤、注意力正则化等）
- **仅限于文本嵌入模型**：未讨论生成式模型（如 GPT 系列）是否存在类似现象
- **检测阈值的选择**：标准差阈值需人工设定，不同模型可能需要不同阈值，缺乏自适应机制
- 改进方向：可以探索在 tokenizer 训练阶段引入粘性检测作为正则化，或在推理阶段动态过滤异常 token

## 相关工作与启发

- **vs Token 异常检测**：此前的 token 异常研究多集中在对抗攻击（adversarial triggers），着眼于刻意制造的攻击输入；本文发现这些异常是模型自身词表的固有缺陷
- **vs 嵌入质量评估**：MTEB 等 benchmark 侧重于下游任务性能，未检测嵌入空间的结构性缺陷；STD 提供了互补的质量评估维度
- 本文的核心发现对所有依赖文本嵌入的系统（RAG、语义搜索、聚类）都有实际警示意义

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统定义和检测粘性token，现象本身很有启发性
- 实验充分度: ⭐⭐⭐⭐ 40个检查点的大规模分析覆盖面广，但缺少修复实验
- 写作质量: ⭐⭐⭐⭐ 定义清晰、分析系统，结构良好
- 价值: ⭐⭐⭐⭐ 对嵌入模型部署和tokenizer设计有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens](a_text_is_worth_several_tokens_text_embedding_from_llms_secretly_aligns_well_wit.md)
- [\[ACL 2025\] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation](dont_reinvent_the_wheel_efficient_instruction-following_text_embedding_based_on_.md)
- [\[ACL 2025\] Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Space](psycholinguistic_visual_semantic.md)

</div>

<!-- RELATED:END -->
