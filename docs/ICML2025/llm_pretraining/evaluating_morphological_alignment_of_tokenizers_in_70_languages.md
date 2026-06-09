---
title: >-
  [论文解读] Evaluating Morphological Alignment of Tokenizers in 70 Languages
description: >-
  [ICML 2025][预训练][分词器评估] 扩展 MorphScore 评估框架至 70 种语言，系统研究分词器的形态边界对齐程度与下游任务性能之间的相关性，发现形态对齐仅能解释极少量的性能方差，且呈负相关，挑战了"形态对齐分词有利于模型性能"的主流假设。
tags:
  - "ICML 2025"
  - "预训练"
  - "分词器评估"
  - "形态对齐"
  - "多语言NLP"
  - "MorphScore"
  - "BPE"
  - "语言模型性能"
---

# Evaluating Morphological Alignment of Tokenizers in 70 Languages

**会议**: ICML 2025  
**arXiv**: [2507.06378](https://arxiv.org/abs/2507.06378)  
**代码**: [GitHub](https://github.com/catherinearnett/morphscore)  
**领域**: LLM预训练  
**关键词**: 分词器评估, 形态对齐, 多语言NLP, MorphScore, BPE, 语言模型性能

## 一句话总结

扩展 MorphScore 评估框架至 70 种语言，系统研究分词器的形态边界对齐程度与下游任务性能之间的相关性，发现形态对齐仅能解释极少量的性能方差，且呈负相关，挑战了"形态对齐分词有利于模型性能"的主流假设。

## 研究背景与动机

分词（tokenization）是语言建模的第一步，对训练效率、模型性能和推理成本均有显著影响。然而，如何有效评估分词器质量仍是一个开放问题。

现有分词器的内在评估指标主要包括：

**压缩度（Compression）**：如 fertility（每词 token 数）和 CTC（语料 token 总数），但已有研究表明压缩度与性能之间无稳健关联。

**Rényi 效率**：考虑了频率分布，但后续工作认为其不能全面衡量分词质量。

**形态对齐（Morphological Alignment）**：衡量 token 边界是否与语素边界一致。例如，英文单词 "books" 的理想分切为 [book + s]，而非 [boo + ks]。

关于形态对齐是否有利于模型性能，已有文献结论高度分歧。一些工作认为对齐有助于性能提升（Park 2020、Hofmann 2021 等），但另一些工作发现无显著帮助（Macháček 2018、Saleva & Lignos 2021）。原始 MorphScore 仅覆盖 22 种语言，且存在多项限制：不包含法语、德语等高资源语言；不包含上下文信息；不考虑词频。

本文的核心动机是：通过大幅扩展语言覆盖范围和参数灵活性，更准确地判断形态对齐是否真正影响模型性能。

## 方法详解

### 整体框架

本文的工作分为三个阶段：（1）基于 Universal Dependencies (UD) 树库创建 70 种语言的形态对齐评估数据集；（2）设计包含多种参数设置的评分函数；（3）将对齐分数与下游任务性能进行相关性分析。

### 评估数据集创建

对每种语言，从 UD 树库中提取多形素词（排除单一形素词），利用词形（wordform）和词元（lemma）确定分切方案：

- 通过识别词形与词元之间最长公共子序列确定词干（stem）
- 前后多余字符分别作为前缀和后缀
- 仅保留可通过拼接重组的规则形式（排除不规则变化和非拼接形态学）
- 该方法仅适用于**屈折语和黏着语**，不适用于闪族语（如阿拉伯语）和孤立语（如中文）

最终创建了 86 种语言的数据集，过滤掉少于 100 条目的语言后保留 70 种。

### 评分函数

扩展原始 MorphScore，引入边界级和子词级两类指标：

- **边界指标（Boundary Metrics）**：评估预测分切是否正确识别了形素边界
    - 宏平均边界精确率和召回率
- **子词指标（Subword Metrics）**：评估预测子词是否与标准形素完全匹配
    - 微/宏平均子词精确率、召回率和 F1

举例：标准分切为 [book + s]，预测分切为 [boo + k + s]，则：
- 边界精确率 = 1/2（仅 k|s 边界正确），边界召回率 = 1/1
- 子词精确率 = 1/3（仅 "s" 完全匹配），子词召回率 = 1/2

### 参数设置实验

**频率加权**：是否根据词频对对齐分数加权。实验发现高频词更可能被形态对齐分切（Spearman $\rho = 0.119$, $p < 0.0001$）。

**单 token 词处理**：是否将被整体存储为单个 token 的词纳入评分。包含单 token 词时分数普遍更高。高频词更可能被整体存储（$\rho = -0.108$, $p < 0.0001$）。

**最优默认设置**：通过线性混合效应模型分析，发现频率加权 + 排除单 token 词的设置对模型性能具有略微更强的预测力。

### 与模型性能的相关性分析

使用五个预训练模型（Llama2 8B, BLOOM, XGLM 7.5B, Llama3, Gemma3）在七个下游任务（XCOPA, XNLI, SIB-200, MultiBLiMP 等）上的表现，通过线性混合效应模型检验形态对齐是否能解释额外方差。

控制变量包括模型参数量和各语言训练数据比例。使用 ANOVA 检验形态对齐是否提供额外解释力。

## 实验关键数据

### 分词器形态对齐主实验

| 分词器 | 召回率 | 精确率 |
|--------|--------|--------|
| BLOOM | 0.33 ± 0.00 | 0.11 ± 0.00 |
| Gemma3 | 0.35 ± 0.00 | 0.12 ± 0.00 |
| Llama2 | 0.56 ± 0.00 | 0.13 ± 0.00 |
| Llama3 | 0.45 ± 0.00 | 0.12 ± 0.00 |
| XGLM | 0.52 ± 0.00 | 0.23 ± 0.00 |

XGLM 在精确率上一致表现最佳；Llama2 召回率最高但这主要源于过度分切。

### 相关性分析结果

- 召回率作为形态对齐指标时，能解释额外方差（$\chi^2(1) = 391.42, p < 0.001$），但精确率不能（$\chi^2(1) = -6.99, p = 1$）
- 整体解释力极低：召回率 $R^2 = 0.024$，精确率 $R^2 = 0.005$
- **相关性方向为负**——形态对齐越高，性能反而越低
- 这与 Arnett & Bergen (2025) 的发现一致

### 过度分切与精确率

使用准确率（accuracy）度量时，字符级分切可以获得完美分数——这是误导性的。Llama 分词器在非拉丁文字语言上经常过度分切（分切到字节级别），导致高召回率但低精确率。本文因此推荐使用精确率和召回率而非准确率。

## 亮点与洞察

1. **挑战主流假设**：在 70 种语言、5 个模型、7 个任务的大规模实验中，形态对齐仅解释了不到 2.5% 的性能方差，且方向为负，直接质疑了"形态对齐分词有利于模型性能"的广泛假设。

2. **评估指标选择至关重要**：使用准确率作为形态对齐度量会严重误导结果（过度分切可获高分），精确率能有效惩罚过度分切，是更合理的指标。

3. **形态对齐可能需与其他指标结合**：单独的形态对齐不足以评判分词质量，未来可能需要与压缩度、Rényi 效率等指标组合使用。

4. **数据集灵活性强**：新数据集包含上下文、词性信息和形态学标注，支持按词性分析等细粒度研究。

## 局限性

- 语言样本仍以欧洲语言为主，闪族语和孤立语由于非拼接形态学被排除
- 形态边界的操作化较为粗糙，主要覆盖屈折形态学
- 下游任务数量有限，且多集中于高资源语言
- 仅考虑自回归 LM，排除了编码器模型
- 模型样本有限，因多数模型不公开训练数据比例

## 相关工作

- **分词器评估**：Rust 2021 (fertility)、Schmidt 2024 (CTC)、Zouhar 2023 (Rényi 效率)
- **形态对齐研究**：Batsuren 2024（分类评估）、Arnett & Bergen 2025（原始 MorphScore）
- **跨语言分词公平性**：Ahia 2023、Petrov 2023

## 评分

⭐⭐⭐ — 实验规模大、结论明确，但核心发现偏向负面结果（形态对齐无用），方法新颖性有限。数据集和评估框架的贡献有实际价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Unsupervised Morphological Tree Tokenizer](../../ACL2025/llm_pretraining/unsupervised_morphological_tree_tokenizer.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](../../ACL2025/llm_pretraining/splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ICML 2025\] Tokenized Bandit for LLM Decoding and Alignment](tokenized_bandit_for_llm_decoding_and_alignment.md)
- [\[ICLR 2026\] Identifying and Evaluating Inactive Heads in Pretrained LLMs](../../ICLR2026/llm_pretraining/identifying_and_evaluating_inactive_heads_in_pretrained_llms.md)
- [\[ACL 2025\] Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases](../../ACL2025/llm_pretraining/between_circuits_chomsky.md)

</div>

<!-- RELATED:END -->
