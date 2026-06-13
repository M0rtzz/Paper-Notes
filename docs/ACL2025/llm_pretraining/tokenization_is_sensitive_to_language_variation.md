---
title: >-
  [论文解读] Tokenization is Sensitive to Language Variation
description: >-
  [ACL2025][预训练][tokenization] 系统研究了 BPE tokenizer 的三个关键设计选择（拟合语料、pre-tokenizer、词表大小）对语言变体鲁棒性任务和敏感性任务下游性能的差异化影响，并提出基于 logistic regression 的 task-aware tokenizer 评估指标，显著优于 Rényi efficiency 等 task-agnostic 指标。
tags:
  - "ACL2025"
  - "预训练"
  - "tokenization"
  - "BPE"
  - "language variation"
  - "pre-tokenizer"
  - "vocabulary size"
  - "robustness"
  - "sensitivity"
---

# Tokenization is Sensitive to Language Variation

**会议**: ACL2025  
**arXiv**: [2502.15343](https://arxiv.org/abs/2502.15343)  
**代码**: [nlpsoc/Tokenization-Language-Variation](https://github.com/nlpsoc/Tokenization-Language-Variation)  
**领域**: LLM预训练  
**关键词**: tokenization, BPE, language variation, pre-tokenizer, vocabulary size, robustness, sensitivity

## 一句话总结

系统研究了 BPE tokenizer 的三个关键设计选择（拟合语料、pre-tokenizer、词表大小）对语言变体鲁棒性任务和敏感性任务下游性能的差异化影响，并提出基于 logistic regression 的 task-aware tokenizer 评估指标，显著优于 Rényi efficiency 等 task-agnostic 指标。

## 研究背景与动机

**领域现状**：语言变体（拼写变体、方言、句法变化等）在自然语言中无处不在，且通常与地域、社会和语境因素系统性关联。BPE 是当前主流 LLM（Llama 3、GPT-4、DeepSeek v3 等）最常用的分词算法。

**现有痛点**：Tokenizer 对不常见的语言形式（如拼写变体 "doin" vs "doing"）处理不一致——常见形式可能是单一 token，而变体被切分为多个 subword。这种不一致可能损害下游 LLM 性能，但现有研究假设"对给定语言存在一个对所有任务都最好的 tokenizer"。

**核心矛盾**：不同类型的下游任务对语言变体有截然不同的需求——语义任务（如 NLI）需要对变体**鲁棒**（英式拼写和美式拼写应得到相同标签），而形式任务（如作者验证）需要对变体**敏感**（区分不同拼写风格是关键信号）。同一个 tokenizer 设置能否同时服务这两类需求？

**本文目标**：(RQ1) 同一组 tokenizer 设置是否在鲁棒性任务和敏感性任务上都表现良好？(RQ2) 能否用简单的 task-aware 指标替代 task-agnostic 指标来预测 tokenizer 的下游表现？

**切入角度**：预训练多个 BERT-base 模型，系统变化 BPE 的三个参数（拟合语料、pre-tokenizer、词表大小），在两类任务上进行对照实验，同时提出 logistic regression 作为新的评估指标。

**核心 idea**：Tokenizer 的最佳设置因任务是否需要对语言变体鲁棒/敏感而不同，pre-tokenizer 是影响最大的设计选项。

## 方法详解

### 整体框架

为每种 tokenizer 配置预训练 BERT-base（110M 参数），在 750M tokens 上训练，然后在两类下游任务上进行微调评估。每种配置训练 3 个不同种子的模型，确保统计显著性。

### 关键设计一：Tokenizer 三维度变化

**功能**：系统变化 BPE tokenizer 的三个参数。
**为什么**：隔离各参数的独立影响。
**怎么做**：

- **拟合语料**：PubMed（生物医学）、Wikipedia（百科）、Twitter（社交媒体）、Miscellaneous（混合多源），各约 15 亿词
- **Pre-tokenizer**（5 种）：None（NO）、仅空格分词（WS）、保留前导空格（_WS）、Llama 3 风格（LLAMA3）、GPT-2 风格（GPT2）
- **词表大小**：500、4k、32k、64k、128k

默认设置为 Miscellaneous 语料 + GPT2 pre-tokenizer + 32k 词表，每次只变一个参数。

### 关键设计二：两类评估任务

**功能**：设计并编译两类任务集。
**为什么**：分别测试对语言变体的鲁棒性和敏感性。
**怎么做**：

- **鲁棒性任务**：GLUE（SST-2、QQP、MNLI、QNLI）+ GLUE+typo（TextFlint 注入拼写错误）+ GLUE+dialect（Multi-VALUE 注入方言变换，覆盖 5 种英语方言）
- **敏感性任务**：AV（作者验证，40.8k 训练样本）、PAN（多作者写作风格分析）、CORE（语域分类）、NUCLE（语法错误分类）、Dialect（方言分类）

### 关键设计三：Logistic Regression 评估指标

**功能**：提出 task-aware 的 tokenizer 评估方法。
**为什么**：传统指标（Rényi efficiency、Corpus Token Count）是 task-agnostic 的，对同一语料上的不同任务给出相同预测，与实际情况不符。
**怎么做**：使用 tokenizer 词表作为特征集，bag-of-token 方式进行 logistic regression，将任务标签作为因变量。对双文本输入任务，使用句子对间的 token 组合作特征。

### 训练策略

- 预训练：BERT-base 110M 参数，512 序列长度，batch size 32，45k steps
- 微调：3 epochs，max_seq_len=128，batch_size=32，lr=2e-5
- 显著性检验：McNemar test + Bonferroni correction

## 实验关键数据

### 主实验：鲁棒性任务表现（Table 2）

| 设置 | GLUE | +typo | +dialect | AVG |
|------|------|-------|----------|-----|
| **拟合语料** | | | | |
| Twitter | **81.1** | 69.1 | 78.8 | **76.4** |
| PubMed | 80.8 | **69.1** | 78.6 | 76.2 |
| Wikipedia | 80.7 | 68.6 | **79.3** | 76.2 |
| **Pre-tokenizer** | | | | |
| GPT2 | **81.3** | 68.2 | 79.2 | 76.2 |
| _WS | 80.8 | **68.9** | 79.0 | 76.2 |
| NO | 72.1 | 61.6 | 70.1 | 67.9 |
| **词表大小** | | | | |
| 500 | 77.2 | **70.3** | 75.6 | 74.4 |
| 32k | **81.3** | 68.2 | **79.2** | **76.2** |
| 128k | 78.7 | 64.6 | 76.1 | 73.1 |

### 主实验：敏感性任务表现（Table 3）

| 设置 | AV | PAN | CORE | NUCLE | Dialect | AVG |
|------|-----|-----|------|-------|---------|-----|
| **拟合语料** | | | | | | |
| Twitter | **82.9** | **66.7** | **56.5** | 21.4 | 88.3 | **63.2** |
| Wikipedia | 81.9 | 65.5 | 55.5 | **23.5** | **88.9** | 63.1 |
| **Pre-tokenizer** | | | | | | |
| _WS | 82.5 | 66.3 | **56.6** | 22.6 | 88.4 | **63.3** |
| GPT2 | **82.6** | 66.6 | 56.3 | 21.8 | 88.4 | 63.1 |
| NO | 81.8 | 59.9 | 51.7 | 16.3 | 77.3 | 57.4 |
| **词表大小** | | | | | | |
| 32k | 82.6 | 66.6 | **56.3** | 21.8 | **88.4** | **63.1** |
| 64k | **82.7** | **67.2** | 54.9 | **22.0** | 88.1 | 63.0 |
| 500 | 78.2 | 62.6 | 51.1 | 13.1 | 85.6 | 58.1 |

### 消融：Tokenizer 评估指标对比（Table 5）

| 指标 | 与鲁棒性任务相关性 | 与敏感性任务相关性 |
|------|-------------------|-------------------|
| Rényi Efficiency | -0.22 | -0.03 |
| Corpus Token Count | -0.45 | +0.37 |
| **Logistic Regression** | **0.85** | **0.84** |

### 关键发现

1. **Pre-tokenizer 影响最大**：在两类任务中，pre-tokenizer 的性能差异范围最大，不用 pre-tokenizer（NO）显著低于其他选项
2. **最佳设置因任务类型而异**：敏感性任务偏好更大的词表（32k-64k），鲁棒性任务中小词表（500）对拼写变体最鲁棒
3. **Twitter 语料出人意料地全面**：在两类任务上都表现良好，不仅限于敏感性任务
4. **Corpus Token Count 相关方向翻转**：对鲁棒性任务负相关（-0.45），对敏感性任务正相关（+0.37），证明 task-agnostic 指标的局限
5. **Logistic regression 与下游性能高度相关**（r=0.85/0.84），远优于传统 intrinsic 指标

## 亮点与洞察

- **新颖视角**：首次将下游任务按"对语言变体的鲁棒性 vs 敏感性"分类来系统评估 tokenizer，揭示了传统"one-size-fits-all"假设的不足
- **实用的三条建议**：(1) 最关注 pre-tokenizer；(2) 敏感性任务用更大词表；(3) 用小型 logistic regression 快速评估 tokenizer
- **低成本高价值的实验设计**：用 110M 参数的 BERT-base（<15 GPU hours/模型）即可得出有指导意义的结论，远比训练 350M-2.5B 模型经济
- **方法论贡献**：提出 task-aware 的 tokenizer 评估范式，打破了 tokenizer 评估只看压缩效率的传统思路

## 局限与展望

1. **模型规模限制**：仅用 110M BERT-base，无法保证结论推广到更大的 decoder-only 模型（如 Llama 3、GPT-4）
2. **语言变体类型未细分**：拼写变体、词汇变体、句法变体的影响被混合在一起，未来应分别研究
3. **参数交互未探索**：每次只变一个参数，未探索拟合语料×pre-tokenizer×词表大小的交互效应
4. **语言局限**：仅研究英语，不同文字系统（如中文、阿拉伯语）可能有不同结论
5. **任务范围有限**：鲁棒性仅用 GLUE；敏感性任务按"语义 vs 形式"的二分法可能过于简化
6. **Logistic regression 局限**：仅适用于分类任务，不适用于文本生成等复杂任务

## 相关工作与启发

### vs Ali et al. (2024) — Tokenizer Choice for LLM Training

Ali et al. 在 350M-2.5B 参数模型上测试了不同 tokenizer，发现不同词表大小对同模型规模下性能差异很小。本文在更小模型上验证了类似发现，但关键创新是区分了两类任务——在鲁棒性 vs 敏感性任务上词表大小的最优值不同，挑战了"存在普适最佳 tokenizer"的假设。

### vs Schmidt et al. (2024) — BPE Tokenizer 的 NLU 评估

Schmidt et al. 也发现 pre-tokenizer 影响下游性能，但未考虑语言变体敏感性任务。本文扩展了评估范围，揭示同一 pre-tokenizer 在两类任务上表现不一致（如 GPT2 在原始 GLUE 上最好，但 _WS 在变体任务上表现更均衡）。

### vs Zouhar et al. (2023) — Rényi Efficiency

Zouhar et al. 提出 Rényi efficiency 作为 tokenizer 的 intrinsic 评估指标。本文发现其与下游性能的相关性很低（-0.22/-0.03），并提出 logistic regression（0.85/0.84）作为更可靠的替代方案。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 任务按语言变体鲁棒/敏感分类的视角新颖，logistic regression 评估指标实用
- **实验充分度**: ⭐⭐⭐⭐ — 系统性变量控制+显著性检验+两类任务充分对比，但未探索参数交互
- **写作质量**: ⭐⭐⭐⭐⭐ — 结构清晰，RQ 驱动，图表丰富且信息量大
- **价值**: ⭐⭐⭐⭐ — 对 tokenizer 设计有直接实践指导意义，适合作为 tokenizer 选型参考

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Adversarial Tokenization](adversarial_tokenization.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ACL 2025\] Incorporating Domain Knowledge into Materials Tokenization](incorporating_domain_knowledge_into_materials_tokenization.md)
- [\[NeurIPS 2025\] Differentiable Hierarchical Visual Tokenization](../../NeurIPS2025/llm_pretraining/differentiable_hierarchical_visual_tokenization.md)

</div>

<!-- RELATED:END -->
