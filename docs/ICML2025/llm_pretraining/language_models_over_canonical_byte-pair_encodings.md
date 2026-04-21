---
title: >-
  [论文解读] Language Models over Canonical Byte-Pair Encodings
description: >-
  [ICML 2025][BPE tokenization] 揭示 BPE 分词下自回归语言模型给指数级数量的非规范 (noncanonical) token 编码分配了不必要的概率质量，提出基于有限状态自动机 (FSA) 的条件化与构造化两套修复方案，在多种模型和语料上一致提升 held-out 似然。
tags:
  - ICML 2025
  - BPE tokenization
  - canonical encoding
  - language modeling
  - probability mass
  - FSA
---

# Language Models over Canonical Byte-Pair Encodings

**会议**: ICML 2025  
**arXiv**: [2506.07956](https://arxiv.org/abs/2506.07956)  
**代码**: [GitHub](https://github.com/genlm/canonical-icml-2025)  
**领域**: LLM Pretraining  
**关键词**: BPE tokenization, canonical encoding, language modeling, probability mass, FSA

## 一句话总结

揭示 BPE 分词下自回归语言模型给指数级数量的非规范 (noncanonical) token 编码分配了不必要的概率质量，提出基于有限状态自动机 (FSA) 的条件化与构造化两套修复方案，在多种模型和语料上一致提升 held-out 似然。

## 研究背景与动机

**领域现状**：现代语言模型 (LM) 通过确定性分词器（如 BPE）将字符串映射为 token 序列，模型在 token 空间上建立概率分布。这一范式是当前几乎所有大规模语言模型训练的基石。

**现有痛点**：BPE 分词器对每个字符串仅生成一种**规范编码**，但同一个字符串可以被分解为指数级数量的不同 token 序列（非规范编码），这些序列虽然能解码回原始字符串，却永远不会出现在任何训练语料中。然而当前模型仍然对这些不可能出现的序列分配了非零概率。

**核心矛盾**：非规范编码的概率分配同时具有两个问题——
1. **错误性**：这些序列在训练数据中不存在，模型不应该给它们正概率
2. **浪费性**：分配给非规范编码的概率质量被从正确的规范序列中"偷走"，降低了模型对合理输出的概率预测

**本文目标** 如何强制让 token 级语言模型只给规范 BPE 编码分配正概率，消除上述浪费和错误。

**切入角度**：将 BPE 分词器的规范性约束形式化为有限状态自动机 (FSA) 的可达性条件，在此基础上设计推理时和训练时的两种修复策略。

**核心 idea**：用 FSA 精确刻画 BPE 规范编码集合，通过条件化推理或参数化约束消除对非规范序列的概率分配。

## 方法详解

### 整体框架

作者提出两种互补方案来实现规范性保证：
1. **Canonicality by Conditioning（条件化方法）**：推理时约束，无需额外训练
2. **Canonicality by Construction（构造化方法）**：模型参数化保证，需重新训练

### 关键设计

1. **规范性 FSA 构造**:
    - 功能：精确描述 BPE 规范编码集合的形式语言约束
    - 核心思路：BPE 的贪心合并规则本质上是一个确定性过程，可以用 FSA 的状态转移来捕捉。在每一步解码时，FSA 维护当前前缀是否仍在规范路径上的状态，只有使得前缀保持规范性的 token 才被允许。FSA 的复杂度与词表大小成线性关系
    - 设计动机：形式语言理论提供了精确且高效的工具来表达确定性分词器的约束，FSA 是表达力恰好匹配的工具——既足够强大又计算高效

2. **条件化方法 (Conditioning)**:
    - 功能：推理时直接使用 FSA 屏蔽不合法的 token
    - 核心思路：在自回归解码每一步中，查询 FSA 当前状态获取合法 token 集合，将非法 token 的 logit 设为 $-\infty$，然后对合法 token 重新归一化概率。等价于计算条件概率 $p_{\text{canonical}}(t | \mathbf{t}_{<i}) = p(t | \mathbf{t}_{<i}) / Z$，其中 $Z = \sum_{t' \in \text{legal}} p(t' | \mathbf{t}_{<i})$
    - 设计动机：类似于 constrained decoding 的思路，无需重新训练即可修正任何现有模型

3. **构造化方法 (Construction)**:
    - 功能：在模型输出层嵌入规范性约束，使非规范序列概率恒为零
    - 核心思路：将 FSA 直接编码进输出概率分布的参数化中。对输出 logit 进行 mask，确保在任何状态下都恒满足 $p(\text{noncanonical token}) = 0$。模型从头训练或微调时学习在规范子空间上分配概率
    - 设计动机：提供更强的理论保证——训练过程中也不浪费概率质量

### 损失函数 / 训练策略

- **条件化方法**：零训练成本，直接在已有模型上应用
- **构造化方法**：使用标准交叉熵损失训练，但输出层包含 FSA 约束的 mask。训练时模型只在规范 token 子集上计算 softmax
- 两种方法可以独立使用，也可以组合使用

## 实验关键数据

### 主实验：规范性修正对困惑度的改善

| 模型 | 语料 | 原始 PPL | 条件化 PPL | 改善比例 |
|------|------|----------|-----------|---------|
| GPT-2 Small (124M) | WikiText-103 | 基线 | 降低 | ~0.5-1% |
| GPT-2 Medium (355M) | WikiText-103 | 基线 | 降低 | ~0.3-0.8% |
| GPT-2 Large (774M) | WikiText-103 | 基线 | 降低 | ~0.2-0.5% |
| 定制小模型 | 多语料 | 基线 | 构造化最佳 | ~1-3% |

### 非规范概率质量分析

| 词表大小 | 非规范编码数量级 | 浪费概率质量 | 说明 |
|----------|-----------------|-------------|------|
| 小词表 (~1K) | 较少 | 较小 | 合并规则少，非规范路径少 |
| 中等词表 (~10K) | 显著 | 中等 | 实际可观的质量浪费 |
| 大词表 (~50K+) | 指数增长 | 最大 | 大词表问题最严重 |

### 关键发现

- 非规范编码问题在所有测试的模型和语料上普遍存在
- 修正后 held-out 数据的似然一致改善，证明浪费的概率质量确实被有效回收
- 词表越大，非规范编码越多，修正带来的改善越显著
- 条件化方法虽然免训练，但会引入推理开销；构造化方法无推理开销但需训练
- 不同语言（英语、多语言）和不同领域的语料都有改善

## 亮点与洞察

1. **揭示了一个长期被忽视的系统性问题**：BPE + 自回归 LM 的组合存在概率漏洞，这不是个别模型的问题，而是范式层面的缺陷
2. **FSA 形式化优雅而高效**：将直觉性的"规范性"概念转化为精确的形式语言约束
3. **两套方案互补性强**：条件化方法适合已训练模型的即插即用修正，构造化方法适合新训练模型
4. **问题的规模令人惊讶**：非规范编码数量与字符串长度呈指数关系增长

## 局限与展望

- 条件化方法在推理时需要维护 FSA 状态，增加了计算开销，特别是在大词表下
- 构造化方法需要从头训练或大规模微调，对于 GPT-4 级别的大模型不太实际
- 论文主要在中小规模模型上验证，对百亿参数以上模型的效果未知
- 未量化对下游任务（如问答、摘要）的具体影响，仅评估了困惑度
- 改善幅度在大模型上趋于减小，是否存在"大模型自动学会了避免非规范编码"的可能性未深入探讨

## 相关工作与启发

- **BPE Dropout / Subword Regularization**：这些工作故意引入非规范编码进行数据增强，而本文指出这些编码不应有概率——两者视角截然不同
- **Constrained Decoding**：条件化方法与 constrained generation 技术密切相关，但约束来源不同（规范性 vs 语法/格式）
- **分词器选择对 LM 的影响**：本文进一步证实分词器不只影响效率，还影响概率分布的正确性
- 启发：其他确定性预处理步骤（如文本归一化）是否也存在类似的概率漏洞？

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 发现并形式化了一个普遍却被忽视的概率论层面缺陷
- 实验充分度: ⭐⭐⭐⭐ 多模型多语料验证，但缺少超大规模模型和下游任务评估
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰，FSA 构造过程自然流畅
- 价值: ⭐⭐⭐⭐ 对 tokenization 和 LM 基础理论有重要贡献，但实际部署收益有限

<!-- RELATED:START -->

## 相关论文

- [Large Language Models are Demonstration Pre-Selectors for Themselves](large_language_models_are_demonstration_pre-selectors_for_themselves.md)
- [Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](../../NeurIPS2025/llm_pretraining/broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)
- [The Curse of Depth in Large Language Models](../../NeurIPS2025/llm_pretraining/the_curse_of_depth_in_large_language_models.md)
- [Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [Scalable Fingerprinting of Large Language Models](../../NeurIPS2025/llm_pretraining/scalable_fingerprinting_of_large_language_models.md)

<!-- RELATED:END -->
