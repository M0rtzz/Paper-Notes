---
title: >-
  [论文解读] If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?
description: >-
   通过 Transformer Grammar (TG) 的注意力机制研究人类记忆检索的表征形式，发现基于句法结构的注意力(TG)与基于 token 序列的注意力(vanilla Transformer)对阅读时间预测有独立贡献，表明人类句子处理涉及双重记忆表征系统。
tags:

---

# If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?

- **会议**: ACL 2025
- **arXiv**: [2502.11469](https://arxiv.org/abs/2502.11469)
- **代码**: [GitHub](https://github.com/osekilab/TG-NAE)
- **领域**: 计算心理语言学 / 句法处理 / 认知建模
- **关键词**: Transformer Grammar, Normalized Attention Entropy, Memory Retrieval, Syntactic Structure, Reading Time Prediction

## 一句话总结

通过 Transformer Grammar (TG) 的注意力机制研究人类记忆检索的表征形式，发现基于句法结构的注意力(TG)与基于 token 序列的注意力(vanilla Transformer)对阅读时间预测有独立贡献，表明人类句子处理涉及双重记忆表征系统。

## 研究背景与动机

- **问题**: 最近研究表明 Transformer 的注意力机制可作为人类记忆检索(cue-based retrieval)的计算实现，但现有工作仅关注 vanilla Transformer 基于 token 级别的表征，忽略了句法结构在人类句子处理中的重要作用。
- **认知科学背景**: 心理语言学中有两大句子处理理论——基于预期的理论(expectation-based, 对应 surprisal)和基于记忆的理论(memory-based, 对应 cue-based retrieval)。Cue-based retrieval 理论认为，当遇到动词时需要从工作记忆中检索其论元，类似元素的存在会造成干扰效应(interference)。
- **核心假设**: 如果注意力机制是人类记忆检索的通用算法，那么在句法结构上操作的注意力机制(TG)应该比在 token 序列上操作的注意力(vanilla Transformer)能更好地捕捉人类记忆检索模式。

## 方法详解

### 整体框架

使用 **Normalized Attention Entropy (NAE)** 作为连接模型与人类的桥梁假设，比较 Transformer Grammar (TG) 和 vanilla Transformer 的注意力机制对自定步速阅读时间(self-paced reading times)的预测能力。

### 关键设计

1. **Transformer Grammar (TG)**: TG 是一种句法语言模型，联合生成 token 序列和句法结构。其核心创新在于 **COMPOSE 注意力机制**：当一个句法短语闭合(X))时，COMPOSE 将短语内所有元素压缩为单一向量表征，后续的 STACK 操作直接引用该短语表征进行预测。这使得注意力操作的记忆单元是**句法结构**(短语)而非 token。

2. **NAE 计算方法**: NAE 衡量注意力权重的扩散程度——NAE 越高表示注意力越分散，对应更严重的检索干扰。计算时对注意力权重进行重归一化(排除自身注意力)并除以最大熵进行标准化。仅使用顶层的所有头的 NAE 之和。

3. **TG-specific 设计决策**: (a) 使用 "perfect oracle" 句法结构(数据集提供的金标准 parse tree)；(b) 仅考虑来自词汇 token 的注意力，排除非词汇符号(如 (NP、NP))的认知负荷归属。

### 统计分析

使用线性混合效应模型(LME)预测对数阅读时间，基线模型包含词位置、词长、n-gram 频率、surprisal、stack count 等控制变量。通过似然比检验(ΔLogLik)评估添加 NAE 后的预测改善。

## 实验

### 主实验

| 模型 | ΔLogLik ↑ | NAE 效应量 (ms) | NAE_so 效应量 (ms) | 显著种子数 |
|------|----------|---------------|-------------------|-----------|
| **TG** | **76.6 (±8.1)** | 1.42 (±0.2)*** | 2.26 (±0.1)*** | 3/3 |
| Transformer | 42.8 (±9.5) | 1.32 (±0.2)*** | 1.46 (±0.2)*** | 3/3 |

(平均阅读时间 334ms，效应量以每标准差对应的毫秒数表示)

### 消融实验

**COMPOSE 注意力的作用** (加入 Transformer NAE 作为基线控制):

| 模型 | ΔLogLik |
|------|---------|
| TG (完整) | 46.1 (±9.1) |
| TG−comp (无 COMPOSE) | 18.1 (±9.3) |

似然比检验: TG 解释了 TG−comp 无法解释的方差 (p<0.001)，反之则不成立 (p=0.478)。

**干扰 vs 衰减效应独立性**:

| 预测变量 | 效应量 (ms) |
|---------|-----------|
| tg_nae | 1.18*** |
| tg_nae_so | 2.38*** |
| clt (Category Locality Theory) | 0.06 (n.s.) |
| clt_so | 1.30*** |

TG NAE 和 CLT 的贡献相互独立 (p<0.001)。

### 关键发现

1. **TG 的 NAE 预测力显著优于 vanilla Transformer** (ΔLogLik 76.6 vs 42.8)，表明基于句法结构的记忆检索在人类句子处理中占主导地位
2. **两种模型具有独立贡献**: TG 在动词(VB, VBG, VBN, VBP)上优势明显，Transformer 在名词(NN, NNP)上更好——分别对应句法驱动和语义驱动的检索操作
3. **COMPOSE 注意力是关键**: 将闭合短语压缩为单一表征是 TG 优势的核心来源，尤其在动词触发的论元检索中
4. **NAE 量化的是干扰效应而非衰减效应**: TG NAE 与 Category Locality Theory (衰减模型)的贡献独立

## 亮点

- **理论贡献深刻**: 首次提供证据表明人类句子处理涉及双重记忆表征(句法结构 + token 序列)，注意力机制作为通用检索算法在两者上运作
- **实验设计严谨**: 控制了 surprisal、stack count 等混淆变量，使用多随机种子、溢出效应建模、似然比嵌套检验
- **连接 NLP 与认知科学**: 从 Marr 三层描述的视角论证了注意力机制作为记忆检索的算法层面解释

## 局限性

- 仅使用英语 Natural Stories 语料库(10 个故事, 10,245 词)，语言和文本类型的泛化性未验证
- 使用 "perfect oracle" 句法结构，回避了人类在线处理中的局部歧义解消问题
- NAE 计算方式(顶层求和、子词聚合)可能不是最优的，替代方案未充分探索
- TG 使用 top-down 解析策略，而心理语言学认为 left-corner 策略可能更接近人类处理

## 相关工作

- **Cue-based retrieval**: Van Dyke & Lewis (2003) 的干扰效应理论
- **注意力与记忆**: Ryu & Lewis (2021) 提出 Attention Entropy, Oh & Schuler (2022) 提出 NAE
- **句法语言模型**: Transformer Grammar (Sartran et al., 2022)、RNNG (Dyer et al., 2016)
- **认知建模**: Surprisal theory (Hale, 2001; Levy, 2008)、Dependency Locality Theory (Gibson, 1998)

## 评分

- **新颖性**: 8/10 — 将 TG 的句法注意力引入认知建模是全新视角，双重记忆表征的发现具有原创性
- **技术深度**: 7/10 — 方法论上较为标准(LME 回归)，但实验设计和控制变量处理非常严谨
- **实验充分度**: 7/10 — 多种消融分析和独立性检验，但语料库和语言覆盖较窄
- **清晰度**: 8/10 — 认知科学概念解释清楚，图表辅助理解，论文组织逻辑清晰
- **总分**: 7.5/10

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness](mindref_mimicking_human_memory_hierarchical_reference_retrieval.md)
- [\[ACL 2025\] Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition](developmentally-plausible_working_memory_shapes_a_critical_period_for_language_a.md)
- [\[ACL 2025\] EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts](epman_episodic_memory_attention_for_generalizing_to_longer_contexts.md)
- [\[ACL 2025\] Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)
- [\[ACL 2025\] In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents](in_prospect_and_retrospect_reflective_memory_management_for_long-term_personaliz.md)

<!-- RELATED:END -->
