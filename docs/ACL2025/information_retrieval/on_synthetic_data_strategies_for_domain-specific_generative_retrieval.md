---
title: >-
  [论文解读] On Synthetic Data Strategies for Domain-Specific Generative Retrieval
description: >-
  [ACL2025][生成式检索] 本文系统研究了针对领域特定语料库训练生成式检索模型的合成数据策略，提出多粒度查询生成、约束条件查询和基于硬负样本的偏好学习方法，显著提升检索性能。
tags:
  - ACL2025
  - 生成式检索
  - 合成数据
  - 偏好学习
  - 领域适配
  - 文档标识符
---

# On Synthetic Data Strategies for Domain-Specific Generative Retrieval

**会议**: ACL2025  
**arXiv**: [2502.17957](https://arxiv.org/abs/2502.17957)  
**代码**: 未公开  
**领域**: others (信息检索)  
**关键词**: 生成式检索, 合成数据, 偏好学习, 领域适配, 文档标识符  

## 一句话总结

本文系统研究了针对领域特定语料库训练生成式检索模型的合成数据策略，提出多粒度查询生成、约束条件查询和基于硬负样本的偏好学习方法，显著提升检索性能。

## 研究背景与动机

生成式检索（Generative Retrieval）是信息检索领域的新兴范式，利用生成模型直接产生与用户查询相关的文档标识符（document identifiers），而非传统的稠密检索方法依赖外部索引。尽管已有工作在训练策略、建模技术和推理方法上取得进展，但**数据策略**——特别是在领域特定语料库上——的作用仍被严重忽视。

生成式检索模型需要将整个语料库"内化"到参数记忆中，因此训练数据的选择和质量至关重要。现有工作主要沿用 DSI-QG 范式，使用 docT5query 从段落生成合成查询，但这种"一刀切"的数据策略在迁移到新领域时往往不够。

与稠密检索不同，生成式检索模型需要同时具备三种核心能力：

**记忆能力**：存储语料库内容并将其映射到文档标识符

**泛化能力**：超越用户查询的显式文本线索进行推断

**相关性打分**：准确排序文档标识符的相关性

领域特定语料库会放大这些挑战，模型必须在适应领域细微差别的同时保持稳健的泛化和排序能力。

## 方法详解

### 整体框架

本文引入了**两阶段训练框架**（见 Figure 1）：

- **第一阶段（监督微调）**：学习将输入映射到文档标识符，重点提升记忆和泛化能力
- **第二阶段（偏好学习）**：通过排序优化进一步增强文档排名性能

### 文档标识符设计

主要使用**语义文档标识符**（semantic identifiers），采用基于关键词的方法：用 LLM 为每篇文档生成描述其内容的关键词列表作为标识符。此外，也扩展验证了**原子标识符**（atomic identifiers，即唯一 token 一步解码）的泛化性。

### 第一阶段：监督微调数据策略

合成数据包含两部分：

#### Context2ID（语境到标识符）
将语料库中的每个文本块与对应文档标识符配对，帮助模型"记忆"文档内容。训练目标不仅优化输出序列（文档标识符），还包括学习解码输入内容，完整损失函数为：

$$\mathcal{L}_{\text{sft}}(q,d) = -\sum_i \log P(q_i | q_{<i}; \theta) - \sum_i \log P(d'_i | d'_{<i}, q; \theta)$$

#### Query2ID（查询到标识符）

使用 LLM（而非 docT5query）进行合成查询生成，具体包括三种策略：

**1. 多粒度查询生成（Multi-Granular Query Generation）**
- **块级查询**：以整个文本块为输入，生成 $m_c$ 个捕获高层语义的查询
- **句级查询**：以单个句子为输入，生成 $m_s$ 个聚焦局部细节的查询

**2. 约束条件查询生成（Constraints-Based Query Generation）**
利用 LLM 的指令遵循能力，在生成查询时融入领域特定的元数据约束（如作者姓名、政治倾向等），生成更专业的领域查询。每篇文档生成 $m_i$ 个带约束的查询。

**3. Context2ID 与 Query2ID 的组合策略**
采用交错（Interleave）方式而非简单拼接（Concat），对较小的 Context2ID 数据进行上采样。

### 第二阶段：偏好学习数据策略

#### 偏好优化目标
采用 **RPO（Regularized Preference Optimization）** 作为排序优化方法。RPO 是 DPO 的扩展版本，增加了监督微调损失以缓解对负样本的过度优化问题。

#### 合成查询策略调整
- 要求 LLM 生成**尽可能困难**的查询
- 同时要求提供查询的**对应答案**，确保困难查询仍然可回答
- 与第一阶段的查询有所区分，避免模型对同一批数据过度优化

#### 负样本候选选择
关键创新在于从模型自身的检索结果中选择负样本：
- 用第一阶段模型对偏好学习的合成查询进行检索
- 选择排名高于正样本的 top-k 负样本候选
- 若正样本已排第一则跳过该查询
- 每个负样本与正样本配对形成训练实例

## 实验

### 实验设置
- **数据集**：MultiHop-RAG、AllSides、AGNews（三个领域特定）+ Natural Questions（通用）
- **基模型**：Mistral 7B 系列
- **查询生成**：Mixtral 8x7B
- **关键词生成**：Claude 3 Sonnet

### 主实验结果

#### 多粒度查询的效果（MultiHop-RAG）

| 方法 | HIT@4 | HIT@10 | MAP@10 | MRR@10 |
|------|-------|--------|--------|--------|
| Chunk only | 43.64 | 66.65 | 13.98 | 31.14 |
| +Sent | **61.64** | **81.69** | **22.13** | **47.20** |

句级查询带来了约 18 个百分点的 HIT@4 提升。

#### 约束条件查询的效果

在 MultiHop-RAG 上 HIT@4 从 61.64 提升至 69.98，在 AllSides 上 HIT@1 从 10.19 提升至 14.20。

#### Context2ID 数据的效果

移除 Context2ID 后，MultiHop-RAG 上 HIT@4 从 69.98 骤降至 41.33；交错组合方式（69.98）远优于简单拼接（44.30）。

#### 偏好学习阶段

| 策略 | HIT@4 | HIT@10 | MRR@10 |
|------|-------|--------|--------|
| SFT only | 69.98 | 88.34 | 52.29 |
| Random 5 负样本 | 58.94 | 82.88 | 43.53 |
| Top-5 负样本 | 71.53 | 89.62 | 55.40 |
| Top-10 负样本 | **71.88** | **89.80** | 54.94 |

随机负样本反而损害性能，高质量硬负样本能稳定提升。

### 消融实验

#### LLM vs docT5query 查询生成
Mixtral 8x7B 生成的合成查询在 MultiHop-RAG 上 HIT@4 为 61.64，远超 docT5query 的 50.86。Jaccard 相似度分析进一步验证 LLM 查询与真实查询分布更接近。

#### 原子标识符泛化性
在原子标识符上的消融结果与语义标识符一致，三种数据类型均贡献显著，其中句级查询贡献最大。

#### 与现成检索器比较
生成式检索模型仅依赖领域内合成数据训练（无检索预训练），已能与 BM25、BGE-large、E5-Mistral-7B 等检索器达到可比甚至更优的性能。

## 亮点与洞察

1. **多粒度+约束条件的合成数据策略**：系统性地挖掘了 LLM 在合成查询生成方面的优势，不同粒度和约束条件的查询互补性强
2. **Context2ID 的重要性**：通过将文档内容记忆作为训练目标的一部分，显著增强了生成式检索的记忆能力
3. **RPO + 硬负样本**：证明了在偏好学习中负样本质量的关键性，随机负样本不仅无用反而有害
4. **数据策略的可迁移性**：所提策略在不同类型标识符（语义/原子）和不同领域间都有效

## 局限性

1. 合成查询主要基于单个文档，未涉及需要多文档推理的复杂查询
2. 未探讨增量学习或泛化到未见文档的场景
3. 合成数据策略仅验证于生成式检索，未系统比较对稠密检索微调的效果
4. 偏好学习中负样本数量的最优选择尚需更深入研究

## 相关工作

- **生成式检索建模**：DSI、GENRE、SEAL、MINDER 等探索了标识符类型、排序损失、约束解码
- **合成查询生成**：InPars、GPL 等在稠密检索中应用合成数据，但生成式检索的数据策略研究不足
- **偏好优化**：DPO、RPO 等方法在 LLM 对齐领域广泛应用，本文将其引入检索排序

## 评分

⭐⭐⭐⭐ — 系统全面的数据策略研究，实验设计扎实且消融充分，对生成式检索的实际部署有重要指导意义。方法虽不复杂但有效，不足之处在于未涉及更复杂的多文档查询场景。

<!-- RELATED:START -->

## 相关论文

- [Understanding Synthetic Context Extension via Retrieval Heads](../../ICML2025/information_retrieval/understanding_synthetic_context_extension_via_retrieval_heads.md)
- [GENIUS: A Generative Framework for Universal Multimodal Search](../../CVPR2025/information_retrieval/genius_a_generative_framework_for_universal_multimodal_search.md)
- [RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information](raemollm_retrieval_augmented_llms_for_cross-domain_misinformation_detection_usin.md)
- [CART: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling](cart_a_generative_cross-modal_retrieval_framework_with_coarse-to-fine_semantic_m.md)
- [Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](../../CVPR2025/information_retrieval/preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)

<!-- RELATED:END -->
