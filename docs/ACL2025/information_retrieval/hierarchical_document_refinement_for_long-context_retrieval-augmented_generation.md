---
title: >-
  [论文解读] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation
description: >-
  [ACL 2025][RAG] 提出 LongRefiner，一个即插即用的长文档精炼系统，通过双层查询分析、层次化文档结构化和自适应精炼三个步骤，在 7 个 QA 数据集上以**仅 1/10 的 token 预算**实现了优于全文输入的性能，同时延迟仅为最佳基线的 1/10。
tags:
  - ACL 2025
  - RAG
  - 文档精炼
  - 层次化结构
  - XML格式
  - 长上下文
  - 多任务LoRA
---

# Hierarchical Document Refinement for Long-context Retrieval-augmented Generation

**会议**: ACL 2025  
**arXiv**: [2505.10413](https://arxiv.org/abs/2505.10413)  
**代码**: [GitHub](https://github.com/ignorejjj/LongRefiner)  
**领域**: 检索增强生成 / 长文本处理  
**关键词**: RAG, 文档精炼, 层次化结构, XML格式, 长上下文, 多任务LoRA

## 一句话总结

提出 LongRefiner，一个即插即用的长文档精炼系统，通过双层查询分析、层次化文档结构化和自适应精炼三个步骤，在 7 个 QA 数据集上以**仅 1/10 的 token 预算**实现了优于全文输入的性能，同时延迟仅为最佳基线的 1/10。

## 研究背景与动机

**领域现状**: RAG 系统通过检索外部文档增强 LLM 的知识覆盖和事实准确性。实际应用中搜索引擎返回的往往是完整长文档，包含大量与查询无关的信息。

**现有痛点**:
   - **信噪比低**: 长文档中有价值的信息被大量无关内容淹没，模型难以聚焦
   - **计算开销大**: 处理完整长文档显著增加输入上下文长度，推理成本高
   - **现有精炼方法的不足**: chunk-based 方法只能处理短文本片段，缺乏全局视角；perplexity-based 方法用困惑度评估 token 相关性过于粗糙

**核心矛盾**: 完整文档虽然包含所需信息但噪声太多，chunk-based 方法精简了长度但丢失了文档的整体结构和上下文关系。

**本文目标**: 如何**高效且保真**地精炼长文档，在大幅减少 token 数量的同时保留关键信息。

**切入角度**: 完整文档天然包含丰富的结构信息（逻辑关联、内容组织层次），利用这些结构可以实现比 chunk-based 方法更精准的信息抽取。

**核心idea**: 将非结构化长文档转化为层次化文档树，基于查询需求自适应精炼，用 XML 格式大幅压缩表示。

## 方法详解

### 整体框架

LongRefiner 集成三项能力于单个基础模型（通过多任务 LoRA 学习）：(1) 双层查询分析——判断查询需要局部还是全局信息；(2) 层次化文档结构化——将非结构化文档转化为文档树；(3) 自适应精炼——根据查询需求选择保留文档的哪些部分。

### 关键设计

1. **双层查询分析 (Dual-Level Query Analysis)**:

    - 定义两种信息层级：**Local Level**（仅需定位特定段落的局部知识）和 **Global Level**（需要广泛背景知识的全局理解）
    - 用教师 LLM 为训练集中的查询标注二值标签
    - 微调 refiner 模型预测特殊 token（[Local] 或 [Global]），对两者的生成概率做 softmax 得到连续的信息范围分数 $r_q$

2. **层次化文档结构化 (Hierarchical Document Structuring)**:

    - 将文档建模为**文档树** $D_{\text{str}} = (\mathcal{N}, \mathcal{R})$，节点为章节/子章节/段落，边为层级包含关系
    - **XML 格式表示**: 设计 `<section>`, `<subsection>`, `<skip>`, `<br>` 四种 XML 标签将树结构平铺为文本。`<skip>` 标签省略段落中间内容（仅保留首尾各 $k$ 个 token），将输出 token 数压缩至原文的约 **1/10**
    - 训练模型学习从原文 $D$ 到 $D_{\text{xml}}$ 的映射，推理时用原文和解析算法恢复完整文档树
    - 生成过程分两步：先生成层次结构（章节标题），再填充内容（用 `<skip>` 跳过中间部分）

3. **自适应精炼 (Adaptive Refinement)**:

    - 基于文档树结构，计算每个段落/章节与查询的相关性分数
    - 根据查询分析结果的信息范围 $r_q$ 自适应决定保留信息的多少
    - Local 类查询保留更少但更精准的内容，Global 类查询保留更多上下文

4. **Wikipedia 标签收集**: 利用 Wikipedia 文章天然的层次结构（标题、子标题、段落）作为 ground truth 训练文档结构化模型。

### 训练与推理优化

- **多任务 LoRA 学习**: 三个任务共享同一基础模型，通过 LoRA adapter 区分
- **离线/在线拆分**: 文档结构化在离线阶段完成（对语料库预处理），在线阶段仅需查询分析 + 自适应精炼，处理输入/输出 token 极少，延迟仅为标准设置的 25%

## 实验与关键数据

### 实验设置

- **数据集**: 7 个 QA 数据集——Single-hop (NQ, TriviaQA, PopQA), Multi-hop (HotpotQA, 2WikiMultiHopQA), Long-form (ASQA, ELI5)
- **生成器**: Llama3.1-8B-Instruct (64K context)
- **Refiner**: Qwen2.5-3B-Instruct + LoRA
- **检索**: 每个查询检索 top-8 完整 Wikipedia 文档
- **预算约束**: 精炼后限制为 2K tokens

### 主实验结果

文中提出的 LongRefiner 在所有 7 个数据集上取得最佳性能，关键发现：

1. **性能**: 所有数据集最优，超越 perplexity-based 方法（LongLLMLingua）9% 以上
2. **效率**: 延迟与检索方法相当，远低于 perplexity-based 方法
3. **vs 全文输入**: 在 6/7 数据集上用 **1/10 的 token** 即超越全文性能（PopQA 除外，因其文档短、噪声少）

### 消融实验 (Table 3)

| 方法 | Single-hop (EM) | Multi-hop (Acc) | Long-form (F1) |
|------|----------------|-----------------|----------------|
| LongRefiner | **62.3** | **37.4** | **30.2** |
| w/o Query Analysis | 60.3 | 36.2 | 29.6 |
| w/o Doc. Structuring | 45.7 | 29.9 | 27.1 |
| w/o Adaptive Refine. | 57.7 | 35.3 | 29.2 |

- 文档结构化模块最关键，移除后性能下降约 20%（退化为基础 chunking）
- 查询分析和自适应精炼各贡献约 2-3% 的提升

### 扩展分析

- **模型规模**: 更大的结构化模型带来更好的文档结构质量和 QA 性能
- **训练数据量**: 训练数据不足时模型倾向生成粗粒度结构（更少章节、更大内容块），看似召回率高但结构信息不准确
- **文档长度**: 短文档中 LongRefiner 优势不明显（因噪声本就少），长文档中显著优于全文方法和 LongLLMLingua
- **评分模型选择**: Reranker 最优，Embedding 模型次之但效率更高，BM25 最差
- **跨生成器验证**: 在 Qwen2.5-7B-Instruct 上同样有效

## 亮点与洞察

1. **文档结构是长文本精炼的关键**: 与基于困惑度或语义相似度的粗糙方法不同，利用文档内在层次结构可以实现更有意义的信息组织和精炼
2. **XML 格式的巧妙设计**: 用少量 XML 标签 + `<skip>` 跳过机制，将文档树的表示压缩到 1/10，大幅降低模型输出 token 数
3. **即插即用**: 作为独立的精炼模块，可以插入任何 RAG pipeline 中，不需要修改生成器
4. **离线/在线拆分**: 文档结构化的高成本部分离线完成，在线推理极轻量
5. **多任务共享**: 三个不同功能共享同一基础模型，通过 LoRA 实现资源高效

## 局限性

1. 仅处理纯文本，不支持含表格、图片、超链接的复杂文档
2. 基于 Wikipedia 训练，迁移到垂直领域（企业、金融等）可能需要重新标注
3. XML 解析中的错误会导致一定程度的信息丢失
4. PopQA 类短文档低噪声场景下精炼反而可能损害性能
5. Refiner 引入额外推理成本，虽然很小但在极低延迟要求场景下仍需考虑

## 相关工作

- **RAG**: RAG、RETRO、REALM 等将检索与生成结合
- **知识精炼方法**: 分为硬精炼（token 删除/摘要/chunk 选择）和软精炼（向量编码）两类
- **长文本处理**: LongLLMLingua 等基于困惑度的 token 压缩方法

## 评分

⭐⭐⭐⭐⭐ — 问题重要（长上下文 RAG 是实际部署中的核心瓶颈），方法设计优雅（层次化结构 + XML 压缩），实验全面（7 个数据集 + 多维分析），工程实用性强（即插即用 + 离线/在线拆分）。在 RAG 知识精炼方向上是一篇高质量的工作。

<!-- RELATED:START -->

## 相关论文

- [Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)
- [A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)
- [SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)
- [FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](faithfulrag_fact_level_conflict.md)

<!-- RELATED:END -->
