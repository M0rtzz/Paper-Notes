---
title: >-
  [论文解读] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches
description: >-
  [ACL 2026][检索增强生成] 本文在四个数据集上从用户意图处理、查询重写、文档精炼和底层 LLM 选择四个维度系统对比了 Enhanced RAG 和 Agentic RAG，发现两者各有优势——Agentic RAG 在意图路由和查询重写上更灵活，Enhanced RAG 在文档重排上更有效，而 Agentic RAG 的成本高达 3.3 倍。
tags:
  - ACL 2026
  - 检索增强生成
  - Agentic RAG
  - Enhanced RAG
  - 查询重写
  - 成本分析
---

# Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches

**会议**: ACL 2026  
**arXiv**: [2601.07711](https://arxiv.org/abs/2601.07711)  
**代码**: 无  
**领域**: 信息检索 / RAG  
**关键词**: 检索增强生成, Agentic RAG, Enhanced RAG, 查询重写, 成本分析

## 一句话总结

本文在四个数据集上从用户意图处理、查询重写、文档精炼和底层 LLM 选择四个维度系统对比了 Enhanced RAG 和 Agentic RAG，发现两者各有优势——Agentic RAG 在意图路由和查询重写上更灵活，Enhanced RAG 在文档重排上更有效，而 Agentic RAG 的成本高达 3.3 倍。

## 研究背景与动机

**领域现状**：RAG 已从研究概念发展为生产级语言系统的核心组件。基础 Naïve RAG（检索+生成）存在多个局限性，催生了两种改进范式：Enhanced RAG（在固定流水线中添加路由、重写、重排等模块）和 Agentic RAG（由 LLM 作为智能体自主编排整个流程）。

**现有痛点**：尽管这两种范式被快速采用，但缺乏系统的实证比较——在什么条件下应选择哪种方案？性能和成本的权衡如何？现有工作要么停留在理论定义层面，要么仅在单一设定下评测。

**核心矛盾**：Agentic RAG 提供了更大的灵活性（动态决策、迭代检索），但这种灵活性是否能转化为实际性能提升？额外的推理成本是否值得？

**本文目标**：在四个维度（用户意图处理、查询重写、文档列表精炼、底层 LLM 变更）上进行受控实验，量化两种范式的性能和成本差异。

**切入角度**：将 Naïve RAG 的四个已知缺陷映射为四个评估维度，分别设计 Enhanced 和 Agentic 的实现方案进行对比。

**核心 idea**：通过严格受控的实验设计，将"Enhanced vs Agentic RAG"这个模糊的架构选择问题转化为可量化的、维度化的决策指南。

## 方法详解

### 整体框架

针对 Naïve RAG 的四个缺陷，分别实现 Enhanced 和 Agentic 两种方案：(1) 用户意图处理——Enhanced 用语义路由器，Agentic 由智能体自主判断；(2) 查询重写——Enhanced 强制 HyDE 重写，Agentic 自主决定是否/如何重写；(3) 文档精炼——Enhanced 用 ELECTRA 重排器，Agentic 可迭代多次检索；(4) 底层 LLM——测试 Qwen3 的 0.6B/4B/8B/32B 四个规模。

### 关键设计

1. **用户意图处理 (User Intent Handling)**:

    - 功能：判断用户查询是否需要触发检索
    - 核心思路：Enhanced 方案使用 semantic-router 框架，基于预定义的有效/无效查询示例做语义相似度分类；Agentic 方案由智能体自主判断是否调用 RAG 工具。评估在 500 个有效 + 500 个无效查询上进行
    - 设计动机：避免对不需要检索的查询进行无意义的检索操作，这是生产环境中的重要需求

2. **查询重写 (Query Rewriting)**:

    - 功能：缩小用户查询和知识库文档之间的语义/格式差距
    - 核心思路：Enhanced 强制使用 HyDE（让 LLM 生成一个假设性回答段落作为查询），Agentic 可以自由选择是否重写以及如何重写。使用 NDCG@10 评估检索质量
    - 设计动机：用户查询通常是简短问题，而知识库文档是长篇文本，格式差异导致检索匹配不佳

3. **文档列表精炼 (Document List Refinement)**:

    - 功能：对检索到的文档列表进行优化，选出最相关的子集
    - 核心思路：Enhanced 使用 ELECTRA 交叉编码器对 top-20 文档重排；Agentic 允许智能体触发多轮检索来迭代改善上下文。实验发现 Agentic 仅在 10% 的情况下选择重新检索，且 53% 的文档与前次相同
    - 设计动机：初始检索结果可能包含噪声或次优文档，需要精炼策略

### 损失函数 / 训练策略

本文不涉及模型训练。Enhanced RAG 的各模块使用预训练模型（OpenAI embedding、ELECTRA reranker），Agentic RAG 使用 PocketFlow 框架实现。评估使用 LLM-as-a-Judge（Selene-70B）。

## 实验关键数据

### 主实验

**用户意图处理 (F1)**

| 设定 | FIQA | FEVER | CQA-EN |
|------|------|-------|--------|
| Naïve | 66.7 | 66.7 | 66.7 |
| Enhanced | 95.7 | **87.9** | 96.6 |
| Agentic | **98.8** | 64.6 | **99.8** |

**查询重写 (NDCG@10)**

| 设定 | FIQA | NQ | FEVER | CQAD | AVG |
|------|------|----|-------|------|-----|
| Naïve | 45.3 | 43.7 | 66.2 | 45.8 | 50.3 |
| Enhanced | 43.5 | 43.9 | 81.1 | 42.8 | 52.8 |
| Agentic | 43.2 | **51.7** | **83.1** | 44.3 | **55.6** |

**文档精炼 (NDCG@10)**

| 设定 | FIQA | CQA-EN | AVG |
|------|------|--------|-----|
| Naïve | 45.0 | 46.0 | 45.5 |
| Enhanced (w/ rewriting) | **51.0** | **48.0** | **49.5** |
| Agentic | 43.4 | 44.4 | 43.9 |

### 消融实验

**成本对比**

| 指标 | Agentic vs Enhanced |
|------|-------------------|
| 输入 token | 3.3× |
| 输出 token | 1.9× |
| 时间 | 1.5× |

### 关键发现

- Agentic RAG 在用户意图判断上表现更好（无需人工构造示例），但在宽泛领域（如 FEVER）会过度使用检索
- Agentic RAG 的自适应查询重写平均比 Enhanced 高 2.8 NDCG@10，在开放域查询（NQ）上优势达 +7.8
- Enhanced RAG 的显式重排远优于 Agentic 的迭代检索——智能体一旦做出决策就很少改变
- 更换底层 LLM 对两种范式的影响模式一致——性能随模型规模同比提升

## 亮点与洞察

- 将模糊的"Enhanced vs Agentic"选择问题拆解为四个独立维度，每个维度给出清晰的建议
- 最终建议是混合方案：用 Agentic 做意图路由和查询重写，用 Enhanced 的重排器做文档精炼
- 成本分析非常实用——Agentic RAG 高达 3.3 倍的 token 消耗提醒从业者需要认真考虑成本

## 局限与展望

- 仅评估单工具 Agentic RAG（只有检索工具），多工具智能体的表现可能不同
- 大规模数据集（NQ、FEVER）上的 Agentic 实验因耗时过长（>7天）而被排除
- 评估使用 LLM-as-a-Judge，存在评估偏差
- 未探索更复杂的 Agentic 策略（如反思、规划等）

## 相关工作与启发

- **vs CRAG/Self-RAG**: 属于 Enhanced RAG 的范畴，本文将其纳入统一框架对比
- **vs LLM Agent 框架**: 本文使用最简单的单工具 Agentic 设计以保持可比性
- **启发**：实际部署中应根据领域特性混合使用两种范式，而非全盘采用一种

## 评分

- 新颖性: ⭐⭐⭐ 问题重要但方法上无创新，主要是实验对比
- 实验充分度: ⭐⭐⭐⭐ 四个数据集×四个维度×成本分析，但部分实验因成本受限
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实用导向强，适合工程参考
- 价值: ⭐⭐⭐⭐ 为 RAG 架构选择提供了急需的实证指南

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] How Retrieved Context Shapes Internal Representations in RAG](how_retrieved_context_shapes_internal_representations_in_rag.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)

<!-- RELATED:END -->
