---
title: >-
  [论文解读] FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems
description: >-
  [ICML 2025][人体理解] FedRAG 提出了一个同时支持集中式和联邦式架构的 RAG 系统微调框架，填补了 RAG 生态系统中缺乏统一微调工具的空白，并通过轻量级抽象实现了从集中式到联邦式训练的无缝转换。
tags:
  - ICML 2025
  - 人体理解
---

# FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems

**会议**: ICML 2025  
**arXiv**: [2506.09200](https://arxiv.org/abs/2506.09200)  
**领域**: 人体理解  

## 一句话总结

FedRAG 提出了一个同时支持集中式和联邦式架构的 RAG 系统微调框架，填补了 RAG 生态系统中缺乏统一微调工具的空白，并通过轻量级抽象实现了从集中式到联邦式训练的无缝转换。

## 研究背景与动机

检索增强生成（RAG）系统通过从外部知识库检索相关信息来补充大语言模型（LLM）的参数记忆，有效缓解了幻觉问题。近年来的研究表明，对 RAG 系统的检索器和生成器进行微调可以进一步提升性能。然而，当前 RAG 生态系统虽然工具丰富（如 LlamaIndex、LangChain 等），但缺乏一个简化 RAG 微调流程且与生态系统深度集成的框架。更重要的是，在数据隐私约束下，联邦学习（FL）成为改进 RAG 系统的不可或缺的工具，但目前几乎没有工具能够将集中式 RAG 微调简单地转换为联邦式任务。

## 方法详解

### 核心设计哲学

FedRAG 遵循三大设计原则：

1. **高级 RAG 微调（Advanced RAG Fine-Tuning）**：全面支持前沿的 RAG 微调方法，并可轻松联邦化
2. **与现有工具协同（Work With Your Tools）**：与 HuggingFace、Unsloth、LlamaIndex 等流行框架深度集成
3. **轻量级抽象（Lightweight Abstractions）**：提供清晰直观的接口，降低学习成本

### 框架架构

FedRAG 采用模块化设计，核心模块包括：

- **`core`**：核心类型定义，包括 `RAGSystem` 类
- **`generators`**：生成器类型（支持 HuggingFace、Unsloth 等）
- **`retrievers`**：检索器类型（支持 HF SentenceTransformer 等）
- **`knowledge_stores`**：知识存储（支持 Qdrant 等）
- **`trainers`**：训练器类型
- **`fl_tasks`**：联邦学习任务定义
- **`evals`**：评估指标和基准

### RAG 系统构建

一个 `RAGSystem` 由三部分组成：`KnowledgeStore`、`Retriever` 和 `Generator`。用户可以通过简洁的 API 快速组装 RAG 系统并执行查询。

### 微调方法

FedRAG 支持两类主要微调方法：

**生成器微调：**
- **RALT（Retrieval-Augmented Language Model Training）**：通过包含检索上下文的指令样本进行微调
- **RAFT（Retrieval-Augmented Fine-Tuning）**：在指令样本中加入 LLM 生成的 CoT 推理链
- **ReSearch**：基于强化学习让 LLM 学习生成包含搜索和检索操作的长 CoT

**检索器微调：**
- **LSR（Language Model Supervised Retriever Training）**：最小化检索分数分布和生成器条件概率分布之间的 KL 散度

$$\mathcal{L}_{\text{LSR}} = D_{\text{KL}}(P_{\text{retrieval}} \| P_{\text{generator}})$$

### 联邦化转换

FedRAG 的核心创新在于从集中式到联邦式的无缝转换。用户只需从训练管理器中提取 `FL_task` 对象，即可获取联邦服务器和客户端，利用联邦平均（FedAvg）进行去中心化训练：

$$\theta_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \theta_{t+1}^k$$

其中 $\theta_{t+1}^k$ 是第 $k$ 个客户端的本地更新参数。

### 评估与基准测试

FedRAG 提供直观的基准测试接口，支持使用 `Benchmarker` 运行指定的 `Benchmark`（如 HuggingFace MMLU），并应用选定的评估指标。

## 实验

### 主实验

论文在附录中展示了轻量级实验，验证了 FedRAG 可以成功且灵活地执行 RAG 微调任务。实验代码和知识库的容器化镜像已随论文发布，以促进可复现性。

### 集成支持

| 库 | 集成内容 |
|---|---|
| HuggingFace | 生成器、检索器、数据集 |
| Unsloth | 快速生成器微调 |
| Qdrant | 知识库存储方案 |
| LlamaIndex | 推理对象桥接 |

### 未来规划

| 优先级 | 开发项目 |
|---|---|
| 高 | MCP RAG 系统与 MCP 知识库集成 |
| 高 | 研究第三方 MCP 提供商适配效果 |
| 中 | 更多微调方法支持 |

## 亮点

- **首个统一的 RAG 微调框架**：同时支持集中式和联邦式架构，填补了生态系统空白
- **极简的联邦化转换**：只需几行代码即可将集中式训练转换为联邦式任务
- **深度生态系统集成**：与 HuggingFace、Unsloth、Qdrant、LlamaIndex 等主流工具无缝对接
- **前瞻性设计**：计划集成 MCP 协议，与去中心化 AI 趋势保持一致

## 局限性

- 论文以系统框架为主，实验验证较为轻量，缺乏大规模基准测试对比
- 联邦学习场景下的通信效率和隐私保护细节未深入讨论
- 目前支持的微调方法有限，尚未覆盖所有前沿 RAG 微调技术
- 缺乏与其他潜在竞争框架的系统性对比

## 评分

⭐⭐⭐ (3/5)

论文作为系统工具论文，设计理念清晰、接口优雅，填补了 RAG 生态系统中微调工具的空白。但实验验证不够充分，作为 ICML 论文略显单薄。

<!-- RELATED:START -->

## 相关论文

- [Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)
- [Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [LLaVA-ReID: Selective Multi-Image Questioner for Interactive Person Re-Identification](llava-reid_selective_multi-image_questioner_for_interactive_person_re-identifica.md)
- [TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration](toping_topologically_interpretable_graph_learning_via_persistent_rationale_filtr.md)
- [Scaling Large Motion Models with Million-Level Human Motions](scaling_large_motion_models_with_million-level_human_motions.md)

<!-- RELATED:END -->
