---
title: >-
  [论文解读] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models
description: >-
  [ACL 2025][代码检索] 提出 CoIR，首个全面的代码信息检索基准，包含 10 个数据集、覆盖 4 大类 8 个子任务和 14 种编程语言，揭示了即使是 SOTA 检索模型在代码检索中也表现不佳，并指出许多模型已在现有排行榜上过拟合。
tags:
  - ACL 2025
  - 代码检索
  - 信息检索基准
  - 嵌入模型
  - 多编程语言
  - 代码理解
---

# CoIR: A Comprehensive Benchmark for Code Information Retrieval Models

**会议**: ACL 2025  
**arXiv**: [2407.02883](https://arxiv.org/abs/2407.02883)  
**代码**: https://github.com/CoIR-team/coir (有)  
**领域**: Others / 代码检索与信息检索  
**关键词**: 代码检索, 信息检索基准, 嵌入模型, 多编程语言, 代码理解

## 一句话总结

提出 CoIR，首个全面的代码信息检索基准，包含 10 个数据集、覆盖 4 大类 8 个子任务和 14 种编程语言，揭示了即使是 SOTA 检索模型在代码检索中也表现不佳，并指出许多模型已在现有排行榜上过拟合。

## 研究背景与动机

信息检索（IR）在文本领域已取得巨大成功，但代码检索（Code Retrieval）作为开发者日常工作的关键功能仍被严重低估和不充分评估。现有代码检索基准存在三大问题：

**任务单一**：CodeSearchNet、CosQA 主要聚焦"文本到代码"单一任务，忽视了实际需求中"代码到代码"、"代码到文本"等多样化场景。开发者实际可能需要输入带 bug 信息的代码片段，检索解释、摘要或修复方案。

**领域缺乏多样性**：CodeSearchNet 仅从 GitHub 提取代码和注释对；XCodeEval 仅关注竞赛编程。这种狭窄的领域覆盖无法全面评估模型在更广泛编码场景中的表现。

**无标准化评估框架**：各基准使用不同的评估指标和格式，导致跨基准比较模型性能困难重重。

更关键的是，许多模型已经在 CodeSearchNet 等常用基准上过拟合——这意味着排行榜上的高分可能并不反映真实的代码检索能力。

## 方法详解

### 整体框架

CoIR 的设计遵循三个原则：多样性（4大任务 × 8子任务 × 10数据集 × 14编程语言）、易用性（pip install 一键评估）、缓解过拟合（多样化任务和领域组合）。

### 关键设计

1. **四大主任务体系**：

    - **Text-to-Code（文本到代码）**：竞赛题代码检索（APPS）、Web查询代码检索（CosQA）、Text-to-SQL检索（Synthetic Text2SQL）
    - **Code-to-Text（代码到文本）**：代码摘要检索（CodeSearchNet）——用代码检索对应注释/摘要
    - **Code-to-Code（代码到代码）**：代码上下文检索（CodeSearchNet-CCR，自建）——给定代码前半段检索后半段；相似代码检索（CodeTransOcean）——跨编程语言/框架的语义等价代码
    - **Hybrid Code（混合代码）**：单轮代码 QA（StackOverflow QA自建 + CodeFeedback-ST）和多轮代码 QA（CodeFeedback-MT）——查询和答案都包含文本和代码混合

2. **CodeSearchNet-CCR（自建数据集）**：将 CodeSearchNet 中每个代码片段随机分为两段（40%~70% 为查询，其余为检索目标），模拟代码补全的检索需求。这是代码上下文检索的首个大规模数据集。

3. **StackOverflow QA（自建数据集）**：从 StackOverflow 原始数据中将问题与最高赞答案配对，得到 19,931 对，并抽样 1,202 条用于测试。

4. **数据质量保障**：所有数据集都经过人工检查和过滤，去除无效答案、歧义实例和无关信息。

5. **标准化评估**：与 BEIR 和 MTEB 数据格式对齐，统一使用 NDCG@10 作为主指标，支持 MAP、Recall、Precision 等。提供 Python 框架 `pip install coir` 一键评估。

### 多样性分析

通过加权 Jaccard 相似度分析数据集间的词汇重叠，发现大部分数据集对间相似度很低（除了同源的 CodeFeedback-ST 和 CodeFeedback-MT），证实了基准的挑战性和多样性。

## 实验关键数据

### 主实验（NDCG@10）

| 模型 (参数量) | APPS | CosQA | Text2SQL | CodeSN | CSN-CCR | CodeTrans-C | CodeTrans-DL | SOQA | CF-ST | CF-MT | 平均 |
|------|------|-------|---------|--------|---------|-------------|-------------|------|-------|-------|------|
| BM25 | 0.95 | 13.96 | 16.92 | 26.75 | 34.69 | 50.13 | 8.69 | 56.80 | 54.32 | 34.73 | 29.79 |
| E5-Base (110M) | 11.52 | 32.59 | 52.31 | 67.99 | 56.87 | 62.50 | 21.87 | 86.86 | 74.52 | 41.99 | 50.90 |
| E5-Mistral (7B) | 21.33 | 31.27 | 65.98 | 54.25 | 65.27 | 82.55 | 33.24 | 91.54 | 72.71 | 33.65 | 55.18 |
| Voyage-Code-002 | 26.52 | 29.79 | 69.26 | 81.79 | 73.45 | 72.77 | 27.28 | 87.68 | 65.35 | 28.74 | **56.26** |
| OpenAI-Ada-002 | 8.70 | 28.88 | 58.32 | 74.21 | 69.13 | 53.34 | 26.04 | 72.40 | 47.12 | 17.74 | 45.59 |

### 效率分析（CodeFeedBack-ST, 156K语料 + 31K查询）

| 模型 | 嵌入维度 | 嵌入延迟/样本 | 检索延迟/查询 | 索引大小 |
|------|---------|-------------|-------------|---------|
| E5-Base | 768 | 7.4ms | 38.1µs | 0.3G |
| BGE-M3 | 1024 | 31.4ms | 42.9µs | 0.6G |
| E5-Mistral | 4096 | 1840ms | 115.5µs | 2.3G |

### 输入长度影响

| 模型 (输入长度) | CodeFB-MT | CodeTO-DL | APPS | SOQA |
|------|-----------|-----------|------|------|
| GTE (512) | 28.48 | 28.80 | 3.24 | 62.71 |
| GTE (4k) | 51.32 | 27.33 | 5.08 | 78.63 |
| BGE-M3 (512) | 33.46 | 31.16 | 7.37 | 61.04 |
| BGE-M3 (4k) | 27.49 | 32.75 | 6.80 | 56.53 |

### 关键发现

- **没有单一模型在所有任务上都占优**：Voyage-Code-002 平均最高（56.26）但方差大；E5-Mistral 在竞赛代码和 SOQA 上最强但在其他任务上表现平庸
- **代码特化训练有效但非万能**：Voyage-Code-002 在 Text-to-Code 和 Code-to-Text 上表现突出，但多轮 QA（CodeFB-MT: 28.74）表现差
- **BM25 在 APPS 上几乎失效**（0.95），说明代码竞赛题的语义理解要求远超词汇匹配
- **过拟合问题严重**：OpenAI-Ada-002 在 CodeSearchNet 上得分 74.21，但 CoIR 平均仅 45.59，差距高达 28.6 分
- **长输入并非总有效**：GTE 受益于长输入（512→4k 在 SOQA 上提升 16分），但 BGE-M3 反而下降，说明代码数据与文本数据特性不同
- **效率-性能权衡**：E5-Mistral 性能强但嵌入延迟是 E5-Base 的 250 倍

## 亮点与洞察

- CoIR 是首个真正全面的代码检索基准，覆盖了从代码补全到跨语言代码匹配的广谱场景
- 揭示了现有排行榜过拟合的关键问题：模型在 CodeSearchNet 上表现优异不等于具备真正的代码检索能力
- LLM-based 检索模型（E5-Mistral）展现了最小的 CodeSearchNet 与 CoIR 分数差距，暗示 LLM 在缓解过拟合方面有潜力
- 多轮代码 QA（查询长达 4K+ token）是极具挑战的新方向，当前所有模型表现都不理想
- 与 BEIR/MTEB 的无缝集成降低了使用门槛

## 局限与展望

- 所有数据集仅覆盖英语，缺少多语言代码检索评测
- 每个查询仅对应一个标准答案（n=1），不反映一个查询可能匹配多个相关代码的现实情况
- 未涉及基于代码元数据（如版本号、库依赖）的多维度检索
- CodeTransOcean 规模较小（数百至千条），统计置信度有限
- 未评估近期的代码专用嵌入模型（如 CodeSage）

## 相关工作与启发

- BEIR 和 MTEB 为文本检索建立了标准化基准，CoIR 将同样的理念推广到代码领域
- CodeSearchNet 的过拟合问题提醒我们：评估基准需要持续更新和多样化
- 结合代码结构信息（AST、control flow）的检索方法可能是突破方向
- 代码-RAG 系统（用代码检索增强 LLM 代码生成）是重要的应用场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个全面的代码检索基准，任务设计考虑周全
- **实验充分度**: ⭐⭐⭐⭐ — 10个模型 × 10个数据集的完整评测，附带效率和过拟合分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，任务分类和数据集描述详尽
- **价值**: ⭐⭐⭐⭐⭐ — 填补了代码检索评估的重大空白，对社区有长期推动作用

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](air-bench_automated_heterogeneous_information_retrieval_benchmark.md)
- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)
- [\[ACL 2025\] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation](flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)
- [\[ACL 2025\] Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models](pandora_box_rag_noise.md)

<!-- RELATED:END -->
