---
title: >-
  [论文解读] Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning
description: >-
  [ICLR 2026][深度搜索] 提出 HybridDeepSearcher，通过构建 HDS-QA 数据集训练大语言推理模型（LRM）区分可并行化和顺序依赖的搜索查询，在 FanOutQA 上 F1 提升 +15.9、BrowseComp 子集上提升 +11.5，同时显著降低推理延迟并展示出一致的测试时搜索扩展能力。
tags:
  - ICLR 2026
  - 深度搜索
  - 并行搜索
  - 检索增强生成
  - 大语言推理模型
  - 测试时搜索扩展
---

# Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning

**会议**: ICLR 2026  
**arXiv**: [2508.19113](https://arxiv.org/abs/2508.19113)  
**代码**: 无  
**领域**: LLM 推理  
**关键词**: 深度搜索, 并行搜索, 检索增强生成, 大语言推理模型, 测试时搜索扩展

## 一句话总结

提出 HybridDeepSearcher，通过构建 HDS-QA 数据集训练大语言推理模型（LRM）区分可并行化和顺序依赖的搜索查询，在 FanOutQA 上 F1 提升 +15.9、BrowseComp 子集上提升 +11.5，同时显著降低推理延迟并展示出一致的测试时搜索扩展能力。

## 研究背景与动机

大语言推理模型（LRM）如 OpenAI o3、DeepSeek-R1 结合检索增强生成（RAG）形成深度研究 agent，通过"推理—查询—检索"循环完成复杂多步任务。然而现有方法存在关键局限：

**延迟过高**：纯顺序查询逐一检索，每个查询都增加延迟

**工作流不连贯**：顺序搜索导致模型过早尝试回答或重复查询

**可扩展性差**：面对需要跨大量文档进行穷举搜索的问题，逐一查询难以覆盖所有证据

以 John Carpenter 电影问题为例：需要查询每部电影的时长。顺序方法逐一查询，不仅慢且容易遗漏；而**同时查询**所有电影时长则高效且准确得多。

核心问题：**如何让 LRM 在深度研究中同时利用并行和顺序搜索策略？**

## 方法详解

### 整体框架

方法包含两个核心部分：(1) HDS-QA 数据集构建，(2) HybridDeepSearcher 模型训练与推理。

### 关键设计

1. **HDS-QA 数据集构建**：自动构建包含混合跳（hybrid-hop）问题的数据集

   **问题生成流水线**（4步）：
   - **实体提取与相关問题收集**：从 Natural Questions 的单跳种子问题出发，提取中心实体，通过 Google "People Also Ask" 收集相关问题，并只保留检索到不同文档的查询以确保多样性
   - **实体特征摘要**：将检索文档摘要为该实体的关键特征
   - **并行跳问题构造**：利用特征组合成隐式引用实体（不直接提及）的并行跳问题
   - **混合跳问题整合**：将并行跳问题嵌入原始单跳问题，替换中心实体，引入额外的顺序跳。验证两个阶段都确实需要多步检索

   最终生成 1,987 个混合跳问题。

   **答案轨迹生成**：
   - 用 Qwen3-32B 迭代执行推理—查询—检索循环，允许每步同时发出多个并行查询
   - 对每个问题推理4次，保留所有正确轨迹以增加推理策略多样性
   - 773个问题得到正确答案，共 2,111 条成功轨迹（成功率约27%，说明任务确实困难）

2. **HybridDeepSearcher 推理流程**：

    - **推理**：模型在 `<think>` 和 `</think>` 标记内进行推理
    - **查询**：基于推理输出，在 `<|begin_search_queries|>` 和 `<|end_search_queries|>` 标记内生成顺序或并行查询，用 `;\n` 分隔多个并行查询
    - **检索**：每个查询通过 Web 搜索 API 执行，检索文档由外部模型（Qwen3-32B）摘要后返回
    - 模型可迭代多轮推理—查询—检索，直到收集到足够信息后产生最终答案

3. **自适应搜索策略**：模型学会动态判断何时使用并行查询（独立子问题）vs. 顺序查询（依赖前步结果的子问题），并在推理中明确展示当前步骤（蓝色）和后续计划（紫色）

### 损失函数 / 训练策略

- 基于 Qwen3-8B 进行全参数微调，使用 2,111 个问答轨迹对训练1个 epoch
- 学习率 3e-5，batch size 4，梯度累积 32 步
- **不对搜索结果部分应用梯度更新**，防止模型记忆检索内容
- 训练仅需 8 块 A100 40GB，约 30 分钟

## 实验关键数据

### 主实验

| 数据集 | 指标 | HybridDeepSearcher | RAG-R1 (SOTA) | 提升 |
|--------|------|-------------------|---------------|------|
| MuSiQue | F1 | 31.2 | 29.7 | +1.5 |
| FanOutQA | F1 | 44.1 | 28.2 | **+15.9** |
| FRAMES | F1 | 39.1 | 35.8 | +3.3 |
| MedBrowseComp | MBE | 30.4 | 28.2 | +2.2 |
| BrowseComp-50 | F1 | 17.2 | 5.7 | **+11.5** |

AUC（效率-效果权衡）：在所有基准上达到最高值，说明模型在更少搜索轮次内达到更高精度。

### 消融实验 / 搜索能力分析

| 方法 | MuSiQue 覆盖率 | FanOutQA 覆盖率 | FRAMES 覆盖率 |
|------|---------------|----------------|--------------|
| Search-o1 | 33.4% | 38.3% | 44.8% |
| DeepResearcher | 38.8% | 49.9% | 49.0% |
| RAG-R1 | 35.9% | 53.2% | 48.0% |
| **HybridDeepSearcher** | **40.7%** | **61.0%** | **55.8%** |

在 FanOutQA 上证据覆盖率提升最大（+7.8pp），该数据集标注证据链接最多，最需要广泛并行检索。

### 关键发现

1. **测试时搜索扩展**（核心优势）：
    - HybridDeepSearcher 的性能随搜索轮次和API调用增加而**持续提升**
    - RAG-R1 等基线在 2-3 轮后**性能停滞**
    - 在 BrowseComp-50 上尤为明显：其他方法几乎无法受益于更多搜索预算

2. **效率优势**：用更少的搜索轮次达到更高精度
    - 在 FanOutQA 上用约3轮搜索就超越其他方法用5轮以上的结果

3. **非迭代方法的失败**：直接生成和标准 RAG 效果极差（BrowseComp-50 上 F1 为 0.0/1.8），证明这些基准确实需要外部知识和多步推理

4. **Case Study 洞察**：
    - 在 FRAMES 的 John Carpenter 问题上，HybridDeepSearcher 并行查询12部电影的时长并找到正确答案（Starman, 115分钟），而 DeepResearcher 先入为主猜测 The Thing、Search-o1 陷入循环查询

## 亮点与洞察

1. **并行+顺序搜索的统一**：首次系统性地训练 LRM 区分可并行化和顺序依赖的查询，填补了现有工作的空白
2. **数据集构建巧妙**：HDS-QA 的自动构建流水线从 NQ 出发，通过"People Also Ask"引入并行性，设计精巧且可扩展
3. **SFT 优于 RL**：仅用 2,111 条轨迹的监督微调就超越了使用 GRPO 的 RL 方法（如 Search-R1、DeepResearcher），说明高质量的混合搜索示范数据极其重要
4. **搜索扩展性**：该方法是少数展示出一致测试时搜索扩展能力的工作，性能随计算预算增长而不饱和
5. **训练成本极低**：仅需 30 分钟在 8 块 A100 上微调，开销远低于 RL 训练方法

## 局限与展望

1. 仅使用 SFT 训练，未结合偏好优化（DPO/RLHF），可利用 HDS-QA 中的成功和失败轨迹进一步提升
2. 搜索查询摘要依赖外部大模型（Qwen3-32B），增加了系统复杂度和API调用成本
3. HDS-QA 仅基于 Natural Questions 构建，领域覆盖可能有限
4. 未探索多 Agent 协作搜索的可能性
5. BrowseComp-50 仅选取了 o3 能解决的50道题，选择偏差可能影响评估公平性

## 相关工作与启发

- **Search-o1**：基于 prompt 的迭代推理-查询-检索框架，单查询顺序搜索
- **Search-R1 / DeepResearcher**：使用 GRPO 训练增强搜索推理能力，但训练数据缺少并行搜索示范
- **RAG-R1**：多查询基线，性能不错但缺乏搜索扩展性
- **APR**：自适应并行推理，但仅在 Countdown 等玩具任务上验证

本文对 RAG 系统设计的启发：将"何时并行、何时顺序"作为显式训练信号，比单纯增加推理能力更有效。混合搜索策略可能是大规模深度研究 Agent 的关键能力。

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Summaries as Centroids for Interpretable and Scalable Text Clustering](summaries_as_centroids_for_interpretable_and_scalable_text_clustering.md)
- [\[NeurIPS 2025\] Deep Research Brings Deeper Harm](../../NeurIPS2025/information_retrieval/deep_research_brings_deeper_harm.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)
- [\[AAAI 2026\] SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention](../../AAAI2026/information_retrieval/sr-ki_scalable_and_real-time_knowledge_integration_into_llms_via_supervised_atte.md)

<!-- RELATED:END -->
