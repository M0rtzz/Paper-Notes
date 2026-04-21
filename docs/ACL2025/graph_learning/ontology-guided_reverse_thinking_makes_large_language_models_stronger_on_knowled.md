---
title: >-
  [论文解读] Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering
description: >-
  [ACL 2025][图学习][知识图谱问答] 提出 ORT（Ontology-Guided Reverse Thinking），利用知识图谱本体结构从目标逆向构建标签推理路径，引导知识检索，显著提升 LLM 的知识图谱问答能力。
tags:
  - ACL 2025
  - 图学习
  - 知识图谱问答
  - 本体引导
  - 逆向推理
  - LLM
  - 标签推理路径
---

# Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering

**会议**: ACL 2025  
**arXiv**: [2502.11491](https://arxiv.org/abs/2502.11491)  
**代码**: 无  
**领域**: 图学习  
**关键词**: 知识图谱问答, 本体引导, 逆向推理, LLM, 标签推理路径

## 一句话总结

提出 ORT（Ontology-Guided Reverse Thinking），利用知识图谱本体结构从目标逆向构建标签推理路径，引导知识检索，显著提升 LLM 的知识图谱问答能力。

## 研究背景与动机

知识图谱问答（KGQA）的核心挑战在于如何高效地建立从问题条件到答案目标的推理路径。现有方法主要分为两类：（1）微调方法（如 RoG、KD-CoT），需要大量高质量训练数据，资源消耗大；（2）嵌入+搜索方法（如 MindMap、Think-on-Graph），依赖实体向量匹配和图遍历，但无法处理概念性目标（如"stadium"是概念而非知识图谱中的具体实体）。

核心问题在于：传统的前向推理方法以实体为中心，通过向量匹配找到实体后扩展邻居，但问题的目标往往是抽象的，难以与具体实体匹配。例如问"1995 Rugby World Cup 的举办地点"，"stadium"只是一个概念标签，实体匹配无法直接定位到"Ellis Park Stadium"。

受人类逆向思维启发，作者提出从目标出发逆向构建推理路径到已知条件，利用知识图谱的本体结构（ontology）实现概念级别的推理导航。

## 方法详解

### 整体框架

ORT 分为三个阶段：（1）从问题中提取条件和目标及其标签；（2）基于知识图谱本体构建逆向标签推理路径；（3）利用标签推理路径引导知识检索并生成答案。

### 关键设计

1. **条件与目标识别（Aim and Condition Recognition）**：利用 LLM 从问题中提取条件实体 $\mathcal{C}_E$、条件标签 $\mathcal{C}_L$、目标实体 $\mathcal{A}_E$ 和目标标签 $\mathcal{A}_L$。通过提供知识图谱的标签列表（Label List），引导 LLM 将问题中的实体映射到知识图谱本体上的标签，解决了纯向量匹配的局限性。

2. **邻居标签字典构建**：遍历知识图谱本体中的关系三元组，为每个标签 $l_i$ 收集所有出现在同一三元组中的其他标签，构建邻居标签字典 $\mathcal{D} = \{l_i: \mathcal{N}(l_i)\}$。这是逆向推理树构建的基础数据结构。

3. **逆向推理树构建（Reverse Reasoning Tree）**：以目标标签 $\mathcal{A}_L$ 为根节点（通过虚拟根节点统一多个目标标签），递归查询邻居标签字典扩展子节点，直到达到最大递归深度（由问题的跳数决定）。这种逆向构建方式从目标出发向条件推进，天然过滤掉大量无关路径。

4. **三重剪枝策略**：

    - **条件剪枝（Prune by Conditions）**：DFS 遍历所有路径，移除不包含条件标签的路径；对包含条件标签的路径，仅保留最后一个条件标签及其之前的部分。
    - **环路剪枝（Prune Cycle Sub-paths）**：DFS 检测并移除路径中的环路，避免双向关系导致的无限循环。
    - **语义剪枝（Prune by Semantics）**：将剩余路径反转为正向路径后，连同问题输入 LLM，由 LLM 筛选出对回答问题有益的路径，消除语义不相关的干扰路径。

5. **引导答案挖掘（Guided Answer Mining）**：利用标签推理路径引导知识图谱的正向查询。从条件节点出发，沿标签路径逐步查询满足标签约束的邻居实体，构建实体推理路径树。最后通过 DFS 收集所有实体路径，输入 LLM 汇总生成最终答案。

### 训练策略

ORT 是即插即用的方法，不需要微调，直接利用现有 LLM 的能力。核心依赖 LLM 的几个 prompt：条件目标提取、语义剪枝、答案生成。

## 实验关键数据

### 主实验

| 数据集 | 指标 | ORT (DeepSeek-v3) | RoG (微调) | MindMap | 纯 LLM (DeepSeek-v3) |
|--------|------|------|----------|------|------|
| WebQSP | Hit@1 | **89.43** | 85.7 | 64.92 | 64.0 |
| WebQSP | F1 | **71.83** | 70.8 | 47.14 | 43.9 |
| CWQ | Hit@1 | **72.91** | 62.6 | 48.83 | 41.12 |
| CWQ | F1 | **62.63** | 56.2 | 43.30 | 33.80 |

### 消融实验

| 配置 | WebQSP Hit@1 | CWQ Hit@1 | 说明 |
|------|---------|------|------|
| ORT 完整 | 89.43 | 72.91 | 全部组件 |
| w/o LLM Filter | 86.58 | 62.58 | 移除语义剪枝，CWQ 下降 10+ |
| Trace Forward | 77.82 | 60.73 | 前向推理替代逆向推理 |
| w/o Rules | 64.00 | 41.12 | 不构建标签路径，退化为纯 LLM |

### 关键发现

- ORT 在不微调的情况下超越了微调方法 RoG（WebQSP: 89.43 vs 85.7）
- 对三种不同 LLM（GPT-4o、DeepSeek-v3、Qwen-max），ORT 均带来 25%+ 的 Hit@1 提升
- 逆向推理相比前向推理在 WebQSP 上 Hit@1 高出 11.61%，证明了逆向思维的有效性
- CWQ 数据集（多跳问题更多）上各方法表现普遍差于 WebQSP，说明多跳推理仍然具有挑战性

## 亮点与洞察

- **逆向思维的创新应用**：首次将人类逆向思维引入 KGQA，从目标出发逆向推理比从条件出发正向搜索更高效，能够天然过滤掉大量不相关的路径
- **本体结构的利用**：通过知识图谱本体在标签/概念层面进行推理，解决了传统方法在实体层面匹配的局限性
- **即插即用**：不需要微调，作为通用增强策略可直接提升各种 LLM 的 KGQA 能力
- 多层剪枝策略（条件剪枝 + 环路剪枝 + 语义剪枝）确保了推理路径的质量和效率

## 局限与展望

- 知识图谱查询时若满足标签约束的实体过多，可能引入大量无关结果
- 所有实体路径输入 LLM 可能引入干扰信息，降低答案准确率
- 依赖 LLM 的标签提取能力，对于复杂问题可能存在标签识别不准确的问题
- 仅在 Freebase 知识图谱上验证，其他知识图谱（如 Wikidata）的迁移性有待验证

## 相关工作与启发

- 与 MindMap 等前向搜索方法相比，逆向推理的路径搜索空间大幅缩小
- 本体结构的利用为 KGQA 提供了新的维度，连接了问题的抽象意图与知识图谱的结构化数据
- 与 RoG 等微调方法相比，ORT 在不微调的情况下达到了更优性能，说明知识图谱的结构信息利用比模型参数优化更关键
- 启发：逆向推理的思想可推广到其他结构化推理任务，如数据库查询优化、因果推理、规划问题等
- WebQSP 以单跳为主（65.49%），CWQ 包含更多多跳问题（20.75% ≥ 3跳），不同数据集的跳数分布影响方法性能

## 评分

- 新颖性: ⭐⭐⭐⭐ 逆向思维+知识图谱本体的结合是一个新颖且直觉上合理的想法
- 实验充分度: ⭐⭐⭐⭐ 两个标准数据集、多种 LLM、详细消融实验，但缺少更多知识图谱的验证
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观，算法伪代码完整
- 价值: ⭐⭐⭐⭐ 即插即用且效果显著，对 KGQA 领域有较大实际价值

<!-- RELATED:START -->

## 相关论文

- [FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)
- [The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering](the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que.md)
- [Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)
- [Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](../../ICML2025/graph_learning/graph-constrained_reasoning_faithful_reasoning_on_knowledge_graphs_with_large_la.md)
- [Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)

<!-- RELATED:END -->
