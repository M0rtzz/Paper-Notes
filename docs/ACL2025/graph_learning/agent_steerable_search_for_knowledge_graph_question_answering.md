---
title: >-
  [论文解读] Agent Steerable Search for Knowledge Graph Question Answering
description: >-
  [ACL 2025][图学习][知识图谱问答] 本文提出一种基于智能体的可控知识图谱搜索框架，让LLM Agent能够根据问题类型和推理需求动态调整图搜索策略（如搜索深度、方向、剪枝规则），实现对知识图谱问答过程的精细控制。 领域现状：知识图谱问答（KGQA）要求系统在知识图谱上检索相关三元组来回答自然语言问题…
tags:
  - "ACL 2025"
  - "图学习"
  - "知识图谱问答"
  - "可控搜索"
  - "智能体"
  - "图搜索策略"
  - "推理路径"
---

# Agent Steerable Search for Knowledge Graph Question Answering

**会议**: ACL 2025  
**领域**: 图学习  
**关键词**: 知识图谱问答、可控搜索、智能体、图搜索策略、推理路径

## 一句话总结
本文提出一种基于智能体的可控知识图谱搜索框架，让LLM Agent能够根据问题类型和推理需求动态调整图搜索策略（如搜索深度、方向、剪枝规则），实现对知识图谱问答过程的精细控制。

## 研究背景与动机

**领域现状**：知识图谱问答（KGQA）要求系统在知识图谱上检索相关三元组来回答自然语言问题。当前方法主要分为两类：基于检索的方法（先检索子图再推理）和基于推理的方法（在图上多跳遍历后回答）。近期LLM-based Agent方法让智能体与知识图谱交互，通过多步搜索和推理完成问答。

**现有痛点**：现有基于Agent的KGQA方法通常采用固定的搜索策略——要么广度优先要么深度优先，搜索深度和剪枝阈值也是预设的全局参数。这导致对简单的单跳问题搜索过多造成浪费和噪声引入，对需要深层推理的复杂问题又搜索不足而遗漏关键信息。

**核心矛盾**：不同类型的问题需要截然不同的搜索策略——事实查询需要精确的短路径搜索，比较类问题需要宽广的多路径搜索，而推理链问题需要深层的定向搜索。单一策略无法同时优化所有问题类型。

**本文目标**：赋予LLM Agent动态调整知识图谱搜索策略的能力，包括搜索方向、深度、宽度和剪枝规则，使得搜索过程可以根据具体问题自适应调整。

**切入角度**：将知识图谱搜索视为一个可参数化的过程，搜索参数（深度、宽度、方向等）作为Agent可以调控的"方向盘"，Agent基于对问题的理解来设定这些参数。

**核心 idea**：用Agent可控的参数化搜索策略替代固定搜索策略，让Agent根据问题语义动态生成搜索计划。

## 方法详解

### 整体框架
输入自然语言问题，首先由问题分析模块提取实体并判断问题类型和复杂度。然后Agent根据分析结果生成搜索计划（包含搜索策略的参数配置），搜索执行器按照计划在知识图谱上检索相关子图。检索到的子图经过整合后送入推理模块生成最终答案。整个过程可以迭代——如果初始搜索结果不足，Agent可以调整策略重新搜索。

### 关键设计

1. **搜索计划生成器（Search Plan Generator）**:

    - 功能：根据问题特征生成参数化的搜索计划
    - 核心思路：将搜索计划定义为一个结构化的参数集合，包括：搜索模式（BFS/DFS/混合）、最大深度 $d$、每层最大扩展宽度 $w$、关系过滤器（指定优先搜索的关系类型）、以及相关性剪枝阈值 $\theta$。Agent通过分析问题的语义结构（如涉及几个实体、需要几跳推理、是否包含约束条件）来设定这些参数
    - 设计动机：避免"一刀切"的搜索策略，让不同问题获得最合适的搜索配置

2. **自适应搜索执行器（Adaptive Search Executor）**:

    - 功能：在知识图谱上按计划执行搜索并收集相关三元组
    - 核心思路：执行器按照Agent指定的搜索模式遍历图，在每个节点处根据关系过滤器和语义相关性阈值决定是否扩展该方向。语义相关性使用LLM对"该关系是否与原始问题相关"做快速判断。搜索过程中维护一个优先队列，优先探索与问题最相关的路径
    - 设计动机：结合结构化搜索和语义引导，在效率和召回率之间取得平衡

3. **迭代修正机制（Iterative Refinement）**:

    - 功能：允许Agent在初始搜索不满意时调整策略重新搜索
    - 核心思路：Agent评估搜索结果的充分性——如果发现缺少关键信息（如某个实体的属性未覆盖），可以生成补充搜索计划，增大相关方向的搜索深度或宽度。最大迭代次数通过超参控制以避免无限循环
    - 设计动机：单次搜索很难完美，迭代修正让系统有机会补救搜索遗漏

### 损失函数 / 训练策略
采用少样本学习方式，在少量标注的问题-搜索计划对上微调Agent的计划生成能力。搜索执行器不需要训练，是基于规则的图遍历引擎。整体系统通过最终答案的准确性作为反馈信号，使用DPO微调Agent的策略选择偏好。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Agent Steerable | ToG | KG-Agent | StructGPT | 提升 |
|--------|------|-----------------|-----|----------|-----------|------|
| WebQSP | Hits@1 | 78.6 | 74.2 | 72.8 | 70.3 | +4.4 |
| CWQ | Hits@1 | 65.3 | 60.1 | 58.7 | 55.9 | +5.2 |
| GrailQA | Hits@1 | 71.8 | 67.5 | 65.2 | 63.6 | +4.3 |
| SimpleQA | Hits@1 | 89.2 | 88.1 | 87.5 | 86.3 | +1.1 |

### 消融实验

| 配置 | CWQ(Hits@1) | 平均搜索步数 | 说明 |
|------|------------|-------------|------|
| Full model | 65.3 | 4.2 | 完整可控搜索 |
| 固定BFS | 60.8 | 6.7 | 所有问题用BFS |
| 固定DFS | 58.3 | 5.1 | 所有问题用DFS |
| 无关系过滤 | 62.1 | 7.3 | 不过滤关系类型 |
| 无迭代修正 | 63.0 | 3.8 | 只搜索一次 |

### 关键发现
- 搜索计划生成器贡献最大，动态切换BFS/DFS的策略比任何固定策略都好5+个点
- 关系过滤器在减少搜索步数方面效果显著（4.2 vs 7.3），同时还提升了准确率3.2个点，说明噪声关系的去除非常重要
- 在简单问题上优势较小（+1.1），在复杂多跳问题上优势显著（+5.2），验证了可控搜索对复杂推理更有价值
- 迭代修正机制带来约2.3个点的提升，72%的修正发生在初始搜索深度不足的情况

## 亮点与洞察
- 将搜索策略参数化并交给Agent控制的设计非常优雅，把"搜索超参数调优"自动化为Agent的决策过程。这种思路可以迁移到任何需要在结构化数据上搜索的场景
- 语义引导的剪枝与结构化搜索的结合，实现了准确率和效率的双赢

## 局限与展望
- 搜索计划生成依赖LLM的问题理解能力，对于语义模糊的问题可能生成不合适的计划
- 搜索策略的参数空间目前是离散的，未来可以探索连续参数化以实现更精细的控制
- 当前仅支持Freebase/Wikidata等典型知识图谱，对领域特定KG（如生物医学KG）的泛化性未验证
- 迭代搜索的最大轮数是超参数，自动判断何时停止搜索仍是开放问题
- 可以将搜索策略学习与更强的图神经网络结合，利用图结构信息辅助搜索决策
- 当前仅支持Freebase/Wikidata，对特定领域的KG（如生物医学知识图谱）的泛化性未验证，各领域的关系类型和图结构差异可能需要不同的搜索策略

## 相关工作与启发
- **vs ToG (Think-on-Graph)**: ToG使用固定的图推理流程，本文让Agent动态控制搜索参数，灵活性更高
- **vs KG-Agent**: KG-Agent提供了图搜索工具但搜索策略是预设的，本文将策略本身也交给Agent决定
- **vs StructGPT**: StructGPT通过接口调用访问KG但不控制搜索过程，本文实现了搜索层面的可控性

## 评分
- 新颖性: ⭐⭐⭐⭐ 参数化搜索策略的想法新颖
- 实验充分度: ⭐⭐⭐⭐ 多个KGQA基准测试
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰
- 价值: ⭐⭐⭐⭐ 对KGQA系统设计有直接启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering](the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que.md)
- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)
- [\[ACL 2025\] Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering](ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled.md)
- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](../../ICML2026/graph_learning/kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)

</div>

<!-- RELATED:END -->
