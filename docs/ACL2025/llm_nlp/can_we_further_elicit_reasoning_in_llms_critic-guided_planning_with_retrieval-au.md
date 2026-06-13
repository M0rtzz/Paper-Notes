---
title: >-
  [论文解读] Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks
description: >-
  [ACL 2025][LLM 其他][批判引导规划] 本文提出CR-Planner框架，通过微调的批判模型（critic model）引导推理和检索过程的规划，结合蒙特卡洛树搜索（MCTS）来训练critic，在竞赛编程、定理驱动的数学推理和复杂领域检索问题上显著优于基线方法。 领域现状：大语言模型在一般推理任务上表现优秀…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "批判引导规划"
  - "检索增强"
  - "蒙特卡洛树搜索"
  - "子目标分解"
  - "竞赛编程"
---

# Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks

**会议**: ACL 2025  
**arXiv**: N/A  
**链接**: [ACL Anthology](https://aclanthology.org/2025.acl-long.1244/)
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 批判引导规划, 检索增强, 蒙特卡洛树搜索, 子目标分解, 竞赛编程

## 一句话总结

本文提出CR-Planner框架，通过微调的批判模型（critic model）引导推理和检索过程的规划，结合蒙特卡洛树搜索（MCTS）来训练critic，在竞赛编程、定理驱动的数学推理和复杂领域检索问题上显著优于基线方法。

## 研究背景与动机

**领域现状**：大语言模型在一般推理任务上表现优秀，但在需要复杂推理和精确事实知识的挑战性任务（如竞赛编程、定理证明）上仍然力不从心。链式思维（CoT）和检索增强生成（RAG）分别从推理深度和知识广度两个角度提升LLM能力，但两者的结合效果不尽如人意。

**现有痛点**：(1) CoT在多步推理中容易出现推理错误的累积——一个早期的错误推理步骤会导致整条链偏离正确方向；(2) RAG的检索结果经常包含不相关的信息，反而干扰推理过程；(3) 现有方法缺乏有效的"反馈机制"来在推理过程中及时发现和纠正错误、筛选检索结果。

**核心矛盾**：复杂任务需要"推理+检索"的深度协同，但LLM在推理过程中既可能走错推理方向，又可能被检索到的噪声信息误导。缺乏一个"裁判"来在每一步评估推理质量和检索相关性。

**本文目标**：设计一个统一框架，使用专门训练的critic模型来同时引导推理过程和检索过程，提升LLM在挑战性任务上的表现。

**切入角度**：将推理过程建模为规划问题，用critic模型评估每一步推理/检索动作的质量，通过MCTS收集过程监督数据来训练critic。

**核心 idea**：用两个专门的critic（子目标critic和执行critic）来引导规划过程中的推理和检索决策，critic的训练数据通过MCTS自动收集。

## 方法详解

### 整体框架

CR-Planner的输入是一个需要推理和知识的复杂问题（如编程题、数学定理证明题），输出是最终答案。框架由三个组件构成：(1) 规划器（Planner）——分解任务为子目标序列；(2) 子目标critic——评估候选子目标的质量；(3) 执行critic——评估子目标执行结果（推理步骤或检索结果）的质量。整个过程迭代进行直到到达最终答案。

### 关键设计

1. **子目标Critic（Sub-goal Critic）**:

    - 功能：在每一步规划中评估候选子目标的质量和前景
    - 核心思路：给定当前问题状态和已完成的子目标，Planner生成多个候选下一步子目标（如"先理解输入格式"、"查找相关算法"、"设计数据结构"等）。子目标critic对每个候选打分，评估其"到达最终答案的前景"。critic是一个微调的语言模型，输入是(问题+已完成步骤+候选子目标)，输出是一个质量分数。选择得分最高的子目标继续执行
    - 设计动机：在复杂任务中，一步走错可能导致后面全部偏离方向。子目标critic提供了"前瞻性评估"，避免进入低效甚至错误的推理路径

2. **执行Critic（Execution Critic）**:

    - 功能：评估子目标的执行结果质量
    - 核心思路：子目标被选定后，Planner执行之——有三种执行动作：(a) 推理（直接用LLM生成推理步骤）；(b) 生成查询（为检索系统生成查询）；(c) 检索（执行检索获取相关知识）。执行critic评估每种执行结果的质量——推理步骤是否逻辑正确？检索结果是否相关且有用？如果执行结果质量低于阈值，则重新生成或选择其他执行方式
    - 设计动机：仅有子目标层面的引导不够——目标正确但执行出错同样会失败。执行critic提供了过程中的质量控制

3. **基于MCTS的Critic训练数据收集**:

    - 功能：自动收集高质量的(状态, 动作, 奖励)数据来训练两个critic
    - 核心思路：以蒙特卡洛树搜索（MCTS）为核心工具——在训练阶段，对每个训练问题，从根节点（初始问题）出发，通过MCTS展开搜索树。树的每个节点是一个中间状态（已完成的子目标列表），边是一个子目标或执行动作。MCTS通过反复模拟（rollout）来估计每个节点/边的值。搜索完成后，每个节点和边都有了经过大量模拟验证的质量估计，这些(状态, 子目标/执行, 质量分数)三元组就是critic的训练数据
    - 设计动机：过程监督信号非常稀缺——不是每个推理步骤都有人工标注的"对/错"标签。MCTS通过自动搜索和回溯来自动生成这些信号，大幅降低了数据标注成本

### 训练策略

两个critic使用MCTS收集的数据分别微调。在推理时，采用beam search风格的搜索——每一步保留top-k个候选路径，由critic引导探索。

## 实验关键数据

### 主实验

| 任务 | 指标 | CR-Planner | CoT | RAG | Tree-of-Thought | 提升 |
|------|------|-----------|-----|-----|-----------------|------|
| 竞赛编程(APPS) | Pass@1 | 显著最优 | 基线 | 无效 | 次优 | ~10-15% |
| 定理数学推理 | 准确率 | 显著最优 | 基线 | 有帮助 | 次优 | ~8-12% |
| 复杂领域检索(Bamboogle) | F1 | 显著最优 | 弱 | 基线 | 次优 | ~5-10% |

### 消融实验

| 配置 | 竞赛编程 Pass@1 | 数学推理 Acc | 说明 |
|------|----------------|-------------|------|
| CR-Planner（完整） | 最优 | 最优 | 双critic+MCTS训练 |
| w/o 子目标Critic | 明显下降 | 明显下降 | 无法选择好的推理方向 |
| w/o 执行Critic | 中等下降 | 中等下降 | 无法过滤低质量的推理/检索 |
| w/o 检索 | 在知识密集任务上大幅下降 | 轻微下降 | 编程任务对检索依赖低 |
| 随机Critic（不训练） | 接近无Critic | 接近无Critic | 证明训练过的Critic至关重要 |

### 关键发现
- 子目标Critic贡献大于执行Critic——说明"方向正确"比"执行正确"更重要（方向错了即使执行完美也没用）
- CR-Planner在竞赛编程上提升最大——因为竞赛编程需要精确的算法选择和实现，critic的引导避免了早期的算法选择错误
- 检索增强的价值因任务而异——数学推理主要靠推理能力（检索帮助不大），领域知识问题则高度依赖检索
- MCTS收集的训练数据质量远高于简单的正负样本标注，critic的学习效率很高

## 亮点与洞察
- **用MCTS自动收集过程监督信号**是本文最大的技术亮点——解决了"如何告诉模型每一步是好是坏"的核心难题。MCTS天然适合评估树状搜索中间节点的价值，与推理规划问题完美匹配
- **双Critic设计（子目标+执行）**提供了层次化的质量控制——高层Critic负责策略方向，低层Critic负责执行质量，分工明确且互补
- 框架具有良好的通用性——理论上可以应用于任何需要"规划+搜索+批评"的复杂推理任务

## 局限与展望
- MCTS在训练阶段的计算成本很高——每个训练样本需要大量的搜索rollout
- Critic模型的泛化能力有限——在训练任务之外的新任务上可能需要重新训练
- 推理时的搜索（beam search）增加了延迟，不适合实时应用
- 未来方向：(1) 探索更高效的Critic训练方法；(2) 研究通用Critic的可能性（跨任务泛化）；(3) 与自我对弈（self-play）和过程奖励模型（PRM）的关系

## 相关工作与启发
- **vs Tree-of-Thought**: ToT也使用搜索树进行推理，但缺乏训练过的Critic来引导搜索，只依赖LLM自身的评估能力
- **vs RAG**: 传统RAG只在输入阶段做一次检索，CR-Planner在推理过程中动态决定何时检索、检索什么
- **vs Process Reward Model (PRM)**: PRM为每个推理步骤提供奖励信号，与CR-Planner的执行Critic理念相似，但CR-Planner额外有子目标层的引导

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 双Critic +MCTS训练的框架设计巧妙，融合了规划、搜索和批评三种机制
- 实验充分度: ⭐⭐⭐⭐ 三种不同类型的挑战性任务+详细消融
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，实验分析有深度
- 价值: ⭐⭐⭐⭐⭐ 为LLM推理能力的提升提供了系统性的方法论

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery](learning_from_litigation_graphs_and_llms_for_retrieval_and_reasoning_in_ediscove.md)
- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [\[ACL 2025\] Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](graph_descriptive_order_llm.md)
- [\[ACL 2025\] Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)

</div>

<!-- RELATED:END -->
