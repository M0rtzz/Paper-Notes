---
title: >-
  [论文解读] MasRouter: Learning to Route LLMs for Multi-Agent Systems
description: >-
  [ACL 2025][LLM 其他][多智能体路由] 首次提出多智能体系统路由（MASR）问题，设计 MasRouter 级联控制器网络，依次决定协作模式、角色分配和 LLM 路由，在保持高性能的同时将 MAS 的推理成本降低最高 52%，实现效果与效率的平衡。 领域现状：基于 LLM 的多智能体系统（MAS）通过多个 LL…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "多智能体路由"
  - "LLM选择"
  - "协作模式"
  - "级联控制器"
  - "成本效率"
---

# MasRouter: Learning to Route LLMs for Multi-Agent Systems

**会议**: ACL 2025  
**arXiv**: [2502.11133](https://arxiv.org/abs/2502.11133)  
**代码**: [https://github.com/yanweiyue/masrouter](https://github.com/yanweiyue/masrouter)  
**领域**: LLM/NLP  
**关键词**: 多智能体路由、LLM选择、协作模式、级联控制器、成本效率

## 一句话总结

首次提出多智能体系统路由（MASR）问题，设计 MasRouter 级联控制器网络，依次决定协作模式、角色分配和 LLM 路由，在保持高性能的同时将 MAS 的推理成本降低最高 52%，实现效果与效率的平衡。

## 研究背景与动机

**领域现状**：基于 LLM 的多智能体系统（MAS）通过多个 LLM agent 的协作来突破单模型的能力边界，在代码生成、数学推理、知识问答等任务上展现了显著优势。常见的 MAS 框架包括 MapCoder（多角色代码生成）和 GPTSwarm（基于图的智能体协作）。

**现有痛点**：当前 MAS 的核心问题是**成本高昂**——每个 agent 都调用强力 LLM（如 GPT-4），多轮交互下 API 花费巨大。现有的 LLM routing 方法（如 RouterBench、RouteLLM）仅针对单 agent 场景设计，只考虑"哪个查询用哪个模型"，完全忽略了 MAS 中更丰富的决策维度：应该用哪种协作模式？每个角色用什么模型？简单任务是否需要多 agent 协作？

**核心矛盾**：MAS 的强大能力来自多 agent 协作，但并非所有任务都需要全配置的 MAS。很多简单查询用单 agent 就够了，复杂查询则需要多 agent 但不一定每个角色都用最贵的模型。缺乏智能路由导致了系统性的资源浪费。

**本文目标**：将 LLM routing 从单 agent 扩展到多 agent 场景，统一管理协作模式选择、角色分配和模型选择三个维度的决策。

**切入角度**：将 MAS 的构建过程建模为一个级联决策问题——先决定"要不要协作"，再决定"谁干什么"，最后决定"用什么模型"。通过学习一个轻量控制器来自动做这些决策。

**核心 idea**：用级联控制器网络在 query 层面动态构建最优的 MAS 配置，实现"该简单时简单、该复杂时复杂"的自适应路由。

## 方法详解

### 整体框架

MasRouter 接收一个用户查询作为输入，通过三级级联控制器依次输出：（1）协作模式（单 agent / 多 agent debate / 多 agent workflow 等）；（2）角色配置（如果是多 agent，每个位置分配什么角色）；（3）LLM 选择（每个 agent 调用哪个 LLM）。最终按照路由结果构建 MAS 并执行查询。整个路由过程本身是轻量的，不需要调用大模型。

### 关键设计

1. **协作模式决定器（Collaboration Mode Determination）**:

    - 功能：判断当前查询是否需要多 agent 协作，以及需要哪种协作模式
    - 核心思路：使用一个轻量分类器（如小型 BERT 或 MLP），输入 query embedding，输出协作模式类别。训练数据通过对历史查询在不同模式下的性能对比来自动标注——对于每个查询，分别在单 agent 和各种多 agent 模式下执行，选择性能最优且成本最低的模式作为标签。预设的协作模式包括：单 agent 直接回答、双 agent debate、多 agent workflow（有编排）等。
    - 设计动机：简单查询（如事实性问答）不需要多 agent 开销，只对真正需要协作的复杂任务启用 MAS，从根本上避免资源浪费。

2. **角色分配器（Role Allocation）**:

    - 功能：为多 agent 配置中的每个位置分配合适的角色
    - 核心思路：如果协作模式决定器选择了多 agent 模式，则激活角色分配器。同样基于 query 特征，从预定义的角色池（如 coder、reviewer、tester、planner 等）中选择合适的角色组合。分配器使用多标签分类或序列预测来生成角色列表。训练信号来自不同角色组合在不同查询上的性能差异。
    - 设计动机：不同任务需要不同的角色组合。代码生成可能需要 coder + reviewer，而数学推理可能需要 solver + verifier。自适应角色分配比固定角色更灵活高效。

3. **LLM 路由器（LLM Router）**:

    - 功能：为每个 agent 角色选择最合适的 LLM
    - 核心思路：在确定了角色后，为每个角色独立路由到一个 LLM。路由决策考虑三个因素：task-LLM 的匹配度（不同 LLM 擅长不同类型的任务）、成本约束（在预算内选择最优模型）、以及角色需求（如 reviewer 不需要最强的模型）。可选的 LLM 池包括 GPT-4、GPT-3.5、Claude、Llama 等不同价格和能力层级的模型。使用一个学习到的评分函数 $s(q, r, m)$ 来评估查询 $q$、角色 $r$ 和模型 $m$ 的三方匹配度。
    - 设计动机：核心 insight 是"不是每个角色都需要最强的模型"。例如，代码 reviewer 可能只需要理解能力而不需要最强的生成能力，用 GPT-3.5 足矣。差异化路由可以在不显著降低效果的情况下大幅节省成本。

### 损失函数 / 训练策略

MasRouter 的训练采用多任务学习框架。每一级控制器有各自的分类损失。此外引入了一个全局的性能-成本权衡目标：$L = L_{perf} + \lambda \cdot L_{cost}$，其中 $L_{perf}$ 衡量路由方案的任务完成质量，$L_{cost}$ 惩罚过高的推理开销，$\lambda$ 是权衡系数。训练数据通过在多种配置下运行查询来构建。

## 实验关键数据

### 主实验

| 数据集 | 指标 | GPT-4 All | SOTA Router | MasRouter | 成本节省 |
|--------|------|-----------|-------------|-----------|---------|
| HumanEval | Pass@1 | 87.8 | 84.3 | 86.5 | 52.07% |
| MBPP | Pass@1 | 78.2 | 76.4 | 82.7 | 38.2% |
| MATH | Accuracy | 76.5 | 73.1 | 75.8 | 41.5% |
| GSM8K | Accuracy | 94.2 | 91.8 | 93.5 | 35.7% |
| MMLU | Accuracy | 86.7 | 84.5 | 86.1 | 28.3% |

### 消融实验

| 配置 | MBPP Pass@1 | HumanEval Pass@1 | 成本（相对） | 说明 |
|------|------------|------------------|------------|------|
| Full MasRouter | 82.7 | 86.5 | 47.93% | 完整级联路由 |
| w/o Mode Selection | 79.1 | 83.2 | 68.5% | 固定多 agent 模式 |
| w/o Role Allocation | 80.3 | 84.8 | 55.2% | 固定角色配置 |
| w/o LLM Routing | 81.5 | 85.7 | 72.1% | 固定使用 GPT-4 |
| Random Routing | 72.6 | 78.3 | 49.8% | 随机路由 |

### 关键发现
- **协作模式选择是成本节省的最大来源**：去掉模式选择后成本从 47.93% 跳到 68.5%，说明很多查询确实不需要多 agent 协作。
- **MBPP 上性能提升最明显**（比 SOTA router 高 8.2%）：这说明代码生成任务中，合理的角色组合（coder + reviewer）比随意的多模型组合更有效。
- **成本最高可节省 52%**：在 HumanEval 上几乎保持原有性能的同时将成本砍半，大幅提升了 MAS 的实用性。
- **与主流 MAS 框架即插即用**：集成到 MapCoder 和 GPTSwarm 后分别降低了 17.21% 和 28.17% 的开销，说明框架的通用性和迁移性好。

## 亮点与洞察
- **首次定义 MASR 问题**：将多 agent 系统的构建从"手动配置"提升为"自动路由"，这是一个有价值的问题定义层面的贡献。MASR 问题的三个维度（协作模式、角色、模型）的解耦和级联设计很优雅。
- **"不是每个 agent 都需要 GPT-4"的实证**：这个看似直觉性的发现，通过系统化的实验得到了量化验证。这对 MAS 的工程实践有直接指导意义——可以用异构模型组成最优性价比的多 agent 团队。
- **即插即用的设计**：MasRouter 不改变底层 MAS 框架的逻辑，只是在"建造"MAS 的阶段做智能选择，因此可以无缝集成到各种已有框架中。

## 局限与展望
- **训练数据收集成本高**：需要在大量查询上尝试多种 MAS 配置来构建训练数据，这本身就需要大量 API 调用。
- **路由决策的泛化性**：在一个数据集上训练的路由器能否泛化到全新的任务类型尚未充分验证。
- **级联错误传播**：如果协作模式选错了，后续的角色分配和 LLM 路由再优也无法补救。
- **仅考虑 API-based 模型**：对于本地部署的开源模型，成本计算方式不同（主要是 GPU 时间而非 API 费用），路由策略需要调整。
- 未来方向：可以探索在线学习的路由策略（根据实际运行反馈持续优化路由），以及面向延迟约束而非仅成本约束的路由。

## 相关工作与启发
- **vs RouteLLM**: RouteLLM 是单 agent 场景下的 LLM 路由方法，只决定用哪个模型。MasRouter 在此基础上新增了协作模式和角色分配两个维度，是 RouteLLM 的"多 agent 版本"。
- **vs FrugalGPT**: FrugalGPT 通过模型级联（先试便宜的，不行再试贵的）来节省成本。MasRouter 的思路更宏观——不仅选模型，还选协作方式和角色，是更全面的优化。
- **vs MapCoder/GPTSwarm**: 这些是具体的 MAS 框架，MasRouter 是它们上层的"路由调度器"，两者是互补关系。实验也验证了 MasRouter 可以直接集成到这些框架中降低开销。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义 MASR 问题，级联控制器设计新颖实用
- 实验充分度: ⭐⭐⭐⭐ 五个数据集上的验证全面，消融和集成实验都有
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述系统
- 价值: ⭐⭐⭐⭐⭐ 对 MAS 的工程落地有重要意义，直接降低部署成本

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Red-Teaming LLM Multi-Agent Systems via Communication Attacks](red-teaming_llm_multi-agent_systems_via_communication_attacks.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](agentdropout_dynamic_agent_elimination_for_token-efficient_and_high-performance_.md)
- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [\[ACL 2025\] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](virsci_multi_agent_idea_gen.md)

</div>

<!-- RELATED:END -->
