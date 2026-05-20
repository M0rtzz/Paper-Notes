---
title: >-
  [论文解读] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration
description: >-
  [ACL 2025][LLM/NLP][多智能体协作] 本文提出AgentDropout，通过在每轮通信中动态删除冗余Agent节点和通信边来优化多智能体系统的通信拓扑，相比SOTA方法平均减少21.6%的prompt token消耗和18.4%的completion token消耗，同时性能提升1.14分。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "多智能体协作"
  - "通信拓扑优化"
  - "节点剪枝"
  - "边剪枝"
  - "Token效率"
---

# AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration

**会议**: ACL 2025  
**arXiv**: [2503.18891](https://arxiv.org/abs/2503.18891)  
**代码**: [https://github.com/wangzx1219/AgentDropout](https://github.com/wangzx1219/AgentDropout)  
**领域**: LLM/NLP  
**关键词**: 多智能体协作、通信拓扑优化、节点剪枝、边剪枝、Token效率

## 一句话总结
本文提出AgentDropout，通过在每轮通信中动态删除冗余Agent节点和通信边来优化多智能体系统的通信拓扑，相比SOTA方法平均减少21.6%的prompt token消耗和18.4%的completion token消耗，同时性能提升1.14分。

## 研究背景与动机

**领域现状**：基于LLM的多智能体系统（MAS）通过让多个Agent相互通信协作来解决复杂任务，已在推理、数学和代码生成等任务上展现了巨大潜力。MAS的通信结构通常建模为有向图，节点代表Agent，边代表通信路径。

**现有痛点**：MAS面临两大核心问题：（1）高token消耗——多个Agent之间频繁的消息生成和传递带来巨大的token开销；（2）冗余通信——并非所有Agent间的信息交换都对最终结果有正向贡献。AgentPrune等方法通过剪枝冗余边来减少通信，但它在所有通信轮次中使用固定参与角色，限制了优化效果。

**核心矛盾**：现有方法只考虑了"哪些通信是冗余的"（边剪枝），忽略了"哪些Agent角色是冗余的"（节点剪枝）。不同讨论阶段可能需要不同的Agent组合，固定角色分配无法适应任务的动态需求。

**本文目标**：同时优化Agent角色分配和通信路径，在每轮讨论中动态调整参与的Agent及其通信方式。

**切入角度**：借鉴管理学中高效团队动态调整角色和协作方式的理论，将多智能体系统类比为人类团队，不同阶段让最适合的成员参与讨论。

**核心 idea**：通过Node Dropout（删除冗余Agent）和Edge Dropout（删除冗余通信边）两步优化策略，动态调整MAS的通信拓扑以提升效率和性能。

## 方法详解

### 整体框架
AgentDropout将MAS的通信结构建模为多轮有向图。输入是一组Agent和它们的通信拓扑，输出是每轮优化后的通信图。方法分为两个阶段：首先通过Node Dropout确定每轮应该参与的Agent子集，然后通过Edge Dropout进一步剪枝Agent间的冗余通信边。

### 关键设计

1. **Node Dropout（节点剪枝）**:

    - 功能：在每轮通信中移除贡献最小的Agent节点
    - 核心思路：首先将通信图的邻接矩阵初始化为可训练的连续权重矩阵（权重范围0-1）。通过策略梯度方法优化轮内邻接矩阵，目标是最大化任务性能。优化后，计算每个节点在每轮中的加权入度和出度之和，使用TopkNodes函数保留度数最大的 $k = (1-\alpha) \times N$ 个节点，删除其余节点及其关联的所有边。这里 $\alpha$ 是节点dropout率
    - 设计动机：不同讨论阶段需要不同的专业角色。例如在代码任务中，第一轮可能需要架构师，第二轮需要具体编码者。固定角色会导致某些Agent在特定阶段"白占"token预算

2. **Edge Dropout（边剪枝）**:

    - 功能：在节点裁剪后的图上进一步删除低贡献的通信边
    - 核心思路：在Node Dropout之后，使用相同的策略梯度优化方法训练轮内和轮间邻接矩阵。重新训练后，根据优化后的边权重，使用top-k选择保留最重要的边，删除权重较低的边以获得更稀疏的通信图。最终得到每轮参与Agent不同、通信路径不同的动态拓扑
    - 设计动机：即使确定了每轮参与的Agent，它们之间也不需要全连接通信。某些信息传递路径可能是冗余的或有害的（如传递错误信息）

3. **DAGSample采样算法**:

    - 功能：从概率化的邻接矩阵中采样离散通信图，保证有向无环
    - 核心思路：以每条边的权重作为采样概率独立采样，并通过后处理确保采样结果是有向无环图（DAG）。这保证了Agent间的通信不会形成死循环
    - 设计动机：通信图需要有向无环才能保证消息传递的正确顺序，随机采样可能生成有环图

### 损失函数 / 训练策略
采用策略梯度（REINFORCE）方法优化邻接矩阵。对每个训练样本采样 $M$ 个通信图，评估其在任务上的性能得分 $\mu(\mathcal{G}_m)$，通过概率加权平均近似期望梯度进行优化。使用gradient ascent更新邻接矩阵权重。

## 实验关键数据

### 主实验

| 方法 | 边剪枝 | 节点剪枝 | MMLU | GSM8K | HumanEval | 平均 |
|------|--------|---------|------|-------|-----------|------|
| Vanilla MAS (T轮) | ✗ | ✗ | 60.13 | 71.48 | 49.17 | 65.72 |
| AgentPrune | ✓ | ✗ | 60.78 | 71.02 | 51.67 | 66.51 |
| **AgentDropout** | ✓ | ✓ | **62.75** | **73.13** | **55.84** | **68.70** |

在Qwen2.5-72B上：

| 方法 | 边剪枝 | 节点剪枝 | MMLU | GSM8K | HumanEval | 平均 |
|------|--------|---------|------|-------|-----------|------|
| Vanilla MAS (T轮) | ✗ | ✗ | 84.31 | 93.28 | 87.08 | 90.76 |
| AgentPrune | ✓ | ✗ | 83.66 | 93.67 | 86.67 | 90.81 |
| **AgentDropout** | ✓ | ✓ | **84.97** | **93.75** | **87.92** | **91.58** |

### 消融实验

| 配置 | 平均性能 | Prompt Token减少 | 说明 |
|------|---------|-----------------|------|
| Full AgentDropout | 68.70 | -21.6% | 完整模型 |
| 仅Edge Dropout | 66.51 | -14.2% | 等同AgentPrune |
| 仅Node Dropout | 67.80 | -18.3% | 节点裁剪贡献更大 |
| 无优化 | 65.72 | 0% | 原始MAS |

### 关键发现
- Node Dropout的贡献大于Edge Dropout，表明"谁参与讨论"比"谁和谁沟通"更重要
- 在更大更强的模型（如Qwen2.5-72B、DeepSeek-V3）上，AgentDropout仍能带来显著提升，说明即使模型能力已经很强，通信拓扑优化仍有价值
- 领域迁移性良好——在一个任务上学到的通信拓扑可以迁移到其他任务
- 结构鲁棒性强——即使初始拓扑结构变化，优化后的结果趋向一致

## 亮点与洞察
- 将Dropout从神经网络正则化迁移到多智能体系统拓扑优化中，概念简洁而有效。这种类比思维将Agent类比为神经元，通信边类比为连接权重，自然地复用了Dropout的正则化效果
- 动态角色分配的思想源于管理学理论，这种跨学科启发值得借鉴。高效团队确实需要根据任务阶段动态调整成员角色
- 方法具有很好的即插即用特性——可以应用于任何基于通信图的MAS框架，无需修改底层LLM

## 局限与展望
- 训练通信拓扑需要多次LLM推理（M次采样 × 多轮训练），计算成本较高
- 节点dropout率 $\alpha$ 是一个需要调优的超参数，不同任务可能需要不同设置
- 当前方法假设所有Agent使用相同的LLM，对异构Agent系统的优化尚未探索
- 仅在非对话类任务上验证，对于需要所有角色持续参与的场景（如辩论）可能不适用

## 相关工作与启发
- **vs AgentPrune**: AgentPrune只做边剪枝且跨轮使用相同策略，AgentDropout同时做节点和边剪枝且每轮动态调整，性能和效率均更优
- **vs GPTSwarm**: GPTSwarm通过图神经网络优化MAS结构，但需要训练额外的GNN。AgentDropout直接优化邻接矩阵，更简洁
- **vs DyLAN**: DyLAN通过动态选择Agent来提升LLM推理，但没有考虑通信边的优化

## 评分
- 新颖性: ⭐⭐⭐⭐ Dropout思想迁移到MAS拓扑优化很有创意，但整体是已有技术的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖三种基座模型、六个数据集，消融和迁移实验充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机阐述有说服力
- 价值: ⭐⭐⭐⭐ 为MAS效率优化提供了实用方案，开源代码促进复现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [\[ACL 2025\] MasRouter: Learning to Route LLMs for Multi-Agent Systems](masrouter_learning_to_route_llms_for_multi-agent_systems.md)
- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)
- [\[ACL 2025\] Dynamic Parallel Tree Search for Efficient LLM Reasoning](dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)

</div>

<!-- RELATED:END -->
