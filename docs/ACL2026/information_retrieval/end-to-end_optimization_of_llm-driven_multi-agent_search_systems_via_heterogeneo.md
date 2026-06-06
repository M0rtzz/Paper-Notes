---
title: >-
  [论文解读] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning
description: >-
  [ACL 2026][信息检索/RAG][多智能体搜索] 本文提出 MHGPO（Multi-Agent Heterogeneous Group Policy Optimization），一种无需 critic 的多智能体 RL 方法，通过异构组相对优势估计和反向奖励传播…
tags:
  - "ACL 2026"
  - "信息检索/RAG"
  - "多智能体搜索"
  - "MARL"
  - "组优化"
  - "端到端优化"
  - "RAG"
---

# End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning

**会议**: ACL 2026  
**arXiv**: [2506.02718](https://arxiv.org/abs/2506.02718)  
**代码**: 无  
**领域**: 信息检索 / 多智能体 RL  
**关键词**: 多智能体搜索, MARL, 组优化, 端到端优化, RAG

## 一句话总结

本文提出 MHGPO（Multi-Agent Heterogeneous Group Policy Optimization），一种无需 critic 的多智能体 RL 方法，通过异构组相对优势估计和反向奖励传播，在三智能体搜索系统（Rewriter→Reranker→Answerer）中实现端到端优化，捕获隐式跨智能体依赖和跨轨迹关联，在 HotpotQA 等多跳 QA 基准上显著优于 MAPPO 和 GRPO 基线。

## 研究背景与动机

**领域现状**：多智能体搜索系统（MASS）通过协调多个专业化 LLM 智能体（配备搜索工具）来分解任务和检索增强推理。常见架构为 Rewriter（将问题分解为检索查询）→ Reranker（从检索结果中选择相关片段）→ Answerer（生成最终答案）。

**现有痛点**：(1) 提示工程和单智能体 SFT 的优化方式工程量大且适应性差；(2) MAPPO 需要大型 critic 网络来评估联合动作，导致不稳定和高内存开销；(3) GRPO 等组优化算法在单上下文设置中有效，但扩展到多上下文 MASS 并非直接——多智能体 rollout 跨越多个有不相交局部上下文的智能体；(4) 上游智能体的输出影响下游行为但没有直接梯度路径（间接依赖），来自同一根查询的 rollout 探索相关但不同的中间决策（隐式跨轨迹关系）。

**核心矛盾**：MASS 需要系统级优化而非单智能体优化——但现有 MARL 方法要么依赖昂贵的 critic（MAPPO），要么无法处理多上下文的跨智能体依赖（GRPO）。

**本文目标**：设计一种高效的无 critic 多智能体 RL 方法，能够捕获间接跨智能体依赖和隐式跨轨迹关联，将优化焦点从局部智能体性能转向全局系统成功。

**切入角度**：参数共享+组优化——所有智能体共享一个 LLM 骨干，通过异构组的相对优势估计来比较来自不同提示的 rollout，并用反向奖励传播将终端奖励归因到上游智能体。

**核心 idea**：异构组优势估计——通过比较来自同一根查询但不同中间决策的 rollout（形成异构组），将优化焦点从"在固定上游输出下选最优本地动作"转向"奖励导致全局成功的系统行为"。

## 方法详解

### 整体框架

输入问题 → 多智能体 rollout 采样（生成 G 个完整轨迹）→ 终端奖励（Answerer 的 F1 与 gold 答案比较）→ 反向奖励传播（从 Answerer 回传到 Reranker 到 Rewriter）→ 异构组优势估计 → 更新共享 LLM 骨干（PPO 目标 + KL 正则化）。

### 关键设计

1. **反向奖励传播**:

    - 功能：将系统级终端奖励归因到上游智能体
    - 核心思路：终端奖励从 Answerer 的输出开始，沿轨迹反向传播到每个上游智能体。对于智能体 k 的第 i 个输出，其共享奖励是所有直接后继智能体（消费了该输出的）的奖励的聚合（默认平均）。加上智能体特定的格式惩罚得到最终奖励
    - 设计动机：即使上游智能体（如 Rewriter）和终端输出之间没有直接梯度路径，反向传播的奖励也能暴露间接依赖——差的检索查询会导致差的最终答案

2. **异构组优势估计**:

    - 功能：从跨轨迹关联中学习全局最优行为
    - 核心思路：标准 GRPO 仅在同一输入的 rollout 间计算相对优势（同构组）。MHGPO 允许组内包含来自不同提示的 rollout（异构组）——例如同一问题但不同 Rewriter 查询导致的不同 Reranker 输入。通过跨轨迹比较，优势信号不再仅选择固定前缀下的最优本地动作，而是奖励导致全局成功的系统行为
    - 设计动机：在 MASS 中，下游智能体的输入取决于上游 rollout——同一智能体在不同上游决策下收到不同输入，形成天然的异构组

3. **三种 Rollout 采样策略**:

    - 功能：平衡采样效率和优化质量
    - 核心思路：IS（独立采样，纯同构组，高冗余）、FoF（在第一个智能体分叉采样 G 次，下游一对一，高效但仅入口智能体有同构组）、RR（轮询，随机化分叉点，平衡全局协调和局部稳定性）
    - 设计动机：IS 冗余严重（n×G 次采样），FoF 高效但下游智能体缺乏同构比较基准，RR 通过概率化分叉点在效率和稳定性间取舍

### 损失函数 / 训练策略

PPO 目标函数 + KL 正则化。参数共享使 MARL 退化为多任务学习。训练 1 epoch，G=4，骨干 Llama3.1-8B-Instruct，Wikipedia dump 作为检索语料库，Contriever 作为检索后端。

## 实验关键数据

### 主实验

**HotpotQA / 2WikiMultihopQA / MuSiQue 上的性能**

| 方法 | HotpotQA F1 | 2WikiMHQA F1(OOD) | MuSiQue F1(OOD) |
|------|------------|-------------------|-----------------|
| Llama3.1-8B（无 RL） | 22.78 | 20.82 | 2.81 |
| PPO | 24.52 | 9.20 | 8.02 |
| GRPO | 27.42 | 11.03 | 9.29 |
| Search-o1 | - | - | - |
| **MHGPO-FoF** | **最高** | **显著更高** | **显著更高** |
| **MHGPO-RR** | **最高级别** | **最高级别** | **最高级别** |

### 消融实验

**采样策略对比**

| 策略 | 采样效率 | 训练稳定性 | 性能 |
|------|---------|----------|------|
| IS | 低（高冗余） | 高 | 中 |
| FoF | 高 | 中 | 高 |
| FoF (os) | 中 | 中 | 高+ |
| RR | 中高 | 高 | **最高** |

### 关键发现

- MHGPO 显著优于 PPO 和 GRPO——无 critic 设计更稳定，异构组捕获了跨智能体依赖
- PPO 训练不稳定且 OOD 性能大幅下降（2WikiMHQA F1 仅 9.20），MHGPO 的 OOD 泛化更好
- RR 策略在效率和性能间取得最佳平衡——概率化的分叉点为所有智能体提供了同构比较机会
- 参数共享+无 critic 设计大幅降低了内存和计算开销

## 亮点与洞察

- 首次系统研究组优化算法在多智能体搜索系统中的应用
- 异构组优势估计是对 GRPO 的自然扩展，将优化焦点从局部转向全局
- 反向奖励传播是处理间接跨智能体依赖的简洁有效方案

## 局限与展望

- 仅在三智能体 MASS 架构上验证，更复杂拓扑的效果未知
- 参数共享可能限制智能体间的角色分化
- 训练仅 1 epoch，更多训练轮次的效果未探索

## 相关工作与启发

- **vs MAPPO**: MAPPO 需要大型 critic 网络，MHGPO 用组相对优势替代，更高效更稳定
- **vs GRPO**: GRPO 仅支持同构组和单上下文，MHGPO 扩展到异构组和多上下文
- **vs Search-o1**: Search-o1 在单模型内集成检索，MHGPO 优化模块化多智能体系统

## 评分

- 新颖性: ⭐⭐⭐⭐ 异构组优势估计和反向奖励传播是对 GRPO/MARL 的有意义扩展
- 实验充分度: ⭐⭐⭐⭐ 多个数据集含 OOD 评估，但智能体架构较简单
- 写作质量: ⭐⭐⭐⭐ 理论形式化严谨，与 GRPO 的连接分析清晰
- 价值: ⭐⭐⭐⭐⭐ 为 LLM 多智能体系统的端到端 RL 优化提供了实用高效的方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)
- [\[ICML 2026\] Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning](../../ICML2026/information_retrieval/graph-r1_towards_agentic_graphrag_framework_via_end-to-end_reinforcement_learnin.md)
- [\[ACL 2025\] Gumbel Reranking: Differentiable End-to-End Reranker Optimization](../../ACL2025/information_retrieval/gumbel_reranking.md)
- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)

</div>

<!-- RELATED:END -->
