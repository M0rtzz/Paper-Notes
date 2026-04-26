---
title: >-
  [论文解读] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation
description: >-
  [ACL 2026][多智能体] MARS2 提出多智能体强化树搜索框架，将多个独立优化的策略嵌入共享搜索树中协作探索，通过 Thompson 采样选择智能体-节点对、树一致性奖励塑形和路径级组优势估计，在代码生成基准上一致提升单模型 Pass@1 最高 8.0%、系统级 Pass@1(MCTS) 最高 6.5%。
tags:
  - ACL 2026
  - 多智能体
  - 树搜索
  - 强化学习
  - 代码生成
  - GRPO
---

# MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation

**会议**: ACL 2026  
**arXiv**: [2604.14564](https://arxiv.org/abs/2604.14564)  
**代码**: https://github.com/TsinghuaC3I/MARTI  
**领域**: 代码智能  
**关键词**: 多智能体, 树搜索, 强化学习, 代码生成, GRPO

## 一句话总结

MARS2 提出多智能体强化树搜索框架，将多个独立优化的策略嵌入共享搜索树中协作探索，通过 Thompson 采样选择智能体-节点对、树一致性奖励塑形和路径级组优势估计，在代码生成基准上一致提升单模型 Pass@1 最高 8.0%、系统级 Pass@1(MCTS) 最高 6.5%。

## 研究背景与动机

**领域现状**：GRPO 等 RLVR 范式在代码生成等推理任务上取得显著进展。搜索增强 RL（如 TreeRL）通过引入 MCTS 树结构提供更多样的探索信号。多智能体 RL（MARL）通过多策略交互产生非平稳数据分布，有望突破单策略探索的限制。

**现有痛点**：(1) 单智能体树搜索受限——整棵搜索树由单一策略分布驱动，训练后期搜索行为集中在少数高概率分支，探索增益递减；(2) 多智能体方法与结构化搜索脱节——现有多智能体推理框架（辩论、投票等）仅做轻量级协调，缺乏分支、回溯和预算分配等结构化搜索支持。

**核心矛盾**：单策略搜索会收敛到局部最优（挑战1），多智能体协作缺乏搜索结构（挑战2）。需要将两者统一。

**本文目标**：构建一个多智能体协作的树搜索 RL 框架，使异构智能体在共享搜索树中协作生成和精炼候选解。

**切入角度**：将搜索树视为可学习的多智能体交互环境，不同智能体贡献不同的策略先验，通过 Thompson 采样动态分配探索预算。

**核心 idea**：多智能体在共享搜索树上协作扩展节点，每个智能体独立优化，奖励信号通过树一致性奖励塑形结合父节点和兄弟节点信息，路径级组优势确保跨复杂搜索轨迹的稳定信用分配。

## 方法详解

### 整体框架

多个异构 LLM（如 Qwen3 + AReaL）作为独立智能体，在共享搜索树上协作探索。每步通过 Thompson 采样先选智能体再选节点——选择生成节点（横向扩展新解）或精炼节点（纵向改进已有解）。扩展后对每个节点执行测试用例获得奖励，通过树一致性奖励塑形和路径级组优势对每个智能体独立优化。

### 关键设计

1. **Thompson 采样智能体-节点选择**:

    - 功能：动态平衡异构智能体的探索预算分配
    - 核心思路：为每个智能体-节点对维护 Beta 先验，先 Thompson 采样选智能体，再对该智能体的可扩展节点 Thompson 采样选节点。区分生成节点（新建候选）和精炼节点（改进已有候选），实现横向探索与纵向深入的动态平衡
    - 设计动机：不同智能体擅长不同类型的问题，Thompson 采样在探索-利用之间自适应平衡

2. **树一致性奖励塑形（Tree-Consistent Reward Shaping）**:

    - 功能：在多智能体共享搜索树上实现层次化的信用分配
    - 核心思路：对每个非根节点 $v$，定义混合基线 $b(v) = (1-\lambda) r_{p(v)} + \lambda \cdot \mu_{C(p(v)) \setminus v}$（父节点奖励和兄弟节点平均奖励的加权组合）。结构一致性增益 $\Delta(v) = r_v - b(v)$，塑形后奖励 $\hat{r}_v = r_v + \gamma \cdot \Delta(v)$
    - 设计动机：子节点不仅要全局奖励高，还要相对父节点有改进（纵向）且优于兄弟候选（横向），鼓励协作中的专业化

3. **路径级组优势估计**:

    - 功能：将 GRPO 的组相对优势从并行采样扩展到树结构
    - 核心思路：搜索树中所有节点来自同一问题，构成天然语义组。用塑形后奖励计算树级组相对优势：$\hat{A}_{v,j} = (\hat{r}_{v,j} - \text{mean}) / \text{std}$，每个智能体仅在自己生成的节点上优化
    - 设计动机：标准 GRPO 的并行轨迹假设在树搜索中不成立，需要考虑父子和兄弟的层次依赖

### 损失函数 / 训练策略

扩展 GRPO 为 MARS2 目标：每个智能体独立优化，使用 DAPO 的 clip-higher 技巧，KL 正则化。训练数据为 DeepCoder 的 7992 个代码生成题目（过滤后）。评估基准为 LiveCodeBench v6（2025.01-05）。

## 实验关键数据

### 主实验

| 模型/系统 | 方法 | Pass@1 | Pass@1(MCTS) | Pass@N |
|-----------|------|--------|-------------|--------|
| Qwen3-8B | Base | 50.3 | 54.3 | 68.6 |
| Qwen3-8B | GRPO | 52.5 (+2.2) | 57.1 (+2.8) | 73.1 |
| Qwen3-8B | RS2 | 55.4 (+5.1) | 60.6 (+6.3) | 71.4 |
| Qwen3-8B | **MARS2** | **58.3 (+8.0)** | **60.8 (+6.5)** | 72.3 |
| AReaL-14B | GRPO | 58.9 (+0.5) | 60.7 (-2.2) | 75.4 |
| AReaL-14B | **MARS2** | **64.6 (+6.2)** | **68.1 (+5.2)** | 80.2 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| GRPO (单智能体无搜索) | +2.2 Pass@1 | 基线 |
| RS2 (单智能体+树搜索) | +5.1 Pass@1 | 搜索结构有用 |
| MARS2 (多智能体+树搜索) | +8.0 Pass@1 | 多智能体进一步提升 |
| 加入弱智能体 (DeepCoder) | 性能仍提升 | 对智能体异质性鲁棒 |

### 关键发现

- MARS2 在所有模型上一致超越 GRPO 和单智能体树搜索（RS2），Pass@1 提升最高 8.0%
- 对已高度优化的代码模型（AReaL），GRPO 几乎无效甚至退化，MARS2 仍能提升 6.2%
- 多智能体系统级 Pass@1(MCTS) 提升最高 6.0%，证明多智能体训练确实产生了互补的策略
- 引入弱智能体（DeepCoder-14B）后性能仍有提升，说明框架对智能体异质性鲁棒
- AReaL-14B 在 MARS2 下达到 64.6% Pass@1，超过 O4-Mini (Low) 的 63.7%

## 亮点与洞察

- 将搜索树视为"可学习的多智能体交互环境"而非静态采样过程，是范式创新。每个节点扩展都是智能体间的协作决策
- 树一致性奖励塑形同时考虑纵向改进（vs 父节点）和横向竞争（vs 兄弟节点），是多智能体信用分配在树结构上的自然推广
- 实验设计严谨：训练和推理配置明确分离，所有方法共享相同数据预算和推理框架

## 局限与展望

- 仅在代码生成上评估，数学推理等其他 RLVR 场景待验证
- 智能体数量固定为 2，更多智能体的 scaling 行为未探索
- Thompson 采样的先验更新规则较简单，更复杂的 bandit 策略可能更优
- 多智能体训练需要同时运行多个模型，GPU 资源需求倍增

## 相关工作与启发

- **vs TreeRL**: TreeRL 用单策略驱动搜索树，训练后期探索增益递减。MARS2 引入多策略打破单策略先验的限制
- **vs MAPoRL**: MAPoRL 用多智能体对话协作但缺乏搜索结构。MARS2 将多智能体嵌入树搜索，提供分支和回溯支持

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将多智能体 RL 与树搜索统一，树一致性奖励塑形设计精巧
- 实验充分度: ⭐⭐⭐⭐ 多模型、多规模，但仅代码生成任务
- 写作质量: ⭐⭐⭐⭐ 框架清晰，公式严谨
- 价值: ⭐⭐⭐⭐⭐ 为搜索增强 RL 提供了新范式，性能提升显著

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] MARS²: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi_agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)

<!-- RELATED:END -->
