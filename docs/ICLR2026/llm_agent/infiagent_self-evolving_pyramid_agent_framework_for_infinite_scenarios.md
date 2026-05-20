---
title: >-
  [论文解读] InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios
description: >-
  [ICLR 2026][LLM Agent][multi-agent system] 提出 InfiAgent，一个基于 DAG 的金字塔式多智能体框架，通过 agent-as-a-tool 机制实现自动化的层级任务分解、双重审计质量保障、智能路由和自演化能力，在多个推理基准上比 ADAS 平均提升 9.9%…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "multi-agent system"
  - "DAG"
  - "agent-as-a-tool"
  - "self-evolution"
  - "task decomposition"
  - "hierarchical agent"
---

# InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios

**会议**: ICLR 2026  
**arXiv**: [2509.22502](https://arxiv.org/abs/2509.22502)  
**代码**: 待公开  
**领域**: LLM Agent  
**关键词**: multi-agent system, DAG, agent-as-a-tool, self-evolution, task decomposition, hierarchical agent

## 一句话总结

提出 InfiAgent，一个基于 DAG 的金字塔式多智能体框架，通过 agent-as-a-tool 机制实现自动化的层级任务分解、双重审计质量保障、智能路由和自演化能力，在多个推理基准上比 ADAS 平均提升 9.9%。

## 研究背景与动机

当前基于 LLM 的智能体系统面临几个核心挑战：

**手工设计瓶颈**：现有智能体需要精心设计的工作流、提示词和迭代调优，这要求同时具备 LLM 技术和领域专业知识，严重限制了跨行业的可扩展性

**协调开销大**：传统多智能体架构采用点对点协作模型，允许智能体之间自由交互，导致协调开销增大、死锁和不可预测的行为

**缺乏自适应性**：现有框架依赖人工编写模板，缺乏系统推理能力，无法自主适应新任务和优化需求

**稳定性问题**：随着系统复杂度增加，智能体间的不可预测交互、资源冲突和涌现行为不稳定性问题日益突出

InfiAgent 旨在提供一个通用框架，无需大量手动配置即可自动适应多样化问题域。

## 方法详解

### 整体框架

InfiAgent 设计为基于 DAG（有向无环图）的分解和路由系统。与传统强调直接执行的架构不同，InfiAgent 中大多数智能体专注于**规划和路由**任务，实际执行委托给底层的功能智能体组。每个高层智能体充当少量专门低层智能体的规划器，编排它们的协作并合并结果。整体框架呈金字塔形，顶层 Router 负责重定向所有用户查询以避免逐层搜索。

### 关键设计

#### 1. Agent-as-a-Tool 机制与智能路由

核心原理是将智能体作为工具进行分解和路由。当任务 $T_0$ 提交给顶层智能体 $\alpha$ 时：

- 自动识别合适的低层智能体 $\{A_1, A_2, \ldots, A_k\}$
- 将任务重新表述为子任务 $\{T_1, T_2, \ldots, T_k\}$
- 每个低层智能体作为专门工具被高层智能体调用

分解过程形式化为：

$$T^{(l)} \mapsto \{T_1^{(l+1)}, T_2^{(l+1)}, \ldots, T_{k_l}^{(l+1)}\}$$

为保持每个智能体的简单性，框架施加严格约束：$k_l \leq K_{\max}$，其中 $K_{\max} = 5$，确保即使系统指数增长，单个智能体也不会面临过大的协调复杂度。

#### 2. 架构可扩展性

利用深度实现功能智能体的指数增长：当平均分支因子为 $b$ 时，深度 $L$ 处可达的功能智能体数为 $N_{\text{func}} \approx b^L$。这使得顶层智能体拥有广泛的泛化能力，同时每个中间智能体只需推理有限数量的子节点。

#### 3. 双重审计质量保障

InfiAgent 实现两层审计机制：

- **执行级审计**：持续监控和验证每个智能体的输出。维护质量评分 $Q_i^{(t+1)} = \alpha \cdot Q_i^{(t)} + (1-\alpha) \cdot \text{validate}(O_i^{(t)})$
- **系统级审计**：通过内置审查机制和回顾性摘要维护系统稳定性，同时进行上下文压缩以节省 token

#### 4. 轻量通信与上下文控制

智能体之间仅交换文件描述符和元数据：$M_{i \to j} = (\text{addr}, \text{desc})$。执行上下文分解为四个结构化组件：

- **系统提示上下文** $C_{\text{sys}}$：引导智能体行为的预定义提示
- **长期记忆索引** $C_{\text{LM}}$：压缩的文件描述符和索引，$C_{\text{LM}} = \text{compress}(\{d(f_i) \mid f_i \in \mathcal{F}\})$
- **短期共享记忆** $C_{\text{SM}}$：动态调用栈，记录活跃的智能体调用树
- **压缩环境交互上下文** $C_{\text{ENV}}$：当 token 长度接近阈值 $\tau$ 时自动压缩

通过约束 $|C| \ll |H|$（$H$ 为完整历史日志），确保上下文长度有界。

#### 5. 自演化机制

三个层次的演化：

- **模型级**：Git 风格工作流，多个轻量模型并行执行，由 Judge 模型 $J$ 评估后合并到主分支 $B_{\text{main}}^{(t+1)} = \text{merge}(B_{\text{main}}^{(t)}, \{\Delta m_i^{(t)} \mid J(\Delta m_i^{(t)}) = 1\})$
- **智能体级**：利用主分支的高质量训练数据持续更新所有并行模型 $m_i^{(t+1)} \leftarrow \text{train}(m_i^{(t)}, D(B_{\text{main}}^{(t)}))$
- **拓扑级**：根据性能模式和新任务需求动态重构 DAG 拓扑，实现领域级专家模型的形成

### 损失函数

框架通过质量评分机制和自演化反馈循环进行优化，核心目标是最大化任务完成质量 $Q_i$ 同时最小化上下文长度 $|C|$。

## 实验关键数据

### 主实验

在五个基准上的表现（均使用 GPT-4o-mini 作为基础模型）：

| 方法 | DROP | HumanEval | MBPP | GSM8K | MATH | 平均 |
|------|------|-----------|------|-------|------|------|
| IO (GPT-4o-mini) | 68.3 | 87.0 | 71.8 | 92.7 | 48.6 | 73.68 |
| CoT | 78.5 | 88.6 | 71.8 | 92.4 | 48.8 | 76.02 |
| CoT SC (5-shots) | 78.8 | 91.6 | 73.6 | 92.7 | 50.4 | 77.42 |
| MedPrompt | 78.0 | 91.6 | 73.6 | 90.0 | 50.0 | 76.64 |
| Self Refine | 70.2 | 87.8 | 69.8 | 89.6 | 46.1 | 72.70 |
| ADAS | 76.6 | 82.4 | 53.4 | 90.8 | 35.4 | 67.72 |
| **InfiAgent** | **82.4** | 89.3 | 71.8 | **93.1** | 35.6 | 74.44 |

### 消融实验

InfiHelper 案例研究——质量评估对比（1-10 分制 peer review）：

| 系统 | 代表论文 | 最佳分数 | 平均分 |
|------|---------|---------|--------|
| AI-Researcher | 多篇 VQ-VAE/GCN | 6 | 4.75 |
| Zochi | Tempest Jailbreak | 6 | 6.0 |
| Sakana-AI | Compositional Regularization | 4 | 4.0 |
| **InfiHelper** | Adaptive Multi-Scale DAS | **7** | **5.67** |

### 关键发现

1. **复杂推理卓越**：DROP 达 82.4%，比最佳基线 CoT SC (78.8%) 高 3.6 个百分点，验证了 agent-as-a-tool 机制在多步推理任务上的优势
2. **数学与代码能力强**：GSM8K 93.1%（最高），HumanEval 89.3% 具有竞争力
3. **专门数学领域受限**：MATH 仅 35.6%，因工具调用框架的 overhead 消耗了本可用于直接数学推演的模型能力
4. **比 ADAS 平均提升 9.9%**：验证层级化分解优于端到端自动生成

## 亮点与洞察

1. **Agent-as-a-Tool 抽象精巧**：将智能体统一视为工具，实现了前所未有的模块化和可重用性，同一框架可处理从科学研究到软件工程的多类任务
2. **金字塔结构 + Router 设计**：通过 Router 直接重定向用户查询避免逐层搜索，显著提升效率
3. **自演化是最大亮点**：类 Git 的模型演化工作流使得系统可自主优化，无需人工干预
4. **轻量通信设计实用**：仅传递 (addr, desc) 对，确保上下文长度有界，解决长期运行任务中的上下文膨胀问题
5. **InfiHelper 生成的论文通过了 IEEE 顶会人工审稿验证**，证明系统在真实科研场景中的实用价值

## 局限性

1. **MATH 基准表现不佳**：对于需要集中推演而非多步分解的挑战性问题，工具调用框架的开销反而成为负担
2. **实验使用统一骨干模型**：为公平比较牺牲了异构模型协作的展示
3. **InfiHelper 案例研究缺乏详细定量评估**：主要依赖 AI 评审打分
4. **自演化机制的收敛性和稳定性未充分分析**
5. **缺乏与更多最新多智能体框架（AutoGen、MetaGPT 等）的直接对比**

## 相关工作与启发

- **ADAS**（Hu et al., 2024）：端到端自动生成智能体框架，InfiAgent 通过结构化分解在平均性能上提升 9.9%
- **AgentGym / EvoAgent**：自我改进和进化智能体，InfiAgent 在此基础上增加了拓扑级演化
- **NADER**（Yang et al., 2025）：协作架构设计框架，强调算法和结构演化对构建弹性系统的重要性
- **启发**：agent-as-a-tool 抽象可推广到更多领域；Git 风格自演化工作流值得在其他系统中借鉴

## 评分

- **创新性**: ⭐⭐⭐⭐ — Agent-as-a-Tool 和金字塔式 DAG 架构是有趣的系统设计创新
- **实用性**: ⭐⭐⭐⭐ — 框架易于扩展部署，InfiHelper 案例展示了真实应用价值
- **实验充分度**: ⭐⭐⭐ — 基准实验充分但案例研究评估偏弱
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数学形式化完整，图示直观
- **综合评分**: ⭐⭐⭐⭐ (7.5/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)
- [\[ICLR 2026\] Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](../../ICML2026/llm_agent/evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ACL 2026\] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](../../ACL2026/llm_agent/agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)
- [\[ACL 2025\] Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](../../ACL2025/llm_agent/gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)

</div>

<!-- RELATED:END -->
