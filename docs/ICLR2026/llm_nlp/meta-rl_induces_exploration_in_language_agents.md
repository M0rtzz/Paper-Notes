---
title: >-
  [论文解读] Meta-RL Induces Exploration in Language Agents
description: >-
  [ICLR 2026][LLM/NLP][Meta-RL] 提出 LaMer 框架，将元强化学习（Meta-RL）引入 LLM agent 训练，通过跨 episode 的奖励优化和基于反思的上下文策略适应，使语言智能体学会主动探索环境，在 Sokoban/MineSweeper/Webshop 上分别获得 11%/14%/19% 的绝对性能提升。
tags:
  - ICLR 2026
  - LLM/NLP
  - Meta-RL
  - LLM Agent
  - 探索与利用
  - 多轮交互
  - 跨episode训练
  - 自我反思
---

# Meta-RL Induces Exploration in Language Agents

**会议**: ICLR 2026  
**arXiv**: [2512.16848](https://arxiv.org/abs/2512.16848)  
**代码**: [mlbio-epfl/LaMer](https://github.com/mlbio-epfl/LaMer)  
**领域**: llm_nlp  
**关键词**: Meta-RL, LLM Agent, 探索与利用, 多轮交互, 跨episode训练, 自我反思

## 一句话总结

提出 LaMer 框架，将元强化学习（Meta-RL）引入 LLM agent 训练，通过跨 episode 的奖励优化和基于反思的上下文策略适应，使语言智能体学会主动探索环境，在 Sokoban/MineSweeper/Webshop 上分别获得 11%/14%/19% 的绝对性能提升。

## 研究背景与动机

### 问题背景

近年来 LLM 从对话系统逐渐转向决策型智能体（如 ReAct、Reflexion），能够在多轮文本观测—动作循环中与环境交互。然而现有 RL 训练的 LLM agent 存在核心缺陷：**缺乏主动探索能力**。在需要试错学习的任务中，agent 往往过早收敛到次优策略，无法像人类一样通过系统性探索快速适应新环境。

### 现有方法的不足

**Prompting 方法**（Zero-shot、ReAct、Reflexion）：依赖冻结的 LLM，探索行为有限，性能天花板低

**标准 RL 训练**（PPO、GRPO、GiGPO）：每个 episode 独立采样，策略固定，无法在测试时通过试错进行适应

**离线蒸馏方法**：依赖离线数据，只能模仿而非主动探索；多聚焦于单轮推理而非多轮 agent 任务

### 核心洞察

多轮任务通常在一个 episode 结束时才有稀疏的成功信号。如果将**多个 episode 视为一个 trial**，探索与利用的平衡就自然地转化为**跨 episode 的 RL 问题**——这正是 Meta-RL 的框架。通过在多个不同但相似的环境上训练，agent 被迫学习通用的探索策略。

## 方法详解

### 整体框架

LaMer（LLM Agent with Meta-RL）包含两个核心设计：

1. **跨 episode 训练框架**：鼓励 agent 在早期 episode 中探索，在后续 episode 中利用积累的经验
2. **基于自我反思的上下文策略适应**：无需梯度更新，通过文本反思在上下文中适应策略

### 跨 episode 训练

每个 trial 由 N 个 episode 顺序组成：

$$\mathcal{T} = (\tau^{(0)}, \tau^{(1)}, \dots, \tau^{(N-1)})$$

其中每个 episode 的策略基于之前积累的历史进行适应。关键是定义**跨 episode 的折扣回报**：

$$G_t^{(n)} = \underbrace{g_t^{(n)}}_{\text{episode内}} + \underbrace{\sum_{m=n+1}^{N-1} \gamma_{\text{traj}}^{m-n} g_0^{(m)}}_{\text{跨episode}}$$

其中 $g_t^{(n)} = \sum_{l=t}^{T-1} \gamma_{\text{step}}^{l-t} r_l^{(n)}$ 是 episode 内回报，$\gamma_{\text{traj}}$ 是跨 episode 折扣因子。

最终 Meta-RL 优化目标为：

$$J(\theta) = \mathbb{E}_{\mathcal{T} \sim \pi_\theta} \left[ \sum_{n=0}^{N-1} \gamma_{\text{traj}}^n \sum_{t=0}^{T-1} \gamma_{\text{step}}^t r_t^{(n)} \right]$$

$\gamma_{\text{traj}}$ 控制探索/利用的权衡：小 $\gamma_{\text{traj}}$ 偏向快速利用，大 $\gamma_{\text{traj}}$ 鼓励长期探索。

### 上下文策略适应（自我反思）

每个 episode 结束后，agent 生成文本反思总结之前的经验：

$$\pi_\theta^{(n)}(\cdot) = \pi_\theta(\cdot | \mathcal{H}^{(n)})$$

其中 $\mathcal{H}^{(n)}$ 是包含历史轨迹和反思的 inter-episode 记忆。反思步骤本身也通过下一个 episode 获得的奖励来训练。

### 与标准 RL 的关键差异

- **标准 RL**：为每个任务独立采样一组 episode，然后独立计算梯度
- **Meta-RL（LaMer）**：同一 trial 中的 episode 顺序生成，每个 episode 条件化于前面的 episode

### 损失函数

梯度估计：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\mathcal{T}} \left[ \sum_{n=0}^{N-1} \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(n)} | s_t^{(n)}, \mathcal{H}^{(n)}) A_t^{(n)} \right]$$

兼容 PPO、GRPO、GiGPO 等主流优化器。默认使用 GiGPO。

## 实验关键数据

### 主实验

基础模型为 Qwen3-4B，N=3 episodes，group size=8（RL 对应 group size=24 保证公平）。

| 方法 | Sokoban p@1/p@2/p@3 | MineSweeper p@1/p@2/p@3 | Webshop p@1/p@2/p@3 |
|------|---------------------|--------------------------|---------------------|
| Zero-shot | 6.8/9.8/12.9 | 4.5/6.6/8.6 | 1.4/2.1/2.3 |
| ReAct | 7.2/9.6/12.5 | 6.3/7.0/10.9 | 3.1/4.5/4.5 |
| Reflexion | 6.4/9.8/12.1 | 5.5/7.2/9.8 | 2.7/3.3/3.5 |
| PPO | 12.5/15.4/16.8 | 29.7/34.2/35.5 | 53.1/54.5/54.9 |
| GiGPO | 41.6/43.6/44.1 | 52.0/54.9/55.1 | 73.4/74.6/75.2 |
| **LaMer** | **42.4/52.0/55.9** | **44.1/66.4/74.4** | **67.8/84.4/89.1** |

LaMer 在 p@3 上全面超越所有基线：Sokoban +11.8%、MineSweeper +19.3%、Webshop +13.9%。

### OOD 泛化实验（ALFWorld）

| 方法 | Pick(i.d.) | Look(i.d.) | Clean(i.d.) | Heat(i.d.) | Cool(o.o.d.) | Pick2(o.o.d.) |
|------|-----------|-----------|------------|-----------|-------------|--------------|
| Prompting | 91.9 | 52.9 | 48.4 | 44.8 | 42.8 | 21.2 |
| RL | 95.5 | 83.0 | 67.9 | 86.6 | 58.1 | 36.0 |
| **Meta-RL** | **97.7** | **100.0** | **90.2** | **89.5** | **81.0** | **50.2** |

在 OOD 任务上，LaMer 比 RL 高出 23%（Cool）和 14%（Pick2）。

### 消融实验

**记忆配置消融**（p@3）：

| 记忆内容 | Sokoban | MineSweeper | Webshop |
|---------|---------|-------------|---------|
| 仅轨迹 | 34.8 | 69.5 | 89.3 |
| 仅反思 | **56.4** | **80.5** | **92.8** |
| 两者兼有 | 55.9 | 74.4 | 89.1 |

反思提供显著收益；仅反思甚至优于默认设置（反思更简洁聚焦）。

**$\gamma_{\text{traj}}$ 影响**：
- Sokoban/Webshop 最优 $\gamma_{\text{traj}}=0.6$（需要平衡即时与长期回报）
- MineSweeper 最优 $\gamma_{\text{traj}}=0.9$（需要更多战略探索）

### 关键发现

1. Meta-RL 保留了更高的轨迹多样性（通过经验分布的熵衡量），实现了更好的探索-利用权衡
2. 在更难任务上（更多箱子/地雷），Meta-RL 始终以 5-10% 的差距领先 RL
3. 测试时 scaling 效果更好：LaMer 从 p@1 到 p@3 的提升远大于 RL（Sokoban: 13.5% vs <5%）

## 亮点与洞察

1. **首次将 Meta-RL 引入 LLM Agent 训练**：将经典 Meta-RL 的跨任务泛化思想适配到 LLM 的多 episode 交互中
2. **优雅的形式化**：$\gamma_{\text{traj}}$ 提供了简洁的探索-利用控制旋钮
3. **自我反思的双重角色**：既是适应机制也是训练信号，消融证实其关键作用
4. **测试时 scaling 的新视角**：Meta-RL 可视为通过训练时多 episode 来摊销测试时计算
5. **无需额外训练数据**：与 RL 使用相同数量的轨迹，只是改变了轨迹的组织方式

## 局限性

1. **训练时间约为 RL 的 2 倍**：trial 内的 episode 必须顺序生成，并行度受限
2. **仅验证了一个基础模型**（Qwen3-4B）：在更大模型上的效果待验证
3. **环境类型有限**：主要是文本格式的游戏/网页环境，真实世界的复杂 agent 任务有待探索
4. **context 长度限制**：多 episode 的历史和反思会快速填满上下文窗口

## 相关工作与启发

- **Reflexion**（Shinn et al., 2023）：使用多 episode + 反思，但冻结 LLM 无训练
- **GiGPO**（Feng et al., 2025）：当前最强单 episode RL 基线，LaMer 在此基础上拓展为多 episode
- **Test-time compute scaling**：LaMer 提供了一种通过训练来改善测试时 scaling 的方法
- **启发**：该框架可与更强的推理模型（如 R1 系列）结合，探索 Reasoning + Exploration 的协同

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将 Meta-RL 适配到 LLM Agent，形式化简洁
- **技术深度**: ⭐⭐⭐⭐ — 跨 episode 奖励传播机制设计成熟，理论分析清晰
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 个环境 + OOD 泛化 + 难度泛化 + 详细消融
- **实用价值**: ⭐⭐⭐⭐ — 框架通用，兼容主流 RL 算法
- **总体推荐**: ⭐⭐⭐⭐ — 扎实的工作，为 LLM Agent 的探索能力训练开辟了新方向

<!-- RELATED:START -->

## 相关论文

- [Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence](../../ICML2025/llm_nlp/beyond_induction_heads_in-context_meta_learning_induces_multi-phase_circuit_emer.md)
- [Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance_estimation_f.md)
- [Enhancing Persona Following at Decoding Time via Dynamic Importance-Guided Token Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance-guided_token.md)
- [System Prompt Optimization with Meta-Learning](../../NeurIPS2025/llm_nlp/system_prompt_optimization_with_meta-learning.md)
- [Q♯: Provably Optimal Distributional RL for LLM Post-Training](../../NeurIPS2025/llm_nlp/qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)

<!-- RELATED:END -->
