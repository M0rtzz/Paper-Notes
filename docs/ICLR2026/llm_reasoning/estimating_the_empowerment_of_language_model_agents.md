---
title: >-
  [论文解读] Estimating the Empowerment of Language Model Agents
description: >-
  [ICLR 2026][LLM推理][empowerment] 提出 EELMA 算法，利用信息论中的"赋权"（empowerment，即 agent 动作与未来状态的互信息）作为目标无关的 LM Agent 能力度量指标，在语言游戏和真实网页浏览场景中与任务表现强相关（$r=0.83$–$0.94$），可用于开放式 agent 监控与安全评估。
tags:
  - ICLR 2026
  - LLM推理
  - empowerment
  - information theory
  - mutual information
  - LM agents
  - goal-agnostic evaluation
  - InfoNCE
  - WebArena
---

# Estimating the Empowerment of Language Model Agents

**会议**: ICLR 2026  
**arXiv**: [2509.22504](https://arxiv.org/abs/2509.22504)  
**代码**: [GitHub](https://github.com/Jinyeop3110/EELMA)  
**领域**: llm_reasoning  
**关键词**: empowerment, information theory, mutual information, LM agents, goal-agnostic evaluation, InfoNCE, WebArena

## 一句话总结

提出 EELMA 算法，利用信息论中的"赋权"（empowerment，即 agent 动作与未来状态的互信息）作为目标无关的 LM Agent 能力度量指标，在语言游戏和真实网页浏览场景中与任务表现强相关（$r=0.83$–$0.94$），可用于开放式 agent 监控与安全评估。

## 研究背景与动机

- **传统评估的局限性**：当前 LM Agent 评估主要依赖目标导向基准（goal-centric benchmarks），需要人工设计大量任务，成本高且无法检测基准范围之外的能力增长，对 AI 安全存在盲区
- **开放式环境的挑战**：随着 LM Agent 能够调用搜索引擎、API、操作系统等工具进行长时间多轮交互，传统的里程碑式评估方法无法捕捉 agent 在开放环境中的真实能力
- **赋权（Empowerment）的启发**：信息论中的赋权度量了 agent 对未来状态的影响力，理论上与任意随机目标下的期望回报存在下界关系，天然适合作为目标无关的能力指标
- **技术瓶颈**：经典赋权估计方法计算开销大，无法在高维文本空间中直接应用，需要新的可扩展算法

## 方法详解

### 整体框架

EELMA（Estimating Empowerment of Language Model Agents）基于标准 MDP 框架 $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$，将 LM Agent 的文本交互建模为状态-动作序列，通过变分互信息估计来量化赋权。

### 关键设计

**1. 有效赋权定义**

引入未来状态随机变量 $s_*$（采样步数 $\tau \sim \text{Geom}(1-\gamma)$），有效赋权定义为动作与未来状态间的平均互信息：

$$\mathcal{E}(\pi_{LM}) \triangleq \mathbb{E}_{s_t, a_t, s_*}\left[\sum_{t=0}^{\infty} \frac{\gamma^t}{1-\gamma} \log \frac{P(s_{t+\tau}=s_* \mid s_t, a_t)}{P(s_{t+\tau}=s_* \mid s_t)}\right]$$

进一步定义了状态条件赋权 $\mathcal{E}(s, \pi_{LM})$ 和状态-动作条件赋权 $\mathcal{E}(s, a, \pi_{LM})$，用于识别高影响力的状态和动作。

**2. 文本嵌入与投影**

从多轮轨迹 $\{(s_t^i, a_t^i)\}_{t=1}^{T_i}$ 中采样元组 $(s_t^i, a_t^i, s_*^i)$，使用预训练嵌入模型（如 Jina Embeddings）配合可微 MLP（参数 $\theta$）将文本映射到紧凑嵌入 $(z_{s,t}^i, z_{a,t}^i, z_{s_*,t}^i)$。

**3. InfoNCE 互信息估计**

使用两个神经编码器 $\phi$（编码当前状态/动作）和 $\psi$（编码未来状态），基于对比学习的 InfoNCE 损失进行变分互信息估计：

$$I_{\text{NCE}}^{\text{State-action}} \geq \mathbb{E}\left[\log \frac{e^{\phi(z_{s,t}^i, z_{a,t}^i)^\top \psi(z_{s,*}^i)}}{\frac{1}{K}\sum_j e^{\phi(z_{s,t}^i, z_{a,t}^i)^\top \psi(z_{s,*}^j)}}\right]$$

负样本来自不同轨迹的目标状态。同时计算仅状态条件的 $I_{\text{NCE}}^{\text{State-only}}$。

**4. 赋权估计公式**

利用学到的表征，有效赋权通过两个点积之差估计：

$$\mathcal{E}(\pi_{LM}) = \mathbb{E}_{i,t,s^*}\left[\phi(z_{s,t}^i, z_{a,t}^i)^\top \psi(z_{s,*}^i) - \phi(z_{s,t}^i)^\top \psi(z_{s,*}^i)\right]$$

### 损失函数

联合最大化两个 NCE 目标（状态-动作版和仅状态版），同时优化编码器 $\phi, \psi$ 和嵌入投影 $\theta$。

### 理论基础

赋权与 agent 能力的关系有理论保证：在均匀奖励假设下，赋权构成平均折扣回报 $\bar{r} = \mathbb{E}_R[\sum_{t=0}^{\infty} \gamma^t r_t]$ 的下界。高赋权意味着 agent 在多轮交互中保持了更多未来选择权，能在任意任务上表现更好。

## 实验关键数据

### 主实验

**语言游戏验证（Gridworld + Tower of Hanoi）**

| 环境 | 方法 | State RMSE (bits) |
|------|------|-------------------|
| Gridworld | EELMA (固定格式) | 0.056 |
| Gridworld | 直接估计 (NL) | 0.302 |
| Gridworld | EELMA (NL) | 0.048 |
| Tower of Hanoi | EELMA (固定格式) | 0.158 |
| Tower of Hanoi | 直接估计 (NL) | 0.438 |
| Tower of Hanoi | EELMA (NL) | 0.127 |

EELMA 在自然语言变体下仍保持鲁棒性，RMSE 甚至低于固定格式时的直接估计。

**WebArena 真实网页浏览**

| 领域 | 赋权-回报相关性 ($R_s$) |
|------|--------------------------|
| GitLab | 0.94 |
| Reddit | 0.83 |
| Shopping Admin | 0.87 |
| Shopping | 弱相关（推理瓶颈） |

GPT-4o 赋权最高、折扣回报最高；o3 成功率与 GPT-4o 相当但步数更多导致折扣回报较低。

### 消融实验

**Agent 子系统对赋权的影响**

| 消融因素 | 赋权变化 |
|----------|----------|
| 移除 CoT | Gridworld 下降 99%（0.19→0.01 bits），ToH 下降 65%（0.29→0.09 bits） |
| 记忆长度 m0→m3 | ToH 赋权从约 0.3 升至 0.4 bits |
| 模型规模 | 闭源模型 > 开源模型；大模型 > 小模型 |
| 环境复杂度 | 4→7 个盒子时赋权单调下降 |

### 关键发现

**认证行为案例研究**

| 动作类型 | 平均赋权 (bits) | 显著性 |
|----------|-----------------|--------|
| 有效密码输入 | 0.210 | p < 0.001 |
| 无效密码输入 | -0.152 | — |
| 有效用户名输入 | 0.170 | p = 0.32（不显著） |
| 总体有效认证 | 0.365 | p < 0.001 |
| 总体无效认证 | -0.127 | — |

成功认证后赋权急剧上升，体现了 agent 获取系统管理权限的"权力扩张"行为。密码输入比用户名输入更关键——因为即使用户名正确，配合错误密码也无法获得未来状态的可达性提升。

## 亮点与洞察

1. **目标无关的能力度量**：赋权是首个不需要目标标注的 LM Agent 通用能力指标，与多种环境下的任务表现高度相关
2. **安全监控价值**：高赋权动作对应关键时刻（如获取认证），可用于检测潜在的权力扩张行为，无需预先枚举危险行为列表
3. **CoT 的定量价值**：首次用信息论方式量化 CoT 的效果——移除 CoT 后赋权下降 99%，提供了 agent 推理能力的理论度量
4. **语言鲁棒性**：EELMA 在自然语言变体下比直接估计更准确，这对现实部署至关重要
5. **理论-实验一致性**：赋权的理论下界关系在从玩具到真实的多种场景中均得到实验支持

## 局限性

1. **赋权不等于权力**：选项更多不一定意味着更强大（类比"一个好 offer 胜过多个差 offer"），且无法捕捉间接影响力（如对其他 agent 的信念和决策的影响）
2. **Shopping 域弱相关**：当瓶颈不在环境控制而在数值推理时，赋权指标失效
3. **计算开销**：需要多轮轨迹收集和嵌入训练，规模化到更复杂的开放环境仍需探索
4. **仅限文本环境**：虽讨论了多模态扩展可能性，但当前仅在文本交互中验证

## 相关工作与启发

- **与 benchmark 评估的互补**：EELMA 不替代而是补充传统基准评估，特别适合发现基准未覆盖的能力增长
- **与 RL 内在激励的区别**：先前工作用互信息作为训练信号（intrinsic reward），本文首次用于评估 LM Agent 而非训练
- **与 AI 安全的连接**：Turner 等人的"权力寻求"理论预测最优策略趋向寻求权力，EELMA 提供了可操作的检测工具
- **对 agent 设计的启发**：赋权分析揭示了 CoT、记忆长度、模型规模对 agent 能力的量化影响，可指导 agent 架构设计

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将信息论赋权概念迁移到 LM Agent 评估，方法和视角均为原创
- **实验充分度**: ⭐⭐⭐⭐ 从受控玩具环境（有真值验证）到真实 WebArena 场景，消融全面
- **实用价值**: ⭐⭐⭐⭐ 为 agent 安全监控和能力评估提供了新范式，但部署开销需优化
- **写作质量**: ⭐⭐⭐⭐⭐ 理论动机清晰，图表丰富，案例研究（认证行为）生动说服力强
- **总评**: ⭐⭐⭐⭐½ 高质量的跨领域创新工作，将信息论与 LM Agent 评估巧妙结合

<!-- RELATED:START -->

## 相关论文

- [Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [Why is Your Language Model a Poor Implicit Reward Model?](why_is_your_language_model_a_poor_implicit_reward_model.md)
- [Agentified Assessment of Logical Reasoning Agents](agentified_assessment_of_logical_reasoning_agents.md)
- [Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)
- [Improve Vision Language Model Chain-of-thought Reasoning](../../ACL2025/llm_reasoning/improve_vlm_cot_reasoning.md)

<!-- RELATED:END -->
