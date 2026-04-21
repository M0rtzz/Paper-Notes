---
title: >-
  [论文解读] Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents
description: >-
  [ICLR 2026][LLM Agent][策略梯度] 提出 EMPG 框架，通过步级熵（uncertainty）动态调制策略梯度的幅度，解决长序列 LLM Agent 任务中稀疏奖励下的信用分配问题，在 WebShop、ALFWorld 和 Deep Search 三个基准上显著超越 GRPO 和 DAPO。
tags:
  - ICLR 2026
  - LLM Agent
  - 策略梯度
  - 熵调制
  - 长序列代理
  - 信用分配
  - 强化学习
---

# Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents

**会议**: ICLR 2026  
**arXiv**: [2509.09265](https://arxiv.org/abs/2509.09265)  
**代码**: [Project Page](https://empgseed-seed.github.io/)  
**领域**: llm_agent  
**关键词**: 策略梯度, 熵调制, 长序列代理, 信用分配, 强化学习  

## 一句话总结

提出 EMPG 框架，通过步级熵（uncertainty）动态调制策略梯度的幅度，解决长序列 LLM Agent 任务中稀疏奖励下的信用分配问题，在 WebShop、ALFWorld 和 Deep Search 三个基准上显著超越 GRPO 和 DAPO。

## 研究背景与动机

在长序列任务（web 导航、软件工程、深度搜索等）中，LLM Agent 面临的根本挑战是**稀疏奖励下的信用分配**：反馈仅在整个生成结束后才可获得，难以识别哪些中间步骤是关键的。

现有方法主要有两条路线：

**隐式奖励引导**：奖励塑形、内在动机（好奇心/新颖性）、逆强化学习等——难以扩展到 LLM 的巨大状态-动作空间

**显式步级监督**：过程奖励模型（PRM）——标注成本高、合成数据噪声大、泛化性差，且在交互式 Agent 任务中定义"正确步骤"本身就极具挑战

本文的核心观察是：**标准策略梯度的梯度幅度与策略熵本质耦合**。具体来说，对 softmax 策略，score function 的期望范数是策略 Renyi-2 熵的单调函数（Proposition 1）。这带来双重问题：
- **自信且正确的步骤**本应被强烈强化，但其自然梯度很小，限制了学习速度
- **不确定的探索步骤**产生大梯度，引入噪声并破坏训练稳定性

## 方法详解

### 整体框架

EMPG（Entropy-Modulated Policy Gradients）是一个对策略梯度进行双重重新校准的框架，包含两个互补组件：

1. **自校准梯度缩放（Self-Calibrating Gradient Scaling）**：根据当前步不确定性重新校准梯度幅度
2. **未来清晰度奖励（Future Clarity Bonus）**：鼓励 Agent 寻找可预测的解决方案路径

### 关键设计

**步级不确定性量化**：对一个"思考-行动"步骤内的 m 个 token，计算平均 token 级熵作为步级熵 H_t。

**调制优势函数**：对轨迹中的步骤 t，定义调制后的优势估计：A_mod(i,t) = A^(i) * g(H_t^(i)) + zeta * f(H_{t+1}^(i))，其中第一项为自校准梯度缩放，第二项为未来清晰度奖励。

**自校准缩放函数** g(H) 使用指数函数并在 mini-batch 内归一化（均值约束为 1）。调制行为：
- 自信且正确的步骤（H_t 低）：g > 1，放大梯度
- 不确定的步骤（H_t 高）：g < 1，衰减梯度
- 自信但错误的步骤（A < 0, H_t 低）：强惩罚信号

**未来清晰度奖励** f(H) 鼓励选择导致下一步低熵状态的动作。

### 损失函数 / 训练策略

EMPG 作为优势调制模块直接应用于 GRPO 或 DAPO 等基线之上。Agent 使用 ReAct 范式（先生成思考再产生动作），整个框架不需要额外的价值模型，属于 value-free 范式。

归一化包含两步：
1. **批次级熵归一化**：对步级熵做 min-max 缩放
2. **最终优势归一化**：计算 A_mod 后做零均值归一化（方差减小）

## 实验关键数据

### 主实验

**ALFWorld 和 WebShop**（表1，平均成功率 %）：

| 方法 | 基座模型 | ALFWorld All | WebShop Succ. |
|------|---------|-------------|---------------|
| GRPO | Qwen2.5-1.5B | 65.6 | 58.2 |
| + EMPG | Qwen2.5-1.5B | **73.7 (+8.1)** | **60.8 (+2.6)** |
| DAPO | Qwen2.5-1.5B | 80.8 | 73.2 |
| + EMPG | Qwen2.5-1.5B | **88.1 (+7.3)** | **73.8 (+0.6)** |
| GRPO | Qwen2.5-7B | 74.8 | 65.6 |
| + EMPG | Qwen2.5-7B | **78.5 (+3.7)** | **69.3 (+3.7)** |
| DAPO | Qwen2.5-7B | 90.0 | 79.6 |
| + EMPG | Qwen2.5-7B | **91.6 (+1.6)** | **82.7 (+3.1)** |

**Deep Search**（表2，Qwen2.5-32B-Instruct）：

| 方法 | ID Avg. | OOD Avg. | Overall |
|------|---------|----------|---------|
| DAPO | 63.5 | 59.8 | 62.0 |
| + EMPG | **66.6 (+3.1)** | **63.7 (+3.9)** | **65.3 (+3.3)** |

注意 EMPG 在 OOD 上的提升（+3.9）大于 ID（+3.1），说明泛化能力增强。

### 消融实验

在 Deep Search（Qwen2.5-32B）上拆解两个组件：

| 变体 | ID Avg. | OOD Avg. | Overall |
|------|---------|----------|---------|
| DAPO 基线 | 63.5 | 59.8 | 62.0 |
| + Gradient Scaling only | 63.7 | 63.7 (+3.9) | 63.7 |
| + Future Bonus only | 66.1 (+2.6) | 61.4 | 64.2 |
| + EMPG (full) | **66.6** | **63.7** | **65.3** |

### 关键发现

1. **两个组件互补**：Future Clarity Bonus 主要提升域内性能（exploitation），Gradient Scaling 主要提升域外泛化（regularization）
2. **训练稳定性**：DAPO 基线在约 240 步后 KL Loss 剧烈波动（policy collapse），EMPG 全程保持稳定
3. **步级 vs token 级**：与 token 级分析不同，即使低熵步骤也会经历显著的熵变化，验证了步级分析的必要性
4. **突破性能瓶颈**：基线在 ALFWorld 和 WebShop 上到达性能平台后停滞，EMPG 能突破这个上限持续提升

## 亮点与洞察

1. **理论洞察深刻**：首次形式化证明策略梯度幅度与策略熵的固有耦合问题（Proposition 1），从梯度动力学角度揭示了长序列 RL 学习效率低下的根本原因
2. **即插即用**：作为优势调制模块，EMPG 可以直接叠加在 GRPO、DAPO 等任何策略梯度方法之上
3. **不需要额外模型**：利用 Agent 自身的策略熵作为内在信号，不需要额外的价值模型或过程奖励模型
4. **双重校准设计精妙**：梯度缩放处理"学多少"，未来清晰度处理"往哪探索"
5. **跨任务跨规模一致有效**：从 1.5B 到 32B 模型，从 web 导航到深度搜索均稳定提升

## 局限与展望

1. **熵估计的粗糙性**：使用平均 token 级熵作为步级不确定性的代理，可能忽略了步内不同 token 的重要性差异
2. **超参数敏感性**：缩放因子 k、k' 和 zeta 的选择需要调优，论文未充分讨论敏感性
3. **任务类型有限**：仅在 web 导航、文本环境交互和搜索任务上验证，缺少代码生成、数学推理等其他长序列任务
4. **与 PRM 的结合**：EMPG 和过程奖励模型是正交的，未来可以探索两者结合
5. **多智能体场景**：论文提到但未验证在多 Agent 协作中的效果

## 相关工作与启发

- **GRPO** [Shao et al.]：通过组内 Z-score 估计优势——EMPG 在此基础上进一步细化到步级
- **DAPO** [Yu et al.]：自适应数据策展——EMPG 提供正交的梯度层面改进
- **SEED-GRPO** [Chen et al.]：用语义不确定性调制响应级优势——限于单轮推理
- **EDGE-GRPO** [Wang et al.]：在数学推理中做熵调制——限于单轮，未解决多步信用分配
- **ReAct** [Yao et al.]：思考-行动范式——EMPG 将每个 ReAct 周期视为一个决策步骤

## 评分

| 维度 | 评分 |
|------|------|
| 理论深度 | ⭐⭐⭐⭐⭐ |
| 新颖性 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 总体评价 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)
- [Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](../../CVPR2026/llm_agent/carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](mc-search_evaluating_and_enhancing_multimodal_agentic_search.md)

<!-- RELATED:END -->
