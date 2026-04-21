---
title: >-
  [论文解读] SR-Scientist: Scientific Equation Discovery With Agentic AI
description: >-
  [ICLR 2026][LLM Agent][symbolic regression] 提出 SR-Scientist 框架，将 LLM 从简单的方程提议者提升为自主 AI 科学家，通过代码解释器工具进行数据分析和方程评估，在长时程交互中自主发现科学方程，并结合强化学习进一步提升能力。
tags:
  - ICLR 2026
  - LLM Agent
  - symbolic regression
  - agentic AI
  - equation discovery
  - reinforcement-learning
  - scientific discovery
---

# SR-Scientist: Scientific Equation Discovery With Agentic AI

**会议**: ICLR 2026  
**arXiv**: [2510.11661](https://arxiv.org/abs/2510.11661)  
**代码**: [GitHub](https://github.com/GAIR-NLP/SR-Scientist)  
**领域**: llm_agent  
**关键词**: symbolic regression, agentic AI, equation discovery, reinforcement-learning, scientific discovery

## 一句话总结

提出 SR-Scientist 框架，将 LLM 从简单的方程提议者提升为自主 AI 科学家，通过代码解释器工具进行数据分析和方程评估，在长时程交互中自主发现科学方程，并结合强化学习进一步提升能力。

## 研究背景与动机

符号回归（Symbolic Regression, SR）旨在从观测数据中发现可解释的数学表达式，是科学发现的基础任务。传统方法主要分为三类：
- **遗传编程（GP）方法**：如 PySR、GPLearn，使用表达式树进行组合搜索
- **深度学习方法**：如 E2E、NeSymReS、DSR，通过神经网络学习从数值到表达式的映射
- **LLM 增强方法**：如 LLM-SR、LaSR，将 LLM 嵌入 GP 算法作为方程提议器

然而，现有 LLM 方法的局限性在于：
1. LLM 仅作为固定流水线中的方程生成器，**缺乏自主性**
2. 无法通过工具直接分析观测数据获取洞察
3. 大多数工作仅关注推理阶段，未探索通过 RL 等方法让模型**自我进化**

本文的核心动机是：构建以 Agentic AI 为核心的科学发现框架，让 LLM 不再是被动工具，而是能驱动整个发现生命周期的自主 Agent。

## 方法详解

### 整体框架

SR-Scientist 的推理框架采用迭代式设计（Algorithm 1）：

1. **每次迭代**设定一个精度目标 $G_i$（基于 MAPE）
2. LLM Agent 在 ReAct 框架下，交替进行推理与工具调用：$(r_1, \mathcal{T}_1, o_1), (r_2, \mathcal{T}_2, o_2), \ldots$
3. 通过经验缓冲区（Experience Buffer）跨迭代传递最优方程
4. 当达到停止条件时，提交最佳方程

目标函数使用 MAPE（平均绝对百分比误差）：

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - f(\mathbf{x}_i)}{y_i} \right|$$

### 关键设计

**工具设计**：将代码解释器封装为两个核心工具：
- **数据分析器 $T_1$**：链接到观测数据，Agent 可编写代码进行统计分析、残差分析等多种数据探索
- **方程评估器 $T_2$**：接受含常数占位符的方程骨架，内部用 BFGS 算法优化常数并报告性能

**经验缓冲区**：维护 $E = \{(e_i, s_i)\}_{i=1}^{N}$ 记录已探索的方程及其 MAPE 分数。每次迭代开始时，从缓冲区取出最优 $K$ 个方程作为上下文示例。这一机制巧妙地绕过了 LLM 上下文长度的限制。

**长时程优化**：每次迭代允许 Agent 进行最多 $M=25$ 轮交互（超过 20 轮），使其有充分时间分析数据和优化方程。

### 损失函数 / 强化学习

训练框架采用 GRPO 算法，奖励函数为对数线性映射：

$$\mathcal{R} = \text{clip}\left(\frac{\lg s_{\max} - \lg s}{\lg s_{\max} - \lg s_{\text{goal}}}, 0, 1\right)$$

其中 $s$ 为最佳方程的 MAPE，$s_{\max}=100\%$，$s_{\text{goal}}=0.1\%$。该连续奖励设计避免了二值奖励的稀疏性问题。训练数据通过混合规则与模型的合成策略构建，覆盖材料科学、化学、生物学、物理学四个领域。

## 实验关键数据

### 主实验

在 LSR-Synth 基准（129 个问题，4 个学科）上的精度结果：

| 方法 | Overall Acc₀.₀₁ | Overall Acc₀.₀₀₁ | 材料科学 Acc₀.₀₁ | 化学 Acc₀.₀₁ | 生物学 Acc₀.₀₁ | 物理学 Acc₀.₀₁ |
|------|-----------------|-------------------|------------------|-------------|---------------|---------------|
| PySR | 29.46 | 14.47 | 53.33 | 25.93 | 16.67 | 25.76 |
| LLM-SR (Qwen-480B) | 41.08 | 18.09 | 80.00 | 36.11 | 30.56 | 28.79 |
| SR-Scientist (GPT-120B) | **63.57** | **49.35** | 74.67 | **81.48** | **66.67** | 40.91 |
| SR-Scientist (GLM) | 48.32 | 25.06 | 81.33 | 45.37 | 40.28 | 36.37 |
| SR-Scientist (Qwen-480B) | 49.09 | 24.55 | **86.67** | 40.74 | 50.00 | 34.09 |
| SR-Scientist (30B) | 32.30 | 16.02 | 81.33 | 22.22 | 22.22 | 18.18 |
| SR-Scientist (30B+RL) | 40.92 | 20.69 | 85.33 | 37.38 | 29.17 | 25.00 |

**核心发现**：SR-Scientist 在四个模型中均超越基线 6%~35%，GPT-OSS-120B 作为骨干时达到最高性能。RL 训练在所有学科上均带来显著提升。

### 消融实验

| 方法 | Acc₀.₀₁ | Acc₀.₀₀₁ |
|------|---------|----------|
| SR-Scientist (GPT) | 63.57 | 49.35 |
| w/o 数据分析器 $T_1$ | 35.66 | 16.28 |
| w/o 经验缓冲区 | 57.36 | 41.86 |
| w/o top-k（随机采样） | 58.14 | 41.86 |

消融分析表明：
- 数据分析工具对 GPT 模型影响最大（下降约 28 个百分点）
- 经验缓冲区对 Qwen 模型影响最大（下降 13.4 个百分点）
- top-k 采样策略优于随机采样

### 关键发现

1. **符号准确率**：SR-Scientist 在完全恢复 ground truth 方程上表现最佳（SA=7.75~8.00），高于 PySR（4.65）和 LLM-SR（5.43）
2. **噪声鲁棒性**：在添加不同标准差的高斯噪声后，SR-Scientist 一致优于其他方法
3. **OOD 泛化**：发现的方程在域外测试数据上仍保持最佳性能
4. **最优交互轮数**：25 轮为最优值，过短（10 轮）不足以深入探索，过长则效益递减
5. **工具使用行为差异**：GPT 系列倾向直接编写残差分析代码，Qwen/GLM 系列更多使用数据统计

## 亮点与洞察

1. **范式转变**：将 LLM 从被动的方程提议者转变为自主的 AI 科学家，这是科学发现领域的重要思路转变
2. **经验缓冲区设计精妙**：用简单的堆结构解决了 LLM 上下文长度限制问题，同时实现了跨迭代的知识传递
3. **连续奖励设计**：利用方程性能可连续度量的特点，设计对数线性奖励避免稀疏性，这比数学/代码任务的二值奖励更加适配
4. **最小人工流水线原则**：Agent 自由决定工作流程，不同模型展现出不同的分析策略（如 GPT 偏好残差分析，Qwen 偏好统计分析）
5. **RL 自我进化有效**：30B 小模型通过 RL 训练后性能接近非 RL 的大模型，验证了 Agent 自我提升的可行性

## 局限性

1. 仅使用文本模型，未利用多模态输入（如图表分析）
2. 噪声场景下仍存在显著性能下降
3. Agent 可能在不同迭代中重复探索已知差劲的方程，记忆系统有优化空间
4. 评估集虽经防记忆设计，但 LSR-Synth 仍为合成数据，与真实科学发现场景存在差距

## 相关工作与启发

- 与 FunSearch（Romera-Paredes et al., 2024）和 AlphaEvolve 等工作相比，SR-Scientist 更强调 Agent 的自主性和长时程交互
- 经验缓冲区+GRPO 的组合为科学发现类 Agent 的 RL 训练提供了范例
- 框架的模块化设计（工具可插拔、骨干模型可替换）具有很好的扩展性，可推广到其他科学发现任务

## 评分

- **创新性**: ⭐⭐⭐⭐ — 将 Agentic AI 范式引入符号回归，结合 RL 自我进化是重要贡献
- **实验充分性**: ⭐⭐⭐⭐⭐ — 4 个学科、5 个骨干模型、精度/泛化/噪声鲁棒/符号准确率全面评估
- **实用性**: ⭐⭐⭐⭐ — 代码开源，框架模块化，但依赖大量 LLM 调用成本较高
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，算法描述规范，但部分内容可更精简
- **综合评分**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

## 相关论文

- [NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics](../../ICML2025/llm_agent/evaluating_retrieval-augmented_generation_agents_for_autonomous_scientific_disco.md)
- [The Controllability Trap: A Governance Framework for Military AI Agents](the_controllability_trap_a_governance_framework_for_military_ai_systems.md)
- [Toward a Dynamic Stackelberg Game-Theoretic Framework for Agentic AI Defense Against LLM Jailbreaking](toward_a_dynamic_stackelberg_game-theoretic_framework_for_agentic_ai_defense_aga.md)
- [Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery](../../ICML2025/llm_agent/open_source_planning_control_system_with_language_agents_for_autonomous_scientif.md)

<!-- RELATED:END -->
