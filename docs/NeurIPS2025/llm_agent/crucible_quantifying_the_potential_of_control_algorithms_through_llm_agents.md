---
title: >-
  [论文解读] Crucible: Quantifying the Potential of Control Algorithms through LLM Agents
description: >-
  [NeurIPS 2025][LLM Agent][tuning potential] 首次将"调优潜能"（Tuning Potential）概念形式化，通过 LLM Agent 模拟多级开发者对控制算法进行参数+逻辑双层调优，在 CartPole 上 Bang-bang 从 34→500 达到 DQN 水平，ABR 任务上相比贝叶斯优化最高提升 44.1%。
tags:
  - NeurIPS 2025
  - LLM Agent
  - tuning potential
  - LLM agent
  - control algorithm
  - parameter optimization
  - Bayesian optimization
---

# Crucible: Quantifying the Potential of Control Algorithms through LLM Agents

**会议**: NeurIPS 2025  
**arXiv**: [2510.18491](https://arxiv.org/abs/2510.18491)  
**代码**: https://github.com/thu-media/Crucible (有)  
**领域**: LLM Agent / 控制算法评估  
**关键词**: tuning potential, LLM agent, control algorithm, parameter optimization, Bayesian optimization

## 一句话总结

首次将"调优潜能"（Tuning Potential）概念形式化，通过 LLM Agent 模拟多级开发者对控制算法进行参数+逻辑双层调优，在 CartPole 上 Bang-bang 从 34→500 达到 DQN 水平，ABR 任务上相比贝叶斯优化最高提升 44.1%。

## 研究背景与动机

**控制算法评估的盲区**。现有研究通常在默认参数或理想条件下评估算法性能，但在生产环境中算法总是由领域专家针对特定场景进行调参和逻辑改造。算法的实际价值不仅取决于其默认设计，更取决于其内在的可调性——即"调优潜能"（Tuning Potential）。然而目前缺乏系统化的方法来度量这一特性，导致算法选型和设计中忽视了这一关键维度。

**评估挑战超越传统参数敏感性分析**。调优潜能的评估不能仅停留在超参搜索层面，还必须覆盖更深层的逻辑级修改——例如增加控制分支、集成新组件等。这些结构性调整高度依赖开发者对算法的主观理解，使得传统评估方法失效。关键矛盾在于如何同时捕捉客观性能指标和主观理解因素的交互作用。

**实际案例引出核心动机**。作者的预实验表明：ABR 场景中简单的 HYB 算法经调优后 QoE 从最差升至第一、视频卡顿时间降低 92%；调度场景中 FIFO 经调优后达到最优累积等待时间——充分说明"简单算法 + 充分调优"可能超越"复杂算法 + 默认配置"。Crucible 的核心 idea 是用 LLM 模拟不同能力水平的开发者进行算法调优，并建立形式化的潜能度量体系。

## 方法详解

### 整体框架

Crucible 由两大核心组件构成：(1) LLM 驱动的多级专家模拟 Agent，模拟不同能力水平的开发者对控制算法进行参数调优和逻辑重构；(2) 基于环境性能特征向量的统一调优潜能度量指标。系统工作流为：对每个测试环境执行 LLM 优化循环 → 收集性能差距最大的 case → LLM 给出优化建议 → 实施修改并可选地应用贝叶斯优化 → 遍历所有环境后进入评估阶段计算潜能。

### 关键设计

1. **多维度领域知识注入**:

    - 功能：为 LLM 构建完整的任务理解上下文
    - 核心思路：通过 system prompt 注入三维度知识——任务描述（输入状态 + 输出行为空间）、优化目标（改进方向 + 评估标准）、环境概述（场景特征 + 约束条件）
    - 设计动机：LLM 需要充分理解控制任务的上下文才能做出有效的逻辑级修改，三维度设计确保覆盖"做什么、优化什么、在哪里做"

2. **参数-逻辑双层优化 Agent**:

    - 功能：同时探索超参空间和算法逻辑空间，挖掘算法的完整调优空间
    - 核心思路：将贝叶斯优化封装为工具接口（评估参数空间内的性能上限），同时 LLM 进行逻辑级修改（增加控制分支、重构算法结构）；每次修改保存为三元组（修改理由、具体操作、观测结果），作为后续优化的经验基础
    - 设计动机：参数优化有天花板（受限于算法逻辑的表征能力），纯 LLM 逻辑修改不稳定（60% 场景无改进）；两者协同才能充分挖掘——贝叶斯优化在 LLM 打开的新解空间中做精细搜索

3. **差异化开发者能力模拟**:

    - 功能：模拟具有不同技能水平和资源预算的开发者
    - 核心思路：通过调整计算预算而非设计不同 prompt 来模拟能力差异——限制贝叶斯优化调用次数（0/10/20 次）和反思迭代步数（1/2/3 步）
    - 设计动机：基于一个关键洞察——专家和新手的核心差异不是"知道更多"而是"能投入更多资源去试错和精细调参"，资源预算是更贴近现实的能力代理指标

### 损失函数 / 训练策略

Crucible 不涉及传统的模型训练。调优潜能的形式化定义为：首先选取探测算法在所有评估环境上运行并归一化，得到每个环境的性能特征向量；两个环境的距离定义为特征向量的 RMSE，相似度为 $\text{sim}(E_i, E_t) = \max(0, 1 - \text{dis}(E_i, E_t))$；最终潜能 $\mathcal{P} = \frac{1}{|\mathcal{T}|} \sum_{E_t} [(S_{t,c} - S_{t,o}) \times \text{sim}(E_i, E_t)]$，即所有测试环境上相似度加权的性能增益均值。这种设计使得与理想环境差异大的环境上的增益被降权，保证度量的鲁棒性和公平性。

## 实验关键数据

### 主实验
| 数据集/任务 | 指标 | 本文(Crucible) | 之前SOTA(贝叶斯/默认) | 提升 |
|--------|------|------|----------|------|
| CartPole Bang-bang | 分数 | 500 | 34(默认)/56(贝叶斯) | 34→500，一次LLM逻辑修改 |
| CartPole PID | 分数 | 500 | 34(默认)/77(贝叶斯) | 两轮迭代达DQN最优 |
| ABR Puffer | QoE提升 | +44.1% vs 贝叶斯 | 贝叶斯基线 | 最高提升 |
| ABR 真实部署(Dash.js) | QoE | HYB/BBA=1.72 | Pensieve(RL)=1.66 | 调优后超越RL基线 |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Bayes=0, LLM迭代 | 60%场景无改进 | 纯LLM逻辑修改不稳定 |
| Bayes=20, LLM迭代 | 20%场景无改进 | 贝叶斯+LLM协同效果显著 |
| Claude 3.7 (HYB) | QoE=1.12 | 更强模型解锁更大潜能 |
| Claude 3.5 (HYB) | QoE=1.03 | 模型间结论一致 |
| GPT-4o-mini (HYB) | QoE=1.04 | 跨模型鲁棒 |

### 关键发现
- 算法的"表征能力"是潜能的核心因素：HYB（双状态输入）潜能 0.068 >> BBA（单状态）0.018
- 算法的"可理解性"同样关键：Pitree（决策树蒸馏自 RL）虽初始性能低但潜能也低（0.033），复杂逻辑阻碍 LLM 优化
- ABR 改进幅度远高于调度算法，因为调度 DAG 输入状态更复杂，LLM 理解难度更大
- 从潜能评估到算法优化：BBA 增加带宽输入变为 BBA_C，初始性能仅差 0.5%，但调优后提升 4%；SJF 经调优后反超初始更优的多级反馈算法

## 亮点与洞察
- 首创"调优潜能"作为算法评估新维度——不仅评估"现在有多好"，还评估"能被优化到多好"
- 参数级利用（贝叶斯）+ 逻辑级探索（LLM）的协同设计思想值得推广到其他 LLM-as-optimizer 场景
- 算法设计的启示：追求简洁可理解 + 宽状态空间的算法，而非复杂黑盒
- 通过资源预算（而非 prompt）模拟不同能力开发者的建模思路新颖

## 局限与展望
- LLM 版本对结果有影响，但作者将其解释为模拟不同水平开发者，逻辑上有些循环
- 无法直接修改黑盒算法内部逻辑，仅能处理决策树等可解释模型
- 纯 LLM 修改成功率仅 40%，对贝叶斯优化的依赖较重
- 标准化接口设计限制了 LLM 只能修改算法代码，不能修改环境设置或评估方式

## 相关工作与启发
- 与 LLM-based human simulation（如 Generative Agents）的联系：同样用 LLM 模拟人类行为，但聚焦于算法调优而非社会行为
- 与 Bayesian optimization 的互补：传统 BO 只搜参数空间，Crucible 同时搜索逻辑空间
- 对 AutoML/NAS 领域的启示：算法潜能可以作为搜索和评估的新目标函数

## 评分
- 新颖性: ⭐⭐⭐⭐ "调优潜能"概念新颖且形式化定义完整，开辟了算法评估的新维度
- 实验充分度: ⭐⭐⭐⭐ 经典控制+计算机系统+真实部署三层验证，跨LLM鲁棒性测试充分
- 写作质量: ⭐⭐⭐⭐ 动机实验有说服力，问题定义清晰，从动机到验证逻辑流畅
- 价值: ⭐⭐⭐⭐ 为算法设计和评估引入全新维度，对 LLM-as-optimizer 范式有的参考意义

<!-- RELATED:START -->

## 相关论文

- [Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models](../../ACL2025/llm_agent/llm_agent_image_classification.md)
- [Reflection-Driven Control for Trustworthy Code Agents](../../AAAI2026/llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)
- [Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery](../../ICML2025/llm_agent/open_source_planning_control_system_with_language_agents_for_autonomous_scientif.md)
- [A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [LLM Agents for Knowledge Discovery in Atomic Layer Processing](llm_agents_for_knowledge_discovery_in_atomic_layer_processing.md)

<!-- RELATED:END -->
