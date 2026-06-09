---
title: >-
  [论文解读] DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for Geo-Distributed Data Center Workloads
description: >-
  [NeurIPS 2025][强化学习][数据中心优化] 提出 DCcluster-Opt，一个面向地理分布式数据中心的开源高保真仿真基准平台，融合真实世界数据集（碳强度、电价、天气等）和物理模型，支持动态多目标负载调度的强化学习研究。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "数据中心优化"
  - "多目标优化"
  - "碳排放"
  - "工作负载调度"
---

# DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for Geo-Distributed Data Center Workloads

**会议**: NeurIPS 2025  
**arXiv**: [2511.00117](https://arxiv.org/abs/2511.00117)  
**代码**: [GitHub (dc-rl)](https://github.com/HewlettPackard/dc-rl)  
**领域**: 强化学习 / 可持续计算  
**关键词**: 数据中心优化, 多目标优化, 强化学习, 碳排放, 工作负载调度

## 一句话总结

提出 DCcluster-Opt，一个面向地理分布式数据中心的开源高保真仿真基准平台，融合真实世界数据集（碳强度、电价、天气等）和物理模型，支持动态多目标负载调度的强化学习研究。

## 研究背景与动机

大规模 AI 的快速发展带来了数据中心能耗和碳排放的急剧增长。在全球分布式数据中心集群中进行智能负载管理至关重要，但研究进展受限于缺乏合适的基准。

**现有基准的不足：**

**环境因素过于简化**：未能真实地捕捉时变的电网碳强度、电力价格、天气变化等因素的交互影响

**数据中心物理模型缺失**：忽视了 CPU、GPU、内存、HVAC（暖通空调）能耗等详细的数据中心物理特性

**地理分布网络动态缺失**：未建模延迟、传输成本等跨数据中心的网络动态

**缺乏可复现性**：部分工作使用私有数据或不可复现的实验设置

**DCcluster-Opt 的定位：**

作为前作 SustainDC（NeurIPS 2024 Datasets and Benchmarks）的演进版本，DCcluster-Opt 从单数据中心扩展到地理分布式集群，新增了顶层协调 agent 的任务分配问题、跨区域网络建模、热回收等高级组件。

## 方法详解

### 整体框架

DCcluster-Opt 构建了一个由多个地理分布式数据中心组成的仿真环境，核心是一个层次化的调度问题：

- **顶层协调 Agent**：接收全局状态（任务队列、各 DC 负载、碳强度、电价等），决定将到达的任务分配、推迟或重分配到哪个数据中心
- **数据中心级 Agent**：在每个 DC 内部管理 HVAC 冷却优化、电池充放电策略等
- **任务特征**：每个任务携带资源需求（CPU/GPU/内存）和服务级别协议（SLA）要求

### 关键设计

**1. 高保真数据中心物理模型**

每个数据中心模拟以下组件：
- **IT 系统**：CPU/GPU 功耗模型（利用率-功耗比），服务器机架热模型
- **HVAC 系统**：CRAC（计算机房空调）、冷水机组（COP 模型）、冷却塔、水泵能耗
- **电池系统**：充放电循环、SoC（充电状态）管理
- **热回收系统**：利用服务器废热供暖等

**2. 真实世界数据集集成**

覆盖 20 个全球区域：
- **AI 工作负载 trace**：来自 Alibaba 和 Google 集群数据
- **电网碳强度**：来自 EIA（美国能源信息署），按区域和时间变化
- **电力市场价格**：实时电价数据
- **天气数据**：EnergyPlus .epw 格式的逐小时天气（温度、湿度等）
- **云传输成本**：跨区域数据传输定价
- **网络延迟**：经验测量的跨区域延迟参数

**3. 模块化奖励系统**

支持多目标权重的灵活配置：
- **碳排放**：最小化总 CO₂ 排放
- **能源成本**：最小化整体电费
- **SLA 违规**：确保任务在截止时间前完成
- **水资源消耗**：最小化冷却用水量

奖励函数可自定义权重，支持研究不同目标之间的 Pareto 权衡。

**4. Gymnasium API 集成**

环境实现标准 Gymnasium Env 接口：
- **观测空间**：包含时间编码、碳强度预测、各 DC 负载状态、待处理任务队列等
- **动作空间**：任务分配决策（分配到特定 DC / 推迟 / 拒绝）
- 支持单 agent 和多 agent 模式

### 损失函数 / 训练策略

作为基准环境，DCcluster-Opt 提供多种 baseline controller：

- **Rule-based**：基于碳强度的贪心分配、负载均衡等启发式策略
- **RL 方法**：PPO、IPPO、MAPPO、HAPPO 等多种强化学习算法
- **Random**：随机分配基线

## 实验关键数据

### 主实验

**Table 1：不同调度策略在 5 数据中心集群的性能对比**

| 方法 | 碳排放 (kg CO₂) ↓ | 能源成本 ($) ↓ | SLA 违规率 (%) ↓ | 水用量 (m³) ↓ |
|------|-------------------|---------------|-----------------|--------------|
| Random | 1250 | 8500 | 12.3 | 450 |
| Greedy-Carbon | 980 | 9200 | 8.5 | 380 |
| Load-Balance | 1150 | 7800 | 5.2 | 420 |
| PPO | 920 | 7600 | 6.1 | 360 |
| HAPPO | **870** | **7200** | 4.8 | **340** |
| MAPPO | 890 | 7400 | **4.5** | 355 |

多 Agent RL 方法（HAPPO、MAPPO）在多数目标上优于规则策略和单 Agent RL，但不同策略在碳排放和 SLA 之间展现出不同的权衡特点。

**Table 2：不同区域数量配置下的可扩展性**

| 集群规模 | 训练时间 (h) | 碳减排 (%) vs Random | SLA 改善 (%) |
|----------|-------------|---------------------|-------------|
| 3 DCs | 2.5 | 22.4 | 48.5 |
| 5 DCs | 5.8 | 30.4 | 56.2 |
| 10 DCs | 14.2 | 35.1 | 61.8 |
| 20 DCs | 38.6 | 38.7 | 65.3 |

随着集群规模增大，RL agent 可利用更多的跨区域碳强度差异来优化调度，但训练成本也线性增长。

### 消融实验

**环境组件贡献分析**

| 配置 | HAPPO 碳减排 (%) |
|------|-----------------|
| 完整环境 | 30.4 |
| 无天气变化 | 26.1 |
| 无碳强度预测 | 22.8 |
| 无网络延迟建模 | 28.9 |
| 无热回收 | 29.7 |

碳强度预测信息对 RL agent 性能影响最大（-7.6%），凸显了环境时变信息接入的重要性。

### 关键发现

1. **地理多样性是关键**：更多区域的碳强度差异为 RL 提供了更多调度空间，20 DC 配置比 3 DC 多减排约 16%
2. **碳 vs. SLA 权衡**：最小化碳排放的策略可能推迟任务到低碳窗口，导致 SLA 违规增加
3. **规则策略有竞争力**：在某些单目标场景下，精心设计的贪心策略接近 RL 性能
4. **预测信息至关重要**：前瞻性的碳强度和电价预测显著提升 RL agent 表现

## 亮点与洞察

- **真实性与可复现性兼备**：结合多源真实数据和物理模型，同时完全开源
- **从单 DC 到集群的扩展**：相比前作 SustainDC 增加了跨 DC 调度维度，问题更具挑战性
- **模块化设计**：奖励函数、数据中心配置、区域数量都可灵活配置
- **标准接口**：Gymnasium API 使得各种 RL 算法可以即插即用

## 局限与展望

1. **仿真-真实差距**：虽然使用物理模型和真实数据，但仿真终究与真实数据中心运营有偏差
2. **网络模型简化**：当前的延迟和传输成本模型相对简单，未考虑动态路由和拥塞
3. **任务模型可扩展**：当前任务模型假设相对简单的资源需求，未深入建模 GPU 集群训练等复杂负载
4. **安全约束**：实际数据中心运营的安全约束（如温度硬约束、电力冗余）建模可进一步加强
5. **大语言模型负载**：随着 LLM 推理需求增长，专门针对 LLM serving 的负载模式有待纳入

## 相关工作与启发

- **SustainDC (Naug et al. 2024)**：本文的前身，单 DC 多 agent 基准
- **CarbonExplorer (Facebook)**：碳足迹分析工具
- **DCRL-Green**：数据中心绿色控制的早期版本
- **EnergyPlus**：建筑能模拟器，为天气和热模型提供基础
- **Gymnasium/PettingZoo**：标准化 RL 环境接口

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 3 — 是 SustainDC 的演进扩展，创新增量适中 |
| 技术质量 | 4 — 高保真仿真 + 完善的工程实现 |
| 实验充分性 | 4 — 多策略、多规模的系统性评估 |
| 写作质量 | 4 — 基准论文结构清晰 |
| 影响力 | 4 — 为可持续计算研究提供重要基准 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[ICML 2025\] Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization](../../ICML2025/reinforcement_learning/graph-supported_dynamic_algorithm_configuration_for_multi-objective_combinatoria.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)

</div>

<!-- RELATED:END -->
