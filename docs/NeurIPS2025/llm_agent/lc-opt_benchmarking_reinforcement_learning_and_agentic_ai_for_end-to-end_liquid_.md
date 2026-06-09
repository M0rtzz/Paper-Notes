---
title: >-
  [论文解读] LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers
description: >-
  [NeurIPS 2025][LLM Agent][液冷优化] 提出 LC-Opt，一个基于 Oak Ridge 国家实验室 Frontier 超级计算机冷却系统高保真数字孪生的液冷基准环境，支持强化学习控制策略的端到端液冷优化…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "液冷优化"
  - "强化学习"
  - "数字孪生"
  - "多智能体RL"
  - "可持续数据中心"
---

# LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers

**会议**: NeurIPS 2025

**arXiv**: [2511.00116](https://arxiv.org/abs/2511.00116)

**代码**: 无（基于 Modelica 的仿真环境）

**领域**: 强化学习 / 智能体AI / 数据中心优化

**关键词**: 液冷优化, 强化学习, 数字孪生, 多智能体RL, 可持续数据中心

## 一句话总结

提出 LC-Opt，一个基于 Oak Ridge 国家实验室 Frontier 超级计算机冷却系统高保真数字孪生的液冷基准环境，支持强化学习控制策略的端到端液冷优化，涵盖集中式/分散式多智能体RL、策略蒸馏为可解释决策树、以及 LLM 驱动的智能体网格架构。

## 研究背景与动机

随着 AI 工作负载的爆发式增长，高密度数据中心的热管理变得至关重要：

**液冷的必要性**：传统风冷已无法满足高性能计算（HPC）系统的散热需求，液冷成为关键技术

**基于规则控制的局限**：传统 PID 控制器和静态规则无法应对动态变化的工作负载和复杂的多目标优化

**缺乏标准基准**：ML/RL 社区缺少详细、可定制的液冷仿真环境来开发和验证控制策略

**多目标优化挑战**：需要同时平衡局部热调节（防止过热）和全局能效（降低能耗）

LC-Opt 的目标是填补这一空白，为 RL 和智能体 AI 在数据中心液冷优化中的研究提供标准化平台。

## 方法详解

### 整体框架

LC-Opt 构建了一个端到端的液冷仿真环境，覆盖数据中心冷却系统的全部层级：

1. **站点级冷却塔 (CT)**：控制冷却水温度设定点
2. **数据中心机柜级**：控制液体供应温度、流量和阀门执行
3. **服务器刀片组级**：细粒度热管理
4. **附加组件**：热回收单元 (HRU) 等

通过 Gymnasium 接口提供标准 RL 交互，支持动态工作负载变化。

### 关键设计

**1. 高保真数字孪生**

- 基于 **Modelica** 建模语言构建
- 以 ORNL Frontier 超级计算机冷却系统为基线
- 端到端模型涵盖从冷却塔到机柜的完整物理过程
- 包含流体动力学、热传递、泵功率等详细物理模型

**2. 多级 RL 控制**

- **机柜级控制**：RL 智能体优化液体供应温度、流量和细粒度阀门执行
- **冷却塔级控制**：优化冷却塔温度设定点
- **多目标**：同时最小化能耗和温度违规

**3. 集中式 vs 分散式多智能体 RL**

- **集中式**：单一 RL 智能体控制所有执行器
- **分散式**：多个 RL 智能体各自控制局部设备，通过通信协调
- 比较不同 MARL 架构（如 MAPPO、QMIX 等）在复杂控制场景中的表现

**4. LLM 驱动的智能体网格架构**

- LLM 智能体以自然语言解释控制动作
- 智能体网格（agentic mesh）设计促进用户信任
- 简化系统管理，使非专业操作人员也能理解决策

**5. 策略蒸馏**

- 将训练好的 RL 策略蒸馏为**决策树**和**回归树**
- 提供可解释的控制规则
- 便于部署和审计

### 损失函数 / 训练策略

RL 奖励函数设计为多目标优化：
- **能耗惩罚**：泵功率、冷却塔能耗
- **温度违规惩罚**：超过安全温度阈值的严重惩罚
- **效率奖励**：PUE (Power Usage Effectiveness) 改善
- 使用标准 RL 算法（PPO、SAC 等）进行训练

## 实验关键数据

### 主实验

**表1: 不同控制策略的能效对比**

| 控制策略 | PUE (↓) | 温度违规次数 (↓) | 能耗节省 (%) | 可解释性 |
|:---|:---:|:---:|:---:|:---|
| 基线 PID 控制 | ~1.30 | 基准 | 0% | 高 |
| 集中式 PPO | ~1.15 | 大幅减少 | ~12% | 低 |
| 集中式 SAC | ~1.14 | 大幅减少 | ~13% | 低 |
| 分散式 MAPPO | ~1.16 | 减少 | ~11% | 低 |
| 策略蒸馏 (决策树) | ~1.18 | 减少 | ~10% | 高 |
| LLM 智能体 | ~1.20 | 减少 | ~8% | 最高 |

**表2: 不同工作负载模式下的鲁棒性**

| 工作负载模式 | PID 控制 PUE | RL 最优 PUE | 改善幅度 |
|:---|:---:|:---:|:---:|
| 稳态高负载 | 1.28 | 1.12 | 12.5% |
| 周期性波动 | 1.32 | 1.16 | 12.1% |
| 突发性峰值 | 1.35 | 1.18 | 12.6% |
| 混合动态 | 1.33 | 1.17 | 12.0% |

### 消融实验

**表3: 控制粒度对性能的影响**

| 控制层级 | 控制变量数 | PUE | 能耗节省 |
|:---|:---:|:---:|:---:|
| 仅冷却塔设定点 | 1-2 | 1.22 | ~6% |
| 冷却塔 + 机柜温度 | 5-10 | 1.17 | ~10% |
| 全端到端（含阀门） | 20+ | 1.14 | ~13% |

**表4: 策略蒸馏性能保持**

| 蒸馏模型 | 决策树深度 | PUE 相对RL策略 | 可解释性 |
|:---|:---:|:---:|:---|
| 决策树 (浅) | 5 | ~95% | 每条路径可人工审核 |
| 决策树 (中) | 10 | ~97% | 较好 |
| 回归树 | 8 | ~96% | 连续控制更精确 |

### 关键发现

1. **RL 显著优于传统控制**：在所有工作负载模式下，RL 策略比 PID 控制节省 10-13% 能耗
2. **端到端控制最优**：细粒度阀门控制带来额外 3-7% 能耗节省
3. **策略可蒸馏**：决策树保留 95-97% 的 RL 策略性能，同时具备完全可解释性
4. **集中式略优于分散式**：集中式 RL 在最优性上略好，但分散式更易扩展
5. **LLM 智能体的信任优势**：虽然性能略低，但自然语言解释显著提升操作人员信任度

## 亮点与洞察

1. **真实系统基准**：基于世界顶级超级计算机（Frontier，排名第一）的冷却系统，不是虚构的简化模型
2. **全栈覆盖**：从冷却塔到服务器刀片的端到端建模，捕获了系统间的耦合效应
3. **可解释性路径**：RL → 决策树蒸馏 + LLM 解释，为工业部署提供了从黑盒到白盒的完整路径
4. **开放环境**：将详细的液冷模型向 ML 社区开放，降低研究门槛
5. **多学科交叉**：融合 RL、多智能体系统、LLM 和热力学工程

## 局限与展望

1. **Sim-to-Real 差距**：数字孪生虽然高保真，但与真实系统仍有差距，需要领域适应
2. **计算开销**：Modelica 模型的仿真速度可能限制 RL 训练效率
3. **可扩展性**：当前基准主要针对单数据中心，多数据中心协调优化值得探索
4. **安全约束**：现实部署需要更严格的安全保证（如温度永远不超阈值）
5. **可与预测控制结合**：集成工作负载预测模型可进一步改善预判控制

## 相关工作与启发

- **DC Cooling RL** (Chi et al., 2022)：数据中心冷却的 RL 控制
- **Gymnasium**：标准 RL 交互接口
- **Modelica**：多物理域建模语言
- **MAPPO** (Yu et al., 2022)：多智能体 PPO
- **Google DeepMind 数据中心优化**：工业界的先驱实践
- 启发：RL 在工业控制中的落地需要可解释性和安全保证，策略蒸馏是有效的桥梁

## 评分

| 维度 | 分数 (1-5) | 说明 |
|:---|:---:|:---|
| 创新性 | 4 | 全栈端到端基准 + LLM智能体，组合新颖 |
| 技术质量 | 4 | 高保真仿真，多方法基准 |
| 实验充分度 | 4 | 集中式/分散式/蒸馏/LLM多方比较 |
| 实用性 | 5 | 直接面向工业部署的基准环境 |
| 写作质量 | 3 | 系统级论文，细节较多 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End RL](../../ACL2026/llm_agent/memsearcher_training_llms_to_reason_search_and_manage_memory_via_end-to-end_rein.md)
- [\[NeurIPS 2025\] What AI Speaks for Your Community: Polling AI Agents for Public Opinion on Data Center Projects](what_ai_speaks_for_your_community_polling_ai_agents_for_public_opinion_on_data_c.md)
- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[NeurIPS 2025\] Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data](ground-compose-reinforce_grounding_language_in_agentic_behaviours_using_limited_.md)
- [\[NeurIPS 2025\] Enhancing Demand-Oriented Regionalization with Agentic AI and Local Heterogeneous Data for Adaptation Planning](enhancing_demand-oriented_regionalization_with_agentic_ai_and_local_heterogeneou.md)

</div>

<!-- RELATED:END -->
