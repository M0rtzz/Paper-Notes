---
title: >-
  [论文解读] Preventing Rogue Agents Improves Multi-Agent Collaboration
description: >-
  [ACL 2025][多智能体协作] 提出一种通过实时监控 Agent 不确定性来检测"失控 Agent"（rogue agent）并进行干预的框架，在自建的 WhoDunitEnv 多智能体协作环境以及代码生成和资源可持续性任务上分别取得高达 17.4%、2.5% 和 20% 的性能提升。
tags:
  - "ACL 2025"
  - "多智能体协作"
  - "异常检测"
  - "不确定性估计"
  - "干预机制"
  - "WhoDunitEnv"
---

# Preventing Rogue Agents Improves Multi-Agent Collaboration

**会议**: ACL 2025  
**arXiv**: [2502.05986](https://arxiv.org/abs/2502.05986)  
**代码**: [有](https://github.com/Ohav/rogue-agents)  
**领域**: 其他  
**关键词**: 多智能体协作, 异常检测, 不确定性估计, 干预机制, WhoDunitEnv

## 一句话总结

提出一种通过实时监控 Agent 不确定性来检测"失控 Agent"（rogue agent）并进行干预的框架，在自建的 WhoDunitEnv 多智能体协作环境以及代码生成和资源可持续性任务上分别取得高达 17.4%、2.5% 和 20% 的性能提升。

## 研究背景与动机

多智能体系统（MAS）让多个专门化 Agent 协作解决共享任务，在推理增强、软件开发模拟、人类行为模拟等方面展现出巨大潜力。但它也有一个核心风险——**一个单独的 Agent 就能导致整个系统失败**。

当前 LLM Agent 在协作中面临的典型问题包括：

- **忽略关键信息**：Agent 可能忽视通信中的重要消息
- **被无关信息干扰**：产生幻觉或引入噪声
- **错误传播**：一个 Agent 的错误会通过通信渠道扩散，且随着多轮交互被放大
- **过早行动**：Agent 可能在信息不足时就做出不可逆的终结性决策

现有改进方法（修改通信协议、加入 belief 系统、增加推理模块）虽有帮助，但都无法阻止一个"失控 Agent"将整个系统拖入失败。

本文受工业监控系统（入侵检测、制造质量控制）和生物免疫系统的启发，提出：**在 Agent 行动之前检测到潜在失败并主动干预**，而不是事后修复。

## 方法详解

### 整体框架

框架包含两个核心组件：

1. **Monitor（监控器）**：基于 Agent 不确定性信号预测系统将要失败的概率
2. **Intervention（干预机制）**：当失败概率超过阈值 τ 时，触发环境干预阻止错误传播

### 关键设计

1. **基于不确定性的监控**：

    - **特征提取**：在 Agent 生成动作时，提取输出概率分布 $\mathbf{p}_i$ 中关键位置（动作选择和前序思考）的三种统计量——**熵（entropy）**、**方差熵（varentropy）** 和 **峰度（kurtosis）** 的最大值，加上当前轮次计数，共计 m≤4 个特征
    - **分类器**：使用简单的多项式岭回归分类器拟合成功概率 $f: \mathbb{R}^m \to [0,1]$，在训练数据上以每轮的布尔标签训练
    - **触发条件**：当 $P(\text{success}) < \tau$ 时触发干预
    - 设计动机：Agent 在动作选择时"困惑"（高熵）意味着可能引入噪声导致系统失败。选择简单分类器是因为特征维度极低（≤4），且实验证明已足够有效

2. **干预策略**：

    - 区分**可逆动作**（信息共享）和**不可逆动作**（指控嫌疑人、提交代码、消耗资源）
    - 干预时回滚到上一个不可逆动作之后的状态——即撤销所有可逆操作，给 Agent 重新协作的机会
    - 在 WhoDunitEnv 中：信息共享可逆，指控不可逆，干预 = 重启整个通信
    - 在 CodeGen 中：代码生成不可逆，干预 = Judge 和 Tester 重写反馈
    - 在 GovSim 中：资源消耗不可逆，干预 = 重置上一轮讨论

3. **WhoDunitEnv 环境设计**：

    - 灵感来自 Guess Who 桌游，Agent 扮演侦探协作找出罪犯
    - **非对称变体（Asym）**：两个 Agent——Accuser（知道罪犯描述，可询问和指控）和 Intel（知道所有嫌疑人描述，可回应）
    - **对称变体（Sym）**：所有 Agent 平等，各自持有罪犯的部分线索（3 条），可共享、指控或跳过
    - **复杂度可调**：嫌疑人数量（6/10/14/20）、属性数量、轮次上限
    - 使用 ReAct 提示让 Agent 先思考再行动

### 损失函数 / 训练策略

- 监控器使用多项式岭回归，在训练集的每轮中间状态上训练（正/负标签 = 最终成功/失败）
- 通过验证集网格搜索最优特征组合、多项式阶数 d∈[1,5] 和阈值 τ∈[0,1]
- 干预次数有上限（WhoDunitEnv: 1-2 次，CodeGen/GovSim: 1 次）

## 实验关键数据

### WhoDunitEnv-Asym 主实验（Success Rate）

| 模型 | 无干预 | + 监控&干预（1次重置） | + 监控&干预（2次重置） |
|------|--------|---------------------|---------------------|
| GPT-4o | ~62% | +6.1% | +11.8%（最终） |
| Llama-3.1-70B | ~60% | +10.6% | — |
| Qwen-2.5-72B | ~60% | +10.3% | — |

### CodeGen 实验（Pass@1，Llama-3.1-70B）

| 方法 | HumanEval | LiveCodeBench |
|------|-----------|---------------|
| Zero-shot | 80.5% | 18.2% |
| Multi-agent（无监控） | 81.6% | 19.3% |
| **Multi-agent + Monitor** | **83.5%** | **21.8%** |

### GovSim 实验

| 模型 | 方法 | Survival Rate | Efficiency |
|------|------|---------------|------------|
| Qwen-1.5-110B | 无干预 | 35.0% | 49.4% |
| Qwen-1.5-110B | **+ Monitor** | **55.0%** | 48.8% |
| GPT-4o | 无干预 | 100% | 69.1% |
| GPT-4o | **+ Monitor** | 100% | **76.0%** |

### 消融实验（Qwen-2.5-72B, WhoDunitEnv-Asym）

| 变体 | Success Rate |
|------|-------------|
| 无干预 | 59.8% |
| 最佳基线（随机重置） | 62.5% |
| 最差监控器 | 62.0% |
| 动作=重采样 Agent（而非重置通信） | 61.3% |
| 次佳监控器 | 69.3% |
| **最佳监控器（单次重置）** | **70.1%** |
| **最佳监控器（双次重置）** | **72.2%** |

### 关键发现

1. **监控+干预一致有效**：在所有环境和模型上都观察到显著提升，WhoDunitEnv 高达 17.4%，GovSim 高达 20%
2. **监控器质量至关重要**：消融实验中最差监控器仅提升 2%（接近随机基线 2.6%），最佳监控器提升 10.3%，说明精确的失败预测是关键
3. **干预方式同样重要**：重采样单个 Agent（不重置通信）仅提升 1.5%，远不如重置通信渠道，说明问题根源在被污染的通信而非单个 Agent
4. **监控器泛化良好**：在 HumanEval 上训练的监控器在 LiveCodeBench 上同样有效（+2.5%）；固定 10 个嫌疑人训练的监控器在 6/14 个嫌疑人上也持续有效
5. **幻觉是最常见的触发原因**：定性分析 50 例触发中 48% 为幻觉、16% 为 Agent 崩坏（重复同一行为）、8% 为角色遗失、4% 为信息回忆失败，共 76% 属于四大类错误

## 亮点与洞察

- **极简但有效的监控器**：仅用 ≤4 个特征 + 多项式岭回归就能有效预测多 Agent 系统的失败，说明 LLM 的不确定性信号（熵等）确实是失败的强指示器
- **"免疫系统"类比优雅**：将工业监控/免疫系统的实时检测-干预思想迁移到 LLM 多 Agent 系统，概念新颖且实用
- **WhoDunitEnv 设计良好**：模块化的对称/非对称变体、可调复杂度、结构化动作空间，为研究多 Agent 通信提供了优秀的测试床
- **分析监控器触发原因**：详细的定性分析（图 7）不仅验证了方法有效性，还提供了 LLM 协作失败的分类学（幻觉 > 崩坏 > 角色遗失 > 回忆失败）
- **代价分析诚实**：明确报告干预带来的额外推理成本（平均多 1.6-1.9 倍轮次）

## 局限与展望

- 需要训练数据中有足够的成功案例来训练监控器——对于 Llama-3-70B 在 GovSim 上近乎零成功率的情况无法应用
- 干预方式较为简单（全局重置），未探索更精细的局部修复（如仅回滚有问题的 Agent 的消息）
- 环境仍相对简单（Guess Who 游戏、单段代码生成），在更复杂的开放式协作场景（如完整软件开发）中的效果有待验证
- 对于闭源模型只能获取 top-k token 概率进行近似，监控精度可能受限
- 监控器是离线训练的，未来可探索在线学习以减少监控器自身的样本需求

## 相关工作与启发

本文桥接了三个方向：(1) 多 Agent 通信协议设计（Li et al. 2023; Hong et al. 2024），(2) 语言模型的不确定性估计（Kadavath et al. 2022; Yona et al. 2024），(3) 多代生成聚合（Wang et al. 2023; Du et al. 2024）。区别于后者的事后聚合策略，本文侧重于"在不可逆行动之前检测并干预"，更适合 Agent 环境中动作有后果的场景。启发在于：对于任何多 Agent 系统，添加一个轻量级"免疫层"来监控通信质量和 Agent 状态可能是提升鲁棒性的通用策略。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 实时监控+干预的思路在多 Agent LLM 系统中属首次系统性提出和验证，WhoDunitEnv 也是有价值的贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ — 3 个环境、3+ 模型、多复杂度级别、消融实验、监控器质量分析、定性分析、泛化测试，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 概念清晰，图示直观，形式化定义准确，环境描述详尽
- **价值**: ⭐⭐⭐⭐ — 提供了一种即插即用的多 Agent 协作增强策略，对日益流行的 Agent 系统有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Multi-Agent Collaboration via Cross-Team Orchestration](multi-agent_collaboration_via_cross-team_orchestration.md)
- [\[ACL 2025\] Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems](beyond_frameworks_multi_agent_collaboration.md)
- [\[ACL 2025\] DocAgent: A Multi-Agent System for Automated Code Documentation Generation](docagent_a_multi-agent_system_for_automated_code_documentation_generation.md)
- [\[ACL 2025\] CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate](cortexdebate_debating_sparsely_and_equally_for_multi-agent_debate.md)
- [\[ACL 2025\] GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning](getreason_enhancing_image_context_extraction_through_hierarchical_multi-agent_re.md)

</div>

<!-- RELATED:END -->
