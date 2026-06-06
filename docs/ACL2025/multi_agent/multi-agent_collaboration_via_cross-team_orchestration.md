---
title: >-
  [论文解读] Multi-Agent Collaboration via Cross-Team Orchestration
description: >-
  [ACL 2025][多智能体][多智能体协作] 提出 Cross-Team Orchestration (Croto)，一个可扩展的多团队协作框架，通过将多个独立 agent 团队组织起来进行跨团队交互…
tags:
  - "ACL 2025"
  - "多智能体"
  - "多智能体协作"
  - "跨团队编排"
  - "解决方案聚合"
  - "软件开发"
  - "故事生成"
---

# Multi-Agent Collaboration via Cross-Team Orchestration

**会议**: ACL 2025  
**arXiv**: [2406.08979](https://arxiv.org/abs/2406.08979)  
**代码**: [GitHub](https://github.com/OpenBMB/ChatDev/tree/macnet)  
**领域**: 其他  
**关键词**: 多智能体协作, 跨团队编排, 解决方案聚合, 软件开发, 故事生成

## 一句话总结

提出 Cross-Team Orchestration (Croto)，一个可扩展的多团队协作框架，通过将多个独立 agent 团队组织起来进行跨团队交互，利用层次化分组 (Hierarchy Partitioning) 和贪心聚合 (Greedy Aggregation) 机制将各团队的多样化解决方案融合为更优结果。

## 研究背景与动机

基于 LLM 的多智能体协作已经在软件开发、故事生成等领域取得进展，但现有方法存在根本性限制：

**单团队单路径**：ChatDev 等链式团队按预定义阶段顺序执行，每个阶段只产生一个结果，形成唯一的决策路径。如果某个特定配置对某类问题容易犯相同错误，就很难自我纠正。

**简单并行不够**：直接运行 n 个团队并集成结果，没有利用中间阶段的协作潜力。类似于并行探索 n 条路径但不利用路径交叉点。

**图结构方法限制**：GPTSwarm 等图结构范式需要对每个节点和边进行任务特定定制，使用复杂且难以泛化。

核心问题：**如何让多个 agent 团队在保持独立性的同时获取和利用其他团队的洞察，产出更优解决方案？**

## 方法详解

### 整体框架

Croto 将多个链式团队（Chain-as-a-Team）组织为一个协作网络：
1. 多个团队并行执行相同任务，各自独立提出解决方案
2. 在预定义的关键阶段（如设计、编码），各团队暂停工作流
3. 提取各团队的解决方案进行跨团队交互
4. 通过层次化分组 + 贪心聚合产出更优方案
5. 聚合结果分发回所有团队，替换原有方案，指导后续阶段

### 关键设计

1. **跨团队交互网络**：

    - 功能：在关键阶段建立团队间的信息交换通道
    - 核心思路：$\mathcal{N} = \{\mathcal{V}, \mathcal{E}\}$，节点 $\mathcal{V}$ 为所有团队的阶段，边 $\mathcal{E}$ 连接不同团队的同名关键阶段
    - 设计动机：只在关键决策点交互，既保持团队独立性又实现协作

2. **贪心聚合 (Greedy Aggregation)**：

    - 功能：将多个团队的解决方案 $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$ 合成为更优方案 $s^* = \alpha(\theta(\mathcal{S}))$
    - 核心思路：不是简单投票选最好的，而是让聚合 agent 提取每个方案的优点和缺点，贪心地整合优点、消除缺点
    - 剪枝机制 $\theta$：聚合前先过滤掉低质量方案（通过 Quality 指标评分），减少聚合负担
    - 设计动机：直接聚合所有方案可能引入噪声，先剪枝再聚合效果更好

3. **层次化分组 (Hierarchy Partitioning)**：

    - 功能：将方案分组、逐层聚合，避免一次性处理过多方案
    - 核心思路：均匀分组（每组 u 个方案），每组内先聚合，聚合结果再分组、再聚合，直到得到最终唯一方案
    - 公式：$s^* = \alpha_x(\tau_x(\alpha_{x-1}(\ldots\alpha_1(\tau_1(\mathcal{S}^0)))))$
    - 设计动机：解决长上下文问题——同时处理 8 个方案可能超出 LLM 的有效处理能力

4. **团队多样性设计**：

    - 不同团队使用不同的 temperature 参数和链长度
    - 链长度多样性可手动设置也可自主演变
    - 确保各团队探索不同的决策空间

### 损失函数 / 训练策略

Croto 是一个纯推理时框架，不涉及训练。关键配置：
- 默认 8 个团队，temperature = 0.2
- 每阶段最多 5 轮 agent 通信
- 使用 GPT-3.5-Turbo 作为基础模型
- 软件开发任务在编码和代码补全阶段设为关键阶段

## 实验关键数据

### 主实验—软件生成（表格）

| 方法 | 范式 | 完整性 | 可执行性 | 一致性 | 质量 |
|------|------|--------|----------|--------|------|
| GPT-Engineer | 单Agent | 0.502 | 0.358 | 0.768 | 0.543 |
| MetaGPT | 单团队 | 0.483 | 0.415 | 0.739 | 0.545 |
| ChatDev | 单团队 | 0.744 | 0.813 | 0.781 | 0.779 |
| AgentVerse | 单团队 | 0.650 | 0.850 | 0.776 | 0.759 |
| GPTSwarm | 图结构 | 0.800 | 0.550 | 0.779 | 0.710 |
| **Croto** | **跨团队** | **0.795** | **0.928** | **0.796** | **0.840** |

### 剪枝机制消融实验（表格）

| 配置 | 完整性 | 可执行性 | 一致性 | 质量 |
|------|--------|----------|--------|------|
| 8-team Croto (无剪枝) | 0.706 | 0.828 | 0.792 | 0.775 |
| 8-team Croto (**+剪枝**) | **0.795** | **0.928** | **0.796** | **0.840** |
| Δ | +0.089 | +0.100 | +0.004 | +0.065 |

### 关键发现

1. **可执行性大幅提升**：Croto 的可执行性达到 0.928，比最强基线 AgentVerse 的 0.850 高 7.8 个百分点。跨团队协作能有效发现和修正代码错误
2. **团队数量的有趣权衡**：不使用剪枝时，4 个团队效果最好（质量 0.789），超过 4 个团队后质量开始下降。原因是 agent 难以有效综合过多方案的特征
3. **剪枝是扩展关键**：8 团队 + 剪枝 (Quality 0.840) > 4 团队无剪枝 (0.789)，说明剪枝机制解决了方案过多带来的噪声问题
4. **可执行性 vs 完整性的反向关系**：团队增加时，可执行性上升但完整性下降，存在内在权衡
5. **故事生成泛化**：Croto 在 ROCStories 故事生成任务上也展现了显著提升，证明框架不限于编程任务

## 亮点与洞察

1. **独立性与协作的平衡**：Croto 的精妙之处在于——团队在自己的阶段中独立工作保持多样性，仅在关键阶段交互获取洞察。这比全程协作或完全独立都更优
2. **层次化聚合应对规模**：将 "多对一" 聚合分解为 "多对少" 的层次过程，是处理 LLM 长上下文限制的实用策略
3. **类比进化算法**：多团队提案 → 评分筛选 → 聚合优化 → 分发继续，类似遗传算法中的群体进化过程

## 局限与展望

1. **Token 成本高**：8 个团队 × 多阶段 × 聚合过程，GPT-3.5-Turbo 的 API 调用成本显著。论文未报告具体的 token 消耗对比
2. **关键阶段需人工定义**：哪些阶段发起跨团队交互需要预先设定，理想情况应自动识别
3. **聚合 agent 的瓶颈**：聚合质量完全依赖 LLM 的综合能力，如果 LLM 不擅长从多个方案中提取优点，整体框架将受限
4. **评估规模较小**：仅在 15 个软件任务和 10 个故事任务上评估，规模偏小
5. **仅测试 GPT-3.5-Turbo**：未验证在更强或更弱的模型上是否有类似效果

## 相关工作与启发

- ChatDev 是 Croto 的单团队基线，Croto 直接在 ChatDev 仓库基础上实现
- MACNET 发现解决方案质量随 agent 数量呈 logistic 增长，Croto 用跨团队协作突破了这一增长曲线的上限
- 与集成学习的思想类似：多个弱方案 → 聚合为强方案，但 Croto 的聚合是语义级别的而非简单投票

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 跨团队编排的概念新颖，层次化分组 + 贪心聚合的设计实用且有洞察
- **实验充分度**: ⭐⭐⭐ — 软件开发和故事生成两个任务，但样本规模小（15+10），仅用一个 LLM 基线
- **写作质量**: ⭐⭐⭐⭐ — 形式化定义清晰（Definition 1-3），图示直观
- **价值**: ⭐⭐⭐⭐ — 为多 agent 系统提供了一个实用的协作范式，已集成到 ChatDev 开源项目中

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Preventing Rogue Agents Improves Multi-Agent Collaboration](preventing_rogue_agents_improves_multi-agent_collaboration.md)
- [\[ACL 2025\] Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems](beyond_frameworks_multi_agent_collaboration.md)
- [\[ICML 2026\] MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks](../../ICML2026/multi_agent/mas-orchestra_understanding_and_improving_multi-agent_reasoning_through_holistic.md)
- [\[ACL 2026\] ConSensus: Multi-Agent Collaboration for Multimodal Sensing](../../ACL2026/multi_agent/consensus_multi-agent_collaboration_for_multimodal_sensing.md)
- [\[ACL 2026\] PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation](../../ACL2026/multi_agent/posterforest_hierarchical_multi-agent_collaboration_for_scientific_poster_genera.md)

</div>

<!-- RELATED:END -->
