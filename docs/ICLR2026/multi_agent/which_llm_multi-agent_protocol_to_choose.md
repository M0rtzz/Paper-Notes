---
title: >-
  [论文解读] Which LLM Multi-Agent Protocol to Choose?
description: >-
  [ICLR 2026][多智能体][多Agent协议] 本文提出ProtocolBench基准和ProtocolRouter路由器，首次系统性比较了多Agent系统中的通信协议（A2A、ACP、ANP、Agora等）在任务成功率、延迟、消息开销和鲁棒性四个维度上的差异…
tags:
  - "ICLR 2026"
  - "多智能体"
  - "多Agent协议"
  - "ProtocolBench"
  - "ProtocolRouter"
  - "A2A"
  - "通信协议评估"
---

# Which LLM Multi-Agent Protocol to Choose?

**会议**: ICLR 2026  
**arXiv**: [2510.17149](https://arxiv.org/abs/2510.17149)  
**代码**: 有（论文附带benchmark artifacts）  
**领域**: LLM评测  
**关键词**: 多Agent协议, ProtocolBench, ProtocolRouter, A2A, 通信协议评估

## 一句话总结

本文提出ProtocolBench基准和ProtocolRouter路由器，首次系统性比较了多Agent系统中的通信协议（A2A、ACP、ANP、Agora等）在任务成功率、延迟、消息开销和鲁棒性四个维度上的差异，并通过可学习的协议路由器实现场景自适应的协议选择，最高降低18.1%的故障恢复时间。

## 研究背景与动机

随着大规模多Agent系统的演进，通信协议层已成为影响系统性能和可靠性的关键但被忽视的因素：

**协议爆发式增长**: 近年来涌现了多种Agent通信协议，包括Google的A2A（Agent-to-Agent）、ACP（Agent Communication Protocol）、ANP（Agent Network Protocol）、Agora等，但缺乏统一的比较标准

**选择困难**: 在实际部署中，协议选择通常基于直觉或经验，缺乏数据驱动的决策支持

**性能差异被低估**: 通常认为协议只是"管道"，对系统性能影响有限，但实际上协议差异可导致高达36.5%的完成时间差异

**缺乏标准化评估**: 不同协议的论文使用不同的任务和指标进行评估，无法直接对比

**单一协议的局限性**: 没有一种协议在所有场景下都是最优的，但现有系统通常只使用单一协议

本文的目标是建立标准化的协议评估框架，并通过自适应路由实现最优协议选择。

## 方法详解

### 整体框架

本文把"选哪个多Agent通信协议"这件常被拍脑袋决定的事拆成两块来做：先用 ProtocolBench 这个标准化基准，把 A2A、ACP、ANP、Agora 等主流协议放到统一的测试场景里，沿任务成功率、延迟、消息开销、故障鲁棒性四个维度量一遍，得到"哪种协议在哪种场景下强"的数据；再用 ProtocolRouter 这个可学习的路由器，把场景需求和运行时信号映射到具体协议，让系统在跑的时候动态切换到当下最合适的那一个。

### 关键设计

**1. ProtocolBench 四维评估框架：让不同协议第一次能被公平比较。** 此前各协议论文各用各的任务和指标，结论无法直接对照，协议层的性能差异因此被长期低估。ProtocolBench 固定四条评估轴——任务成功率（Task Success，协议能否支撑 Agent 正确完成任务）、端到端延迟（End-to-End Latency，从任务下发到完成的总耗时）、消息/字节开销（通信引入的额外成本）、故障鲁棒性（Robustness Under Failures，面对网络故障、Agent 崩溃时的恢复能力），并在三类代表性场景里跑：Streaming Queue 压吞吐与延迟，Fail-Storm Recovery 制造大规模故障考验恢复能力，GAIA 检验通用 Agent 任务的完成质量。所有协议在同一测试环境下评测，保证对比公平、可复现，这也是后续路由器训练数据的来源。

**2. 统一实现下的主流协议横评：覆盖面决定结论的可推广性。** 框架把当前几种代表性协议都纳入同一套接口里实现并测评：A2A（Agent-to-Agent，Google 提出，强调跨平台互操作）、ACP（Agent Communication Protocol，基于 FIPA 标准的结构化消息）、ANP（Agent Network Protocol，面向大规模 Agent 网络的分布式设计）、Agora（支持灵活消息路由的开放协议）。把它们放进同一环境横比，才使得"协议选择最高带来 36.5% 完成时间差异"这样的结论具有跨场景的说服力，而不是某一篇论文的孤例。

**3. ProtocolRouter 动态路由器：把协议选择从人工经验变成数据驱动决策。** 既然没有一种协议在所有场景下最优，固定单协议必然在某些场景上吃亏。路由器接收两类输入——场景需求描述（如延迟敏感度、容错要求）和运行时监控信号（如当前网络状态、Agent 负载），用一个基于 ProtocolBench 历史性能数据训练的轻量模型，把这些特征映射到协议选择。它支持两种粒度：场景级路由（整个场景统一用一种协议）和模块级路由（同一系统里不同模块各用最合适的协议）；实验显示模块级更优，印证了"一个系统内部的不同组件可能本就适配不同协议"这一洞察。

**4. ProtocolRouterBench：给路由器本身一把可复现的尺子。** 评估路由器不能复用评估单协议的设置，因此本文额外构造了专门衡量路由器性能的标准化基准，包含多样的场景配置和对应指标，让"路由策略好不好"也能被一致、可复现地比较，避免路由器研究重蹈各测各的覆辙。

### 损失函数 / 训练策略

ProtocolRouter 以监督学习训练，标注来自 ProtocolBench 的历史运行记录——每条记录给出"某场景特征下哪种协议表现最好"，构成场景特征到协议选择的映射样本；模型上线后再通过在线学习持续吸收新场景与新的运行时条件。由于路由模型刻意做得轻量，单次路由决策的延迟开销极低，几乎不影响系统整体性能。

## 实验关键数据

### 主实验

| 场景 | 指标 | 最佳协议 | 最差协议 | 差异幅度 |
|------|------|----------|----------|----------|
| Streaming Queue | 完成时间 | - | - | 最大36.5%差异 |
| Streaming Queue | 端到端延迟 | - | - | 均值差3.48s |
| Fail-Storm Recovery | 恢复时间 | - | - | 显著差异 |
| GAIA | 任务成功率 | 场景依赖 | 场景依赖 | 协议间差异一致 |

### ProtocolRouter性能

| 对比基线 | 指标 | ProtocolRouter | 最佳单协议 | 提升 |
|----------|------|----------------|------------|------|
| Fail-Storm Recovery | 恢复时间 | 最优 | 次优 | 降低18.1% |
| GAIA | 成功率 | 更高 | 次优 | 场景相关提升 |
| 综合 | 加权得分 | 最优 | 协议依赖 | 一致提升 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 场景级 vs 模块级路由 | 模块级更优 | 更细粒度的路由带来更好的适配 |
| 有/无运行时信号 | 有信号更优 | 实时监控信息提升路由准确性 |
| 路由器模型大小 | 轻量即可 | 小模型足以实现有效路由 |
| 固定协议 vs 路由 | 路由一致更优 | 验证了自适应的必要性 |

### 关键发现

1. **协议选择显著影响性能**: 不同协议在同一场景下的性能差异可达36.5%，远超预期
2. **没有万能协议**: 在不同场景下，最优协议不同，单一协议策略必然存在妥协
3. **延迟差异突出**: Streaming Queue场景中端到端延迟差异达3.48秒，对实时应用影响巨大
4. **鲁棒性差异一致**: 在故障场景下，不同协议的恢复能力存在稳定的差异模式
5. **自适应路由有效**: ProtocolRouter在所有场景下均优于最佳单一协议，证明了动态路由的价值
6. **模块级路由更优**: 同一系统中不同模块可能适合不同协议，细粒度路由效果更好

## 亮点与洞察

1. **首个协议基准**: ProtocolBench填补了多Agent协议评估领域的空白，为协议设计和选择提供了数据支撑
2. **实用的路由机制**: ProtocolRouter将"选哪个协议"从人工决策转变为数据驱动决策，降低了部署门槛
3. **四维评估体系完备**: 任务成功率、延迟、开销、鲁棒性四个维度覆盖了实际部署中的核心关注点
4. **模块级路由洞察**: 揭示了同一系统内部不同组件可能适合不同协议的现象，为异构协议架构提供了理论支持
5. **36.5%的性能差异**: 这一数字有力地证明了协议选择不是"无关紧要的细节"，而是系统设计的关键决策

## 局限与展望

1. **协议覆盖范围**: 目前评估的协议种类有限，新兴协议（如MCP相关协议）尚未纳入
2. **场景多样性**: 评估场景虽然具有代表性，但可能无法覆盖所有实际使用模式
3. **路由延迟**: 虽然路由器本身很轻量，但在超低延迟场景下额外的路由开销仍需关注
4. **安全性考量**: 未充分评估不同协议在安全性（如消息加密、认证）方面的差异
5. **大规模验证**: 评估的Agent数量有限，千级或万级Agent场景下的表现有待验证
6. **协议混合的兼容性**: 模块级路由意味着系统中同时存在多种协议，兼容性和调试复杂度需要更多讨论
7. **动态环境适应**: 路由器对运行时环境变化（如网络拓扑变化、Agent动态加入/退出）的适应能力有待加强

## 相关工作与启发

- **A2A Protocol (Google)**: 面向Agent互操作的协议标准，强调跨平台兼容
- **MCP (Model Context Protocol)**: Anthropic的模型上下文协议，虽然不直接针对Agent间通信，但影响了协议设计思路
- **FIPA-ACL**: 传统的Agent通信语言标准，ACP在其基础上发展而来
- **AutoGen / CrewAI**: 多Agent框架，通常使用固定的通信模式
- **启发**: 
    - 协议层的研究可能成为多Agent系统性能优化的新突破口
    - 自适应协议路由的思路可以扩展到更多系统层面（如模型选择、工具选择）
    - 需要建立类似网络协议栈的Agent协议分层标准

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] ProtocolBench: Which LLM MultiAgent Protocol to Choose?](../../ICML2026/multi_agent/protocolbench_which_llm_multiagent_protocol_to_choose.md)
- [\[ICLR 2026\] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)
- [\[ICLR 2026\] KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](kvcomm_enabling_efficient_llm_communication_through_selective_kv_sharing.md)
- [\[ICLR 2026\] LH-Deception: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions](lh-deception_simulating_and_understanding_llm_deceptive_behaviors_in_long-horizo.md)
- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)

</div>

<!-- RELATED:END -->
