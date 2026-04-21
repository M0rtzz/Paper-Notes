---
title: >-
  [论文解读] Which LLM Multi-Agent Protocol to Choose?
description: >-
  [ICLR 2026][多Agent协议] 本文提出ProtocolBench基准和ProtocolRouter路由器，首次系统性比较了多Agent系统中的通信协议（A2A、ACP、ANP、Agora等）在任务成功率、延迟、消息开销和鲁棒性四个维度上的差异，并通过可学习的协议路由器实现场景自适应的协议选择，最高降低18.1%的故障恢复时间。
tags:
  - ICLR 2026
  - 多Agent协议
  - ProtocolBench
  - ProtocolRouter
  - A2A
  - 通信协议评估
---

# Which LLM Multi-Agent Protocol to Choose?

**会议**: ICLR 2026  
**arXiv**: [2510.17149](https://arxiv.org/abs/2510.17149)  
**代码**: 有（论文附带benchmark artifacts）  
**领域**: LLM / NLP / 多Agent系统  
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

本文工作包含两个核心组件：

1. **ProtocolBench**: 一个标准化的协议评估基准，在多个场景下从四个维度系统比较Agent协议
2. **ProtocolRouter**: 一个可学习的协议路由器，根据场景需求和运行时信号动态选择最优协议

### 关键设计

1. **ProtocolBench 评估框架**:

    - **四维评估轴**: 
        - 任务成功率（Task Success）: 衡量协议是否能支撑Agent正确完成任务
        - 端到端延迟（End-to-End Latency）: 从任务下发到完成的总时间
        - 消息/字节开销（Message/Byte Overhead）: 协议通信的额外开销
        - 故障鲁棒性（Robustness Under Failures）: 面对网络故障、Agent崩溃等异常时的恢复能力
    - **评估场景**: 
        - Streaming Queue: 流式任务处理场景，测试吞吐和延迟
        - Fail-Storm Recovery: 大规模故障恢复场景，测试鲁棒性
        - GAIA: 通用Agent智能评估
    - **设计动机**: 提供全面、公平、可复现的协议对比框架

2. **协议实现与比较**:

    - **A2A (Agent-to-Agent)**: Google开发的Agent间通信协议，侧重互操作性
    - **ACP (Agent Communication Protocol)**: 基于FIPA标准的结构化消息协议
    - **ANP (Agent Network Protocol)**: 面向大规模Agent网络的分布式协议
    - **Agora**: 支持灵活消息路由的开放协议
    - 每种协议在统一的测试环境下评估，确保公平对比
    - **设计动机**: 覆盖当前主流协议，确保结论的广泛适用性

3. **ProtocolRouter 动态路由器**:

    - **输入信号**: 场景需求描述（如延迟敏感度、容错要求）和运行时监控信号（如当前网络状态、Agent负载）
    - **路由粒度**: 支持场景级路由（整个场景使用一种协议）和模块级路由（不同模块使用不同协议）
    - **学习方法**: 基于历史性能数据训练的轻量级路由模型，将场景特征映射到协议选择
    - **设计动机**: 没有万能协议，自适应选择可以充分利用各协议的优势

4. **ProtocolRouterBench**:

    - 专门用于评估协议路由器性能的标准化基准
    - 包含多样化的场景配置和性能指标
    - **设计动机**: 为路由器研究提供可复现的评估标准

### 训练策略

- ProtocolRouter使用监督学习方法训练，训练数据来自ProtocolBench的历史运行记录
- 通过在线学习不断适应新的场景和运行时条件
- 路由决策的延迟开销极低，不影响系统整体性能

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

## 相关论文

- [Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)
- [MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning](../../AAAI2026/llm_evaluation/maps_multi-agent_personality_shaping_for_collaborative_reaso.md)
- [Model Context Protocol for Vision Systems: Audit, Security, and Protocol Extensions](../../NeurIPS2025/llm_evaluation/model_context_protocol_for_vision_systems_audit_security_and_protocol_extensions.md)
- [Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)
- [UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)

<!-- RELATED:END -->
