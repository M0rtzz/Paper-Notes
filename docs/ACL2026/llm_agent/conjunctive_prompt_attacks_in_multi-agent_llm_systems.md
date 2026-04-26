---
title: >-
  [论文解读] Conjunctive Prompt Attacks in Multi-Agent LLM Systems
description: >-
  [ACL 2026][LLM Agent][提示注入攻击] 本文研究多智能体 LLM 系统中的联合提示攻击（conjunctive prompt attacks）：用户查询中嵌入的触发键和被入侵远程代理中的隐藏模板各自看起来无害，但当路由将它们带到同一代理时会激活有害行为，现有防御（PromptGuard、Llama-Guard 等）均无法可靠阻止。
tags:
  - ACL 2026
  - LLM Agent
  - 提示注入攻击
  - 多智能体安全
  - 联合激活
  - 拓扑感知
  - 供应链威胁
---

# Conjunctive Prompt Attacks in Multi-Agent LLM Systems

**会议**: ACL 2026  
**arXiv**: [2604.16543](https://arxiv.org/abs/2604.16543)  
**代码**: [GitHub](https://github.com/UCF-ML-Research/ConjunctiveAgents)  
**领域**: AI安全 / 多智能体系统  
**关键词**: 提示注入攻击, 多智能体安全, 联合激活, 拓扑感知, 供应链威胁

## 一句话总结
本文研究多智能体 LLM 系统中的联合提示攻击（conjunctive prompt attacks）：用户查询中嵌入的触发键和被入侵远程代理中的隐藏模板各自看起来无害，但当路由将它们带到同一代理时会激活有害行为，现有防御（PromptGuard、Llama-Guard 等）均无法可靠阻止。

## 研究背景与动机

**领域现状**：LLM 安全研究主要关注单代理场景，但实际部署中多个专用代理通过任务分解、路由和工具调用协作。多智能体管线中，远程代理通常是黑盒——其权重、提示和系统模板可能由第三方托管。

**现有痛点**：单代理安全评估无法捕捉多智能体的新攻击面——提示分段、代理间路由、隐藏包装器创造了单点检查无法发现的漏洞。现有防御（PromptGuard、Llama-Guard）只检查孤立消息，无法检测跨代理组合后才产生的恶意行为。

**核心矛盾**：模块化设计提升了系统能力但也引入了供应链风险——攻击者不需要修改任何模型权重或客户端代理，只需在一个远程代理中注入看似无害的模板，就可能造成端到端的入侵。

**本文目标**：形式化联合提示攻击的威胁模型，开发拓扑感知的攻击优化框架，评估现有防御的有效性。

**切入角度**：将攻击成功建模为三个条件的联合（conjunction）：触发键存在于查询段中 + 该段被路由到被入侵代理 + 被入侵代理的模板被激活。

**核心 idea**：联合激活——攻击的两个组成部分各自无害，只有路由将它们带到一起时才激活，这使得逐点检查的防御天然失效。

## 方法详解

### 整体框架
攻击框架分为两阶段：优化阶段通过可微代理（Gumbel-Softmax）学习最优触发键放置位置、模板放置位置和路由偏置参数 $\theta^*$；推理阶段将学到的配置应用到黑盒多智能体系统中，评估端到端攻击成功率。

### 关键设计

1. **联合激活条件（Conjunctive Activation）**:

    - 功能：定义攻击成功的精确条件——需要三个要素同时满足
    - 核心思路：攻击激活当且仅当 $\exists j$ 使得 $(k \in s_j) \land (a_j = a^*)$，即触发键 $k$ 所在的查询段 $s_j$ 被路由到被入侵代理 $a^*$。触发键本身和模板本身都不具有恶意性——触发键可以是"请检查我的账户余额"这样的正常请求，模板可以是"用特殊格式输出结果"这样的无害指令
    - 设计动机：这种联合属性是本攻击与传统单点提示注入的根本区别——没有任何单一组件看起来可疑，安全审查无从下手

2. **拓扑和路由感知的优化**:

    - 功能：最大化攻击成功率同时最小化误激活
    - 核心思路：路由概率建模为 $\Pr[a=a^*|s] = \text{clip}(\alpha I_{acc}(s) + \rho I_{acc}(s) I_k(s))$，其中 $\alpha$ 是基线账户亲和度，$\rho$ 是攻击者可控的路由偏置。使用 Gumbel-Softmax 对离散决策变量（触发键位置、模板位置 $\tau \in \{prefix, wrap, suffix\}$）做可微松弛，梯度优化联合 ASR 目标
    - 设计动机：攻击成功本质上是概率性的且依赖拓扑——在星型、链式和 DAG 拓扑中路由动态完全不同，需要拓扑感知的优化策略

3. **四象限评估体制**:

    - 功能：严格隔离联合效应，排除单组件贡献
    - 核心思路：评估四种条件：clean（无键无模板）、key_only（有键无模板）、template_only（无键有模板）、both（键+模板）。只有 both 条件下 ASR 高而其余三个条件 ASR 低才证明真正的联合激活。使用确定性标记 token（__ACTIVATED__）做激活判定
    - 设计动机：如果 key_only 或 template_only 就能触发攻击，则不是联合攻击而是传统注入；四象限评估确保实验结论的因果有效性

### 损失函数 / 训练策略
攻击优化使用可微代理目标，通过 Gumbel-Softmax 松弛离散变量，梯度下降优化攻击配置 $\theta = (j, \tau, \rho)$。不修改任何模型权重。

## 实验关键数据

### 主实验

| 拓扑 | 优化后 ASR (both) | 非优化 ASR | key_only ASR | template_only ASR |
|------|-------------------|-----------|-------------|-------------------|
| Star | 高 | 低 | ~0 | ~0 |
| Chain | 高 | 低 | ~0 | ~0 |
| DAG | 高 | 低 | ~0 | ~0 |

### 消融实验

| 防御方法 | 是否阻止联合攻击 | 说明 |
|---------|-----------------|------|
| PromptGuard | 否 | 逐消息检查，各组件单独无害 |
| Llama-Guard 变体 | 否 | 同上，无法检测跨代理组合 |
| 工具限制 | 否 | 攻击不依赖工具调用 |
| 系统级控制 | 否 | 攻击在提示层面操作 |

### 关键发现
- 路由感知优化显著提升攻击成功率（相比非优化基线）同时保持低误激活率
- 攻击在星型、链式和 DAG 三种拓扑中均可迁移，但成功率随拓扑不同而变化
- 所有现有防御机制均无法可靠阻止联合攻击——因为它们的检查粒度是单条消息而非跨代理组合
- 模板位置（prefix vs wrap vs suffix）显著影响攻击效果

## 亮点与洞察
- **联合激活的威胁模型**非常有洞察力——这暴露了多智能体系统的结构性脆弱性：安全不能通过逐点检查实现，必须推理路由和跨代理组合
- 这种攻击与现实中的供应链攻击高度类似——第三方服务提供商的一个微小修改可能在特定条件下触发系统级入侵
- 启示：多智能体系统需要"全局上下文感知"的安全机制，而非孤立的消息级防御

## 局限与展望
- 假设攻击者能控制用户输入和一个远程代理的模板，这在某些部署场景中可能过于强大
- 激活判定使用人工标记 token，实际场景中恶意行为的判定更复杂
- 仅测试了文本域的攻击，多模态代理系统可能有额外的攻击面
- 未提出有效的防御方案，主要是暴露问题

## 相关工作与启发
- **vs 传统提示注入**: 传统注入是单点恶意提示，联合攻击中没有任何单点是恶意的
- **vs 多跳传播攻击 (Tan et al., 2024)**: 传播攻击传递单个恶意指令，联合攻击需要两个无害组件的对齐
- **vs IPIGuard**: IPIGuard 限制间接指令在工具依赖中传播，但联合攻击不走工具通道

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 联合激活概念新颖，暴露了多智能体系统的结构性安全盲区
- 实验充分度: ⭐⭐⭐⭐ 多拓扑、多骨干模型、四象限评估设计严谨
- 写作质量: ⭐⭐⭐⭐ 威胁模型形式化清晰，数学描述精确

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks](../../ACL2025/llm_agent/agents_under_siege_breaking_pragmatic_multi-agent_llm_systems_with_optimized_pro.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)

<!-- RELATED:END -->
