---
title: >-
  [论文解读] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems
description: >-
  [ICLR 2026][多智能体系统] 提出 SupervisorAgent，一个轻量级的实时自适应监督框架，通过无 LLM 的自适应过滤器在关键交互节点主动干预（纠错、指导、观察净化），在 GAIA 基准上将 Smolagent 的 token 消耗降低 29.68% 而不损失成功率。
tags:
  - ICLR 2026
  - 多智能体系统
  - Token 效率
  - 运行时监督
  - 自适应过滤
  - 错误纠正
---

# Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems

**会议**: ICLR 2026  
**arXiv**: [2510.26585](https://arxiv.org/abs/2510.26585)  
**代码**: 无  
**领域**: 模型压缩 / 多智能体效率  
**关键词**: 多智能体系统, Token 效率, 运行时监督, 自适应过滤, 错误纠正

## 一句话总结

提出 SupervisorAgent，一个轻量级的实时自适应监督框架，通过无 LLM 的自适应过滤器在关键交互节点主动干预（纠错、指导、观察净化），在 GAIA 基准上将 Smolagent 的 token 消耗降低 29.68% 而不损失成功率。

## 研究背景与动机

- 多智能体系统（MAS）在复杂任务上表现出色，但面临**效率与鲁棒性悖论**：
    - **错误传播**：单个幻觉信息污染整个推理链的下游代理
    - **低效行为**：代理进入重复操作循环、选择不必要的复杂路径
    - **上下文膨胀**：冗长的工具返回（如原始 HTML）充斥上下文窗口
- 现有方法主要关注**事后归因**（post-hoc failure attribution），缺乏**实时主动干预**

## 方法详解

### 整体框架：监督型多智能体系统（SMAS）

在原有 MAS 基础上增加一个元级控制代理——SupervisorAgent，实时监控三类高风险交互：

1. **Agent-Agent 交互**：代理间通信/委托，易传播幻觉信息
2. **Agent-Tool 交互**：外部工具调用，返回不准确/无关/过时数据
3. **Agent-Memory 交互**：记忆检索，可能使用过期或有缺陷的历史信息

### 自适应过滤器（When to Supervise）

**无 LLM** 的轻量级启发式过滤器，仅在关键节点触发监督：

- **错误发生** $c_{error}$：显式错误（工具调用失败、代码执行错误）
- **低效行为** $c_{inefficient}$：重复操作循环（如反复 page_down）
- **观察过长** $c_{excessive}$：工具返回超长内容（如原始 HTML）

### 上下文窗口

$$\mathcal{W} = (N, Q_g, Q_l, T_l, S)$$

- $N$: 被监督代理名称
- $Q_g, Q_l$: 全局目标和局部任务
- $T_l$: 局部行动轨迹
- $S$: 最新交互摘要
- 扩展版 $\mathcal{W}_{ext}$ 包含全局轨迹 $T_g$ 用于诊断系统级低效

### 多级干预动作空间（How to Supervise）

| 动作 | 强度 | 触发条件 | 描述 |
|------|------|---------|------|
| **approve** | 最低 | $c_{inefficient}$ | 允许有效的重复行为继续 |
| **provide_guidance** | 中等 | $c_{error}, c_{inefficient}$ | 追加引导提示纠正推理路径 |
| **correct_observation** | 高 | $c_{error}, c_{excessive}$ | 替换/净化原始观察内容 |
| **run_verification** | 最高 | $c_{error}$ | 调用验证子代理做外部事实检查 |

### 核心设计原则

- **非侵入式**：不修改基础代理架构
- **自适应**：不对每个交互都监督，仅在高风险节点触发
- **记忆增强**：SupervisorAgent 拥有比任何单个代理更全面的系统状态视角

## 实验关键数据

### GAIA 基准主实验

| 方法 | 平均准确率 | 平均 Token (K) | L2 Token (K) |
|------|----------|--------------|-------------|
| CodeAgent | 40.00 | 120.40 | — |
| Smolagent (pass@1) | — | 基线 | 基线 |
| **SMAS (pass@1)** | 持平 | **-29.68%** | **-35%** |

### GAIA Level 2 详细分析

| 指标 | Smolagent | SMAS | 改善 |
|------|----------|------|------|
| Token 消耗 | 基线 | -35% | 显著 |
| 方差 | 基线 | -63% | 大幅降低 |
| 步骤数 (案例) | 基线 | -43% | 显著 |

### 跨基准验证

| 基准 | 领域 | Token 减少 | 准确率变化 |
|------|------|----------|----------|
| HumanEval | 代码生成 | **-23.74%** | +提升 |
| MBPP | 代码生成 | 显著减少 | 持平/提升 |
| AIME 2024 | 数学推理 | 减少 | 持平 |
| GSM8k-Hard | 数学推理 | 减少 | 持平 |
| DROP | 问答 | 减少 | 持平 |

### 关键发现

1. HumanEval 上实现 23.74% token 减少的同时准确率还提升了
2. SupervisorAgent 跨 GPT-4.1、Gemini-2.5-pro、Qwen3 系列均有效
3. 自适应过滤器有效控制了监督开销，避免了对每个交互的冗余检查
4. 案例分析显示一次成功的监督干预可减少 70%+ 的 token 消耗

## 亮点与洞察

- **实时主动干预 vs 事后分析**：从 reactive 到 proactive 的范式转变
- **Pareto 改善**：降低 token 消耗的同时不损失（甚至提升）成功率
- **无 LLM 过滤器**：关键创新在于用简单启发式替代 LLM 来决定"何时监督"
- **与现有方法正交**：可叠加到任意现有 MAS 框架上
- **方差大幅减少**：更稳定可靠的系统行为

## 局限性

- 自适应过滤器基于预定义的启发式规则，可能错过某些新类型的高风险交互
- SupervisorAgent 本身的 LLM 调用也有成本，需要权衡监督收益与监督开销
- 主要在 Smolagent 框架上验证，在其他 MAS 框架上的适配可能需要调整
- 对于不使用工具的纯对话任务，框架的适用性有限

## 相关工作

- 失败归因：Aegis、AgenTracer 等事后分析方法
- 效率优化：AgentDropout（剪枝代理）、MetaAgent（设计时优化拓扑）
- 上下文压缩：观察摘要/蒸馏

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 运行时监督的概念新颖，非侵入式设计实用
- **技术深度**: ⭐⭐⭐ — 方法相对直觉，技术复杂度适中
- **实验充分性**: ⭐⭐⭐⭐⭐ — 6 基准 × 多基础模型 × 详细案例分析
- **实用性**: ⭐⭐⭐⭐⭐ — 直接可部署，对降低 MAS 运营成本有重要价值

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Is Your LLM-Based Multi-Agent a Reliable Real-World Planner? Exploring Fraud Detection in Travel Planning](../../ICML2025/social_computing/is_your_llm-based_multi-agent_a_reliable_real-world_planner_exploring_fraud_dete.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/social_computing/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ICLR 2026\] Scalable Multi-Task Low-Rank Model Adaptation](scalable_multi-task_low-rank_model_adaptation.md)
- [\[CVPR 2026\] Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning](../../CVPR2026/social_computing/revisiting_unknowns_towards_effective_and_efficient_open-set_active_learning.md)
- [\[ICLR 2026\] Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI](functional_embeddings_enable_aggregation_of_multi-area_seeg_recordings_over_subj.md)

<!-- RELATED:END -->
