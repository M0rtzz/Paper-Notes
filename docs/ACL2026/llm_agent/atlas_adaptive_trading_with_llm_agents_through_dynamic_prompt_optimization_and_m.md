---
title: >-
  [论文解读] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination
description: >-
  [ACL 2026][LLM Agent][LLM交易智能体] 提出 ATLAS 多智能体金融交易框架和 Adaptive-OPRO 提示优化方法，通过专业化分析师智能体准备异构市场信息，并基于延迟噪声反馈动态优化中央交易智能体的指令提示，在多种市场波动环境中显著超越基线。
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "LLM交易智能体"
  - "提示优化"
  - "多智能体协作"
  - "金融决策"
  - "自适应策略"
---

# ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination

**会议**: ACL 2026  
**arXiv**: [2510.15949](https://arxiv.org/abs/2510.15949)  
**代码**: 待发布  
**领域**: LLM Agent / Finance  
**关键词**: LLM交易智能体, 提示优化, 多智能体协作, 金融决策, 自适应策略

## 一句话总结

提出 ATLAS 多智能体金融交易框架和 Adaptive-OPRO 提示优化方法，通过专业化分析师智能体准备异构市场信息，并基于延迟噪声反馈动态优化中央交易智能体的指令提示，在多种市场波动环境中显著超越基线。

## 研究背景与动机

**领域现状**: LLM 在金融决策领域展现出处理多源数据和推理复杂场景的潜力，但从能力到可靠交易系统的转化面临重大挑战。

**现有痛点**: (1) 异构信息源（技术指标、新闻、基本面）的系统化整合缺乏统一框架；(2) 在延迟且噪声的奖励信号下，静态决策策略不足以应对市场动态变化；(3) 现有方法通常使用手工提示，无法适应不同市场环境。

**核心矛盾**: 金融交易本质上是序列决策问题，决策之间存在时序耦合，奖励信号延迟到达——但现有提示优化方法（如 OPRO）假设即时反馈和独立实例。

**本文目标**: 构建统一的 LLM 交易智能体框架，解决信息整合和行为适应两大核心问题。

**切入角度**: 将提示优化从单轮即时反馈扩展到序列决策的延迟噪声反馈场景。

**核心 idea**: Adaptive-OPRO——将 OPRO 的元优化思想适配到交易场景，通过滚动评估窗口和模板分离实现稳定的提示迭代优化。

## 方法详解

### 整体框架

ATLAS 包含三个核心组件：(1) 市场智能管线（Market Intelligence Pipeline）——三个专业分析师智能体分别处理技术、新闻和基本面信息；(2) 决策与执行层——中央交易智能体（CTA）生成订单并在 StockSim 模拟器中执行；(3) 反馈机制——Adaptive-OPRO 基于执行反馈动态优化 CTA 的指令提示。

### 关键设计

1. **市场智能管线（三专家架构）**:

    - 功能：将异构信息源结构化为一致的决策输入
    - 核心思路：Market Analyst 生成多时间尺度技术摘要（2年/6月/3月），News Analyst 将新闻聚合为结构化字段（情感、关键事件、市场相关性），Fundamental Analyst 从财报和企业事件中提取实质变化
    - 设计动机：信息准备与决策分离，每个分析师专注于特定模态

2. **Adaptive-OPRO 提示优化**:

    - 功能：基于延迟噪声反馈动态更新交易指令
    - 核心思路：维护指令提示 $P_t$ 和优化历史 $\mathcal{H} = \{(P_i, s_i)\}$，每 $K=5$ 个交易日评估一次，通过优化器 LLM 生成新指令 $P_{t+1} = U(M, \mathcal{H}, s_t, \text{summary})$；得分映射 $s = \text{clip}_{[0,100]}(50 + 250 \cdot \text{ROI})$
    - 设计动机：原始 OPRO 假设即时反馈，无法处理交易中的信用分配和延迟奖励问题

3. **模板分离稳定性机制**:

    - 功能：防止提示更新破坏运行时接口
    - 核心思路：将提示分为 (a) 可编辑的静态指令（策略、优先级、约束）和 (b) 不可编辑的动态运行时内容（状态、观测、工具输出），仅允许编辑静态部分
    - 设计动机：序列系统中提示更新可能意外破坏占位符或输出格式，需要强制编辑局部性

### 损失函数 / 训练策略

非传统训练，而是在线提示优化。每个评估窗口（5个交易日）后计算 ROI 并映射为 [0,100] 分数，优化器 LLM 诊断失败模式、提出修订、总结变更并预测行为影响。候选提示仅在保持模板完整性时被接受。

## 实验关键数据

### 主实验

| 模型 | 方法 | ROI(%) ↑ | Sharpe ↑ | Max DD(%) ↓ | Win Rate(%) |
|------|------|---------|---------|------------|-------------|
| LLaMA-3.3-70B | Baseline | -9.19±1.54 | -0.091 | 16.90 | 30.28 |
| LLaMA-3.3-70B | Adaptive-OPRO | **-6.16±2.08** | **-0.066** | **14.05** | **54.36** |
| GPT-o4-mini | Baseline | -1.30±1.71 | -0.017 | 9.68 | 29.17 |
| GPT-o4-mini | Adaptive-OPRO | **9.06±0.73** | **0.094** | 11.48 | **65.28** |
| GPT-o3 | Baseline | -6.11 | -0.080 | 11.58 | 42.59 |
| Claude Sonnet 4 | Adaptive-OPRO | **0.35±1.78** | **0.008** | 14.76 | 43.45 |
| Buy & Hold | - | -8.59 | -0.071 | 20.45 | 0.00 |

### 消融实验

| 对比 | 发现 |
|------|------|
| Baseline vs Reflection | Reflection 方法不稳定，部分模型上反而更差 |
| Baseline vs Adaptive-OPRO | Adaptive-OPRO 在所有模型上一致优于 Baseline |
| 不同信息模态 | 增加信息源不一定有益，取决于市场环境 |
| 高波动 vs 低波动 | Adaptive-OPRO 在高波动市场优势更明显 |

### 关键发现

- Adaptive-OPRO 在所有 LLM 家族上一致优于 Baseline 和 Reflection 方法
- GPT-o4-mini + Adaptive-OPRO 是唯一实现正 ROI（9.06%）的配置
- 额外信息模态（新闻、基本面）并非总是有益——在噪声市场中可能降低性能
- 多次运行报告（mean±std）对评估随机性至关重要

## 亮点与洞察

- Adaptive-OPRO 是 OPRO 在序列决策场景下的首次系统扩展
- 模板分离设计优雅地解决了提示优化中的接口稳定性问题
- "更多信息不一定更好"的发现有重要实践指导意义
- 订单级决策（类型、大小、时机、价格）比简单方向评分更接近真实交易

## 局限与展望

- 仅在单股票交易场景下评估，未考虑投资组合管理
- 评估窗口大小 $K=5$ 的敏感性未充分分析
- 模拟器 StockSim 可能无法完全反映真实市场微观结构
- 未来可扩展至多资产、多市场和更长投资周期

## 相关工作与启发

- OPRO（Yang et al., 2024）：原始提示优化方法，假设即时反馈
- CryptoTrade（Li et al., 2024）：整合链上/链下信号的反思式交易
- TradingAgents（Xiao et al., 2025）：结构化辩论的多智能体交易
- FINCON（Yu et al., 2024）：概念化语言强化的多智能体协作
- 本文的 Adaptive-OPRO 可推广到其他延迟反馈的序列决策场景

## 评分

- 新颖性: ⭐⭐⭐⭐ Adaptive-OPRO 将提示优化扩展到序列决策的延迟反馈场景
- 实验充分度: ⭐⭐⭐⭐ 覆盖 7 个 LLM 家族、多种市场环境、多次重复
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，公式化合理
- 价值: ⭐⭐⭐⭐ 对 LLM 金融应用和提示优化均有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[ACL 2026\] IntrAgent: An LLM Agent for Content-Grounded Information Retrieval through Literature Review](intragent_an_llm_agent_for_content-grounded_information_retrieval_through_litera.md)
- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](../../NeurIPS2025/llm_agent/mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[ACL 2026\] AdaRubric: Task-Adaptive Rubrics for Reliable LLM Agent Evaluation and Reward Learning](adarubric_task-adaptive_rubrics_for_reliable_llm_agent_evaluation_and_reward_lea.md)
- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)

</div>

<!-- RELATED:END -->
