---
title: >-
  [论文解读] Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading
description: >-
  [NeurIPS 2025][LLM Agent][金融智能体] 提出 FinAgent 编排框架，将传统算法交易系统的各组件映射为 AI 智能体（规划器、编排器、Alpha/风控/组合/回测/执行/审计/记忆智能体），使用 MCP 协议进行控制通信、A2A 协议进行智能体间通信，在股票和 BTC 交易任务上验证了可行性。
tags:
  - NeurIPS 2025
  - LLM Agent
  - 金融智能体
  - 算法交易
  - 多智能体编排
  - MCP协议
  - 量化交易
---

# Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading

**会议**: NeurIPS 2025  
**arXiv**: [2512.02227](https://arxiv.org/abs/2512.02227)  
**代码**: [GitHub](https://github.com/)  
**领域**: llm_agent  
**关键词**: 金融智能体, 算法交易, 多智能体编排, MCP协议, 量化交易

## 一句话总结

提出 FinAgent 编排框架，将传统算法交易系统的各组件映射为 AI 智能体（规划器、编排器、Alpha/风控/组合/回测/执行/审计/记忆智能体），使用 MCP 协议进行控制通信、A2A 协议进行智能体间通信，在股票和 BTC 交易任务上验证了可行性。

## 研究背景与动机

传统算法交易（Algorithmic Trading, AT）系统的开发需要专业团队花费数年时间，流程包括：数据处理 → 信号提取 → 组合管理 → 执行 → 评估。本文的核心动机是：

**AI 智能体技术的成熟**：ReAct 推理-行动框架、工具使用、生成式智能体、反思与记忆、多智能体协调等技术已展现出巨大潜力

**金融市场的特殊挑战**：时序动态性强、信噪比极低，是 AI 智能体的"高难度试炼场"

**民主化金融智能**：目标是让普通用户也能获得专业级交易策略能力

**核心思路**：将 AT 系统中的每个组件一一映射为对应的 AI 智能体，构建端到端的编排框架。

## 方法详解

### 整体框架

FinAgent 框架包含多个智能体池（Agent Pool），每个池负责一个交易阶段：

| 智能体池 | 功能 |
|:---|:---|
| **Planner** | 制定整体交易计划 |
| **Orchestrator** | 编排调度各智能体池 |
| **Data Agents** | 从多数据源获取、清洗、对齐数据 |
| **Alpha Agents** | 基于文献提出因子结构（不接触评估期数据） |
| **Risk Agents** | 计算风险暴露、设定约束（集中度、波动率、回撤） |
| **Portfolio Agents** | 测试多空规则、资金和换手率约束 |
| **Execution Agents** | 将权重转化为订单、考虑滑点和交易成本 |
| **Backtest Agents** | Walk-forward 回测、归因分析 |
| **Audit Agents** | 验证收益曲线、回撤、贡献 |
| **Memory Agent** | 记录状态、提示词、工具调用和决策 |

### 通信协议设计

**MCP（Model Context Protocol）用于控制消息**：
- 编排器通过 MCP 向各智能体池发送任务描述（节点类型、任务ID、输入模式、策略标志、超时、重试预算）
- 等待回复（确认、状态、日志、产物ID）
- 心跳监控和完成追踪

**A2A（Agent-to-Agent）协议用于智能体间通信**：
- 每个智能体池通过 A2A 与记忆智能体交互（读取历史上下文、上传日志和关键结果）
- 消息类型：ask, tell, propose, confirm
- 带角色标签和上下文 ID
- 定时共享进度，失败可由同伴接管或编排器重新分配

### 关键设计：数据隔离

**核心原则**：LLM 智能体永远不接触评估窗口的数据。具体来说：
- Alpha Agents 的因子设计仅基于已发表文献，不访问评估期数据
- 所有数值信号构造和收益映射由工具模块（非 LLM）完成
- 实现收益和性能指标对 LLM 智能体隐藏
- 信号诊断（如 rank-IC）由工具模块计算

### BTC 交易策略细节

**特征工程**：超过 100 个输入特征，涵盖价格、成交量、波动率、趋势
- 基础价格特征：5分钟平滑收益、15分钟波动率、60分钟 EWM 波动率、VWAP偏差
- 技术指标：RSI(14/30)、MACD、布林带
- 多时间尺度动量：1/3/5/10/15/30/60/240 分钟
- 波动率特征：GARCH 风格项、波动率聚类

**预测模型**：XGBoost 回归，预测下一分钟收益 $r_{t+1}$
- 300 棵树，最大深度 6，学习率 0.08
- 滚动 walk-forward 训练，每 24 小时重训
- 特征重要性筛选：保留 top 70%

**信号混合**：模型预测与价格行为规则的加权组合

$$w_{\text{model}}(q_t) = \begin{cases} 0.10, & q_t < 0.05 \\ 0.20, & 0.05 \leq q_t < 0.10 \\ 0.40, & q_t \geq 0.10 \end{cases}$$

**市场状态识别**：强趋势、突破、横盘、高波动，每种状态使用不同信号权重

## 实验关键数据

### 主实验：交易绩效对比

| 指标 | 本方法（股票） | SPY | QQQ | 等权组合 | 本方法（BTC） | BTC Buy&Hold |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 总收益 ↑ | 20.42% | 16.60% | 21.59% | 47.46% | 8.39% | 3.80% |
| 年化收益 ↑ | 31.08% | 25.07% | 32.94% | 76.07% | - | - |
| 波动率 ↓ | **11.83%** | 13.49% | 18.38% | 22.54% | **24.23%** | 25.82% |
| Sharpe ↑ | 2.63 | 1.86 | 1.79 | 3.37 | 0.378 | 0.170 |
| 最大回撤 ↑ | **-3.59%** | -8.89% | -14.13% | -16.21% | **-2.80%** | -5.26% |

### BTC 策略详细绩效

| 指标 | 智能体策略 | Buy&Hold |
|:---|:---:|:---:|
| 累计收益 | 8.39% | 3.80% |
| 超额收益 | +4.59pp | - |
| Sharpe | 0.380 | 0.168 |
| Calmar | 166.06 | 23.30 |
| 最大回撤 | -2.80% | -5.26% |
| 胜率 | 64.7% | 58.8% |
| 交易次数 | 17 次（1.04次/天） | - |
| 平均持仓 | 16.07 小时 | - |

### 关键发现

1. **风险控制优势明显**：本方法在股票和 BTC 上均实现了最低波动率和最小回撤
2. **股票实验中等权基准实际收益更高**（47.46% vs 20.42%），说明在强势市场中保守策略可能不如满仓持有
3. **BTC 实验更具说服力**：在 17 天的高频交易中，超额收益 +4.6pp，回撤仅为 Buy&Hold 的一半
4. **交易频率适中**：BTC 策略平均每天仅 1.04 次交易，不是超高频策略

## 亮点与洞察

1. **AT → Agentic Trading 的映射思路清晰**：将成熟的算法交易流程中的每一环映射为智能体，既继承了 AT 的结构化优势，又引入了智能体的灵活性
2. **MCP + A2A 双协议设计合理**：MCP 用于编排器的控制面，A2A 用于智能体的数据面，职责分离
3. **数据隔离设计**是亮点中的亮点：防止 LLM 通过评估期数据"作弊"，保证了回测的公正性
4. **Memory Agent 的设计**使系统具有可审计性和状态持久性
5. **端到端框架可复用性强**：同一 DAG 拓扑适用于股票和 BTC，仅数据源和调度不同

## 局限与展望

1. **回测时间窗口过短**：股票仅 8 个月（04-12/2024），BTC 仅 17 天，统计显著性不足
2. **股票实验不如等权基准**：在 2024 年科技股强势行情中，20.42% 对比等权 47.46%，缺少对不同市场环境的鲁棒性分析
3. **股票宇宙过小**：仅 7 只超大盘科技股，不具代表性
4. **缺少与其他智能体交易系统的直接对比**（如 TradingAgents、AI Hedge Fund 等）
5. **LLM 幻觉风险**：Alpha Agent 基于文献生成因子结构，可能产生不合理因子
6. **实盘验证缺失**：所有结果均为模拟回测，未涉及实盘交易的延迟、流动性等问题
7. **成本分析缺失**：大量 LLM API 调用的成本可能不菲

## 相关工作与启发

- **TradingAgents**（24.8K stars）、**AI Hedge Fund**（42.3K stars）：社区热度很高的开源智能体交易项目
- **FinGPT**、**FinMem**：金融领域 LLM 和记忆机制的基础工作
- **MCP 协议**：Anthropic 推出的模型上下文协议，本文是较早将其应用于金融领域的工作

## 评分

- ⭐⭐⭐ (3/5)
- **创新性** ⭐⭐⭐：框架设计思路清晰，但核心是将已有组件组合
- **实验充分性** ⭐⭐：回测时间短、宇宙小、缺少基线对比，实验说服力不足
- **写作质量** ⭐⭐⭐⭐：系统架构描述详细，BTC 策略的技术细节特别充分
- **实用价值** ⭐⭐⭐⭐：框架设计有参考意义，MCP/A2A 协议选择有实践指导
- **理论深度** ⭐⭐：偏工程实践，理论分析较少

<!-- RELATED:START -->

## 相关论文

- [A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](../../ACL2025/llm_agent/agentic_reasoning_tools.md)
- [Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)
- [Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents](agentic_plan_caching_test-time_memory_for_fast_and_cost-efficient_llm_agents.md)
- [Agentic NL2SQL to Reduce Computational Costs](agentic_nl2sql_to_reduce_computational_costs.md)

<!-- RELATED:END -->
