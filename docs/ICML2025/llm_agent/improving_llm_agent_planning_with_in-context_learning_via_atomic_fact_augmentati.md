---
title: >-
  [论文解读] Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search
description: >-
  [ICML 2025][LLM Agent][LLM Agent] 提出 LWM-Planner，从交互轨迹中提取"原子事实"增强 LLM 世界模型模拟，结合递归前瞻搜索实现纯 in-context 的 Agent 规划改进，在 ALFWorld 等任务上显著优于 ReAct 和 Reflexion。
tags:
  - ICML 2025
  - LLM Agent
  - Atomic Facts
  - Lookahead Search
  - in-context learning
  - world model
---

# Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search

**会议**: ICML 2025  
**arXiv**: [2506.09171](https://arxiv.org/abs/2506.09171)  
**代码**: 待确认  
**领域**: llm_agent  
**关键词**: LLM Agent, Atomic Facts, Lookahead Search, in-context learning, world model

## 一句话总结

提出 LWM-Planner，从交互轨迹中提取"原子事实"增强 LLM 世界模型模拟，结合递归前瞻搜索实现纯 in-context 的 Agent 规划改进，在 ALFWorld 等任务上显著优于 ReAct 和 Reflexion。

## 研究背景与动机

### LLM Agent 规划的核心挑战
- 无法高效利用历史经验（靠整条轨迹塞入上下文效率低）
- 缺乏显式世界模型来模拟未来状态

### 现有方法不足
- ReAct：无跨 episode 学习能力
- Reflexion：只产出高层建议，不够结构化
- RAP(MCTS)：需要环境交互展开搜索树，代价大

### 核心洞察
LLM 拥有大量关于世界动态的先验知识，如果能从经验中提取简洁的"原子事实"（如"物体 X 在容器 Y 中"），显著提升模拟和规划能力。

## 方法详解

### 整体框架：LWM-Planner
维护短期交互历史 + 长期原子事实集合，通过递归前瞻搜索选择动作。

### 关键设计 1：原子事实提取
每 episode 结束后从轨迹中提取形如"obstacle_at(3,0)"的最小知识单元，加入长期事实库。

### 关键设计 2：LLM 作为潜在世界模型
LLM 同时充当三个角色（均接收原子事实增强）：
- **Action Proposer**：生成候选动作
- **World Model**：预测下一状态+奖励+终止
- **Value Estimator**：估计叶节点长期价值

### 关键设计 3：递归前瞻搜索
- 深度限制 d=3，分支因子 b=4
- $Q(o,a) = r' - \lambda_{step} + \gamma \cdot \hat{V}(o')$
- 结果缓存减少重复 LLM 调用，temperature=0 保证确定性

### 理论动机
框架形式化为基于事实的状态抽象 MDP，性能损失由 $\epsilon_{sim}$、$\delta_{model}$、$\epsilon_{plan}$ 三误差控制。

## 实验关键数据

### 主实验（归一化累积回报）

| 方法 | TextFrozenLake | CrafterMini | ALFWorld-A | ALFWorld-B | ALFWorld-C |
|------|---------------|-------------|------------|------------|------------|
| LWM-Planner | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| ReAct + FEC | 89.6 | 99.9 | 22.0 | 67.7 | 54.4 |
| ReAct | -165.7 | 86.7 | 59.1 | 55.9 | 64.1 |

### 步数效率（每次成功所需步数）

| 方法 | TextFrozenLake | CrafterMini | ALFWorld-A |
|------|---------------|-------------|------------|
| LWM-Planner | **6.0** | **46.5** | **8.4** |
| ReAct + FEC | — | 41.4 | 14.6 |
| ReAct | — | 50.7 | 24.7 |

### 关键发现
1. LWM-Planner 在所有环境上达到最高回报且步数接近最优
2. 消融 ReAct+FEC 说明事实提取有效，前瞻搜索进一步提升
3. 完全 in-context learning，无需权重更新
4. 原子事实随 episode 积累，Agent 持续自我改进

## 亮点与洞察

1. "原子事实"概念精炼优雅：比整条轨迹检索高效，比高层反思更精确。
2. LLM 同时当 world model + value function + policy 的统一设计。
3. 搜索+经验的组合比纯反思（Reflexion）或纯搜索都更鲁棒。
4. 纯 in-context 学习意味着对新环境的适应完全不需要训练。

## 局限与展望

1. LLM 推理调用量大，每步多次 LLM 调用成本高。
2. 原子事实质量完全依赖 LLM 反思能力。
3. 实验环境相对简单，更复杂真实环境效果待验证。
4. 事实集合的长期增长管理可能成为瓶颈。

## 相关工作与启发

- **vs ReAct (Yao et al. 2023)**：ReAct 交替推理和执行但无世界模型；本文用原子事实增强 LLM 做多步仿真
- **vs Reflexion (Shinn et al. 2023)**：Reflexion 产出高层反思建议，本文提取结构化原子事实用于 world model，信息粒度更细
- **vs RAP (Kagaya et al. 2024)**：RAP 检索完整历史轨迹做 MCTS，上下文长且冗余；本文用精炼事实做 LLM 前瞻搜索，更高效
- **vs Dyna 架构 (Sutton 1990)**：本文是 Dyna 的 LLM 实现——用事实集替代传统参数化世界模型，用 LLM 仿真做规划
- **启发**：原子事实可与符号推理系统结合，或扩展到多 Agent 协作场景中的共享世界知识
- **潜在方向**：将原子事实与 RAG 结合，在大规模环境中做分层事实管理

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 原子事实 + LLM 世界模型 + 前瞻搜索的三位一体设计高度原创
- 实验充分度: ⭐⭐⭐⭐ TextFrozenLake 和 ALFWorld 验证充分，但更多复杂环境待测
- 写作质量: ⭐⭐⭐⭐ 理论动机和算法描述清晰，Dyna 类比自然
- 价值: ⭐⭐⭐⭐⭐ 为 LLM Agent 的在线纯上下文学习提供了系统性解决方案

<!-- RELATED:START -->

## 相关论文

- [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](../../ICLR2026/llm_agent/agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)
- [FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models](../../ACL2025/llm_agent/fact_audit_factcheck.md)
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](../../ACL2025/llm_agent/pasa_an_llm_agent_for_comprehensive_academic_paper_search.md)
- [ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](../../ICLR2026/llm_agent/tooltree_efficient_llm_agent_tool_planning_via_dual-feedback_monte_carlo_tree_se.md)
- [LLM Agents for Knowledge Discovery in Atomic Layer Processing](../../NeurIPS2025/llm_agent/llm_agents_for_knowledge_discovery_in_atomic_layer_processing.md)

<!-- RELATED:END -->
