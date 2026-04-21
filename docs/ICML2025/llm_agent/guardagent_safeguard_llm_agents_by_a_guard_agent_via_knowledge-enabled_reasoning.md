---
title: >-
  [论文解读] GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning
description: >-
  [ICML2025][LLM Agent][Agent 安全] GuardAgent 是首个"用 Agent 守护 Agent"的框架，通过将安全规则动态转化为可执行的护栏代码来检查目标 Agent 的动作是否违规，在医疗访问控制和 Web 安全控制两个新基准上分别达到 98%+ 和 83%+ 的护栏准确率。
tags:
  - ICML2025
  - LLM Agent
  - Agent 安全
  - 护栏代理
  - 代码生成执行
  - 访问控制
  - 安全策略
---

# GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning

**会议**: ICML2025  
**arXiv**: [2406.09187](https://arxiv.org/abs/2406.09187)  
**代码**: [GuardAgent 项目页](https://guardagent.github.io/)  
**领域**: llm_agent  
**关键词**: Agent 安全, 护栏代理, 代码生成执行, 访问控制, 安全策略

## 一句话总结
GuardAgent 是首个"用 Agent 守护 Agent"的框架，通过将安全规则动态转化为可执行的护栏代码来检查目标 Agent 的动作是否违规，在医疗访问控制和 Web 安全控制两个新基准上分别达到 98%+ 和 83%+ 的护栏准确率。

## 研究背景与动机

### 现有 LLM 护栏的局限
传统 LLM 护栏（如 NVIDIA NeMo Guardrails、Llama Guard 等）主要面向文本有害性检测（暴力/色情/仇恨等类别），依赖分类器对输出进行内容审核。

然而 LLM Agent 的输出不仅仅是文本：
- Web Agent 可能点击按钮、填写表单
- 医疗 Agent 可能查询数据库、修改病人记录
- 自动驾驶 Agent 可能生成轨迹指令

这些输出模态远超纯文本范畴，且安全要求高度领域相关。传统文本护栏完全无法处理"禁止未授权人员查看特定病人实验室数据"之类的需求。

### 核心矛盾
已有的 Agent 安全控制要么硬编码在目标 Agent 内部（不可迁移），要么只覆盖通用文本风险（粒度不够）。需要一种"非侵入、可迁移、可编程"的 Agent 级护栏方案。

## 方法详解

### 整体框架
GuardAgent 本身就是一个 LLM Agent，用来"看守"另一个目标 Agent：
1. 接收安全规则（纯文本描述）
2. 接收目标 Agent 的输入/输出/日志
3. 生成 action plan → 生成 guardrail code → 执行代码 → 返回通过/拒绝

### 关键设计 1：Task Plan 生成
LLM 分析安全规则和目标 Agent 的动作日志，生成结构化的检查计划。
该步骤由 ICL 驱动，从 memory module 检索相关历史示例作为上下文。

### 关键设计 2：Guardrail Code 生成与执行
将 plan 翻译成可执行的 Python 代码，代码可以调用预定义的工具函数（如数据库查询、字段比较、权限验证等）。代码执行是确定性的，避免了纯 NL 推理的不可靠性。

### 关键设计 3：Memory Module
存储之前成功处理的安全规则及其对应的 plan/code，为新规则提供 few-shot 参考。通过检索相似历史案例来引导当前任务。

### 三大优势
- 非侵入：与目标 Agent 并行部署，不修改目标 Agent 内部逻辑
- 代码驱动：比纯 NL 判断更可靠、可审计
- 无需训练：基于 ICL，直接使用现成 LLM

## 实验关键数据

### 新提出的两个基准

| 基准 | 场景 | 安全需求类型 | 规模 |
|------|------|------------|------|
| EICU-AC | 医疗 Agent（EHRAgent） | 访问控制 | 多角色权限组合 |
| Mind2Web-SC | Web Agent（SeeAct） | 安全策略 | 多网站操作类型 |

### 主实验：护栏准确率

| 方法 | EICU-AC 准确率 | Mind2Web-SC 准确率 |
|------|---------------|-------------------|
| LLM-based guardrail（纯 NL） | ~85% | ~65% |
| GuardAgent (GPT-4) | **98%+** | **83%+** |
| GuardAgent (Claude) | 97%+ | 82%+ |
| GuardAgent (GPT-3.5) | 95%+ | 78%+ |

### 关键发现
1. "Agent 守 Agent" 显著优于"模型守 Agent"（纯 NL 推理方案）
2. 代码执行的确定性是准确率提升的关键因素
3. 不同核心 LLM 都能驱动 GuardAgent，说明框架通用性
4. GuardAgent 不影响目标 Agent 的任务性能（仅做过滤）
5. Memory module 的检索质量直接影响复杂规则的处理效果

## 亮点与洞察

1. 开创性地提出"Agent 守 Agent"范式，填补了 LLM Agent 安全领域空白。
2. 代码生成 + 执行的设计巧妙：把模糊安全规则转化为确定性检查逻辑。
3. 两个新基准具有实际意义：医疗访问控制和 Web 安全策略都是真实世界高频需求。
4. 可扩展性好：新增安全规则只需更新 toolbox 和 memory，无需重新训练。
5. 对产业界的启示：任何部署 Agent 的场景都需要配套的护栏 Agent。

## 局限与展望

1. 代码生成依赖 LLM 能力，复杂规则可能生成错误代码。
2. Mind2Web-SC 上准确率低于 EICU-AC，Web 场景多样性更难覆盖。
3. 目前 toolbox 需人工预定义，自动化工具发现是潜在方向。
4. 安全规则冲突的处理未深入讨论。
5. 攻击者可能通过间接方式绕过护栏（对抗鲁棒性待评估）。

## 相关工作与启发

- 与 NeMo Guardrails、Llama Guard 的本质区别在于：GuardAgent 处理结构化动作而非纯文本。
- 与 Agent 内置安全机制相比，GuardAgent 是外置且通用的。
- 启发后续研究：可将护栏 Agent 与形式化验证结合，或研究多层级护栏体系。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（5.0/5）— 首个 Agent-guarding-Agent 框架
- 实验充分度: ⭐⭐⭐⭐☆（4.0/5）— 两个新基准，但场景可更多样
- 写作质量: ⭐⭐⭐⭐☆（4.0/5）
- 价值: ⭐⭐⭐⭐⭐（5.0/5）— Agent 安全是刚需

<!-- RELATED:START -->

## 相关论文

- [LLM Agents for Knowledge Discovery in Atomic Layer Processing](../../NeurIPS2025/llm_agent/llm_agents_for_knowledge_discovery_in_atomic_layer_processing.md)
- [Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_in_web_agents.md)
- [Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](../../ACL2025/llm_agent/agentic_reasoning_tools.md)
- [From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium](from_debate_to_equilibrium_belief-driven_multi-agent_llm_reasoning_via_bayesian_.md)

<!-- RELATED:END -->
