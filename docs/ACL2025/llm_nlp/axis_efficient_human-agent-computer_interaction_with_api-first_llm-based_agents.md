---
title: >-
  [论文解读] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents
description: >-
  [ACL 2025][LLM/NLP][API-first] 提出 AXIS 框架，通过让 LLM Agent 优先调用 API 而非模拟人类 UI 操作来完成应用任务，在 Microsoft Word 实验中将任务完成时间缩短 65-70%，认知负荷降低 38-53%，同时保持 97-98% 的准确率。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "API-first"
  - "UI Agent"
  - "技能探索"
  - "认知负荷"
  - "Agent OS"
  - "Human-Agent-Computer Interaction"
---

# AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents

**会议**: ACL 2025  
**arXiv**: [2409.17140](https://arxiv.org/abs/2409.17140)  
**代码**: 未公开  
**领域**: LLM/NLP  
**关键词**: API-first, UI Agent, 技能探索, 认知负荷, Agent OS, Human-Agent-Computer Interaction

## 一句话总结

提出 AXIS 框架，通过让 LLM Agent 优先调用 API 而非模拟人类 UI 操作来完成应用任务，在 Microsoft Word 实验中将任务完成时间缩短 65-70%，认知负荷降低 38-53%，同时保持 97-98% 的准确率。

## 研究背景与动机

**领域现状**: 基于多模态 LLM 的 UI Agent（如 UFO）能够直接操作应用界面完成用户任务，成为研究热点。然而现有应用的 UI 是为**人类**设计的，不适合 Agent 高效操作。

**现有痛点**:
   - **高延迟**: 每步 UI 交互需要一次 LLM 推理，多步操作累积延迟严重
   - **低可靠性**: 长链 UI 交互中 LLM 易产生幻觉，错误逐步累积
   - **泛化困难**: LLM 对预训练阶段未见过的 UI 控件难以正确交互

**核心矛盾**: 现有 UI 是 HCI（人-计算机交互）范式的产物，用于 HACI（人-Agent-计算机交互）范式效率低下。类比蒸汽时代到电力时代的工厂改造——不能只替换动力源，需要重新设计整个流程。

**本文目标**: 如何让 LLM Agent 高效、可靠地完成应用操作任务。

**切入角度**: API 调用比 UI 操作更高效——一次 API 调用可替代多步 UI 交互（如"插入 2×2 表格"从 UI 的"Insert→Table→2×2"三步变为一次 API 调用）。

**核心idea**: Agent 应**优先**调用 API，仅在 API 不可用时才退回 UI 操作；框架应能自动探索应用并构建新 API。

## 方法详解

### 整体框架

AXIS 系统分三阶段：(1) 轨迹收集——Agent 在应用中探索并记录交互轨迹；(2) 技能生成——从轨迹中提取技能、翻译为 API 代码；(3) 技能验证——通过静态和动态测试确保技能可靠。

### 关键设计

1. **技能 (Skill) 定义**: 每个技能包含描述、代码和使用示例，按代码成分分为 5 类——原子 UI 技能、原子 API 技能、复合 UI 技能、复合 API 技能、API-UI 混合技能。技能支持嵌套调用，形成层次结构。

2. **轨迹收集 (Stage I)**: 

    - **Follower 模式**: Agent 按照应用帮助文档中的分步指令执行任务，严格遵循指导
    - **Explorer 模式**: Agent 利用 LLM 的头脑风暴能力自主探索应用功能。为增加状态多样性，采用随机初始状态、纵横探索策略（纵向深入子菜单/横向切换功能区）和三级技能水平（对应 Microsoft Office Specialist 认证课程）

3. **技能生成 (Stage II)**: 三个 Agent 协作

    - **Monitor**: 审查技能库，从轨迹中提取有意义的片段，整合为自然语言技能洞察
    - **Generator**: 将技能洞察转化为可执行代码（原始仍含大量 UI 操作）
    - **Translator**: 连接 RAG 模块，参考应用文档和现有技能库，将 UI 操作代码翻译为 API 调用（"UI→API 翻译"）

4. **技能验证 (Stage III)**:

    - **静态验证**: 检查代码结构兼容性（参数、方法调用、依赖技能）
    - **动态验证**: Validator 生成多种测试输入，Evaluator 检查最终状态，确保技能在真实环境中可泛化

### API-First 策略

Agent 在执行任务时，优先从技能库中查找可用的 API 技能；如果技能可以用 API 或 UI 两种方式实现，**仅保留 API 版本**；仅在缺乏对应 API 时才退回到 UI 交互。

## 实验与关键数据

### 可行性研究 (Table 1-2)

在 Microsoft Word 上探索获得 73 个技能（44 个层次-1，24 个层次-2，5 个层次-3/4），随后在 50 个 Word 任务上评估：

| 指标 | UI Agent (UFO) | AXIS |
|------|----------------|------|
| 平均耗时 (s) | 59.5 | **29.9** |
| 成功率 (%) | 52.0 | **84.0** |
| 平均步数 | 3.2 | **2.0** |
| 平均成本 ($) | 0.4 | **0.2** |

- API 调用率：AXIS 55.7% vs UI Agent 8.1%
- 高级 API 使用率（层次≥2）：AXIS 23.1%

### 用户研究 (Tables 3-5)

20 名参与者在 L1（低难度）和 L2（高难度）任务中对比手动操作、UI Agent 和 AXIS：

**效率**:

| 指标 | 手动 | UI Agent | AXIS |
|------|------|----------|------|
| L1 时间 (s) | 61.8 | 104.6 | **18.2** |
| L2 时间 (s) | 167.6 | 155.5 | **57.1** |
| L1 成功率 (%) | 100 | 75 | **98.3** |
| L2 成功率 (%) | 97.5 | 45 | **95** |

**认知负荷 (NASA-TLX)**:

| 指标 | 手动 (L2) | Agent (L2) |
|------|-----------|------------|
| 心理需求 | 70.0 | **7.5** |
| 身体需求 | 57.5 | **6.3** |
| 挫败感 | 62.5 | **10.0** |

- AXIS 在所有主观偏好维度（流畅度、可靠性、速度感知）上优于 UI Agent
- 复杂任务中 AXIS 与人类决策一致性更高

## 亮点与洞察

1. **范式转变的洞察**: 从 HCI 到 HACI 的转变不应只是"给 UI 加个 Agent"，而需要重新设计交互模式——API-first 是关键
2. **自动技能发现**: Agent 可以自行探索应用功能并构建可复用技能库，无需人工定义 API
3. **UI→API 翻译**: 通过 RAG 检索应用文档实现自动"升级"UI 操作为 API 调用，巧妙且实用
4. **技能嵌套层次**: 从原子操作组合出复杂技能的层次结构设计，类似编程中的函数抽象
5. **全面的用户研究**: 不仅评估技术指标，还通过 NASA-TLX 等标准化量表评估认知负荷，偏应用导向

## 局限性

1. 目前主要依赖 Python API，无原生 Python 接口的应用难以支持
2. 探索过程的稳定性和效率仍需优化
3. 仅在 Microsoft Word 上验证，泛化到其他应用（如 Photoshop、Excel）未经测试
4. 技能库的维护和更新（应用版本变化后API变动）尚未讨论
5. 安全性考量不足——Agent 直接执行 API 操作可能带来权限和安全风险

## 相关工作

- **UI Agent**: AppAgent、UFO、CogAgent 等利用 MLLM 操作应用 UI
- **Agent OS**: Apple Intelligence、Microsoft Copilot、Agent OS 概念
- **UI 设计**: MUD 利用 LLM 挖掘 UI 数据，SimUser 模拟用户反馈

## 评分

⭐⭐⭐⭐ — 洞察深刻（API-first 范式），实验扎实（包含完整用户研究），实践价值高。不足在于仅在 Word 上验证，且框架复杂度较高（三阶段多 Agent 协作）。对 Agent OS 方向有启发意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Game Development as Human-LLM Interaction](game_development_as_human-llm_interaction.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](agentdropout_dynamic_agent_elimination_for_token-efficient_and_high-performance_.md)
- [\[ACL 2025\] Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition](sample-efficient_human_evaluation_of_large_language_models_via_maximum_discrepan.md)
- [\[ACL 2025\] BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving](bfs-prover_scalable_best-first_tree_search_for_llm-based_automatic_theorem_provi.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)

</div>

<!-- RELATED:END -->
