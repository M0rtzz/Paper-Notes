---
title: >-
  [论文解读] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution
description: >-
  [ICLR 2026][LLM Agent][language agent] 提出 Toolathlon，一个覆盖 32 个软件应用、604 个工具和 108 个任务的语言 Agent 基准，强调真实多样的环境状态和长程多步交互（平均约 20 轮工具调用），最强模型 Claude-4.5-Sonnet 仅达 38.6% 成功率。
tags:
  - ICLR 2026
  - LLM Agent
  - language agent
  - benchmark
  - MCP
  - tool calling
  - long-horizon
  - 多应用交互
  - 执行式评估
---

# The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution

**会议**: ICLR 2026  
**arXiv**: [2510.25726](https://arxiv.org/abs/2510.25726)  
**代码**: 待确认 (基于 MCP 服务器)  
**领域**: LLM Agent / Benchmark / 工具使用  
**关键词**: language agent, benchmark, MCP, tool calling, long-horizon, 多应用交互, 执行式评估

## 一句话总结
提出 Toolathlon，一个覆盖 32 个软件应用、604 个工具和 108 个任务的语言 Agent 基准，强调真实多样的环境状态和长程多步交互（平均约 20 轮工具调用），最强模型 Claude-4.5-Sonnet 仅达 38.6% 成功率。

## 研究背景与动机
**领域现状**：语言 Agent 需要在真实世界中完成跨应用、多步骤的复杂工作流，如管理日历邮件联动、监控数据库生成报告等。这要求 Agent 具备工具发现、多轮推理、状态追踪和跨系统协调的综合能力。

**现有痛点**：现有 Agent 基准存在三大不足——(1) 应用领域狭窄（多聚焦于单一工具或 API），(2) 任务过于简化（一两步调用即可完成），(3) 环境状态不真实（使用空白或极简的初始状态，而非真实软件中的复杂数据），无法充分评估 Agent 的实际部署能力。

**核心矛盾**：研究者需要可靠评估 Agent 的真实世界表现，但现有基准的"玩具级"任务与真实工作流之间存在巨大鸿沟，导致基准得分与实际能力脱节——Agent 在简单基准上表现良好，但在真实场景中频繁失败。

**MCP 的机遇**：Model Context Protocol (MCP) 的出现为构建标准化工具接口提供了基础设施。Toolathlon 基于高质量的 MCP 服务器构建工具层，部分由团队自行实现或修订，确保接口质量和一致性。

**评估可靠性**：现有基准多依赖 LLM 判断或字符串匹配来评估 Agent 输出，误差大且不可复现。需要基于程序执行的严格验证机制，确保每个任务的完成与否有确定性判定。

**核心 idea**：构建首个同时满足"多样应用覆盖 + 真实环境状态 + 长程复杂任务 + 执行式验证"四个维度的 Agent 综合基准。

## 方法详解

### 整体框架
Toolathlon 的设计围绕四个核心原则：应用多样性（Diverse Apps）、环境真实性（Realistic Setup）、任务复杂性（Long-Horizon Tasks）和评估可靠性（Execution-Based Evaluation）。

### 关键设计

1. **应用与工具覆盖**：

    - 涵盖 32 个软件应用，横跨日常办公（Google Calendar、Notion、Gmail）到专业工具（WooCommerce 电商、Kubernetes 容器编排、BigQuery 数据分析），覆盖 604 个不同的工具/API。
    - 工具层基于 MCP 服务器实现，团队对部分 MCP 服务器进行了修订或重新实现，确保接口的质量和行为一致性。
    - 设计意图：Agent 需要面对"工具发现"挑战——在海量工具中找到正确的工具组合来完成任务。

2. **真实环境状态**：

    - 区别于仅提供"功能可用"但状态空白的环境，Toolathlon 为每个任务提供来自真实软件的初始环境状态。
    - 例如：Canvas 课程中包含数十名学生的真实课程数据、真实的财务电子表格、已有订单的电商系统等。
    - 价值：Agent 需要在复杂的现有数据中查找、筛选、关联信息，而非在空白环境中简单创建。

3. **任务设计与复杂性**：

    - 108 个任务，全部手动编写或从真实工作场景中提炼，每个任务需要 Agent 与多个应用交互。
    - 平均每个任务需要约 20 轮工具调用，属于长程任务（long-horizon），要求 Agent 维持长期上下文和多步规划。
    - 任务类型涵盖：信息检索聚合、跨系统数据同步、条件判断与流程执行、报告生成等。

4. **执行式评估**：

    - 每个任务配备专用的评估脚本（evaluation script），通过检查系统状态变化（如数据库记录、文件内容、API 状态）来严格验证任务是否完成。
    - 评估是确定性的——不依赖 LLM 判断或模糊匹配，避免评估本身引入噪声。
    - 评估脚本覆盖了正确性（是否完成）和完整性（是否遗漏步骤）两个维度，部分任务有中间步骤检查点。

## 实验关键数据

### 主实验：模型成功率对比

| 模型 | 成功率(%) | 平均工具调用轮数 | 类型 | 说明 |
|------|----------|---------------|------|------|
| Claude-4.5-Sonnet | **38.6** | 20.2 | 闭源 | 最强模型 |
| GPT-4o / GPT-5 系列 | 25-35 (推测) | ~20 | 闭源 | 中等表现 |
| DeepSeek-V3.2-Exp | **20.1** | ~20 | 开源最强 | 开源权重最佳 |
| 其他开源模型 | <20 | 变化大 | 开源 | 普遍不足 |

### 不同应用类别的表现差异

| 应用类别 | 典型应用 | 模型表现趋势 | 难点分析 |
|---------|---------|------------|---------|
| 日常办公 | Calendar, Gmail | 相对较好 | API 结构化程度高 |
| 项目管理 | Notion, Canvas | 中等 | 需理解复杂数据结构 |
| 开发运维 | Kubernetes, Git | 较差 | 要求专业领域知识 |
| 数据分析 | BigQuery, Sheets | 较差 | 需要多步数据处理 |
| 电子商务 | WooCommerce | 较差 | 业务逻辑复杂 |

### 关键发现
- 即使最强的 Claude-4.5-Sonnet 也只达到 38.6% 的成功率，说明当前 Agent 在处理真实、长程、多应用任务时仍有巨大不足。
- 开源模型和闭源模型之间存在约 18 个百分点的差距（38.6% vs 20.1%），工具使用能力仍是开源模型的明显短板。
- 任务失败的主要原因包括：工具选择错误、长上下文中丢失关键信息、跨应用数据格式适配失败、缺乏错误恢复能力。
- 专业领域工具（如 Kubernetes、BigQuery）的成功率显著低于日常办公工具，说明领域知识是当前 Agent 的重要瓶颈。
- MCP 服务器的标准化接口降低了工具集成的复杂度，但 Agent 在"理解工具能力边界"方面仍有欠缺。

## 亮点与洞察
- **"真实环境状态"的设计哲学**：Toolathlon 不仅提供工具接口，还提供复杂的真实数据环境。这迫使 Agent 学会"在混乱中工作"——类似人类新员工面对已有数据和系统时的挑战。这比空白环境测试更能反映实际部署中的困难。
- **MCP 作为基础设施**：以 MCP 服务器为工具层的设计具有前瞻性——随着 MCP 生态系统的增长，Toolathlon 可以自然扩展到更多应用和工具。
- **长程任务的评估价值**：平均 20 轮调用的任务长度有效测试了 Agent 的规划、上下文维护和错误恢复能力，这些在短任务中无法体现。

## 局限与展望
- **任务规模有限**：108 个任务虽然手动高质量构建，但数量相对较少，可能不足以评估模型在更广泛场景中的泛化能力。
- **静态环境**：任务环境在评估开始时固定，不涉及动态变化的环境（如实时通知、并发用户操作），这在真实世界中很常见。
- **人工标注成本高**：每个任务需要手动编写评估脚本，扩展到更多任务/应用的成本较高。
- **缺少交互式场景**：所有任务都是单向的——Agent 执行任务，没有用户在中途提供反馈或修改需求的场景。
- **模型调用成本**：20 轮平均调用意味着大量的 API 开销，这限制了对更多模型进行全面评估的可行性。
- **地理与语言局限**：当前应用和任务以英语为主，面向北美/全球化的软件工具，对非英语环境和本地化应用的覆盖不足。
- **安全性维度缺失**：基准主要评估任务完成率，未系统评估 Agent 在工具使用中的安全性（如数据泄露、权限越界、误操作风险）。

## 相关工作与对比
- **vs AgentBench (Liu et al., 2023)**：AgentBench 覆盖了操作系统、数据库、Web 等场景，但任务长度和环境状态复杂度远不如 Toolathlon。Toolathlon 在应用多样性和环境真实性上是显著升级。
- **vs ToolBench (Qin et al., 2024)**：ToolBench 提供了大规模的 API 集合和自动化任务生成，但任务多为单步或少步调用，环境状态简单。Toolathlon 强调长程多步和真实状态。
- **vs SWE-bench (Jimenez et al., 2024)**：SWE-bench 聚焦软件工程单一领域（代码修复），Toolathlon 则覆盖 32 个不同领域的应用，测试 Agent 的通用工具使用能力。
- **vs GAIA (Mialon et al., 2023)**：GAIA 测试通用 AI 助手的网络搜索和推理能力，任务较短。Toolathlon 的 MCP 框架使其更接近真实的应用集成场景。
- **vs τ-bench (Yao et al., 2024)**：τ-bench 同样关注工具使用，但任务设计更偏向于推理能力验证。Toolathlon 更强调应用覆盖的广度和环境状态的真实性。
- **vs AgentDojo (Debenedetti et al., 2024)**：AgentDojo 侧重安全性（prompt injection 防御），Toolathlon 侧重功能性（任务完成率），二者评估视角互补。

## 评分
- 新颖性: ⭐⭐⭐⭐ MCP + 真实环境状态 + 长程任务的组合是 Agent 评估的新范式
- 实验充分度: ⭐⭐⭐⭐⭐ SOTA 模型全面测试，32 应用 × 604 工具 × 108 任务
- 写作质量: ⭐⭐⭐⭐ 基准描述清晰，但部分细节较密集
- 价值: ⭐⭐⭐⭐⭐ 填补了 Agent 评估中"真实性"的关键空缺，有望成为社区标准基准

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ICLR 2026\] Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents](harnessing_uncertainty_entropy-modulated_policy_gradients_for_long-horizon_llm_a.md)
- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](../../CVPR2026/llm_agent/carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [\[AAAI 2026\] Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution](../../AAAI2026/llm_agent/cook_and_clean_together_teaching_embodied_agents_for_paralle.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)

<!-- RELATED:END -->
