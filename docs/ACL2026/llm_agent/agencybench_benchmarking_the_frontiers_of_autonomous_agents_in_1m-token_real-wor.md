---
title: >-
  [论文解读] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts
description: >-
  [ACL 2026][LLM Agent][自主智能体] 提出AgencyBench——一个包含138个真实世界任务的综合基准，评估6种核心智能体能力，每个场景平均需90次工具调用和100万token，通过用户模拟agent和Docker沙箱实现全自动化评估。
tags:
  - ACL 2026
  - LLM Agent
  - 自主智能体
  - 长程任务
  - 真实世界基准
  - 用户模拟
  - Docker沙箱评估
---

# AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts

**会议**: ACL 2026  
**arXiv**: [2601.11044](https://arxiv.org/abs/2601.11044)  
**代码**: [GitHub](https://github.com/GAIR-NLP/AgencyBench)  
**领域**: LLM Agent / Benchmark  
**关键词**: 自主智能体, 长程任务, 真实世界基准, 用户模拟, Docker沙箱评估

## 一句话总结

提出AgencyBench——一个包含138个真实世界任务的综合基准，评估6种核心智能体能力，每个场景平均需90次工具调用和100万token，通过用户模拟agent和Docker沙箱实现全自动化评估。

## 研究背景与动机

**领域现状**：LLM-based自主智能体正在渗透软件开发、科学研究、日常使用等多个领域，但评估基准严重滞后于智能体能力的发展。

**现有痛点**：(1) 现有基准聚焦单一能力（如工具使用或软件工程），无法捕捉真实世界任务的多维性和长程性；(2) 真实任务评估依赖human-in-the-loop反馈，成为自动化评估的瓶颈；(3) 任务复杂度不够——大多数基准仅需数十次工具调用。

**核心矛盾**：前沿智能体的能力已远超现有基准的测试范围，亟需更具挑战性的评测。

**本文目标**：构建高复杂度、多维度、全自动化评估的真实世界智能体基准。

**切入角度**：由20位人类专家（AI研究者、开发者）收集真实工作场景中的任务，构建层次化的能力-场景-任务体系。

**核心 idea**：通过用户模拟agent替代人类反馈、Docker沙箱执行可视化评估，实现长程复杂任务的全自动化rollout收集和评分。

## 方法详解

### 整体框架

层次化设计：6种核心能力（游戏开发、前端、后端、代码生成、研究、MCP工具）→ 32个真实场景 → 138个具体任务。每个场景包含1-5个难度递增的顺序任务，前序任务结果影响后续。评估通过workspace-sandbox-evalspace三空间分离确保隔离性。

### 关键设计

1. **用户模拟Agent**:

    - 功能：在多轮交互中替代人类提供迭代反馈
    - 核心思路：模拟真实用户行为——当智能体提交中间结果时，模拟agent基于任务描述和rubric提供修改建议和确认
    - 设计动机：消除human-in-the-loop瓶颈，使长达数小时的rollout可以全自动完成

2. **Docker沙箱评估**:

    - 功能：对智能体产出的代码/文件进行可视化和功能性评估
    - 核心思路：将deliverables同步到Docker容器中，模拟人机操作（UI渲染、鼠标点击、屏幕录制），生成可视化artifacts，再由评估脚本和LLM judge基于rubric打分
    - 设计动机：很多真实任务的输出（如游戏、网页）无法仅靠文本评估，需要实际运行和视觉检查

3. **层次化任务设计**:

    - 功能：模拟真实工作流的渐进式复杂度
    - 核心思路：每个场景的1-5个任务难度递增，前序完成结果影响后续——如"五子棋游戏"场景从基础棋盘到添加AI对手、回退功能、主题切换等
    - 设计动机：真实世界任务从不是一步完成的，这种设计测试了智能体的上下文保持和长程规划能力

### 损失函数 / 训练策略

评估采用基于rubric的0-10分评分，结合规则评估脚本和LLM-based judge。全一致同意策略用于数据质量——4位专家需全部同意任务才能纳入。

## 实验关键数据

### 主实验

| 模型类型 | 平均分 | 最高 | 最低 |
|---------|--------|------|------|
| 闭源模型 | 48.4% | GPT-5.2 (56.5%) | Grok-4.1-Fast (44.3%) |
| 开源模型 | 32.1% | GLM-4.6 (38.6%) | Qwen-3-235B (27.0%) |

### 关键行为差异

| 模型 | 特点 | 说明 |
|------|------|------|
| GPT-5.2 | 反馈自校正强 | 最善于利用用户反馈改进 |
| Grok-4.1-Fast | Token效率高 | 用更少token完成任务 |
| Claude-4.5-Opus | 偏好Shell工具 | 更多使用命令行操作 |
| Gemini-3-Pro | 偏好文件管理 | 更多使用文件和记忆管理工具 |

### 关键发现
- 闭源模型显著优于开源模型（48.4% vs 32.1%），差距比短任务基准上更大
- "主场优势"效应明显——模型在原生框架（如Claude+Claude-Agent-SDK）中表现最佳
- 当前最强模型也仅达56.5%，说明长程真实世界任务仍是巨大挑战
- 不同模型有明显的工具使用偏好差异，暗示架构和训练数据的影响

## 亮点与洞察
- 任务复杂度远超现有基准——平均90次工具调用和100万token是质的飞跃
- 用户模拟agent和Docker沙箱的组合解决了长程任务自动评估的核心难题
- "主场优势"发现对智能体框架设计有重要启示——通用框架可能不如专用框架

## 局限与展望
- 138个任务可能仍不足以全面覆盖真实世界场景
- 用户模拟agent的质量是评估可靠性的上限
- Docker沙箱的环境配置复杂，可能限制社区采用
- 未来可扩展到更多领域（如数据分析、设计、写作）

## 相关工作与启发
- **vs SWE-bench**: SWE-bench聚焦软件工程单一能力，AgencyBench覆盖6种能力
- **vs GAIA**: GAIA平均仅10K token，AgencyBench是100倍复杂度
- **vs ToolLLM**: ToolLLM关注工具调用正确性，AgencyBench关注端到端任务完成

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 在规模和真实性上显著超越现有基准
- 实验充分度: ⭐⭐⭐⭐ 多模型对比、行为分析、框架对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，案例丰富
- 价值: ⭐⭐⭐⭐⭐ 为下一代智能体评估设立新标杆

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)
- [\[AAAI 2026\] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](../../AAAI2026/llm_agent/d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](../../ICLR2026/llm_agent/openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ACL 2026\] CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents](ci-work_benchmarking_contextual_integrity_in_enterprise_llm_agents.md)
- [\[ACL 2026\] FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems](fedgui_benchmarking_federated_gui_agents_across_heterogeneous_platforms_devices_.md)

<!-- RELATED:END -->
