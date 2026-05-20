---
title: >-
  [论文解读] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games
description: >-
  [ACL 2026][多模态VLM][不完全信息推理] 提出一个协作式多智能体框架用于自动生成高质量剧本杀游戏脚本和训练数据，通过两阶段训练策略（CoT 微调 + GRPO 强化学习配合 ScoreAgent 奖励塑形）增强 VLM 在不完全信息下的多跳推理能力…
tags:
  - "ACL 2026"
  - "多模态VLM"
  - "不完全信息推理"
  - "剧本杀"
  - "多智能体数据生成"
  - "视觉语言模型"
  - "强化学习"
---

# Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games

**会议**: ACL 2026  
**arXiv**: [2604.11741](https://arxiv.org/abs/2604.11741)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 不完全信息推理, 剧本杀, 多智能体数据生成, 视觉语言模型, 强化学习

## 一句话总结
提出一个协作式多智能体框架用于自动生成高质量剧本杀游戏脚本和训练数据，通过两阶段训练策略（CoT 微调 + GRPO 强化学习配合 ScoreAgent 奖励塑形）增强 VLM 在不完全信息下的多跳推理能力，在 WhodunitBench 上显著提升 VLM 的叙事推理、事实提取和欺骗抵御能力。

## 研究背景与动机

**领域现状**：视觉语言模型（VLM）在感知任务上表现出色，但在涉及不完全信息、欺骗和多玩家社交互动的复杂多跳推理中仍然退化。剧本杀（Murder Mystery）作为一种社交推理游戏，要求玩家基于部分线索推断隐藏真相，是研究这类推理的理想测试平台。

**现有痛点**：(1) 剧本杀场景缺乏大规模高质量数据集用于微调和评估 VLM；(2) 人工生产高质量剧本杀脚本成本高且难以规模化；(3) 现有 VLM 在角色一致性（凶手需要欺骗、无辜者需要合作）和多模态多跳推理（结合文本和视觉线索）方面表现不佳；(4) 角色扮演和互动讨论没有标准答案，纯 SFT 不足以训练这类行为。

**核心矛盾**：VLM 需要在不完全、欺骗性信息环境中进行可靠推理，但缺乏合适的训练数据和训练方法。

**本文目标**：(1) 构建可扩展的多智能体数据合成框架；(2) 设计适合不完全信息推理的两阶段训练策略。

**切入角度**：用强力 LLM（Gemini 2.5 Pro）作为 Agent 协作生成游戏脚本，然后用 Agent 监控的训练策略增强目标 VLM。

**核心 idea**：生成 Agent（故事大纲→角色脚本→线索→对话→QA）+ 评估 Agent（质量控制+奖励塑形）协作构建训练数据，两阶段训练（SFT + GRPO with ScoreAgent）增强 VLM。

## 方法详解

### 整体框架
两大模块：(1) 数据生成模块：多个专职 Agent（OutlineAgent、CharacterAgent、ClueAgent、RoleplayAgent、QaAgent、CriticAgent）协作生成剧本杀脚本和训练数据；(2) 模型增强模块：Stage 1 SFT 建立基础推理能力 → Stage 2 GRPO 强化学习在 ScoreAgent 监控下优化角色特定行为。

### 关键设计

1. **多智能体脚本生成框架**:

    - 功能：自动生成多样化、高质量的剧本杀游戏脚本和训练数据
    - 核心思路：六个专职 Agent 流水线协作：OutlineAgent 构建犯罪日叙事（动机+秘密）→ CharacterAgent 细化角色日常行动和交互 → CriticAgent 从剧情复杂性、角色发展、难度、逻辑合理性四个维度评估并反馈 → ClueAgent 生成多模态线索（视觉+文本）→ RoleplayAgent 模拟多轮对话 → QaAgent 生成从单跳到多跳的推理链和 QA 对
    - 设计动机：单一模型生成整个剧本容易出现逻辑不一致。专职分工 + CriticAgent 评估反馈确保了脚本质量

2. **ScoreAgent 监控的 GRPO 强化学习**:

    - 功能：优化 VLM 的角色一致性和推理质量
    - 核心思路：对不同类型的训练数据设计不同的奖励函数。**不可验证数据**（自我介绍、讨论）：ScoreAgent（LLM-as-Judge）评分角色一致性，讨论还加入 $S_{\text{choice}}$（选择询问嫌疑人得 1 分，问其他人得 0.5 分，问自己得 0 分）。**可验证数据**（QA）：答案正确性 + 格式正确性 + 线索匹配正确性的加权组合
    - 设计动机：SFT 可以建立基础能力但无法处理没有标准答案的角色扮演行为。GRPO 通过 ScoreAgent 评估来区分好的和差的角色扮演表现

3. **不完全信息下的推理链生成**:

    - 功能：为训练提供在信息不完整条件下的推理示例
    - 核心思路：自动生成基于不完整信息的推理链——玩家只能看到自己的线索和公共信息，需要从中进行多跳推理。这与传统 CoT（假设信息完整）形成对比
    - 设计动机：传统推理数据假设信息完整，但剧本杀的核心挑战正是信息不完全和欺骗

## 实验关键数据

### 主实验（WhodunitBench）

| 方法 | MMR | CMD | RP | DM | LSU | TIU | MIU |
|------|-----|-----|----|----|-----|-----|-----|
| GPT-4V | 58.75 | 26.43 | 6.43 | 24.2% | 92.40 | 51.88 | 69.25 |
| Gemini-1.5-Pro | 57.39 | 19.20 | 7.22 | 16.9% | - | - | - |
| Qwen2.5-VL-3B | baseline | - | - | - | - | - | - |
| **Qwen2.5-VL-3B + Ours** | **显著提升** | **提升** | **提升** | **提升** | **提升** | **提升** | **提升** |

### 消融实验

| 配置 | 说明 |
|------|------|
| 仅 SFT | 建立基础推理但角色一致性差 |
| SFT + 无 ScoreAgent 的 RL | 奖励信号不准确，改进有限 |
| **SFT + ScoreAgent GRPO** | 角色一致性和推理质量双提升 |

### 关键发现
- **多智能体框架成功生成了多样化、逻辑一致的剧本杀数据**，CriticAgent 的反馈机制显著提升脚本质量
- **两阶段训练在 3B 和 7B 两个规模上一致有效**
- **ScoreAgent 的角色特定奖励设计**使得模型学会了凶手和无辜者的不同行为模式
- **GRPO 对角色扮演行为的改进特别显著**——SFT 对没有标准答案的行为训练效果有限
- **低分示例特征清晰**：偏离主题、自相矛盾、过早暴露身份等

## 亮点与洞察
- **将剧本杀建模为 VLM 的推理训练平台**是巧妙的任务选择——涵盖了不完全信息、欺骗检测、多跳推理、多模态整合等多个挑战
- **ScoreAgent 的差异化奖励设计**（可验证 vs 不可验证数据的不同奖励函数）是实用的解决方案，避免了为没有标准答案的任务训练独立奖励模型
- **数据生成框架的可扩展性**：通过添加或调整专职 Agent，可以适配其他博弈论任务（如狼人杀、法庭模拟）

## 局限与展望
- WhodunitBench 只有 50 个剧本，评估规模有限
- 生成的脚本质量依赖于 Gemini 2.5 Pro 的能力，成本较高
- 角色扮演评估仍主要依赖 LLM-as-Judge，主观性较强
- 未探索多个 VLM 之间的真实多人交互训练
- 视觉线索目前较简单，未涉及复杂的场景理解（如监控视频分析）
- 训练数据的多样性受限于生成 Agent 的创造力

## 相关工作与启发
- **vs WhodunitBench (Xie et al., 2024)**: WhodunitBench 提供评测平台但数据不足。本文提供了数据生成框架和训练方法
- **vs AgentInstruct / MATRIX**: 这些做通用合成数据，本文专注于不完全信息博弈场景的结构化数据生成
- **vs Reason-RFT / SRPO**: 通用推理增强方法，本文的 ScoreAgent 设计针对角色一致性做了特化

## 评分
- 新颖性: ⭐⭐⭐⭐ 将剧本杀作为 VLM 推理训练场景新颖，多智能体数据生成框架设计周到
- 实验充分度: ⭐⭐⭐ WhodunitBench 规模有限，具体数字不够完整
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但篇幅较长
- 价值: ⭐⭐⭐⭐ 对不完全信息下的 VLM 推理训练有独特贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning](../../ACL2025/multi_agent/getreason_enhancing_image_context_extraction_through_hierarchical_multi-agent_re.md)
- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents](hachimi_scalable_and_controllable_student_persona_generation_via_orchestrated_ag.md)
- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[ACL 2026\] When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning](when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)

</div>

<!-- RELATED:END -->
