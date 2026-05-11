---
title: >-
  [论文解读] From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?
description: >-
  [ICML 2025][LLM Agent][Active Reasoning] 本文提出 AR-Bench，一个专门评估 LLM 主动推理能力的基准，包含侦探案件、情境谜题和猜数字三类任务，实验发现 GPT-4o 等最先进模型在需要主动提问获取缺失信息的场景中表现远逊于人类，揭示了被动推理与主动推理之间的巨大鸿沟。
tags:
  - ICML 2025
  - LLM Agent
  - Active Reasoning
  - benchmark
  - 信息获取
  - 多轮交互
  - 推理评估
---

# From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?

**会议**: ICML 2025  
**arXiv**: [2506.08295](https://arxiv.org/abs/2506.08295)  
**代码**: [https://github.com/tmlr-group/AR-Bench](https://github.com/tmlr-group/AR-Bench)  
**领域**: LLM Agent  
**关键词**: Active Reasoning, benchmark, 信息获取, 多轮交互, 推理评估

## 一句话总结
本文提出 AR-Bench，一个专门评估 LLM 主动推理能力的基准，包含侦探案件、情境谜题和猜数字三类任务，实验发现 GPT-4o 等最先进模型在需要主动提问获取缺失信息的场景中表现远逊于人类，揭示了被动推理与主动推理之间的巨大鸿沟。

## 研究背景与动机
**领域现状**：LLM 在数学、编程等被动推理（Passive Reasoning）任务中表现出色，这些任务都提供了充分信息来推导答案。

**现有痛点**：现实世界大量场景需要在信息不完整条件下工作——旅行规划要询问用户偏好、医疗诊断要追问症状——这些需要模型主动获取信息。

**核心矛盾**：现有评测几乎全聚焦被动推理，少量已有主动推理数据集（如 20 Questions、MediQ）要么太简单、要么缺少符号反馈和复杂推理。

**本文目标**：构建全面的主动推理基准 AR-Bench，系统评估 LLM 通过多轮交互获取关键信息并推理答案的能力。

**切入角度**：设计三类互补任务——侦探案件（常识推理）、情境谜题（逻辑推理）、猜数字（符号推理），覆盖不同反馈类型。

**核心 idea**：主动推理的本质不是"推导正确答案"，而是"提出正确的问题"来获取缺失信息。

## 方法详解

### 整体框架
AR-Bench 包含 6040 个谜题，分为三个任务家族。评估流程模拟多轮对话：被评估模型作为 player 在 25 轮内向 NPC（Llama-3.1-405B 或规则函数）提问获取信息，最终给出答案。设计了结果指标（正确率/F1/精确匹配）和过程指标（关键问题覆盖率）两套评测体系。

### 关键设计

1. **侦探案件 (Detective Cases, DC)**:

    - 功能：模拟侦探审讯 5 名嫌疑人的过程，每名嫌疑人有不同角色（帮助或干扰）
    - 核心思路：player 轮流选择嫌疑人并提问，收集线索后判断真凶
    - 设计动机：测试模型在复杂嘈杂反馈下的常识推理和信息整合能力
    - 规模：400 训练 / 100 测试，平均问题 564 tokens，答案空间 5

2. **情境谜题 (Situation Puzzles, SP)**:

    - 功能：经典横向思维谜题，player 通过 Yes/No 问题逐步还原反常谜面的真相
    - 核心思路：需要间接创造性思维，从碎片线索拼凑完整故事
    - 设计动机：测试开放答案空间（无穷大）下的逻辑推理能力
    - 规模：400 训练 / 100 测试，平均 178 tokens

3. **猜数字 (Guessing Numbers, GN)**:

    - 功能：猜 4 位不重复数字，每次猜测获得"几个位置正确+几个数字对但位置错"的反馈
    - 核心思路：每次猜测即信息查询，需最大化信息增益，通过符号推理缩小假设空间
    - 设计动机：测试利用精确符号反馈进行系统性推理的能力
    - 答案空间：5040 种可能

4. **数据集构建与评估**:

    - 功能：四阶段自动化生成（核心采样→树扩展→关键问题提取→谜题合成）+ 人工验证
    - 过程评估公式：$\text{Score}(Q, s_t) = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(f(s_t, q_i) = 1)$
    - 评判函数 f 由 Llama-3.1-405B 实现，判断当前对话状态能否回答各关键问题

### 评估方法体系
涵盖 8 个模型（Llama-3.1-8B/70B/405B、Qwen-2.5-3B/7B、QwQ-32B、GPT-4o-mini、GPT-4o）和 6+2 种方法（zero-shot、few-shot、few-shot+instruction、ToT、SFT、DPO、Proactive CoT、UoT）。

## 实验关键数据

### 主实验

| 模型/方法 | DC 准确率 | SP F1 | GN 精确匹配 |
|-----------|----------|-------|------------|
| GPT-4o (zero-shot) | ~60% | ~50% | 35% |
| Llama-3.1-405B | 中等偏上 | 中等偏上 | 中等 |
| Llama-3.1-8B + SFT | 一般 | 一般 | **0%** |
| Llama-3.1-8B + DPO | 一般 | 低于zero-shot | 低于zero-shot |
| Proactive CoT | 边际提升(SP) | — | — |
| UoT | 低于zero-shot | — | — |
| **Human** | **显著高于所有模型** | **显著高于所有模型** | **显著高于所有模型** |

### 过程分析（信息获取效率）

| 交互阶段 | 平均过程得分提升 | 说明 |
|---------|---------------|------|
| 5-10 轮 | +7.7% | 早期快速进步 |
| 20-25 轮 | +2.5% | 后期严重停滞 |
| 前50轮 scaling | +45.8% | 累计过程得分增长 |
| 50-100轮 scaling | 仅+6.7% | 边际收益递减明显 |

### 错误模式分析

| 任务 | 错误类型 | Llama-8B | GPT-4o |
|------|---------|---------|--------|
| DC | 时间线误解 | 10% | 31% |
| DC | 忽略证据 | 61% | 15% |
| SP | 无支撑假设 | 90% | 72% |
| GN | 反馈理解错误 | 78% | 61% |
| GN | 不完整测试 | 81% | 55% |

### 关键发现
- GPT-4o 在被动推理榜单领先，但主动推理中 GN 仅 35% 精确匹配
- SFT 在 GN 中得分为 0，DPO 在 SP、GN 中低于 zero-shot
- 模型倾向于提出模糊、重复的问题，后期"收敛到局部最优"
- 即使 scaling 到 100 轮交互也无法完全解决任务
- 验证器可靠性因任务而异：GN 确定性验证最好，DC/SP 启发式验证效果差
- 更大模型在信息检索（提问质量）和信息处理（不完整信息下推理）两方面都更强

## 亮点与洞察
- 首次系统区分并评估被动推理与主动推理，填补重要评测空白
- 核心洞见：**推理能力 ≠ 提问能力**——模型擅长推导已知信息，但不擅长识别和获取缺失信息
- 过程评估指标设计巧妙，不仅看最终答案还追踪每轮信息获取质量
- 三任务覆盖不同反馈类型（叙事/布尔/符号）和答案空间维度（5/∞/5040）
- "早期快速进步＋后期严重停滞"指向上下文管理和长期策略规划的根本性挑战

## 局限与展望
- NPC 答案质量依赖 Llama-3.1-405B，可能引入评估偏差
- 三任务仍是简化的 toy 场景，距真实世界应用（如医疗诊断）有差距
- 侧重揭示问题而非提供解决方案，缺少有效提升主动推理的新方法
- 未探索 RAG 或工具调用等更实际的信息获取方式

## 相关工作与启发
- 与 Proactive CoT 和 UoT 对比表明，现有主动推理方法在复杂场景下效果有限
- 指出多个研究方向：交互式学习、实时反馈循环、环境感知训练目标
- 与长上下文研究关联：后期退化可能源于上下文过长导致注意力稀释
- 启发：未来 LLM agent 需要具备"提出好问题"的能力，不仅是"回答好问题"

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统定义并评测被动vs主动推理范式，开辟全新评测维度
- 实验充分度: ⭐⭐⭐⭐⭐ 8个模型×8种方法×3个任务，外加人类对照、过程分析、错误模式分析和多角度消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰、观察点逐条编号便于引用，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 揭示了LLM推理能力的根本性盲区，对agent系统设计有深远启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)
- [\[NeurIPS 2025\] Are Large Language Models Sensitive to the Motives Behind Communication?](../../NeurIPS2025/llm_agent/are_large_language_models_sensitive_to_the_motives_behind_communication.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](../../ACL2025/llm_agent/toolhop_multi_hop_tool_use.md)
- [\[ACL 2025\] Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger](../../ACL2025/llm_agent/meco_metacognition_tool_use.md)
- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](../../AAAI2026/llm_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)

</div>

<!-- RELATED:END -->
