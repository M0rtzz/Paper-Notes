---
title: >-
  [论文解读] SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks
description: >-
  [NeurIPS 2025 (Datasets & Benchmarks Track, Spotlight)][scientific literature evaluation] 构建 SciArena 社区驱动的科学文献评估开放平台，采用 Chatbot Arena 式的人类偏好投票方式对 47 个基础模型进行排名，收集超过 20,000 条投票数据，并发布 SciArena-Eval 元基准来评测自动评估系统对文献任务答案质量的判断能力。
tags:
  - "NeurIPS 2025 (Datasets & Benchmarks Track, Spotlight)"
  - scientific literature evaluation
  - foundation models
  - human preference
  - Chatbot Arena
  - meta-benchmark
---

# SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks

**会议**: NeurIPS 2025 (Datasets & Benchmarks Track, Spotlight)  
**arXiv**: [2507.01001](https://arxiv.org/abs/2507.01001)  
**代码**: 有  
**领域**: NLP / 科学文献评估  
**关键词**: scientific literature evaluation, foundation models, human preference, Chatbot Arena, meta-benchmark

## 一句话总结
构建 SciArena 社区驱动的科学文献评估开放平台，采用 Chatbot Arena 式的人类偏好投票方式对 47 个基础模型进行排名，收集超过 20,000 条投票数据，并发布 SciArena-Eval 元基准来评测自动评估系统对文献任务答案质量的判断能力。

## 研究背景与动机

**领域现状**：科学文献理解与综合是基础模型应用的重要场景，涵盖论文问答、综述生成、假设提出等开放性任务。传统基准测试通常依赖可自动验证答案的闭集任务（如多选题、简答题），但科学文献任务的答案往往是长篇的、开放的，难以通过程序化方式自动判定好坏。

**现有痛点**：现有的科学文献基准要么局限于简单的抽取式问答，要么依赖固定的标准答案评分，无法捕捉研究者对模型回答的整体偏好。自动评估系统（如 LLM-as-judge）在评估此类非可验证任务时，与人类判断的一致性又难以保证。

**核心矛盾**：评估科学文献相关任务的模型能力需要领域专家的参与，但大规模、持续性的人类评估成本极高。如何在保证评估质量的同时实现可扩展的社区驱动评估，是一个关键挑战。本文借鉴 Chatbot Arena 的成功经验，构建面向科学文献任务的社区驱动评估平台，让研究者通过投票对模型输出进行两两比较，同时发布元评估基准来推动更可靠的自动评估方法研究。

## 方法详解

### 整体框架
SciArena 采用"人类偏好投票 + 自动评估元基准"的双层框架。底层是一个基于网页的交互平台，研究者提交科学文献相关的任务请求，系统将请求发送给两个匿名模型生成回答，研究者选择更好的回答进行投票。上层基于收集到的偏好数据构建 SciArena-Eval 元基准，用于评测自动评估器。

### 关键设计

1. **社区驱动评估平台**:

    - 功能：提供一个开放、协作的网页平台，研究者可以在上面提交科学文献任务（如论文问答、文献综合等）并对模型输出进行两两比较投票
    - 核心思路：采用 Chatbot Arena 的 pairwise comparison 范式，用户提交查询后系统随机匹配两个匿名模型，用户根据回答质量投票（A 更好 / B 更好 / 平局）
    - 设计动机：科学文献任务的答案质量难以程序化判断，需要领域研究者的专业判断；Chatbot Arena 已在通用对话场景验证了社区投票的有效性

2. **模型排名与 Elo 评分系统**:

    - 功能：基于偏好投票数据生成可信的模型排名
    - 核心思路：利用 Elo 评分或 Bradley-Terry 模型将两两比较的投票转化为全局排名，支持 47 个基础模型的公平比较
    - 设计动机：两两比较相对绝对打分更稳定、更符合人类判断习惯；排名系统可以在持续收集新数据时动态更新

3. **SciArena-Eval 元评估基准**:

    - 功能：衡量自动评估系统（如 LLM-as-judge）判断文献任务回答质量的准确性
    - 核心思路：以人类投票作为金标准，检查自动评估器的两两判断是否与人类偏好一致
    - 设计动机：高质量人类评估成本高，若能开发可靠的自动评估方法将大幅降低评估成本；但现有方法在科学文献任务上的准确度尚不清楚，需要标准化的元基准

### 数据收集与质量保障
平台通过投票一致性分析和统计检验确认数据质量，确保收集到的偏好投票反映真实的模型能力差异而非随机噪声。所有投票来自跨多个科学领域的人类研究者，涵盖计算机科学、物理学、生物学、医学等多个学科。平台设计考虑了控制偏差的措施，如模型匿名化（消除品牌偏好）、随机配对（避免特定模型对比的过度采样）、以及对投票模式的统计异常检测。

## 实验关键数据

### 主实验

| 统计维度 | 数据 |
|---------|------|
| 支持模型数 | 47 个基础模型 |
| 总投票数 | 20,000+ 条来自人类研究者 |
| 覆盖范围 | 多个科学领域 |
| 任务类型 | 开放式科学文献任务，需长篇文献支撑的回答 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| LLM-as-judge 基线 | 与人类投票一致率 | 自动评估系统在科学文献任务上与人类判断的一致性仍有较大提升空间 |
| 不同模型作为评估器 | 偏好对齐度 | 各模型作为 judge 时与人类偏好的对齐程度存在显著差异 |
| 数据质量分析 | 投票一致性 | 统计分析确认收集数据的高质量性 |

### 关键发现
- 顶级模型（如 GPT-4、Claude 系列）在科学文献理解和综合任务上排名领先，能力分层明显
- 自动评估方法与人类偏好的一致性仍然不高，说明科学文献任务的自动评估是一个有待攻克的难题
- 社区投票数据经统计分析确认质量可靠
- 不同科学领域的投票分布和模型表现存在差异，表明科学文献评估需要考虑领域特异性
- 模型在需要深度文献理解和综合的复杂任务上差距更大，简单事实性问答上差异较小

## 亮点与洞察
- 首个面向科学文献非可验证任务的社区驱动评估平台，获 NeurIPS Spotlight，填补了该领域评估空白
- 将 Chatbot Arena 范式成功迁移到科学文献领域，验证了社区驱动评估在垂直领域的可行性和有效性
- SciArena-Eval 量化了自动评估方法的不足，为后续研究提供了清晰的改进方向和基准线
- 基础设施型贡献，价值将随社区扩大和数据积累持续增长
- 涵盖多种科学任务类型（文献问答、综述综合、假设评估等），任务设计贴近真实科研需求

## 局限与展望
- 投票质量依赖研究者的参与意愿和专业水平，可能存在领域覆盖不均衡的问题
- 两两比较无法直接给出模型在特定子任务上的绝对能力评分，只能提供相对排名
- 不同子任务（如问答 vs 综述生成）的评估需求差异大，难以用单一排名体系覆盖
- 需要探索结合引文验证、事实核查等方法来提高自动评估可靠性
- 目前主要覆盖英文科学文献，多语言扩展和非英文学术社区的参与仍是挑战
- 投票者的学科背景可能影响对跨学科任务回答的判断公正性
- 随着参与模型数量增多，每个模型获得的投票对可能被稀释，影响排名精度

## 相关工作与启发
- **Chatbot Arena**：通用对话评估平台，本文的直接灵感来源，证明了社区投票评估的可扩展性和可靠性
- **LLM-as-judge**（如 MT-Bench）：SciArena-Eval 直接评测了这类方法在科学文献领域的可靠性，发现提升空间很大
- **MMLU / SciQ 等科学基准**：传统闭集评估基准，无法评估开放式科学文献任务的模型能力
- **Semantic Scholar / OpenReview**：学术文献平台，SciArena 在这些平台积累的文献基础上进行模型评估
- 启发：科学文献评估的核心难点在于"非可验证性"——没有唯一正确答案。这一问题在代码审查、医学诊断、法律推理等领域同样存在，SciArena 的平台化思路具有推广价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个面向科学文献的社区驱动评估平台，填补重要空白
- 实验充分度: ⭐⭐⭐ 规模可观（47模型/20K+投票），但详细排名和元评估的深入分析受限于可获取的论文信息
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，平台设计合理，动机论述充分
- 价值: ⭐⭐⭐⭐⭐ 基础设施级贡献，NeurIPS Spotlight，对科学文献评估领域有长期推动作用

<!-- end -->

<!-- RELATED:START -->

## 相关论文

- [PersonaLens: A Benchmark for Personalization Evaluation in Conversational AI Assistants](../../ACL2025/dialogue/personalens_a_benchmark_for_personalization_evaluation_in_conversational_ai_assi.md)
- [Investigating Non-Transitivity in LLM-as-a-Judge](../../ICML2025/dialogue/investigating_non-transitivity_in_llm-as-a-judge.md)
- [Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [Canoe: Teaching LLMs to Maintain Contextual Faithfulness via Synthetic Tasks and RL](../../AAAI2026/dialogue/teaching_large_language_models_to_maintain_contextual_faithfulness_via_synthetic.md)
- [MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)

<!-- RELATED:END -->
