---
title: >-
  [论文解读] WebDS: An End-to-End Benchmark for Web-based Data Science
description: >-
  [多模态VLM] 提出首个端到端 Web 数据科学基准 WebDS（870 个任务，29 个网站，10 个领域），当前最强 Agent（BrowserUse + GPT-4o）仅完成 15% 的任务，而人类达到 90%，揭示了真实数据科学工作流中 Agent 的巨大性能差距。
tags:
  - "多模态VLM"
---

# WebDS: An End-to-End Benchmark for Web-based Data Science

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2508.01222](https://arxiv.org/abs/2508.01222)
- **代码**: [WebDS Benchmark](https://webdsbenchmark.github.io/)
- **领域**: 多模态大模型 / Web Agent / 数据科学
- **关键词**: web agent, 数据科学, benchmark, 端到端评估, 多步推理

## 一句话总结

提出首个端到端 Web 数据科学基准 WebDS（870 个任务，29 个网站，10 个领域），当前最强 Agent（BrowserUse + GPT-4o）仅完成 15% 的任务，而人类达到 90%，揭示了真实数据科学工作流中 Agent 的巨大性能差距。

## 研究背景与动机

现实中的数据科学任务涉及复杂的 Web 交互：在互联网上寻找合适的数据、从不同位置综合多模态数据、生成汇总分析。然而现有基准存在两个关键缺陷：

**Web Agent 基准**（如 WebVoyager、WebArena）聚焦简单交互（发帖、购物），不要求多样化的工具使用能力和数据分析

**数据科学基准**（如 InfiAgent-DABench、DSBench）集中于静态结构化数据集，不涵盖从数据获取到分析的端到端工作流

**核心矛盾**：真实数据科学工作流通常从浏览 Web 开始，跨多个网站导航和综合信息，但这一关键环节被现有基准忽略。例如，BrowserUse 在 WebVoyager 上达到 80%，但在 WebDS 上仅 15%。

## 方法详解

### 整体框架

WebDS 不是一个模型，而是一个完整刻画"真实数据科学管道"的基准：Agent 要先在真实网站上浏览并获取数据，再对获取到的多模态数据做分析或可视化，最后生成有理据支撑的产出。整套设计围绕三件事展开——任务从哪来、怎么覆盖真实复杂度、又如何被客观评分。

### 关键设计

**1. 专家驱动的任务来源：让任务贴近真实工作流而非人工臆造。** 多数 Web Agent 基准的任务是研究者凭空设计的简单交互，与真实数据从业者每天做的事相去甚远。WebDS 改为对 8 名记者、数据科学家和领域专家做访谈，从他们的实际工作中归纳出两类核心任务：一类要产出下游产品（分析报告、可视化图表），另一类要解答关键的分析性问题。由此最终人工编写出 870 个任务，覆盖 29 个数据丰富的网站（CDC、政府数据门户、新闻媒体等）和 10 个高风险领域，数据形态同时包含结构化（CSV、表格）和非结构化（文本、图形），保证基准触及的是真实工作中那种"先要找对数据、再做综合"的难点。

**2. 七维属性标注：把模糊的"难"拆成可解释的能力维度。** 单看成功率无法说明 Agent 究竟卡在哪一步，因此每个任务都被打上 7 种属性标签，使失败可以被定位到具体能力短板。其中最主要的是问答（QA）与行动（Action）的区分：问答类含 344 个单跳与 117 个多跳，行动类含 97 个单跳、134 个多跳与 139 个需调用工具的任务。其余维度刻画了任务对不同能力的要求——是否需要组合多个数据源（单跳 vs 多跳）、数据是结构化还是非结构化、是否需要 Python/SQL 等外部工具、是否需要网站导航交互、是否涉及跨网站信息聚合。这套标注让后续分析能精确回答"Agent 在多跳还是工具使用上更弱"这类问题。

**3. 基于能力组合的难度分级：用任务包含多少种硬能力来定难度。** 难度不靠主观打分，而是由任务是否触及"多跳、非文本、行动、工具、多网站"这些硬能力来机械判定，从而保证分级客观可复现。具体规则为：$$\text{Difficulty} = \begin{cases} \text{Easy (247)} & \text{不含多跳/非文本/行动/工具，单网站} \\ \text{Medium (275)} & \text{恰好包含上述一项，单网站} \\ \text{Hard (348)} & \text{包含两项以上或多网站} \end{cases}$$ 这样一来，难度直接对应任务所需的能力数量，Agent 在三档上的得分曲线就能反映它随复杂度上升的退化情况。

**4. 双轨部署：在真实性与可复现性之间两头兼顾。** 真实网站会随时间变化、难以复现，但容器化又会丢失真实 Web 的复杂度，二者难以兼得。WebDS 因此提供两条评估轨道：WebDS-live 让 Agent 直接在真实网站上交互，最大程度捕捉真实 Web 的混乱与多变；WebDS-dockerized 则把子集网站容器化部署，牺牲一部分真实性换取严格的可复现实验环境。研究者可按需选择，既能做贴近现实的能力评估，也能做可重复的对照实验。

**5. 三级评估协议：让端到端的开放产出也能被客观打分。** 数据科学任务的产出往往是报告或图表这类开放结果，单一的对错判定不够用。WebDS 因此叠加三层评估：对有参考答案的任务做自动二元评估，由 LLM 比较输出与标准答案给出 SUCCESSFUL/UNSUCCESSFUL；对开放任务扩展 WebVoyager 的做法，由 LLM 对完整轨迹（而非仅看最终截图）给出 1–5 的五级评分并附失败分析；最后用人工验证兜底，对 400 对任务-轨迹做独立评审，结果显示该评估体系与人工判断达到 93% 一致率，说明自动评分足够可信。

## 实验

### 主要结果

| Agent | 框架 | SR% |
|-------|------|-----|
| GPT-4o + BrowserUse | BrowserUse | 13.2% |
| GPT-4o + AgentOccam | AgentOccam | 4.8% |
| Claude Sonnet-4.5 + WebArena | WebArena | ~10% |
| GPT-5.1 + WebArena | WebArena | ~12% |
| **人类基线** | 浏览器 | **90% (±3%)** |

### 关键发现

1. **巨大的人机差距**：最强 Agent 仅 13.2%，人类 90%，差距高达 ~77 个百分点
2. **增加模型容量无显著提升**：GPT-4o、GPT-4o-mini 和 Qwen2.5-72B 表现相似
3. **新型失败模式**：
    - **信息锚定失误**：锚定知识与潜在知识矛盾
    - **重复行为**：在多跳任务中陷入循环
    - **走捷径**：跳过必要的数据获取步骤
4. **难度梯度明显**：Agent 在 Easy 任务上得分约为 Medium/Hard 的 2.5 倍
5. **跨基准差距**：WebVoyager 上 81.1% vs WebDS 上 13.2%（同一 Agent）

### 对比 WebVoyager / WebArena

| 特征 | WebVoyager | WebArena | WebDS |
|------|-----------|----------|-------|
| 多跳 | ✗ | ✓ | ✓ |
| 结构化数据 | ✗ | ✗ | ✓ |
| 非结构化数据 | ✗ | ✗ | ✓ |
| 多网站 | ✗ | ✓ | ✓ |
| 工具使用 | ✗ | ✓ | ✓ |
| 端到端数据科学 | ✗ | ✗ | ✓ |

## 亮点

- 首个端到端 Web 数据科学基准，弥合了 Web 交互与数据科学能力之间的鸿沟
- 870 个人工编写的高质量任务，粒度覆盖 7 种属性和 3 种难度
- 双轨设计（live + dockerized）兼顾真实性与可复现性
- 完整轨迹评估 + 细粒度评分，超越简单的二元判定
- 量化了巨大的人机差距，为社区指明方向

## 局限性

- 当前仅覆盖 29 个网站，领域代表性有限
- 容器化部署仅为子集，部分任务依赖 live 网站可能随时间变化
- 人工标注成本高，870 个任务规模可能不足以覆盖所有真实场景
- 评估仍依赖 LLM-as-Judge，对复杂分析报告的质量评判可能不够精确
- 未深入分析不同类型工具使用的能力差异

## 相关工作

- **数据分析基准**：SQuAD、HotpotQA（结构化 QA），InfiAgent-DABench、DSBench（数据科学 agent），Spider 2.0（企业 SQL）
- **Web Agent 基准**：WebArena（功能正确性），WebVoyager（最终截图），Mind2Web（动作序列）
- **端到端工作流**：GAIA（多模态推理），AssistantBench（Web 辅助）— 均不专注数据科学管道

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个端到端 Web 数据科学基准，问题定义新颖
- **技术深度**: ⭐⭐⭐⭐ — 任务设计严谨，评估体系全面
- **实验充分度**: ⭐⭐⭐⭐ — 9 个 SOTA agent + 人类基线，多维度分析
- **实用价值**: ⭐⭐⭐⭐⭐ — 揭示 Agent 在真实数据科学中的关键不足，指导未来发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[ICLR 2026\] VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?](vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning.md)

</div>

<!-- RELATED:END -->
