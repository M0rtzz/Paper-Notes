---
title: >-
  [论文解读] WebDS: An End-to-End Benchmark for Web-based Data Science
description: >-
  [多模态] 提出首个端到端 Web 数据科学基准 WebDS（870 个任务，29 个网站，10 个领域），当前最强 Agent（BrowserUse + GPT-4o）仅完成 15% 的任务，而人类达到 90%，揭示了真实数据科学工作流中 Agent 的巨大性能差距。
tags:
  - 多模态
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

WebDS 是首个评估完整数据科学管道的基准：**Web 浏览获取数据** → **数据分析/可视化** → **生成有理据的输出**。

### 基准设计

**基于专家访谈**：对 8 名记者、数据科学家和领域专家进行访谈，归纳出两大任务类别：
- 产出下游产品的任务（报告、可视化）
- 解答关键分析问题的任务

**网站覆盖**：29 个数据丰富的网站（CDC、政府数据门户、新闻媒体等），跨 10 个高风险领域，涵盖结构化数据（CSV、表格）和非结构化数据（文本、图形）。

### 任务设计

870 个人工编写的任务，每个任务标注 7 种属性：

- **QA vs Action**：问答任务（344 单跳 + 117 多跳）vs 行动任务（97 单跳 + 134 多跳 + 139 需工具）
- **单跳 vs 多跳**：是否需要组合多个数据源
- **结构化 vs 非结构化**：数据格式
- **工具使用**：是否需要 Python、SQL 等外部工具
- **Web 导航**：是否需要网站交互
- **多网站**：是否涉及跨网站信息聚合

### 难度分类

$$
\text{Difficulty} = \begin{cases}
\text{Easy (247)} & \text{不含多跳/非文本/行动/工具，单网站} \\
\text{Medium (275)} & \text{恰好包含上述一项，单网站} \\
\text{Hard (348)} & \text{包含两项以上或多网站}
\end{cases}
$$

### 双轨评估

- **WebDS-live**：直接在真实网站上交互，捕捉真实 Web 复杂性
- **WebDS-dockerized**：容器化部署子集网站，确保可复现性

### 评估协议

1. **自动二元评估**：有参考答案的任务，LLM 比较输出与答案 → SUCCESSFUL/UNSUCCESSFUL
2. **LLM 主观评分（1-5）**：扩展 WebVoyager 方法，评估完整轨迹而非仅最终截图，提供五级评分 + 失败分析
3. **人工验证**：400 对任务-轨迹独立评审，评估体系与人工达到 93% 一致率

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

## 相关论文

- [MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/multimodal_vlm/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)
- [VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)

<!-- RELATED:END -->
