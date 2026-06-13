---
title: >-
  [论文解读] Recommendations and Reporting Checklist for Rigorous & Transparent Human Baselines in Model Evaluations
description: >-
  [ICML 2025][推荐系统][Human Baseline] 本文对 AI 评估中"人类基线"（human baseline）的方法论进行了系统审查，发现现有 115 项人类基线研究在严谨性和透明度方面存在严重不足，并提出了覆盖基线全生命周期的方法建议和报告清单。
tags:
  - "ICML 2025"
  - "推荐系统"
  - "Human Baseline"
  - "AI评估方法论"
  - "评估透明度"
  - "测量理论"
  - "报告规范"
---

# Recommendations and Reporting Checklist for Rigorous & Transparent Human Baselines in Model Evaluations

**会议**: ICML 2025 Spotlight  
**arXiv**: [2506.13776](https://arxiv.org/abs/2506.13776)  
**代码**: [https://github.com/kevinlwei/human-baselines](https://github.com/kevinlwei/human-baselines)  
**领域**: 推荐系统  
**关键词**: Human Baseline, AI评估方法论, 评估透明度, 测量理论, 报告规范

## 一句话总结

本文对 AI 评估中"人类基线"（human baseline）的方法论进行了系统审查，发现现有 115 项人类基线研究在严谨性和透明度方面存在严重不足，并提出了覆盖基线全生命周期的方法建议和报告清单。

## 研究背景与动机

在 AI 评估中，将模型与人类表现进行对比是一种广泛使用的做法。从 ImageNet 到 MMLU 再到各种推理基准，"超越人类表现"往往被视为里程碑式的成就。然而，这些对比的前提是：我们有一个**可靠的人类基线**（human baseline）作为参照。

但现实中，人类基线的构建往往存在严重问题：测试集不一致（AI 和人类在不同子集上评估）、被试样本量不足、缺乏质量控制、不同群体间的方法效应（method effects）未被控制、不确定性未被量化。这导致"AI 超越人类"的声明可能具有误导性——我们并不是在公平条件下进行比较。

这一问题的核心矛盾在于：AI 评估社区越来越重视模型评估的严谨性（如标准化基准、排行榜），**但对人类基线的构建却缺乏同等程度的方法论关注**。一个不严谨的人类基线会导致三种风险：(1) 高估 AI 能力（因为人类基线被低估了），(2) 低估 AI 能力，(3) 误导后续研究方向和政策决策。

本文的切入角度来自测量理论（measurement theory）——社会科学中长期积累的问卷设计、抽样、信效度检验方法论，将其系统性地迁移到 AI 评估场景中。

## 方法详解

### 整体框架

本文以"人类基线的生命周期"为核心组织架构，将整个流程分为五个阶段，每个阶段提出具体的方法论建议：

1. **基线设计与实现**（Baseline Design & Implementation）
2. **被试招募**（Baseliner Recruitment）
3. **基线执行**（Baseline Execution）
4. **基线分析**（Baseline Analysis）
5. **基线文档化**（Baseline Documentation）

### 关键设计

1. **一致性与代表性测试集**: 要求人类基线和 AI 必须在**完全相同的测试集**上评估。如果因成本限制使用子集，则 AI 结果也必须在该子集上计算，且子集应**随机抽取或按难度/主题分层**。设计动机是：很多现有研究让人类在子集上测试，但报告 AI 在全集上的分数，导致不可比。

2. **充分的被试样本量**: 对于通才（非专家）基线，建议进行**统计功效分析**（power analysis），经验法则是需要约 1000 名受试者才能代表美国成年群体。对于专家基线，可以使用便利样本（convenience sample），但需明确定义专家资格标准。设计动机是：很多研究仅用 3-5 名标注者，样本量远不足以代表"人类水平"。

3. **控制方法效应并使用一致任务**: 方法效应指的是由数据收集方式（而非真实能力差异）导致的结果偏差。例如，AI 可以在一次推理中处理整篇文章，但人类可能受到时间压力或界面限制。建议使用**完全相同的任务**（相同的指令、示例、上下文），并随机化题目和选项顺序。设计动机是：人类和 AI 面对不同形式的同一任务时，表现差异可能源于形式而非能力。

4. **控制努力程度**: 在相似的投入水平下进行公平比较——例如相同的时间限制或相当的经济成本。AI 的"努力"可通过推理成本、采样策略等衡量，人类的"努力"则受培训、报酬、时限影响。

5. **不确定性量化**: 报告统计检验、置信区间或性能分布，而非仅报告点估计。使用一致的评估指标和评分方法（如 pass@k、majority vote 在人类和 AI 上统一使用）。

### 报告清单

作者在附录B中提供了详细的报告清单（checklist），涵盖：
- 被试人口统计学特征
- 抽样策略和纳入/排除标准
- 知情同意和伦理审查
- 任务指令和培训材料
- 质量控制措施
- 评分方法和指标定义
- 不确定性度量

## 实验关键数据

### 主实验

作者对 115 项现有人类基线研究进行了系统综述，主要发现：

| 方面 | 问题发现 | 涉及比例 |
|------|---------|---------|
| 测试集一致性 | 人类和AI使用不同测试集 | 大量研究 |
| 样本量 | 被试数量普遍不足 | 多数研究 |
| 抽样策略 | 便利抽样为主，缺乏代表性 | 多数研究 |
| 质量控制 | 缺乏注意力检查、排除标准 | 常见 |
| 不确定性报告 | 仅报告点估计 | 非常普遍 |
| 伦理审查 | 未报告IRB或知情同意 | 显著比例 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 同一测试集 vs 不同测试集 | 比较结论可能反转 | 子集选择偏差可严重影响结果 |
| 3-5人 vs 1000人被试 | 估计误差差异显著 | 小样本下置信区间极宽 |
| 有/无方法效应控制 | 影响未知但可能显著 | 形式差异可能被误认为能力差异 |

### 关键发现

- 现有人类基线在严谨性和透明度方面**系统性不足**，无法为人类-AI 比较提供可靠的参照
- 许多"AI 超越人类"的声明可能建立在不充分的人类基线之上
- 来自测量理论和调查方法学的成熟工具（如分层抽样、功效分析、注意力检查等）在 AI 评估中严重未被利用
- 即使无法达到最大严谨度（如预算限制），研究者也应**明确讨论方法局限性并缩窄结论范围**

## 亮点与洞察

- **跨学科视角的迁移**：将社会科学中数十年积累的测量理论系统性引入 AI 评估，弥补了一个长期被忽视的方法论空白
- **生命周期框架**：从设计、招募到分析、文档化的完整覆盖，不仅提供了"应该做什么"，还解释了"为什么这样做"
- **实用导向**：提供了可直接使用的报告清单（附录B），降低了采纳门槛
- 对 AI 政策和治理具有重要启示：如果人类基线不够可靠，基于此做出的关于 AI 能力的政策判断可能有偏

## 局限与展望

- 115 项研究的综述范围可能不够全面，某些领域（如机器人、自动驾驶）的人类基线可能有不同特点
- 建议虽然全面，但在高成本场景下（如专家评估、长时间任务）的可操作性仍有挑战
- 未深入讨论"人类表现本身是否是一个有意义的参照"——在某些任务上，人类表现可能不代表合理的上界或基准线
- 方法效应在人类和 AI 之间的不对称性（如 AI 不会疲劳、不受顺序效应影响）是一个根本性挑战，文中虽提及但未提供完整解决方案

## 相关工作与启发

- 与 Bowman & Dahl (2021) 关于 NLU 基准设计的反思相呼应
- 补充了 Liang et al. (2023) HELM 框架中对评估标准化的追求——HELM 标准化了模型评估，本文则标准化了人类基线构建
- 启发思考：在推荐系统评估中，离线实验往往使用隐式反馈作为"人类基线"，但这些反馈同样存在代表性、噪声和方法效应等问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 视角新颖，将测量理论系统引入AI评估
- 实验充分度: ⭐⭐⭐⭐ 115项研究的综述提供了扎实的实证基础
- 写作质量: ⭐⭐⭐⭐⭐ 结构极其清晰，分层阅读指南设计贴心，表格信息密度高
- 价值: ⭐⭐⭐⭐⭐ 对整个AI评估社区具有基础性的方法论贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization](recommendations_with_sparse_comparison_data_provably_fast_convergence_for_noncon.md)
- [\[ICML 2025\] RLTHF: Targeted Human Feedback for LLM Alignment](rlthf_targeted_human_feedback_for_llm_alignment.md)
- [\[ICML 2025\] How to Set AdamW's Weight Decay as You Scale Model and Dataset Size](how_to_set_adamws_weight_decay_as_you_scale_model_and_dataset_size.md)
- [\[ICML 2025\] MATCHA: Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition](toward_safe_and_human-aligned_game_conversational_recommendation_via_multi-agent.md)
- [\[ICML 2025\] PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](parm_multi-objective_test-time_alignment_via_preference-aware_autoregressive_rew.md)

</div>

<!-- RELATED:END -->
