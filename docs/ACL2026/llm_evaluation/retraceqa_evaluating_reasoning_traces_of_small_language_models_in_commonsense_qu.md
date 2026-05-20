---
title: >-
  [论文解读] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering
description: >-
  [ACL 2026][LLM评测][推理过程评估] 本文提出 ReTraceQA，首个面向常识推理任务的推理过程评测基准，包含 2421 条由专家标注的步骤级错误定位和错误分类标注，揭示 14-24% 的 SLM 虽给出正确答案但推理过程有误，当采用推理感知评估替代仅答案评估时…
tags:
  - "ACL 2026"
  - "LLM评测"
  - "推理过程评估"
  - "小语言模型"
  - "常识推理"
  - "过程奖励模型"
  - "LLM-as-Judge"
---

# ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering

**会议**: ACL 2026  
**arXiv**: [2510.09351](https://arxiv.org/abs/2510.09351)  
**代码**: [https://github.com/SapienzaNLP/ReTraceQA](https://github.com/SapienzaNLP/ReTraceQA)  
**领域**: LLM评测/常识推理  
**关键词**: 推理过程评估, 小语言模型, 常识推理, 过程奖励模型, LLM-as-Judge

## 一句话总结

本文提出 ReTraceQA，首个面向常识推理任务的推理过程评测基准，包含 2421 条由专家标注的步骤级错误定位和错误分类标注，揭示 14-24% 的 SLM 虽给出正确答案但推理过程有误，当采用推理感知评估替代仅答案评估时，SLM 性能最多下降 25 个百分点。

## 研究背景与动机

**领域现状**：小语言模型（SLM，≤10B 参数）在各种常识推理基准上表现越来越好，但当前评估实践几乎完全依赖最终答案的正确性——只要模型预测与标准答案匹配就算正确，完全不关注推理过程是否合理。

**现有痛点**：(1) 模型可以通过无效推理路径到达正确答案（如捷径推理、错误前提下的偶然正确），仅答案评估会人为膨胀性能指标；(2) 现有推理过程评测基准（ProcessBench、MR-Ben 等）集中在数学/科学领域，常识推理的过程评估完全空白；(3) 过程奖励模型（PRM）和 LLM 评委主要被用于 Best-of-N 选择以优化性能，而非用于审视正确答案是否经由有效推理路径获得。

**核心矛盾**：SLM 在排行榜上的高分与其真实推理能力之间存在显著落差——答案正确不等于推理正确，但目前的评估体系无法区分两者。

**本文目标**：构建首个面向常识推理的步骤级推理过程评测基准，量化仅答案评估对 SLM 能力的高估程度，并评估 LLM 作为推理评委和 PRM 在常识推理领域的表现。

**切入角度**：关注"过程错误"（process error）——即答案正确但推理过程有误的实例，通过专家标注建立黄金标准，然后用它衡量自动评估方法的可靠性。

**核心 idea**：用 7 个 SLM 在 4 个常识推理数据集上生成 CoT 推理链，经三位博士级专家标注步骤级错误位置和错误类别（Misinterpretation/Hallucination/Reasoning），构建 2421 个实例的基准，在无参考和有参考两种设置下评测 LLM 评委和 PRM。

## 方法详解

### 整体框架

ReTraceQA 的构建流程包括：(1) 从 CSQA、OBQA、QASC、StrategyQA 四个常识推理数据集中选取问题；(2) 用 7 个 SLM（Llama 3.2/3.1、Qwen 2.5、Phi-4-mini 的指令微调版本）通过零样本 CoT 生成推理链；(3) 对推理链进行步骤分割；(4) 平衡采样确保正确/错误链、各模型和问题唯一性的平衡；(5) 三位专家标注每条链的首个错误步骤位置和错误类别。

### 关键设计

1. **三级层次化错误分类体系**:
    - 功能：按错误的认知层次对推理错误进行互斥分类
    - 核心思路：从底层到高层定义三类错误——Misinterpretation（接地层：误解问题、选项含义或任务要求，包括引用不存在的选项或给出多个答案）、Hallucination（内容层：引入经验上错误或不可验证的世界知识，仅当逻辑结构可能正确但事实"积木"有误时使用，如"狼不在北极地区生存"）、Reasoning（推理层：在正确前提之间进行无效逻辑跳跃，如正确陈述"盐降低冰点"后错误推断"这使冰更容易形成"）。分类时遵循"从接地到推理"的优先级。
    - 设计动机：区分"不理解问题"、"不知道事实"和"不会逻辑推理"三种本质不同的失败模式，为改进 SLM 提供有针对性的诊断信息。

2. **首错定位任务定义**:
    - 功能：将推理过程评估形式化为可量化的任务
    - 核心思路：给定问题 $q$ 和推理链 $S = [s_0, s_1, \ldots, s_n]$，预测索引 $i \in \{-1, 0, \ldots, n\}$，其中 $i = -1$ 表示所有步骤正确，$i \geq 0$ 表示首个错误出现在步骤 $s_i$。仅关注首个错误，因为后续步骤建立在错误前提上，其正确性变得模糊。
    - 设计动机：与 ProcessBench 保持一致的任务定义便于跨领域比较，且首错定位避免了级联错误归因的歧义性。

3. **双轴评估框架（无参考 + 有参考 × 评委 + PRM）**:
    - 功能：全面评估自动推理评估方法在常识推理上的表现
    - 核心思路：无参考设置（仅提供推理链，不提供正确答案）测试 LLM 评委和 PRM 作为训练反馈/Best-of-N 选择的可靠性；有参考设置（提供正确答案和推理链）测试其作为评估工具的能力。两种设置下都用 correct（识别全正确链的准确率）、error（定位首错步骤的准确率）和 F1（两者的调和平均）来评估。
    - 设计动机：无参考设置反映实际部署场景（训练时无法获得标准答案），有参考设置反映评估场景，两者结合揭示不同模型在不同条件下的优劣。

### 损失函数 / 训练策略

本文为评测基准论文，不涉及模型训练。LLM 评委使用略微适配的 ProcessBench 提示模板，PRM 使用 sigmoid 激活输出的阈值化判断或 F1 最大化的阈值选择。所有开源模型使用贪心解码，o1-mini 和 DeepSeek-R1 因 API 限制使用温度 1.0。

## 实验关键数据

### 主实验

| 模型 | CSQA F1 | OBQA F1 | QASC F1 | StrategyQA F1 | 平均 F1 |
|------|---------|---------|---------|-------------|--------|
| **有参考 LLM 评委** | | | | | |
| o1-mini | 65.7 | 79.2 | 74.2 | 78.3 | 74.4 |
| GPT-4o | 67.9 | 76.6 | 66.2 | 65.3 | 69.0 |
| Qwen2.5-72B | 64.7 | 69.9 | 69.7 | 67.3 | 67.9 |
| Gemini-2.0-Flash | 65.2 | 74.5 | 68.4 | 62.4 | 67.6 |
| DeepSeek-R1 | 57.4 | 56.4 | 56.7 | 47.2 | 54.4 |
| **无参考 PRM** | | | | | |
| Qwen2.5-Math-PRM-7B | 33.8 | 42.8 | 48.6 | 37.4 | 40.7 |
| Math-Shepherd-PRM-7B | 8.0 | 11.5 | 17.9 | 28.4 | 16.5 |

| SLM 模型 | 仅答案准确率 | 推理感知准确率 | 性能膨胀Δ |
|---------|------------|-------------|---------|
| Qwen2.5-7B | 81.0 | 67.5 | 13.5 |
| Llama-3.1-8B | 76.3 | 63.1 | 13.2 |
| Qwen2.5-3B | 70.4 | 48.5 | 22.0 |
| Llama-3.2-1B | 49.0 | 23.4 | 25.6 |
| 平均 | 68.3 | 49.7 | 18.6 |

### 消融实验

| 数据集 | 过程错误比例（答案正确但推理有误） |
|--------|---------------------------|
| CSQA | 16.3% |
| OBQA | 14.7% |
| QASC | 16.6% |
| StrategyQA | 24.0% |
| 平均 | 17.9% |

### 关键发现

- **17.9% 的正确答案来自错误推理**：平均每 5-6 个"正确"回答中就有一个推理过程有误，StrategyQA 上高达 24%，说明仅答案评估严重高估 SLM 能力。
- **推理感知评估导致性能大幅下降**：使用 o1-mini 作为推理评委后，SLM 平均准确率从 68.3% 降至 49.7%（下降 18.6pp），最差的 Llama-3.2-1B 从 49.0% 降至 23.4%（下降 25.6pp）。
- **幻觉是 SLM 推理的主要失败模式**：幻觉错误占所有错误的 41.9%-62.5%，其次是推理错误（27.9%-35.4%），误解错误占比最低（9.6%-24.1%）。SLM 能理解问题但常制造虚假"事实"。
- **数学 PRM 无法迁移到常识推理**：最强数学 PRM 的平均 F1 仅 40.7%，而最强 LLM 评委达 74.4%，表明 PRM 的泛化能力极为有限。
- **LLM 评委擅长整体判断但弱于错误定位**：模型检测链整体正确性的 correct 分数远高于定位具体错误步骤的 error 分数，说明精确定位推理错误仍是开放挑战。
- **错误多发于中间步骤（3-4 步）**：早期上下文建立通常成功，错误出现在中级推理阶段。o1-mini 的预测分布与人工标注高度吻合，但对后期步骤有过度归因倾向。

## 亮点与洞察

- **首次量化"答案正确≠推理正确"在常识推理中的严重程度**：17.9% 的过程错误率和高达 25pp 的性能膨胀，为社区敲响了警钟——排行榜分数比实际能力高出近 19 个百分点。
- **层次化错误分类的实用价值**：Hallucination > Reasoning > Misinterpretation 的错误分布规律，清晰揭示 SLM 的核心弱点在于事实接地而非逻辑推理或问题理解，为改进方向提供了明确指引。
- **跨领域迁移的警示**：数学 PRM 在常识推理上的惨败（平均 F1 仅 21.1% 无参考）证明了"数学推理≠通用推理"，呼吁构建领域特定的过程奖励模型。
- **标注质量极高**：三位 PhD 级专家标注，Fleiss's Kappa 0.84（"几乎完美一致"），为该领域提供了可靠的黄金标准。

## 局限与展望

- 仅评估了 ≤10B 参数的 SLM，未涉及更大模型的推理过程质量。
- 常识推理的"正确性"本身有主观性——不同标注者可能对某些世界知识的"正确"与否有分歧。
- 仅使用零样本 CoT 生成推理链，未探索少样本或其他提示策略下的推理质量。
- 未来需构建面向常识推理的专用 PRM，而非依赖数学领域的迁移。
- 可扩展到更多推理领域（法律、伦理、社会推理等）。

## 相关工作与启发

- **vs ProcessBench**: ProcessBench 仅覆盖数学推理的错误定位，ReTraceQA 首次将过程评估扩展到常识推理领域。
- **vs MR-Ben/MR-GSM8K**: 这些基准提供错误定位+解释+纠正，但同样局限于数学/科学，ReTraceQA 证明了常识推理需要不同的评估框架。
- **vs MMErroR**: MMErroR 评估 VLM 对给定错误推理链的诊断能力，ReTraceQA 评估对 SLM 自身生成的推理链的过程级评估，两者互补。
- **vs PRM (Math-Shepherd/Qwen2.5-Math-PRM)**: ReTraceQA 的实验证明数学 PRM 无法迁移到常识推理，凸显了领域特定评估的必要性。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个面向常识推理的步骤级推理过程评测基准，问题定义清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个 PRM + 8 个 LLM 评委、无参考/有参考双设置、7 个 SLM 的下游评估，分析极为全面
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，任务定义严谨，统计分析详尽
- 价值: ⭐⭐⭐⭐ 揭示了仅答案评估的严重缺陷，为推理感知评估提供了实用基准和工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)
- [\[ACL 2026\] Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics](evaluating_legal_reasoning_traces_with_legal_issue_tree_rubrics.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)
- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ACL 2026\] Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring](question_difficulty_estimation_for_large_language_models_via_answer_plausibility.md)

</div>

<!-- RELATED:END -->
