---
title: >-
  [论文解读] Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?
description: >-
  [ACL2025][LLM推理][test-time scaling] 系统性地揭示了 QwQ/DeepSeek-R1/LIMO 等 o1-like 模型在测试时并不具备真正的顺序扩展 (sequential scaling) 能力——更长的 CoT 并不带来更高准确率，根因是自我修正 (self-revision) 能力不足——并据此提出 Shortest Majority Vote 并行扩展方法显著超越传统多数投票。
tags:
  - "ACL2025"
  - "LLM推理"
  - "test-time scaling"
  - "o1-like models"
  - "self-revision"
  - "majority vote"
  - "chain-of-thought"
---

# Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?

**会议**: ACL2025  
**arXiv**: [2502.12215](https://arxiv.org/abs/2502.12215)  
**代码**: [GitHub](https://github.com/ZhiYuanZeng/test-time-scaling-eval)  
**领域**: LLM推理  
**关键词**: test-time scaling, o1-like models, self-revision, majority vote, chain-of-thought

## 一句话总结

系统性地揭示了 QwQ/DeepSeek-R1/LIMO 等 o1-like 模型在测试时并不具备真正的顺序扩展 (sequential scaling) 能力——更长的 CoT 并不带来更高准确率，根因是自我修正 (self-revision) 能力不足——并据此提出 Shortest Majority Vote 并行扩展方法显著超越传统多数投票。

## 研究背景与动机

OpenAI o1 系列开创了"测试时扩展" (test-time scaling) 新范式，通过在推理阶段分配更多计算资源来持续提升推理能力。o1 的成功催生了 QwQ、DeepSeek-R1、LIMO 等开源复现模型，它们都能生成很长的 CoT，但一个关键问题尚未被充分验证：**这些模型是否真正具备测试时扩展能力？** 即更长的推理链是否必然带来更好的表现？

作者发现一个反直觉的现象：对同一问题，正确解答的平均长度反而**短于**错误解答。这促使他们深入探究 o1-like 模型的测试时扩展机制，区分顺序扩展 (延长 CoT) 和并行扩展 (多次采样取最优) 的实际效果。

## 方法详解

### 1. 顺序扩展的失效分析

**实验设计**：对每个问题采样 5 次解答，按长度排序分为 5 组，分别统计各组的平均长度和准确率。

**核心发现**：
- 最长解答的平均长度约为最短解答的 2 倍，但准确率无显著提升，甚至在 AIME、Omni-MATH 等难题上出现反向扩展 (inverse scaling)
- 对同一问题，正确解答的平均长度始终短于错误解答，且弱模型 (QwQ、R1-Distill-1.5B) 差距更大

### 2. 失效根因：自我修正能力不足

**长短 CoT 差异分析**：长 CoT 中包含更多自我修正标记 ("Wait"、"Alternatively")，解答长度与 "Wait" 出现频率呈强线性相关。

**主动触发自我修正实验**：
- 移除解答末尾的 "final answer" 部分，用 "Wait" 或 "Alternatively" 作为 prompt 继续推理 40 步
- QwQ 和 R1-Distill-1.5b 准确率随修正步数持续下降
- R1-Distill-32b/14b 和 LIMO 初期略有提升，随后振荡不再改善

**修正行为的详细分析**：
- 将错误答案修正为正确的成功修正率 (successful-revision rate) 极低，始终低于 10%
- QwQ 和 R1-Distill-1.5b 的失败修正率 (正确→错误) 甚至高于成功修正率
- 当原始答案错误时，R1-Distill-32b/14b 在超过 70% 的情况下保持原答案不变

### 3. 并行扩展 vs 顺序扩展

在相同 token 预算下比较两种策略：
- **覆盖率 (pass@k)**：并行采样 10 个解答的覆盖率远高于顺序修正 40 步
- **准确率**：Majority Vote (并行) 的可扩展性优于顺序修正
- 顺序扩展因需要对更长上下文做 attention，计算成本也更高

### 4. Shortest Majority Vote

基于"正确解答更短"的洞察，提出改进的并行扩展算法：

对第 i 类答案，令 c_i 为该类解答数量，l_i 为平均长度，则其得分为：

    s_i = c_i / log(l_i)

最终选择得分最高的类别作为答案。

**设计动机**：正确答案更可能出现在解答数量多且长度短的类别中。当候选解答只有 2 个时（传统多数投票失效），解答长度可作为额外的引导信号。

## 实验关键数据

### Table 1: 模型修正时保持原错误答案的比例

| 模型 | 保持原错误答案比例 |
|------|---------------------|
| R1-Distill-32b | 72% |
| R1-Distill-14b | 70% |
| R1-Distill-1.5b | 58% |
| QwQ | 32% |
| LIMO | 54% |

自我修正时大部分情况下模型"固执己见"，即使答案错误也不更改，削弱了顺序扩展的效果。

### Table 2: Shortest Majority Vote vs Majority Vote (16 solutions)

| 模型 | AIME MV | AIME Shortest MV | GPQA MV | GPQA Shortest MV |
|------|---------|-------------------|---------|-------------------|
| R1-Distill-32b | 72.88 | **73.77** | 63.33 | **63.53** |
| R1-Distill-14b | 71.77 | 71.55 | 56.16 | 56.46 |
| R1-Distill-1.5b | 40.00 | **42.22** | 29.59 | **30.20** |
| QwQ | 51.33 | 50.88 | 62.25 | 62.25 |
| LIMO | 68.88 | **70.00** | 55.58 | 55.89 |

Shortest Majority Vote 在多数模型和数据集上优于传统 Majority Vote，尤其在 AIME 上效果显著。

## 亮点

- 用大规模实验系统性证伪了"更长 CoT = 更好推理"的直觉假设
- 将顺序扩展失效的根因精确归因到自我修正能力不足（成功修正率 <10%、高比例保持错误答案不变）
- Shortest Majority Vote 方法简洁优雅：仅用解答长度作为额外信号，无需训练额外的 reward model
- 实验覆盖 QwQ/R1 全系列 + LIMO，跨 4 个 benchmark (MATH-500, AIME, Omni-MATH, GPQA)，结论稳健

## 局限性

- R1-671b 的完整评估受限于计算成本，仅在部分实验中使用
- 实验基于静态模型 checkpoint，未考虑强化学习训练过程中 test-time scaling 行为的动态变化
- Shortest Majority Vote 对具有强顺序扩展能力的模型可能效果有限（此时可能需要反转为 Longest Majority Vote）
- 自我修正能力不足的根因分析停留在现象层面，未深入探讨模型架构或训练方法的影响

## 相关工作

- **并行扩展方法**：Best-of-N Search (Cobbe et al., 2021)、Majority Vote (Wang et al., 2023)、Tree Search (Beam Search, MCTS)
- **顺序扩展/自我修正**：Self-Refine (Madaan et al., 2023)、CRITIC (Gou et al., 2024)；关于自我修正是否有效的争论 (Huang et al., 2024 vs Kumar et al., 2024)
- **o1-like 模型分析**：Wang et al. (2025) 从 underthinking 角度解释类似现象；Chen et al. (2024) 和 Arora & Zanette (2025) 发现缩短 CoT 不影响性能
- **区别**：本文首次系统对比了 o1-like 模型的顺序/并行扩展能力，并提出基于长度信号的改进投票方法

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从实证角度质疑 test-time scaling 的主流假设，发现有价值
- 实验充分度: ⭐⭐⭐⭐⭐ — 多模型多数据集全面覆盖，分析层层深入
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，论证逻辑性强
- 价值: ⭐⭐⭐⭐ — 对理解和改进 o1-like 模型的推理扩展策略有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](mclm_multilingual_test_time_scaling.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](../../NeurIPS2025/llm_reasoning/limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](../../NeurIPS2025/llm_reasoning/atom_of_thoughts_for_markov_llm_testtime_scaling.md)
- [\[ACL 2025\] Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory](rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
