---
title: >-
  [论文解读] Evaluating Judges as Evaluators: The JETTS Benchmark of LLM-as-Judges as Test-Time Scaling Evaluators
description: >-
  [ICML 2025][LLM推理][LLM-as-Judge] 本文提出 JETTS 基准，系统评估 LLM-judge 在 test-time scaling 场景（response reranking、step-level beam search、critique-based refinement）中作为评估器的表现，发现 judge 在 reranking 中与 outcome reward model 竞争力相当但在 beam search 中显著弱于 process reward model，且自然语言 critique 目前无法有效引导生成器改进。
tags:
  - ICML 2025
  - LLM推理
  - LLM-as-Judge
  - Test-Time Scaling
  - Reward Model
  - Reranking
  - Beam Search
  - Critique
  - JETTS
---

# Evaluating Judges as Evaluators: The JETTS Benchmark of LLM-as-Judges as Test-Time Scaling Evaluators

**会议**: ICML 2025  
**arXiv**: [2504.15253](https://arxiv.org/abs/2504.15253)  
**代码**: 有（见论文）  
**领域**: LLM推理 / 评估系统 / Test-Time Scaling  
**关键词**: LLM-as-Judge, Test-Time Scaling, Reward Model, Reranking, Beam Search, Critique, JETTS  

## 一句话总结

本文提出 JETTS 基准，系统评估 LLM-judge 在 test-time scaling 场景（response reranking、step-level beam search、critique-based refinement）中作为评估器的表现，发现 judge 在 reranking 中与 outcome reward model 竞争力相当但在 beam search 中显著弱于 process reward model，且自然语言 critique 目前无法有效引导生成器改进。

## 研究背景与动机

Test-time scaling（推理时计算扩展）是提升 LLM 性能的重要范式。其核心组件是**评估器**——用于判断候选回答质量。现有评估器分为两类：

**Reward Models (RM)**：非生成式，输出标量分数
   - Outcome RM (ORM)：对完整回答评分
   - Process RM (PRM)：对推理步骤逐步评分

**LLM-Judges**：生成式，输出自然语言评价和评分

LLM-judge 的独特优势在于能生成**自然语言 critique**，可解释性强。然而，它们在 test-time scaling 场景下的有效性从未被系统评估。核心问题：

- LLM-judge 能否替代 reward model 作为 test-time scaling 的评估器？
- Judge 的自然语言 critique 能否帮助生成器改进回答？

## 方法详解

### JETTS 基准设计

JETTS 覆盖三个领域和三种任务设置：

**三个领域**：
- 数学推理（GSM8K, MATH500）
- 代码生成（MBPP+, HumanEval+）
- 指令遵循（IFEval）

**三种任务设置**：

#### 1. Response Reranking

生成器产生 $N$ 个候选回答，评估器对每个完整回答打分，选出最佳：

$$\hat{y} = \arg\max_{y_i \in \{y_1, \ldots, y_N\}} s(y_i | x)$$

其中 $s(\cdot)$ 是评估器的打分函数。LLM-judge 通过 prompt 生成评价并提取分数。

#### 2. Step-Level Beam Search

将推理过程分解为多步，每步维护 $k$ 个最优路径。评估器对每步评分，剪枝低质量路径：

$$\text{beam}_t = \text{TopK}\left(\bigcup_{b \in \text{beam}_{t-1}} \{b \oplus s_t^{(j)}\}_{j=1}^{m}, k\right)$$

此任务要求评估器具备**过程级评估能力**，能判断中间推理步骤的质量。

#### 3. Critique-Based Response Refinement

Judge 对初始回答生成自然语言 critique，然后将 critique 反馈给生成器进行回答改进。这是 LLM-judge 独有的能力（RM 无法提供文本反馈）：

$$y_{\text{refined}} = \text{Generator}(x, y_{\text{initial}}, \text{critique})$$

### 评估规模

- **10 种 Judge 模型**：7B–70B 参数，包括通用 LLM 和专门训练的 judge
- **8 种 Generator 模型**：6.7B–72B 参数
- 对照组：ORM 和 PRM

## 实验

### 主实验：Reranking 性能

| 评估器类型 | 数学推理 Acc | 代码生成 Acc | 指令遵循 Acc |
|-----------|------------|------------|------------|
| 随机选择（基线） | 45.2% | 62.1% | 55.8% |
| Outcome RM | 68.3% | 74.5% | 71.2% |
| 最佳 LLM-Judge | 66.8% | 73.1% | 70.5% |
| Oracle（上界） | 82.1% | 86.3% | 84.7% |

**发现**：LLM-judge 在 reranking 中与 ORM 表现接近，差距在 2% 以内。

### 主实验：Beam Search 性能

| 评估器类型 | 数学推理（beam=4） | 数学推理（beam=8） |
|-----------|------------------|------------------|
| Process RM | 72.5% | 76.8% |
| 最佳 LLM-Judge | 58.3% | 61.2% |
| Outcome RM | 55.1% | 57.6% |

**发现**：LLM-judge 在 beam search 中**显著弱于** PRM，差距可达 15+ 个百分点。Judge 缺乏对中间推理步骤的精细评估能力。

### 主实验：Critique Refinement 效果

| 设置 | 改进率 | 退步率 | 净效果 |
|------|--------|--------|--------|
| 无 critique（重新生成） | 35% | 30% | +5% |
| LLM-Judge critique | 38% | 33% | +5% |
| Oracle critique | 62% | 12% | +50% |

**发现**：Judge 的自然语言 critique 对生成器的引导效果与不提供 critique 时几乎无差异。Critique 的改进潜力远未被释放。

### 关键消融

- Judge 规模越大，reranking 性能越好，但 beam search 改进有限
- 专门训练的 judge 比通用 LLM 在评分一致性上更优
- Critique 的具体性与改进效果正相关，但当前 judge 生成的 critique 往往过于笼统

## 亮点与洞察

- **首个系统评估 LLM-judge 在 test-time scaling 中表现的基准**
- 揭示了 judge 的"评分能力-critique 能力"不对称：能给出合理分数但无法生成有效改进建议
- Beam search 中 judge 的失败暴露了其**缺乏过程级推理评估能力**——这是 PRM 的核心优势
- 提示未来方向：训练具有过程级评估和高质量 critique 生成能力的 judge

## 局限性

- 评估主要基于准确率指标，未深入分析 judge 的失败模式
- Critique refinement 仅测试单轮改进，未探索多轮迭代
- 未测试 GPT-4、Claude 等闭源强模型作为 judge 的表现
- Judge 的 prompt 设计对结果影响较大，未充分探索最优 prompt

## 相关工作与启发

- **LLM-as-Judge (Zheng et al., 2024)**：使用 LLM 进行自动评估的范式
- **Process Reward Models (Lightman et al., 2023)**：对推理步骤逐步标注和评分
- **Test-Time Scaling (Snell et al., 2024)**：推理时计算扩展的理论和实践
- 本文桥接了 LLM-judge 和 test-time scaling 两个方向，揭示了当前 judge 的能力边界

## 评分

⭐⭐⭐⭐ — 实验全面系统，结论清晰有实践指导意义，揭示了 LLM-judge 在 test-time scaling 中的关键短板

<!-- RELATED:START -->

## 相关论文

- [Atom of Thoughts for Markov LLM Test-Time Scaling](../../NeurIPS2025/llm_reasoning/atom_of_thoughts_for_markov_llm_testtime_scaling.md)
- [Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?](../../ACL2025/llm_reasoning/revisiting_the_test-time_scaling_of_o1-like_models_do_they_truly_possess_test-ti.md)
- [Are Large Reasoning Models Good Translation Evaluators? Analysis and Performance Boost](../../NeurIPS2025/llm_reasoning/are_large_reasoning_models_good_translation_evaluators_analysis_and_performance_.md)
- [Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](../../ACL2025/llm_reasoning/mclm_multilingual_test_time_scaling.md)

<!-- RELATED:END -->
