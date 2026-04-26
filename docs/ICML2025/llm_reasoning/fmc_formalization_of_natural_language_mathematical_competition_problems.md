---
title: >-
  [论文解读] FMC: Formalization of Natural Language Mathematical Competition Problems
description: >-
  [ICML 2025 (AI4MATH Workshop)][LLM推理][Autoformalization] 本文提出基于 LLM 错误反馈的全自动形式化流水线，将自然语言数学竞赛题转化为 Lean 形式化表示，构建了包含 3,922 道自然语言与 9,787 条 Lean 形式化对齐的奥赛级数据集 FMC，并验证了其作为自动定理证明基准的价值。
tags:
  - ICML 2025 (AI4MATH Workshop)
  - LLM推理
  - Autoformalization
  - Lean
  - Error Feedback
  - Olympiad Math
  - Formal Reasoning
  - Theorem Proving
---

# FMC: Formalization of Natural Language Mathematical Competition Problems

**会议**: ICML 2025 (AI4MATH Workshop)  
**arXiv**: [2507.11275](https://arxiv.org/abs/2507.11275)  
**代码**: —  
**领域**: LLM推理 / 数学形式化 / 自动定理证明  
**关键词**: Autoformalization, Lean, Error Feedback, Olympiad Math, Formal Reasoning, Theorem Proving  

## 一句话总结

本文提出基于 LLM 错误反馈的全自动形式化流水线，将自然语言数学竞赛题转化为 Lean 形式化表示，构建了包含 3,922 道自然语言与 9,787 条 Lean 形式化对齐的奥赛级数据集 FMC，并验证了其作为自动定理证明基准的价值。

## 研究背景与动机

形式化数学推理要求将自然语言数学问题精确翻译为形式语言（如 Lean、Isabelle），以便自动定理证明器进行机器验证。这一过程（autoformalization）面临几个挑战：

**人工标注成本极高**：数学竞赛题的形式化需要形式化验证专家，人均标注速度极慢

**现有数据集规模有限**：miniF2F 仅 488 题，ProofNet 仅 371 题，难以支撑大规模训练和评估

**LLM 形式化能力未被充分挖掘**：LLM 能否实现全自动可靠的数学问题形式化？

核心目标：设计无需训练的全自动流水线，利用 LLM 将大量自然语言数学竞赛题转化为高质量 Lean 形式化，构建大规模基准。

## 方法详解

### 整体流水线

```
自然语言数学题 → LLM 初始形式化 → Lean 编译检查 → 错误反馈 → LLM 修正 → 反复迭代 → 最终形式化
```

### 1. 数据收集与预处理

从多个来源收集奥赛级数学竞赛题：
- AMC/AIME（美国数学竞赛）
- IMO Shortlist（国际数学奥林匹克候选题）
- 各国国家级数学竞赛题

总计收集 3,922 道自然语言数学题，覆盖代数、组合、几何、数论四大领域。

### 2. LLM 自动形式化

使用 LLM（如 GPT-4、DeepSeek）将自然语言题目翻译为 Lean 4 形式化表示。关键设计：

**Few-shot Prompting**：提供 3-5 个高质量的"自然语言-Lean"对照示例，帮助 LLM 理解形式化模式：

$$P_{\text{prompt}} = \{(x_1, f_1), (x_2, f_2), \ldots, (x_k, f_k)\} \cup \{x_{\text{target}}\}$$

**多次采样**：对每道题生成多个形式化候选，增加命中概率：

$$\{f_1, f_2, \ldots, f_n\} = \text{LLM}(x, P_{\text{prompt}}, T, n)$$

### 3. 错误反馈机制

Lean 编译器提供精确的语法和类型错误信息。当形式化失败时，将错误信息反馈给 LLM 进行修正：

$$f_{t+1} = \text{LLM}(x, f_t, e_t)$$

其中 $e_t$ 是第 $t$ 轮的编译错误信息。每道题最多迭代 $T$ 轮反馈修正。

### 4. 质量评估

采用多维度评估体系：
- **编译通过率**：形式化是否能被 Lean 编译器接受
- **语义一致性**：形式化是否忠实表达原始题意（由人工或 LLM 评判）
- **质量分级**：将形式化分为"高质量"、"中等质量"、"低质量"三级

## 实验

### 数据集统计

| 指标 | 值 |
|------|------|
| 自然语言数学题 | 3,922 道 |
| Lean 形式化 | 9,787 条（含多个候选） |
| 编译通过率 | ~70%（经错误反馈后） |
| Above-average 质量占比 | 64.46% |
| 覆盖领域 | 代数/组合/几何/数论 |

### 形式化能力对比

| 模型 | 初始编译通过率 | 错误反馈后通过率 | 提升 |
|------|--------------|----------------|------|
| GPT-4 | 45.2% | 68.7% | +23.5% |
| DeepSeek-V2 | 38.6% | 61.3% | +22.7% |
| Claude-3 | 41.8% | 64.2% | +22.4% |
| LLaMA-3-70B | 28.3% | 48.9% | +20.6% |

### 错误反馈轮次分析

| 反馈轮次 | 累计编译通过率 |
|----------|--------------|
| 0（初始） | ~40% |
| 1 | ~55% |
| 2 | ~63% |
| 3 | ~67% |
| 5（饱和） | ~70% |

### 自动定理证明器评估

| 证明器 | FMC 通过率 | miniF2F 通过率 |
|--------|-----------|---------------|
| Lean Tactic | 12.3% | 35.2% |
| DSP-v1.5 | 18.7% | 54.9% |
| ReProver | 15.1% | 42.8% |

FMC 通过率远低于 miniF2F，表明 FMC 更具挑战性。

### 关键发现

- **错误反馈显著提升形式化质量**：平均提升 20+ 个百分点
- **Few-shot > Zero-shot**：提供形式化示例使编译通过率提升约 15%
- **采样数增加有效**：从 1 次到 16 次采样，通过率提升约 25%
- **FMC 比 miniF2F 更具挑战性**：最强证明器通过率不足 20%

## 亮点与洞察

- **全自动无训练**：流水线不需要额外训练，利用现有 LLM 即可完成形式化
- **错误反馈是关键**：Lean 编译器的精确错误信息是提升形式化质量的核心
- **大规模奥赛级形式化数据集**：FMC 规模远超 miniF2F（3,922 vs 488），填补了大规模形式化基准的空白
- 揭示了 LLM 形式化能力的上限：即使用最强模型+错误反馈，通过率仍远未达到 100%

## 局限性

- Lean 编译通过不等于语义正确，部分形式化可能类型检查通过但题意偏差
- 64.46% 的"above-average"质量意味着约 1/3 的数据存在质量问题
- 仅支持 Lean 4，未扩展到 Isabelle、Coq 等其他形式化语言
- 几何题的形式化质量明显低于代数和数论题

## 相关工作与启发

- **miniF2F (Zheng et al., 2022)**：经典小规模形式化数学基准
- **ProofNet (Azerbayev et al., 2023)**：本科数学形式化数据集
- **DeepSeek-Prover (Xin et al., 2024)**：LLM 驱动的形式化定理证明
- 本文的贡献在于 data side：不改进证明器，而是构建更大更难的形式化基准

## 评分

⭐⭐⭐⭐ — 实用性强的大规模形式化数据集，流水线设计简洁有效，为数学形式化推理研究提供了重要基础设施

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Complex Reasoning with Natural Language Contexts and Background Knowledge](../../ACL2025/llm_reasoning/complex_reasoning_with_natural_language_contexts_and_background_knowledge.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](../../NeurIPS2025/llm_reasoning/sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[ACL 2025\] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models](../../ACL2025/llm_reasoning/chain-of-reasoning_towards_unified_mathematical_reasoning_in_large_language_mode.md)
- [\[ACL 2025\] ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](../../ACL2025/llm_reasoning/clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)
- [\[NeurIPS 2025\] I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](../../NeurIPS2025/llm_reasoning/i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)

<!-- RELATED:END -->
