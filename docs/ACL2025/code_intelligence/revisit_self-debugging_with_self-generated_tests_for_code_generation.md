---
title: >-
  [论文解读] Revisit Self-Debugging with Self-Generated Tests for Code Generation
description: >-
   系统性地研究了使用 LLM 自生成测试进行自调试（self-debugging）的效果，发现基于后执行信息的自调试在基础编程问题上反而降低性能（因自生成测试偏差），但基于执行中间状态（in-execution）的自调试可有效规避该偏差，在基础和竞赛题上均有提升。
tags:

---

# Revisit Self-Debugging with Self-Generated Tests for Code Generation

| 项目 | 内容 |
|------|------|
| 标题 | Revisit Self-Debugging with Self-Generated Tests for Code Generation |
| 会议 | ACL 2025 |
| arXiv | 2501.12793 |
| 代码 | - |
| 领域 | nlp_generation |
| 关键词 | code generation, self-debugging, self-generated tests, execution feedback, LLM |

## 一句话总结

系统性地研究了使用 LLM 自生成测试进行自调试（self-debugging）的效果，发现基于后执行信息的自调试在基础编程问题上反而降低性能（因自生成测试偏差），但基于执行中间状态（in-execution）的自调试可有效规避该偏差，在基础和竞赛题上均有提升。

## 研究背景与动机

- **Self-debugging** 是近年来提升 LLM 代码生成质量的热门方法：生成代码 → 执行测试 → 获取反馈 → 修复代码
- 但现有方法（如 Self-Debugging、Reflexion、AlphaCodium）多依赖**预设的 oracle 测试**，现实中往往没有高质量测试用例
- **自生成测试的困境**：用模型自身生成测试来自调试是自然的解决方案，但其有效性未被充分探索
    - Reflexion 虽用自生成测试做反馈，但用 oracle 测试评估修复前代码
    - AlphaCodium 先迭代 oracle 测试再迭代自生成测试
- **核心问题**：自生成测试的质量有限（测试输出准确率仅 ~85%），这对自调试带来什么影响？

## 方法详解

### 整体框架

提出自调试的统一框架并区分两种范式：

1. **Post-Execution Self-Debugging（后执行自调试）**
    - 执行代码后，比较实际输出与预期输出
    - 如果不匹配，将失败的测试用例、实际输出、错误信息作为反馈，让模型修复程序
    - 两种反馈粒度：label（仅告知对错）、detail（包含测试输入/预期输出/实际输出）

2. **In-Execution Self-Debugging（执行中自调试）**
    - 将程序分解为基本块（basic blocks），根据控制流图（CFG）
    - 收集每个基本块执行前后的中间变量状态（execution trace）
    - 模型仅根据测试输入和中间状态判断程序正确性并修复
    - **不使用后执行信息**（不知道最终输出是否匹配）

### 关键区别

- Post-execution 依赖自生成测试的**输出标签**——但标签可能错误
- In-execution 仅关注**执行过程的中间状态**——绕开了标签偏差问题

### 偏差分析框架

定义四种情况：
- **True Positive (TP)**：正确程序通过测试
- **True Negative (TN)**：错误程序未通过测试
- **False Positive (FP)**：错误程序通过了有问题的测试
- **False Negative (FN)**：正确程序因错误测试而未通过

## 实验

### 实验设置

- **模型**：GPT-4o、Claude-3.5-Sonnet、LLaMA-3-70B-Instruct、Qwen2.5-Coder-7B-Instruct
- **基准**：HumanEval(+)、MBPP(+)（基础）、LiveCodeBench（竞赛级，450 题）
- 贪心解码，每题生成 10 个测试用例
- 迭代次数：1-2 轮

### 后执行自调试 + Oracle 测试（对照基线）

使用 oracle 测试时自调试带来一致提升，例如：
- GPT-4o 在 HumanEval：92.1 → 95.1 (+3.0)
- Claude-3.5 在 MBPP+：77.0 → 86.0 (+9.0)

### 后执行自调试 + 自生成测试

**在基础问题上表现糟糕**：
- Claude-3.5 在 HumanEval：94.5 → 87.2 (**-7.3**)
- LLaMA-3-70B 在 HumanEval：79.9 → 73.8 (**-6.1**)
- 所有模型在 HumanEval 上均出现下降

**在竞赛问题上有潜力**：
- GPT-4o 在 LiveCodeBench：46.0 → 49.3 (+3.3)（label 反馈）
- 但 detail 反馈反而导致 easy 题下降

### 自生成测试质量分析

| 模型 | 测试输入准确率 | 测试输出准确率 | 测试套件有效率 |
|------|--------------|--------------|--------------|
| GPT-4o | 97.63% | 89.77% | 59.15% |
| Claude-3.5 | 97.68% | 89.14% | 56.71% |
| LLaMA-3-70B | 94.53% | 84.69% | 49.39% |
| Qwen2.5-Coder-7B | 97.19% | 84.85% | 44.50% |

- 生成测试输入容易，生成正确输出难（~85%）
- 完整测试套件有效率仅 ~50-60%

### 偏差分析的关键发现

- 在 HumanEval/MBPP 上，**False Negative 多于 True Negative**——正确程序被错误测试标记为失败，导致不必要的修改反而引入 bug
- 在 LiveCodeBench 上，**True Negative 占比更高**——因为竞赛题本身通过率低，自测标签更可能正确

### 执行中自调试（In-Execution）

**在基础问题上表现积极**：
- GPT-4o 在 HumanEval/MBPP+：87.8/76.5 → 89.0/79.1（2 轮迭代）
- Qwen2.5-Coder-7B 在 MBPP+：70.6 → 72.0
- 多数模型至少不降

**在竞赛问题上**：
- GPT-4o 在 LiveCodeBench：46.0 → 47.6 (+1.6)

### 对比总结

| 范式 | 基础题 | 竞赛题 |
|------|--------|--------|
| Post-execution + self-tests | ❌ 普遍下降 | ⚠️ label 可提升 |
| In-execution + self-tests | ✅ 温和提升 | ✅ 一致提升 |

## 亮点与洞察

1. **首次系统揭示自生成测试自调试的失败模式**：在基础题上后执行自调试反而有害，这是一个重要且反直觉的发现
2. **偏差分析框架（TP/TN/FP/FN）**精准解释了不一致性的根源：基础题 FN 过高，竞赛题 TN 更合理
3. **In-execution 范式是实用方案**：通过分析中间状态而非依赖不可靠的测试标签，有效绕过偏差
4. **洞察**：自调试的有效性不仅取决于修复能力，还取决于**识别错误反馈的能力**
5. **实验覆盖全面**：4 个模型 × 3 个基准 × 2 种范式 × 2 种反馈粒度

## 局限性

- 仅限 Python 编程任务，未验证多语言场景
- In-execution 需要完整的执行 trace，对复杂程序（深层循环/递归）可能产生过长 trace，截断或压缩策略未讨论
- 自生成测试仅生成 10 个，更多测试是否能减轻偏差未充分探索
- 未探索 oracle 测试 + 自生成测试的混合方案
- 仅使用贪心解码，未考虑采样多个候选程序的场景

## 相关工作

- **代码生成**：CodeT（Chen et al., 2023）用双重一致性筛选；Self-Edit（Zhang et al., 2023a）用示例测试做执行反馈
- **Self-debugging**：Self-Debugging（Chen et al., 2024b）迭代式调试；LDB（Zhong et al., 2024）利用运行时信息调试；Reflexion（Shinn et al., 2023）用自生成测试反馈但评估用 oracle
- **代码质量评估**：EvalPlus（Liu et al., 2023）扩展测试集；LiveCodeBench（Jain et al., 2024）持续收集竞赛题

## 评分 ⭐⭐⭐⭐

研究问题清晰、分析深入（偏差框架）、发现有价值（后执行自调试在基础题有害），提出了可行的替代方案（in-execution），但 in-execution 提升幅度有限，实用性有待进一步验证。

<!-- RELATED:START -->

## 相关论文

- [CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](codedpo_code_alignment.md)
- [A Self-Improving Coding Agent](../../NeurIPS2025/code_intelligence/a_selfimproving_coding_agent.md)
- [MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios](mldebugging_towards_benchmarking_code_debugging_across_multi-library_scenarios.md)
- [MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings](../../AAAI2026/code_intelligence/mose_hierarchical_self-distillation_enhances_early_layer_embeddings.md)

<!-- RELATED:END -->
