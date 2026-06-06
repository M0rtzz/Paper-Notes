---
title: >-
  [论文解读] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4
description: >-
  [ACL 2026][代码智能][自动定理证明] DAP 提出了 Hard Mode ATP 的概念（AI 必须自行发现答案再构造证明，而非使用嵌入答案的 Easy Mode 声明），发布了 MiniF2F-Hard 和 FIMO-Hard 基准…
tags:
  - "ACL 2026"
  - "代码智能"
  - "自动定理证明"
  - "Hard Mode"
  - "Lean 4"
  - "答案发现"
  - "形式化验证"
---

# Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4

**会议**: ACL 2026  
**arXiv**: [2604.15839](https://arxiv.org/abs/2604.15839)  
**代码**: [GitHub](https://github.com/liuchengwucn/discover-and-prove)  
**领域**: LLM Agent  
**关键词**: 自动定理证明, Hard Mode, Lean 4, 答案发现, 形式化验证

## 一句话总结
DAP 提出了 Hard Mode ATP 的概念（AI 必须自行发现答案再构造证明，而非使用嵌入答案的 Easy Mode 声明），发布了 MiniF2F-Hard 和 FIMO-Hard 基准，并设计了"发现+证明"两阶段框架——用 LLM 自然语言推理发现答案后改写为 Easy Mode 声明交给形式化证明器，在 CombiBench 上将解题数从 7 提升到 10，首次在 PutnamBench Hard Mode 上证明 36 个定理。

## 研究背景与动机

**领域现状**：自动定理证明（ATP）取得了快速进展，Seed-Prover 在 MiniF2F 上趋近饱和。但现有基准普遍采用"Easy Mode"——将最终答案嵌入形式化声明中——这降低了任务难度，因为人类参赛者必须自己发现答案。

**现有痛点**：(1) Easy Mode 大幅降低了问题难度——对许多竞赛题，发现答案才是主要挑战，知道答案后的证明相对简单；(2) 部分形式化声明与原题语义不完全对齐——如只证明了单方向蕴含而原题要求充要条件；(3) LLM 在非形式推理上超过 80% 答案准确率，但形式化证明器只能处理不到 10%，暴露了巨大的能力差距。

**核心矛盾**：Easy Mode 让 ATP 基准过于乐观地估计了 AI 的数学能力，因为它省略了人类解题中最具挑战性的"发现"环节。

**本文目标**：(1) 建立更公平的 Hard Mode ATP 基准；(2) 设计能处理 Hard Mode 问题的框架。

**切入角度**：将 Hard Mode 问题分解为两步——先用非形式 LLM 推理发现答案（Discovery），再用形式化证明器证明（Proving），模拟人类数学家的思考流程。

**核心 idea**：解耦"发现答案"和"构造证明"——用 LLM 的强非形式推理能力弥补形式化证明器的弱点。

## 方法详解

### 整体框架
两模块管线：(1) Discovery Module——推理 LLM 生成自然语言解题步骤，经自验证和自纠正后，将 Hard Mode 声明改写为 Easy Mode（填入发现的答案）；(2) Proving Module——将改写后的 Easy Mode 声明交给现成 ATP 证明器（Goedel-Prover-V2）生成形式化证明。

### 关键设计

1. **Discovery Module（答案发现）**:

    - 功能：从数学问题中独立发现答案
    - 核心思路：四步流程——(a) 解题生成：推理 LLM（GPT-OSS-120B）生成详细的思维链解题过程；(b) 自验证：LLM 检查自身步骤中的潜在错误并生成错误报告；(c) 自纠正：基于错误报告生成修正后的解答（仅在发现错误时执行）；(d) 改写：将发现的答案填入 Hard Mode Lean 4 声明的第一个 sorry 占位符，生成 Easy Mode 声明。
    - 设计动机：分离发现和证明允许各自使用最强的工具——非形式推理用强推理 LLM，形式化证明用专门的 ATP 系统

2. **Hard Mode 基准数据策展**:

    - 功能：提供更公平的 ATP 评估标准
    - 核心思路：专家重新标注 MiniF2F 和 FIMO——将"答案导向"问题编码为两个 sorry 占位符（第一个填答案，第二个填证明），修正已知的语义对齐问题（如 Easy Mode 只证明单方向而原题要求充要条件），提供 Lean 4 版本的 FIMO。
    - 设计动机：Easy Mode 形式化可能语义弱于原题（如省略了可达性证明），Hard Mode 确保 AI 面对的任务与人类参赛者一致

3. **模块解耦与可扩展性**:

    - 功能：允许独立升级各组件
    - 核心思路：Discovery Module 和 Proving Module 使用不同的 LLM，互不依赖。任何推理模型或 ATP 系统的进步都能直接提升 DAP 的整体性能。
    - 设计动机：当前 LLM 在非形式推理上进步迅速但形式化证明仍有限，解耦设计最大化利用两个方向的进展

### 损失函数 / 训练策略
无训练。Discovery Module 通过精心设计的 prompt 驱动，Proving Module 使用预训练的 Goedel-Prover-V2。

## 实验关键数据

### 主实验

| 基准 | 方法 | 解题数 | 说明 |
|------|------|--------|------|
| CombiBench Hard | 此前SOTA (Kimina) | 7-8 | Pass@16 |
| CombiBench Hard | **DAP** | **10** | 新SOTA |
| PutnamBench Hard | 此前 | 0 (无公开结果) | 首次评估 |
| PutnamBench Hard | **DAP** | **36** | 首个Hard Mode结果 |

### 消融实验

| 配置 | 说明 |
|------|------|
| 仅Discovery（无Proving） | 答案准确率>80%，但无形式化保证 |
| 仅Proving（Easy Mode） | 形式化证明率<10% |
| Discovery+Proving | 两者互补，证明数显著提升 |

### 关键发现
- LLM 在非形式推理上的答案准确率（>80%）远超形式化证明率（<10%），揭示了 Hard Mode 基准独特的度量价值
- DAP 的 Discovery Module 对最终性能贡献最大——错误的答案发现直接导致不可证明的声明
- 自验证和自纠正步骤显著提升答案准确率
- Easy Mode 和 Hard Mode 之间的性能差距在困难问题上更加显著

## 亮点与洞察
- **Easy Mode vs Hard Mode 的区分**是对 ATP 评估方法论的重要贡献——暴露了现有基准可能过于乐观的问题
- 非形式推理 80% vs 形式化证明 10% 的差距量化了"知道答案"与"严格证明"之间的鸿沟
- DAP 框架设计简洁但效果显著——不需要复杂的搜索或 RL 训练，仅靠 prompt 工程和模块解耦

## 局限与展望
- Discovery Module 的答案准确率仍非 100%，错误答案导致的不可证明声明浪费了证明器的计算
- 仅使用单一推理 LLM，集成多个推理模型可能提升答案发现率
- 自验证的可靠性有限——LLM 可能无法检测自身的微妙推理错误

## 相关工作与启发
- **vs DSP/DSP+**: DSP 用自然语言草稿指导形式化证明，DAP 用自然语言发现答案后改写声明——DAP 直接解决 Hard Mode 的答案发现问题
- **vs Seed-Prover**: Seed-Prover 是 lemma 风格的全证明推理模型，DAP 解耦了发现和证明，更灵活
- **vs AlphaProof**: AlphaProof 使用强化学习，DAP 完全开源且基于 prompt，更可复现

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Hard Mode ATP 概念+基准+框架三位一体的贡献
- 实验充分度: ⭐⭐⭐⭐ 多基准评估+消融，但数据规模有限
- 写作质量: ⭐⭐⭐⭐⭐ Easy/Hard Mode 区分的动机论述极为清晰
- 价值: ⭐⭐⭐⭐⭐ 对 ATP 评估方法论和实践都有重要影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Benchmarking Testing in Automated Theorem Proving](benchmarking_testing_in_automated_theorem_proving.md)
- [\[ACL 2026\] FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean](formalscience_scalable_human-in-the-loop_autoformalisation_of_science_with_agent.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ACL 2026\] CuBridge: An LLM-Based Framework for Understanding and Reconstructing High-Performance Attention Kernels](cubridge_an_llm-based_framework_for_understanding_and_reconstructing_high-perfor.md)
- [\[AAAI 2026\] Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study](../../AAAI2026/code_intelligence/why_do_open-source_llms_struggle_with_data_analysis_a_systematic_empirical_study.md)

</div>

<!-- RELATED:END -->
