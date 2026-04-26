---
title: >-
  [论文解读] InductionBench: LLMs Fail in the Simplest Complexity Class
description: >-
  [ACL 2025][LLM/NLP][归纳推理] 本文提出 InductionBench，一个基于子正则函数层次（subregular hierarchy）的归纳推理基准，揭示即使是最强的 LLM（如 o3-mini）也难以掌握最简单复杂度类的归纳推理任务，暴露了当前 LLM 在从观测数据中归纳规则方面的根本缺陷。
tags:
  - ACL 2025
  - LLM/NLP
  - 归纳推理
  - 基准测试
  - 子正则层次
  - 函数学习
  - LLM评估
---

# InductionBench: LLMs Fail in the Simplest Complexity Class

**会议**: ACL 2025  
**arXiv**: [2502.15823](https://arxiv.org/abs/2502.15823)  
**代码**: [GitHub](https://github.com/Wenyueh/inductive_reasoning_benchmark)  
**领域**: LLM/NLP  
**关键词**: 归纳推理、基准测试、子正则层次、函数学习、LLM评估  

## 一句话总结

本文提出 InductionBench，一个基于子正则函数层次（subregular hierarchy）的归纳推理基准，揭示即使是最强的 LLM（如 o3-mini）也难以掌握最简单复杂度类的归纳推理任务，暴露了当前 LLM 在从观测数据中归纳规则方面的根本缺陷。

## 研究背景与动机

**领域现状**：现有的 LLM 推理基准大多聚焦于演绎推理（deductive reasoning），如数学证明、代码生成等——这些任务中规则（公理、语法）是明确给定的，LLM 只需要在已知规则上做搜索和规划。o1、o3 等推理模型在这类任务上已经取得了接近饱和的表现。

**现有痛点**：归纳推理（inductive reasoning）——即从有限观测中推断出底层规则——是科学发现的核心能力，但现有基准对这一能力的评估非常不足。少数涉及归纳的测试（如 ARC）缺乏严格的形式化框架，难以系统地评估 LLM 的归纳能力边界。

**核心矛盾**：我们不知道 LLM 的归纳推理能力究竟有多强——如果无法掌握形式语言理论中最简单的函数类别，那更复杂的科学归纳更无从谈起。需要一个基于计算复杂度理论的系统化基准。

**本文目标**：（1）构建一个基于子正则层次的归纳推理基准；（2）系统评估 LLM 在不同复杂度类上的归纳推理能力；（3）量化假设空间大小对推理难度的影响。

**切入角度**：作者从形式语言理论中的子正则层次出发——这个层次定义了字符串到字符串变换函数的复杂度等级序列，从最简单的 Input Strictly Local (ISL) 到更复杂的类别。如果 LLM 连最底层都学不会，就说明其归纳能力存在根本性不足。

**核心 idea**：用子正则层次中的函数类作为归纳推理的测试床，通过控制函数类别（ISL → OSL → ...）和假设空间参数（k、词表大小、规则数量）来精确测量 LLM 的归纳推理能力边界。

## 方法详解

### 整体框架

InductionBench 的设计基于形式语言理论中的子正则层次。整体流程为：（1）选定一个函数类（如 ISL-k）和假设空间参数；（2）从该类中随机采样一个函数作为目标；（3）生成若干输入-输出对作为示例；（4）要求 LLM 根据这些示例推断出底层规则并预测新输入的输出。基准分为 Standard Benchmark（有多项式时间可证正确算法的函数类）和 Exploration Benchmark（没有已知高效算法的更复杂类别）。

### 关键设计

1. **子正则函数层次体系**:

    - 功能：提供一个从简到繁的函数复杂度分类框架
    - 核心思路：子正则层次将字符串变换函数按局部性和方向性分层。最简单的 ISL-k（Input Strictly Local）函数仅依赖输入字符串中长度为 k 的局部窗口来决定每个位置的输出；OSL-k（Output Strictly Local）则依赖输出端的局部窗口。更高阶的类别如 Subsequential 和 Regular 函数具有更长程依赖。每个层级都有严格的形式化定义和包含关系。
    - 设计动机：这种层次结构使得我们可以精确控制任务难度——如果 LLM 在层次底部就失败了，就不需要测试更高层级，从而快速定位能力边界

2. **可控假设空间参数化**:

    - 功能：通过参数组合精确控制每个测试实例的难度
    - 核心思路：每个函数类由三个参数控制——k（局部窗口大小）、词表大小 $|\Sigma|$、规则数量 $n$。假设空间大小可以精确计算为 $\binom{|\Sigma|^k}{n} \times (|\Sigma|)^n$。通过遍历这些参数的组合，可以绘制出 LLM 性能随假设空间大小变化的精确曲线。
    - 设计动机：现有归纳推理评估缺乏对任务难度的量化控制。参数化设计让我们可以区分"模型完全不会归纳"和"模型只在简单情况下能归纳"

3. **双层基准设计（Standard + Exploration）**:

    - 功能：区分"已知可解"和"未知难度"的归纳推理任务
    - 核心思路：Standard Benchmark 包含 ISL 和 OSL 等存在已知多项式时间学习算法的函数类，提供理论上的性能上界参考。Exploration Benchmark 包含更复杂的函数类（如 non-monotone subsequential），没有已知的高效可证正确算法。两层基准的对比可以揭示 LLM 是否能通过"暴力搜索"或"模式匹配"来部分解决问题。
    - 设计动机：通过区分有算法解和无算法解的类别，可以更好地理解 LLM 的推理策略是否接近于系统性搜索还是仅仅是浅层模式匹配

### 损失函数 / 训练策略

本文为评估性基准，不涉及训练。评估指标为加权准确率，权重与假设空间大小成反比（越难的设置权重越大），也提供 log 权重的变体以减少极端设置的影响。

## 实验关键数据

### 主实验

| 模型 | ISL (Standard) | OSL (Standard) | ISL (Exploration) |
|------|----------------|-----------------|---------------------|
| Llama-3.1 8B | 0.00 | 0.00 | 0.00 |
| Qwen2.5-Coder-32B | 0.073 | 0.007 | 0.0003 |
| Llama-3.3 70B | 0.066 | 0.058 | 0.001 |
| DeepSeek-R1-Distill-70B | 0.038 | 0.051 | 0.008 |
| o3-mini | 0.289 | 0.431 | 0.057 |

### 消融实验（假设空间大小影响）

| 参数设置 (k, vocab, rules) | 假设空间大小 | o3-mini 准确率 | 说明 |
|---------------------------|-------------|---------------|------|
| (2, 2, 1) 最简单 | ~6 | ~0.8+ | 几乎可解 |
| (3, 3, 2) 中等 | ~数百 | ~0.3 | 急剧下降 |
| (4, 4, 3) 较难 | ~数万 | <0.1 | 几乎失败 |
| (4, 4, 4) 最难 | ~数十万 | ~0.01 | 完全不可行 |

### 关键发现

- **所有模型在最简单的 ISL 类上表现都极差**：即使是 o3-mini，加权准确率也仅 0.29，说明归纳推理确实是当前 LLM 的根本弱点
- **假设空间大小是决定难度的主要因素**：性能随假设空间指数级下降，LLM 无法有效搜索组合空间
- **推理增强模型（如 o3-mini）有明显优势但远不够**：o3-mini 在简单设置下有一定能力，但假设空间稍大就崩溃
- **小模型（8B）完全无能力**：Llama-3.1 8B 在所有设置上获得 0 分

## 亮点与洞察

- **用形式语言理论严格定义归纳推理难度等级**——这比 ARC 等基于直觉设计的基准更有科学基础，因为每个函数类的复杂度和假设空间大小都可以精确计算，实现了真正的可控评估
- **揭示了 LLM 推理能力的根本盲区**——现有讨论中 LLM 推理能力的炒作集中在演绎推理上，本文清晰地指出归纳推理是一个完全不同的、且当前远未解决的维度
- **参数化的假设空间控制方法**可以迁移到其他认知能力评估中——比如用类似框架评估 LLM 的概念学习、因果推断等能力

## 局限与展望

- 目前仅测试了字符串到字符串的变换函数，未覆盖更自然的归纳推理场景（如从图像序列中归纳规则）
- 子正则层次只是形式语言理论的一个子领域，更复杂的函数类（如图灵可计算函数）未被涉及
- 评估是零样本/少样本的 prompt-based，未探索 fine-tuning 或 in-context learning 的长序列变体是否能改善表现
- 未分析 LLM 失败的具体原因——是搜索策略不当还是根本缺乏归纳归约能力

## 相关工作与启发

- **vs ARC Benchmark**: ARC 也测试归纳推理但基于视觉模式，缺乏形式化复杂度分级。InductionBench 的优势在于理论根基更扎实，可以精确量化难度
- **vs BIG-Bench (Induction tasks)**: BIG-Bench 中的归纳任务偏向于简单的序列外推，没有系统性的复杂度层次
- **vs Program Synthesis**: 程序综合也涉及从示例中归纳规则，但通常有明确的 DSL 约束。InductionBench 更关注纯粹的归纳能力而非编程能力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将子正则层次引入 LLM 评估，角度独特且有深刻理论基础
- 实验充分度: ⭐⭐⭐⭐ 覆盖多个模型和参数组合，但缺乏 fine-tuning 和长 context 的探索
- 写作质量: ⭐⭐⭐⭐ 形式化定义清晰，但对非形式语言理论背景的读者门槛较高
- 价值: ⭐⭐⭐⭐⭐ 指出了 LLM 能力的一个根本性盲区，对未来推理增强研究有重要启示

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)
- [\[ICLR 2026\] Trapped by simplicity: When Transformers fail to learn from noisy features](../../ICLR2026/llm_nlp/trapped_by_simplicity_when_transformers_fail_to_learn_from_noisy_features.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[NeurIPS 2025\] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](../../NeurIPS2025/llm_nlp/c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)

<!-- RELATED:END -->
