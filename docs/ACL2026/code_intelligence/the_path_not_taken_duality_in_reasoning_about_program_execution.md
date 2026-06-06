---
title: >-
  [论文解读] The Path Not Taken: Duality in Reasoning about Program Execution
description: >-
  [ACL 2026][代码智能][程序执行推理] 本文提出程序执行推理的对偶性概念，通过DexBench基准（445个配对实例）联合评估LLM的正向执行推理（预测给定输入下的代码覆盖）和反向反事实推理（推断使执行流转向目标分支的输入变异），发现单一方向上的强表现不能转化为联合评估下的成功…
tags:
  - "ACL 2026"
  - "代码智能"
  - "程序执行推理"
  - "反事实推理"
  - "双路径推理"
  - "代码覆盖"
  - "LLM代码理解"
---

# The Path Not Taken: Duality in Reasoning about Program Execution

**会议**: ACL 2026  
**arXiv**: [2604.20917](https://arxiv.org/abs/2604.20917)  
**代码**: [github.com/sail-ucf/dexbench](https://github.com/sail-ucf/dexbench)  
**领域**: 代码智能 / 程序推理  
**关键词**: 程序执行推理、反事实推理、双路径推理、代码覆盖、LLM代码理解

## 一句话总结
本文提出程序执行推理的对偶性概念，通过DexBench基准（445个配对实例）联合评估LLM的正向执行推理（预测给定输入下的代码覆盖）和反向反事实推理（推断使执行流转向目标分支的输入变异），发现单一方向上的强表现不能转化为联合评估下的成功，揭示了模型对程序因果理解的不足。

## 研究背景与动机

**领域现状**：LLM在软件工程任务中被广泛采用（代码生成、测试、Bug修复等），但其表现不一致——能解决复杂编程问题，却在基本循环推理上失败。这可能源于模仿表面模式而非真正理解程序。近年来的工作开始关注细粒度的推理能力评估，如代码覆盖预测、输入输出映射、中间状态跟踪等。

**现有痛点**：现有的程序执行基准仅在单一测试用例下考察运行时行为（即沿一条执行路径推理），但给定程序可能根据不同输入遍历多条路径。单测试用例评估只能提供模型程序理解能力的狭隘视角。此外，固定输入输出对定义的基准容易被训练数据污染（模型可能记住了答案）。

**核心矛盾**：程序的两条路径共享执行空间，仅在分支点因程序状态不同而分叉。现有评估忽略了执行路径之间的因果关系——仅评估模型能否预测"走了哪条路"，而不评估"为什么走这条路"以及"怎样才能走另一条路"。

**本文目标**：设计一个联合评估框架，同时测试LLM的正向执行推理和反向反事实推理能力，从而更鲁棒地衡量其对程序执行的因果理解。

**切入角度**：两条程序路径共享公共执行空间，仅在分支条件处分叉。作者认为理解程序执行需要同时理解"观察到的行为如何发生"和"在什么条件下执行会转向另一条路径"。

**核心 idea**：定义程序执行推理的对偶性——正向推理预测执行路径的可观察行为，反向推理推断使执行流转向反事实路径的输入变异——两者联合组成双路径推理 $\mathcal{R}_{dual} = \mathcal{R}_{exec} \oplus \mathcal{R}_{cf}$。

## 方法详解

### 整体框架
DexBench包含445个评估实例，每个实例由一对正向和反向推理任务组成。给定程序P和测试输入$I_{exec}$：正向任务要求预测代码覆盖（哪些行被执行）；反向任务要求变异输入$I_{exec}$为$I_{cf}$，使执行流覆盖原本未覆盖的目标分支。两个任务必须同时成功才算双路径推理成功。数据来源于CruxEval（298个，简单控制流）、HumanEval（100个，中等复杂度）和PythonSaga（47个，深度嵌套+递归）。

### 关键设计

1. **正向执行推理（$\mathcal{R}_{exec}$）**:

    - 功能：预测程序在给定输入下的语句覆盖（代码覆盖率预测）。
    - 核心思路：给定程序P和输入$I_{exec}$，模型需预测被执行的行号集合$\phi(\tau_{exec})$。成功标准是精确匹配——至少一个生成的候选答案完全正确地预测了代码覆盖。使用SlipCover工具收集ground-truth覆盖信息。
    - 设计动机：代码覆盖预测要求模型维持程序状态的表示并根据每条语句的语义进行更新，是评估执行推理的自然任务。难度随执行轨迹长度和状态转换复杂度（如非平凡控制流）增加。

2. **反向反事实推理（$\mathcal{R}_{cf}$）**:

    - 功能：推断需要如何变异原始输入，使程序执行流转向覆盖特定目标分支。
    - 核心思路：给定程序P、原始输入$I_{exec}$和反事实目标（覆盖特定未覆盖分支b），模型需生成变异输入$I_{cf}$，使得$b \in \phi(\tau_{cf})$。选择覆盖增量最大的分支作为反事实目标。通过实际执行生成的输入来验证正确性。
    - 设计动机：与覆盖引导的模糊测试不同，DexBench要求模型通过推理而非随机变异来识别必要的输入变化。反事实推理要求分析输入变化如何影响控制流和程序状态，是测试因果理解的关键任务。

3. **双路径推理（$\mathcal{R}_{dual}$）**:

    - 功能：联合评估正向和反向推理能力，提供对程序执行理解的鲁棒评估。
    - 核心思路：$S_{dual} = S_{exec} \wedge S_{cf}$——一个实例在双路径评估下成功当且仅当模型同时正确预测了原始覆盖并生成了至少一个覆盖目标分支的输入。这避免了单路径评估的狭隘性和不完整性。
    - 设计动机：实验发现许多模型仅在一个方向上成功——能预测覆盖但无法变异输入（或反之），表明单路径评估会高估模型的程序理解能力。

### 损失函数 / 训练策略
DexBench是评估基准，不涉及训练。评估使用pass@k指标（k=1和k=5），采用one-shot prompting。

## 实验关键数据

### 主实验（双路径推理 pass@5，%）

| 模型 | CruxEval | HumanEval | PythonSaga |
|------|----------|-----------|------------|
| Llama-3.2-3B-Inst. | 0.0 | 0.0 | 0.0 |
| Qwen2.5-32B | 57.0 | 31.0 | 6.4 |
| QwQ-32B | 33.2 | 25.0 | 2.1 |
| Gemini 2.5 Flash | 73.8 | 62.0 | 8.5 |
| GPT-5 Mini | 91.6 | 76.0 | 44.7 |
| Claude Sonnet 4 | **95.3** | **79.0** | **70.2** |

### 消融实验（输入生成 vs 输入变异，GPT-5 Mini）

| 设置 | CruxEval pass@5 | HumanEval pass@5 | PythonSaga pass@5 |
|------|-----------------|-------------------|-------------------|
| 输入生成（无原始输入） | 97.7 | 99.0 | 95.7 |
| 输入变异（本文） | 97.0 | 88.0 | 95.7 |

### 关键发现
- **单一方向的成功不转化为联合成功**：许多模型在执行推理或反事实推理中的一个方向表现良好，但在双路径评估下大幅下降，证明单路径评估的不完整性
- **推理后训练无效**：在开源模型中，非推理变体（Qwen2.5-32B、Mistral Small 24B）在双路径推理上反而优于其推理变体（QwQ-32B、Magistral Small），平均高出63.9%和47.9%
- **规模效应非单调**：Qwen2.5-32B在双路径推理上一致优于Qwen2.5-72B；所有<10B模型在执行推理上近乎为零但在反事实推理上有一定表现
- **复杂度影响显著**：从CruxEval到PythonSaga，双路径性能大幅下降（如Gemini 2.5 Flash从73.8%降至8.5%）
- **输入变异比输入生成更难**：去掉原始输入后模型表现反而更好，因为输入变异需要推理输入扰动如何在中间程序状态中传播

## 亮点与洞察
- **对偶性框架**：程序执行推理天然具有正向/反向的对偶结构，联合评估比单独评估更能反映真实理解。这个思路可以推广到其他推理任务——如数学推理的"正向求解"与"反向构造题目"。
- **反事实推理作为因果理解的探针**：与随机模糊测试不同，要求模型通过推理而非试错来识别必要的输入变化，直接检验了因果理解能力。
- **推理后训练的反直觉发现**：推理专用的后训练不仅未提升程序执行推理，反而造成退化，暗示当前推理增强策略可能过度偏向特定推理模式。

## 局限与展望
- 仅覆盖Python语言，扩展到其他语言需要构建安全的沙箱运行环境
- 数据来自公开基准，存在潜在的数据污染风险（但反事实推理的设计部分缓解了此问题）
- 默认选择覆盖增量最大的反事实路径，可能偏向更易推理的分支
- 目前仅以代码覆盖为可观察行为，未来可扩展到程序输出、中间状态等其他执行属性

## 相关工作与启发
- **vs CruxEval**：CruxEval配对输出预测与输入预测，但每个方向独立评估。DexBench通过共享执行上下文联合评估因果推理
- **vs R-Eval (IC-Score)**：R-Eval从多个执行任务组合IC-Score，但仍是单路径评估。DexBench引入反事实维度
- **vs CES**：CES评估跨多个输入的推理一致性，但独立处理各路径，不探索导致路径分叉的因果逻辑

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 对偶性框架是评估程序推理的全新视角，设计精巧
- 实验充分度: ⭐⭐⭐⭐ 13个模型、三种复杂度、完整的敏感性分析，但数据规模偏小（445实例）
- 写作质量: ⭐⭐⭐⭐⭐ 形式化定义清晰，实验分析深入
- 价值: ⭐⭐⭐⭐ 为代码LLM评估提供了更鲁棒的框架，揭示了推理后训练的反直觉现象

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](../../ICML2026/code_intelligence/boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)
- [\[NeurIPS 2025\] Once Upon an Input: Reasoning via Per-Instance Program Synthesis](../../NeurIPS2025/code_intelligence/once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing](to_diff_or_not_to_diff_structure-aware_and_adaptive_output_formats_for_efficient.md)
- [\[ICML 2025\] Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation](../../ICML2025/code_intelligence/reasoning_through_execution_unifying_process_and_outcome_rewards_for_code_genera.md)

</div>

<!-- RELATED:END -->
