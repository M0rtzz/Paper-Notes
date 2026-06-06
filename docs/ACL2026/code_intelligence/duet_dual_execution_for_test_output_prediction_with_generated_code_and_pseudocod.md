---
title: >-
  [论文解读] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode
description: >-
  [ACL 2026][代码智能][测试输出预测] 本文提出 DUET，一个结合直接代码执行和 LLM 伪代码执行的双路框架，通过功能多数投票融合两种互补的执行路径——前者在代码正确时可靠但受实现错误影响…
tags:
  - "ACL 2026"
  - "代码智能"
  - "测试输出预测"
  - "伪代码执行"
  - "双路执行"
  - "代码生成"
  - "功能多数投票"
---

# DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode

**会议**: ACL 2026  
**arXiv**: [2604.11514](https://arxiv.org/abs/2604.11514)  
**代码**: [GitHub](https://github.com/ldilab/DuET)  
**领域**: LLM安全  
**关键词**: 测试输出预测, 伪代码执行, 双路执行, 代码生成, 功能多数投票

## 一句话总结

本文提出 DUET，一个结合直接代码执行和 LLM 伪代码执行的双路框架，通过功能多数投票融合两种互补的执行路径——前者在代码正确时可靠但受实现错误影响，后者绕过实现细节但可能产生执行幻觉——在 LiveCodeBench 测试输出预测上提升 Pass@1 13.6 个百分点。

## 研究背景与动机

**领域现状**：测试用例生成是代码生成管道的关键环节，其中测试输出预测（给定问题描述和测试输入，预测正确输出）是一个需要精确程序推理的难题。TestChain 等方法通过先生成代码再直接执行来进行预测。

**现有痛点**：(1) 直接代码执行的致命问题——模型可能理解了正确的算法逻辑（能生成正确的伪代码）但在实现为可执行代码时引入了细微错误（如用 `len(nums)` 而非 `i+1` 来计算累积平均），导致执行失败或输出错误；(2) 在端到端代码生成中，用生成代码的执行结果来过滤候选程序存在"零优势问题"——如果生成的测试代码也有错，过滤就失效。

**核心矛盾**：直接代码执行依赖代码正确性（确定性但脆弱），LLM 推理不依赖代码但可能产生执行幻觉（灵活但不确定）——两者的失败模式互补。

**本文目标**：(1) 提出 LLM 伪代码执行来解耦正确逻辑和实现错误；(2) 设计双路框架 DUET 融合两种执行路径的互补优势。

**切入角度**：将"代码生成中的逻辑正确性"和"代码实现的正确性"解耦——伪代码捕获算法意图而不受语法细节约束，LLM 在伪代码上的模拟执行可以绕过实现错误。

**核心 idea**：测试输出预测的两种路径——直接执行（代码正确时可靠）和伪代码模拟执行（绕过实现错误但可能幻觉）——是天然互补的，通过功能多数投票可以利用两者的优势。

## 方法详解

### 整体框架

给定问题描述和测试输入：路径 1 生成可执行代码并直接执行获得输出；路径 2 生成伪代码并用 LLM 模拟执行（逐步推理）获得输出。两条路径各采样多次，通过功能多数投票选择最终预测输出。

### 关键设计

1. **LLM 伪代码执行**:

    - 功能：在不依赖代码实现正确性的情况下预测测试输出
    - 核心思路：让 LLM 先生成问题的伪代码描述（高层算法意图），然后用 LLM 对伪代码进行逐步模拟执行——跟踪变量状态并推导最终输出。伪代码提供了比自然语言更精确的算法描述，但比可执行代码更容错
    - 设计动机：实现错误（如循环边界、变量名混淆）是直接执行失败的主要原因，伪代码通过更高层次的抽象绕过这些细节

2. **功能多数投票融合**:

    - 功能：结合两种执行路径的互补优势
    - 核心思路：对代码执行和伪代码执行各采样 N 次（如各 5 次），收集所有 2N 个输出。如果某个输出在两种路径中都出现（或在总投票中获得多数），则更有可能是正确答案。功能多数投票以输出的功能等价性（而非字面相同）为准
    - 设计动机：两种路径的失败模式不同——代码执行在实现错误时失败（如图 2a），伪代码执行在深层嵌套循环时幻觉（如图 2b）。互补融合可以显著提升可靠性

3. **零优势问题的解决**:

    - 功能：解决端到端代码生成中测试过滤失效的问题
    - 核心思路：在 CODET 管道中用预测的测试输出过滤候选代码时，如果测试输出来自生成代码的直接执行，当测试代码有错时过滤就失效（零优势）。伪代码执行路径与代码正确性解耦，因此即使候选代码有错，伪代码路径仍可提供有效的过滤信号
    - 设计动机：TestChain 在 CODET 管道中反而降低了性能（-5.6pp），正是因为零优势问题；DUET 通过伪代码路径避免了这个问题

### 损失函数 / 训练策略

不涉及模型训练。使用现有 LLM（如 Llama-3.1-8B-Instruct）进行推理。代码执行和伪代码执行各采样 5 次（总共 10 次调用），通过功能多数投票选择最终输出。

## 实验关键数据

### 主实验

**LiveCodeBench 测试输出预测**

| 方法 | Pass@1 | 相对提升 |
|------|--------|---------|
| Direct（仅代码执行） | 基线 | - |
| TestChain | +5.6pp | - |
| Pseudocode Exec | 竞争 | - |
| **DUET** | **+13.6pp** | **SOTA** |

**端到端代码生成（CODET 管道 + Llama-3.1-8B-Instruct）**

| 预测方法 | Pass@1 变化 |
|---------|-----------|
| 无过滤 | 基线 |
| TestChain | -5.6pp（零优势问题） |
| **DUET** | **+3.2pp** |

### 消融实验

**在 LiveCodeBench-Easy/BigCodeBench-Hard/DevEval/HumanEval(+) 上的端到端代码生成**

| 方法 | LCB-Easy | BCB-Hard | DevEval | HumanEval(+) |
|------|----------|----------|---------|---------------|
| DUET | **最佳** | **最佳** | **最佳** | **最佳** |

### 关键发现

- DUET 在测试输出预测上 +13.6pp，远超单路方法——两种路径的互补性得到充分验证
- TestChain 在 CODET 管道中反而降低性能 5.6pp，而 DUET 提升 3.2pp——零优势问题是实际部署中的关键障碍
- 实现错误和执行幻觉的分布互补：功能逻辑复杂的问题更易出现实现错误（伪代码路径更可靠），深层嵌套循环更易出现执行幻觉（代码执行更可靠）
- 伪代码的抽象层次使其天然比可执行代码更容错——即使逻辑相同，伪代码的"实现"更不容易出错

## 亮点与洞察

- 将"正确逻辑"和"正确实现"解耦是优雅的问题分析——伪代码执行正是这种解耦的自然产物
- 零优势问题的发现和解决对代码生成管道的实际部署有重要意义
- 功能多数投票是一种通用的融合策略，可扩展到更多执行路径

## 局限与展望

- 伪代码执行需要额外的 LLM 调用（2N 次 vs N 次），计算成本翻倍
- 在极复杂的算法问题上，伪代码和代码可能同时出错
- 伪代码的生成质量依赖 LLM 的元语言能力
- 未探索对伪代码执行进行专门训练以减少执行幻觉

## 相关工作与启发

- **vs TestChain**: TestChain 仅用直接代码执行，受零优势问题影响；DUET 的双路设计解决了这个问题
- **vs AlphaCode**: AlphaCode 仅关注测试输入生成，DUET 同时解决输出预测
- **vs CODET**: CODET 框架依赖准确的测试输出过滤，DUET 提供了更可靠的测试输出

## 评分

- 新颖性: ⭐⭐⭐⭐ 伪代码执行和双路融合的思路新颖，零优势问题的发现有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 多个基准上的测试输出预测和端到端评估全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析深入清晰，示例图解直观易懂
- 价值: ⭐⭐⭐⭐ 对代码生成管道中的测试预测有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] PaT: Planning-after-Trial for Efficient Test-Time Code Generation](pat_planning-after-trial_for_efficient_test-time_code_generation.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[ICML 2026\] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](../../ICML2026/code_intelligence/boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)
- [\[ACL 2026\] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)

</div>

<!-- RELATED:END -->
