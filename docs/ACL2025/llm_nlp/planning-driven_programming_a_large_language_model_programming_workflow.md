---
title: >-
  [论文解读] Planning-Driven Programming: A Large Language Model Programming Workflow
description: >-
  [ACL 2025][LLM 其他][代码生成] 提出 LPW（LLM Programming Workflow），通过"方案生成→计划验证→代码实现→基于计划验证的精准调试"的两阶段工作流，显著提升 LLM 代码生成准确率，在 GPT-4o 上实现 HumanEval 98.2%、MBPP 84.8%、LiveCode 59.3% 的新 SOTA。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "代码生成"
  - "计划验证"
  - "LLM工作流"
  - "程序修复"
  - "测试驱动开发"
---

# Planning-Driven Programming: A Large Language Model Programming Workflow

**会议**: ACL 2025  
**arXiv**: [2411.14503](https://arxiv.org/abs/2411.14503)  
**代码**: [github](https://github.com/you68681/lpw)  
**领域**: LLM/NLP  
**关键词**: 代码生成, 计划验证, LLM工作流, 程序修复, 测试驱动开发

## 一句话总结

提出 LPW（LLM Programming Workflow），通过"方案生成→计划验证→代码实现→基于计划验证的精准调试"的两阶段工作流，显著提升 LLM 代码生成准确率，在 GPT-4o 上实现 HumanEval 98.2%、MBPP 84.8%、LiveCode 59.3% 的新 SOTA。

## 研究背景与动机

LLM 代码生成虽表现出色，但仍面临以下核心挑战：

**调试方向偏离**：现有方法（如 Self-Debugging）基于执行结果和错误解释进行代码修复，但反馈信息中缺乏精确的修正指令，导致修复过程多次偏离预期解决方案。

**计划和测试不可靠**：多智能体协作方法（如 MapCoder）引入额外的可见测试和解决方案计划，但缺乏验证生成的测试和计划正确性的方法论，不正确的计划会误导后续代码生成。

**推理能力有限**：LLM 在严格的词法、语法和语义约束下的代码生成仍具挑战性，且大幅偏离问题描述的程序修复仍是开放问题。

**资源消耗大**：多智能体协作需要大量 token 资源用于通信，效率低下。

核心洞察：人类程序员在编码前会先验证自己的解题思路是否正确（类比测试驱动开发 TDD），而现有 LLM 方法跳过了这一关键步骤。

## 方法详解

### 整体框架

LPW 分为两个阶段：

**阶段一：方案生成（Solution Generation）**
- 计划创建 → 计划验证 → 验证审查 → 迭代修正

**阶段二：代码实现（Code Implementation）**
- 初始代码生成 → 执行测试 → 错误分析（对比计划验证与运行轨迹）→ 代码修复

### 关键设计

1. **解决方案计划（Plan）**：利用 Self-Planning 方法将问题描述分解为若干可处理的子问题（中间步骤），为代码生成提供结构化指导。

2. **计划验证（Plan Verification）**：LPW 的核心创新。对每个可见测试，让 LLM 基于计划进行逐步分析，推导每个中间步骤的输出和最终输出，然后与测试真值对比。这一过程相当于在自然语言层面验证方案的正确性，包含了解决问题所需的完整条件和逻辑规约。

3. **验证审查（Verification Review）**：即使最终输出正确，LLM 还会审查所有中间步骤的结果，检测上下文不一致、数学误算或逻辑缺陷，确保中间结果的准确性（因为后续调试依赖这些中间结果）。

4. **基于计划验证的调试**：当代码在可见测试上失败时，LPW 比较代码执行轨迹（通过自动插入 print 语句获得）与计划验证中记录的期望中间输出，精确定位 bug 位置并生成详细的修复建议（Error Analysis）。结合代码解释（Code Explanation）作为反馈来修复代码。

5. **迭代更新机制**：

    - 方案生成阶段：如果验证输出与测试真值不匹配，自动修订计划；如果中间结果有误，重新生成验证
    - 代码实现阶段：失败则用修复版本替换原代码，迭代直到通过或达到最大迭代次数

### 损失函数 / 训练策略

LPW 是纯推理时方法，不需要额外训练：
- 使用 2-shot prompting
- 最大迭代次数：方案生成和代码实现各 12 次
- 所有组件均由 LLM 通过 few-shot prompting 自主生成
- 仅依赖运行时信息和 LLM 生成信息，无需标注语料

## 实验关键数据

### 主实验

**GPT-3.5 backbone**：

| 方法 | HumanEval | HumanEval-ET | MBPP | MBPP-ET |
|------|-----------|-------------|------|---------|
| Baseline | 74.4 | 66.5 | 67.4 | 52.8 |
| Self-Planning | 77.4 | 69.5 | 69.2 | 52.4 |
| MapCoder | 77.4 | 66.5 | 72.0 | 56.6 |
| Self-Debugging | 81.1 | 72.0 | 71.2 | 56.0 |
| LDB | 82.9 | 72.6 | 72.4 | 55.6 |
| **LPW** | **89.0** | **77.4** | **76.0** | **57.6** |

**GPT-4o backbone（新 SOTA）**：

| 方法 | HumanEval | MBPP | LiveCode | APPS | CodeContests |
|------|-----------|------|----------|------|-------------|
| Baseline | 91.5 | 78.4 | 45.7 | 41.7 | 28.0 |
| LDB | 92.1 | 82.4 | 54.3 | 53.2 | 29.3 |
| **LPW** | **98.2** | **84.8** | **59.3** | **62.6** | **34.7** |

**Llama-3 backbone 提升最大**：

| 基准 | Baseline | LDB | LPW | LPW vs LDB |
|------|---------|-----|-----|-----------|
| HumanEval | 73.2 | 84.1 | 88.4 | +4.3 |
| MBPP | 44.0 | 57.2 | **73.6** | **+16.4** |

### 消融实验

| 配置 | HumanEval | MBPP | 说明 |
|------|-----------|------|------|
| LPW（完整） | 89.0 | 76.0 | - |
| LPW-V（去掉计划验证） | 86.0 (-3.0) | 73.2 (-2.8) | 计划验证对两阶段都重要 |
| LPW-S（去掉方案生成阶段） | 86.0 (-3.0) | 73.0 (-3.0) | 直接调试 Baseline 代码 |
| LPW-C（去掉代码修复） | 79.9 (-9.1) | 72.2 (-3.8) | 仅基于计划生成代码 |
| 更多可见测试（MBPP-ET → MBPP-ET-3） | - | +4.4 | LPW 利用额外测试效率最高 |

### 关键发现

1. **计划验证是关键**：LPW-V 的下降证明了计划验证在初始代码生成和调试中的双重价值；未验证的计划效果有限（LPW-V 与 LPW-S 近似）
2. **两阶段缺一不可**：去掉任一阶段都导致性能下降，但去掉代码修复影响更大（-9.1% on HumanEval）
3. **在困难基准上优势更大**：在 LiveCode（+5%）、APPS（+10%）、CodeContests（+5%）等挑战性基准上优势突出
4. **迭代效率高**：LPW 仅需 1 次迭代就超越 LDB/SD 的最优表现
5. **初始代码质量更高**：LPW 的初始代码在 0 次迭代时就达到 79.9%（vs Baseline 74.4%），说明计划验证直接提升了初始代码质量

## 亮点与洞察

- **"先想清楚再编码"的自动化**：将人类程序员先验证思路、再编码、再对比分析的工作流首次完整自动化，是对 TDD 理念在 LLM 领域的精妙翻译
- **自然语言作为调试的中间表示**：计划验证提供了比代码更高层次的"预期行为规范"，使得调试从"猜测修复"变为"对比差异"，本质上将调试问题转化为文本对齐问题
- **模型无关性**：在 GPT-3.5、Llama-3、Phi-3、GPT-4o 四种不同模型上均有效，证明了工作流设计的通用性
- **计划验证对可见测试的高效利用**：MBPP-ET-3 实验表明 LPW 在利用额外测试信息方面效率最高（+4.4% vs LDB +2.0%）

## 局限与展望

1. **Token 消耗大**：计划和验证生成需要大量 token，对于简单问题可能过度engineered
2. **计划翻译瓶颈**：代码的准确率仍低于计划验证的准确率，说明从自然语言方案到代码实现的翻译仍有提升空间
3. **依赖可见测试**：LPW 需要可见测试用例来验证计划，在无测试场景下无法直接使用
4. **LLM 推理能力限制**：底层仍受限于 LLM 的推理能力，无法解决超出模型能力范围的复杂逻辑问题
5. **XML 格式兼容性**：MapCoder 在 Phi-3 上失败的案例说明严格格式要求可能影响小模型适用性，LPW 虽未报告此问题但值得关注

## 相关工作与启发

- **Self-Planning (Jiang et al., 2023)**：仅生成计划不验证，效果有限（仅比 Baseline +1-3%）
- **Self-Debugging (Chen et al., 2023)**：橡皮鸭调试法，缺乏精确的修正指导
- **LDB (Zhong et al., 2024)**：基于控制流图分段调试，提供运行时信息但反馈粗糙
- **MapCoder (Islam et al., 2024)**：多智能体生成多计划但不验证
- 启发：在 LLM 代码生成中，自然语言层面的方案验证可能比代码层面的调试更有效率，因为 LLM 在自然语言推理上比代码执行追踪更擅长

## 评分

- 新颖性: ⭐⭐⭐⭐ 计划验证机制和基于验证的精准调试策略设计新颖，TDD思想的巧妙转化
- 实验充分度: ⭐⭐⭐⭐⭐ 7个基准、4种模型、完整消融、成本分析、案例研究，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 流程图清晰，案例说明详实，公式化问题定义规范
- 价值: ⭐⭐⭐⭐⭐ 新SOTA，方法实用且模型无关，对LLM代码生成领域有显著推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] APPL: A Prompt Programming Language for Harmonious Integration of Programs and Large Language Model Prompts](appl_a_prompt_programming_language_for_harmonious_integration_of_programs_and_la.md)
- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](../../NeurIPS2025/llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] On the Limit of Language Models as Planning Formalizers](limit_llm_planning_formalizer.md)
- [\[ACL 2025\] INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models](interact_enabling_interactive_question-driven_learning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
