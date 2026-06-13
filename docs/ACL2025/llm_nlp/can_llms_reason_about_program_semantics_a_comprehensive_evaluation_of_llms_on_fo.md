---
title: >-
  [论文解读] Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference
description: >-
  [ACL 2025][LLM 其他][程序语义推理] 提出 FormalBench 基准，通过形式化程序规格（formal specifications）推断任务系统评估LLM的程序语义推理能力，发现LLM在简单控制流上表现良好但在循环等复杂结构上挣扎，并设计了自修复提示（self-repair prompts）将成功率提升25%。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "程序语义推理"
  - "形式化验证"
  - "规格推断"
  - "代码理解"
  - "自修复提示"
---

# Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference

**会议**: ACL 2025  
**arXiv**: [2503.04779](https://arxiv.org/abs/2503.04779)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 程序语义推理, 形式化验证, 规格推断, 代码理解, 自修复提示

## 一句话总结

提出 FormalBench 基准，通过形式化程序规格（formal specifications）推断任务系统评估LLM的程序语义推理能力，发现LLM在简单控制流上表现良好但在循环等复杂结构上挣扎，并设计了自修复提示（self-repair prompts）将成功率提升25%。

## 研究背景与动机

**领域现状**：大语言模型在编程任务自动化中的应用越来越广泛，包括代码生成、代码补全、Bug修复等。然而，大多数评测聚焦于LLM生成"看起来正确"的代码的能力，对模型是否真正理解程序的语义——即程序在所有可能执行路径上的行为——研究不足。

**现有痛点**：现有的代码能力评测（如HumanEval、MBPP）主要通过运行测试用例来验证生成代码的正确性，但这有两个根本问题：（1）测试用例只能覆盖有限的输入，无法保证程序在所有情况下都正确；（2）通过测试用例并不意味着LLM真正理解了程序的语义——模型可能只是模式匹配到了正确的代码模板。形式化验证（formal verification）需要程序规格（specifications），如前置条件、后置条件和循环不变式，这些规格精确描述了程序的预期行为。

**核心矛盾**：通过测试用例评测代码的方法无法准确衡量LLM对程序语义的理解深度。一个能通过所有测试的模型可能对程序的实际行为毫无理解，只是在做表面的模式匹配。

**本文目标**：设计一个专门评估LLM程序语义推理能力的基准，通过让LLM生成形式化程序规格来测试其对程序行为的理解。

**切入角度**：作者选择"形式化规格推断"（formal specification inference）作为评测任务——这个任务要求模型既要全面推理程序的所有可能执行路径，又要生成精确的、符合形式化语法语义的表达式，是一个高要求的程序理解任务。

**核心 idea**：用形式化规格推断作为代理任务来评估LLM的程序语义推理能力，构建涵盖不同复杂度程序结构的基准FormalBench。

## 方法详解

### 整体框架

FormalBench 基准的设计和评估包含三个部分：（1）**基准构建**：收集涵盖不同控制流复杂度的C程序，为每个程序手动编写正确的形式化规格作为ground truth；（2）**评测设置**：设计多种prompting策略评估LLM的规格推断能力；（3）**深入分析**：分析失败模式、鲁棒性、以及自修复的效果。

### 关键设计

1. **FormalBench 基准构建 (Benchmark Construction)**:

    - 功能：提供一个涵盖不同程序复杂度的形式化规格推断基准
    - 核心思路：基准中的程序涵盖四个复杂度级别：（a）**纯顺序结构**：没有分支和循环，只有赋值语句；（b）**条件分支**：包含 if-else 等条件语句；（c）**简单循环**：包含单层循环，循环体相对简单；（d）**复杂循环**：嵌套循环、依赖于多个变量的循环条件等。对每个程序，需要推断的规格包括前置条件（preconditions）、后置条件（postconditions）和循环不变式（loop invariants）。所有规格使用ACSL（ANSI/ISO C Specification Language）语法编写
    - 设计动机：不同的程序结构对推理能力的要求不同——顺序结构只需线性推理，而循环需要归纳推理找到不变式，这是一个根本性的挑战

2. **多策略Prompting评测 (Multi-Strategy Prompting Evaluation)**:

    - 功能：系统评估不同提示策略对LLM规格推断能力的影响
    - 核心思路：设计了多种prompting策略：（a）**Zero-shot**：直接给出程序，要求生成规格；（b）**Few-shot**：提供几个"程序→规格"的示例；（c）**Chain-of-Thought**：要求模型先分析程序的执行流程，再生成规格；（d）**基于Hoare逻辑的提示**：在prompt中引入Hoare三元组的概念，引导模型按形式化推理方式思考。每种策略在多个主流LLM上评测，包括GPT-4、Claude、LLaMA、CodeLlama等
    - 设计动机：不同的提示策略对应不同的推理引导方式，系统比较能揭示LLM的推理瓶颈

3. **自修复提示机制 (Self-Repair Prompting)**:

    - 功能：让LLM在初次尝试失败后修复其生成的规格
    - 核心思路：当LLM生成的规格通不过形式化验证工具（如Frama-C）的检查时，将验证器返回的错误信息（如"循环不变式在第二次迭代后不成立"）反馈给LLM，让其修正规格。这个过程可以迭代多轮——每次修复后重新验证，失败则继续修复。实验中通常允许最多3轮修复
    - 设计动机：形式化验证工具提供了精确的反馈信号，与一般的代码生成（只有测试通过/失败的粗粒度信号）不同。利用这种精确反馈来引导修复是一个自然的想法

### 评测指标

主要使用两个指标：（1）**一致性（Consistency）**：生成的规格是否能通过形式化验证工具的检查，即规格是否与程序行为一致；（2）**完备性（Completeness）**：生成的规格是否足够强，能完全描述程序行为，而非一个平凡的（trivially true）规格。

## 实验关键数据

### 主实验

不同LLM在不同程序复杂度上的一致性通过率：

| 模型 | 顺序结构 | 条件分支 | 简单循环 | 复杂循环 | 平均 |
|------|---------|---------|---------|---------|------|
| GPT-4 | 92.3% | 85.7% | 61.2% | 38.5% | 69.4% |
| Claude-3 | 89.1% | 82.3% | 57.8% | 34.2% | 65.9% |
| GPT-3.5-Turbo | 78.5% | 68.2% | 42.1% | 22.3% | 52.8% |
| CodeLlama-34B | 75.3% | 64.8% | 38.5% | 18.7% | 49.3% |
| LLaMA-3-70B | 82.1% | 73.5% | 48.2% | 28.1% | 58.0% |
| DeepSeek-Coder-33B | 80.7% | 71.2% | 46.5% | 26.8% | 56.3% |

### 消融实验

| 提示策略 | GPT-4平均通过率 | 相对Zero-shot提升 | 说明 |
|---------|----------------|------------------|------|
| Zero-shot | 60.2% | — | 基线 |
| Few-shot (3 examples) | 65.8% | +5.6% | 示例帮助理解格式 |
| Chain-of-Thought | 67.3% | +7.1% | 逐步推理有效 |
| Hoare Logic提示 | 69.4% | +9.2% | 形式化思维引导最有效 |
| Self-repair (1轮) | 78.1% | +17.9% | 验证反馈显著有效 |
| Self-repair (3轮) | 85.2% | +25.0% | 多轮修复持续改善 |

### 关键发现

- **循环是根本瓶颈**：LLM在顺序结构上表现优秀（>90%），但在复杂循环上急剧下降（<40%）。循环需要归纳推理来找到不变式，这对LLM来说是一个根本性的挑战
- **自修复提示非常有效**：利用形式化验证器的精确反馈，成功率提升了25%。这表明LLM的"初次尝试"可能接近正确，验证反馈帮助其修正细节错误
- **鲁棒性不足**：对程序进行语义保持变换（如变量重命名、等价代码重写）后，LLM的输出可能完全不同，说明模型可能在做表面模式匹配而非真正的语义推理
- **常见失败模式**：（a）循环不变式过弱——生成的不变式为 `true`，虽然一致但无用；（b）变量关系遗漏——忽略多个变量之间的关系；（c）语法错误——生成的ACSL表达式不符合语法规范
- **模型规模有帮助但不能根本解决问题**：GPT-4在复杂循环上仍只有38.5%，说明规模扩展不是万能的

## 亮点与洞察

- 选择形式化规格推断作为评测任务非常巧妙——它既要求深度的程序理解，又有明确的正确性标准（通过形式化验证器检查），避免了评测指标本身的不可靠性
- 自修复机制利用了形式化验证的独特优势——精确的错误反馈。这在代码生成领域很少被利用
- 不同复杂度级别的分析揭示了LLM推理能力的"断崖式下降"发生在循环结构上，为理解LLM的推理边界提供了重要线索

## 局限与展望

- 基准规模相对较小，程序主要是教科书式的简单程序，与工业级代码有差距
- 仅测试了C语言的ACSL规格，未涉及其他形式化规格语言（如JML、Dafny）
- 自修复的轮数有限（3轮），更多轮次是否能持续改善尚不清楚
- 未来可以探索：将形式化推理能力作为训练目标来强化LLM、或结合符号推理器的混合方法

## 相关工作与启发

- 与 HumanEval、MBPP 等代码生成评测互补——它们评测"生成正确代码"的能力，FormalBench评测"理解程序行为"的能力
- 与 Lemur 等形式化推理工作相关，但 FormalBench 更聚焦于评测而非方法改进
- 自修复机制的成功启发了在更多形式化验证场景中利用工具反馈来增强LLM的可能性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次系统评估LLM在形式化规格推断上的能力，任务选择巧妙且有深度
- **实验充分度**: ⭐⭐⭐⭐ — 多模型、多策略的系统评估，消融分析和错误分析深入
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，实验设计合理
- **价值**: ⭐⭐⭐⭐⭐ — 揭示了LLM在程序语义推理上的根本局限，ACL 2025主会录取说明了其重要性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can Language Models Reason about Individualistic Human Values and Preferences?](can_language_models_reason_about_individualistic_human_values_and_preferences.md)
- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs](can_llms_help_uncover_insights_about_llms_a_large-scale_evolving_literature_anal.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)

</div>

<!-- RELATED:END -->
