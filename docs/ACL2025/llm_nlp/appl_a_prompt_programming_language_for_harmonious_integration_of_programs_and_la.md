---
title: >-
  [论文解读] APPL: A Prompt Programming Language for Harmonious Integration of Programs and Large Language Model Prompts
description: >-
  [ACL 2025][LLM 其他][提示编程语言] 本文提出APPL——一种将LLM提示（prompt）无缝嵌入Python程序的提示编程语言，提供Python原生语法、异步并行运行时和可追溯调试模块，简化了复杂LLM工作流的开发与维护。 领域现状：随着LLM能力的增强，基于LLM的应用越来越复杂——从简单的单轮对话发展到…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "提示编程语言"
  - "LLM工作流"
  - "Python集成"
  - "异步执行"
  - "可复现性"
---

# APPL: A Prompt Programming Language for Harmonious Integration of Programs and Large Language Model Prompts

**会议**: ACL 2025  
**arXiv**: [2406.13161](https://arxiv.org/abs/2406.13161)  
**代码**: [https://github.com/appl-team/appl](https://github.com/appl-team/appl)  
**领域**: LLM / 提示工程  
**关键词**: 提示编程语言, LLM工作流, Python集成, 异步执行, 可复现性

## 一句话总结
本文提出APPL——一种将LLM提示（prompt）无缝嵌入Python程序的提示编程语言，提供Python原生语法、异步并行运行时和可追溯调试模块，简化了复杂LLM工作流的开发与维护。

## 研究背景与动机

**领域现状**：随着LLM能力的增强，基于LLM的应用越来越复杂——从简单的单轮对话发展到多步推理、工具调用、多Agent协作等复杂工作流。开发者需要将LLM调用与传统程序逻辑（条件判断、循环、数据处理）紧密结合。

**现有痛点**：（1）现有LLM开发框架（如LangChain、DSPy）虽然提供了高级抽象，但往往引入复杂的新概念和API，学习曲线陡峭。（2）prompt的编写与程序逻辑脱节——prompt通常以字符串模板形式存在，与代码的数据流很难自然融合。（3）当LLM调用之间存在依赖关系时，串行执行效率低下；而手动管理异步并行又增加了代码复杂度。（4）复杂工作流的调试和错误追踪极其困难。

**核心矛盾**：prompt是自然语言，程序是代码——两者的表达范式根本不同。现有方案要么牺牲prompt的灵活性将其压缩为模板，要么牺牲程序的结构性在长字符串中拼接提示，两者都不理想。

**本文目标**：设计一种编程语言/框架，让LLM prompt像函数一样可以被Python代码自然调用和组合，同时自动处理并行化、缓存和调试。

**切入角度**：观察到大多数LLM工作流实质上是"prompt函数"的有向无环图（DAG），利用Python的协程机制可以自然地表达这种DAG结构并实现自动并行化。

**核心 idea**：用Python装饰器将普通函数标记为"prompt函数"，在函数体内自然地混合Python代码和LLM调用，运行时自动分析依赖关系并行执行独立的LLM调用。

## 方法详解

### 整体框架
APPL的架构包含三层：（1）语言层：提供Python原生的prompt构建语法，通过装饰器 `@ppl` 标记prompt函数；（2）运行时层：自动检测数据依赖关系，使用协程实现异步并行执行；（3）工具层：支持执行追踪（tracing）用于调试和结果重放。

### 关键设计

1. **Python-native的Prompt语法**:

    - 功能：让开发者用自然的Python代码构建动态prompt
    - 核心思路：通过 `@ppl` 装饰器标记函数为prompt函数。函数内部可以使用普通Python语句构建上下文，使用 `gen()` 调用LLM生成。prompt的上下文通过函数的局部状态自动管理，支持变量插值、条件分支和循环。函数调用本身就可以组合更复杂的prompt——一个prompt函数可以调用另一个prompt函数，形成层级结构
    - 设计动机：Pythonic的设计降低学习成本，利用Python已有的抽象机制（函数、类、模块）来组织prompt，无需引入新的概念

2. **异步并行运行时**:

    - 功能：自动并行化独立的LLM调用以提升吞吐量
    - 核心思路：当prompt函数中包含多个 `gen()` 调用且它们之间没有数据依赖时，运行时自动将这些调用包装为异步协程并行发送。开发者编写的是看似顺序的代码，但执行是自动并行的——APPL使用延迟求值（lazy evaluation）技术，只在真正需要结果时等待LLM响应，其余时间继续执行后续代码
    - 设计动机：LLM API调用的延迟通常在百毫秒到秒级，是工作流的主要瓶颈。自动并行化可以在不改变开发者代码逻辑的情况下显著提升效率

3. **执行追踪与重放模块**:

    - 功能：记录LLM调用的输入输出，支持故障诊断和无成本重放
    - 核心思路：每次 `gen()` 调用的prompt输入和LLM响应都被记录到trace文件中。当工作流失败或需要调试时，可以从trace中重放，跳过实际的LLM调用，快速重现问题。trace还支持差异对比——当修改了prompt后，只有变化的调用需要重新执行，未变化的从缓存中读取
    - 设计动机：复杂LLM工作流的调试是开发者的巨大痛点。trace机制同时解决了可复现性和调试成本两个问题

### 损失函数 / 训练策略
APPL是一个工程框架而非模型，不涉及训练。评估主要通过开发效率（代码行数、开发时间）、运行效率（延迟、吞吐量）和与现有框架的功能对比来进行。

## 实验关键数据

### 主实验

| 框架 | 代码行数(RAG) | 代码行数(Multi-Agent) | 延迟(5并行) | 可追溯性 |
|------|-------------|---------------------|------------|---------|
| 原生Python | 85 | 210 | 15.2s | 无 |
| LangChain | 62 | 175 | 14.8s | 部分 |
| DSPy | 45 | 130 | 13.5s | 部分 |
| **APPL** | **38** | **95** | **4.2s** | **完整** |

### 消融实验

| 配置 | 延迟(Multi-Agent, 5轮) | 说明 |
|------|----------------------|------|
| APPL（自动并行） | 4.2s | 5个独立调用并行执行 |
| APPL（串行模式） | 15.0s | 禁用并行，与原生Python相当 |
| APPL + Trace重放 | 0.3s | 从缓存重放，无LLM调用 |
| 部分修改后重放 | 2.1s | 只重新执行修改的调用 |

### 关键发现
- APPL的自动并行化在Multi-Agent场景中实现了约3.6倍的延迟降低
- 代码量减少约55%（相比原生Python），且可读性显著提升
- Trace重放将调试迭代的时间从秒级降低到毫秒级。在开发复杂工作流时，这意味着每次修改后可以几乎瞬时看到结果
- APPL的Python原生设计使得现有IDE的代码补全、类型检查等工具可以无缝使用

## 亮点与洞察
- "prompt as function"的设计哲学非常精准——函数是编程最基本的抽象单元，将prompt函数化可以无缝利用已有的编程范式（组合、复用、测试）。
- 延迟求值+自动并行化是一个非常工程化但极其实用的创新。开发者写顺序代码，运行时自动优化为并行，这种透明性是工具设计的高水平体现。
- Trace机制对于LLM应用的可靠工程化至关重要——当LLM调用的成本是实际金钱时，能够缓存和重放就不只是方便，而是省钱。

## 局限与展望
- APPL目前仅支持Python，对于其他语言（如JavaScript/TypeScript）的LLM开发者不适用
- 自动并行化依赖于正确的依赖分析，对于有隐式副作用的代码可能无法正确处理
- 与LangChain、DSPy等框架的生态整合尚不完善
- 未来可以扩展到支持流式输出（streaming）和实时交互场景

## 相关工作与启发
- **vs LangChain**: LangChain提供高级抽象但引入大量新概念，APPL更贴近原生Python，学习成本低
- **vs DSPy**: DSPy聚焦于prompt优化，APPL聚焦于程序与prompt的无缝融合，两者可以互补
- **vs LMQL**: LMQL也是prompt编程语言，但使用独立的语法，而APPL完全基于Python语法
- **vs SGLang**: SGLang关注高效的LLM服务运行时，APPL关注前端编程体验，二者处于不同层次

## 评分
- 新颖性: ⭐⭐⭐⭐ Prompt-as-function的设计哲学和自动并行化运行时具有创新性
- 实验充分度: ⭐⭐⭐ 以工程评估为主，缺少大规模用户研究
- 写作质量: ⭐⭐⭐⭐ 技术描述清晰，代码示例直观
- 价值: ⭐⭐⭐⭐ 对LLM工作流开发具有实际工程价值，开源且活跃维护

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Planning-Driven Programming: A Large Language Model Programming Workflow](planning-driven_programming_a_large_language_model_programming_workflow.md)
- [\[ACL 2025\] JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution](jopa_explaining_large_language_models_generation_via_joint_prompt_attribution.md)
- [\[ACL 2025\] When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)
- [\[ACL 2025\] Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models](dynamic_knowledge_integration_for_evidence-driven_counter-argument_generation_wi.md)
- [\[ACL 2025\] Representation Bending for Large Language Model Safety](repbend_representation_bending_safety.md)

</div>

<!-- RELATED:END -->
