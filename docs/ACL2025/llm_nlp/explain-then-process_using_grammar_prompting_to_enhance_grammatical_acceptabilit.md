---
title: >-
  [论文解读] Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments
description: >-
  [ACL 2025][LLM/NLP][提示学习] 提出 grammar prompting 的 explain-then-process 范式——先让 LLM 生成目标语法现象的解释，再将该解释作为上下文反馈给目标模型（LLM 或 SLM）辅助最小对语法判断。在英语 BLiMP、中文 SLING、俄语 RuBLiMP 三个跨语言基准上显著提升准确率，SLM 搭配 GP+CoT 将 LLM-SLM 平均差距从 13.0pp 缩小到 5.8pp（缩小 56%）。
tags:
  - ACL 2025
  - LLM/NLP
  - 提示学习
  - 语法可接受性
  - 最小对
  - explain-then-process
  - 多语言
---

# Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments

**会议**: ACL 2025  
**arXiv**: [2506.02302](https://arxiv.org/abs/2506.02302)  
**代码**: 无  
**领域**: 语言学评估 / Prompting  
**关键词**: grammar prompting, 语法可接受性, 最小对, explain-then-process, 多语言

## 一句话总结

提出 grammar prompting 的 explain-then-process 范式——先让 LLM 生成目标语法现象的解释，再将该解释作为上下文反馈给目标模型（LLM 或 SLM）辅助最小对语法判断。在英语 BLiMP、中文 SLING、俄语 RuBLiMP 三个跨语言基准上显著提升准确率，SLM 搭配 GP+CoT 将 LLM-SLM 平均差距从 13.0pp 缩小到 5.8pp（缩小 56%）。

## 研究背景与动机

**领域现状**：LLM 展现出强大的语言使用能力（功能性能力），但在显式语法判断任务上暴露出意外的弱点。例如 Claude Sonnet 在判断 NPI（否定极性项）许可时，会先释义句子，在释义过程中丢失关键的句法约束信息，导致错误判断。

**现有痛点**：LLM 做语法判断时倾向于先释义/翻译再分析，而释义过程系统性地掩盖了关键语法特征——有一个关键的"知道规则"和"应用规则"之间的鸿沟。LLM 能解释语法规则但常常不能在判断时正确应用它们。

**核心矛盾**：LLM 拥有隐式的语法知识（可以生成流畅文本），但在显式判断中无法系统地激活和应用这些知识——形式能力（知道规则）与功能能力（使用规则）脱节。

**本文目标** 如何帮助模型在做语法判断时聚焦于语言结构而非释义？

**切入角度**：参考心理语言学和 MTOB（Machine Translation from One Book）的成功经验：显式提供语法知识可以激活模型的内在语言能力。

**核心 idea**：先解释再处理——将 LLM 自生成的语法解释反馈给自身或 SLM，弥合"知道规则"和"使用规则"之间的鸿沟。

## 方法详解

### 整体框架

两步 explain-then-process 流程：(1) Explain——使用指令模板让 LLM（如 Sonnet、GPT-o1）为特定语法现象（如 NPI 许可、填充-空位依赖）生成简洁的语法解释，解释不包含完整示例句（避免模式匹配）；(2) Process——将生成的语法解释作为上下文提示，输入目标模型进行最小对判断（从一对仅差一个句法特征的句子中选出语法正确的那个）。

### 关键设计

1. **语法解释生成（Explain）**:

    - 功能：设计指令模板引导 LLM 生成特定语法范式的解释
    - 核心思路：模板包含语法范式名称（如"NPI licensing"）、示例最小对和指令（要求避免完整例句、指定目标受众），生成面向"初学者"或"专家"的解释
    - 设计动机：(1) 规避完整示例可防止模型做表面模式匹配而非真正推理；(2) 初学者解释强调实用识别方法（如"用 who/what 检查"），专家解释使用技术术语（如"长距离依赖"、"选择限制"）
    - 发现：初学者解释在宏观分析中以小但显著的优势优于专家解释（-1.9% ± 5.7%, p=0.002）

2. **提示策略组合（Process）**:

    - 功能：测试多种提示策略及其组合
    - 核心思路：
        - **Base**: 直接问"哪个句子更语法正确"
        - **CoT**: 要求逐步推理后回答
        - **GP (Grammar Prompting)**: 将语法解释作为上下文前缀
        - **GP+CoT**: 先提供语法解释，再要求逐步推理
    - 控制条件：Control（提供无关语法现象的解释）和 Textbook（提供多个语法解释由模型自行选择）
    - 设计动机：GP 和 CoT 针对不同瓶颈——GP 提供缺失的规则知识，CoT 激活规则应用能力。组合使用可同时解决两个瓶颈

3. **多语言最小对评估**:

    - 功能：在英语 BLiMP（67 范式取困难子集 8 类）、中文 SLING（38 范式取 6 类）、俄语 RuBLiMP（45 范式取 7 类）上评估，每范式取前 50 对
    - 核心思路：三次 A/B 呈现实验（正序+反序+随机）取平均消除位置偏见；使用 prompt-based 方法而非 perplexity
    - 设计动机：多语言设计验证方法的语言不可知性；取困难子集聚焦于模型真正薄弱的语法现象

## 实验关键数据

### 主实验（GPT-4o + Grammar Prompting，各基准困难子集）

| 基准 | Base | CoT | GPb (Sonnet) | GPb+CoT (o1) |
|------|------|-----|-------------|-------------|
| BLiMP (英语) | 77.0 | 79.9 | 85.2 | **96.7** |
| SLING (中文) | 93.1 | 96.7 | 97.1 | **99.2** |
| RuBLiMP (俄语) | 93.3 | 97.6 | 98.0 | **100.0** |

### SLM 实验（Haiku + Grammar Prompting）

| 基准 | Base | CoT | GPb+CoT (Sonnet) | GPb+CoT (o1) |
|------|------|-----|-----------------|-------------|
| BLiMP (英语) | 61.2 | 72.0 | 82.3 | **86.5** |
| SLING (中文) | 78.3 | 83.6 | 89.2 | **93.3** |
| RuBLiMP (俄语) | 78.3 | 86.3 | 93.2 | **95.8** |

### 消融：控制条件 vs GP（GPT-4o，BLiMP）

| 条件 | gpt-3.5 Avg | gpt-4o Avg |
|------|-------------|------------|
| Control (无关解释) | 64.1 | 75.8 |
| Textbook (多规则混合) | 61.3 | 77.8 |
| GPb (目标规则解释) | **72.5** | **90.2** |

### 关键发现
- Grammar Prompting 单独即可在 BLiMP 上将 gpt-3.5 从 67.9% 提升到 73.6%（+5.7pp），gpt-4o 从 77.0% 提升到 85.2%（+8.2pp）
- GP+CoT 组合效果最强：gpt-4o 在 BLiMP 上达到 96.7%，Sonnet 在 RuBLiMP 达 100%
- 控制条件（无关解释）有时反而降低性能，证明提升来自目标语法知识而非通用指令遵循
- 初学者解释整体优于专家解释（p=0.002），但在填充-空位依赖等特定范式中专家解释更优
- SLM（Haiku）搭配 GP+CoT 将与 LLM 的差距从 13.0pp 缩小到 5.8pp——GP 单独缩小 20%，GP+CoT 缩小 56%
- 3-shot 在 SLM 上效果极差（可能引发模式匹配式捷径），GP 是更原则性的方法

## 亮点与洞察
- **"知道规则"vs"应用规则"的鸿沟洞察**——LLM 能解释语法但做不好语法判断，因为判断时倾向于释义而非结构分析。GP 通过显式提供规则引导注意力回到结构层面，是一种优雅的解决方案。
- **SLM 赋能的实际意义**——GP 让低成本 SLM 接近前沿 LLM 的语法判断性能，这对资源受限场景和教育应用有实际价值。GP+CoT 的组合尤其强大。
- **多语言零成本泛化**——方法在英/中/俄三种类型学差异大的语言上均有效，且语法解释用英语提示即可（即使目标句子是中文/俄语），说明方法具有语言不可知性。

## 局限与展望
- 语法解释需要对每个范式生成一次，但范式识别本身未自动化（测试中已知范式标签）
- 仅测试了 GPT、Claude 和 Llama 家族的 5 个模型
- 困难范式的选择基于 gpt-4o 的初始表现，可能引入选择偏差
- 未测试更多真实应用场景（如语法纠错、写作辅助），仅限于最小对判断任务
- 语法解释质量依赖于 LLM 自身的元语言知识，对少数语言或罕见语法现象可能失效

## 相关工作与启发
- **vs MTOB (Tanzer et al., 2024)**: MTOB 用一本语法书改善零资源翻译；GP 类似但将 LLM 自身作为"语法书"来源
- **vs CoT**: CoT 激活推理过程但不提供新知识；GP 提供缺失的领域知识。两者正交且互补
- **vs Few-shot**: Few-shot 可能引导模式匹配而非规则理解；GP 提供规则而非示例，更原则性

## 评分
- 新颖性: ⭐⭐⭐⭐ explain-then-process 范式和 GP 与 CoT 的正交互补关系是有价值的贡献
- 实验充分度: ⭐⭐⭐⭐ 3 语言 × 5 模型 × 多种条件 × 控制实验，设计严谨
- 写作质量: ⭐⭐⭐⭐⭐ 引入示例生动，实验设计逻辑清晰
- 价值: ⭐⭐⭐⭐ 对 LLM 语言学评估和 prompting 方法论有实用贡献

<!-- RELATED:START -->

## 相关论文

- [Can Input Attributions Explain Inductive Reasoning in In-Context Learning?](can_input_attributions_explain_inductive_reasoning_in_in-context_learning.md)
- [CodeTool: Enhancing Programmatic Tool Invocation of LLMs via Process Supervision](codetool_enhancing_programmatic_tool_invocation_of_llms_via_process_supervision.md)
- [Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)
- [HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)
- [Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](deontological_keyword_bias.md)

<!-- RELATED:END -->
