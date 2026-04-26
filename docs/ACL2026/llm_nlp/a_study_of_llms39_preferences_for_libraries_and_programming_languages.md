---
title: >-
  [论文解读] A Study of LLMs' Preferences for Libraries and Programming Languages
description: >-
  [ACL 2026 (Findings)][LLM/NLP][代码生成偏好] 首次系统研究8个LLM在代码生成中对库和编程语言的偏好行为，发现LLM严重偏好NumPy等流行库（45%的使用不必要）和Python语言（58%的高性能任务仍选Python），且自然语言推荐与实际代码选择不一致。
tags:
  - ACL 2026 (Findings)
  - LLM/NLP
  - 代码生成偏好
  - 库选择偏差
  - 编程语言偏好
  - LLM行为分析
  - 技术多样性
---

# A Study of LLMs' Preferences for Libraries and Programming Languages

**会议**: ACL 2026 (Findings)  
**arXiv**: [2503.17181](https://arxiv.org/abs/2503.17181)  
**代码**: [GitHub](https://github.com/itsluketwist/llm-code-bias)  
**领域**: Code Intelligence / LLM Evaluation  
**关键词**: 代码生成偏好, 库选择偏差, 编程语言偏好, LLM行为分析, 技术多样性

## 一句话总结

首次系统研究8个LLM在代码生成中对库和编程语言的偏好行为，发现LLM严重偏好NumPy等流行库（45%的使用不必要）和Python语言（58%的高性能任务仍选Python），且自然语言推荐与实际代码选择不一致。

## 研究背景与动机

**领域现状**：LLM在代码生成方面取得了巨大进展，但现有评估主要关注功能正确性和语法有效性，忽略了LLM在生成代码时做出的关键设计决策——选择什么库、使用什么编程语言。

**现有痛点**：开发者在prompting LLM时往往不指定具体库，很多终端用户也缺乏判断LLM语言选择是否恰当的专业知识。这意味着LLM的技术偏好可能深刻影响软件生态系统的多样性。

**核心矛盾**：LLM应根据任务需求选择最适合的技术栈，但训练数据中的频率分布可能导致它们系统性地偏向流行技术，即使这并非最优选择。

**本文目标**：量化LLM在库和编程语言选择中的偏好模式，评估这些偏好的合理性和潜在风险。

**切入角度**：设计三组实验——benchmark任务的库选择、项目初始化的库/语言选择、自然语言推荐与代码行为的一致性检验。

**核心 idea**：LLM在代码生成中表现出显著的"熟悉度偏好"，优先选择流行技术而非最适合任务的技术。

## 方法详解

### 整体框架

三组实验覆盖两个维度（库和语言）×两个场景（benchmark任务和项目初始化），外加一致性检验。使用8个多样化的LLM（GPT-4o-mini, GPT-3.5-turbo, Claude-3.5 Sonnet/Haiku, Llama-3.2-3B, Mistral-7B, Qwen-2.5-Coder, DeepSeek-LLM），每个任务生成3-100次响应以减少随机性。

### 关键设计

1. **库偏好实验（Experiment 1）**:

    - 功能：量化LLM在不指定库时的Python库选择偏好
    - 核心思路：使用BigCodeBench的525个任务（过滤掉prompt中提及ground-truth库的任务），要求LLM生成使用外部库的代码，统计库使用频率并与ground-truth对比
    - 设计动机：开发者常不指定库就让LLM写代码，需要理解LLM在此场景下的行为模式

2. **语言偏好实验（Experiment 2）**:

    - 功能：量化LLM在不指定语言时的编程语言选择偏好
    - 核心思路：使用6个语言无关数据集（Multi-HumanEval, MBXP, AixBench, CoNaLa, APPS, CodeContests）测试benchmark任务；额外设计5个高性能场景的项目初始化任务（并发web服务器、跨平台GUI、低延迟交易平台等），Python被认为不是最优选择
    - 设计动机：测试LLM是否会根据任务特点（如高性能需求）选择合适的语言，还是一律默认Python

3. **推荐一致性实验（Experiment 3）**:

    - 功能：检验LLM自然语言推荐的技术是否与其代码生成中实际使用的一致
    - 核心思路：让LLM以文字列出"最佳"库/语言的排名，再与Experiment 1/2中的实际使用频率排名对比，用Kendall's $\tau_b$ 系数度量一致性
    - 设计动机：如果LLM"知道"最优选择（NL推荐正确）但代码中不这么做（实际偏好不同），说明偏好嵌入在代码生成行为而非知识缺失

### 损失函数 / 训练策略

本文是实证研究，不涉及模型训练。所有LLM使用默认API参数，每次交互使用新会话避免缓存偏差，不使用system prompt以反映基线行为。

## 实验关键数据

### 主实验

| 发现 | 具体数据 | 影响 |
|------|---------|------|
| NumPy过度使用 | 在305个不需要NumPy的任务中有192个（63%）使用了NumPy | 严重偏好 |
| 库多样性不足 | 每个LLM仅使用32-39个不同库 | 生态单一化 |
| Python偏好 | 高性能任务中58%的情况仍选Python | 技术不适配 |
| Rust缺失 | 高性能项目中Rust使用率为0 | 极端偏好 |
| NL-Code不一致 | Kendall's τ_b 极低 | 知行不合一 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Prompt敏感性 | 偏好模式不变 | 不同严格程度的prompt得到类似结果 |
| 跨LLM一致性 | Top-3库相同 | 所有8个LLM的前三名库完全一致（NumPy, pandas, Matplotlib）|

### 关键发现
- 所有LLM的库使用分布高度相似，前三名一致（NumPy > pandas > Matplotlib），与模型大小、开闭源无关
- 即便任务明确需要高性能（低延迟交易、并行处理），Python仍占主导，Rust完全缺席
- LLM"推荐"的技术栈与其实际使用的之间一致性极低，说明偏好根植于生成行为而非知识层面

## 亮点与洞察
- "LLM知道什么好但不一定这么做"是一个重要发现，说明代码生成中的偏好可能源于训练数据分布而非推理
- 对软件生态有警示意义：LLM大规模使用可能形成正反馈循环——偏好流行库→生成更多该库代码→更多训练数据→更强偏好
- 实验设计简洁有力，三个实验形成完整的论证链

## 局限与展望
- 仅测试8个LLM，未覆盖最新的推理增强型模型（如o1, DeepSeek-R1）
- Python的库分析较深入，但其他语言的库偏好未探索
- 未提出具体的去偏方法，主要停留在现象描述
- 未来可研究fine-tuning和RLHF如何影响技术偏好

## 相关工作与启发
- **vs LLM社会偏见研究**: 将偏见分析从社会维度扩展到技术维度，开辟了新的研究方向
- **vs 代码生成评估**: 提出了"设计决策质量"这一被忽视但重要的评估维度
- **vs 工具推荐系统**: 揭示了LLM作为"隐式推荐系统"的局限性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统研究LLM的技术偏好，开辟新方向
- 实验充分度: ⭐⭐⭐⭐ 8个模型、多场景、一致性检验，设计完整
- 写作质量: ⭐⭐⭐⭐⭐ 问题清晰、实验简洁、发现有冲击力
- 价值: ⭐⭐⭐⭐ 对LLM开发者和用户都有重要警示意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Planning-Driven Programming: A Large Language Model Programming Workflow](../../ACL2025/llm_nlp/planning-driven_programming_a_large_language_model_programming_workflow.md)
- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](../../ACL2025/llm_nlp/knowledge_boundary_crosslingual.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](../../ACL2025/llm_nlp/can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs](../../ACL2025/llm_nlp/how_llms_comprehend_temporal_meaning_in_narratives_a_case_study_in_cognitive_eva.md)
- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](../../ACL2025/llm_nlp/is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)

<!-- RELATED:END -->
