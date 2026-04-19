---
title: >-
  [论文解读] SR-LLM: Rethinking the Structured Representation in Large Language Model
description: >-
  [ACL 2025][LLM/NLP][抽象语义表示] 提出 SR-LLM 框架，通过两种设置（training-free 的自然语言描述转换和 training-dependent 的混合数据微调）将结构化表示（AMR、PST、FOL）有效集成到 LLM 中，在 PAWS 等下游任务上分别取得 3.17% 和 12.38% 的提升，首次证明结构化表示能实质性增强 LLM 推理能力。
tags:
  - ACL 2025
  - LLM/NLP
  - 抽象语义表示
  - 结构化信息
  - LLM增强
  - 自然语言描述
  - 微调
---

# SR-LLM: Rethinking the Structured Representation in Large Language Model

**会议**: ACL 2025  
**arXiv**: [2502.14352](https://arxiv.org/abs/2502.14352)  
**代码**: 无  
**领域**: NLP / 结构化表示  
**关键词**: 抽象语义表示, 结构化信息, LLM增强, 自然语言描述, 微调

## 一句话总结

提出 SR-LLM 框架，通过两种设置（training-free 的自然语言描述转换和 training-dependent 的混合数据微调）将结构化表示（AMR、PST、FOL）有效集成到 LLM 中，在 PAWS 等下游任务上分别取得 3.17% 和 12.38% 的提升，首次证明结构化表示能实质性增强 LLM 推理能力。

## 研究背景与动机

结构化表示（如 AMR 抽象语义表示、PST 句法分析树、FOL 一阶逻辑）在传统 NLP 中至关重要，但在 LLM 时代其角色变得模糊：

1. **直接集成反而有害**：先前工作（如 AMRCOT）将 AMR 等结构以代码格式直接加入 prompt，结果性能反而下降（PAWS 上 -5.18%），令人质疑结构化表示在 LLM 中的价值
2. **编码格式不匹配猜想**：作者假设下降原因是结构化信息以 LLM 训练语料中不常见的代码格式传入，而非结构化信息本身无用
3. **缺乏有效集成范式**：LLM 时代如何最优地利用结构化表示仍是开放挑战

核心思路：将结构化表示转换为 LLM 更熟悉的自然语言描述形式，让模型能真正理解并利用结构化信息。

## 方法详解

### 整体框架

SR-LLM 包含两种配置：(1) Training-Free：通过 SR-to-NLD 模块将结构化表示转换为自然语言描述，直接加入 prompt；(2) Training-Dependent：构建包含结构化表示的混合数据集 Gen-SR 进行 SFT 微调，建立任务与结构化信息的内部关联。

### 关键设计

1. **SR-to-NLD（结构化表示到自然语言描述）**：以 AMR 为例，分四个阶段——(Phase 0) 用 Penman 库将 AMR 图转为三元组；(Phase 1) 标识符实例化，将抽象标识符替换为具体概念；(Phase 2) 用预定义规则字典将三元组映射为自然语言句子；(Phase 3) 用 GPT-4o Mini 精炼生成的描述，确保流畅和连贯，并通过多次生成的投票机制减少幻觉。设计动机是让 LLM 看到的是它训练语料中常见的自然语言形式，而非陌生的代码格式。关键区别于传统 SR-to-Text（生成一句完整句子），SR-to-NLD 通过多句协作描述结构信息。

2. **Gen-SR 混合数据微调**：构建两部分数据——G(text) 仅包含原始文本的指令对，G(SR) 在指令对中加入结构化表示。采用 50% G(text) + 50% G(SR) 的混合比例进行联合训练。设计动机是让模型不仅学习下游任务，还建立任务与结构信息之间的内部关联，使推理时能更有效利用结构化信息。

3. **多种结构化表示支持**：框架支持三种 SR——AMR（抽象语义表示）、PST（句法分析树）和 FOL（一阶逻辑），分别捕捉语义、句法和逻辑层面的信息。每种 SR 都有对应的 NLD 转换方法。

### 损失函数 / 训练策略

- Training-Free 设置不需要任何训练，直接在 prompt 中添加 SR-NLD
- Training-Dependent 设置使用 Llama3.1-8B-Instruct 进行 SFT
- 联合训练跨 10 个任务的数据（非针对单一任务）
- 实验在 AMR、PST、FOL 三种 SR 上分别独立进行，结果取平均
- 使用 CoT 和 One-Shot 两种 prompt 策略

## 实验关键数据

### 主实验（Training-Free, Llama3.1-8b-Instruct）

| 数据集 | 指标 | 原始 Prompt | + SR 代码格式 | + SR-NLD (本文) | 提升 |
|--------|------|------------|-------------|----------------|------|
| PAWS | F1 | 41.59 | 36.41 (-5.18) | 44.77 (+3.17) | +8.36 vs SR |
| Logic | F1 | 15.48 | 14.20 | 18.27 (+2.79) | +4.07 vs SR |
| AGNEWS | F1 | 53.88 | 48.17 | 56.67 (+2.79) | +8.50 vs SR |
| WiC | F1 | 43.99 | 42.05 | 48.17 (+4.18) | +6.12 vs SR |
| SPIDER | F1 | 24.80 | 21.53 | 29.60 (+4.80) | +8.07 vs SR |

### 主实验（Training-Dependent, Llama3.1-8b-Instruct）

| 微调策略 | PAWS (F1) | SNLI (F1) | WiC (F1) | SST-2 (F1) | SPIDER (EM) |
|---------|-----------|-----------|----------|------------|-------------|
| 100% G(text) | 68.94 | 35.53 | 66.97 | 75.59 | 41.20 |
| 100% G(SR) + SR prompt | 75.39 | 56.62 | 70.82 | 81.62 | 40.60 |
| **50% G(SR) + 50% G(text) + SR prompt** | **81.04** | **54.92** | **74.68** | **83.72** | **48.93** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| AMRBART 生成 AMR vs GPT-4o | 差异 <1% | SR 来源对结果影响小 |
| Flawed AMR vs Gold AMR | 差异显著 (4-9%) | SR 质量很重要 |
| Gold AMR + NLD | 最佳表现 | NLD 转换放大了高质量 SR 的收益 |
| 不同混合比例 | 50-50 最优 | 平衡文本和 SR 至关重要 |

### 关键发现

- 将 SR 以自然语言描述形式集成一致优于代码格式，证实了编码格式不匹配假说
- 弱模型从结构化信息中获益更多：Llama3.1-8b 改善显著且一致，GPT-4o-mini 改善最小
- Training-Dependent 在 PAWS 上获得 +12.38% 的巨大提升（从 68.66 到 81.04）
- SR 质量很重要：低质量/有缺陷的 AMR 会显著损害性能（-4 到 -9%），而高质量 AMR-NLD 大幅提升
- 训练数据中 SR 与任务的关联建立后，推理时 SR 才能被有效利用
- 纯 SR 数据训练不如文本+SR 混合训练

## 亮点与洞察

- **首次正面证明 SR 能增强 LLM**：扭转了"结构化表示在 LLM 时代无用"的判断，核心在于表示形式而非信息本身
- **自然语言是 LLM 的通用接口**：无论多抽象的结构化信息，转化为自然语言描述后都能被 LLM 有效吸收
- **弱模型受益更多**：结构化信息对缺乏自身推理能力的模型是有效补充，对强模型则边际效用递减
- **混合训练的重要性**：纯结构数据不如混合数据，模型需要同时维持对文本和结构的理解

## 局限与展望

- SR 生成依赖外部工具（AMR parser / GPT-4o），引入额外计算开销和潜在错误
- NLD 转换中使用 GPT-4o Mini 精炼可能引入成本和不确定性
- 10 个任务覆盖有限，未在生成式任务（如摘要、对话）中验证
- 训练依赖设置的成本未详细分析
- 不同 SR 类型（AMR vs PST vs FOL）的适用任务差异未深入分析

## 相关工作与启发

- 与 AMRCOT (Jin et al., 2024) 形成对比：同样使用 AMR，但表示形式不同导致相反结果
- 结构化知识与 LLM 的集成是重要趋势，本文为"如何集成"提供了新视角
- 启发：知识图谱、表格数据等结构化信息也可尝试 NLD 转换后集成到 LLM
- 弱模型更受益的发现对模型压缩后的增强策略有指导意义

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 简单但有效的洞察：形式而非内容是障碍 |
| 实用性 | 3 | NLD 转换流程可复用，但依赖外部工具 |
| 实验充分度 | 4 | 10 个任务、3 个模型、多种 SR 的全面实验 |
| 写作质量 | 3 | 结构清晰但符号较多，部分地方冗长 |
| 总分 | 3.5 | 有价值的洞察，打破了 LLM 时代 SR 无用的迷思 |

<!-- RELATED:START -->

## 相关论文

- [Representation Bending for Large Language Model Safety](repbend_representation_bending_safety.md)
- [Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights](praise_enhancing_product_descriptions_with_llm-driven_structured_insights.md)
- [Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [Pre³: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation](pre3_deterministic_pda_structured_gen.md)

<!-- RELATED:END -->
