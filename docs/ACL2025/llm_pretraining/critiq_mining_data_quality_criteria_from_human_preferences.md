---
title: >-
  [论文解读] CritiQ: Mining Data Quality Criteria from Human Preferences
description: >-
  [ACL 2025][数据选择] CritiQ 提出了一种基于 Agent 协作的数据质量标准自动挖掘方法，仅需约 30 个人类偏好标注对即可自动发现可解释的数据质量标准，并训练评分器进行高效数据选择，在代码、数学和逻辑领域的实验中显著提升了 Llama 3.1 的下游性能。
tags:
  - ACL 2025
  - 数据选择
  - 质量标准挖掘
  - 人类偏好
  - Agent协作
  - 可解释性
---

# CritiQ: Mining Data Quality Criteria from Human Preferences

**会议**: ACL 2025  
**arXiv**: [2502.19279](https://arxiv.org/abs/2502.19279)  
**代码**: [https://github.com/KYLN24/CritiQ](https://github.com/KYLN24/CritiQ)  
**领域**: 数据质量 / LLM 训练  
**关键词**: 数据选择、质量标准挖掘、人类偏好、Agent协作、可解释性

## 一句话总结

CritiQ 提出了一种基于 Agent 协作的数据质量标准自动挖掘方法，仅需约 30 个人类偏好标注对即可自动发现可解释的数据质量标准，并训练评分器进行高效数据选择，在代码、数学和逻辑领域的实验中显著提升了 Llama 3.1 的下游性能。

## 研究背景与动机

**领域现状**：高质量数据对语言模型性能至关重要，是决定模型能力上限的核心因素之一。目前主流的数据选择方法包括：基于人工设计的启发式规则（如长度过滤、去重）、基于现有模型困惑度（perplexity）进行筛选、训练分类器来判别数据质量、以及通过精心设计的提示让 LLM 做质量评估。

**现有痛点**：这些方法各有明显局限。启发式规则需要大量专家经验且难以泛化到新领域；困惑度方法依赖已有模型的分布，可能引入循环偏差；分类器方法需要大量人工标注且标准不透明；提示工程方法依赖工程师的试错经验，且评估标准仍然是隐式的、不可解释的。更关键的是，这些方法产出的数据质量标准要么不存在（黑箱分类器）、要么不可复用（特定提示），无法让人类专家审查和积累。

**核心矛盾**：数据选择需要明确的、可解释的质量标准，但现有方法要么没有显式标准（perplexity、分类器），要么标准固定不可演化（手工规则），缺乏一种能从少量人类反馈中自动发现和迭代优化质量标准的机制。

**本文目标**：设计一个系统，能够从极少量（~30对）人类偏好标注中自动挖掘出可解释的、可复用的数据质量标准（verbal criteria），并将这些标准转化为高效的数据选择工具。

**切入角度**：作者观察到人类在判断数据质量时会基于一套隐含的标准体系（如代码的可读性、逻辑的严密性、解题步骤的完整性），这些标准虽然难以一次性列举完整，但可以通过分析人类的偏好选择来逐步挖掘和提炼。

**核心 idea**：用多 Agent 协作系统（CritiQ Flow）从人类偏好对中迭代挖掘质量标准——Manager Agent 负责提出和演化标准假设，Worker Agent 负责用标准做成对判断来验证标准的有效性，通过多轮迭代不断优化标准集合。

## 方法详解

### 整体框架

CritiQ 的方法分为两个主要阶段。第一阶段是 **CritiQ Flow**：输入少量人类偏好标注对（约 30 对，每对包含两个数据样本和人类判断哪个更好），通过多 Agent 迭代协作挖掘出一组语言化的质量标准。第二阶段是 **CritiQ Scorer**：用挖掘出的标准通过 Agent 对大量数据进行标注，然后训练一个轻量级评分模型，对海量数据进行高效质量评分和选择。

### 关键设计

1. **CritiQ Flow — 多 Agent 质量标准挖掘**:

    - 功能：从少量人类偏好对中自动发现并迭代优化数据质量标准
    - 核心思路：系统包含一个 Manager Agent 和多个 Worker Agent。Manager Agent 负责根据当前的判断结果反思和提出新的质量标准假设（如"代码应该有清晰的变量命名"、"数学推理步骤应该完整"）。Worker Agent 负责根据给定的标准集合对人类偏好对进行成对判断。每轮迭代中，Worker 的判断结果与人类标注对比，Manager 根据错误案例分析标准的不足之处，增加新标准或修改现有标准。如此迭代直到标准集合在验证集上达到满意的准确率
    - 设计动机：单个 Agent 难以同时承担标准提出和验证的双重角色，分离为 Manager 和 Worker 可以形成"假设-验证"的科学发现循环，系统性地探索标准空间

2. **知识库增强（Knowledge Base Boosting）**:

    - 功能：利用先前工作中已发现的质量标准加速 CritiQ Flow 的收敛
    - 核心思路：从数据质量相关的先前研究论文中提取已知的质量标准（如 QuRating 等工作中的标准），构建一个结构化的知识库。在 CritiQ Flow 初始化时，Manager Agent 可以参考知识库中的已有标准作为起点，而不是完全从零开始探索。知识库的格式为 JSON，包含标准名称和描述
    - 设计动机：完全从零开始的标准挖掘效率较低且可能遗漏已被验证的重要维度，利用领域知识积累可以大幅提升效率和标准质量

3. **CritiQ Scorer — 高效质量评分器**:

    - 功能：将语言化标准转化为可大规模应用的数值评分工具
    - 核心思路：先用 CritiQ Flow 挖掘出的标准指导 Worker Agent 对大量数据进行标注（成对比较），生成带有质量排序信息的标注数据。然后基于 Qwen2 等模型训练一个 reward model 形式的评分器，输入一个数据样本输出一个质量分数。训练使用标准的 pairwise ranking loss。训练好的 Scorer 可以快速为海量数据打分，根据分数进行温度采样（Gumbel distribution sampling）来选择训练数据子集
    - 设计动机：LLM Agent 标注虽然准确但速度慢且成本高，训练轻量级 Scorer 可以在保留标准精度的同时实现大规模数据选择

### 损失函数 / 训练策略

CritiQ Scorer 使用 pairwise ranking loss 进行训练：给定一对数据样本 $(x_w, x_l)$（其中 $x_w$ 被判断为质量更高），模型学习让 $f(x_w) > f(x_l)$。具体采用 margin-based 损失函数。训练使用 DeepSpeed ZeRO-2 进行分布式训练，8卡并行，学习率 2e-5，训练 3 个 epoch，warmup ratio 0.2。

## 实验关键数据

### 主实验

在代码（Code）、数学（Math）和逻辑（Logic）三个领域验证数据选择效果。使用持续训练（continual training）Llama 3.1 模型来评估选出数据的质量。

| 领域 | 方法 | 人类偏好准确率 | 下游任务提升 |
|------|------|---------------|-------------|
| Code | 随机采样 | — | 基线 |
| Code | Perplexity | 61.2% | +1.3% |
| Code | 分类器 | 67.8% | +2.1% |
| Code | CritiQ | 82.5% | +4.7% |
| Math | 随机采样 | — | 基线 |
| Math | Perplexity | 58.9% | +0.8% |
| Math | CritiQ | 79.3% | +3.9% |
| Logic | 随机采样 | — | 基线 |
| Logic | CritiQ | 80.1% | +3.2% |

### 消融实验

| 配置 | Code 准确率 | Math 准确率 | 说明 |
|------|------------|------------|------|
| Full CritiQ Flow | 82.5% | 79.3% | 完整系统 |
| w/o 知识库 | 76.8% | 73.5% | 去掉知识库增强，从零开始挖掘 |
| w/o 反思机制 | 74.2% | 71.8% | Manager 不做错误分析和标准修正 |
| w/o 多数投票 | 78.1% | 75.6% | Worker 只用单次判断不做投票 |
| 固定标准（不迭代） | 70.5% | 67.2% | 只用初始标准不做迭代优化 |

### 关键发现

- 知识库增强贡献了约 5-6% 的准确率提升，说明领域先验知识对标准挖掘很有价值，但即使没有知识库，CritiQ 仍然显著优于 perplexity 和分类器方法
- 反思机制是第二大贡献因素，Manager Agent 根据错误案例调整标准的能力至关重要
- 标准在迭代过程中展现出有趣的演化模式：初期标准较为笼统（如"代码应该正确"），随着迭代逐步细化为更具体的标准（如"代码应该有错误处理机制"、"变量命名应该有语义"）
- 仅使用约 30 个标注对就能达到 80%+ 的人类偏好预测准确率，数据效率极高
- 挖掘出的标准具有可解释性和可复用性，可以直接供人类专家审查和编辑

## 亮点与洞察

- **极低标注成本下的高质量标准挖掘**：仅需约 30 个人类偏好标注对就能自动发现有效的质量标准，这个成本远低于训练分类器所需的数百甚至数千个标注。这让数据选择的门槛大幅降低
- **可解释性是核心优势**：与 perplexity 或分类器等黑箱方法不同，CritiQ 产出的是自然语言描述的质量标准，人类可以直接阅读、理解、修改和复用，这在实际生产环境中极具价值
- **多 Agent 协作的"假设-验证"范式**：Manager-Worker 的设计模式可以推广到其他需要从少量反馈中发现隐式规则的场景，如标注指南自动生成、评估标准发现等

## 局限与展望

- 目前仅在代码、数学和逻辑三个相对结构化的领域验证，未测试在更开放的领域（如创意写作、对话）中的效果
- CritiQ Flow 的迭代优化依赖强大的 LLM（如 GPT-4）作为 Manager 和 Worker，API 调用成本不低
- 知识库的构建仍然需要人工从先前文献中提取标准，尚未实现完全自动化
- 标准的覆盖度取决于初始偏好标注对的多样性，如果标注对覆盖的质量维度不全面，可能遗漏重要标准

## 相关工作与启发

- **vs QuRating**: QuRating 使用 LLM 在预定义的质量维度上打分，维度本身是人工设计的；CritiQ 的标准是自动挖掘的，更灵活且能发现人工可能忽略的维度
- **vs DSIR/DsDm**: 这些方法使用 perplexity 或分布匹配进行数据选择，是黑箱的数值方法；CritiQ 产出可解释的语言标准，在透明度和可调整性上有本质优势
- **vs AlpaGasus/LIMA**: 这些工作通过精选少量高质量数据来训练模型，但选择标准仍然是隐式的；CritiQ 可以为这类工作提供自动化的、可解释的选择标准

## 评分

- 新颖性: ⭐⭐⭐⭐ 多 Agent 协作自动挖掘数据质量标准的思路新颖，将数据选择从"判断质量"提升到"发现标准"
- 实验充分度: ⭐⭐⭐⭐ 三个领域的验证加上详细的消融和标准演化分析，实验设计完整
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机链条完整，图表设计合理
- 价值: ⭐⭐⭐⭐⭐ 解决了数据选择中标准不透明的核心痛点，30对标注就能工作的低成本特性使其极具实用价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Predict Training Data Quality via Its Geometry in Metric Space](../../NeurIPS2025/llm_pretraining/predict_training_data_quality_via_its_geometry_in_metric_space.md)
- [\[AAAI 2026\] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](../../AAAI2026/llm_pretraining/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)
- [\[ACL 2025\] Data-Constrained Synthesis of Training Data for De-Identification](data-constrained_synthesis_of_training_data_for_de-identification.md)
- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2025\] SCAR: Data Selection via Style Consistency-Aware Response Ranking for Efficient Instruction-Tuning](scar_style_consistency_data_selection.md)

<!-- RELATED:END -->
