---
title: >-
  [论文解读] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models
description: >-
  [ACL 2025][LLM 其他][语言可接受性] 本文比较了多种自回归语言模型（GPT系列、Llama系列等）在语言可接受性判断任务上的表现，通过系统实验揭示了模型规模、训练数据和架构对语法判断能力的影响，并探讨了模型的语法知识是否与人类语言直觉一致。 领域现状：语言可接受性判断（Linguistic Acceptabi…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "语言可接受性"
  - "自回归语言模型"
  - "语法判断"
  - "语言学评估"
  - "困惑度"
---

# Comparing Linguistic Acceptability Judgments of Autoregressive Language Models

**会议**: ACL 2025  
**arXiv**: N/A  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 语言可接受性, 自回归语言模型, 语法判断, 语言学评估, 困惑度

## 一句话总结
本文比较了多种自回归语言模型（GPT系列、Llama系列等）在语言可接受性判断任务上的表现，通过系统实验揭示了模型规模、训练数据和架构对语法判断能力的影响，并探讨了模型的语法知识是否与人类语言直觉一致。

## 研究背景与动机

**领域现状**：语言可接受性判断（Linguistic Acceptability Judgment）是评估语言模型是否真正"理解"语言结构的经典任务。给定一个句子，判断其在语法上是否可接受（如 "The cat sat on the mat" vs "*The cat sat on"）。CoLA（Corpus of Linguistic Acceptability）等基准数据集被广泛用于评估模型的语法能力。

**现有痛点**：（1）早期NLP研究主要在编码器模型（BERT等）上评估语言可接受性，但现代主流已转向自回归（decoder-only）模型，其语法判断能力尚未被系统评估；（2）自回归模型没有天然的句子级表示，判断可接受性需要借助困惑度（perplexity）等代理指标，但不同代理指标的可靠性未被充分比较；（3）模型规模急剧增长，但更大的模型是否必然在语法判断上更好仍不清楚。

**核心矛盾**：自回归模型在生成任务上表现卓越，但这并不意味着它们拥有精确的语法知识。模型可能"说出"语法正确的句子却无法正确"判断"句子的语法性，因为生成和判断是不同的语言能力。

**本文目标**：（1）系统比较不同自回归模型的语法判断能力；（2）比较不同评估方案（困惑度、提示判断、概率对比等）的可靠性；（3）分析模型在细粒度语法现象（一致性、岛屿约束、绑定理论等）上的表现差异。

**切入角度**：作者从语言学视角出发，不仅使用CoLA等通用基准，还精心设计了覆盖多种语法现象的最小对测试（minimal pair tests），每对仅改变一个语法特征，从而精确诊断模型的语法知识。

**核心 idea**：通过困惑度差异、直接提示和概率对比三种评估范式的系统比较，发现自回归模型的语法判断能力与规模呈对数关系而非线性关系，且不同语法现象对模型的挑战程度差异悬殊。

## 方法详解

### 整体框架
评估框架包含三层设计：（1）测试集层——综合使用CoLA、BLiMP和自定义最小对数据集；（2）评估方法层——设计三种从自回归模型中提取语法判断的方案；（3）分析层——按语法现象类别、模型规模和训练数据细分结果。评估涵盖GPT-2/3/4、Llama-2/3、Mistral、Phi等模型家族。

### 关键设计

1. **多范式评估方案**:

    - 功能：从自回归模型中提取语言可接受性判断
    - 核心思路：设计三种评估范式——（a）困惑度法：计算句子的困惑度（perplexity），低困惑度对应高可接受性；（b）直接提示法：通过提示让模型直接回答句子是否语法正确（"Is the following sentence grammatically correct? ..."）；（c）概率对比法：给定最小对（一个语法正确，一个不正确），比较两者的对数概率差。三种方法各有优劣——困惑度不需要任何提示但受频率效应干扰，直接提示依赖模型的指令遵循能力，概率对比最精确但需要配对数据
    - 设计动机：不同评估方案可能给出不同结论，比较它们的一致性和可靠性对后续研究有重要参考价值

2. **细粒度语法现象分类**:

    - 功能：诊断模型在具体语法现象上的知识掌握程度
    - 核心思路：将测试集按语法现象分为六大类：（a）主谓一致（Subject-verb agreement）；（b）论元结构（Argument structure）；（c）岛屿约束（Island constraints）；（d）绑定理论（Binding theory）；（e）否定极性项（NPI licensing）；（f）时态/体/语气（TAM）。每类包含至少100个最小对示例，确保统计显著性。分析每种模型在哪些语法现象上表现最好/最差
    - 设计动机：汇总的准确率掩盖了模型在不同语法现象上的巨大差异，细粒度分析才能揭示模型"知道"什么语法和"不知道"什么语法

3. **规模效应分析**:

    - 功能：研究模型参数量对语法判断能力的影响规律
    - 核心思路：在同一模型家族内（如Llama-2的7B/13B/70B）比较不同规模的语法判断准确率，绘制准确率-参数量曲线。使用对数坐标系拟合曲线，测量"语法涌现"是否存在突变点。同时比较训练数据量（tokens数）与参数量各自的贡献
    - 设计动机：理解规模效应的形态（线性/对数/阶跃）对预测未来模型的语法能力和决定是否需要专门的语法训练至关重要

### 损失函数 / 训练策略
本文不涉及模型训练，纯评估性工作。使用的模型均为公开发布的预训练/指令微调版本。

## 实验关键数据

### 主实验

| 模型 | CoLA-MCC | BLiMP准确率 | 概率对比准确率 | 直接提示准确率 |
|------|----------|------------|--------------|-------------|
| GPT-2 (124M) | 0.21 | 68.3% | 71.5% | 52.1% |
| GPT-2 (1.5B) | 0.35 | 76.8% | 79.2% | 58.3% |
| Llama-2-7B | 0.41 | 81.2% | 84.3% | 69.7% |
| Llama-2-70B | 0.52 | 86.7% | 89.1% | 78.4% |
| Llama-3-8B | 0.48 | 84.5% | 87.2% | 74.3% |
| GPT-4 | 0.58 | **89.3%** | **91.5%** | **83.6%** |

### 消融实验（按语法现象类别，GPT-4概率对比）

| 语法现象 | 准确率 | 说明 |
|---------|--------|------|
| 主谓一致 | 95.2% | 最高，模型掌握最好的语法规则 |
| 论元结构 | 90.8% | 较好，及物/不及物区分清晰 |
| 时态/体/语气 | 88.3% | 良好，但某些罕见时态有困难 |
| 否定极性项 | 82.1% | 中等，"any"的许可条件有时判断错误 |
| 绑定理论 | 76.5% | 较弱，长距离照应关系判断困难 |
| 岛屿约束 | 71.3% | 最弱，复杂的句法移动约束掌握不足 |

### 关键发现
- 概率对比法是最可靠的评估方案，一致性和灵敏度都最高；直接提示法最不可靠，小模型几乎无法遵循指令
- 语法判断准确率与模型规模呈对数关系，从124M到1.5B提升巨大，从7B到70B提升递减
- 所有模型在岛屿约束和绑定理论上表现最差，这些涉及长距离依赖和复杂句法结构的现象仍是重大挑战
- Llama-3-8B在某些语法测试上接近Llama-2-70B，说明训练数据质量和训练方法的进步也很重要

## 亮点与洞察
- 三种评估范式的系统比较非常有价值——此前文献中不同论文使用不同方案导致结果不可比，本文提供了直接对比
- 细粒度语法现象分析揭示了有趣的层次结构：表层语法（一致性）几乎完美，深层语法（岛屿约束）仍然薄弱，这与理论语言学中对语法复杂度的排序一致
- 对数增长曲线的发现暗示单纯增大模型可能无法解决深层语法问题，可能需要引入语法归纳偏置

## 局限与展望
- 仅评估英语，不同语言的语法现象差异巨大（如自由语序语言的评估更困难）
- 最小对测试虽然精确但人工性强，与自然语言中遇到的语法错误分布不同
- 未分析指令微调（RLHF/DPO）对语法判断能力的影响，这是一个有趣的方向
- 模型可能在判断时利用了表面线索（如词频差异）而非真正的语法知识

## 相关工作与启发
- **vs BLiMP**: BLiMP是本文使用的基准之一，但本文扩展了评估的模型范围和评估方法
- **vs CoLA/GLUE**: CoLA通过MCC指标评估，本文发现该指标与困惑度法的相关性适中，概率对比法更有区分力
- **vs Syntax-Probing**: 之前的句法探针工作聚焦于模型内部表示，本文关注行为层面的语法判断，两者互补

## 评分
- 新颖性: ⭐⭐⭐ 评估方法论为主，无新模型或新方法
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖广泛的模型和语法现象，分析细致入微
- 写作质量: ⭐⭐⭐⭐ 语言学背景介绍清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ 为评估LLM语法能力提供了标准化方案和重要实证发现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments](explain-then-process_using_grammar_prompting_to_enhance_grammatical_acceptabilit.md)
- [\[ACL 2025\] Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models](detecting_referring_expressions_in_visually_grounded_dialogue_with_autoregressiv.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)
- [\[ACL 2025\] Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](deontological_keyword_bias.md)
- [\[ACL 2025\] AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models](ai_as_a_novel_ethical_agent_exploring_moral_judgments_by_large_language_models.md)

</div>

<!-- RELATED:END -->
