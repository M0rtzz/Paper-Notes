---
title: >-
  [论文解读] From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts
description: >-
  [ACL 2025][LLM/NLP][sample efficiency] 首次直接研究事实在预训练数据中出现频次与 LLM 能否回忆该事实之间的关系，提出两种样本效率指标，发现不同架构/规模的模型在高频事实上表现相似但在低频事实上差异显著——低频事实的学习能力是区分模型样本效率的关键。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "sample efficiency"
  - "fact learning"
  - "knowledge probing"
  - "pre-training"
  - "power law"
---

# From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts

**会议**: ACL 2025  
**arXiv**: [2506.16912](https://arxiv.org/abs/2506.16912)  
**代码**: [https://github.com/Jabbawukis/sample-efficiency-evaluation](https://github.com/Jabbawukis/sample-efficiency-evaluation)  
**领域**: LLM/NLP  
**关键词**: sample efficiency, fact learning, knowledge probing, pre-training, power law

## 一句话总结
首次直接研究事实在预训练数据中出现频次与 LLM 能否回忆该事实之间的关系，提出两种样本效率指标，发现不同架构/规模的模型在高频事实上表现相似但在低频事实上差异显著——低频事实的学习能力是区分模型样本效率的关键。

## 研究背景与动机

### 领域现状

**领域现状**：LLM 通过预训练存储大量事实知识，但样本效率（需要看到多少次才能学会）未被系统研究。

**现有痛点**：真实文本中信息遵循长尾分布，模型需要从少量出现中学习稀有事实。现有工作比较模型性能时不考虑训练数据中的频率信息。

**核心矛盾**：两个在相同数据上训练的模型，哪个更能从少量曝光中学习事实？

**本文目标** 建立事实频率→回忆能力的量化框架来衡量样本效率。

**切入角度**：在同一预训练数据上训练多种模型，标注每个事实在训练数据中的出现频次，用 BEAR probe 评估回忆能力。

**核心 idea**：样本效率应被建模为事实回忆概率关于训练曝光次数的函数——斜率 $\alpha_m$ 越大，模型越高效。

## 方法详解

### 整体框架
Wikipedia 语料统计事实频率 -> 在同一语料上预训练多种模型 -> BEAR 知识探测 -> 按频率分桶分析 -> 计算加权准确率和幂律拟合的样本效率指标。

### 关键设计

1. **事实频率统计**

    - 对 BEAR probe 中的每个事实三元组 (s,r,o)，在训练语料中搜索 s 和 o 在同一句中出现的次数
    - 使用别名和词形还原增加匹配率
    - 设计动机：估计模型在预训练中"看到"某事实的次数

2. **两种样本效率指标**

    - **加权准确率**：按频率分桶，低频桶权重更高 ($w_i = \exp(-0.05 \cdot l_i)$)
    - **幂律拟合 α_m**：$F(x) = 1 - (L_0 + \frac{x_0}{(1+x)^{\alpha_m}})$，α_m 越大样本效率越高
    - 设计动机：加权准确率直观但难以比较，α_m 提供单一可比指标

3. **模型训练**

    - 在 ~5B tokens Wikipedia 上训练
    - 三种架构 × 两种规模 = 6 个模型
    - 保存中间检查点跟踪学习动态
    - 设计动机：控制训练数据变量，只比较架构和规模的影响

## 实验关键数据

### 主实验 -- 按频率分桶的准确率

| 频率桶 | 模型A (大) | 模型A (小) | 模型B (大) | 模型B (小) |
|--------|----------|----------|----------|----------|
| 1-5 次 | ~25% | ~15% | ~20% | ~12% |
| 6-20 次 | ~40% | ~30% | ~35% | ~25% |
| 21-100 次 | ~60% | ~50% | ~55% | ~45% |
| 100+ 次 | ~75% | ~70% | ~73% | ~68% |

### 样本效率指标 α_m

| 模型 | α_m | 说明 |
|------|-----|------|
| 大模型 A | **0.35** | 最高效 |
| 大模型 B | 0.30 | |
| 小模型 A | 0.25 | |
| 小模型 B | 0.20 | 最低效 |

### 关键发现
- **高频事实上模型差异小**（都能学会），**低频事实上差异大**
- **更大的模型样本效率更高**：α_m 随规模增大
- **架构差异在低频事实上最明显**
- **幂律函数很好地拟合了频率-准确率关系**
- **样本效率在训练过程中逐步提高**（通过检查点跟踪验证）

## 亮点与洞察
- **首次将事实频率与知识探测直接关联**——填补了"训练数据特征→模型行为"的研究空白
- **幂律拟合提供了优雅的单一指标 α_m**来比较模型效率
- **低频事实是关键鉴别因子**的发现对训练数据策略有直接启示

## 局限与展望
- 频率估计方法（同句共现）可能不精确
- 仅在 Wikipedia 上训练，可能不代表多样化预训练
- 改进方向：更精确的频率估计、大规模预训练数据的分析

## 相关工作与启发
- **vs Kandpal et al. (2023)**：他们发现 LLM 在低频实体上表现差，本文量化了"差多少"
- **vs Neural Scaling Laws (Kaplan et al.)**：他们研究 loss 随数据量的幂律关系，本文研究事实回忆概率随频率的幂律关系

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次建立事实频率→回忆能力的量化框架
- 实验充分度: ⭐⭐⭐⭐ 控制变量充分，多架构多规模
- 写作质量: ⭐⭐⭐⭐ 指标设计精巧
- 价值: ⭐⭐⭐⭐ 对预训练数据策略和模型比较有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [\[ACL 2025\] Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](condor_enhance_llm_alignment_with_knowledge-driven_data_synthesis_and_refinement.md)
- [\[ACL 2025\] UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions](uaqfact_evaluating_factual_knowledge_utilization_of_llms_on_unanswerable_questio.md)
- [\[ACL 2025\] CogSteer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models](cogsteer_cognition-inspired_selective_layer_intervention_for_efficiently_steerin.md)

</div>

<!-- RELATED:END -->
