---
title: >-
  [论文解读] DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing
description: >-
  [ACL 2025][自动作文评分] 发布DREsS大规模标准化评分准则数据集，包含三个子集（DREsS_New真实课堂数据1.7K + DREsS_Std标准化历史数据集6.5K + DREsS_CASE增强数据40.1K），提出基于腐蚀的作文增强策略CASE，将BERT基线的QWK分数从0.471提升至0.685（提升45.44%）。
tags:
  - ACL 2025
  - 自动作文评分
  - AES
  - EFL写作
  - 评分准则
  - 数据增强
  - CASE
---

# DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing

**会议**: ACL 2025  
**arXiv**: [2402.16733](https://arxiv.org/abs/2402.16733)  
**代码**: 未公开（数据集需提交consent form获取）  
**领域**: NLP教育 / 自动作文评分  
**关键词**: 自动作文评分, AES, EFL写作, 评分准则, 数据增强, CASE  

## 一句话总结
发布DREsS大规模标准化评分准则数据集，包含三个子集（DREsS_New真实课堂数据1.7K + DREsS_Std标准化历史数据集6.5K + DREsS_CASE增强数据40.1K），提出基于腐蚀的作文增强策略CASE，将BERT基线的QWK分数从0.471提升至0.685（提升45.44%）。

## 研究背景与动机

**领域现状**：自动作文评分（AES）在EFL（English as a Foreign Language）写作教育中是重要工具，可以为学生和教师提供实时评分反馈。

**现有痛点**：(1) **数据集与教学场景脱节**——ASAP等广泛使用的AES数据集由非专家标注，评分对象不是EFL学习者，且主要提供整体分数而非基于评分准则的分析性分数；(2) **评分准则不统一**——现有少量基于准则的数据集（ASAP P7-8、ASAP++、ICNALE EE）各有不同的准则定义和分数范围，无法联合训练；(3) **数据量不足**——可用的评分准则数据集规模小，严重限制模型性能。

**核心矛盾**：EFL写作教育需要基于评分准则（content, organization, language）的分析性评分，但缺乏(a)由领域专家标注的大规模数据集，(b)统一标准化的评分框架，(c)有效的数据增强方法。

**本文目标** 构建一个大规模、标准化的评分准则作文评分数据集，并提出数据增强策略解决数据稀缺问题。

**核心 idea**：整合新采集+标准化+增强三条路线构建48.9K样本的统一评分准则数据集，CASE增强通过向高分作文注入准则特定的腐蚀来生成带标签的合成样本。

## 方法详解

### 整体框架
DREsS数据集由三部分组成：
- **DREsS_New**：全新采集的真实课堂EFL作文（1,782篇），由11位英语教育专家按统一准则评分
- **DREsS_Std**：标准化整合4个历史数据集（ASAP P7-8、ASAP++ P1-2、ICNALE EE），统一至相同准则和分数范围（6,516篇）
- **DREsS_CASE**：使用CASE增强策略生成的合成样本（40,101篇）

统一三个评分准则：**Content**（内容）、**Organization**（结构）、**Language**（语言），分数范围1-5（步长0.5）。

### 关键设计

1. **DREsS_New数据集采集**：
    - 来源：韩国一所大学2020-2023年EFL写作课程
    - 学生：TOEFL写作分数15-21分的本科生
    - 任务：40分钟限时议论文（包含学期初pre-test和期末post-test）
    - 标注者：11位英语教育/语言学专家教师
    - 质量保证：标注前进行评分培训和标准化会议；ANOVA和Tukey HSD检验确认标注者间无显著差异（p<0.05）

2. **CASE腐蚀增强策略（Corruption-based Augmentation Strategy for Essays）**：
    - 核心思路：从高分作文（4.5-5.0分）出发，对三个准则分别注入不同类型的腐蚀，生成不同分数等级的合成样本
    - 腐蚀句子数公式：$n(S_c) = \lfloor n(S_E) \times (5.0 - x_i) / 5.0 \rceil$，其中$n(S_E)$为原始作文句子数，$x_i$为目标合成分数
    - **Content腐蚀**：随机替换句子为来自不同prompt的句子（跑题）——替换越多，内容质量越差
    - **Organization腐蚀**：随机交换作文中两个句子的位置——交换越多，结构越混乱
    - **Language腐蚀**：替换句子为BEA-2019 GEC数据集中的不合语法句子（编辑数>10的句子）——替换越多，语法错误越多

3. **历史数据集标准化**：
    - ASAP P7（4准则→3准则）：style×0.66 + convention×0.33 = Language
    - ASAP P8（6准则→3准则）：voice + word choice + sentence fluency + convention等权平均 = Language
    - ICNALE EE：vocabulary×0.4 + language use×0.5 + mechanics×0.1 = Language
    - 所有分数重缩放到1-5范围

### 损失函数
基线模型（BERT fine-tuning）使用标准回归损失，评估指标为Quadratic Weighted Kappa (QWK)。

## 实验

### 主实验

| 模型 | 训练数据 | Content | Organization | Language | Total QWK |
|:---:|:---:|:---:|:---:|:---:|:---:|
| gpt-3.5-turbo | N/A (zero-shot) | 0.239 | 0.371 | 0.246 | 0.307 |
| EASE (SVR) | DREsS | - | - | - | 0.360 |
| NPCR | DREsS | - | - | - | 0.507 |
| BERT | DREsS_New | 0.414 | 0.311 | 0.487 | 0.471 |
| BERT | + DREsS_Std | 0.599 | 0.593 | 0.587 | 0.551 |
| BERT | + DREsS_Std + CASE | **0.642** | **0.750** | **0.607** | **0.685** |

**主要发现**：三种数据来源的叠加效果显著——仅用DREsS_New训练QWK为0.471，加入标准化数据提升至0.551，再加入CASE增强数据达到0.685，总提升**45.44%**。

### 消融实验

| 增强参数$n_{aug}$ | Content最优 | Organization最优 | Language最优 |
|:---:|:---:|:---:|:---:|
| 最优值 | 0.5 | 2 | 0.125 |

**CASE增强量的影响**：三个准则的最优增强量差异大——Organization最高（因为腐蚀在作文内部进行，不依赖外部数据源大小），Content中等，Language最低（因为不合语法句子来源有限——仅605条）。

### 不同预训练模型对比

| 模型 | Content | Organization | Language | Total |
|:---:|:---:|:---:|:---:|:---:|
| BERT | 0.414 | 0.311 | 0.487 | 0.471 |
| Longformer | 0.409 | 0.312 | 0.475 | 0.463 |
| BigBird | 0.412 | 0.317 | 0.473 | 0.469 |
| GPT-NeoX | 0.410 | 0.313 | 0.446 | 0.475 |

**关键发现**：不同预训练语言模型对AES性能无显著差异，与Xie et al. (2022)的观察一致。

### ChatGPT评分实验

| Prompt策略 | Content | Organization | Language | Total |
|:---:|:---:|:---:|:---:|:---:|
| (A) 标准zero-shot | 0.320 | 0.248 | 0.359 | 0.336 |
| (B) 2-shot | 0.330 | 0.328 | 0.306 | 0.346 |
| (C) zero-shot + 准则解释 | 0.357 | 0.278 | 0.342 | 0.364 |
| (D) zero-shot + 反馈生成 | 0.336 | 0.361 | 0.272 | 0.385 |

**关键发现**：gpt-3.5-turbo在评分任务上表现最差，方差大且评分不一致，说明LLM直接做AES仍不可靠。

## 亮点
- **数据集贡献突出**：首个面向EFL写作教育、由领域专家标注的大规模评分准则数据集
- **CASE增强策略巧妙**：利用准则特定的腐蚀模式生成合成数据，直觉清晰且效果显著（+45.44%）
- **标准化工作实用**：统一了4个历史数据集到相同准则，便于后续研究直接使用
- **评估全面**：覆盖传统ML、fine-tuned LM、ChatGPT等多种baseline

## 局限性
- 仅关注英语写作，未覆盖其他L2语言
- DREsS_New主要来自韩国学生，可能存在文化/语言背景偏差
- CASE策略只能从高分作文生成低分样本，无法合成高分作文（需依赖LLM）
- 评分准则对齐过程中涉及主观权重设定（如style×0.66 + convention×0.33），可能引入偏差
- 数据集访问需提交consent form，非完全开放

## 相关工作
- **整体评分AES**：ASAP P1-6（10K样本，非专家标注），TOEFL11（不可公开获取），EASE/NPCR等模型
- **评分准则AES**：ASAP P7-8（仅2.3K样本），ASAP++（非专家标注），ICNALE EE（仅639样本）
- **本文定位**：第一个满足(1)面向EFL教学场景(2)专家标注(3)统一评分准则(4)大规模(5)公开可用五个条件的AES数据集

## 评分
- **创新性**: ⭐⭐⭐⭐ — CASE增强策略设计巧妙，评分准则标准化贡献大
- **实用性**: ⭐⭐⭐⭐⭐ — 直接可用于EFL写作教育系统，实际需求明确
- **技术深度**: ⭐⭐⭐ — 数据集驱动的工作，建模方法相对简单
- **实验充分度**: ⭐⭐⭐⭐ — 多模型、多策略对比充分，消融完整
- **总体推荐**: ⭐⭐⭐⭐ — 重要的数据集贡献，CASE策略有启发性

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Rubrik's Cube: Testing a New Rubric for Evaluating Explanations on the CUBE Dataset](rubriks_cube_testing_a_new_rubric_for_evaluating_explanations_on_the_cube_datase.md)
- [\[ACL 2025\] Research Borderlands: Analysing Writing Across Research Cultures](research_borderlands_analysing_writing_across_research_cultures.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [\[ACL 2025\] TabXEval: Why this is a Bad Table? An eXhaustive Rubric for Table Evaluation](tabxeval_why_this_is_a_bad_table_an_exhaustive_rubric_for_table_evaluation.md)
- [\[ACL 2025\] Enhancing Marker Scoring Accuracy through Ordinal Confidence Modelling in Educational Assessments](enhancing_marker_scoring_accuracy_through_ordinal_confidence_modelling_in_educat.md)

<!-- RELATED:END -->
