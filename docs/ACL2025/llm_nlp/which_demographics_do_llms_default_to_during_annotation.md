---
title: >-
  [论文解读] Which Demographics Do LLMs Default to During Annotation?
description: >-
  [ACL2025][LLM/NLP][LLM annotation] 通过对比 LLM 在无人口统计信息(N)、有人口统计(SD)、安慰剂信息(P)三种 prompt 条件下的标注行为，揭示 LLM 在主观标注任务(冒犯性/礼貌性)中默认更接近白人、年轻、高学历群体的标注模式，且人口统计 prompting 确实产生了比安慰剂信息更系统性的影响。
tags:
  - ACL2025
  - LLM/NLP
  - LLM annotation
  - demographic bias
  - 提示学习
  - perspectivism
  - offensiveness
  - politeness
---

# Which Demographics Do LLMs Default to During Annotation?

**会议**: ACL2025  
**arXiv**: [2410.08820](https://arxiv.org/abs/2410.08820)  
**代码**: [uni-bamberg.de/nlproc/resources/llms-default-demographics](https://www.uni-bamberg.de/en/nlproc/resources/llms-default-demographics/)  
**领域**: llm_nlp  
**关键词**: LLM annotation, demographic bias, socio-demographic prompting, perspectivism, placebo prompting, offensiveness, politeness

## 一句话总结

通过对比 LLM 在无人口统计信息(N)、有人口统计(SD)、安慰剂信息(P)三种 prompt 条件下的标注行为，揭示 LLM 在主观标注任务(冒犯性/礼貌性)中默认更接近白人、年轻、高学历群体的标注模式，且人口统计 prompting 确实产生了比安慰剂信息更系统性的影响。

## 研究背景与动机

**领域现状**：LLM 越来越多地被用于自动数据标注，尤其是零样本/少样本场景。然而标注本质上是主观的——不同人口统计背景的标注者对同一文本的冒犯性、礼貌性评价不同（如老年女性可能认为"bro"一词冒犯，但青少年男性觉得正常）。

**现有痛点**：LLM 标注缺乏人类标注者的多样性。之前的研究（Beck et al. 2024, Mukherjee et al. 2024）试图用人口统计 prompting 注入多样性，但**未发现一致的模式**，甚至认为 prompt 中的人口统计信息效果不显著。

**核心矛盾**：LLM 必然有某种"默认人设"——当不给定人口统计信息时，其标注行为更接近某些群体而非另一些。这种隐含偏见会导致少数群体观点被边缘化，但目前**缺乏系统性的实证研究来量化这种默认偏见**。

**本文目标**：(RQ1) LLM 默认模仿哪些人口统计群体？(RQ2) 人口统计 prompting 的影响是否比安慰剂信息更显著？(RQ3) 任务属性（冒犯性 vs 礼貌性评分）如何影响人口统计信息的作用？(RQ4) 不同模型是否表现一致？

**切入角度**：使用 Popquorn 数据集（专门为研究标注者特征与标注差异而设计，包含详细人口统计信息），同时引入安慰剂 prompting 作为对照组，系统比较三种 prompt 条件下 LLM 的标注行为。

**核心 idea**：LLM 在无人口统计信息的标注中隐含地"默认"为更接近白人和年轻群体的视角，而人口统计 prompting 确实能产生比安慰剂信息更系统的行为变化。

## 方法详解

### 整体框架

设计三类 prompt（SD/P/N），在两个主观评分任务上（冒犯性 1-5、礼貌性 1-5），用两个 LLM（GPT-4o、Claude 3.5 Sonnet）进行标注，然后将 LLM 标注与具有详细人口统计信息的人类标注进行对比分析。

### 关键设计一：三类 Prompt 对照

**功能**：设计社会人口统计（SD）、安慰剂（P）、无信息（N）三种 prompt 模板。
**为什么**：SD vs N 揭示人口统计信息的影响；P vs N 检测是否任意额外信息就能改变模型行为；SD vs P 对比人口统计信息的独特效应。
**怎么做**：

- **SD prompt**：注入性别、种族、年龄、职业、学历等真实标注者属性，如"You are a person of gender [gender], race [race], age [age]..."
- **P prompt**：注入无关属性（身高、星座、门牌号、爱好、最喜欢的颜色），如"You are a person of height [height], Zodiac sign [zodiac sign]..."
- **N prompt**：不含任何人口统计信息，或使用"any gender, any race..."的通用描述
- 每种类型设计 2 个模板变体以检验模板稳定性

### 关键设计二：数据集选择——Popquorn

**功能**：使用 Popquorn 数据集进行实验。
**为什么**：该数据集专为研究标注者人口统计属性对标注的影响而设计，包含 45000 条标注（1484 名标注者），含性别、种族、年龄、职业、学历等信息。
**怎么做**：

- 使用冒犯性评分（4500 条标注/1500 实例）和礼貌性评分（11151 条标注/3717 实例）两个子集
- 每个实例统一采样 3 名标注者，避免标注数量不均对分析的干扰
- 剔除拒绝披露人口统计信息的标注者

### 关键设计三：分析方法

**功能**：用两种分析方法推断 LLM 的"默认人设"。
**为什么**：单一分析维度可能遗漏信息。
**怎么做**：

- **混合效应回归分析**（Table 3）：以 LLM 标注与人类标注的距离为因变量，标注者人口统计为自变量，含随机截距（标注者和实例），检验哪些群体的标注与 LLM 默认输出差距最大
- **均值距离对比分析**（Table 4）：计算 SD prompt 输出与 N prompt 输出的均值距离 Δμ，距离越大说明该人口统计值偏离模型默认越远

### 模型和成本

- GPT-4o：$54；Claude 3.5 Sonnet：$60
- 每个实验仅运行一次（成本限制）
- 输出解析失败率极低（GPT-4o 22 例，Claude 3 例）

## 实验关键数据

### 主实验：LLM 默认人设的回归分析（Table 3）

| 人口统计属性 | 冒犯性(GPT-4o) | 冒犯性(Claude) | 礼貌性(GPT-4o) | 礼貌性(Claude) |
|---|---|---|---|---|
| **年龄**（每增1岁） | **+0.01** | **+0.01** | 0.00 | 0.00 |
| **种族（基准：白人）** | | | | |
| 黑人/非裔美国人 | ***+0.22** | **+0.19** | **+0.14** | **+0.15** |
| 亚裔 | +0.09 | +0.03 | -0.08 | 0.00 |
| 西裔/拉丁裔 | -0.11 | -0.05 | +0.09 | +0.12 |
| **学历（基准：<高中）** | | | | |
| 大学学位 | +0.05 | -0.09 | *-0.43 | *-0.48 |
| 研究生学位 | +0.06 | -0.01 | *-0.36 | *-0.44 |
| **性别（基准：男性）** | | | | |
| 女性 | 0.00 | -0.03 | -0.05 | -0.05 |
| 非二元 | -0.06 | -0.01 | -0.05 | -0.05 |

（正系数 = LLM 与该群体距离更远 = 不默认该群体；* p≤0.05, ** p≤0.01, *** p≤0.001）

### 消融：人口统计 Prompting 效应对比（Table 4 摘要）

| 属性 | Δμ GPT-4o 冒犯性 | Δμ Claude 冒犯性 |
|---|---|---|
| 男性 | 0.18 | 0.17 |
| 女性 | 0.20 | 0.15 |
| 非二元 | **0.29** | 0.17 |
| 白人 | 0.18 | 0.16 |
| 黑人/非裔 | 0.22 | 0.15 |
| 18-24岁 | 0.21 | 0.16 |
| >65岁 | 0.20 | 0.15 |

（安慰剂 prompt 分数在各属性值间保持一致稳定，无系统性差异）

### 关键发现

1. **种族偏见最显著**：LLM 与黑人/非裔美国人标注者的距离显著大于白人（冒犯性任务 +0.19~0.22 Likert 点），说明 LLM 默认更接近白人视角
2. **年龄效应**：在冒犯性任务中，LLM 与年龄更大的标注者距离更远（p≤0.01），默认偏向年轻人
3. **学历效应仅在礼貌性任务显著**：LLM 与低学历群体距离更大（高达 0.57 点），默认偏向高学历视角
4. **性别和职业无显著效应**：模型在这两个维度上未表现出明确偏好
5. **SD prompting > 安慰剂**：人口统计 prompting 产生系统性预测变化，而安慰剂 prompt 的变化无特定方向性
6. **礼貌性比冒犯性更受人口统计影响**：GPT-4o 在礼貌性任务的平均预测差异（0.25）高于冒犯性（0.19）
7. **Claude 特有模式**：随年龄增大，Claude 在礼貌性任务上的预测差异从 0.16 增至 0.26

## 亮点与洞察

- **安慰剂对照设计**：借鉴 Mukherjee et al. (2024) 的安慰剂概念（星座、门牌号等），巧妙地将人口统计 prompting 的效应与"任何额外信息"的效应区分开来
- **双分析法互补**：回归分析（LLM vs 人类标注距离）和均值距离分析（SD vs N prompt 差异）从两个角度交叉验证，增强了结论可信度
- **与前人工作的矛盾发现**：Beck et al. (2024) 和 Mukherjee et al. (2024) 报告人口统计 prompting 无一致效果，本文使用更适合的数据集（Popquorn）观察到了显著效应，说明数据集质量对研究结论至关重要
- **实际影响的审慎讨论**：尽管统计显著，效应量相对 1-5 Likert 量表较小（< 0.5 点），作者对过度解读保持谨慎

## 局限与展望

1. **仅 2 个模型、1 次运行**：受预算限制（共 $114），每个实验仅运行一次，统计功效有限
2. **数据集文化偏见**：Popquorn 人口统计变量以美国社会为中心（如种族分类），不适用于其他文化
3. **人口统计子群不均衡**：非二元性别仅 124 个冒犯性标注样本，难以得出可靠结论
4. **采样可能引入偏差**：为统一每实例 3 名标注者而进行下采样，可能扭曲某些人口统计群体的代表性
5. **缺乏事后统计检验**：回归模型未进行同方差性和共线性的事后检验
6. **效应量较小**：观察到的差异在 0.1-0.5 Likert 点之间，实际应用中的意义尚需验证
7. **仅 2 个任务**：冒犯性和礼貌性都属情感/社会判断类任务，应扩展到命名实体识别、情感分析等更多任务

## 相关工作与启发

### vs Beck et al. (2024) — 人口统计 Prompting 效果

Beck et al. 发现 prompt 中的人口统计信息影响远小于 prompt 格式技巧（如措辞方式）。本文使用专门为 perspectivism 研究设计的 Popquorn 数据集，观察到了种族和年龄的显著效应。关键区别在于数据集选择——Popquorn 的控制性设计使其更适合检测细微的人口统计效应。

### vs Mukherjee et al. (2024) — 安慰剂 Prompting

Mukherjee et al. 提出安慰剂 prompting 概念，发现大多数 LLM 表现出显著的响应变异性，质疑社会人口统计 prompting 的可靠性。本文采纳并扩展了安慰剂设计，发现 SD prompting 确实比安慰剂产生更系统性的效果（安慰剂分数跨属性值保持稳定，SD 则有方向性差异），部分化解了 Mukherjee 的质疑。

### vs Sun et al. (2025) — LLM 人口统计偏见

Sun et al. 发现 LLM 倾向于与白人参与者的感知更一致。本文验证了这一发现（白人标注与 LLM 距离最近），并进一步扩展到更多人口统计维度（年龄、学历、职业），提供了更全面的偏见画像。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 安慰剂+人口统计双对照设计有巧思，Popquorn 首次用于 LLM 分析
- **实验充分度**: ⭐⭐⭐ — 仅 2 个模型、2 个任务、1 次运行，规模较小；但分析方法多样且互补
- **写作质量**: ⭐⭐⭐⭐ — RQ 驱动结构清晰，相关工作覆盖全面，对局限性讨论坦诚
- **价值**: ⭐⭐⭐⭐ — 对 LLM 标注公平性有实际警示意义，安慰剂对照方法值得推广

<!-- RELATED:START -->

## 相关论文

- [Which of These Best Describes Multiple Choice Evaluation with LLMs?](multiple_choice_eval.md)
- [Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?](do_llms_give_psychometrically_plausible_responses_in_educational_assessments.md)
- [Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs](do_language_models_mirror_human_confidence_exploring_psychological_insights_to_a.md)
- [Do Language Models Understand Honorific Systems in Javanese?](do_language_models_understand_honorific_systems_in_javanese.md)
- [Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm](do_language_models_understand_the_cognitive_tasks_given_to_them_investigations_w.md)

<!-- RELATED:END -->
