---
title: >-
  [论文解读] Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study
description: >-
  [ACL 2025][LLM/NLP][algorithmic fidelity] 基于德国纵向选举调查(GLES)的开放式问题数据，系统评估三个开源LLM（Llama2、Gemma、Mixtral）通过人口统计persona提示生成合成德国公众舆论的算法保真度，发现Llama2在亚群体代表性上表现最佳（JS距离0.28），但所有模型均表现出左倾政治偏见和群体内多样性降低的问题。
tags:
  - ACL 2025
  - LLM/NLP
  - algorithmic fidelity
  - opinion simulation
  - 提示学习
  - German survey
  - Jensen-Shannon distance
  - political bias
---

# Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study

**会议**: ACL 2025  
**arXiv**: [2412.13169](https://arxiv.org/abs/2412.13169)  
**代码**: [soda-lmu/llm-opinion-german](https://github.com/soda-lmu/llm-opinion-german)  
**机构**: LMU Munich, Munich Center for Machine Learning, University of Maryland
**领域**: LLM/NLP  
**关键词**: algorithmic fidelity, opinion simulation, persona prompting, German survey, Jensen-Shannon distance, political bias

## 一句话总结

基于德国纵向选举调查(GLES)的开放式问题数据，系统评估三个开源LLM（Llama2、Gemma、Mixtral）通过人口统计persona提示生成合成德国公众舆论的算法保真度，发现Llama2在亚群体代表性上表现最佳（JS距离0.28），但所有模型均表现出左倾政治偏见和群体内多样性降低的问题。

## 研究背景与动机

**领域现状**：近年来LLM被广泛尝试用于合成调查数据生成，即通过"silicon sampling"方法让LLM扮演特定人口统计背景的persona来模拟人类的调查回答。Argyle et al. (2023) 提出"算法保真度"概念，衡量LLM复制不同人类亚群体社会文化背景和观点差异的能力，在美国选举数据上展示了一定可行性。

**现有痛点**：现有研究存在三个显著局限。第一，绝大多数研究聚焦于英语和美国语境，对非英语国家（如德国多党制体系）的评估极为稀缺。第二，已有实验几乎全部使用封闭式选择题（如投票意向的单选），而开放式自由文本回答的场景更贴近真实调查但更难评估。第三，von der Heyde et al. (2025) 用GPT-3.5研究德国投票行为发现模型偏向绿党和左翼党，但仅限于单选题设置，无法揭示LLM在自由文本生成中的细粒度偏见。

**核心矛盾**：开放式回答的多样性和语义丰富度远超封闭式问卷，但这恰恰是LLM最容易生成刻板化、低多样性输出的场景——模型需要同时理解人口统计背景的条件分布并保持群体内的观点多样性，这两个目标之间存在张力。

**本文目标** 首次在德语开放式调查问题上系统评估多个开源LLM的算法保真度，回答三个具体问题：(1) 哪个模型在群体级别代表性最好？(2) 代表性在不同亚群体和时间波次上如何变化？(3) prompt中包含哪些人口统计变量对保真度影响最大？

**切入角度**：使用GLES Panel（德国纵向选举研究）的21波次面板数据，选取"德国当前面临的最重要问题是什么？"这一开放式问题，涵盖2019-2021年（含COVID前后）。用信息论指标（JS距离、条件熵、互信息、Cramér's V）全面量化分布对齐和变量关联，并设计消融实验解耦各变量的贡献。

**核心 idea**：通过persona提示让LLM模拟德国公民回答开放式政治问题，用分类器将自由文本映射到16类编码后，以信息论指标系统评估分布保真度和亚群体偏差。

## 方法详解

### 整体框架

整个流程分为四步：(1) 从GLES数据集提取受访者的人口统计特征（年龄、性别、政党倾向、地区、学历、职业资质），构造德语persona prompt；(2) 用三个开源LLM（Llama2-13B、Gemma-7B、Mixtral-8x7B）生成文本回答；(3) 训练BERT分类器将LLM输出映射到16个粗粒度类别；(4) 用JS距离、条件熵、信息增益和Cramér's V等指标比较合成数据与真实调查数据的分布对齐度。

### 关键设计

1. **Persona Prompt构造与多变量编码**:

    - 功能：将真实调查受访者的人口统计画像转化为LLM可理解的德语提示
    - 核心思路：使用德语模板，将6个人口统计变量（年龄、性别、政党倾向、地区、学历、职业资质）填入占位符，指示LLM以该角色身份回答"德国当前面临的最重要问题"。提示语言选择德语而非英语，与原始GLES调查保持一致
    - 设计动机：用德语提示确保领域一致性，避免翻译偏差；6个变量覆盖社会经济的核心维度，在保证信息量的同时不过度约束模型

2. **LLM输出分类Pipeline（标注→BERT→全量推理）**:

    - 功能：将LLM生成的开放式文本归入16个预定义类别，使分布比较成为可能
    - 核心思路：首先从三个LLM各随机抽取500条输出进行人工标注（共1,500条），然后微调德语BERT分类器，在测试集上达到加权F1=0.93后用于全量自动分类。16类编码方案遵循GESIS的编码标准，将原始50+细粒度类合并为16个粗粒度类
    - 设计动机：开放式回答不能直接比较分布，必须先标准化到同一类别空间。训练分类器相比直接用LLM分类更稳定，且避免了循环使用被评估对象作为评估工具的方法论问题

3. **信息论评估指标体系**:

    - 功能：从多个维度量化合成数据与真实调查的分布对齐程度
    - 核心思路：JS距离衡量整体分布对齐（取值0-1，越低越好）；条件熵测量已知亚群体后回答的剩余不确定性；信息增益（互信息）衡量人口统计变量对回答的预测力；Cramér's V检验LLM是否保持了输入变量与输出类别之间的关联模式
    - 设计动机：单一指标无法全面评估保真度——JS距离衡量群体级分布对齐，条件熵和信息增益揭示亚群体级的建模质量，Cramér's V检验变量间关联模式是否失真

## 实验关键数据

### 主实验

**实验1：三模型比较（Wave 12, COVID前）**

| 指标 | Gemma | Llama2 | Mixtral | 真实调查 |
|------|-------|--------|---------|----------|
| COVID正则匹配率 | **0.42** | 0.03 | 0.002 | 0 |
| JS距离(↓) | 0.62 | **0.28** | 0.29 | - |
| 回答熵 | 2.26 | **2.90** | 2.56 | 2.93 |
| 非德语回答率 | 0.02 | 0.06 | 0.03 | - |
| 无回答率 | 0 | 0 | 0.05 | 0.04 |

**实验2：Llama2纵向分析（Wave 12-21）**

| 波次 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 均值 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| LLM熵 | 2.90 | 0.58 | 1.67 | 1.31 | 2.12 | 2.20 | 2.27 | 2.46 | 2.46 | 2.49 | 2.04 |
| Survey熵 | 2.93 | 2.02 | 2.24 | 2.31 | 2.53 | 2.82 | 2.75 | 2.85 | 2.92 | 2.19 | 2.55 |
| JS距离 | 0.29 | 0.29 | 0.24 | 0.22 | 0.20 | 0.23 | 0.23 | 0.22 | 0.24 | 0.30 | 0.24 |

### 消融实验

**实验3：变量消融（Cramér's V对比）**

| Prompt变量 | 真实调查 | LLM-仅一个变量 | LLM-全部变量 |
|------------|----------|----------------|-------------|
| 年龄 | 0.09 | 0.09 | 0.07 |
| 学历 | 0.06 | **0.25** | 0.05 |
| 性别 | 0.08 | **0.20** | 0.16 |
| 政党倾向 | 0.16 | **0.35** | 0.17 |
| 地区 | 0.06 | **0.42** | 0.15 |
| 职业资质 | 0.08 | 0.12 | 0.07 |

### 关键发现

- **Gemma严重幻觉**：42%回答涉及COVID-19（数据采集于2019年11月，COVID爆发前），直接被排除后续分析，说明LLM时间知识的覆盖可能导致严重的分布偏移
- **多样性与代表性的负相关**：Survey熵与JS距离的Pearson相关 $r=-0.35$，调查回答越多样化，LLM代表性越低。模型在共识度高（如COVID爆发时92.4%回答集中在健康政策）的场景更准确
- **系统性左翼偏见**：Llama2对绿党和左翼党的建模JS距离显著低于AfD（右翼），且这一偏差不能完全用亚群体内多样性差异解释，指向模型训练数据和RLHF的系统性倾向
- **单变量过度聚焦效应**：仅提供一个变量时模型过度依赖（Cramér's V从0.16跳到0.35），多变量共同提供时关联回到合理范围
- **政党变量影响最大**：消融分析中仅添加政党变量的JS距离降幅最大，仅排除政党变量则性能下降最严重

## 亮点与洞察

- **开放式回答的评估范式创新**：通过"开放式文本→人工标注→BERT分类→分布指标"的pipeline，解决了自由文本无法直接做分布比较的难题。这个流程对其他开放式survey模拟研究有直接的方法论价值
- **信息增益分析揭示刻板化机制**：Mixtral在亚群体（尤其是绿党和AfD）的条件熵骤降，意味着模型为特定群体生成高度同质的回答。这种通过信息论指标量化"刻板化程度"的方法可迁移到其他人群偏见评估场景
- **消融实验的variable交互效应**：揭示了一个反直觉现象——给模型提供更少的人口统计变量反而导致更强的变量-回答关联，说明LLM的条件生成并非简单的贝叶斯后验采样，而是受prompt结构强烈影响

## 局限与展望

- **模型覆盖有限**：仅测试3个开源LLM，未包含GPT-4、Claude等闭源模型，后者可能因更大规模训练和更细致的RLHF有不同表现
- **分类粒度损失**：16类粗粒度编码必然丢失自由文本回答中的细微语义差异，未来可结合embedding空间的连续距离度量
- **零样本限制**：仅使用零样本提示，未探索few-shot示例或微调在改善保真度方面的潜力
- **单一问题泛化性**：评估仅基于一个调查问题（"最重要问题"），不同类型问题（如价值观、政策偏好）的保真度可能不同

## 相关工作与启发

- **vs Argyle et al. (2023)**：他们在美国选举数据上用GPT-3的Silicon Sampling方法评估算法保真度，但仅限英语封闭式问题。本文扩展到德语开放式场景，发现保真度挑战更大
- **vs von der Heyde et al. (2025)**：他们用GPT-3.5的单选投票预测也发现左翼偏见，但无法揭示刻板化的程度。本文通过条件熵和信息增益提供了更细粒度的偏见量化
- **vs Santurkar et al. (2023)**：他们发现LLM观点分布与美国调查有显著差异。本文在非英语场景中确认了类似的发现，且提供了变量消融证据

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在德语开放式调查上系统评估LLM算法保真度，场景和方法都有新意
- 实验充分度: ⭐⭐⭐⭐ 三个实验（模型比较+纵向分析+消融）设计严谨，信息论指标体系完整
- 写作质量: ⭐⭐⭐⭐ 数据呈现清晰，图表丰富，讨论与结论逻辑连贯
- 价值: ⭐⭐⭐ 主要是分析性工作，缺少改进方法的提出，对实际应用的指导有限

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] FoodTaxo: Generating Food Taxonomies with Large Language Models](foodtaxo_generating_food_taxonomies_with_large_language_models.md)
- [\[ACL 2025\] Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions](language_model_fine-tuning_on_scaled_survey_data_for_predicting_distributions_of.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)
- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)

<!-- RELATED:END -->
