---
title: >-
  [论文解读] UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions
description: >-
  [ACL 2025][不可回答问题] 提出双语不可回答问题数据集UAQFact（13,970题），每个问题附带知识图谱事实知识，定义三个评估任务分别衡量LLM区分不可回答问题（UAQ）与可回答问题（ABQ）、利用内部/外部事实知识处理UAQ的能力，实验揭示即使LLM已存储相关知识也难以有效利用。
tags:
  - ACL 2025
  - 不可回答问题
  - 事实性知识
  - 知识图谱
  - LLM评估
  - 双语基准
  - 拒答率
---

# UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions

**会议**: ACL 2025  
**arXiv**: [2505.23461](https://arxiv.org/abs/2505.23461)  
**作者**: Chuanyuan Tan, Wenbiao Shao, Hao Xiong, Tong Zhu, Zhenhua Liu, Kai Shi, Wenliang Chen (苏州大学 & OPPO AI Center)  
**代码**: [github.com/cytan17726/UAQ_Fact](https://github.com/cytan17726/UAQ_Fact)  
**领域**: llm_nlp  
**关键词**: 不可回答问题, 事实性知识, 知识图谱, LLM评估, 双语基准, 拒答率  

## 一句话总结

提出双语不可回答问题数据集UAQFact（13,970题），每个问题附带知识图谱事实知识，定义三个评估任务分别衡量LLM区分不可回答问题（UAQ）与可回答问题（ABQ）、利用内部/外部事实知识处理UAQ的能力，实验揭示即使LLM已存储相关知识也难以有效利用。

## 研究背景与动机

### 问题背景
LLM在传统问答任务上表现优异，但现实场景中用户的提问可能没有确定的事实答案——即不可回答问题（UAQ）。例如"Nero Caesar的兄弟中谁还是Seti I的父亲？"——这类问题没有满足条件的事实答案。若LLM对UAQ给出虚构回答，将误导用户并产生不良后果。

### 已有工作不足

**缺乏事实知识支撑**：现有UAQ数据集（SelfAware、FalseQA、UnknownBench等）的问题来自网络抓取、头脑风暴或实体替换，仅提供答案/标签而不包含支撑性事实知识，无法评估LLM利用知识处理UAQ的能力

**仅支持英语**：已有数据集均为英文，无法评估跨语言泛化能力

**评测维度单一**：已有数据集仅支持UAQ/ABQ二分类任务，缺少对内部知识利用和外部知识利用的深度评估

### 核心动机
构建带有辅助事实知识的双语UAQ数据集，定义新任务来深度评估LLM是否能有效利用其已有和外部提供的事实知识来正确处理不可回答问题。

## 方法详解

### 数据集构建流程

#### 1. 问题类型定义
定义三种问题类型（QType）：
- **Inter（交集型）**：要求LLM返回两个集合的交集。对于UAQ，两个子问题的答案集交集为空集。例："谁既是Enneads的编辑又是The Sixth Sense的演员？"
- **Time（时间约束型）**：要求LLM在给定时间约束下回答。对于UAQ，时间约束不可满足。例："Erfurt在1957到1962年间与哪个城市结为友好城市？"（实际起始年份为1971）
- **Dilemma（候选答案型）**：提供候选答案供LLM选择。对于UAQ，所有候选答案均不正确。例："Segestes属于Mohawk还是Khamti部落？"（正确答案Cherusci不在候选中）

#### 2. 事实三元组采样
从Wikidata知识图谱中采样双语事实三元组作为知识支撑：
- 查询获取724个属性（如editor、cast member）及描述
- 按属性可理解性、出现频次≥5、可提供事实知识三个标准筛选
- 对每种QType构建不同查询获取相关实体和双语标签

#### 3. 模板生成与填充
利用GPT-3.5根据属性描述生成问题模板，人工检查所有864个模板（每种语言），修正或丢弃错误模板，最终将实体填入模板生成问题。

### 三个评估任务

#### Task 1：UAQ/ABQ判别
直接向LLM提供问题，评估其区分UAQ和ABQ的基本能力。

#### Task 2：内部知识利用评估
先通过多选题探测LLM是否存储了与UAQ相关的事实知识（知识通过率KPR），再结合Task 1结果计算知识感知拒答率（KRR）：

$$\text{KRR} = \left(1 + e^{-R_\Delta \cdot \text{KPR}^{-1}}\right)^{-1}$$

其中 $R_\Delta = R_{ua} - R_{ab}$，$R_{ua}$为UAQ拒答率，$R_{ab}$为ABQ拒答率。KRR越高说明LLM知识利用能力越强。

#### Task 3：外部知识利用评估
向LLM提供精心设计的Chain-of-Thought推理线索作为外部知识，包含：(1) 将问题分解为子问题，(2) 给出相关事实知识，(3) 基于前述信息回答。评估LLM能否利用外部知识正确处理UAQ。

### 数据集规模
- 总计13,970个问题（6,985 UAQ + 6,985 ABQ）
- 9,021个实体，724个属性
- 8,686个知识探测题 + 13,970条推理线索
- 全部数据双语（英文 + 中文）
- 人工检查通过率99.2%

## 实验关键数据

### Task 1：UAQ/ABQ判别结果

| 模型 | 英文 $R_{ua}$↑ | 英文 $R_{ab}$↓ | 英文 $R_\Delta$↑ | 英文 Acc↑ | 中文 $R_\Delta$↑ | 中文 Acc↑ |
|------|--------|--------|---------|------|---------|------|
| Llama3 | 38.80 | 17.91 | 20.89 | 53.21 | 9.95 | 35.88 |
| Mistral0.2 | 62.15 | 31.27 | 30.88 | 54.32 | 7.92 | 19.31 |
| Qwen2.5 | 59.10 | 29.03 | 30.07 | 47.73 | 16.63 | 43.77 |
| GLM4 | 55.05 | 31.22 | 23.83 | 49.63 | 14.13 | 41.53 |
| Gemini-1.5-pro | 66.50 | 12.25 | **54.24** | **69.62** | **38.40** | **53.98** |
| GPT-4o-mini | 85.05 | 42.15 | 42.91 | 50.51 | 18.14 | 47.02 |
| GPT-4 | 85.70 | 22.43 | **63.26** | 66.79 | 34.52 | 51.02 |

- 所有LLM的 $R_\Delta$ 均为正值，说明有一定UAQ/ABQ区分能力
- 最佳英文 $R_\Delta$ 仅63.26（GPT-4），中文仅38.40（Gemini-1.5-pro），UAQFact极具挑战性
- 中文整体 $R_\Delta$ 低于英文，说明中文UAQ更难判别

### Task 2：内部知识利用结果

| 模型 | 英文KPR↑ | 英文 $R_\Delta$↑ | 英文KRR↑ | 中文KPR↑ | 中文KRR↑ |
|------|---------|---------|---------|---------|---------|
| Llama3 | 73.11 | 20.89 | 57.10 | 52.71 | 54.71 |
| Mistral0.2 | 68.92 | 30.88 | 61.02 | 40.21 | 54.91 |
| Qwen2.5 | 70.57 | 30.07 | 60.49 | 60.44 | 56.84 |
| GPT-4 | **81.80** | **63.26** | 68.42 | **83.21** | 60.23 |
| Gemini-1.5-pro | 69.03 | 54.24 | **68.69** | 76.74 | **62.26** |

- LLM普遍具有较高KPR（68-82%英文），说明已存储大量事实知识
- 但KRR仅54-69%，暴露知识存储与知识利用之间的显著差距
- Gemini-1.5-pro的KPR不是最高，但KRR最优，知识利用效率最强

### Task 3：外部知识利用结果

提供CoT推理线索后，所有LLM表现均有提升：
- Llama3提升最显著：英文 $R_\Delta$ 从20.89跃升至74.54（+53.65）
- 开源模型获益比闭源模型更大（相对提升幅度更高）
- 闭源模型在中文设定下提升更明显，平均中文 $R_\Delta$ 甚至超过英文（73.08 vs 64.96）
- 但即使有验证过的外部知识，最佳英文 $R_\Delta$ 仅约75%，远未达到理想水平

## 关键发现

1. **知识存储≠知识利用**：LLM存储了丰富的事实知识（KPR高），但在面对UAQ时无法有效调用这些知识（KRR低），知识"有"和"用"之间存在明显鸿沟
2. **中文更具挑战性**：几乎所有模型在中文UAQ判别上 $R_\Delta$ 显著低于英文，即使KPR相近，中文KRR也普遍下降
3. **规模效应不均匀**：Qwen2.5系列从0.5B扩展到72B时，$R_\Delta$ 呈上升趋势，但 $R_{ua}$ 和 $R_{ab}$ 呈波动同步变化——拒绝更多UAQ的同时也错误拒绝了更多ABQ
4. **外部知识有帮助但不够**：CoT线索使所有模型提升，但闭源模型提供知识后趋向给出更确定回答（$R_{ua}$ 和 $R_{ab}$ 同时下降），不一定带来净收益
5. **知识利用与知识量级解耦**：Mistral0.2英文KPR低于Qwen2.5，但KRR更高，说明知识量少但利用效率更高的模型反而在UAQ判别上更优

## 亮点

- **首个知识增强的UAQ基准**：每个问题附带Wikidata知识图谱事实知识，支持深度评测，而非仅做分类
- **三任务评估体系**：基础判别 + 内部知识利用 + 外部知识利用，层层递进地剖析LLM处理UAQ的能力瓶颈
- **知识感知拒答率KRR**：创新性地将知识探测结果和拒答表现整合为单一指标，实现跨模型公平对比
- **双语设计**：英文+中文全覆盖，揭示跨语言性能差异，为多语言LLM评估提供新视角
- **高质量数据**：基于Wikidata严格采样+GPT-3.5模板生成+人工校验（99.2%通过率），保证问题质量可控

## 局限与展望

- **词法匹配评估**：拒答率通过关键词匹配计算（识别拒绝、道歉、弃权关键词），虽人工评估Cohen's Kappa达94.90，但仍与人工判断存在差距
- **知识来源单一**：仅基于Wikidata，受限于其覆盖范围和结构化特性，未涵盖需要常识推理或领域专业知识的UAQ
- **问题类型有限**：三种QType（Inter/Time/Dilemma）覆盖面有限，现实中UAQ形式更加多样（如错误前提、过度限定等）
- **模板生成质量**：GPT-3.5生成的模板存在语义错误和槽位缺失，依赖人工检查
- **开源模型规模**：开源LLM评估集中在7B左右，缺少与闭源模型参数量级相当的对比
- **知识探测设计**：多选题形式的知识探测可能不完全反映LLM真实的知识掌握程度

## 与相关工作的对比

- **SelfAware (Yin et al. 2023)**：3,369题，仅英文，无知识支撑，单任务——UAQFact规模更大（13,970题），双语且附带知识
- **FalseQA (Hu et al. 2023)**：4,730题，基于头脑风暴，提供可行回复而非事实答案——UAQFact基于KG构建，答案有知识图谱验证
- **UnknownBench (Liu et al. 2023)**：6,323题，通过替换实体为虚假实体构建UAQ——UAQFact保持所有实体真实，通过组合关系构造不可回答性
- **CREPE (Yu et al. 2022)**：8,466题，网络来源，无知识支撑——UAQFact每题均有Wikidata三元组支撑
- **MMLU/C-Eval**：评估LLM内部知识但非UAQ场景——UAQFact专注于UAQ下的知识利用评估

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将事实知识引入UAQ评估，三任务设计有独创性
- 实验充分度: ⭐⭐⭐⭐ — 覆盖7个模型系列、Qwen2.5全参数规模分析、双语对比，但缺少最新模型（如GPT-4o、Claude 3）
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，任务定义明确，图表排版规范
- 价值: ⭐⭐⭐⭐ — 揭示知识存储与利用的鸿沟这一关键洞察，对LLM诚实性和可靠性研究有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)
- [\[ACL 2025\] Factual Knowledge in Language Models: Robustness and Anomalies under Simple Temporal Context Variations](factual_knowledge_in_language_models_robustness_and_anomalies_under_simple_tempo.md)
- [\[ACL 2025\] Towards Effective Extraction and Evaluation of Factual Claims](towards_effective_extraction_and_evaluation_of_factual_claims.md)
- [\[ACL 2025\] Mamba Knockout for Unraveling Factual Information Flow](mamba_knockout_for_unraveling_factual_information_flow.md)
- [\[ACL 2025\] Truth Knows No Language: Evaluating Truthfulness Beyond English](truth_knows_no_language_evaluating_truthfulness_beyond_english.md)

<!-- RELATED:END -->
