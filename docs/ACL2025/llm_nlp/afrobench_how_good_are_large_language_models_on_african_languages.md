---
title: >-
  [论文解读] AfroBench: How Good are Large Language Models on African Languages?
description: >-
  [ACL 2025][LLM/NLP][非洲语言] 提出AfroBench——覆盖64种非洲语言、15个NLP任务、22个数据集的综合评测基准，评估12个LLM发现闭源模型(GPT-4o)领先最佳开源模型(Gemma 2 27B)约12分，但所有LLM仍落后于微调基线，与英语的差距在开源模型上超过40分。
tags:
  - ACL 2025
  - LLM/NLP
  - 非洲语言
  - 低资源语言
  - 多语言基准
  - LLM评测
  - 公平性
---

# AfroBench: How Good are Large Language Models on African Languages?

**会议**: ACL 2025  
**arXiv**: [2311.07978](https://arxiv.org/abs/2311.07978)  
**代码**: [GitHub](https://mcgill-nlp.github.io/AfroBench/)  
**领域**: 多语言NLP / LLM评测  
**关键词**: 非洲语言, 低资源语言, 多语言基准, LLM评测, 公平性

## 一句话总结

提出AfroBench——覆盖64种非洲语言、15个NLP任务、22个数据集的综合评测基准，评估12个LLM发现闭源模型(GPT-4o)领先最佳开源模型(Gemma 2 27B)约12分，但所有LLM仍落后于微调基线，与英语的差距在开源模型上超过40分。

## 研究背景与动机

**领域现状**：LLM在高资源语言上表现优异，但在非洲语言等低资源语言上的能力严重不足且评估匮乏。现有多语言评测(MEGA, Megaverse)仅覆盖11-16种非洲语言和有限任务。

**现有痛点**：(a) 非洲语言数据集分散且难以发现；(b) 评估覆盖语言少、任务单一（多集中在NER/POS）；(c) 评估成本高导致模型覆盖不全；(d) LLM不断迭代但缺乏持续追踪非洲语言进展的平台。

**核心矛盾**：全球7000+语言中90%被NLP社区忽视，非洲语言的NLP技术差距亟需量化和缩小。

**本文目标**：构建最全面的非洲语言LLM评测基准，并系统性地揭示当前LLM在非洲语言上的能力边界。

## 方法详解

### 整体框架

AfroBench聚合22个数据集，覆盖15个任务(9个NLU + 6个NLG + 6个知识/QA + 1个数学推理)，跨64种非洲语言(7个语族)。所有任务建模为文本生成问题，使用多个prompt模板评估。

### 关键设计

1. **任务全面覆盖**:
    - NLU：POS、NER、情感分析、主题分类、意图分类、仇恨言论检测、NLI
    - NLG：机器翻译(4个数据集)、摘要、变音符号还原(AfriADR, 新数据集)
    - 知识/QA：跨语言QA、阅读理解、MMLU、科学QA
    - 推理：数学推理(AfriMGSM)

2. **AfroBench-Lite**:
    - 功能：提供含14种代表性语言和7个任务的轻量版本
    - 核心思路：语言选择涵盖不同资源水平和类型学多样性(Swahili, Hausa, Amharic, Igbo, Yorùbá等)
    - 设计动机：降低评估成本，方便新模型快速上榜

3. **AfriADR新数据集**:
    - 功能：自动变音符号还原任务，覆盖5种语言(Ghomálá', Fon, Igbo, Wolof, Yorùbá)
    - 核心思路：去除句子中所有变音符号作为输入，要求模型恢复正确变音
    - 设计动机：变音符号对非洲语言的语义至关重要，且此任务LLM不熟悉

## 实验关键数据

### 主实验

12个LLM在15个任务上的平均得分：

| 模型 | 总体平均 | vs 英语差距 |
|------|---------|-----------|
| GPT-4o | 59.6 | -25.5 |
| Gemini 1.5 pro | 58.5 | -24.1 |
| Gemma 2 27B | 47.7 | -32.9 |
| LLaMa 3.1 70B | 43.3 | -36.7 |
| Aya-101 13B | 40.1 | (N/A) |
| LLaMa 2 7B | 22.5 | (N/A) |
| 微调基线(AfroXLMR等) | (任务相关) | (N/A) |

AfroBench-Lite上英语vs非洲语言表现：

| 模型 | 英语 | 非洲语言 | 差距 |
|------|------|---------|------|
| GPT-4o | 85.1 | 66.0 | -19.1 |
| Gemma 2 27B | 80.6 | 43.5 | -37.1 |
| LLaMa 3.1 70B | 80.0 | 39.9 | -40.1 |

### 消融实验

Few-shot效果(GPT-4o, 0-shot vs 5-shot)：

| 任务 | 0-shot | 5-shot | 提升 |
|------|--------|--------|------|
| ADR(变音符号还原) | 54.9 | 62.7 | +7.8 |
| 仇恨言论 | 63.5 | 69.3 | +5.8 |
| 数学推理 | 49.8 | 54.7 | +4.9 |
| 摘要 | 66.5 | 67.9 | +1.4 |

### 关键发现

1. **闭源vs开源差距比英语大得多**：在英语上差距仅2-5分，在非洲语言上差距超过12分
2. **知识密集型任务差距最大**：Arc-Easy(+29.4)、Math(+22.6)、MMLU(+19.9)
3. **性能与单语数据量正相关**：Swahili(2.4GB单语数据)最佳，Wolof(5MB)最差
4. **所有LLM仍落后于微调基线约11.5分**：说明为低资源语言收集标注数据仍然有价值
5. **Prompt敏感性**：Gemini-1.5 pro对prompt最不敏感，小模型(Gemma 2 9B)最敏感
6. **Few-shot对NLG任务和新任务(ADR)帮助最大**，对翻译帮助最小

## 亮点与洞察

- **规模空前**：64种非洲语言、15个任务、22个数据集，远超此前最大的非洲语言评测
- **AfriADR是创新贡献**：变音符号还原是非洲语言特有的重要任务，few-shot能大幅提升
- **质性分析有说服力**：展示了Ghomálá'语变音符号还原中0-shot vs 5-shot的巨大差异(ChrF从21.4到81.6)，以及数学推理中few-shot帮助模型用目标语言正确推理
- **实用价值**：建立了持续更新的leaderboard，已追加GPT-4.1, Gemini-2.0-Flash, LLaMa 4等新模型

## 局限与展望

- 训练数据透明度不足导致无法评估数据污染
- 评估成本高（GPT-4o和Gemini-1.5各约$2500）限制了模型覆盖
- 60%语言出现在少于5个数据集中，长尾分布限制了某些语言的可靠评估
- 翻译评估受限于chrF等指标，缺乏高质量的COMET/AfriCOMET评估

## 相关工作与启发

- **IrokoBench**：ACL 2025同期工作，聚焦16种非洲语言和3个任务，本文涵盖更广
- **Belebele**：覆盖28种非洲语言但仅QA任务
- 启发：非洲语言LLM能力的关键瓶颈在于单语数据量而非模型架构，投资语言资源建设比改进模型更关键

## 评分

- 新颖性: ⭐⭐⭐ 主要是资源和评测贡献，方法创新不多(AfriADR除外)
- 实验充分度: ⭐⭐⭐⭐⭐ 64种语言、12个模型、15个任务，多维分析详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分析全面，但表格较多阅读门槛高
- 价值: ⭐⭐⭐⭐⭐ 填补非洲语言LLM评测的重大空白，持续更新的leaderboard有长期价值

<!-- RELATED:START -->

## 相关论文

- [Large Language Models are Good Relational Learners](large_language_models_are_good_relational_learners.md)
- [Large Language Models for Predictive Analysis: How Far Are They?](large_language_models_for_predictive_analysis_how_far_are_they.md)
- [What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)
- [MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [TigerLLM - A Family of Bangla Large Language Models](tigerllm_-_a_family_of_bangla_large_language_models.md)

<!-- RELATED:END -->
