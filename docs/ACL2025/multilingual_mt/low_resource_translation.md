---
title: >-
  [论文解读] Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books
description: >-
  [ACL 2025][extremely low-resource translation] 将语法书辅助的极低资源翻译分解为**语法规则检索**和**规则应用**两步，提出 Rule-by-Rule 检索策略和代码格式语法规则表示，在壮语翻译上端到端提升 13.1% BLEU。
tags:
  - ACL 2025
  - extremely low-resource translation
  - grammar book
  - code representation
  - Zhuang language
  - rule retrieval
---

# Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books

**会议**: ACL 2025  
**arXiv**: [2506.01796](https://arxiv.org/abs/2506.01796)  
**代码**: [GitHub - ZhuangRules](https://github.com/Infinite-set/ZhuangRules)  
**机构**: Peking University (Wangxuan Institute of Computer Technology)
**领域**: Multilingual MT / Extremely Low-Resource Translation  
**关键词**: extremely low-resource translation, grammar book, code representation, Zhuang language, rule retrieval

## 一句话总结

将语法书辅助的极低资源翻译分解为**语法规则检索**和**规则应用**两步，提出 Rule-by-Rule 检索策略和代码格式语法规则表示，在壮语翻译上端到端提升 13.1% BLEU。

## 研究背景与动机

**领域现状**：全球大多数语言属于极低资源（XLR）语言，可用平行语料仅数千句，传统预训练/微调方法完全不可行。LLM通过上下文学习（ICL）利用词典、平行句对等小规模语言学资源进行XLR MT展现了潜力。语法书作为最系统的语言学描述，理论上最适合指导翻译。

**现有痛点**：Aycock et al. (2024) 指出LLM可能仅从语法书中提取双语词汇解释作为"捷径"（lexical leakage），而非真正理解语法规则。现有研究对语法书是否真正有效莫衷一是——这源于缺乏能排除词汇知识干扰、仅聚焦语法理解能力的受控评估手段。

**核心矛盾**：语法书包含大量细粒度规则，一次性提供整本书给LLM时，模型难以定位相关规则（recall仅约50%），且规则表述为自然语言文本时LLM理解和执行的准确度有限。如何让LLM高效检索并准确应用语法规则是关键瓶颈。

**本文目标** (1) 构建ZhuangRules受控数据集回答"LLM是否真正理解语法规则"；(2) 将语法书翻译解耦为检索+应用两步，精确定位瓶颈并针对性解决。

**切入角度**：利用语法操作与代码结构的天然相似性（加词缀→算术运算、条件选择→if-else），将语法规则转换为Python伪代码格式，同时提升检索和应用两步的性能。

**核心 idea**：通过Rule-by-Rule检索策略和代码格式语法规则表示，将语法书辅助的XLR翻译端到端提升13.1% BLEU。

## 方法详解

### 整体框架

将语法书辅助的极低资源翻译分解为两个独立阶段：

1. **语法规则检索（Rule Retrieval）**：给定待翻译句子，从语法书中定位所需的语法规则
2. **语法规则应用（Rule Application）**：根据检索到的规则完成翻译

核心创新在于：(a) 提出 Rule-by-Rule 检索策略将长上下文理解简化为二分类；(b) 用代码格式表示语法规则，同时提升检索和应用两步的性能。

### 关键设计

1. **ZhuangRules受控数据集**:

    - 功能：提供排除词汇知识干扰、仅聚焦语法理解能力的受控评测基准
    - 核心思路：构建面向壮语的模块化语法规则数据集：109条原子语法规则，每条平均配5.6个壮中平行句对（共608对）。关键设计是为每个测试句对提供覆盖所有相关词汇的壮中词典，将语法理解与词汇知识彻底解耦。规则按三个维度标注：动作类型（加词缀、词序交换等）、难度（Easy/Medium/Hard，平均操作数1.2/1.5/2.1）、语言学领域（形态学、词序等，遵循WALS分类）
    - 设计动机：此前无法判断LLM是在认真运用语法规则还是利用词汇捷径。严格控制词汇变量后，才能真正回答"LLM能否理解语法规则"

2. **Rule-by-Rule检索策略**:

    - 功能：将长上下文语法书理解问题转化为高效的二分类问题，大幅提升规则检索recall
    - 核心思路：Pilot study发现Full-Book方式（一次性提供整本语法书）recall仅约50%，且无关规则数量增加时翻译性能急剧下降。Rule-by-Rule策略逐条判断每条规则是否与待翻译句子相关（二分类），用更多API调用（109次vs 1次）换取显著更高的accuracy，recall提升至~89%
    - 设计动机：精确定位瓶颈在检索而非应用，用计算开销换准确率是合理的工程权衡。二分类问题远比从109条规则中选择相关子集简单

3. **代码格式语法规则表示**:

    - 功能：利用LLM对代码的强理解能力提升语法规则的检索和应用
    - 核心思路：利用语法操作与代码结构的天然相似性（加词缀→算术运算、条件选择→if-else、词序变换→序列操作），用GPT-4o以5-shot ICL将文本规则转换为Python伪代码函数。每条代码规则包含简要注释+伪代码函数体。质量检查：随机抽样10条，全部符合Python语法，仅1条遗漏次要信息
    - 设计动机：LLM在代码理解和执行上的能力远超自然语言指令遵循，代码的过程化表示对复杂多步语法操作尤其有效

### 训练策略

本文不训练模型，全部基于 ICL prompting。翻译实验中平行例句用 2-shot ICL。IGT（Interlinear Glossed Text）由 GPT-4o 生成，以 123 个手工标注 IGT 作为 ICL 示例，morpheme 正确率约 72%。

## 实验关键数据

### 主实验

在 ZhuangRules 上比较不同检索策略的 recall：

| 检索策略 | 模型 | za→zh recall | zh→za recall | 平均检索规则数 |
|---------|------|:----------:|:----------:|:----------:|
| BM25 | — | 41.6 (rec@5) | 27.3 (rec@5) | 5 |
| Full-Book | Qwen-72B | 52.8 | 49.4 | 1.8 |
| Rule-by-Rule (text) | Qwen-72B | 89.4 | 84.7 | 4.1 |
| Rule-by-Rule (code) | Qwen-72B | **89.6** | **87.1** | 3.9 |
| Rule-by-Rule (text) | Llama-70B | 69.7 | 75.8 | 2.2 |
| Rule-by-Rule (code) | Llama-70B | **82.2** | **87.5** | 4.2 |
| Rule-by-Rule (text) | Qwen-7B | 55.1 | 67.9 | 2.5 |
| Rule-by-Rule (code) | Qwen-7B | **68.4** | **80.3** | 3.8 |

代码格式在小模型上提升最为显著：Qwen-7B 提升 +13.3/+12.4 recall，Llama-70B 提升 +12.5/+11.7 recall。

### 规则应用性能

在 ZhuangRules 上比较不同规则应用设置（6 模型-方向组合的平均 BLEU/chrF++）：

| 设置 | 平均 BLEU | 平均 chrF++ |
|------|:--------:|:----------:|
| No Rule (w/o Lexicon) | 0.9 | 3.0 |
| No Rule | 25.5 | 38.0 |
| Parallel Examples | 60.2 | 67.4 |
| Gold Textual Rule | 45.7 | 60.7 |
| Gold Textual Rule + Parallel Examples | 70.2 | 75.4 |
| Gold Code Rule | 57.9 | 69.2 |
| **Gold Code Rule + Parallel Examples** | **72.4** | **77.9** |

代码规则 vs 文本规则：+12.2 BLEU（45.7→57.9）；结合平行例句后达到最优 72.4 BLEU。

### 消融实验

| 规则格式 | Easy (za→zh) | Medium | Hard | Easy (zh→za) | Medium | Hard |
|---------|:----------:|:------:|:----:|:----------:|:------:|:----:|
| Text Rule | 65.6 | 51.3 | 34.6 | 85.5 | 82.4 | 69.3 |
| Code Rule | 76.3 | 57.9 | 48.6 | 93.0 | 87.5 | 76.8 |
| **Δ** | +10.7 | +6.6 | **+14.0** | +7.5 | +5.2 | +7.5 |

代码格式对 hard 规则提升最大（za→zh: +14.0 BLEU），说明代码的过程化表示对复杂多步操作尤其有效。

### 跨语言泛化（MTOB, Kalamang, Qwen-72B）

| 规则格式 | kgv→eng BLEU | eng→kgv BLEU |
|---------|:----------:|:----------:|
| Gold Textual Rule | 14.6 | 43.8 |
| Gold Code Rule | **16.0** | **44.5** |

在另一种 XLR 语言 Kalamang 上同样有效，验证了代码格式的跨语言泛化能力。

### 关键发现

- 代码格式对hard규则提升最大（za→zh: +14.0 BLEU），说明代码的过程化表示对复杂多步操作尤其有效
- Rule-by-Rule检索策略将recall从Full-Book的~50%提升至~89%，代价是109次API调用 vs 1次
- 代码格式在小模型上提升最为显著：Qwen-7B提升+13.3/+12.4 recall
- 在Kalamang语上的跨语言泛化验证了方法的普适性
- 端到端最佳组合（Code Rule + Rule-by-Rule）比Full-Book + Textual Rule提升13.1% BLEU

## 亮点与洞察

- **问题分解思想精妙**：将端到端语法书翻译拆为检索+应用两步，精确定位瓶颈（检索），并针对性提出Rule-by-Rule策略。这个分解思路可迁移到其他需要从大量知识中检索并应用特定规则的NLP任务
- **代码增强推理的语言学首次应用**：利用LLM对代码的强理解能力来提升语法规则的处理效果，是代码增强推理在语言学领域的创新应用
- **受控评估设计考究**：ZhuangRules通过提供词典排除词汇干扰，通过原子规则+难度标注支持受控分析，为XLR翻译评估树立了方法论标杆

## 局限与展望

- 仅在壮语和Kalamang两种语言上实验，对其他语系的泛化性有待验证
- Rule-by-Rule策略需对每条规则单独查询LLM（109次vs 1次），计算开销显著
- 代码规则转换依赖GPT-4o，对更复杂/不规则语法的适用性未知
- IGT生成质量（72% morpheme正确率）仍有提升空间

## 相关工作与启发

- **vs Aycock et al. (2024)**：他们指出LLM从语法书中提取词汇捷径（lexical leakage），本文通过ZhuangRules数据集的词典控制设计直接回应了这一质疑
- **vs MTOB (Tanzer et al., 2024)**：MTOB提供语法书+词典的翻译基准，但未对语法规则做模块化分解。本文的Rule-by-Rule策略在其数据上也验证了有效性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 两步分解+代码格式语法规则的组合非常新颖且有效
- 实验充分度: ⭐⭐⭐⭐ 3个模型、2个数据集、多维度消融实验，覆盖检索和应用两阶段
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验设计受控，分析细致
- 价值: ⭐⭐⭐⭐ 为极低资源翻译提供了可行的新范式

<!-- RELATED:START -->

## 相关论文

- [Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages](multilingual_encoder_knows_more_than_you_realize_shared_weights_pretraining_for_.md)
- [The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages](the_esethu_framework_reimagining_sustainable_dataset_governance_and_curation_for.md)
- [Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)
- [GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning](grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)
- [KnowCoder-X: Boosting Multilingual Information Extraction via Code](knowcoder-x_boosting_multilingual_information_extraction_via_code.md)

<!-- RELATED:END -->
