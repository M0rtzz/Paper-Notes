---
title: >-
  [论文解读] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora
description: >-
  [LLM/NLP] 提出利用LLM的语境理解能力来检测和量化有语法性别语言（如西班牙语、巴伦西亚语）训练语料中的性别表征偏差（representation bias），发现严重的男性主导不平衡，并验证了通过反向偏差数据进行持续预训练可有效缓解模型输出偏差。
tags:
  - LLM/NLP
---

# Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora

| 信息 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| arXiv | [2406.13677](https://arxiv.org/abs/2406.13677) |
| 代码 | [GitHub](https://github.com/ellisalicante/grb-corpora) |
| 领域 | NLP / 偏差检测 / 多语言 |
| 关键词 | 性别表征偏差, 有语法性别语言, LLM偏差检测, 语料分析, 偏差缓解 |

## 一句话总结

提出利用LLM的语境理解能力来检测和量化有语法性别语言（如西班牙语、巴伦西亚语）训练语料中的性别表征偏差（representation bias），发现严重的男性主导不平衡，并验证了通过反向偏差数据进行持续预训练可有效缓解模型输出偏差。

## 研究背景与动机

- **问题定义**：性别表征偏差（gender representation bias）指文本中对不同性别个体引用频率的不平等。这种训练数据中的上游偏差是模型偏差传播和放大的源头。
- **现有不足**：(1) 已有研究主要关注**刻板印象偏差**（stereotyping bias，将特定角色与性别关联），而非**表征偏差**（频率不平等）；(2) 现有方法（如gender polarity）专为英语设计，使用预定义的性别词列表匹配，无法处理**有语法性别的语言**——因为这些语言中所有名词都有语法性别（如西班牙语"el coche"是阳性"car"但非人类引用）
- **为何重要**：全球约38%人口使用有语法性别的语言。在这些语言中，阳性复数形式通常默认代表混合性别群体（如西班牙语"los profesores"既指男教师也泛指所有教师），这种语言惯例本身就构成隐性的性别表征偏差。
- **核心动机**：需要一种能区分"指代人类的名词/代词"与"非人类名词"的方法，并正确分类其语法性别——这超越了简单的词表匹配，需要语义理解能力。

## 方法详解

### 整体框架

三步流水线，利用LLM的语境理解能力：

1. **识别**：给定文本中所有名词和代词
2. **分类**：判断每个名词/代词是否指代人类（P）或非人类（N）
3. **性别判定**：确定每个词的语法性别——阳性（M）或阴性（F）

最终计算 $L_{P,M} : L_{P,F}$ 比率来量化性别表征偏差。

### 关键设计

1. **LLM-based方法**：通过精心设计的prompt + few-shot示例，让LLM在单次查询中完成名词/代词识别、人类引用判断、语法性别分类三个任务。逐句处理文本，充分利用LLM的上下文语义理解能力。
2. **排除形容词**：形容词的性别标记通常依赖关联名词，不独立传达人类引用信息，排除可降低复杂度。
3. **持续预训练验证偏差传播**：构建三种合成5000句数据集（male-biased/female-biased/balanced），对三个开源LLM进行QLoRA持续预训练（<20步），验证训练数据偏差如何传播到模型输出。

### 损失函数

持续预训练使用标准语言模型损失（next token prediction），配合QLoRA进行参数高效训练。核心不在于训练新模型，而在于验证偏差传播假说。

## 实验关键数据

### 主实验：西班牙语-英语语料偏差

| 数据集 | 英语 GM:GF | 西班牙语 L_{P,M}:L_{P,F} |
|--------|-----------|------------------------|
| Europarl | 1.39:1 ~ 1.46:1 | **3.94:1 ~ 3.98:1** |
| CCAligned | 1.07:1 | **4.03:1 ~ 4.54:1** |
| Global Voices | 1.43:1 | **4.39:1 ~ 4.48:1** |
| WMT-News | 3.08:1 ~ 3.44:1 | **5.22:1 ~ 6.04:1** |

### 巴伦西亚语语料偏差

| 数据集 | L_{P,M}:L_{P,F} |
|--------|-----------------|
| BOUA | 2.21:1 ~ 2.88:1 |
| DOGV+DOGCV | 2.41:1 ~ 2.72:1 |
| DSCV+DSCCV | 2.03:1 ~ 2.38:1 |

### 消融：偏差传播实验

| 训练数据 | 模型输出男女比（示例：llama3.1-8B巴伦西亚语）|
|---------|------|
| 基座模型（无持续预训练）| 3.21:1 |
| Male-biased训练 | 6.63:1 ↑ |
| Balanced训练 | 接近1:1 |
| **Female-biased训练** | **≈1:1** ✓ |

### 关键发现

1. **西班牙语偏差远大于英语**：同一平行语料，西班牙语的男性表征偏差是英语的3-4倍，与语法性别惯例有关
2. **巴伦西亚语偏差较低**：2:1~3:1，可能因为官方文档有更正式的包容性用语惯例
3. **偏差可传播**：male-biased数据训练后模型男性引用比例显著增加
4. **反向偏差有效缓解**：仅5000句female-biased数据持续预训练即可将模型输出偏差从3:1降至接近1:1
5. **方法验证**：gpt-4-turbo在西班牙语/巴伦西亚语验证集上F-score分别达90.24%/84.43%，方法稳定可靠

## 亮点与洞察

- 填补了有语法性别语言训练语料性别表征偏差量化的空白，方法设计简洁但有效
- 区分"指代人类的名词"与"所有名词"这一设计至关重要——避免了将"桌子（阴性）""车（阳性）"等非人类引用计入偏差
- 反向偏差缓解策略的发现极具实用价值：不需要大量平衡数据，仅5000句反向偏差数据即可显著纠偏
- 方法可扩展到其他有语法性别的语言（法语、德语、捷克语等），为多语言NLP的公平性研究提供工具

## 局限性

- 仅在西班牙语和巴伦西亚语上验证，未在其他语法性别语言（法语、德语等）上测试
- 方法依赖高端LLM（gpt-4-turbo），计算成本较高
- 每个子集仅采样1000句，虽统计上合理但可能错过长尾分布
- 持续预训练实验使用合成数据，与真实分布可能有差异
- 仅考虑二元性别（阳性/阴性），未覆盖非二元性别表达

## 相关工作与启发

- **Gender polarity** (Dhamala et al., 2021)：英语性别极性方法，基于预定义词表，不适用于有语法性别的语言
- **BOLD** (Dhamala et al., 2021)：社会偏差基准，侧重于stereotyping bias
- **Biesialska et al. (2024)**：报告stereotyping bias和representation bias之间的关联
- 对多语言模型公平性有启发：需要针对不同语言类型设计不同的偏差检测方法

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)

<!-- RELATED:END -->
