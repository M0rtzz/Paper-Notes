---
title: >-
  [论文解读] KnowCoder-X: Boosting Multilingual Information Extraction via Code
description: >-
  [ACL 2025][跨语言IE] 提出 KnowCoder-X，通过统一的 Python 类表示多语言 IE schema，并引入 IE 跨语言对齐指令微调阶段（含高质量 ParallelNER 数据集），在 64 个 IE 基准上大幅提升跨语言信息抽取性能。
tags:
  - ACL 2025
  - 跨语言IE
  - 代码生成
  - NER
  - schema统一
  - 平行数据
---

# KnowCoder-X: Boosting Multilingual Information Extraction via Code

**会议**: ACL 2025  
**arXiv**: [2411.04794](https://arxiv.org/abs/2411.04794)  
**代码**: [ICT-GoKnow/KnowCoder](https://github.com/ICT-GoKnow/KnowCoder) (有)  
**领域**: NLP-信息抽取  
**关键词**: 跨语言IE、代码生成、NER、schema统一、平行数据  

## 一句话总结

提出 KnowCoder-X，通过统一的 Python 类表示多语言 IE schema，并引入 IE 跨语言对齐指令微调阶段（含高质量 ParallelNER 数据集），在 64 个 IE 基准上大幅提升跨语言信息抽取性能。

## 研究背景与动机

- **领域现状**: 大语言模型在多语言语料上预训练后展现出自发的跨语言对齐能力，但在信息抽取（IE）任务中，不同语言之间的性能差距仍然巨大
- **现有痛点**: 实验表明，即使在英文 NER 数据上训练后，中文并行数据集上的 F1 仅为 52.7（英文 95.1），说明 LLM 中 IE 的跨语言对齐仍然较弱；现有跨语言 IE 方法缺乏统一的 schema 表示
- **核心矛盾**: IE 任务的 schema（实体类型、关系类型等）在不同语言中名称不同但语义相同（如韩语"사람"和中文"人物"都对应 PER），缺乏统一表示阻碍了跨语言知识迁移
- **本文目标**: 增强 LLM 在 IE 任务中的跨语言对齐能力，使得在中英文训练的模型能迁移到 29 种未见过的语言
- **切入角度**: 利用代码（Python 类）统一多语言 schema 表示，并通过翻译实例预测任务进行跨语言对齐训练
- **核心 idea**: 用 Python 类统一多语言 IE schema + 跨语言对齐指令微调 = 强大的零样本跨语言 IE 迁移

## 方法详解

### 整体框架

KnowCoder-X 采用两阶段指令微调：(1) IE 跨语言对齐阶段——在翻译实例预测任务上训练以增强跨语言迁移；(2) 中英文 UIE 指令微调阶段——在 46 个 IE 数据集上训练获得最终模型。

### 关键设计

#### 1. 基于 Python 类的多语言 Schema 统一表示
- **功能**: 将所有语言的 IE schema 映射为统一的 Python 类定义
- **核心思路**: 定义 Entity、Relation、Event 三个基类，每个具体概念继承对应基类。非英语 schema 首先映射到英语对应类，如韩语"사람"和中文"人物"都映射为 `PER(Entity)` 类。类的注释包含实例样本和概念描述
- **设计动机**: 面向对象的代码特性天然适合统一 schema 表示和跨语言知识共享；一致的 schema 让模型可以高效地在不同语言间共享同一本体的知识

#### 2. IE 跨语言对齐指令微调
- **功能**: 设计翻译实例预测任务，将源语言的IE输入输出与目标语言的IE输入拼接为指令，目标语言的IE输出作为 completion
- **核心思路**: 给定源语言句子 $s^{src}$ 和已标注 span $I^{src}$，拼接目标语言句子 $s^{tgt}$，预测 $I^{tgt}$
- **设计动机**: 不同于直接预测完整的 IE 平行数据（侧重句子对齐），本方法优先对齐翻译后的实例，这才是 IE 跨语言对齐的核心目标

#### 3. 三阶段 IE 平行数据构建管道
- **功能**: 高质量地自动构建 IE 平行语料
- **核心思路**:
    - **Stage 1 - 联合翻译**: 同时翻译句子和 span（而非先翻句子再对齐 span），减少误差累积
    - **Stage 2 - Span 改写**: 对翻译后不在目标句子中的 span 进行改写（类似检索式纠正）
    - **Stage 3 - 句子改写**: 对仍有缺失 span 的情况，改写目标句子以包含所有 span（span-then-sentence 策略）
- **设计动机**: 传统 label projection 方法（先翻句子再找 span）存在误差传播问题；三阶段融合了并行处理、句子-后-span、span-后-句子三种翻译策略
- **效果**: 在 WikiANN 上 10 种语言平均达到 **99% faithfulness**

#### 4. ParallelNER 数据集
- **功能**: 构建高质量中英 NER 平行数据集，共 257,190 样本
- **核心思路**: 使用 WikiNeural（en→zh）和 CLUENER2020（zh→en）作为源数据，GPT-4o-mini 为管道基础 LLM，失败时切换 GPT-4o 重处理，剩余 97 个难例人工标注
- **缺失率**: WikiNeural 仅 82/92720（8.84‱），CLUENER2020 仅 15/10000

### 损失函数

使用标准的语言建模交叉熵损失进行 LoRA 微调（rank=32），基座模型为 Baichuan2-7B-Base。

## 实验关键数据

### 主实验：Multiconer22 跨语言零样本评估（9 种未见语言）

| 模型 | 英文 | 中文 | Avg_cross (9种未见语言) | Avg (全部) |
|------|------|------|----------------------|-----------|
| ChatGPT | 37.20 | 18.80 | 30.37 | 29.94 |
| B2NER | 54.80 | 45.40 | - | - |
| IEPILE | 53.19 | 39.26 | 28.48 | 31.71 |
| **KnowCoder-X** | **56.37** | **47.53** | **39.53** | **41.79** |
| Supervised | 62.70 | 53.10 | 54.22 | 54.89 |

- 超越 ChatGPT **+30.17%**，超越 SoTA **+20.03%**
- 在西班牙语和土耳其语上零样本性能接近监督方法

### 消融实验：有监督 NER 评估

| 数据集 | YAYI-UIE | IEPILE | B2NER | KnowCoder-X |
|--------|---------|--------|-------|-------------|
| CoNLL 2003 | 96.77 | 92.49 | 92.56 | **94.69** |
| MultiNERD | 88.42 | 94.60 | 93.98 | **95.94** |
| 中文 MSRA | 95.97 | 87.99 | 92.22 | **96.01** |
| 中文 Avg | - | 90.96 | 94.06 | **96.03** |

- 中文 IE 全面达到 SoTA，EAE 任务平均提升 +4.12

### 关键发现

1. **跨语言对齐的涌现效果**: 仅在中英文数据上训练，就能迁移到 29 种未见语言，且在 20 种低资源非洲语言上平均提升 +11.43%
2. **NER 对齐惠及全部 IE 任务**: 跨语言 NER 能力的提升通过基础的 spotting 能力带动了 RE、ED、EAE 的全面提升
3. **代码表示的优势**: 统一的 Python 类确保了不同语言的 schema 语义一致，是跨语言迁移的关键

## 亮点与洞察

- **代码统一 schema 表示**的设计简洁而有效——利用面向对象的继承和类型系统天然适配 IE 本体结构
- **三阶段平行数据管道**达到近 100% 的标注忠实度，解决了传统 label projection 的误差传播问题
- 首次系统验证了**代码基础的 IE 方法**在跨语言场景下的有效性
- ParallelNER 是有价值的社区资源，可用于后续跨语言 IE 研究

## 局限与展望

- 仅构建了中英平行数据，可以扩展到更多语言对
- 基座模型 Baichuan2-7B 较小，更大模型可能获得更好效果
- RE 和 EE 任务的平行数据因 schema 概念过多而受上下文长度限制，未用于对齐训练
- 三阶段管道依赖 GPT-4o 系列的能力，对于 LLM 质量较差的低资源语言可能效果受限
- 未探索无监督的跨语言对齐方法

## 相关工作与启发

- **KnowCoder**: 用代码表示 IE 的先驱工作，KnowCoder-X 是其多语言扩展
- **GoLLIE**: 将标注指南作为类注释嵌入 schema，类似思路
- **IEPILE / B2NER**: 主要的 UIE 基线，但忽略了 schema 在不同语言间的统一表示
- **CLaP**: 传统 label projection 的代表，KnowCoder-X 的三阶段管道显著优于此方法
- **启发**: 利用代码的结构化特性来桥接不同语言的语义鸿沟是一个有前景的方向

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐: 代码统一 schema + 跨语言对齐微调的组合思路新颖实用
- **实验** ⭐⭐⭐⭐⭐: 64 个基准的全面评估令人信服，跨语言零样本结果突出
- **资源贡献** ⭐⭐⭐⭐: ParallelNER 数据集和三阶段管道有独立价值
- **写作** ⭐⭐⭐: 论文偏长，部分公式表述较冗余

<!-- RELATED:START -->

## 相关论文

- [Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction](translation_and_fusion_improves_cross-lingual_information_extraction.md)
- [CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation](m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)
- [Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books](low_resource_translation.md)
- [A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)

<!-- RELATED:END -->
