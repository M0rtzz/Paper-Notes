---
title: >-
  [论文解读] A Multilingual Dataset and Empirical Validation for the Mutual Reinforcement Effect in Information Extraction
description: >-
  [ACL 2026][多语言/翻译][互增强效应] 构建首个多语言MRE Mix数据集（MMM，21个子集覆盖英中日），并通过大规模消融实验系统验证了词级与文本级信息抽取任务的互增强效应（MRE）跨语言普遍存在。
tags:
  - "ACL 2026"
  - "多语言/翻译"
  - "互增强效应"
  - "多语言信息抽取"
  - "词级-文本级联合建模"
  - "数据集构建"
  - "LLM辅助翻译"
---

# A Multilingual Dataset and Empirical Validation for the Mutual Reinforcement Effect in Information Extraction

**会议**: ACL 2026  
**arXiv**: [2407.10953](https://arxiv.org/abs/2407.10953)  
**代码**: [GitHub/HuggingFace](https://ganchengguang.github.io/MRE/)  
**领域**: Information Extraction / Multilingual NLP  
**关键词**: 互增强效应, 多语言信息抽取, 词级-文本级联合建模, 数据集构建, LLM辅助翻译

## 一句话总结

构建首个多语言MRE Mix数据集（MMM，21个子集覆盖英中日），并通过大规模消融实验系统验证了词级与文本级信息抽取任务的互增强效应（MRE）跨语言普遍存在。

## 研究背景与动机

**领域现状**：信息抽取（IE）包含命名实体识别、关系抽取、情感分析等多个子任务，传统做法将其独立建模。多任务学习虽共享表示，但并未显式建模任务间的语义交互。

**现有痛点**：互增强效应（MRE）——词级和文本级IE任务在联合建模时能相互提升——此前仅在日语上有验证，缺乏多语言MRE数据集严重阻碍了跨语言验证和更广泛的应用。

**核心矛盾**：MRE是语言特定的现象还是跨语言的通用机制？这个根本问题因数据缺失而无法回答。

**本文目标**：构建多语言MRE数据集并系统验证MRE在不同语言和任务组合中的普遍性。

**切入角度**：提出LLM辅助的数据集翻译对齐框架，将日语MRE数据集扩展到英语和中文，同时构建新的开放域数据集。

**核心 idea**：MRE不是语言特定artifact，而是IE任务中词级细粒度语义与文本级全局语义之间双向依赖的通用机制。

## 方法详解

### 整体框架

工作包含三部分：(1) 构建MMM数据集——通过LLM辅助翻译框架将日语MRE数据集扩展为多语言版本；(2) 设计OIELLM——统一输入输出格式的开放域信息抽取模型；(3) 进行系统性消融实验验证MRE跨语言有效性。

### 关键设计

1. **LLM辅助数据集翻译框架**:

    - 功能：将日语MRE数据集高效翻译为英语和中文，同时保持标注一致性
    - 核心思路：先用规则匹配确定性翻译固定标签集，再用GPT-3.5-Turbo辅助翻译自由文本部分，最后通过两阶段规则过滤（去除未翻译字符、对齐实体span）和人工校验确保质量
    - 设计动机：标签集固定可用确定性映射消除歧义；LLM用于减少重复劳动而非替代人工，人工保留在质量控制环节

2. **统一输入输出的OIELLM模型**:

    - 功能：在单一解码过程中联合生成文本级标签和词级标签-实体对
    - 核心思路：输入为原文+任务指令词（用"/"前缀标记），输出按固定格式先给出文本级标签再给出词级抽取结果，使用分隔符":"和";"确保跨任务跨语言的解析一致性
    - 设计动机：避免对话式prompt的长度开销和prompt引导偏差，让模型专注学习文本级与词级信息的结构性依赖

3. **Knowledgeable Verbalizer扩展**:

    - 功能：将MRE数据中的词级监督信号注入到文本级分类器中
    - 核心思路：利用MRE Mix数据中的词级标注构建知识增强的verbalizer，增强prompt-based文本分类中标签词的表示
    - 设计动机：从另一个角度验证MRE——如果词级信息确实有助于文本级任务，那么将其显式注入应能提升分类性能

### 损失函数 / 训练策略

OIELLM基于开源LLM进行全量微调，使用标准的自回归语言模型训练目标。训练数据涵盖MMM全部21个子集。

## 实验关键数据

### 主实验

| 模型 | SCNM TL | SCNM WL | SCNM ALL |
|------|---------|---------|----------|
| GPT-4o | 58.30 | 23.42 | 8.57 |
| OIELLM-8B | 84.73 | 88.53 | 61.93 |
| OIELLM-8B* | 87.30 | 89.28 | 64.00 |
| OIELLM-13B | 89.00 | 86.33 | 57.70 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| MRE存在率 | 76% | 21个子集中16个展现显著MRE |
| 跨语言一致性 | 英/中/日均有效 | MRE非语言特定现象 |
| Verbalizer增益 | 正向 | 词级监督注入提升文本级分类 |

### 关键发现
- 76%的MMM子数据集在消融中显示出稳定的互增强效应，证明MRE是跨语言通用机制
- OIELLM在联合训练设置下全面超越零样本LLM（GPT-3.5, GPT-4o），证明MRE的实际价值
- 将词级信息注入Knowledgeable Verbalizer带来了一致的文本级分类提升

## 亮点与洞察
- "Point-Line"抽象优雅地统一了词级和文本级IE任务的关系——词级是点，文本级是线，互相约束
- LLM辅助翻译框架的设计实用：确定性映射+LLM翻译+规则过滤+人工校验，每步都有明确分工
- 实验设计周到——不仅证明MRE存在，还通过Verbalizer实验展示了其可操作的应用价值

## 局限与展望
- 目前仅覆盖英中日三种语言，低资源语言的MRE有效性未验证
- 翻译框架仍需10名多语言研究生参与人工校验，规模化成本不低
- MRE的理论解释（为何词级和文本级会互增强）仍不充分
- 未来可扩展到更多语言和更多IE任务组合

## 相关工作与启发
- **vs 传统多任务IE**: 不仅共享表示，更显式建模和验证任务间的双向增强
- **vs UIE/USM等统一IE模型**: 聚焦于MRE现象的实证验证而非模型架构创新
- **vs LLM零样本IE**: 微调后的OIELLM显著优于GPT-4o零样本，说明任务特定训练仍然重要

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个多语言MRE数据集和系统性跨语言验证
- 实验充分度: ⭐⭐⭐⭐ 21个子集的全面消融，多模型对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Point-Line抽象生动
- 价值: ⭐⭐⭐⭐ 为多语言IE提供重要数据资源和实证基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] KnowCoder-X: Boosting Multilingual Information Extraction via Code](../../ACL2025/multilingual_mt/knowcoder-x_boosting_multilingual_information_extraction_via_code.md)
- [\[ACL 2025\] Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction](../../ACL2025/multilingual_mt/translation_and_fusion_improves_cross-lingual_information_extraction.md)
- [\[ACL 2026\] NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning](neoamt_neologism-aware_agentic_machine_translation_with_reinforcement_learning.md)
- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)

</div>

<!-- RELATED:END -->
