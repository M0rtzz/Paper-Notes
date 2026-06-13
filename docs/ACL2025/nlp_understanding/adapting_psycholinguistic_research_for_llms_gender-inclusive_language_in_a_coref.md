---
title: >-
  [论文解读] Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context
description: >-
  [ACL 2025][NLP理解][性别包容性语言] 将 Tibblin et al. (2023) 的心理语言学实验从法语适配到英语和德语 LLM，通过测量共指词概率和生成内容分析发现：英语 LLM 基本保持先行词-共指词性别一致但 they 单数几乎不被使用且存在底层男性偏见；德语 Leo Mistral 7B 的男性偏见更强烈（压倒所有 8 种包容策略），但包容策略仍能增加女性/中性性别的出现概率，与心理语言学人类实验结果一致。
tags:
  - "ACL 2025"
  - "NLP理解"
  - "性别包容性语言"
  - "共指消解"
  - "LLM偏见"
  - "心理语言学"
  - "德语"
---

# Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context

**会议**: ACL 2025  
**arXiv**: [2502.13120](https://arxiv.org/abs/2502.13120)  
**代码**: [https://github.com/marionbartl/GIL-coref-context](https://github.com/marionbartl/GIL-coref-context)  
**领域**: NLP理解  
**关键词**: 性别包容性语言, 共指消解, LLM偏见, 心理语言学, 德语

## 一句话总结

将 Tibblin et al. (2023) 的心理语言学实验从法语适配到英语和德语 LLM，通过测量共指词概率和生成内容分析发现：英语 LLM 基本保持先行词-共指词性别一致但 they 单数几乎不被使用且存在底层男性偏见；德语 Leo Mistral 7B 的男性偏见更强烈（压倒所有 8 种包容策略），但包容策略仍能增加女性/中性性别的出现概率，与心理语言学人类实验结果一致。

## 研究背景与动机

**领域现状**：性别包容性语言（如 chairman→chairperson、he/she→they）旨在消除"男性默认"偏见。心理语言学已证明性别包容表达可增加女性和非二元性别的心理表征。LLM 被广泛用作写作助手和内容生成器，其生成的语言可能影响用户的语言习惯。

**现有痛点**：LLM 性别偏见研究已有不少，但对性别包容性语言的处理研究极少。英语 LLM 中 they 单数的处理仅开始被探索，德语 LLM 中的性别包容语言处理完全未被研究过。而德语因语法性别系统（名词/冠词/形容词均有性别标记），其包容策略更复杂——包括大写 I、性别星号 (*)、冒号 (:)、下划线 (_) 和并列式等多种形式。

**核心矛盾**：性别包容性语言旨在促进性别中性解读，但 LLM 是否真的中性解读这些表达？如果 LLM 生成的语言仍然偏向男性，其作为写作工具反而可能强化男性默认偏见。

**本文目标** 量化英语和德语 LLM 在遇到性别包容性先行词时，生成的共指词是否保持性别一致还是反映模型偏见。

**切入角度**：将 Tibblin et al. (2023) 的法语心理语言学实验范式（句对中先行词与共指词的性别关系）转化为 LLM 评估方法，同时比较 8 种德语包容策略的效果。

**核心 idea**：将心理语言学的共指消解实验设计转化为 LLM 性别偏见探针，首次系统评估德语 8 种性别包容策略在 LLM 中的效果。

## 方法详解

### 整体框架

两类实验：(1) 测量特定共指词的条件概率——给定先行词后特定性别共指词的概率分布，(2) 分析 LLM 自由生成的续写中出现的性别——观察模型实际生成行为。英语（6个模型）和德语（1个模型）分别评估。

### 关键设计

1. **数据集构建**:

    - 功能：基于 Tibblin et al. (2023) 构建句对模板，句1含先行词（性别化/中性），句2含共指词（men/women/people 或 he/she/they）
    - 核心思路——英语：
        - 复数条件(PL)：34 个先行词三元组（如 swordsman/swordswoman/fencer）× 3 个共指词(men/women/people) × 44 模板 = 13,464 实例
        - 单数条件(SG)：37 三元组 × 3 代词(he/she/they) × 44 模板 = 14,652 实例
    - 核心思路——德语：10 个先行词 × 8 种性别策略（阳性/阴性/并列阳先/并列阴先/大写I/星号/冒号/下划线） × 3 共指词(Männer/Frauen/Personen) × 44 模板 = 10,560 实例
    - 设计动机：保持与心理语言学原始设计的可比性，同时利用 LLM 特点（可精确测量概率）扩展分析维度

2. **共指词概率测量**:

    - 功能：测量 LLM 在给定先行词后对特定共指词的预测概率
    - 核心思路：用 LLM 预测句对到共指词位置的联合概率，提取共指词的对数概率 $\log(p)$。对多 token 共指词取首 token 概率（避免后续 token 的自预测膨胀）
    - 统计分析：双因素 ANOVA 检验先行词性别 × 共指词性别的交互效应，Tukey post-hoc 检验具体对比

3. **共指词生成分析**:

    - 功能：让 LLM 自由生成续写，标注生成内容中出现的性别
    - 核心思路：英语生成 8 个 token、德语生成 10 个 token 的续写，由人工标注生成内容中提及的性别和是否为共指词
    - 英语标注：3 名标注者，Fleiss' $\kappa = 0.757$（性别）/ $0.671$（共指），均在"substantial agreement"范围
    - 德语：作者之一作为母语者标注（试点实验）

### 评估模型

英语：GPT-2（基线）、GPT-2 性别中性微调版、OLMo-1B/7B/13B、Qwen-2.5-32B
德语：Leo Mistral 7B

## 实验关键数据

### 主实验（英语 Qwen-2.5 共指词概率, ANOVA）

| 效应 | PL 条件 | SG 条件 |
|------|---------|---------|
| 先行词性别主效应 | $F(2,13455)=138.59, p<.001, \eta^2=0.02$ | 显著 |
| 共指词性别主效应 | $F(2,13455)=178.33, p<.001, \eta^2=0.03$ | 显著 |
| **交互效应** | $F(4,13455)=809.94, p<.001, \eta^2=0.19$ (大) | 显著 |

### 消融实验（德语 Leo Mistral 7B, 8种包容策略对比）

| 策略 | 阴性共指词概率排名 | 中性共指词概率排名 | 说明 |
|------|-------------------|-------------------|------|
| 阳性通称（基线） | 最低 | 最低 | 男性偏见最强 |
| 阴性形式 | **最高** | 中等 | 符合预期 |
| 星号策略 (*) | 第二高 | 中等 | 含 -innen 后缀 |
| 并列式（两种） | 中等 | 中等 | 效果适中 |
| 大写I/冒号/下划线 | 中等 | 中等 | 效果弱于星号 |

### 关键发现

- **英语底层男性偏见**：中性先行词后男性共指词概率是女性的 3 倍（Tukey: $e^{1.107} \approx 3.03, p<.001$）
- **they 单数几乎不被接受**：中性先行词单数条件下，he 的概率比 they 高 88%（$e^{-2.16} \approx 0.12$），说明模型不把 they 当单数代词
- **德语男性偏见压倒一切**：无论使用何种包容策略，Männer（男性）共指词概率始终最高。与英语不同，德语中共指词性别（而非交互效应）是最大预测因子（$\eta^2 = 0.33$）
- **包容策略部分有效**：尽管无法消除男性偏见，所有包容策略都增加了阴性和中性共指词概率，与 Tibblin et al. (2023) 的人类心理语言学发现一致
- **生成实验印证概率实验**：英语 LLM 生成共指词时基本保持先行词性别一致，但中性先行词后生成共指词的频率最低

## 亮点与洞察

- 首次系统研究德语 LLM 中性别包容语言的处理——8 种策略的比较为德语 NLP 社区提供了数据支撑
- 心理语言学→NLP 的方法论转化清晰有效——概率测量和生成分析互补验证，为偏见研究提供新的实验范式
- "they 单数使用率极低"的发现对英语 NLP 系统设计有直接影响——写作助手如果不主动使用 they 单数，可能擦除非二元性别人群的语言表征
- 包容策略"部分有效"的发现既有理论意义（LLM 中偏见可被部分缓解）也有实践价值（推荐在 LLM 输入中使用包容性表达）

## 局限与展望

- 模型规模有限（最大 32B），最新的 >100B 模型表现可能不同
- 德语仅测试 Leo Mistral 7B 一个模型，需更多模型验证
- 模板化句对可能不反映自然文本中的共指模式
- 仅考虑两句间共指消解——实际 LLM 交互通常涉及更长上下文
- 共指词种类固定（men/women/people），未测试更多候选共指词

## 相关工作与启发

- **vs Watson et al. (2023)**: 在 BERT 上测试角色名词可接受性判断；本文在生成式 LLM 上测试共指词概率和实际生成
- **vs Brandl et al. (2022)**: 测试瑞典语新代词的处理难度；本文系统比较德语 8 种包容策略
- **vs Winograd Schema 偏见评估**: 单句内代词消解评估；本文跨句共指关系评估

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将心理语言学共指范式完整适配到LLM评估，德语包容策略研究填补空白
- 实验充分度: ⭐⭐⭐⭐ 概率+生成双验证，ANOVA统计严谨，但德语仅单模型试点
- 写作质量: ⭐⭐⭐⭐ 实验设计描述清晰，统计报告规范（效应量+置信区间）
- 价值: ⭐⭐⭐⭐ 对LLM公平性和多语言偏见研究有方法论贡献，实践意义在于推动包容性语言使用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] BookCoref: Coreference Resolution at Book Scale](bookcoref_book_scale.md)
- [\[ACL 2025\] On Synthesizing Data for Context Attribution in Question Answering](on_synthesizing_data_for_context_attribution_in_question_answering.md)
- [\[ACL 2025\] End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems](end-to-end_dialog_neural_coreference_resolution_balancing_efficiency_and_accurac.md)
- [\[ACL 2025\] BQA: Body Language Question Answering Dataset for Video Large Language Models](bqa_body_language_question_answering_dataset_for_video_large_language_models.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)

</div>

<!-- RELATED:END -->
