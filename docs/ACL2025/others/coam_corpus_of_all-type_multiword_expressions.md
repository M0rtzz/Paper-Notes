---
title: >-
  [论文解读] CoAM: Corpus of All-Type Multiword Expressions
description: >-
  [ACL 2025][多词表达] 构建了一个高质量的全类型多词表达(MWE)识别数据集 CoAM（1.3K句），通过多步质量保障流程解决了现有数据集标注不一致的问题，并发现微调大语言模型在 MWE 识别任务上显著优于此前的 SOTA 方法 MWEasWSD。 多词表达（Multiword Expressions…
tags:
  - "ACL 2025"
  - "多词表达"
  - "MWE识别"
  - "数据集构建"
  - "序列标注"
  - "LLM微调"
---

# CoAM: Corpus of All-Type Multiword Expressions

**会议**: ACL 2025  
**arXiv**: [2412.18151](https://arxiv.org/abs/2412.18151)  
**代码**: [https://github.com/Yusuke196/CAIGen](https://github.com/Yusuke196/CAIGen)  
**领域**: 其他  
**关键词**: 多词表达, MWE识别, 数据集构建, 序列标注, LLM微调

## 一句话总结

构建了一个高质量的全类型多词表达(MWE)识别数据集 CoAM（1.3K句），通过多步质量保障流程解决了现有数据集标注不一致的问题，并发现微调大语言模型在 MWE 识别任务上显著优于此前的 SOTA 方法 MWEasWSD。

## 研究背景与动机

多词表达（Multiword Expressions, MWEs）是指语义不能从组成词直接推导出的惯用多词序列，例如 "under the weather"（身体不适）、"real estate"（房地产）等。MWE 识别（MWEI）对机器翻译、阅读辅助等下游任务至关重要——Briakou et al. (2024) 证明在翻译的初始步骤中识别 MWE 可以提升翻译质量。

然而，现有 MWEI 数据集存在三大问题：

**标注不一致**：DiMSUM 数据集中超过 80% 的假阳性实际上是由标注不一致引起的（同一个 MWE 在一处标注、在另一处遗漏）

**类型单一**：PARSEME 语料库质量较高但仅关注动词类 MWE

**规模受限**：ID10M 每种语言仅 200 句评估数据，且未标注不连续 MWE

这些限制使得 MWEI 系统无法被可靠地评估。CoAM 的动机正是构建一个标注一致、覆盖全类型、质量有保障的数据集。

## 方法详解

### 整体框架

CoAM 的构建采用四阶段流水线：数据选择与预处理 → 双人标注 → 人工审查 → 自动一致性检查。同时开发了全新的标注界面生成器 CAIGen，支持任意形式（包括不连续、重叠）的 MWE 标注。

### 关键设计

1. **数据来源多样化**：选取四类文本源——新闻（EMM NewsBrief）、新闻评论（WMT23）、TED 演讲转录、Universal Dependencies 网络文本。覆盖书面文本和口语转录，确保标准英语且语法错误较少。

2. **严格的 MWE 定义**：MWE 需满足三个条件——(a) 至少两个词且总是由相同词素实现；(b) 展现语义、词汇或句法的惯用性（排除透明搭配如 "stuck at"）；(c) 不是专有名词。这一定义比 DiMSUM（包含多词专有名词）更精确。

3. **CAIGen 标注界面**：基于 Google Sheets 的复选框标注界面。与 brat 和 FLAT 相比，CAIGen 支持不连续/重叠 MWE 标注、无需管理服务器、界面直观且高度可定制。

4. **双人标注 + 审查机制**：每个句子分配两名标注者（一名雇佣标注者 + 一名作者），确保至少一人为英语母语者。标注完成后由两位母语作者审查所有标注，解决分歧。

5. **自动一致性检查**：利用基于规则的流水线查找可能被遗漏的 MWE，再由人工确认。在 CoAM 中发现 147 处不一致并全部修正；对照实验发现 DiMSUM 同等规模数据中有 118 处不一致未被修正。

6. **MWE 类型标注**：将 MWE 分为 5 类——Noun、Verb、Mod/Conn（修饰/连接）、Clause、Other，基于依存结构自动标注后人工修正测试集（错误率 7.8%）。

### MWEI 方法

评估了两种方法：

- **MWEasWSD (MaW)**：使用 WordNet 词典 + 规则流水线识别候选 MWE，再用双编码器过滤。基于 BERT-base-uncased。
- **LLM 微调**：对 Llama-3.1 和 Qwen-2.5 系列进行 QLoRA 微调，采用 tsv_to_tsv 的输入输出格式。

## 实验关键数据

### 主实验

| 方法 | F1 | Precision | Recall | 说明 |
|------|-----|-----------|--------|------|
| MaW Rule-only | 32.4 | 27.9 | 38.6 | 纯规则基线 |
| MaW Rule+DCA | 41.9 | 49.0 | 36.7 | 此前 SOTA |
| FT Llama-8B | 24.9 | 92.0 | 14.4 | 精度高但召回极低 |
| FT Qwen-7B | 48.1 | 60.9 | 39.7 | 中等规模有提升 |
| **FT Qwen-72B** | **57.8** | **63.8** | **52.8** | 全面最优 |

### 消融实验

| 配置 | F1 (Qwen-72B) | 说明 |
|------|---------------|------|
| Fine-tuning | 57.8 | 主实验结果 |
| Few-shot (5-shot) | 12.9 | 大幅下降 |
| Zero-shot | 1.4 | 几乎无法完成任务 |

### 分类型召回率分析

| MWE 类型 | Qwen-72B 召回 | MaW Rule+DCA 召回 |
|----------|---------------|-------------------|
| Verb (139个) | 59.7% | 45.3% |
| Noun (121个) | 42.1% | 29.8% |
| Mod/Conn (111个) | 57.7% | 36.6% |
| Clause (7个) | 28.6% | 0.0% |

### 关键发现

- **微调 LLM 显著优于规则方法**：Qwen-72B 的 F1 比 MaW 最佳系统高 15.9 个百分点
- **所有方法召回率都低**：即使最好的 Qwen-72B 也只有 52.8%，接近一半的 MWE 被遗漏
- **不连续 MWE 是关键挑战**：连续 MWE 召回 57.3%，不连续 MWE 仅 17.1%
- **不在 WordNet 中的 MWE 特别难识别**：在 WordNet 中的 MWE 召回 62.7%，不在的仅 45.5%
- **人类标注者的平均召回率也仅约 51%**，与 Qwen-72B 相当，说明 MWEI 本身是一个极具挑战性的任务

## 亮点与洞察

- CoAM 是首个带有 MWE 类型标签的全类型 MWE 数据集，支持细粒度错误分析
- 一致性检查方法有效且通用，可应用于其他标注任务的质量保障
- 人类标注者的表现分析揭示了一个重要事实：顶级标注者（语言学背景）F1 达 72%，而平均水平仅 56.5%，说明 MWE 识别高度依赖专业训练
- 首次将 LLM 微调应用于 MWEI 并取得 SOTA，证明 LLM 在预训练过程中确实习得了 MWE 知识

## 局限与展望

- 数据集规模较小（1.3K句），虽然评估足够但限制了训练规模
- 仅覆盖英语，多语言扩展是重要方向
- 不连续 MWE 识别仍是巨大挑战（召回仅 17%），需要专门建模
- 未探索 BERT 级别的序列标注方法（如 BIO 标注 + 微调）
- CAIGen 界面存在可用性问题——标注者可能忘记为不同 MWE 使用不同行

## 相关工作与启发

- MWEasWSD 利用 WordNet 作为外部知识的思路值得借鉴，但依赖词典覆盖率
- 一致性检查的方法论（先收集所有已标注 MWE，再在全文中搜索遗漏实例）可推广到其他 NLP 标注任务
- LLM 微调在序列标注类任务中的有效性进一步得到验证

## 评分

- **新颖性**: ⭐⭐⭐ — 核心贡献是数据集构建，技术方法并非全新，但全类型 MWE + 一致性检查的组合有新意
- **实验充分度**: ⭐⭐⭐⭐ — 多维度的分析（类型、连续性、是否在 WordNet 中、seen/unseen）非常详尽
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数据集构建过程描述详细
- **价值**: ⭐⭐⭐⭐ — 为 MWE 识别社区提供了可靠的评估基准，一致性检查方法有推广价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LoGU: Long-form Generation with Uncertainty Expressions](logu_longform_gen_uncertainty.md)
- [\[ACL 2025\] Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus](mapping_the_podcast_ecosystem_with_the_structured_podcast_research_corpus.md)
- [\[ACL 2025\] Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)
- [\[ACL 2025\] All That Glitters is Not Novel: Plagiarism in AI Generated Research](plagiarism_ai_generated_research.md)
- [\[ACL 2025\] One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit](one_for_all_update_parameterized_knowledge_across_multiple_models_with_once_edit.md)

</div>

<!-- RELATED:END -->
