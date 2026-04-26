---
title: >-
  [论文解读] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs
description: >-
  [ACL 2025][Cross-Lingual Generalization] 系统评估 LLM 在三种古典语言（梵语、古希腊语、拉丁语）上的零样本跨语言泛化能力，涵盖 NER、机器翻译和问答三个 NLU 任务，同时贡献 1501 对梵语问答数据集并验证 RAG 策略的有效性，揭示模型规模是跨语言泛化的决定性因素。
tags:
  - ACL 2025
  - Cross-Lingual Generalization
  - Classical Languages
  - Sanskrit
  - Zero-Shot
  - RAG
  - NER
---

# A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs

**会议**: ACL 2025  
**arXiv**: [2505.13173](https://arxiv.org/abs/2505.13173)  
**代码**: [GitHub](https://github.com/mahesh-ak/SktQA)  
**机构**: University of Tübingen, University of Lyon 1, Indian Institute of Technology Kanpur
**领域**: NLP / 多语言与机器翻译 / 跨语言泛化  
**关键词**: Cross-Lingual Generalization, Classical Languages, Sanskrit, Zero-Shot, RAG, NER

## 一句话总结

系统评估 LLM 在三种古典语言（梵语、古希腊语、拉丁语）上的零样本跨语言泛化能力，涵盖 NER、机器翻译和问答三个 NLU 任务，同时贡献 1501 对梵语问答数据集并验证 RAG 策略的有效性，揭示模型规模是跨语言泛化的决定性因素。

## 研究背景与动机

LLM 已展现强大的跨语言泛化能力，但对古典语言的零样本 NLU 能力研究极其有限，此前仅 Volk et al. (2024) 探索了拉丁语翻译和摘要，缺乏多任务的系统评估。

- **古典语言的独特地位**：梵语、古希腊语和拉丁语虽然数字化下游任务数据极度稀缺，但拥有丰富的古代文献资源，且对高资源语言有深远影响（如拉丁语贡献了约 28% 的英语词汇），同时其高度屈折变化给 NLP 处理带来独特挑战
- **现有数据资源严重不足**：梵语 NER 数据集仅 139 句（1558 tokens），拉丁语 3410 句；梵语 QA 数据集更少，仅有 80 个亲属关系问题（Terdalkar & Bhattacharya, 2019），无法支撑有效的评估研究
- **跨语言迁移机制缺乏验证**：尽管前人工作（Cahyawijaya et al., 2024; Han et al., 2024）在低资源语言上展示了 LLM 泛化能力，但古典语言因其特殊的形态学复杂性和有限的预训练数据覆盖，是否受益于跨语言迁移尚未被系统检验
- **核心研究问题**：LLM 是否能通过跨语言迁移（而非专门训练）理解古典语言？模型规模、提示语言和文字体系分别扮演什么角色？

## 方法详解

### 整体框架

研究设计包含两组零样本实验：(1) 在三种古典语言上评估 NER 和机器翻译（翻译到英语）两个 NLU 任务，使用已有公开数据集；(2) 聚焦梵语，贡献新的事实型问答数据集，结合 BM25 检索的 RAG 方法评估深层理解能力。评估模型为两对大小模型：闭源 GPT-4o / GPT-4o-mini，开源 LLaMA-3.1-405B-Instruct / LLaMA-3.1-8B-Instruct（知识截止日期 2023 年底，多数数据集发布于此后）。

### 关键设计

1. **梵语事实型 QA 数据集构建**：填补梵语 QA 资源空白，构建 1501 个事实型问答对，覆盖两个代表性领域——古代史诗《罗摩衍那》(Rāmāyaṇa) 和阿育吠陀经典医学文本《Bhāvaprakāśanighaṇṭu》。所有 QA 对由人工标注，答案为单词级精确匹配格式，支持闭卷和 RAG 两种评估模式。

2. **BM25 + 词形还原的 RAG 增强方案**：采用 BM25 检索 top-k 相关段落（k=4 最优），与基于 FastText 和 GloVe 的嵌入式检索器对比——BM25 在所有指标上一致优于嵌入方法。针对梵语高度屈折的特点，引入基于 Transformer Seq2Seq 的梵语词形还原步骤（在 DCS 语料库上训练，F1=0.94），将屈折形式还原为词元以提升 BM25 的词汇匹配召回率。

3. **多维度系统对比实验设计**：设计四个正交分析维度——(a) 模型规模：GPT-4o / LLaMA-405B vs. GPT-4o-mini / LLaMA-8B；(b) 提示语言：英语提示 vs. 目标语言提示；(c) 文字体系：天城体 (Devanagari) vs. IAST 罗马化转写；(d) 上下文有效性：有答案的检索上下文 vs. 无答案的检索上下文，系统分析各因素对跨语言泛化的影响。

## 实验关键数据

### 数据集总览

| 任务 | 语言 | 数据来源 | 测试集大小 |
|------|------|---------|-----------|
| NER | 梵语 | Terdalkar (2023) | 139 句 |
| NER | 拉丁语 | Erdmann et al. (2019) | 3,410 句 |
| NER | 古希腊语 | Myerston (2025) | 4,957 句 |
| MT→en | 梵语 | Maheshwari et al. (2024) | 6,464 句 |
| MT→en | 拉丁语 | Rosenthal (2023) | 1,014 句 |
| MT→en | 古希腊语 | Palladino et al. (2023) | 274 句 |
| QA | 梵语 | **本文贡献** | 1,501 对 |

### 零样本核心性能对比

| 任务 | 指标 | GPT-4o | LLaMA-405B | GPT-4o-mini | LLaMA-8B |
|------|------|--------|-----------|-------------|----------|
| NER (梵语) | Macro F1 | 0.637 | 0.561 | 0.359 | 0.164 |
| MT (梵语) | BLEU | 0.179 | 0.193 | 0.135 | 0.120 |
| QA 闭卷 (梵语) | EM | 0.36 | 0.41 | 0.18 | 0.13 |
| QA + RAG (梵语) | EM | **0.46** | 0.42 | 0.25 | 0.09 |

### 屈折变化对 QA 的影响（英语提示）

| 模型 | 闭卷-屈折 | 闭卷-词元 | +RAG-屈折 | +RAG-词元 |
|------|----------|----------|----------|----------|
| GPT-4o | 0.36 | 0.37 | 0.46 | 0.48 |
| LLaMA-405B | 0.41 | 0.40 | 0.42 | 0.44 |
| GPT-4o-mini | 0.18 | 0.20 | 0.25 | 0.28 |
| LLaMA-8B | 0.13 | 0.15 | 0.09 | 0.10 |

### 文字体系对比（梵语，英语提示）

| 模型 | MT-BLEU (天城体) | MT-BLEU (IAST) | NER-F1 (天城体) | NER-F1 (IAST) |
|------|-----------------|----------------|----------------|---------------|
| GPT-4o | 0.179 | 0.165 | 0.637 | 0.599 |
| LLaMA-405B | 0.193 | 0.148 | 0.561 | 0.556 |
| GPT-4o-mini | 0.135 | 0.099 | 0.359 | 0.318 |
| LLaMA-8B | 0.120 | 0.063 | 0.164 | 0.149 |

### 多维度分析总结

| 分析维度 | 关键发现 |
|---------|---------|
| 模型规模 | 大模型在所有任务和语言上一致优于小模型，小模型在特殊实体类型和 RAG 利用上尤其退化 |
| 提示语言 | 英语提示通常优于原语言提示（尤其小模型），间接证明模型未在古典语言上做过指令微调 |
| 文字体系 | 天城体稍优于 IAST 罗马化转写，但差距不大——两种形式在预训练数据中都有一定覆盖 |
| NER 错误分析 | 小模型在语义相近实体（Deva/Asura/Rakshasa, Kingdom/City/Forest）上混淆严重，大模型有清晰决策边界 |
| RAG 上下文 | 有答案上下文时 EM=0.46 远优于无答案上下文，证明大模型具备梵语段落理解能力 |
| 屈折变化 | 词形还原后 EM 仅略有提升（+1～3%），说明屈折错误不是主要瓶颈 |

## 论文亮点与不足

### 亮点

- 首个系统性评估 LLM 在三种古典语言上多任务（NER/MT/QA）零样本 NLU 能力的工作
- 贡献 1501 对覆盖梵语史诗和医学文本的事实型 QA 数据集，填补资源空白
- 多维度正交对比设计（模型规模 × 提示语言 × 文字体系 × 检索方式）提供全面分析
- NER 混淆矩阵分析深入揭示了不同规模模型在古典语言特有实体上的行为差异
- 验证 BM25 + 词形还原在高屈折语言上的 RAG 有效性，为古典语言 NLP 提供实用方案

### 不足

- 梵语数据集规模有限（QA 1501 对、NER 仅 139 句），统计置信度可能不足
- 部分数据集发布在模型知识截止日期前，存在数据污染风险（古希腊语 MT 表现异常高）
- 仅使用 BM25 作为主要检索方法，未探索跨语言密集检索模型（如 mDPR）
- 仅评估 4 个模型（2 个闭源 + 2 个开源），未覆盖 Gemini、Claude、Mistral 等其他模型家族
- KG-QA 探索因知识图谱不完整而无法得出有效结论，方向有待完善

## 相关工作与启发

- **跨语言泛化**：Cahyawijaya et al. (2024) 展示 LLM 在低资源语言的少样本上下文学习能力，本文将评估扩展至古典语言的零样本设定
- **古典语言 NLP**：Riemenschneider & Frank (2023)、Nehrdich et al. (2024) 构建古典语言专用模型，但限于形态学解析等任务；本文验证通用 LLM 的跨语言迁移可作为替代方案
- **RAG 范式**：Lewis et al. (2020) 提出 RAG，本文首次将其应用于梵语 QA 并验证词形还原步骤的必要性
- **文字体系与迁移**：Muller et al. (2021)、Fujinuma et al. (2022) 发现正字法是跨语言迁移的关键因素，本文通过天城体 vs. IAST 对比在梵语上进一步验证
- **启发**：古典语言虽然下游数据稀缺，但凭借其对高资源语言的深层影响，大型 LLM 可能已通过间接路径（共享词根、句法结构）获得一定的理解能力

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 选题独特，古典语言的零样本泛化是有价值且未被充分探索的研究方向
- **实用性**: ⭐⭐⭐ — 应用场景偏窄（数字人文、语言学研究），但 QA 数据集和 RAG 方案有实际价值
- **实验充分度**: ⭐⭐⭐⭐ — 多任务、多语言、多模型、多维度正交对比，分析细致
- **写作质量**: ⭐⭐⭐⭐ — 案例研究风格恰当，结构清晰，混淆矩阵分析深入

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction](translation_and_fusion_improves_cross-lingual_information_extraction.md)
- [\[ACL 2025\] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)
- [\[ACL 2025\] Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons](cross_lingual_neurons_compression.md)
- [\[ACL 2025\] Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)
- [\[ACL 2025\] Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models](statement-tuning_enables_efficient_cross-lingual_generalization_in_encoder-only_.md)

<!-- RELATED:END -->
