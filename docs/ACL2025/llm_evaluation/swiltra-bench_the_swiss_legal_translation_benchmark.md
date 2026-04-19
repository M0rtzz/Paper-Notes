---
title: >-
  [论文解读] SwiLTra-Bench: The Swiss Legal Translation Benchmark
description: >-
  [ACL 2025][法律翻译] 构建了 SwiLTra-Bench——一个包含超过 18 万对齐瑞士法律翻译对的大规模多语言基准（覆盖法律、判例摘要、新闻稿，涵盖德法意罗英五种语言），系统评估了前沿 LLM 和微调开源 SLM 在法律翻译上的表现，并提出 SwiLTra-Judge 自动评估方法。
tags:
  - ACL 2025
  - 法律翻译
  - 多语言基准
  - 瑞士法律
  - LLM翻译评估
  - 微调
---

# SwiLTra-Bench: The Swiss Legal Translation Benchmark

**会议**: ACL 2025  
**arXiv**: [2503.01372](https://arxiv.org/abs/2503.01372)  
**代码**: 有（论文提供了 Datasets 和 Code 链接）  
**领域**: 机器翻译 / 法律NLP  
**关键词**: 法律翻译, 多语言基准, 瑞士法律, LLM翻译评估, 微调

## 一句话总结

构建了 SwiLTra-Bench——一个包含超过 18 万对齐瑞士法律翻译对的大规模多语言基准（覆盖法律、判例摘要、新闻稿，涵盖德法意罗英五种语言），系统评估了前沿 LLM 和微调开源 SLM 在法律翻译上的表现，并提出 SwiLTra-Judge 自动评估方法。

## 研究背景与动机

瑞士是一个拥有四种官方语言（德语、法语、意大利语、罗曼什语）的国家，法律文件必须翻译为多种语言。传统法律翻译依赖既懂法律又懂语言的专业人员，形成严重的翻译瓶颈，影响了司法公平获取。

现有的神经机器翻译（NMT）系统在法律文本上表现受限，原因在于：
1. 法律语言具有独特的话语结构和专业术语
2. 缺乏高质量的多语言法律平行语料
3. 低资源语言（如罗曼什语）的翻译覆盖尤其困难

此前虽有初步探索，但尚不清楚当前 LLM 在大规模瑞士法律翻译基准上的实际表现，无论是零样本还是微调场景。

## 方法详解

### 整体框架

本工作包含三个核心贡献：
1. **SwiLTra-Bench 数据集**：大规模多语言法律翻译基准
2. **全面模型评估**：首次大规模比较前沿 LLM 和微调 SLM
3. **SwiLTra-Judge**：与人类专家评估对齐的 LLM 评估方法

### 关键设计

1. **三类法律文本子数据集**: 
    - CH-Law-Trans（瑞士法律翻译）：包含法律级、条款级、段落级翻译，覆盖 5 种语言（德法意罗英），段落级训练集 15 万+对
    - CH-Headnote-Trans（判例摘要翻译）：来自瑞士联邦最高法院的标志性判例，包含 BGE/Regest/Text 三个层级，训练集 2.6 万+对
    - CH-Press-Trans（新闻稿翻译）：法院新闻稿，训练集 867 对
    - 所有数据利用政府官方 HTML 结构进行高质量对齐，而非自动句对齐

2. **五类模型全面评估**: 系统比较了翻译专用模型（MADLAD-400、Tower-Instruct）、前沿模型（Claude-3.5-Sonnet、GPT-4o、Gemini-1.5-Pro 等）、推理模型（o1）、开源 SLM 和微调模型，覆盖零样本和微调两种设置。评估指标包括词汇级（BLEU、ChrF、METEOR）和模型级（BERTScore、BLEURT、XCOMET、GEMBA-MQM）。

3. **SwiLTra-Judge 评估系统**: 设计了专门的 LLM 评估系统，用于自动评估翻译质量。通过与人类专家标注的对比验证，SwiLTra-Judge 与专家评估的一致性最高，为法律翻译提供了可靠的自动评估框架。

### 损失函数 / 训练策略

微调设置：
- 使用 4-bit 量化 + 8-bit AdamW 优化器
- Rank Stabilized LoRA（rank=16, alpha=16）
- 序列长度 512（覆盖 99%+训练数据）
- 使用 packing 技术，batch size 128
- 线性学习率调度，1000 步 warmup，学习率 1e-4
- 早停（patience=3），大多数模型在 1 epoch 后达到最低验证损失
- 微调了 13 个开源模型（Gemma、Llama、Phi、Qwen 系列）

## 实验关键数据

### 主实验

翻译模型对比（平均分，越高越好）：

| 模型 | 大小 | GEMBA-MQM | XCOMET | METEOR | ChrF |
|------|------|-----------|--------|--------|------|
| Google Translate | N/A | 53.20 | 64.61 | 41.15 | 47.81 |
| MADLAD-400-7B | 7B | 62.66 | 87.40 | 43.70 | 51.67 |
| Tower-Instruct-13B | 13B | 57.38 | 75.94 | 43.95 | 48.46 |
| Claude-3.5-Sonnet | large | 80.66 | 90.70 | 56.71 | 65.87 |
| GPT-4o | large | 80.27 | 80.96 | 55.56 | 63.27 |
| Gemini-1.5-Pro | large | 81.88 | 87.13 | 57.92 | 70.07 |
| o1 | large | **85.81** | **91.35** | **58.91** | **70.11** |
| GPT-4o-mini | small | 82.59 | 87.90 | 54.03 | 59.86 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 翻译模型 vs 前沿模型 | MADLAD-400 在 XCOMET 上超过 GPT-4o | 翻译专用模型在法律文本上具有竞争力 |
| 零样本 vs 微调 SLM | 微调大幅提升质量但仍落后前沿零样本模型 | 微调后差距缩小但未消除 |
| 法律 vs 判例摘要 | 翻译模型在法律上强但判例摘要上弱 | 文本类型影响模型表现 |
| 跨语言表现 | 各语言间翻译质量相对均匀 | 多语言覆盖较为平衡 |

### 关键发现

- **o1 推理模型总分最高**（GEMBA-MQM 85.81），但成本远高于 Claude-3.5-Sonnet，后者性价比最优
- **MADLAD-400 在法律翻译上出人意料地强**，XCOMET 超过 GPT-4o（87.40 vs 80.96）
- **微调开源 SLM 显著提升质量**但仍落后于最佳零样本前沿模型
- **Google Translate 表现意外糟糕**（GEMBA-MQM 仅 53.20）
- **人类专家在法律翻译上的一致性高于判例摘要**，反映法律文本的标准化程度更高
- Claude-3.5-Haiku 等小型前沿模型的成本效益值得关注

## 亮点与洞察

- 数据集质量高：利用政府官方 HTML 结构对齐，避免了传统自动句对齐的噪声
- 评估全面：五类模型 × 七种指标 × 三种文本类型的完整矩阵
- 实际应用价值大：直接服务于瑞士政府的法律翻译需求和司法公平
- 对低资源语言（罗曼什语）的覆盖有特殊意义

## 局限与展望

- 罗曼什语和英语在法律数据集中的覆盖有限（分别仅有约 2 万和 3 万段落级样本）
- 微调仅使用 LoRA 且序列长度限制在 512，对长法律文本可能不够
- 未评估翻译系统在实际法律工作流中的端到端效用
- SwiLTra-Judge 的评估本身依赖 GPT-4o（GEMBA-MQM），存在循环依赖风险

## 相关工作与启发

- 法律 NLP 领域的翻译基准稀缺，本文填补了瑞士法律场景的重要空白
- MADLAD-400 作为翻译专用模型的强劲表现，提示在专业领域微调的价值
- 启发：对于多语言法律场景，前沿 LLM 的零样本能力已相当可用，但仍需法律专家验证

## 评分

- 新颖性: ⭐⭐⭐ — 主要贡献是数据集和评估，方法上无显著创新
- 实验: ⭐⭐⭐⭐⭐ — 模型覆盖极广，评估指标全面，包含人类专家验证
- 写作: ⭐⭐⭐⭐ — 结构清晰，数据统计详尽
- 实用性: ⭐⭐⭐⭐ — 数据集和评估框架对法律翻译有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)
- [PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](../../NeurIPS2025/llm_evaluation/parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](../../NeurIPS2025/llm_evaluation/rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)
- [TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)

<!-- RELATED:END -->
