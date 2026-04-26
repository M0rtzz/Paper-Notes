---
title: >-
  [论文解读] NameTag 3: A Tool and a Service for Multilingual/Multitagset NER
description: >-
  [ACL 2025][命名实体识别] 本文介绍 NameTag 3，一个开源的多语言、多数据集、多标签集命名实体识别工具和云服务，基于微调的预训练语言模型，单个 355M 参数模型在 15 种语言的 21 个测试集上达到 SOTA，同时比 DeepSeek-R1 等 LLM 快 10,000 倍以上。
tags:
  - ACL 2025
  - 命名实体识别
  - 多语言NER
  - 嵌套NER
  - 多标签集
  - 开源工具
---

# NameTag 3: A Tool and a Service for Multilingual/Multitagset NER

**会议**: ACL 2025  
**arXiv**: [2506.05949](https://arxiv.org/abs/2506.05949)  
**代码**: https://github.com/ufal/nametag3 (有)  
**领域**: nlp_understanding  
**关键词**: 命名实体识别, 多语言NER, 嵌套NER, 多标签集, 开源工具

## 一句话总结
本文介绍 NameTag 3，一个开源的多语言、多数据集、多标签集命名实体识别工具和云服务，基于微调的预训练语言模型，单个 355M 参数模型在 15 种语言的 21 个测试集上达到 SOTA，同时比 DeepSeek-R1 等 LLM 快 10,000 倍以上。

## 研究背景与动机

命名实体识别（NER）是 NLP 和知识提取系统中的基础预处理步骤，需要识别文本中的人名、地名、机构名等。虽然 NER 研究（尤其是英语）已经非常成熟，但实际可用的多语言开源 NER 工具仍然匮乏。

**现有痛点**：

**工具覆盖不足**：现有工具如 Stanza 和 SpaCy 虽然支持多种语言，但每种语言需要独立训练模型，无法实现跨语言迁移。Stanza 基于冻结的 Flair 嵌入 + Bi-LSTM + CRF 架构，技术路线已经落后。

**不支持嵌套 NER**：Stanza 和 SpaCy 都不支持嵌套实体识别，但许多语言（如捷克语的 CNEC 2.0 语料库）有复杂的嵌套标注。

**缺乏灵活的标签集支持**：不同数据集使用不同的标签集（CoNLL、UNER、OntoNotes），现有工具无法在一个模型中同时支持多种标签集。

**LLM 做 NER 效率极低**：虽然 GPT、DeepSeek 等大模型可以零样本做 NER，但实际上准确率远低于微调模型，且速度慢数万倍。

**核心矛盾**：需要一个轻量、高效、支持多语言/多标签集/嵌套NER的统一工具，同时要优于 LLM 零样本性能。

**切入角度**：在一个微调的多语言预训练模型上联合训练，通过多标签集分类头设计实现标签集灵活性，通过序列到序列解码头支持嵌套 NER。

## 方法详解

### 整体框架
NameTag 3 基于预训练语言模型（XLM-R Large 355M 或 RobeCzech Base 126M）进行微调，提供两种识别模式：扁平 NER（softmax 分类头）和嵌套 NER（seq2seq 解码头）。单个多语言模型联合训练 21 个数据集、17 种语言和 3 种标签集。

### 关键设计

1. **多标签集学习（Multitagset Learning）**：

    - 核心思路：为每种标签集分配独立的分类头，共享底层编码器
    - 训练时联合优化编码器和所有分类头
    - 推理时根据请求的标签集选择对应的分类头，确保只预测有效标签
    - 支持 CoNLL（PER/LOC/ORG/MISC）、UNER（通用 NER v1）和 OntoNotes 三种标签集
    - 设计动机：避免为每个标签集维护独立模型，实现统一服务

2. **嵌套 NER 的 seq2seq 解码**：

    - 用 Transformer seq2seq 解码器替换 softmax 分类头
    - 解码器对每个输入 token 生成线性化（展平）的嵌套标签序列
    - 使用硬注意力机制聚焦当前 token
    - 先冻结编码器权重预训练几个 epoch，让解码器适应编码器表示，再联合微调
    - 设计动机：嵌套实体需要为同一文本区间输出多个标签，分类头无法实现

3. **平方根温度采样训练**：

    - 训练批次中每个语料库的采样概率正比于其句子数的平方根
    - 效果：对大语料库降采样、对小语料库升采样
    - 使用宏平均 span F1 评估确保各数据集平衡性能
    - 设计动机：解决多语料库联合训练中的数据不平衡问题

### 训练策略
- 扁平多语言模型：XLM-R Large，30 epochs，学习率 2e-5，cosine 衰减
- 嵌套 NER 模型：RoBERTa Large（英语）/ RobeCzech Base（捷克语），先冻结 20 epochs 预训练解码器，再联合微调 50-60 epochs
- 批大小 4-16，取决于具体数据集

## 实验关键数据

### 主实验 — 扁平 NER F1（多语言模型 355M）

| 语言/数据集 | NameTag 3 (Multi) | NameTag 3 (Mono) | Stanza | 之前 SOTA | SOTA 模型大小 |
|------------|-------------------|------------------|--------|----------|-------------|
| English CoNLL-2003 | **94.09** | 93.80 | 92.1 | 94.60 | 1853M |
| Chinese OntoNotes v5 | **81.63** | 81.76 | 79.2 | 80.20 | 147M |
| Croatian UNER SET | **95.55** | 94.08 | - | 95.00 | 355M |
| Ukrainian Lang-uk | **92.88** | 90.45 | 86.1 | 88.73 | 110M |
| Czech CNEC 2.0 | **86.24** | 85.31 | - | - | - |

### 消融实验 — 与 LLM 对比（English CoNLL-2003 完整测试集）

| 方法 | F1 | 速度（句/秒） | 总用时 |
|------|-----|-------------|--------|
| NameTag 3 (355M) | **94.09** | 801 | **4.6秒** |
| DeepSeek R1 70B 5-shot | 74.00 | 0.04 | 25小时 |
| DeepSeek R1 32B 5-shot | 74.26 | 0.06 | 16小时 |
| ChatGPT 3.5 ICL | 74.99 | - | - |

### 嵌套 NER 和跨语言迁移

| 任务 | NameTag 3 F1 | 之前 SOTA F1 | 说明 |
|------|-------------|-------------|------|
| ACE-2004 嵌套 | 88.39 | 88.72 | 接近 SOTA |
| CNEC 2.0 嵌套 (46类) | **86.39** | 83.44 | 新 SOTA |
| Cebuano 跨语言 | **96.97** | 82.2 | 未见语言，大幅超越 |
| Tagalog 跨语言 | **97.78** | 83.7 | 未见语言，+14 点 |

### 关键发现
- 多语言联合训练模型在大多数数据集上优于单语言模型，证明了跨语言迁移的有效性
- 在未见过的语言（宿务语、塔加洛语）上的跨语言迁移效果惊人（+14点F1）
- 微调的 355M 模型比 70B 的 DeepSeek R1 高出 20 个百分点的 F1，且快 10,000 倍以上
- 多标签集设计使得一个模型即可服务所有标签集需求

## 亮点与洞察
- 在 LLM 时代证明了"有训练数据时，小模型微调仍然是最优选择"的论点，对比数据极具说服力
- 多标签集设计优雅地解决了不同数据集标签不统一的实际部署问题
- 作为工具论文，提供了命令行、Web 应用和 REST API 三种使用方式，覆盖各类部署场景
- 跨语言迁移到未见语言的效果证明了多语言联合训练的巨大价值

## 局限与展望
- 训练数据以拉丁字母为主，非拉丁字母语言覆盖不足（仅中文、阿拉伯语、乌克兰语）
- 模型使用 CC BY-NC-SA 4.0 许可证，限制了商业使用
- 作为监督微调模型，高度依赖高质量人工标注数据的可用性
- 扁平 NER 将嵌套标注简化为 4 类标签，损失了原始标注的细粒度信息
- 未探索与 LLM 的协作方案，如用 LLM 辅助银标注数据生成

## 相关工作与启发
- 是 NameTag 系列的第三代（2014年→2019年→2025年），演化路径清晰
- 与 Stanza 的核心区别：微调 PLM（而非冻结嵌入 + Bi-LSTM）和单模型多语言（而非逐语言独立模型）
- 为 NER 工具开发提供了一个标杆范式：多语言联合训练 + 多标签集分类头 + seq2seq 嵌套解码
- 启示：在有标注数据的传统 NLP 任务上，精心微调的小模型仍远优于 LLM

## 评分
- 新颖性: ⭐⭐⭐ 方法上是成熟技术的组合（微调PLM + 多头 + seq2seq），创新性有限
- 实验充分度: ⭐⭐⭐⭐⭐ 21个数据集、15种语言、与SOTA/Stanza/SpaCy/LLM全面对比，极其充分
- 写作质量: ⭐⭐⭐⭐ 工具论文写作规范，信息量大，但略显冗长
- 价值: ⭐⭐⭐⭐⭐ 作为开源NER工具的实用价值极高，对学术社区和非英语NLP研究者尤其有用

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] M-RewardBench: Evaluating Reward Models in Multilingual Settings](m_rewardbench.md)
- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](execute_a_multilingual_benchmark_for_llm_token_understanding.md)
- [\[ACL 2025\] LangMark: A Multilingual Dataset for Automatic Post-Editing](langmark_a_multilingual_dataset_for_automatic_post-editing.md)
- [\[ACL 2025\] LangSAMP: Language-Script Aware Multilingual Pretraining](langsamp_multilingual_pretraining.md)
- [\[ACL 2025\] LexGen: Domain-aware Multilingual Lexicon Generation](lexgen_domain-aware_multilingual_lexicon_generation.md)

<!-- RELATED:END -->
