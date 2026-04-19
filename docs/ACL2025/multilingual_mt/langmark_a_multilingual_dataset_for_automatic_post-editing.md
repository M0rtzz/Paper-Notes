---
title: >-
  [论文解读] LangMark: A Multilingual Dataset for Automatic Post-Editing
description: >-
  [ACL 2025][automatic post-editing] 发布 LangMark——一个包含 206,983 个三元组、覆盖英语到七种语言的大规模多语言自动后编辑（APE）数据集，并证明 LLM 配合 few-shot prompting 能有效改善专有 NMT 引擎的输出质量。
tags:
  - ACL 2025
  - automatic post-editing
  - multilingual dataset
  - machine translation
  - LLM
  - 提示学习
---

# LangMark: A Multilingual Dataset for Automatic Post-Editing

**会议**: ACL 2025  
**arXiv**: [2511.17153](https://arxiv.org/abs/2511.17153)  
**代码**: 无（数据集发布于 [Zenodo](https://zenodo.org/records/15553365)）  
**领域**: NLP / 机器翻译  
**关键词**: automatic post-editing, multilingual dataset, machine translation, LLM, few-shot prompting

## 一句话总结

发布 LangMark——一个包含 206,983 个三元组、覆盖英语到七种语言的大规模多语言自动后编辑（APE）数据集，并证明 LLM 配合 few-shot prompting 能有效改善专有 NMT 引擎的输出质量。

## 研究背景与动机

自动后编辑（Automatic Post-Editing, APE）旨在自动纠正机器翻译输出中的错误，在保证翻译质量的同时减少人工干预。尽管神经机器翻译（NMT）已取得长足进步，但当前 APE 研究面临几个关键瓶颈：

**数据集规模不足**：WMT APE 共享任务仅有 15K-18K 三元组，SubEdits 虽有 161K 但仅覆盖英德一种语言对

**语言多样性缺乏**：大部分现有数据集仅覆盖 1-2 种语言对，难以支持多语言 APE 研究

**合成数据的局限**：eSCAPE 等合成数据集虽规模大（数百万级），但无法捕捉高级 NMT 系统所需的细微编辑

**NMT 输出仍有缺陷**：即便是领先的 NMT 系统，在市场营销等专业领域仍会产生上下文不当的翻译（如"our people"被误译为"我们的民族"，"pitch"被按"焦油"含义翻译）

这些问题共同指向：**需要一个大规模、多语言、人工标注的 NMT 后编辑数据集**。

## 方法详解

### 整体框架

LangMark 本质是一个数据集贡献工作，附带基线评估实验。核心流程为：

1. 数据收集与标注 → 2. 数据集统计分析 → 3. 检索增强的 few-shot APE 评估

### 关键设计

1. **数据集构建**

    - 来源：Smartsheet 平台的市场营销类文档，由翻译管理系统（TMS）分段为句子/短语
    - 翻译：使用针对 Smartsheet 领域训练的专有 NMT 引擎
    - 后编辑：由专业语言学家（5+ 年行业经验）在 TMS 中完成
    - 隐私保护：使用 Google DLP 工具移除个人身份信息
    - 去重处理：对每种语言对移除重复三元组
    - 设计动机：保留真实工业数据原貌，使评估更贴近实际应用场景

2. **语言覆盖**

    - 英语 → 德语（33.3K）、西班牙语（32.8K）、法语（33.0K）、意大利语（32.5K）、日语（28.2K）、巴西葡萄牙语（32.0K）、俄语（8.6K）
    - 总计 206,983 个三元组
    - 设计动机：覆盖多样化语系（日耳曼语、罗曼语、斯拉夫语、日语），增强基准的通用性

3. **评估框架**

    - 90%/10% 训练/测试划分，训练集用于检索示例
    - 使用 OpenAI text-embedding-3-small 对源段落进行嵌入
    - 余弦相似度检索 20 个最相似的源-后编辑对作为 few-shot 示例
    - 统一 20-shot prompting 格式评估所有模型
    - 设计动机：zero-shot 方式无法超越强 NMT 基线，需要领域内示例引导

### 损失函数 / 训练策略

本文为数据集工作，不涉及模型训练。评估使用现有 LLM 的推理能力，不做微调。

## 实验关键数据

### 主实验（CHRF 分数，20-shot APE）

| 模型 | EN-RU | EN-BR | EN-JP | EN-IT | EN-FR | EN-ES | EN-DE |
|------|-------|-------|-------|-------|-------|-------|-------|
| NMT Baseline | 68.90 | 89.44 | 70.22 | 89.58 | 81.96 | 86.07 | 81.29 |
| GPT-4o | **69.68** | 89.21 | **73.94** | **89.79** | **82.75** | **86.62** | **81.41** |
| Qwen2.5-72B | 70.13 | 89.03 | 72.93 | 89.10 | 82.34 | 86.44 | 81.16 |
| Claude 3.5-Haiku | 69.08 | 88.81 | 71.64 | 88.76 | 82.21 | 86.08 | 80.66 |
| Gemini-1.5 Flash | 68.92 | 89.18 | 71.69 | 89.40 | 82.20 | 86.24 | 81.01 |
| Llama 3.1-70B | 69.55 | 86.82 | 68.37 | 86.80 | 80.97 | 83.75 | 79.12 |

GPT-4o 是唯一在大多数语言对上一致超越 NMT 基线的闭源模型；Qwen2.5-72B 在俄语上表现最佳。

### 商用 MT 引擎对比（全数据集 CHRF）

| MT 引擎 | EN-DE | EN-ES | EN-FR | EN-IT | EN-JP | EN-BR | EN-RU |
|---------|-------|-------|-------|-------|-------|-------|-------|
| 专有 NMT (本数据集) | 81.09 | 86.04 | 81.54 | 89.73 | 69.77 | 89.13 | — |
| Google Translate | 73.95 | 79.79 | 76.57 | 79.80 | 62.11 | 83.70 | 64.34 |
| Microsoft Translator | 75.74 | 80.32 | 76.07 | 82.57 | 62.82 | 84.97 | 64.38 |
| DeepL | 73.03 | 75.01 | 74.74 | 76.96 | 55.26 | 83.93 | 67.74 |

专有 NMT 在所有语言上显著优于通用 MT 引擎，说明 LangMark 的 APE 任务确实具有挑战性。

### 关键发现

1. **高基线难以超越**：除 GPT-4o 外的所有模型无法一致性地改善 NMT 输出，证明该数据集作为 APE 基准的挑战性
2. **编辑保守性问题**：所有 LLM 的编辑次数都显著少于人工基线，表明"何时编辑"仍是关键挑战
3. **开源模型潜力**：Qwen2.5-72B 的表现接近最佳闭源模型，在俄语上甚至更优
4. **语言差异明显**：日语和俄语（需要大量编辑的语言）是 LLM 改善空间最大的方向
5. **领域特异性**：营销领域的翻译需要对上下文的深度理解，而非简单的语法修正

## 亮点与洞察

- **来自工业界的真实数据**：不同于学术构造的数据集，LangMark 反映了真实翻译工作流中的挑战
- **专有 NMT 作为基线**：使用高质量领域内 NMT 引擎作为起点，确保 APE 任务的"天花板"足够高
- **"编辑还是不编辑"问题**：首次系统性地讨论了 APE 模型的编辑决策问题（precision/recall 分析）
- **评估方法论**：通过检索增强的 few-shot 方式统一了不同模型的评估框架

## 局限与展望

1. 数据来源仅为市场营销领域，不代表其他翻译场景（法律、医学、文学等）
2. 俄语三元组仅 8.6K，数据量显著少于其他语言对
3. 未探索微调方案（如在 LangMark 上微调开源 LLM 进行 APE）
4. 专有 NMT 引擎不可复现，限制了结果的完全可复现性
5. 检索方法仅基于源段落相似度，未考虑错误类型匹配

## 相关工作与启发

- 相比 WMT APE (15-18K, 单语言对) 和 SubEdits (161K, 仅英德)，LangMark 在规模和语言覆盖上有显著提升
- 启发：可在此数据集上研究自适应编辑策略（高置信度不编辑、低置信度细粒度修改）
- 可结合质量估计（QE）方法来辅助"是否编辑"的决策

## 评分

- **新颖性**: ⭐⭐⭐ — 主要是数据集贡献，方法层面创新有限
- **实验充分度**: ⭐⭐⭐⭐ — 涵盖多种闭源和开源模型，多语言对比全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数据集描述详细，可视化丰富
- **价值**: ⭐⭐⭐⭐ — 填补多语言 NMT APE 数据集的重要空白，工业价值高

<!-- RELATED:START -->

## 相关论文

- [Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)
- [An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT)](an_expanded_massive_multilingual_dataset_for_high-performance_language_technolog.md)
- [AskQE: Question Answering as Automatic Evaluation for Machine Translation](askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)
- [M3FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset](m3finmeeting_a_multilingual_multi-sector_and_multi-task_financial_meeting_unders.md)

<!-- RELATED:END -->
