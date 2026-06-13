---
title: >-
  [论文解读] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models
description: >-
  [ACL 2025][LLM 其他][知识蒸馏] 提出 PiFi 框架，将 LLM 的单层冻结参数插入到 SLM 中并微调，以极低计算开销显著提升 SLM 在 NLU 和 NLG 任务上的性能。 核心问题：LLM 具有强大的泛化能力和语言知识，但计算开销大、难以部署在资源受限环境；SLM 高效但泛化能力不足…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "知识蒸馏"
  - "小语言模型"
  - "大语言模型"
  - "模型压缩"
  - "迁移学习"
---

# PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models

**会议**: ACL 2025  
**arXiv**: [2506.07424](https://arxiv.org/abs/2506.07424)  
**代码**: 未公开  
**领域**: LLM/NLP  
**关键词**: 知识蒸馏, 小语言模型, 大语言模型, 模型压缩, 迁移学习  

## 一句话总结

提出 PiFi 框架，将 LLM 的单层冻结参数插入到 SLM 中并微调，以极低计算开销显著提升 SLM 在 NLU 和 NLG 任务上的性能。

## 研究背景与动机

**核心问题：** LLM 具有强大的泛化能力和语言知识，但计算开销大、难以部署在资源受限环境；SLM 高效但泛化能力不足。如何让 SLM 获得 LLM 的知识优势？

**现有方案局限：**
- 知识蒸馏方法（参数化/非参数化）需要完整的 LLM 作为教师模型或生成合成数据，计算成本高
- 直接微调 LLM 需要大量显存和计算资源
- 现有 SLM 增强方法（领域预训练、数据增强）无法引入 LLM 级别的外部知识

**核心动机：** 受视觉领域将 LLM 层用作视觉编码器的启发，LLM 的单个 Transformer 层已经编码了丰富的语言知识，通过"插入+冻结"的方式可以在几乎不增加训练参数量的情况下有效传递 LLM 的知识到 SLM。

## 方法详解

### 整体框架

PiFi（Plug-in and Fine-tuning）框架的核心思想是：从 LLM（如 Llama-3.1-8B）中提取单层 Transformer，冻结后插入 SLM 的特定位置，然后对组合模型进行微调。框架支持 encoder-only 模型（如 BERT）和 encoder-decoder 模型（如 T5）两种架构。

### 关键设计

1. **维度适配层：** 由于 SLM 和 LLM 的隐藏维度不同（如 BERT 768 vs Llama 4096），引入线性映射层 $L_{in}$（升维）和 $L_{out}$（降维）进行维度适配。对 Encoder-only 模型：$h_{enc} = Enc(x)$, $h_{LLM} = L_{LLM}(L_{in}(h_{enc}))$, $\hat{y} = Head(L_{out}(h_{LLM}))$

2. **LLM 层冻结策略：** 在微调阶段冻结 $L_{LLM}$ 的参数，只训练 SLM 原有参数、$L_{in}$、$L_{out}$ 和分类头，避免灾难性遗忘并最小化额外参数

3. **灵活的架构适配：** 对于 Encoder-only 模型，LLM 层插入 encoder 和分类头之间；对于 Encoder-Decoder 模型，LLM 层插入 encoder 和 decoder 之间

### 损失函数

使用标准的任务损失函数：分类任务使用交叉熵损失，生成任务使用自回归语言建模损失，无需额外蒸馏损失。

## 实验

### 主实验：NLU 任务性能

| 模型 | SST2 | IMDB | Tweet(Sent) | Tweet(Off) | CoLA | MNLI | SNLI | SQuAD | Avg |
|------|------|------|------|------|------|------|------|------|------|
| BERT_base | 89.41 | 85.10 | 86.90 | 83.15 | 80.10 | 82.00 | 89.10 | 63.81 | 82.45 |
| +PiFi | **91.50** | **87.09** | **92.95** | **86.03** | **82.07** | **82.74** | **89.48** | **66.17** | **84.75** |
| ELECTRA_base | 93.42 | 88.31 | 90.58 | 83.52 | 83.99 | 85.41 | 90.11 | 44.44 | 82.00 |
| +PiFi | **94.13** | **89.40** | **93.31** | **84.99** | **86.26** | **86.47** | **90.48** | **67.99** | **86.71** |
| DeBERTa-V3_base | 93.74 | 89.45 | 91.29 | 83.60 | 84.75 | 87.52 | 90.94 | 69.40 | 86.34 |
| +PiFi | **95.01** | **89.83** | **93.80** | **85.60** | **86.07** | **87.98** | **91.05** | **69.87** | **87.40** |

**NLG 任务**：T5_base + PiFi 在 Multi30K 翻译上 BLEU 从 0.5301→0.5413，BART_base + PiFi 在 CNN/DailyMail 摘要上 BLEU 从 0.2270→0.2331。

### 消融实验

| 消融维度 | 关键发现 |
|------|------|
| LLM 层位置（第1/16/32层） | 最后一层效果最好，包含最丰富的高层语言知识 |
| 是否冻结 LLM 层 | 冻结效果优于解冻，解冻导致灾难性遗忘 |
| 不同 LLM 来源 | Llama-3.1-8B 整体最优，不同 LLM 均有提升 |
| Instruction-tuned LLM | 使用 instruction-tuned 版本效果类似 |

### 关键发现

1. PiFi 在所有 5 个 SLM 架构上均实现一致提升，BERT_base 平均提升 2.3%，ELECTRA_base 提升 4.7%
2. 跨域泛化实验中，PiFi 在 IMDB→Tweet 上从 70.40 提升到 83.68（+13.28%），展示了强大的领域迁移能力
3. 多语言分类实验表明，使用目标语言预训练的 LLM 层可显著提升 SLM 的多语言能力
4. 仅增加极少量参数（两个线性层），计算开销可忽略不计

## 亮点

- **设计极简却有效**：仅插入一个 LLM 层+两个线性映射，就能显著提升 SLM 性能
- **通用性强**：适用于 Encoder-only 和 Encoder-Decoder 两种架构，覆盖 NLU 和 NLG 多种任务
- **跨域和多语言能力迁移效果突出**：实验表明 PiFi 不仅提升任务内性能，还增强了对未见领域和语言的泛化能力
- **训练高效**：LLM 层冻结训练，额外参数量极少

## 局限性

- 目前仅验证了单层 LLM 层的效果，多层组合或中间层的潜力未被充分探索
- 线性映射层的维度投影可能造成信息损失，更高级的适配网络（如 MLP）可能更好
- 对更大的 SLM（如 1-3B 级别）是否仍有显著提升未充分验证
- 推理阶段需要加载 LLM 的一层参数（如 Llama-8B 一层约 600M），增加存储需求
- 缺少与 LoRA 等主流 PEFT 方法的直接对比

## 相关工作

- **知识蒸馏**：参数化方法（Zhong et al.）使用教师输出分布训练学生，非参数化方法（Ye et al.）使用 LLM 生成合成数据
- **SLM 增强**：继续预训练（Gururangan et al.）、数据增强（Gao et al.）等
- **LLM 跨模态应用**：LLM 层作为视觉编码器（Pang et al.），LLM 重写文本描述增强 CLIP（Fan et al.）

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | 7/10 |
| 有效性 | 8/10 |
| 实验充分度 | 8/10 |
| 写作质量 | 7/10 |
| 总分 | 7.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models](behaviorbox_automated_discovery_of_fine-grained_performance_differences_between_.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](gradot_offsite_tuning.md)
- [\[ACL 2025\] Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA](team_anotheroption_at_semeval-2025_task_8_bridging_the_gap_between_open-source_a.md)

</div>

<!-- RELATED:END -->
