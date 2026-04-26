---
title: >-
  [论文解读] When Claims Evolve: Evaluating and Enhancing the Robustness of Embedding Models Against Misinformation Edits
description: >-
  [ACL 2025 Findings][事实核查] 提出一个扰动框架来系统评估句子嵌入模型在处理经过编辑的虚假信息声明时的鲁棒性，发现标准嵌入模型性能显著下降，并通过知识蒸馏和声明规范化两种缓解方法将领域内鲁棒性提升最高 17 个百分点、跨域泛化提升 10 个百分点。
tags:
  - ACL 2025 Findings
  - 事实核查
  - 声明匹配
  - 嵌入模型鲁棒性
  - 虚假信息变体
  - 检索系统
---

# When Claims Evolve: Evaluating and Enhancing the Robustness of Embedding Models Against Misinformation Edits

**会议**: ACL 2025 Findings  
**arXiv**: [2503.03417](https://arxiv.org/abs/2503.03417)  
**代码**: https://github.com/JabezNzomo99/claim-matching-robustness  
**领域**: NLP理解  
**关键词**: 事实核查、声明匹配、嵌入模型鲁棒性、虚假信息变体、检索系统

## 一句话总结

提出一个扰动框架来系统评估句子嵌入模型在处理经过编辑的虚假信息声明时的鲁棒性，发现标准嵌入模型性能显著下降，并通过知识蒸馏和声明规范化两种缓解方法将领域内鲁棒性提升最高 17 个百分点、跨域泛化提升 10 个百分点。

## 研究背景与动机

**领域现状**：事实核查员越来越依赖自动化的"声明匹配"系统——当一条新的声明出现时，系统使用句子嵌入模型在已核查声明数据库中检索最相关的 fact-check，帮助核查员快速判断该声明是否已被验证过。这个管线通常包括两阶段：第一阶段用嵌入模型做初步检索（recall），第二阶段用重排序器（reranker）精排。

**现有痛点**：网络用户在传播虚假信息时经常对原始声明进行各种编辑——改写措辞（rewrite）、添加否定（negation）、引入拼写错误（typos）、夸大或淡化语气（amplify/minimize）、替换实体名（entity replacement）甚至转换为方言（dialect）。这些编辑后的声明在语义上仍指向同一条 fact-check，但嵌入模型能否在这些变体上保持鲁棒的检索性能尚不清楚。

**核心矛盾**：嵌入模型被训练为将语义相似的句子映射到向量空间中相近的位置，但微妙的文本编辑（如加个否定词、换种方言表达）可能导致向量偏移，使得原本应该匹配的声明在向量空间中"漂移"到无法检索的位置。

**本文目标**：(1) 设计一个系统的扰动框架来生成合理且自然的声明变体；(2) 评估主流嵌入模型在多种编辑类型下的鲁棒性；(3) 提出训练时和推理时的缓解方法来提升鲁棒性。

**切入角度**：利用 GPT-4o 作为"扰动器"生成声明变体，并通过验证步骤确保变体仍指向同一 fact-check，然后在这些经过验证的变体上系统性地评测嵌入模型。

**核心 idea**：用 LLM 自动生成多类型的声明编辑变体来压力测试嵌入模型，然后用知识蒸馏训练更鲁棒的嵌入模型，以及用 LLM 做推理时的声明规范化来消除噪声。

## 方法详解

### 整体框架

分为三个阶段：(1) 扰动生成——用 GPT-4o 对原始声明进行多类型编辑并验证；(2) 鲁棒性评估——在多阶段检索管线（嵌入检索 + 重排序）中评测多种模型；(3) 缓解方法——分别在训练时（知识蒸馏）和推理时（声明规范化）提升鲁棒性。评估使用 CheckThat22、FactCheckTweet 两个领域内数据集和一个 OOD 数据集。

### 关键设计

1. **LLM 驱动的扰动框架**:

    - 功能：自动且系统地生成多类型的、语义保持的声明变体
    - 核心思路：使用 GPT-4o 对每条声明生成 5 种编辑变体，编辑类型包括改写（rewrite）、否定（negation）、拼写错误（typos）、夸大/淡化（amplify/minimize）、实体替换（entity replacement）和方言转换（dialect）。生成后再用 GPT-4o 做验证——检查每个变体是否仍对应原始 fact-check、是否传达相同核心声明、是否读起来自然。通过验证的变体被保留，然后从中选择"最小编辑"（baseline）和"最大编辑"（worst-case）两个子集用于评估。
    - 设计动机：手动编写声明变体不可扩展，且难以系统覆盖所有编辑类型。LLM 生成 + LLM 验证的两步流程既保证了多样性又保证了质量。区分 baseline 和 worst-case 能够评估模型在不同编辑程度下的表现。

2. **多阶段检索管线鲁棒性评估**:

    - 功能：全面评测嵌入模型在实际部署场景中的鲁棒性
    - 核心思路：评估采用两阶段检索管线。第一阶段使用 10+ 种嵌入模型（包括 all-mpnet-base-v2、all-MiniLM-L12-v2、sentence-t5、INSTRUCTOR、NV-Embed-v2、SFR-Embedding-Mistral 等）检索 top-k 候选，第二阶段使用 BGE-LLM 等重排序器精排。分别对原始声明和编辑后声明计算 MAP@k、Recall@k 等指标，用性能差值（$\Delta$）衡量鲁棒性。还包括 BM25 作为稀疏检索的 baseline。
    - 设计动机：单独测嵌入模型不够——需要在完整管线中评估，因为重排序器可能弥补第一阶段的损失。实际使用中这两个阶段通常串联部署。

3. **训练时缓解：鲁棒性知识蒸馏**:

    - 功能：通过微调使嵌入模型对声明编辑更鲁棒
    - 核心思路：使用扰动框架生成的"原始-变体"声明对作为训练数据，对嵌入模型进行知识蒸馏微调。教师模型在原始声明上的嵌入作为目标，学生模型在编辑后声明上的嵌入尽量逼近教师的输出。训练集包含约 70K 声明对（或 lite 版本 11.5K 对）。
    - 设计动机：直接在编辑声明上微调可能导致过拟合特定编辑类型，而知识蒸馏让模型学习"忽略表面编辑，保持语义不变"的能力，泛化性更好。

### 推理时缓解：声明规范化

在推理时，使用 GPT-4o 将输入的噪声声明（包含拼写错误、方言等）规范化为标准形式，然后再送入检索管线。这种方法不需要修改模型，只需在输入端加一个预处理步骤。

## 实验关键数据

### 主实验

| 嵌入模型 | 原始 MAP@5 | 改写后 MAP@5 | Δ (pp) | 重排序后 Δ |
|---------|-----------|-------------|--------|-----------|
| all-mpnet-base-v2 | 78.4 | 64.2 | -14.2 | -8.7 |
| all-MiniLM-L12-v2 | 74.8 | 59.1 | -15.7 | -9.3 |
| sentence-t5-large | 76.2 | 63.5 | -12.7 | -7.8 |
| INSTRUCTOR-large | 79.1 | 68.3 | -10.8 | -6.5 |
| NV-Embed-v2 | 83.6 | 75.1 | -8.5 | -4.2 |
| SFR-Embedding-Mistral | 85.2 | 78.4 | -6.8 | -3.5 |
| BM25 | 62.3 | 48.7 | -13.6 | - |

### 缓解方法效果

| 方法 | 领域内 Δ 改善 (pp) | 跨域 Δ 改善 (pp) | 说明 |
|-----|-------------------|-----------------|------|
| 无缓解 (baseline) | 0 | 0 | 原始性能下降 |
| 知识蒸馏 (full, 70K) | +17.0 | +10.0 | 训练时方法，效果最好 |
| 知识蒸馏 (lite, 11.5K) | +12.3 | +7.2 | 轻量版训练数据也有效 |
| 声明规范化 (GPT-4o) | +9.5 | +6.8 | 推理时方法，无需重训练 |
| 重排序器 (BGE-LLM) | +5.8 | +3.4 | 能缓解但不能完全弥补 |

### 关键发现

- LLM 蒸馏的嵌入模型（如 NV-Embed-v2、SFR-Embedding-Mistral）天然比传统嵌入模型更鲁棒，但计算成本更高
- 重排序器能部分弥补第一阶段的性能损失，但无法完全补偿——如果第一阶段 recall 就漏掉了正确结果，重排序器也无能为力
- 知识蒸馏是最有效的缓解方法，能在领域内提升 17pp，跨域提升 10pp
- 不同编辑类型中，否定（negation）对嵌入模型的冲击最大，因为它直接改变了语义方向；方言编辑（dialect）的影响也出人意料地大

## 亮点与洞察

- **将 LLM 同时用于攻击（生成扰动）和防御（声明规范化）**是非常巧妙的设计——用同一个工具构建了评测和解决的完整架构
- 知识蒸馏方法简单但效果显著，它的训练数据可以用扰动框架自动生成，形成了一个自给自足的"生成扰动→训练鲁棒模型"闭环
- "重排序器无法完全弥补第一阶段损失"这个发现很有实践意义——说明在部署事实核查系统时，不能只关注 reranker 的精度，第一阶段检索的鲁棒性同样至关重要

## 局限与展望

- 扰动生成依赖 GPT-4o，成本较高且受模型能力限制
- 只评估了英文声明，多语言场景下声明编辑的模式可能不同
- OOD 数据集只有一个来源（Meedan WhatsApp Tiplines），跨域泛化结论需要更多验证
- 知识蒸馏需要为每个嵌入模型单独训练，部署成本随模型数量线性增长

## 相关工作与启发

- **vs Adversarial NLI/TextFooler**: 这些工作关注对抗性攻击文本分类模型，本文关注的是检索系统的鲁棒性——对抗目标不同（分类翻转 vs 检索失败），扰动方式也更贴近真实用户行为
- **vs CheckThat Lab**: CheckThat 系列评测已有声明匹配任务，但未考虑声明编辑带来的鲁棒性问题，本文填补了这一空白
- **vs Sentence-BERT AdvTraining**: 以往对嵌入模型的对抗训练主要关注同义替换，本文系统覆盖了 6+ 种编辑类型，评估更全面

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统评估声明匹配场景下嵌入模型对多种编辑的鲁棒性
- 实验充分度: ⭐⭐⭐⭐⭐ 10+ 种嵌入模型、6 种编辑类型、3 个数据集、多种缓解方法，非常全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验设计系统，结论有实践指导意义
- 价值: ⭐⭐⭐⭐ 对事实核查系统的实际部署有直接帮助，代码和数据公开

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](../../NeurIPS2025/information_retrieval/worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Semantic Outlier Removal with Embedding Models and LLMs](semantic_outlier_removal_with_embedding_models_and_llms.md)
- [\[ACL 2025\] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [\[ACL 2025\] Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models](sticking_to_the_mean_detecting_sticky_tokens_in_text_embedding_models.md)

<!-- RELATED:END -->
