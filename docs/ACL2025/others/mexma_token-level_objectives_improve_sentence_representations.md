---
title: >-
  [论文解读] MEXMA: Token-level Objectives Improve Sentence Representations
description: >-
  [ACL 2025][跨语言句子编码器] 提出 MEXMA，一种结合句子级和 token 级目标的跨语言句子编码器训练方法：用一种语言的句子表示去预测另一种语言的被掩码 token，同时让句子和 token 的梯度都直接更新编码器，在双文本挖掘和多项下游任务上超越 SONAR 和 LaBSE。
tags:
  - ACL 2025
  - 跨语言句子编码器
  - token级目标
  - 句子表示
  - 多语言对齐
  - 掩码语言模型
---

# MEXMA: Token-level Objectives Improve Sentence Representations

**会议**: ACL 2025  
**arXiv**: [2409.12737](https://arxiv.org/abs/2409.12737)  
**代码**: 无  
**领域**: 其他  
**关键词**: 跨语言句子编码器, token级目标, 句子表示, 多语言对齐, 掩码语言模型

## 一句话总结

提出 MEXMA，一种结合句子级和 token 级目标的跨语言句子编码器训练方法：用一种语言的句子表示去预测另一种语言的被掩码 token，同时让句子和 token 的梯度都直接更新编码器，在双文本挖掘和多项下游任务上超越 SONAR 和 LaBSE。

## 研究背景与动机

跨语言句子编码器（CLSE）旨在创建跨语言对齐的定长句子表示。现有方法存在一个核心矛盾：

**预训练阶段使用 token 级目标**：CLSE 通常基于 XLM-RoBERTa、NLLB 等预训练编码器，这些模型通过掩码语言建模（MLM）等 token 级目标更新每个 token 的表示

**微调阶段仅使用句子级目标**：CLSE 训练时（如 LaBSE 的对比学习、SONAR 的翻译瓶颈）仅通过句子表示更新编码器，不再有 token 级目标

**后果**：这导致 token 级信息（尤其是词汇信息）退化，进而影响句子表示质量

**假设**：在 CLSE 训练中保留 token 级目标，配合句子级目标，能更好地更新编码器并提升句子表示质量。

**与现有混合方法的区别**：
- DAP：有 token 级目标但不通过它更新句子表示
- RetroMAE：句子表示用于指导 token 去掩码，但编码器本身不接收来自 token 的直接梯度

## 方法详解

### 整体框架

MEXMA 的架构是对称的：给定两种语言的翻译对，每种语言创建两个视图（一个掩码版本、一个干净版本），共四个编码器实例（共享参数）。核心操作是**跨语言去掩码**：用语言 A 的干净句子表示来预测语言 B 被掩码的 token，反之亦然。

### 关键设计

1. **跨语言去掩码（Cross-Unmasking）**：

    - 对语言 A 的输入进行高比例掩码（40%），使编码器和 MLM head 难以在没有额外上下文的情况下恢复缺失 token
    - 提供语言 B 的干净句子向量 $S_B$ 作为额外上下文，强制模型利用 $S_B$ 中的信息来预测语言 A 中被掩码的 token
    - 对称性操作：同时反向进行（用 $S_A$ 预测语言 B 的掩码 token）
    - 损失函数：$\mathcal{L}_{mlm} = CE([S_B, \hat{A}], A) + CE([S_A, \hat{B}], B)$
    - **关键区别**：梯度同时流经句子表示和各个 token 表示回到编码器

2. **对齐损失（Alignment Loss）**：

    - 跨语言去掩码产生了隐式对齐，但不足以强制相同语义的句子在嵌入空间中足够接近
    - 使用 MSE 损失强制两种语言的句子表示对齐：$\mathcal{L}_{alignment} = MSE(S_A, S_B)$
    - 这是一种非对比式的对齐方法（因为掩码操作防止了表示坍塌）

3. **KoLeo 损失**：

    - 解决表示的各向异性（anisotropy）问题
    - 基于 Kozachenko-Leonenko 差分熵估计器，鼓励句子表示在潜在空间中均匀分布
    - $\mathcal{L}_{KoLeo} = -\frac{1}{n}\sum_{i=1}^n \log(d_{n,i})$，其中 $d_{n,i}$ 是 $x_i$ 与批次中最近邻的距离

### 损失函数 / 训练策略

总损失为三部分的加权组合：
$$\mathcal{L}_{MEXMA} = \alpha \cdot \mathcal{L}_{alignment} + \beta \cdot \mathcal{L}_{mlm} + \gamma \cdot \mathcal{L}_{K}$$

- 编码器基于 XLM-RoBERTa（base: 277M / large: 559M）
- 训练数据来自 NLLB-200 语料库的子集，覆盖 81 种语言（所有数据均为英语与其他 80 种语言的配对）
- 每种语言 15M~25M 句子，对数据量少的语言补充挖掘数据
- 掩码比例 40%（最优区间 30%-60%）

## 实验关键数据

### 主实验

**双文本挖掘（Bitext Mining）**：

| 模型 | xsim ↓ | xsim++ ↓ | BUCC F1 ↑ |
|------|--------|----------|-----------|
| DAP | — | — | 98.68 |
| SONAR (766M) | 0.09 | 12.08 | 98.25 |
| LaBSE (471M) | 0.92 | 18.65 | 98.75 |
| **MEXMA (559M)** | **0.06** | **9.60** | **98.93** |

xsim++ 绝对提升 2.48%（vs SONAR），表明对困难负样本的鲁棒性显著增强。

**分类任务**：

| 模型 | SentEval | MTEB 平均 |
|------|----------|-----------|
| SONAR | 85.82 | 63.02 |
| LaBSE | 85.63 | 62.77 |
| **MEXMA** | **86.38** | **65.35** |

**配对分类（Average Precision）**：

| 模型 | 平均 AP |
|------|---------|
| SONAR | 69.70 |
| LaBSE | 68.47 |
| **MEXMA** | **71.55** |

### 消融实验

| 配置 | xsim ↓ | xsim++ ↓ | SentEval ↑ |
|------|--------|----------|-----------|
| 仅句子级梯度 | 0.15 | 11.37 | 85.06 |
| + Token 级梯度 | 0.10 (↓0.05) | 9.67 (↓1.7) | 85.98 (↑0.92) |
| + KoLeo (完整 MEXMA) | 0.06 (↓0.04) | 9.60 (↓0.07) | 86.38 (↑0.4) |

**模型规模消融**：

| 模型 | 参数量 | xsim++ ↓ | SentEval ↑ |
|------|--------|----------|-----------|
| MEXMA-base | 277M | 13.03 | 85.30 |
| LaBSE | 471M | 18.65 | 85.63 |
| MEXMA | 559M | 9.60 | 86.38 |
| SONAR | 766M | 12.08 | 85.82 |

MEXMA-base (277M) 以 LaBSE 58.8% 的参数量即超越 LaBSE，并接近 SONAR（2.77 倍参数）。

### 关键发现

1. **Token 级梯度是核心贡献**：从仅句子级梯度到加入 token 级梯度，xsim++ 降低 1.7 个百分点，是最大的单一改进因素
2. **小模型也强大**：MEXMA-base (277M) 的 xsim++ 已达 13.03%，显著优于 LaBSE (471M) 的 18.65%
3. **与对比学习兼容**：将 MSE 对齐损失替换为对比损失后，MEXMA 的 token 级梯度仍然带来显著提升
4. **Token 嵌入分析**：MEXMA 的 token 展现出强跨语言语义对齐（97.88% 匹配到翻译），同时保留更多词汇信息（1.33% same language vs SONAR 的 0.13%）
5. **STS 任务是例外**：唯一 LaBSE 优于 MEXMA 的任务，表明对比损失更适合 STS

## 亮点与洞察

- **设计简洁而有效**：核心想法非常直觉——训练句子编码器时不应只通过句子更新编码器，还要通过 token 级目标直接更新。实现上只需要让梯度流经 token 即可
- **对称跨语言去掩码**：巧妙地同时实现了两件事——迫使句子向量编码充分信息（用于去掩码）、保持 token 表示质量（直接接收梯度）
- **KoLeo 防各向异性**：从视觉模型（DINOv2）借鉴的想法，用于解决非对比方法的各向异性问题
- **Token 近邻分析**：通过分析 token 嵌入的最近邻类别（同语言/同句子/翻译/其他），直观展示了不同模型 token 表示的特性差异

## 局限与展望

1. **STS 任务表现不佳**：非对比式对齐在语义文本相似度任务上不如对比方法
2. **仅支持 81 种语言**：相比 SONAR 的 200 种语言覆盖面较小
3. **训练数据依赖**：大量依赖 NLLB 的挖掘数据，低资源语言的数据质量可能参差不齐
4. **未探索生成式下游任务**：主要在分类和挖掘任务上评估，生成任务的效果未知
5. **掩码比例的自适应**：当前使用固定 40% 掩码率，不同语言对/句子长度可能有不同的最优值

## 相关工作与启发

- **与 RetroMAE 的关系**：RetroMAE 首先提出用句子表示指导 token 去掩码的 IR 方法，但编码器不接收 token 级梯度；MEXMA 在此基础上让梯度双向流动
- **与 SONAR 的关系**：SONAR 使用翻译瓶颈实现对齐，但瓶颈阻止了解码器梯度直接更新 token；MEXMA 无此限制
- **启发方向**：token 级和句子级目标的协同训练范式可推广到其他需要层级表示的任务（如文档检索、段落表示）

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心思想简洁深刻——让梯度同时从 token 和句子两个层级流动
- 实验充分度: ⭐⭐⭐⭐⭐ 多个基准（xsim/xsim++/BUCC/MTEB/SentEval）、完整消融、模型规模分析、token 嵌入分析
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，实验详实，分析深入到 token 级别
- 价值: ⭐⭐⭐⭐⭐ 在多个关键基准上建立新 SOTA，且方法可与对比学习组合使用，实用性强

<!-- RELATED:START -->

## 相关论文

- [Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)
- [LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)
- [Cautious Next Token Prediction](cautious_next_token_prediction.md)
- [Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)
- [TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](trove_a_challenge_for_finegrained_text.md)

<!-- RELATED:END -->
