---
title: >-
  [论文解读] Partial Colexifications Improve Concept Embeddings
description: >-
   首次将部分共词化（affix/overlap colexification）引入概念嵌入训练，在语义相似性建模、语义变化预测和词语联想预测三个任务上均优于仅使用完全共词化的基线。
tags:

---

# Partial Colexifications Improve Concept Embeddings

- **会议**: ACL 2025
- **arXiv**: [2502.09743](https://arxiv.org/abs/2502.09743)
- **代码**: [GitHub](https://github.com/calc-project/concept-embeddings)
- **领域**: 计算语言学 / 词汇语义学
- **关键词**: 概念嵌入, 共词化, 部分共词化, 图嵌入, 跨语言语义, Node2Vec, ProNE

## 一句话总结

首次将部分共词化（affix/overlap colexification）引入概念嵌入训练，在语义相似性建模、语义变化预测和词语联想预测三个任务上均优于仅使用完全共词化的基线。

## 研究背景与动机

- **核心问题**: 概念嵌入（concept embeddings）为跨语言 NLP 任务提供了语言无关的语义表示，但现有方法仅利用完全共词化（full colexification，即同一词形表达两个义项）来构建共词化网络，忽略了词的部分（如词缀、词干共享）所蕴含的语义关系。
- **现有方法局限**:
    - **仅用完全共词化**: Harvill et al. 2022 和 Chen et al. 2023 仅基于完全共词化的图嵌入，遗漏了词法派生/合成中隐含的概念关系（如"树"和"树皮"通过共享词根相关联，但极少完全共词化）。
    - **自动构建的共词化图噪声大**: Liu et al. 2023 从平行语料自动推断共词化，但质量不如手工标注数据。
    - **覆盖范围有限**: 任何单一语言的共词化数量很少，需要跨语言聚合才能构建有意义的网络。
- **本文动机**: 利用 List (2023) 提出的部分共词化推断方法（affix + overlap），在手工标注的 IDS 数据集（329 种语言）上构建更丰富的共词化网络，训练更好的概念嵌入。

## 方法详解

### 整体框架

从 IDS 数据集（329 种语言、1,310 个概念）中推断三种共词化网络：完全共词化（相同词形）、词缀共词化（一词是另一词的前/后缀）、重叠共词化（共享子串）。分别用三种图嵌入方法（SDNE、Node2Vec、ProNE）学习 128 维概念向量，通过 PCA 拼接组合不同类型的嵌入。

### 关键设计

1. **三种共词化网络**: 完全共词化（1,246 节点 / 4,008 边）、词缀共词化（1,308 节点 / 38,215 边，有方向性但转为无向图）、重叠共词化（926 节点 / 12,974 边）。边权按语言族数加权，跨越更多语言族的共词化权重更高。
2. **嵌入组合策略**: 分别在各类型网络上训练嵌入，然后拼接 + PCA 降维回 128 维。测试了 6 种组合：full / affix / overlap / full+affix / full+overlap / full+affix+overlap。
3. **三项评估任务**: (a) 语义相似性建模（Multi-SimLex 538 对，Spearman 相关性）；(b) 语义变化预测（DatSemShift 547 对，逻辑回归二分类）；(c) 词语联想预测（EAT 780 条边，链接预测二分类）。

### 损失函数

- **SDNE**: 自编码器重构损失 + 一阶/二阶邻域保持损失
- **Node2Vec**: 基于随机游走的 Skip-gram (Word2Vec) 目标
- **ProNE**: 稀疏矩阵分解 + 谱传播

## 实验

### 主实验结果

**任务 (a) 语义相似性（Spearman 相关系数）**:

| 方法 | full | affix | full+affix | full+affix+overlap |
|------|------|-------|-----------|-------------------|
| ProNE | 0.64 | 0.63 | **0.72** | 0.66 |
| Node2Vec | 0.64 | 0.58 | 0.69 | 0.66 |
| fastText-ZH (best) | 0.44 | — | — | — |

**任务 (b) 语义变化预测（准确率）**:

| 方法 | full | full+affix | full+affix+overlap |
|------|------|-----------|-------------------|
| Node2Vec | 0.79 | **0.83** | 0.82 |
| ProNE | 0.78 | 0.82 | **0.83** |
| fastText-ET (best) | 0.82 | — | — |

**任务 (c) 词语联想预测（准确率）**:

| 方法 | full | full+affix | full+affix+overlap |
|------|------|-----------|-------------------|
| ProNE | 0.71 | 0.80 | **0.81** |
| Node2Vec | 0.71 | 0.78 | 0.79 |
| fastText-EN (best) | 0.87 | — | — |

### 消融分析（不同共词化类型的贡献）

| 共词化类型 | 语义相似性 | 语义变化预测 | 联想预测 |
|-----------|-----------|------------|---------|
| full alone | 0.64 | 0.79 | 0.71 |
| +affix | **0.72 (+0.08)** | **0.83 (+0.04)** | 0.80 (+0.09) |
| +overlap | 0.62 (-0.02) | 0.80 (+0.01) | 0.77 (+0.06) |
| +affix+overlap | 0.66 (+0.02) | 0.83 (+0.04) | **0.81 (+0.10)** |

### 关键发现

1. **词缀共词化是最有价值的补充信息**: full+affix 在所有三个任务上均显著优于 full alone，相关系数提升 0.08，准确率提升 4-9 个百分点。
2. **重叠共词化价值因任务而异**: 在语义相似性任务上有害（-0.02），但在联想预测上有益（+0.06），可能因为重叠共词化捕捉的是更远的语义关联。
3. **概念嵌入在语义相似性上远超词嵌入**: ProNE full+affix (0.72) vs fastText best (0.44)，说明跨语言共词化比单语言分布信息更好地捕捉语义相似性。
4. **词嵌入在联想预测上仍有优势**: fastText-EN (0.87) > ProNE best (0.81)，因为 EAT 是英语单语联想数据，词嵌入的分布信息更直接。
5. **SDNE 表现最差**: 在所有任务上均不如 Node2Vec 和 ProNE，不适合嵌入共词化图。

## 亮点

- 首次将部分共词化（词缀+重叠）引入概念嵌入，提出了系统性的评估框架
- 基于手工标注的高质量跨语言数据（329 种语言、IDS），比自动推断的共词化更可靠
- 概念嵌入在跨语言语义相似性建模上大幅超越传统词嵌入

## 局限性

- 仅覆盖约 1,000 个核心概念，受限于基本词汇的比较词表
- 图中孤立节点（无共词化的概念）无法嵌入
- 重叠共词化的作用机制尚不清晰，需更多研究
- 评估任务主要基于英语和少数高资源语言，未直接验证低资源语言的效用

## 相关工作

- **概念嵌入**: Harvill et al. 2022 (BabelNet 图嵌入), Chen et al. 2023 (Colex2Lang), Liu et al. 2023
- **共词化研究**: François 2008, List 2023 (部分共词化), CLICS 数据库
- **图嵌入**: Node2Vec (Grover & Leskovec 2016), ProNE (Zhang et al. 2019), SDNE (Wang et al. 2016)
- **词嵌入**: Word2Vec, fastText, GloVE

## 评分

- **创新性**: ⭐⭐⭐⭐ — 部分共词化用于概念嵌入是自然但未被探索的方向
- **实用性**: ⭐⭐⭐ — 概念集有限，直接 NLP 应用需等覆盖扩展
- **严谨性**: ⭐⭐⭐⭐⭐ — 三个互补评估任务 + 多种基线 + 负采样均衡设计
- **综合**: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport](../../ICML2025/others/theoretical_performance_guarantees_for_partial_domain_adaptation_via_partial_opt.md)
- [\[ACL 2025\] Better Embeddings with Coupled Adam](better_embeddings_with_coupled_adam.md)
- [\[ACL 2025\] Synthia: Novel Concept Design with Affordance Composition](synthia_novel_concept_design_with_affordance_composition.md)
- [\[ACL 2025\] MEXMA: Token-level Objectives Improve Sentence Representations](mexma_token-level_objectives_improve_sentence_representations.md)
- [\[ACL 2025\] Enhancing the Comprehensibility of Text Explanations via Unsupervised Concept Discovery](enhancing_the_comprehensibility_of_text_explanations_via_unsupervised_concept_di.md)

<!-- RELATED:END -->
