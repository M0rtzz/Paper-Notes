---
title: >-
  [论文解读] Enhancing Lexicon-Based Text Embeddings with Large Language Models
description: >-
  [ACL 2025][词汇嵌入] 提出 LENS 框架，首次将 LLM 用于通用词汇级文本嵌入（lexicon-based embedding），通过 token 嵌入聚类解决 LLM 词表冗余问题、引入双向注意力克服因果 LLM 的限制，在 MTEB 上超越同数据训练的稠密嵌入，且与稠密嵌入结合后在 BEIR 上达到 SOTA。
tags:
  - ACL 2025
  - 词汇嵌入
  - LLM
  - 文本嵌入
  - token聚类
  - 稀疏检索
---

# Enhancing Lexicon-Based Text Embeddings with Large Language Models

**会议**: ACL 2025  
**arXiv**: [2501.09749](https://arxiv.org/abs/2501.09749)  
**代码**: https://github.com/Yibin-Lei/LENS  
**领域**: 信息检索 / 文本嵌入  
**关键词**: 词汇嵌入, LLM, 文本嵌入, token聚类, 稀疏检索

## 一句话总结

提出 LENS 框架，首次将 LLM 用于通用词汇级文本嵌入（lexicon-based embedding），通过 token 嵌入聚类解决 LLM 词表冗余问题、引入双向注意力克服因果 LLM 的限制，在 MTEB 上超越同数据训练的稠密嵌入，且与稠密嵌入结合后在 BEIR 上达到 SOTA。

## 研究背景与动机

**领域现状**：文本嵌入是检索、语义相似度、分类等任务的基础。当前主流是稠密嵌入（dense embedding），将文本编码为低维实值向量。基于词汇的稀疏嵌入（lexicon-based embedding，如 SPLADE）虽然在检索任务上表现优越（匹配性好、可解释性强），但研究主要限于 BERT 量级的 MLM 模型，未在 LLM 时代得到充分探索。

**现有痛点**：直接将 LLM 用于词汇嵌入面临两个关键挑战：(1) LLM 词表冗余严重——子词分词导致语义等价 token 以多种形式出现（如 "what"/"What"/" what"），破坏词汇匹配一致性；(2) 因果 LLM 使用单向注意力，每个 token 只能看到前面的 token，而词汇嵌入需要从所有 token 的输出中聚合信息。

**核心矛盾**：LLM 的能力在不断增强，但其架构特性（子词词表 + 单向注意力）阻碍了高质量词汇级嵌入的生成。

**本文目标** 设计一个框架让 LLM 产生紧凑、高效的通用词汇嵌入，性能媲美或超越稠密嵌入。

**切入角度**：从"修改 LLM 的语言建模头和注意力机制"入手，而非使用 prompt 工程等外部方案。

**核心 idea**：通过 KMeans 聚类合并语义相似的 token 来降维去噪，配合双向注意力和 max-pooling，让 LLM 生成与稠密嵌入同维度的高质量词汇嵌入。

## 方法详解

### 整体框架

LENS 基于 Mistral-7B 构建，核心修改三个方面：(1) 用 KMeans 聚类将 32K 词表压缩到 4000/8000 个语义簇，簇中心替换 LM head 中的 token 嵌入；(2) 将单向注意力改为双向注意力；(3) 对所有 token 的 logits 做 log-saturation + max-pooling 得到最终嵌入。训练严格复用 BGE-en-ICL 的公开数据和流程。

### 关键设计

1. **Token 嵌入聚类（词表压缩）**:

    - 功能：对 LM head 中的 token 嵌入矩阵做 KMeans 聚类，将 ~32K 个 token 合并为 $k$ 个语义簇（$k$ = 4000 或 8000），用簇中心嵌入替换原始 token 嵌入
    - 核心思路：聚类后，logits 表示的是对簇的打分而非对单个 token 的打分，从而消除了词表冗余（如 "What"/"what"/" what" 被合并到同一簇）并大幅降低嵌入维度。输入端 token 嵌入保持不变，只修改输出端的 LM head
    - 设计动机：直接使用 32K 维稀疏嵌入在非检索任务（聚类、分类）上效率低下，且 FAISS 等现有框架不支持高维稀疏向量。聚类后 4000 维的嵌入可直接用于 dense 框架中
    - 聚类质量示例：{"quickly", "rapid", "rapidly", "swift"} 被分到同一簇；{"cannot", "impossible", "Unable"} 被分到同一簇

2. **双向注意力机制**:

    - 功能：在微调阶段将 LLM 的因果注意力掩码替换为全连接注意力（双向）
    - 核心思路：词汇嵌入需要对所有 token 的输出做 max-pooling，单向注意力导致前面的 token 无法利用后面的上下文信息，严重限制了嵌入质量
    - 设计动机：与稠密嵌入文献中"保持原始单向注意力通常最优"的结论相反，对词汇嵌入而言双向注意力是关键——实验证明双向比单向在所有 pooling 策略下平均高 3+ 点

3. **表示生成与 Pooling**:

    - 功能：将 logits 通过 log-saturation 变换后做 max-pooling 得到最终嵌入
    - 核心思路：log-saturation 变换 $w_{ij} = \log(1 + \text{ReLU}(l_{ij}))$ 将权重压缩到非负范围；max-pooling 在序列维度上取最大值 $w_j = \max_{i} w_{ij}$，只保留每个簇在整个文本中的最强信号。对 query 部分只使用原始 query token（排除 task instruction token）；并做 logit 位移（每个 token 用左邻居的 logit）以适应自回归特性
    - 设计动机：max-pooling 在词汇嵌入中优于 sum-pooling 和 last-token pooling（实验验证），因为它天然实现了稀疏化——只有最相关的簇有大权重

### 损失函数 / 训练策略

- 使用 InfoNCE 对比学习损失：$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, p)/\tau)}{\exp(\text{sim}(q, p)/\tau) + \sum_j \exp(\text{sim}(q, p_j^-)/\tau)}$，温度 $\tau = 0.02$
- 检索任务额外使用 KL 散度蒸馏 BGE-reranker 的排序分数
- LoRA 微调（rank=32, alpha=64），学习率 1e-4，训练 1 epoch
- 各类任务的 batch size：检索 512，其余 256

## 实验关键数据

### 主实验：MTEB（56 数据集，7 类任务）

| 模型 | 维度 | 检索 | 重排 | 聚类 | 配对分类 | 分类 | STS | 平均 |
|------|------|------|------|------|---------|------|-----|------|
| BGE-en-ICL (稠密) | 4096 | 61.67 | 59.66 | 57.51 | 86.93 | 88.62 | 83.74 | 71.24 |
| NV-Embed-v2 (稠密) | 4096 | 62.65 | 60.65 | 58.46 | 88.67 | 90.37 | 84.31 | 72.31 |
| **LENS-4000** | 4000 | 60.76 | 60.86 | 57.92 | 87.93 | 88.13 | 84.35 | 71.22 |
| **LENS-8000** | 8000 | 61.86 | 60.91 | 58.02 | 87.98 | 88.43 | 84.67 | **71.63** |

LENS-8000 在公开数据训练的模型中 MTEB 平均分最高，在 7 类任务中 6 类超越稠密对应物 BGE-en-ICL。

### 消融实验：注意力 + Pooling 组合

| 注意力 | Pooling | 检索 | 聚类 | 分类 | STS | 平均 |
|--------|---------|------|------|------|-----|------|
| 单向 | Last-token | 73.84 | 60.46 | 58.66 | 89.26 | 67.73 |
| 单向 | Max-pooling | 75.18 | 50.93 | 57.58 | 82.74 | 64.15 |
| **双向** | **Max-pooling** | **76.19** | **63.05** | **62.30** | **88.92** | **69.07** |

双向注意力 + max-pooling 是最优组合，平均比单向 last-token 高 1.34 点。

### 关键发现

- LENS 证明词汇嵌入在 LLM 时代可以全面媲美甚至超越稠密嵌入——这打破了"稠密嵌入在通用任务上更优"的固有认知
- 聚类数从 32K（原始词表）降到 8K 和 4K 时性能反而提升，说明词表去噪本身就有增益
- Top-K 剪枝（仅保留 256/4000 维）几乎不损失性能，天然支持嵌入压缩，无需 Matryoshka 训练
- LENS 与稠密嵌入结合后在 BEIR 检索子集上达到 SOTA，验证了词汇嵌入和稠密嵌入的互补性

## 亮点与洞察

- 首次证明 LLM 可以生成通用的高质量词汇嵌入，而非仅限于检索任务——这为嵌入研究打开了新的方向
- Token 嵌入聚类是一个简洁优雅的设计，一举解决词表冗余、维度爆炸和匹配不一致三个问题
- 双向 vs 单向注意力的实验结果与稠密嵌入文献的结论截然相反，揭示了词汇嵌入和稠密嵌入在架构需求上的根本差异
- 定性分析展示了 LENS 的深层语义理解能力——例如对 "causes of hypoxia in adults" 生成 "oxygen" 簇权重最高

## 局限与展望

- LENS-4000 在部分任务（如 AIR-Bench）上仍落后于稠密嵌入，簇数过少可能导致信息过度压缩
- 仅基于 Mistral-7B 实验，其他 LLM backbone 的效果未知
- KMeans 聚类是静态的，不同任务可能需要不同的聚类粒度
- 未探索多语言场景，LLM 词表的多语言冗余问题可能更严重

## 相关工作与启发

- **vs SPLADE**: SPLADE 基于 MLM（BERT）生成词汇嵌入并在检索上表现优异，但从未扩展到通用任务；LENS 证明 LLM 可以做到更好且更通用
- **vs PromptReps**: PromptReps 用 prompt 工程让 LLM 生成词汇嵌入，但性能远逊于稠密嵌入（MRR 34.15 vs 41.86）；LENS 通过架构修改而非 prompt 工程取得了本质的突破
- **vs BGE-en-ICL**: 使用完全相同的数据和训练流程，LENS-8000 的词汇嵌入在 MTEB 上超越了 BGE-en-ICL 的稠密嵌入

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将 LLM 用于通用词汇嵌入，token 聚类的设计简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ MTEB（56数据集）+ AIR-Bench + 详细消融（聚类数/注意力/pooling），极为充分
- 写作质量: ⭐⭐⭐⭐ 背景梳理清晰，实验设计公平（严格复用 BGE-en-ICL 配置）
- 价值: ⭐⭐⭐⭐ 证明了词汇嵌入在 LLM 时代的竞争力，对嵌入研究具有方向性意义

<!-- RELATED:START -->

## 相关论文

- [Redundancy, Isotropy and Intrinsic Dimensionality of Prompt-Based Text Embeddings](redundancy_isotropy_and_intrinsic_dimensionality_of_prompt-based_text_embeddings.md)
- [RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](rare_retrieval_augmented_reasoning.md)
- [Drama: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers](drama_diverse_augmentation_from_large_language_models_to_smaller_dense_retriever.md)
- [Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)

<!-- RELATED:END -->
