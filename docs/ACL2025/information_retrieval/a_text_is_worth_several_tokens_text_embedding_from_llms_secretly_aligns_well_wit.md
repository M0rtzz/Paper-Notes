---
title: >-
  [论文解读] A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens
description: >-
  [ACL 2025][Text Embedding] 揭示 LLM 文本嵌入的有趣现象：将嵌入向量通过解码层映射回词表空间后，解码概率最高的 token 与输入文本的关键词高度对齐；进一步通过谱分析发现这一现象主要受第一主成分控制，并据此提出一种简洁的稀疏检索方法，达到原密集检索 80%+ 的效果。
tags:
  - ACL 2025
  - Text Embedding
  - LLM
  - Token Alignment
  - Sparse Retrieval
  - Spectral Analysis
---

# A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens

**会议**: ACL 2025  
**arXiv**: [2406.17378](https://arxiv.org/abs/2406.17378)  
**代码**: [https://github.com/Arthurizijar/Text_aligns_tokens](https://github.com/Arthurizijar/Text_aligns_tokens)  
**领域**: NLP / Text Embedding  
**关键词**: Text Embedding, LLM, Token Alignment, Sparse Retrieval, Spectral Analysis

## 一句话总结

揭示 LLM 文本嵌入的有趣现象：将嵌入向量通过解码层映射回词表空间后，解码概率最高的 token 与输入文本的关键词高度对齐；进一步通过谱分析发现这一现象主要受第一主成分控制，并据此提出一种简洁的稀疏检索方法，达到原密集检索 80%+ 的效果。

## 研究背景与动机

基于 LLM 的文本嵌入方法已在信息检索、语义相似度等任务上取得优秀表现。现有方法通常将 LLM 分为两部分：特征提取模块 $f$ 和解码层 $g$，将 $f$ 转化为嵌入器 $\hat{f}$（通过 prompt engineering 或对比学习），丢弃 $g$，用池化策略得到文本嵌入。

作者发现一个**有趣现象**：将 $\hat{f}$ 得到的文本嵌入重新送入 $g$（原始 LLM 的解码层），**解码概率最高的 token 与输入文本的关键词高度相关**。例如输入 "What diseases are parrots prone to?"，解码出 "disease"、"birds"、"suscept" 等语义相关的 token。

这一现象具有**普遍性**，不受模型架构（GPT-Neo, OPT, LLaMA, Mistral）、训练策略（prompt engineering, 对比学习）和嵌入方法（last pooling, mean pooling）的影响。

## 方法详解

### 整体框架

**分析框架**：给定文本 $s_i$，通过 tokenizer 获取字面 token 集合 $T_{s_i}$，通过嵌入器 $\hat{f}$ 获取文本嵌入 $\mathbf{h}_i$，将 $\mathbf{h}_i$ 与解码层 token 嵌入矩阵 $\mathbf{E}_g$ 做点积，按分数降序排列获得对齐 token 集 $\hat{T}_{s_i}^K$。

### 关键设计

**1. 对齐现象的定性与定量分析**

- 8 个 LLM 嵌入器全面分析，包括 SGPT、OPT-EOL、LLaMA-EOL、GritLM、LLM2Vec 等
- 三个量化指标：
    - **Hit@K**：top-K 对齐 token 中是否包含输入文本 token（命中率）
    - **LAR (Local Alignment Rate)**：top-$|T_{s_i}|$ 对齐 token 与输入 token 的重叠比例
    - **GAR (Global Alignment Rate)**：数据集级别的全局重叠比例
- 关键发现：
    - 原始 $f$ 也能对齐某些 token，但多为无意义词（"and", "the"）
    - $\hat{f}$ 对齐的 token 更**有意义**（"game", "November"）且更**多样化**（GAR 更高）
    - 对比学习进一步增强了对齐的多样性和意义性

**2. 谱分析解释**

通过 SVD 分解文本嵌入矩阵获得主成分，分析 $f$ 和 $\hat{f}$ 嵌入空间的差异：

$$v_j = \mathbb{E}_{s_i \in D}\left[(\hat{\mathbf{h}}_i - \mathbf{h}_i)^\top \mathbf{u}_j\right]$$

两个核心观察：
- **Observation 1**：$\hat{f}$ 相对于 $f$ 的变化**集中在第一主成分** $v_1$
- **Observation 2**：第一主成分主要贡献于**无意义 token**（如 "the", "and"），而非关键 token

由此提出假设：**原始 LLM 的嵌入实际上已与关键 token 对齐，但被第一主成分的影响所掩盖**。

验证方法：手动调整嵌入 $\mathbf{h}_i^{adj} = \mathbf{h}_i + \lambda \mathbf{u}_1$，当 $\lambda \approx v_1$ 时，$f$ 的嵌入也能对齐有意义的关键 token → 假设成立。

**3. 稀疏检索应用**

基于对齐发现，提出 training-free 的稀疏检索：
- 文档端：保留 top-K 对齐 token 及其分数，构成稀疏向量（仅 K 个非零维度）
- 查询端：字面 token 集合 + 前 M 个对齐 token 作为扩展
- 相似度：查询扩展 token 集与文档稀疏向量的交集权重之和

**4. 可解释性应用**

- **Instruction-Following 解释**：同一文本在不同 instruction 下对齐到不同 token（情感分类 instruction → 对齐情感词，无 instruction → 对齐主题词）
- **语义相似 vs 语义相关解释**：NLI 数据训练的模型（SGPTnli）倾向对齐"dislike"类 token → 低相似度；MSMARCO 训练的模型（SGPTmsmarco）平衡对齐"dislike"+"apple" → 高相关度

### 损失函数 / 训练策略

本文以分析为主，不涉及新的训练。稀疏检索方法完全是 training-free 的。

各嵌入器使用的训练方法包括：
- Prompt Engineering（PromptEOL）
- 对比学习（InfoNCE loss）+ LoRA 微调
- 混合训练（对比学习 + 下一词预测，如 GritLM）

## 实验关键数据

### 主实验

**稀疏检索 vs 密集检索（nDCG@10）**：

| 方法 | FiQA | NFCorpus | SciFact | ArguAna |
|------|------|----------|---------|---------|
| BM25 | 0.236 | 0.325 | 0.665 | 0.315 |
| SPLADEv2 | 0.336 | 0.334 | 0.693 | 0.479 |
| LLM2Vec (密集) | 0.531 | 0.393 | 0.789 | 0.575 |
| LLM2Vec → Sparse | 0.404 | 0.326 | 0.669 | 0.481 |
| GritLM (密集) | 0.600 | 0.409 | 0.792 | 0.632 |
| GritLM → Sparse | 0.457 | 0.336 | 0.703 | 0.526 |

稀疏方法保留了密集检索 **~80%** 的性能，同时显著超越 BM25 和 SPLADEv2，推理 FLOPs 仅为密集方法的约 **13%**。

### 消融实验

**对齐指标对比（8 个嵌入器 vs 对应原始 LLM）**：

- 所有 $\hat{f}$ 的 Hit@10 接近 100%（至少一个输入 token 在 top-10 对齐结果中）
- 对比学习训练的 $\hat{f}$ 的 GAR 明显高于 prompt engineering 训练的
- Instruction-Following 实验证实了 aligned token 随 instruction 动态变化

### 关键发现

1. 文本嵌入与关键 token 的对齐是 LLM 嵌入器的**普遍现象**，跨架构、跨训练策略
2. 嵌入空间的主要变化集中在第一主成分——**减少第一主成分即可让原始 LLM 也展现出对齐行为**
3. 对比学习方法比 prompt engineering 方法产生更多样、更有意义的对齐 token
4. GritLM（同时训练对比学习+NTP）是唯一第一主成分增大的模型，与其他方法行为不同

## 亮点与洞察

1. **发现驱动**的工作：先发现有趣现象，再分析成因，最后展示应用，逻辑链条完整
2. **第一主成分假设**极具洞察力——各种不同的嵌入改进方法（PE、CL、混合训练）本质上都在做同一件事：调整第一主成分
3. **Instruction-Following 的直觉解释**：不同 instruction 引导 LLM 将嵌入对齐到不同的关键 token，首次给出了可视化的理解
4. 稀疏检索方法虽然简单，但展示了找到的规律的实际价值

## 局限与展望

1. 稀疏检索方法目前仅保留 ~80% 性能，可能通过学习最优 K 和 M 进一步提升
2. 分析基于 Wikipedia 数据，其他领域（代码、数学）的对齐行为待验证
3. 第一主成分的调整是全局的，不同文本可能需要不同程度的调整
4. 对齐的"有意义"目前缺乏定量定义（红色标记的 token 被判为无关，但可能存在深层语义联系）

## 相关工作与启发

- **Geva et al. (2022)**：将 FFN 值向量乘以 token 嵌入矩阵解释 FFN 的更新行为 → 本文将此思路应用于整体文本嵌入
- **PromptEOL / CSE**：基于 prompt engineering 和对比学习的 LLM 嵌入方法 → 本文为它们提供了统一的理解视角
- **SPLADE**：基于 BERT 的稀疏检索方法 → 本文的 LLM 稀疏检索可视为 SPLADE 的 LLM 版本
- **Dar et al. (2022)**：分析预训练 Transformer 中参数的 token 空间投影 → 同一分析思路

## 评分

- **创新性**: ★★★★★ — 发现本身极具启发性，为 LLM 嵌入领域提供了全新视角
- **实用性**: ★★★★☆ — 稀疏检索应用有实际价值，但主要贡献在分析性发现
- **实验充分度**: ★★★★☆ — 8 个嵌入器、多组定性定量分析、4 个 IR 数据集
- **写作质量**: ★★★★★ — 叙事流畅，从发现到解释到应用层层递进

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models](sticking_to_the_mean_detecting_sticky_tokens_in_text_embedding_models.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [\[ACL 2025\] LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations](ldir_low-dimensional_dense_and_interpretable_text_embeddings_with_relative_repre.md)
- [\[ACL 2025\] Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation](dont_reinvent_the_wheel_efficient_instruction-following_text_embedding_based_on_.md)

<!-- RELATED:END -->
