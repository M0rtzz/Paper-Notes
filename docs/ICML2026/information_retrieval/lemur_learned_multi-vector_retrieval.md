---
title: >-
  [论文解读] LEMUR: Learned Multi-Vector Retrieval
description: >-
  [ICML 2026][信息检索/RAG][多向量检索] Lemur 将多向量相似性搜索转化为监督学习问题，用一个两层 MLP 将 token 级嵌入映射到低维潜空间，再利用现有单向量 ANNS 索引完成检索，比 PLAID/MUVERA 等方法快一个数量级。
tags:
  - "ICML 2026"
  - "信息检索/RAG"
  - "多向量检索"
  - "近似最近邻搜索"
  - "MaxSim"
  - "监督学习降维"
  - "ColBERT"
---

# LEMUR: Learned Multi-Vector Retrieval

**会议**: ICML 2026  
**arXiv**: [2601.21853](https://arxiv.org/abs/2601.21853)  
**代码**: [github.com/ejaasaari/lemur](https://github.com/ejaasaari/lemur)  
**领域**: 信息检索  
**关键词**: 多向量检索, 近似最近邻搜索, MaxSim, 监督学习降维, ColBERT  

## 一句话总结
Lemur 将多向量相似性搜索转化为监督学习问题，用一个两层 MLP 将 token 级嵌入映射到低维潜空间，再利用现有单向量 ANNS 索引完成检索，比 PLAID/MUVERA 等方法快一个数量级。

## 研究背景与动机

**领域现状**：以 ColBERT 为代表的 late interaction 模型通过每个 token 一个 embedding 的多向量表示实现了比单向量模型更高的检索精度。查询与文档之间的相似度通过 MaxSim 度量，即每个 query token 与最相似 document token 的内积之和。

**现有痛点**：MaxSim 的计算代价极高——需要评估所有 query embedding 与所有 document embedding 之间的内积。现有加速方法（PLAID、DESSERT、EMVB、IGP）依赖 token 级剪枝作为第一步，但单个 token 的相似度是文档级 MaxSim 的不精确代理，导致候选集必须很大才能保证召回率。MUVERA 通过固定维度编码（FDE）将问题归约为单向量搜索，但需要 10240 维才能获得足够精度，内存和延迟代价仍然很高。

**核心矛盾**：多向量检索的高精度优势与其高延迟之间存在根本矛盾。现有方法要么依赖不精确的 token 级代理（PLAID 系列），要么需要极高维的数据无关编码（MUVERA），都无法高效弥合这一差距。

**本文目标**：设计一个轻量级、语料库特定的搜索降维框架，将多向量搜索归约为低维单向量搜索，同时保持高召回率。

**切入角度**：MaxSim 可以分解为逐 token 贡献的加和 $\text{MaxSim}(X,C) = \sum_{x \in X} \max_{c \in C} \langle x,c \rangle$，每个 token 对所有文档的贡献 $g(x) \in \mathbb{R}^m$ 是一个从 $\mathbb{R}^d$ 到 $\mathbb{R}^m$ 的回归问题，可以用 MLP 来学习。

**核心 idea**：用一个两层 MLP 学习 token 到文档相似度的映射，然后利用其线性输出层的结构将多向量搜索归约为潜空间中的单向量 MIPS 问题。

## 方法详解

### 整体框架
Lemur 分为离线索引和在线查询两个阶段。离线阶段训练一个两层 MLP 来估计每个 token embedding 对所有文档的 MaxSim 贡献，然后将输出层权重矩阵的行向量作为文档的单向量表示存入 ANNS 索引。在线查询时，将所有 query token 通过 MLP 隐藏层映射到潜空间并求和池化得到单向量查询表示，用 ANNS 检索 $k'$ 个候选文档，最后用精确 MaxSim 重排得到最终 top-$k$ 结果。

### 关键设计

1. **MaxSim 的监督学习重构**:

    - 功能：将多向量相似度估计转化为标准的多输出回归问题
    - 核心思路：目标函数 $g_l(x) = \max_{c \in C_l} \langle x,c \rangle$ 将每个 token embedding $x$ 映射到它对第 $l$ 个文档的 MaxSim 贡献。训练一个 MLP $\phi(x) = W \psi(x)$ 来拟合这个映射，其中 $\psi$ 是特征编码器（隐藏层），$W \in \mathbb{R}^{m \times d'}$ 是线性输出层。由于 $g_l$ 是凸分段线性函数，两层网络已有足够拟合能力
    - 设计动机：直接优化 MaxSim 估计精度而非使用数据无关的编码，使得低维（2048 维）表示就能超过 MUVERA 的 10240 维 FDE

2. **线性输出层到单向量 MIPS 的归约**:

    - 功能：将多输出回归的推理归约为标准单向量近似最近邻搜索
    - 核心思路：由于输出层是线性的，MaxSim 估计可以写成 $f(X) \approx W \Psi(X)$，其中 $\Psi(X) = \sum_{x \in X} \psi(x)$ 是池化后的查询向量。找 MaxSim 最大的 $k'$ 个文档等价于在 $d'$ 维空间中找与 $\Psi(X)$ 内积最大的 $k'$ 个权重行向量 $w_j$——这正是单向量 MIPS 问题，可直接用 HNSW 等高度优化的 ANNS 库加速
    - 设计动机：避免了专门设计多向量搜索数据结构的复杂性，直接复用成熟的单向量搜索生态

3. **可扩展的两阶段训练**:

    - 功能：将 MLP 训练扩展到百万级语料库
    - 核心思路：先在 $m' \ll m$ 个采样文档上预训练特征编码器 $\psi$，然后固定 $\psi$，对每个文档 $j$ 用 OLS 回归解析求解输出层的第 $j$ 行 $w_j = Z^+ y_j$，其中 $Z$ 是训练样本的特征矩阵。OLS 解对每个文档独立计算，天然可并行
    - 设计动机：直接训练 $m$ 输出的 MLP 内存不可行，两阶段拆分让特征学习和线性拟合解耦，索引构建线性可扩展

## 实验关键数据

### 主实验（ColBERTv2, k=100, QPS@≥80% Recall）

| 数据集 | Lemur | MUVERA | IGP | DESSERT | PLAID |
|--------|-------|--------|-----|---------|-------|
| MSMARCO (8.8M docs) | **799** | 150 | 62 | — | 13 |
| HotpotQA (5.2M docs) | **426** | 22 | 37 | — | 10 |
| NQ (2.7M docs) | **869** | 79 | 107 | 38 | 16 |
| Quora (523K docs) | **4068** | 787 | 679 | 284 | 89 |
| FiQA (58K docs) | **2416** | 239 | 310 | 242 | 51 |
| SCIDOCS (26K docs) | **2591** | 391 | 320 | 285 | 85 |

### 消融实验（隐藏层维度 $d'$ 对性能的影响）

| 配置 | MaxSim 近似精度 | 端到端延迟趋势 | 说明 |
|------|----------------|---------------|------|
| $d'=1024$ | 7/8 数据集超过 MUVERA 10240 维 FDE | 最快 ANNS | 精度已超 10x 大的 FDE |
| $d'=2048$（默认） | 显著优于 $d'=1024$ | 最佳性价比 | 延迟增量可忽略 |
| $d'=4096$ | 略优于 $d'=2048$ | 收益递减 | ANNS 代价部分抵消精度增益 |

### 关键发现
- Lemur 在所有 8 个 BEIR 数据集上在 ≥80% 召回时比最佳基线快 5–16 倍
- 1024 维 Lemur 嵌入在 7/8 数据集上召回率超过 10240 维 MUVERA FDE，说明监督学习的表示远优于数据无关编码
- 在非 ColBERTv2 模型（answerai-colbert-small、GTE-ModernColBERT、LFM2-ColBERT）上，MUVERA 召回率甚至无法超过 60%，而 Lemur 仍然稳定
- Pearson 相关系数和 Spearman 秩相关系数在所有数据集上均 > 0.94，表明 MaxSim 估计质量极高

## 亮点与洞察
- 将多向量搜索转化为监督回归再归约为单向量 MIPS 的"双重归约"思路非常优雅——关键洞察在于线性输出层的行向量天然就是文档在潜空间的单向量表示，无需额外投影
- 特征编码器用随机权重（ELM 模式）仍然有效，说明隐藏层主要起非线性特征扩展作用而非学习高度特化的表示，这一发现对理解 late interaction 嵌入空间的结构有启发价值
- 索引支持增量更新（新文档只需一次 OLS 回归 + HNSW 插入），这在生产系统中至关重要
- 用语料库文档本身作为训练数据（无需单独的训练查询集）也能工作，极大降低了部署门槛；使用真实查询训练时性能更好

## 局限与展望。不过两阶段训练设计使得重训成本较低（最大数据集 8.8M 文档总索引耗时约 4.8 小时）
- 未探索与极低精度向量压缩（如 2-bit 量化）的兼容性，标准标量量化和乘积量化应可直接适用但需验证
- 在视觉文档检索（ViDoRe）上优势缩小，因为图像文档的 embedding 数量远大于文本（平均 1073 vs 68），重排阶段成为瓶颈
- 未来可探索跨语料库迁移学习和合成训练集，减少对特定语料库的依赖
- 在视觉文档检索（ViDoRe）上优势缩小，因为图像文档的 embedding 数量远大于文本（平均 1073 vs 68），重排阶段成为瓶颈

## 相关工作与启发
- **vs MUVERA**: MUVERA 用数据无关的 FDE 做单向量归约，Lemur 用监督学习做语料库特定归约，后者在 1/10 维度下就超过前者，代价是需要训练
- **vs PLAID/DESSERT/IGP**: 这些方法依赖 token 级剪枝做候选筛选，proxy 不精确需要大候选集；Lemur 直接在文档级做 MaxSim 估计，候选集更小更精准
- **vs TCT-ColBERT**: TCT 训练一个通用的单向量检索器来蒸馏 MaxSim，Lemur 只训练一个轻量级搜索降维而非端到端检索器，更灵活且可直接适配任意 late interaction 模型
- **vs BGE-M3**: BGE-M3 通过自蒸馏联合训练 dense/sparse/multi-vector 模式，是通用 embedding 模型；Lemur 不训练编码器，只为已有模型构建高效搜索索引

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 双重归约思路原创性强，将检索问题转化为回归+MIPS 的视角令人眼前一亮
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个 BEIR 数据集 × 5 个文本模型 + 2 个视觉模型，消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，从问题定义到算法设计一气呵成
- 价值: ⭐⭐⭐⭐⭐ 直接可用于加速 ColBERT 系列检索系统，已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](../../ACL2026/information_retrieval/hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)
- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](../../ACL2026/information_retrieval/sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[ICML 2025\] POQD: Performance-Oriented Query Decomposer for Multi-Vector Retrieval](../../ICML2025/information_retrieval/poqd_performance-oriented_query_decomposer_for_multi-vector_retrieval.md)
- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](../../CVPR2026/information_retrieval/bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [\[ICML 2026\] HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling](hgmem_hypergraph-based_working_memory_to_improve_multi-step_rag_for_long-context.md)

</div>

<!-- RELATED:END -->
