---
title: >-
  [论文解读] DTCRS: Dynamic Tree Construction for Recursive Summarization
description: >-
  [ACL 2025][文本生成][Recursive Summarization] 提出 DTCRS 方法，根据文档结构和查询语义动态构建摘要树，通过问题分解和子问题引导聚类减少冗余摘要节点，在三个 QA 数据集上显著优于静态摘要树方法 RAPTOR。
tags:
  - ACL 2025
  - 文本生成
  - Recursive Summarization
  - RAG
  - Summary Tree
  - Question Decomposition
  - GMM Clustering
---

# DTCRS: Dynamic Tree Construction for Recursive Summarization

**会议**: ACL 2025  
**arXiv**: [2604.07012](https://arxiv.org/abs/2604.07012)  
**代码**: 未公开  
**领域**: NLP Generation / RAG / 文本摘要  
**关键词**: Recursive Summarization, RAG, Summary Tree, Question Decomposition, GMM Clustering  

## 一句话总结

提出 DTCRS 方法，根据文档结构和查询语义动态构建摘要树，通过问题分解和子问题引导聚类减少冗余摘要节点，在三个 QA 数据集上显著优于静态摘要树方法 RAPTOR。

---

## 研究背景与动机

### 问题背景
检索增强生成（RAG）通过整合外部知识来缓解大语言模型（LLM）的幻觉问题。对于需要多步推理的抽象性问题，通常需要文档多个部分的知识。递归摘要（Recursive Summarization）通过对文本块进行层次化聚类和摘要，构建层级摘要树来整合分散信息，为抽象性问题提供证据支持。

### 现有方法的不足

**冗余摘要节点过多**：传统摘要树（如 RAPTOR）是基于文档的静态树，生成大量与查询无关的冗余摘要节点，不仅增加构建时间，还可能负面影响问答效果

**摘要与查询缺乏相关性**：静态摘要树仅反映文档本身结构，未捕获查询语义，导致生成的摘要与用户问题关联弱

**不区分问题类型**：递归摘要对抽象性问题有益，但对简单的抽取式或布尔式问题可能没有优势甚至产生干扰，盲目使用递归摘要不合理

### 研究动机
设计一种动态摘要树构建方法：能根据文档结构和查询语义按需构建摘要树，增强摘要与问题的相关性，同时减少冗余节点和构建时间。

---

## 方法详解

### 整体框架
DTCRS 的整体流程分为三步：
1. **问题类型分类**：判断问题是否需要递归摘要
2. **动态摘要树构建**：对需要摘要的复杂问题，通过问题分解和子问题引导聚类构建摘要树
3. **检索与回答**：使用 collapsed tree 方法从摘要树中检索相关节点

### 关键设计

#### 1. 问题类型分类
- 利用 LLM 生成文档的目录（Table of Contents, ToC）
- 将 ToC 和原始问题输入 LLM 分类器，输出二元标签：$y = f_{\text{LLM}}(q, c) \in \{0, 1\}$
- 标签为 1 表示需要构建动态摘要树；标签为 0 则使用 DPR 检索 top-k 文本块
- 分类准则：问题是否复杂，是否需要综合目录多个部分的信息

#### 2. 问题分解（Question Decomposition）
- 将 ToC $c$ 和原始问题 $q$ 输入 LLM 问题分解器，生成子问题集合 $Q' = \{q_1, q_2, \dots, q_j\}$
- 引入 ToC 的原因：限制子问题范围在参考文档主题内，并使子问题更好地对齐文档各部分
- 子问题数量 $N_{Q'}$ 远小于文本块数量 $N_d$，从而大幅减少聚类数量和计算开销

#### 3. 文本块分割
- 使用固定分割方法，文本块大小 500 tokens
- 跨越 500 token 边界的句子整体移到下一文本块，避免不完整句子

#### 4. 聚类
- 使用 SBERT 编码文本块和子问题，获得嵌入向量
- 将文本块嵌入 $E_T$ 和子问题嵌入 $E_{Q'}$ 合并后用 UMAP 统一降维，保持语义一致性：$E_{\text{reduced}} = \Phi_{\text{UMAP}}(E_T \oplus E_{Q'})$
- 采用 GMM（高斯混合模型）进行软聚类，允许文本块属于多个类别
- **第一层聚类**：使用子问题数量作为聚类数，子问题嵌入作为初始聚类中心
- **后续层**：不再考虑子问题，使用 BIC 确定聚类数和随机初始中心
- 采用全局聚类而非层次聚类，因为子问题数量少且全局聚类更高效

#### 5. 递归摘要生成
- 聚类后的文本块输入 LLM 生成对应摘要
- 递归重复降维、聚类、摘要过程直到无法进一步聚类

#### 6. 检索方法
- 使用 collapsed tree 方法检索：将所有节点展开为单层，选择与查询余弦相似度最高的 top-k 节点

### 计算复杂度分析
- RAPTOR 的总计算量为 $2N_d$（几何级数求和）
- DTCRS 的总计算量约为 $N_d + N_{Q'} + \frac{N_{Q'}}{2} + \cdots$，由于 $N_{Q'} \ll N_d$，近似为 $N_d$
- 时间效率提升约 50%

---

## 实验

### 实验设置
- **LLM**：GPT-4、GPT-4o-mini、DeepSeek-V2-Lite-Chat (7B)
- **嵌入模型**：SBERT
- **GMM 聚类阈值**：0.5
- **DPR top-k**：5，collapsed tree 最大 token 限制：3500
- **硬件**：NVIDIA A800 80GB GPU

### 数据集

| 数据集 | 问题数 | 平均文档字符数 |
|--------|--------|---------------|
| QASPER | 1451 | 21,889 |
| QuALITY | 2128 | 24,723 |
| NarrativeQA | 10,558 | 332,134 |

### 主实验结果

**QASPER (F1 score)**：

| 方法 | F1 |
|------|-----|
| DPR + GPT-4 | 51.3 |
| CoLT5 XL | 53.9 |
| RAPTOR + GPT-4 | 55.7 |
| **DTCRS + GPT-4** | **58.5** |

**QuALITY (准确率)**：

| 方法 | Test Set | Hard Subset |
|------|----------|-------------|
| DPR + GPT-4o-mini | 62.8 | 50.3 |
| RAPTOR + GPT-4o-mini | 67.6 | 57.0 |
| **DTCRS + GPT-4o-mini** | **74.7** | **62.9** |

**NarrativeQA**：DTCRS 在 METEOR 指标上达到最佳 14.4%，但整体不优于 DPR/RAPTOR，作者归因于 NarrativeQA 以简单抽取式问题为主。

### 消融实验（QASPER, F1）

| 变体 | GPT-4o-mini | DeepSeek |
|------|-------------|---------|
| DPR | 33.0 | 31.3 |
| w/o 全局聚类 | 43.8 | 38.6 |
| w/o 分类器 | 37.2 | 33.7 |
| w/o ToC | 44.1 | 38.1 |
| w/o 问题分解 | 37.5 | 32.9 |
| **DTCRS** | **44.5** | **38.3** |

### 关键发现
1. **分类器最关键**：移除分类器导致最大性能下降，因为抽取式问题用 DPR 更好，盲目引入摘要会降低性能
2. **问题分解重要**：移除问题分解器后性能明显下降
3. **效率提升显著**：摘要层节点数从 RAPTOR 的 64 减少到 DTCRS 的 4.99（减少 92.2%），构建时间从 214.39s 减少到 40.82s（减少 80.95%）
4. **抽象性问题受益更大**：DTCRS 在抽象性问题上的优势最为明显
5. **问题类型分布影响效果**：QASPER 和 QuALITY 抽象性问题占比超 20%，故效果好；NarrativeQA 几乎无抽象性问题，故效果有限

---

## 亮点与洞察

1. **查询导向的动态摘要树**是核心创新，将静态文档摘要转变为动态查询相关摘要，思路简洁有效
2. **子问题嵌入作为聚类中心**的设计巧妙，同时解决了聚类数量和相关性两个问题
3. **问题类型分类**的引入体现了实用主义思维——不是所有问题都需要递归摘要
4. **效率提升**显著（节点减少 92%，时间减少 81%），对长文档场景有实际意义
5. 对「递归摘要适用于哪些问题类型」提供了有价值的实证分析

## 局限性

1. NarrativeQA 上效果有限，说明方法对问题类型组成敏感
2. 依赖 LLM 进行多次调用（分类、ToC 生成、问题分解、摘要），可能带来额外延迟和成本
3. 问题分类器的准确性直接影响最终效果，但论文未深入分析分类器本身的性能
4. 仅在英文数据集上验证，跨语言泛化性未知
5. 摘要生成可能引入轻微幻觉（虽然论文认为不影响 QA 结果）

## 相关工作

- **RAG**：Self-RAG (Asai et al. 2023)、Self-Adaptive RAG (Jeong et al. 2024)
- **递归摘要**：RAPTOR (Sarthi et al. 2024)、Recursively Summarizing Books (Wu et al. 2021)、HIBRIDS (Cao & Wang 2022)
- **查询增强**：查询重写 (Mo et al. 2024)、问题分解 (Reppert et al. 2023)
- **检索方法**：DPR (Karpukhin et al. 2020)、ColBERT (Khattab & Zaharia 2020)

---

## 评分 ⭐⭐⭐⭐

方法设计简洁且有效，问题驱动的动态摘要树是对静态 RAPTOR 的合理改进，实验充分。不过，对 LLM 调用次数的成本分析不够深入，且方法的适用范围与问题类型分布高度相关。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] AlphaContext: An Evolutionary Tree-based Psychometric Context Generator for Creativity Assessment](../../ACL2026/nlp_generation/alphacontext_an_evolutionary_tree-based_psychometric_context_generator_for_creat.md)
- [\[ACL 2025\] Decomposed Opinion Summarization with Verified Aspect-Aware Modules](decomposed_opinion_summarization_with_verified_aspect-aware_modules.md)
- [\[ACL 2025\] Context-Aware Hierarchical Merging for Long Document Summarization](context-aware_hierarchical_merging_for_long_document_summarization.md)
- [\[ACL 2025\] Theme-Explanation Structure for Table Summarization Using Large Language Models](theme-explanation_structure_for_table_summarization_using_large_language_models_.md)
- [\[ACL 2025\] An Empirical Study of Many-to-Many Summarization with Large Language Models](an_empirical_study_of_manytomany_summarization.md)

<!-- RELATED:END -->
