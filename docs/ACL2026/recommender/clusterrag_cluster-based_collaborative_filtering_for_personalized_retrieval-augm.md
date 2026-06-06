---
title: >-
  [论文解读] ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation
description: >-
  [ACL2026][推荐系统][个性化RAG] ClusterRAG 把协同过滤引入个性化 RAG：先用用户历史文档构建用户表示并用 HDBSCAN 聚类，再从目标用户和相似用户中分层检索 profile 文档组成 prompt…
tags:
  - "ACL2026"
  - "推荐系统"
  - "个性化RAG"
  - "协同过滤"
  - "用户聚类"
  - "HDBSCAN"
  - "LaMP"
---

# ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation

**会议**: ACL2026  
**arXiv**: [2605.18769](https://arxiv.org/abs/2605.18769)  
**代码**: https://github.com/academicprojects44/anonymous  
**领域**: 个性化推荐 / Personalized RAG  
**关键词**: 个性化RAG, 协同过滤, 用户聚类, HDBSCAN, LaMP

## 一句话总结
ClusterRAG 把协同过滤引入个性化 RAG：先用用户历史文档构建用户表示并用 HDBSCAN 聚类，再从目标用户和相似用户中分层检索 profile 文档组成 prompt，在 LaMP 多任务基准上使 hybrid 模式全面优于 vanillaRAG、LaMP-IPA、ROPG 和 CFRAG。

## 研究背景与动机
**领域现状**：RAG 已经是降低幻觉、增强事实性的主流范式，通常做法是根据当前 query 检索外部文档，再把检索结果拼进生成模型上下文。个性化 RAG 进一步引入用户历史，让回答符合用户偏好、写作风格或长期兴趣。

**现有痛点**：很多个性化 RAG 只看目标用户自己的 profile，这在用户历史稀疏、噪声大或当前 query 与历史记录不完全匹配时很脆弱。另一端是非个性化 RAG，完全忽视用户长期偏好。已有协同式方法尝试找相似用户，但如果直接在大规模用户集合中两两计算相似度，成本高；找到相似用户后，如何选择其文档并与目标用户 profile 混合也缺少系统设计。

**核心矛盾**：个性化需要充分利用目标用户历史，但单个用户历史往往不完整；协同过滤能补足稀疏性，却会带来检索复杂度、隐私和噪声邻居问题。一个可扩展的 Personalized RAG 需要在“个人信号”和“相似群体信号”之间找到可控混合方式。

**本文目标**：ClusterRAG 希望构建一个模型无关、检索器可替换、能利用协同信号但不强依赖模型参数微调的 RAG pipeline。它要解决三件事：用户如何表示、相似用户如何高效找、目标用户文档与相似用户文档如何一起进入生成 prompt。

**切入角度**：作者把推荐系统里的聚类式协同过滤迁移到 RAG 检索前端。用户先被组织为语义一致的 cohort，搜索相似用户时只在同簇内部排序；文档检索也通过 profile 文档聚类先选主题簇再做细粒度 rerank。

**核心 idea**：用 HDBSCAN 构建用户/文档的层次化检索空间，再用 hybrid profile retrieval 同时注入目标用户和相似用户的证据，让 LLM 生成更稳定的个性化输出。

## 方法详解

### 整体框架
ClusterRAG 包含三个阶段：用户表示与相似用户检索、profile 文档检索、个性化生成。首先，系统把每个用户的历史文档编码成 dense embeddings，并对文档向量取平均得到用户表示。随后用 HDBSCAN 将用户分入可变密度簇，并在每个簇内用 ColBERTv2 计算细粒度用户相似度，保留每个用户的 top-$k$ 相似邻居。给定当前 query 后，系统可以只检索目标用户文档、只检索相似用户文档，或使用二者混合。最后，候选文档还会再被聚成主题簇，先选相关簇，再在簇内取 top-$m$ 文档拼入 IPA prompt，交给生成模型输出。

### 关键设计
1. **用户级 HDBSCAN 聚类与簇内排序**:

	- 功能：在不做全局两两用户比较的情况下找到行为相似用户。
	- 核心思路：每个用户 $u$ 的表示为其 profile 文档 embedding 的平均值 $\mathbf{z}_u=\frac{1}{n_u}\sum_i f(d_i)$。HDBSCAN 根据这些表示自动发现可变密度用户簇；对同一簇内的用户，再用 ColBERTv2 构造簇内相似度矩阵 $R^C_{u,v}=ColBERTv2(\mathbf{z}_u,\mathbf{z}_v)$，保留 top-$k$ 邻居。
	- 设计动机：全局用户相似度计算在大规模场景中昂贵，且容易引入主题不一致的噪声邻居；先聚类再排序能把比较限制在行为更一致的 cohort 内。

2. **三种 profile retrieval 模式**:

	- 功能：控制个体信号与协同信号的来源。
	- 核心思路：User-only 模式只用目标用户自己的 profile；Collaborative 模式从 top-$k$ 相似用户的 profile 中取文档；Hybrid 模式把两类文档合并作为候选池。论文默认 $k=1$、$m=2$，因此能在很少 profile 文档下测试协同信号是否有帮助。
	- 设计动机：单一用户 profile 对冷启动或稀疏用户不够，纯协同又可能稀释个人偏好；hybrid 允许系统同时保留个人历史与相似用户补充证据。

3. **文档主题簇检索与 IPA 生成**:

	- 功能：从候选 profile 文档中选出最相关、最适合进入 prompt 的上下文。
	- 核心思路：候选文档先经 dense retriever 编码，并由 HDBSCAN 形成主题簇及中心；query embedding 先与簇中心比较选 top-$B$ 簇，再在选中簇内 rerank 取 top-$m$ 文档。生成阶段采用 In-Prompt Augmentation，把 query 与检索文档按任务模板拼接。prompt 中 profile 长度由 $|U_p|=\mathcal{G}_t(L_{max}-\min(|q|,\lfloor \gamma L_{max}\rfloor))$ 控制，默认 $\gamma=0.55$。
	- 设计动机：直接把所有用户文档扔进 prompt 会浪费长度，也会引入不相关历史；层次检索能降低复杂度到 $\mathcal{O}(K+B\cdot N/K)$，同时让 prompt 只包含当前任务最相关的证据。

### 损失函数或训练策略
ClusterRAG 本身是检索与 prompt 组织框架，不要求改造生成模型结构。主实验用 fine-tuned FlanT5-base；扩展实验还用 FlanT5-XXL 和 Qwen2-7B-Instruct 做 zero-shot 个性化测试。训练采用 AdamW，学习率 $5\times10^{-5}$，weight decay $10^{-4}$，warm-up ratio 0.05，最多 30 epoch，batch size 16，最大 prompt 长度 512，最大输出长度 128，beam size 4。实验在 Quadro RTX 8000 48GB 上运行，每个任务约 10-24 小时。

## 实验关键数据

### 主实验
LaMP 基准包含个性化 citation、movie tagging、product rating、headline/title generation 和 tweet paraphrasing 等任务。下表摘取主表中的代表指标；分类任务越高越好，LaMP-3 的 MAE/RMSE 越低越好。

| 方法 | LaMP-1 Acc/F1 | LaMP-2 Acc/F1 | LaMP-3 MAE/RMSE | LaMP-7 R-1/R-L |
|------|---------------|---------------|-----------------|----------------|
| vanillaRAG | 0.630 / 0.630 | 0.520 / 0.440 | 0.371 / 0.709 | 0.310 / 0.273 |
| LaMP-IPA | 0.674 / 0.664 | 0.570 / 0.522 | 0.289 / 0.608 | 0.508 / 0.457 |
| CFRAG | 0.633 / 0.327 | 0.534 / 0.036 | 0.354 / 0.707 | 0.375 / 0.306 |
| ClusterRAG-C | 0.674 / 0.673 | 0.644 / 0.607 | 0.284 / 0.624 | 0.507 / 0.454 |
| ClusterRAG-U | 0.645 / 0.645 | 0.649 / 0.612 | 0.271 / 0.599 | 0.514 / 0.464 |
| ClusterRAG-H | 0.690 / 0.690 | 0.661 / 0.620 | 0.270 / 0.594 | 0.521 / 0.470 |

### 消融实验
| 变体 | LaMP-3 MAE | LaMP-3 RMSE | LaMP-7 R-1 | LaMP-7 R-L | 说明 |
|------|-----------:|------------:|-----------:|-----------:|------|
| w/o user clustering | 0.320 | 0.637 | 0.458 | 0.371 | 随机相似用户，协同信号变噪 |
| w/o intra-cluster sim | 0.329 | 0.639 | 0.501 | 0.442 | 没有簇内精排，相似用户质量下降 |
| w/o doc ranking | 0.331 | 0.642 | 0.462 | 0.413 | 文档级证据未被有效排序 |
| Centroids only | 0.400 | 0.643 | 0.472 | 0.438 | 只用中心表示，丢失具体证据 |
| k-means | 0.291 | 0.610 | 0.502 | 0.453 | 聚类可替换，但弱于 HDBSCAN |
| ClusterRAG | 0.270 | 0.594 | 0.521 | 0.470 | 完整方法 |

### 关键发现
- Hybrid 模式在所有 LaMP 任务上取得最优，说明目标用户 profile 与相似用户 profile 的互补性很强。
- 只用 2 个 profile 文档就能超过需要更多文档的基线，说明“选对文档”比“塞更多文档”更重要。
- ColBERTv2 是最强 retriever：在 LaMP-1/2/7 上达到 0.690/0.690、0.661/0.620、0.521/0.470；Random、Recency、BM25 明显落后，BGE 和 Contriever 居中。
- zero-shot LLM 也受益于 ClusterRAG：pFlan 在 LaMP-1 上从 0.546/0.540 提升到 0.648/0.647，pQwen2 在 LaMP-2 上从 0.521/0.521 提升到 0.610/0.606。

## 亮点与洞察
- **把协同过滤做成 RAG 前端，而不是改 LLM**：这让方法更容易接到现有 personalized generation 系统里，也避免了为每个用户微调模型。
- **用户聚类和文档聚类分层对应两个问题**：用户聚类解决“向谁借信号”，文档聚类解决“借哪些证据”，结构上很清楚。
- **hybrid 检索是核心收益点**：实验显示 user-only 和 collaborative-only 都有用，但二者混合最稳，说明个性化不是单纯替换上下文，而是补充上下文。
- **对冷启动有实际意义**：当目标用户历史稀疏时，相似用户文档可以填补缺口；当协同信号不可信时，框架也能退回 user-only。

## 局限与展望
- 生成端依赖 IPA prompt，作者也承认 prompt formulation 不是最优；更结构化的 prompt 或检索-生成联合优化可能进一步提升结果。
- LaMP-1 和 LaMP-5 只有论文摘要而非全文，因此 citation/title 类任务的信息上限受限。
- 实验限制在英文、文本数据，尚未验证多语言、多模态用户历史或跨平台推荐场景。
- 框架性能仍依赖底层 LLM 与 retriever；如果用户 embedding 或文档 embedding 本身带偏差，协同过滤可能放大群体偏见。
- 后续可以研究增量聚类、在线用户重分配、隐私保护 embedding 聚合，以及把生成反馈回流到检索排序中。

## 相关工作与启发
- **vs vanillaRAG / Self-RAG**: 传统 RAG 主要围绕当前 query 检索共享知识；ClusterRAG 进一步把用户长期历史和相似用户历史纳入上下文。
- **vs LaMP-IPA / ROPG**: 这些个性化方法更侧重目标用户自己的 profile；ClusterRAG 的增量在于显式建模跨用户协同信号。
- **vs CFRAG**: CFRAG 使用 contrastive learning 找相似用户；ClusterRAG 用 HDBSCAN 和簇内 ColBERTv2 排序，强调可扩展的 cohort 检索和文档级 rerank。
- **启发**：在企业知识助手、学习助手或长周期写作助手中，可以把相似用户或相似项目的历史交互作为“协同记忆”，但需要用聚类与权限控制约束检索范围。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 把协同过滤和 Personalized RAG 结合得系统，单个模块不陌生，但组合完整。
- 实验充分度: ⭐⭐⭐⭐☆ LaMP 覆盖任务多，并有 retriever、LLM、消融分析；真实大规模线上延迟和隐私评估仍欠缺。
- 写作质量: ⭐⭐⭐⭐☆ 方法拆解清楚，表格覆盖全面；部分符号和 prompt 细节略显密集。
- 价值: ⭐⭐⭐⭐☆ 对个性化 RAG 很有工程启发，尤其适合用户历史稀疏但群体行为可用的场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MemRec: Collaborative Memory-Augmented Agentic Recommender System](memrec_collaborative_memory-augmented_agentic_recommender_system.md)
- [\[AAAI 2026\] SlideTailor: Personalized Presentation Slide Generation for Scientific Papers](../../AAAI2026/recommender/slidetailor_personalized_presentation_slide_generation_for_scientific_papers.md)
- [\[ICML 2026\] Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy](../../ICML2026/recommender/rethinking_contrastive_learning_for_graph_collaborative_filtering_limitations_an.md)
- [\[NeurIPS 2025\] FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens](../../NeurIPS2025/recommender/face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)

</div>

<!-- RELATED:END -->
