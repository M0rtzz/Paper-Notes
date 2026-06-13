---
title: >-
  [论文解读] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search
description: >-
  [推荐系统] 提出 CroPS 数据引擎，通过 query 改写行为、推荐系统交互、LLM 世界知识三个视角扩充正样本集合，配合分层标签分配（HLA）和 H-InfoNCE 损失函数，打破工业级稠密检索系统中的信息茧房效应，已在快手搜索全量部署。 核心问题：自我强化训练范式导致的信息茧房 工业级短视频搜索系统普遍采用双塔（d…
tags:
  - "推荐系统"
---

# CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search

## 基本信息

- **论文链接**: [arXiv:2511.15443](https://arxiv.org/abs/2511.15443)
- **作者**: Ao Xie, Jiahui Chen, Quanzhi Zhu, Xiaoze Jiang, Zhiheng Qin, Enyun Yu, Han Li (快手科技)
- **会议**: AAAI 2026
- **代码**: 无
- **领域**: 信息检索 / 推荐系统 / 短视频搜索

## 一句话总结

提出 CroPS 数据引擎，通过 query 改写行为、推荐系统交互、LLM 世界知识三个视角扩充正样本集合，配合分层标签分配（HLA）和 H-InfoNCE 损失函数，打破工业级稠密检索系统中的信息茧房效应，已在快手搜索全量部署。

## 研究背景与动机

### 核心问题：自我强化训练范式导致的信息茧房

工业级短视频搜索系统普遍采用双塔（dual-encoder）架构做稠密检索，训练数据来源于线上系统的历史曝光交互日志：用户点击/观看的视频作为正样本，未曝光或被过滤的视频作为负样本。这种**自我强化（self-reinforcing）训练范式**存在根本性缺陷——只有历史上被系统曝光过的内容才有机会成为正样本，语义相关但从未被检索到的内容会被系统性地排除在正样本集之外，甚至被错误标记为负样本。

论文用一个直观的例子说明：当用户搜索"transformer"时，由于历史数据中深度学习领域内容的主导地位，关于"电力变压器"的视频虽然语义相关，却因从未被曝光而被错误地归为负样本。这种偏差会导致模型检索行为越来越保守和单一，用户体验持续下降。

### 已有工作的不足

先前研究主要集中在两个方向：（1）架构改进，如 ColBERT 的 late interaction 设计；（2）负样本采样策略，如 ANCE 的动态难负样本、TriSampler 等。然而这些方法都没有跳出自我强化训练范式的框架——无论负样本怎么选，正样本始终局限在历史曝光集合中，信息茧房的根源未被触及。

### 本文动机

作者敏锐地指出：**正样本扩充（positive sample enrichment）是一个被严重忽视但极具潜力的方向**。通过从多个视角引入历史曝光之外的语义相关正样本，可以有效打破数据层面的信息茧房边界。这一洞察构成了 CroPS 的核心出发点。

## 方法详解

### 整体框架

CroPS 由三个主要模块组成：

1. **CroPS 数据引擎**：从三个互补视角（query 级别、系统级别、世界知识级别）扩充正样本集合 $\mathcal{P} = \mathcal{P}_0 \cup \mathcal{P}_1 \cup \mathcal{P}_2 \cup \mathcal{P}_3$
2. **分层标签分配（HLA）**：为不同来源的正负样本赋予 0-5 的层级标签，替代传统的二元标签
3. **H-InfoNCE 损失函数**：支持多层级对比学习的损失函数，高效且适配 HLA

### 关键设计一：三视角正样本扩充

**（1）Query 级别正样本扩充（$\mathcal{P}_1$）**

利用用户的 query 改写行为（query reformulation）。当用户对初始搜索结果不满意时，会在短时间内（90秒窗口）发出语义相近的后续 query。CroPS 将用户在改写 query 下交互的视频视为原始 query 的潜在正样本。通过一个预训练的 6 层 Transformer 语义判别器 $\theta(\cdot)$ 评估原始 query 与候选视频的相关性，阈值 $\alpha = 0.6$：

$$\mathcal{P}_1 = \bigcup_{q_i \in \mathcal{Q}} \{d_{ij} \in \mathcal{D}_i \mid \theta(q, d_{ij}) > \alpha\}$$

这一设计的巧妙之处在于：改写行为本身编码了用户"真正想找什么"的信号，这些正样本通常处于原始 query 的检索盲区，恰好补充了自我强化范式遗漏的内容。

**（2）系统级别正样本扩充（$\mathcal{P}_2$）**

打破搜索系统与推荐系统之间的数据壁垒。对于 query $q$，找到发出过该 query 的用户集合 $\mathcal{U}$，检索每个用户在 query 时间戳附近在推荐流中交互的视频（上限 100 条），再通过同一语义判别器筛选语义相关的视频：

$$\mathcal{P}_2 = \bigcup_{u_i \in \mathcal{U}} \{d_{ij} \in \mathcal{D}_i \mid \theta(q, d_{ij}) > \alpha\}$$

推荐系统的交互数据通常更新更快、更贴近用户个人兴趣，与搜索数据形成互补。

**（3）世界知识扩充（$\mathcal{P}_3$）**

利用 LLM（Qwen2.5-14B）作为"伪检索器"。采用 one-shot 策略，给 LLM 提供 query 和一个已知相关视频作示例，让其生成与 query 匹配的其他视频描述作为合成正样本。共生成 3500 万条合成正样本。这一策略模拟了用户在 App 内找不到满意内容时转向外部信息源的行为，将平台外部的语义关联和事实知识注入训练过程。

### 关键设计二：分层标签分配（HLA）

不同来源的正样本可靠性和重要性不同，简单统一处理会导致次优学习效果。HLA 将样本划分为 6 个层级（0-5）：

| 层级 | 样本类型 | 含义 |
|:---:|:---:|:---:|
| 5 | Query 改写正样本 | 最直接反映用户精确意图 |
| 4 | 系统级正样本 / 世界知识正样本 / 点击视频 | 强相关信号 |
| 3 | 排序阶段曝光但未点击的视频 | 中等相关 |
| 2 | 排序阶段未曝光的视频 | 弱/不确定相关 |
| 1 | 预排序到排序间被过滤的视频 | 低相关 |
| 0 | 批内负样本 | 不相关 |

为何 query 改写正样本获得最高标签（5）？因为用户的改写行为代表了对初始搜索结果不满意后的主动修正，后续交互最真实地反映了用户的底层需求。赋予最高权重可以引导模型学会主动理解模糊 query 背后的多义性，从而减少用户改写频率。

### 关键设计三：H-InfoNCE 损失函数

标准 InfoNCE 假设二元相关性（正/负），无法利用 HLA 提供的多层级监督信号。H-InfoNCE 引入层级感知的对比结构：对于标签为 $l$ 的正样本，只有标签严格小于 $l$ 的样本才被视为负样本：

$$\mathcal{L} = -\sum_{d_i \in \mathcal{S}} \log \frac{\exp(\text{sim}(q, d_i) / \tau)}{\sum_{d_j \in \{d_i\} \cup \{d_k \in S | l_i > l_k\}} \exp(\text{sim}(q, d_j) / \tau)}$$

实现上使用掩码矩阵过滤不可比较的样本，并用标签索引的数据结构组织输入，所有层级的对比损失在一次前向传播中计算完成，速度与标准 InfoNCE 相当。

## 实验关键数据

### 表1：主实验对比（CPSQA 数据集）

| 方法 | Recall@100 CT(%) | Recall@100 QR(%) | NDCG@4(%) |
|:---:|:---:|:---:|:---:|
| BM25 | 42.9 | 22.5 | 64.8 |
| DPR | 56.0 | 30.7 | 66.5 |
| ANCE | 56.9 | 31.3 | 67.1 |
| ADORE+STAR | 59.4 | 31.9 | 67.4 |
| TriSampler | 59.8 | 32.2 | 66.9 |
| FS-LR | 59.6 | 33.0 | 66.0 |
| **CroPS** | **69.1** | **40.1** | **67.0** |

CroPS 在 CT 上比最强 baseline（TriSampler）提升 9.3%，在 QR 上比 FS-LR 提升 7.1%，提升幅度非常显著。QR 指标的大幅提升意味着用户在首次搜索就能找到想要的内容，减少了改写需求。

### 表2：线上 A/B 测试结果

| 模型类型 | CTR 提升 | LPR 提升 | RQR 降低 |
|:---:|:---:|:---:|:---:|
| Dense Model | +0.869% | +0.483% | -0.646% |
| Sparse Model | +0.783% | +0.423% | -0.614% |

在快手搜索的线上 A/B 测试中，CroPS 在 Dense Model 上使点击率提升 0.869%，长播放率提升 0.483%，query 改写率降低 0.646%。这些指标在工业界大规模系统上的提升幅度是非常可观的。同时 CroPS 在 Sparse Model 上也取得一致性提升，验证了方法的架构无关性。

## 亮点与洞察

1. **问题定位精准**：将信息茧房效应的根因定位在正样本空间的局限性而非负样本策略，这一视角新颖且切中要害。先前工作过度关注负采样，而正样本扩充这一"低垂果实"被长期忽视。

2. **三视角互补设计合理**：Query 改写（捕获意图连续性）、跨系统数据（打破信息孤岛）、LLM 世界知识（引入外部语义）三者各解决信息茧房的不同维度，消融实验证明三者的增益是叠加的。

3. **HLA 的精巧设计**：将 query 改写正样本赋予最高权重的决策有深刻洞察——它既反映了用户最真实的信息需求，又通过训练时的高权重引导模型主动减少用户改写行为，形成正向循环。

4. **工业部署友好**：H-InfoNCE 训练速度与标准 InfoNCE 相当（88h vs 178h 甚至更快），CroPS 不引入额外推理开销、架构无关，已全量部署服务数亿用户。

## 局限性

1. **语义判别器依赖**：Query 级和系统级正样本的质量高度依赖轻量判别器 $\theta(\cdot)$ 的准确性，阈值 $\alpha = 0.6$ 的选择缺乏深入分析，不同取值对噪声引入的影响未充分讨论。

2. **LLM 合成样本质量**：3500 万条合成正样本的质量控制流程未详述，LLM 幻觉可能引入错误的语义关联。对合成样本的过滤和质量评估策略不够透明。

3. **数据集非公开**：CPSQA 数据集基于快手内部数据构建，实验无法复现，外部研究者难以公平比较。

4. **标签层级设计的通用性**：HLA 的层级划分（0-5）和具体赋值是针对快手搜索场景经验设定的，迁移到其他搜索场景时可能需要重新设计。

5. **仅评估文本模态**：文档编码器仅使用视频的文本信息（标题、字幕等），未利用视频的视觉和音频信息，可能在某些查询类型上存在语义表达的瓶颈。

## 相关工作与启发

- **DPR / ANCE / ADORE+STAR**：代表了稠密检索在负采样策略上的演进，CroPS 从正样本角度与这些方法互补。
- **FS-LR (Zheng et al., 2024)**：引入多级负样本标签，是 CroPS HLA 思想在负样本侧的前身，CroPS 将层级思想扩展到正负样本统一框架。
- **ColBERT / Poly-encoder**：结构增强型方法，但 late interaction 难以集成 ANN 索引，CroPS 的架构无关性是明确优势。
- **对比学习中的层级/加权策略**：RINCE 等探索了分级对比，CroPS 的 H-InfoNCE 提供了更系统的层级对比框架。

**启发**：这项工作对推荐/搜索系统中数据层面的系统性偏差给出了一个范式级别的解法。核心启发在于——当模型性能瓶颈在于训练数据本身的偏差时，优化模型架构或损失函数只是治标，从数据源头引入多视角信号才是治本之策。CroPS 用搜索-推荐跨系统数据桥接的思路，对任何存在多个数据孤岛的工业系统都有借鉴意义。

## 评分

**4/5 ⭐**

扎实的工业系统论文。问题定位精准、方法设计系统完整、线上部署验证充分。扣分主要因为数据集非公开难以复现，且核心判别器和标签设计缺少充分的敏感性分析。HLA + H-InfoNCE 的层级对比学习框架对稠密检索领域有明确的方法论贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[ICLR 2026\] Search Arena: Analyzing Search-Augmented LLMs](../../ICLR2026/recommender/search_arena_analyzing_search-augmented_llms.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](inductive_generative_recommendation_via_retrieval-based_speculation.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)

</div>

<!-- RELATED:END -->
