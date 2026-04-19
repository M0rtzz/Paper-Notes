---
title: >-
  [论文解读] Large Language Models are Demonstration Pre-Selectors for Themselves
description: >-
  [ICML 2025][in-context learning] 提出 FEEDER（FEw yet Essential Demonstration prE-selectoR），一个基于"充分性"和"必要性"度量的示例预选框架，利用 LLM 自身能力从训练数据中识别代表性子集，在 ICL 和微调两个场景下均可减少 20%+ 数据量同时保持甚至提升性能。
tags:
  - ICML 2025
  - in-context learning
  - demonstration selection
  - data pre-selection
  - "sufficiency & necessity"
  - bilevel optimization
---

# Large Language Models are Demonstration Pre-Selectors for Themselves

**会议**: ICML 2025  
**arXiv**: [2506.06033](https://arxiv.org/abs/2506.06033)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: in-context learning, demonstration selection, data pre-selection, sufficiency & necessity, bilevel optimization

## 一句话总结

提出 FEEDER（FEw yet Essential Demonstration prE-selectoR），一个基于"充分性"和"必要性"度量的示例预选框架，利用 LLM 自身能力从训练数据中识别代表性子集，在 ICL 和微调两个场景下均可减少 20%+ 数据量同时保持甚至提升性能。

## 研究背景与动机

In-context learning（ICL）通过选取少量示例（demonstration）作为 prompt 上下文，让 LLM 在不微调的情况下完成下游任务。核心挑战是**如何从大规模训练数据中选出最具代表性的示例**。

现有方法存在两大问题：

**计算开销大**：现有示例选择器（如基于相似度、多样性、聚类的方法）需要对每个测试查询从整个训练集中反复检索，随着 shot 数增加和选择标准变复杂，计算代价急剧上升。

**忽略 LLM 自身特性**：不同 LLM 具有不同的能力和知识领域，选择示例时应考虑特定 LLM 的特点，而非通用的相似度/多样性度量。

作者观察到：与其每次查询都从全量训练集中选择，不如先用 LLM 自身做一次**预选（pre-selection）**，筛出一个小而精的代表性子集（FEEDER subset），后续所有查询都在这个子集上做 demonstration selection，从而同时提升效率和效果。

## 方法详解

### 整体框架

FEEDER 将 demonstration selection 分为两个阶段：

1. **预选阶段（Pre-selection Stage）**：从全量训练集 $\mathcal{D}_{\text{TRAIN}}$ 中预选出代表性子集 $\mathcal{D}_{\text{FEEDER}}$，目标是让子集尽可能小同时能充分代表全量数据。
2. **选择阶段（Selection Stage）**：在 $\mathcal{D}_{\text{FEEDER}}$ 上应用现有的 demonstration selector（如 Random、Similarity、Diversity 等），为具体测试输入选取 n-shot 示例。

此外，FEEDER 还可应用于**微调场景**：用预选子集微调 LLM，通过双层优化（bi-level optimization）交替优化子集选择和模型参数。

### 关键设计

#### 1. **充分性度量（Sufficiency Metric）**：评估示例的代表能力

核心思想：如果把样本 $(x_n, y_n)$ 作为上下文"插入"（plug in）给 LLM，能让 LLM 对另一个样本 $x_m$ 产生正确输出，则称 $(x_n, y_n)$ 对 $(x_m, y_m)$ 是**充分的**。

形式化定义：$Y_{x_m}=1 \mid \text{plug}((x_n, y_n)); C, S=(Y_{x_m}=0)$

即在原始状态 LLM 对 $x_m$ 回答错误的条件下，插入 $(x_n, y_n)$ 后 LLM 能纠正答案。充分性回答了一个关键问题：**这个示例是否足以代表其他样本？**

#### 2. **必要性度量（Necessity Metric）**：评估示例的不可替代性

核心思想：如果把已在上下文中的样本 $(x_n, y_n)$ "拔出"（unplug），导致 LLM 对 $x_m$ 的输出由正确变为错误，则称 $(x_n, y_n)$ 对 $(x_m, y_m)$ 是**必要的**。

形式化定义：$Y_{x_m}=0 \mid \text{unplug}((x_n, y_n)); C, S=(Y_{x_m}=1)$

必要性解决了另一个问题：**移除这个示例是否会导致信息缺失？** 如果某个示例不是必要的，说明它提供的信息是冗余的，可以安全移除。

#### 3. **树形近似算法（Tree-based Approximation）**：高效发现 FEEDER 子集

直接枚举所有可能子集的复杂度为 $O(2^N)$，不可行。作者设计了一个从底向上的树形算法：

- **初始化**：每个训练样本作为树的底层叶节点
- **每一轮（round）**：对当前层的节点两两配对，用 LLM 检查充分性关系
    - 若 $W_i$ 和 $W_j$ 互相充分：保留元素更少的那个
    - 若仅单向充分（如 $W_i$ 对 $W_j$ 充分）：保留充分的一方 $W_i$
    - 若互不充分：合并为 $W_i \cup W_j$
- **终止**：当只剩一个节点时停止，其包含的样本即为 FEEDER 子集

复杂度为 $O(RK \log_2 |\mathcal{D}_{\text{TRAIN}}|)$，其中 $R$ 为算法运行次数，$K$ 为树深度。实验表明 $K=1, R=1$ 已足够，简化为 $O(\log_2 |\mathcal{D}_{\text{TRAIN}}|)$。

作者还证明了 **Proposition 1**：在充分性传递性假设下，树形算法产生的子集仍能充分代表整个训练集。

### 损失函数 / 训练策略

**ICL 设置**下的优化目标为最小化 FEEDER 子集大小，同时保证其在训练集上的表现不低于全量数据：

$$\min_{\mathcal{D}_{\text{FEEDER}} \subseteq \mathcal{D}_{\text{TRAIN}}} |\mathcal{D}_{\text{FEEDER}}|, \quad \text{s.t.} \; \mathcal{L}(\mathcal{D}_{\text{FEEDER}}, \mathcal{D}_{\text{TRAIN}}) \leq \mathcal{L}(\mathcal{D}_{\text{TRAIN}}, \mathcal{D}_{\text{TRAIN}})$$

**微调设置**下的双层优化：
- **外层**：冻结 LLM，用树形算法更新 $\mathcal{D}_{\text{FEEDER}}$
- **内层**：固定 $\mathcal{D}_{\text{FEEDER}}$，微调 LLM 参数

两层交替迭代，实现子集选择和模型优化的联合提升。

## 实验关键数据

### 主实验

实验覆盖 6 个文本分类数据集（SST-2, SST-5, COLA, TREC, SUBJ, FPB）、推理数据集 GSM8K、语义解析 SMCALFlow、科学问答 GPQA。LLM 涵盖 GPT-2 (335M/774M)、GPT-neo (1.3B)、GPT-3 (6B)、Gemma-2 (2B)、Llama-2 (7B)、Llama-3 (8B)、Qwen-2.5 (32B)。

**ICL 设置关键结果**（以 Gemma-2 (2B) 在文本分类为例，n=5 shots）：

| 数据集 | 选择器 | $\mathcal{D}_{\text{TRAIN}}$ | $\mathcal{D}_{\text{FEEDER}}$ | 提升 |
|--------|--------|------|------|------|
| SUBJ | Similarity | 91.5 | **94.5** | +3.0 |
| SST-2 | Similarity | 80.5 | **83.6** | +3.1 |
| COLA | Similarity | 67.2 | **77.6** | +10.4 |
| GSM8K | Similarity | 20.31 | **22.58** | +2.27 |
| SMCALFlow | Diversity | 27.89 | **32.54** | +4.65 |

**Qwen-2.5 (32B) 在 GSM8K/GPQA（n=10 shots）**：

| 数据集 | 选择器 | $\mathcal{D}_{\text{TRAIN}}$ | $\mathcal{D}_{\text{FEEDER}}$ | 提升 |
|--------|--------|------|------|------|
| GSM8K | Similarity | 90.41 | **91.23** | +0.82 |
| GSM8K | Uncertainty | 90.20 | **91.96** | +1.76 |
| GPQA | Similarity | 46.85 | **47.80** | +0.95 |
| GPQA | Diversity | 46.71 | **47.93** | +1.22 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| $K=1, R=1$（默认） | 最优性价比 | 一次 one-shot 推理 + 单轮运行即可取得优异效果 |
| 增大 $R$（多轮运行） | 性能先升后降 | 子集过小会限制性能，存在数据量 vs 质量的权衡 |
| 增大 $K$（加深树） | 更鲁棒的提升趋势 | two-shot 推理提供了更严格的过滤机制 |
| 双层优化（微调） | 大幅提升（如 SUBJ: 89.2→95.6） | 微调能更有效地利用高质量子集 |
| 重复训练集实验 | FEEDER 最小化噪声影响 | 验证了 FEEDER 在数据冗余场景下的稳健性 |

### 关键发现

1. **FEEDER 可减少 20%~50% 训练数据**，同时在 ICL 中保持甚至超越全量数据的性能
2. **FEEDER + Similarity 可媲美甚至超越 Diversity/Clustering 等复杂选择器**，说明预选阶段比选择策略更重要
3. **微调场景提升更显著**：在双层优化框架下，FEEDER 可将 GPT-2 (0.8B) 在 SUBJ 上 10-shot 准确率从 94.0 提升到 95.5
4. **大 shot 数下优势明显**：当 shot 从 5 增到 10 时，全量数据经常出现性能下降（噪声/冗余），FEEDER 有效缓解此问题
5. **不同 LLM 需要不同的 FEEDER**：Case study 证实不同 LLM 对同一事实所需的充分必要示例不同，验证了 LLM-aware 预选的必要性
6. **时间复杂度近线性**：与数据量几乎线性相关，实际部署 $K=1,R=1$ 即可，开销可控

## 亮点与洞察

- **"预选"理念新颖**：将 demonstration selection 拆分为预选 + 选择两阶段的思路清晰且实用，预选阶段是 query-agnostic 的，一次计算后所有查询复用
- **因果推理视角**：借鉴因果推理中的 sufficiency 和 necessity 概念（与 do-calculus 中的 intervention 操作关联），为示例质量评估提供了理论基础
- **LLM-aware 设计**：预选过程考虑特定 LLM 的能力边界，不同模型产生不同的 FEEDER 子集，体现了以模型为中心的数据选择思想
- **双场景适用**：FEEDER 同时服务于 ICL 和微调，通过双层优化实现无缝衔接

## 局限与展望

1. **充分性传递性假设**：树形算法依赖充分性的传递性假设，实际中可能不完全成立，可能导致次优子集
2. **预选阶段本身有计算成本**：虽然复杂度为 $O(\log N)$，但每次充分性检查都需要 LLM 推理，对于大规模数据集仍有开销
3. **数据集和任务覆盖有限**：主要在文本分类和简单推理任务上验证，对生成任务（如摘要、翻译）的效果未知
4. **可能放大模型偏见**：FEEDER 依赖 LLM 自身判断来筛选数据，如果模型存在偏见，预选可能进一步强化偏见（论文 Impact Statement 中也承认了这点）
5. **未探索与更先进选择器的组合**：如与 retrieval-augmented ICL 方法结合的效果

## 相关工作与启发

- **Demonstration Selection**：FEEDER 与现有的 similarity/diversity/clustering/uncertainty 选择器正交，可作为它们的前置步骤
- **Core-set Selection**：FEEDER 的子集选择与 core-set selection 文献（Feldman, 2020; Guo et al., 2022）相呼应，但引入了 LLM 特定的评估标准
- **Data-centric AI**：体现了数据质量 > 数据数量的理念，与 Alpagasus（Chen et al., 2023）等数据筛选工作异曲同工
- **启发**：这种 LLM-as-judge 的范式（用 LLM 评估数据对 LLM 的价值）可以推广到其他数据选择场景，如训练数据清洗、课程学习等

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 预选阶段 + sufficiency/necessity 框架有新意，但核心思路是子集选择的变体
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖 8 种 LLM（335M~32B）、9 个数据集、6 种选择器，ICL+微调双场景
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，形式化严谨，但符号较多读起来有一定门槛
- **价值**: ⭐⭐⭐⭐ — 实用性强，即插即用的预选模块，但核心提升在小模型上更明显

<!-- RELATED:START -->

## 相关论文

- [Towards Effective and Efficient Continual Pre-training of Large Language Models](../../ACL2025/llm_pretraining/towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [Large Vocabulary Size Improves Large Language Models](../../ACL2025/llm_pretraining/large_vocabulary_size_improves_large_language_models.md)
- [Scalable Fingerprinting of Large Language Models](../../NeurIPS2025/llm_pretraining/scalable_fingerprinting_of_large_language_models.md)
- [The Curse of Depth in Large Language Models](../../NeurIPS2025/llm_pretraining/the_curse_of_depth_in_large_language_models.md)
- [Language Models over Canonical Byte-Pair Encodings](language_models_over_canonical_byte-pair_encodings.md)

<!-- RELATED:END -->
