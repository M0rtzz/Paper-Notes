---
title: >-
  [论文解读] Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees
description: >-
  [ICML 2025][图学习][图基础模型] 提出 Task-Tree 作为统一学习实例，通过引入虚拟任务节点将节点/边/图级任务对齐到同一表示空间，配合重构目标预训练 GNN，构建图基础模型 GIT，在 32 个图、5 个领域上实现微调/上下文学习/零样本三种范式的跨域跨任务泛化。
tags:
  - "ICML 2025"
  - "图学习"
  - "图基础模型"
  - "Task-Tree"
  - "跨任务泛化"
  - "图神经网络"
  - "零样本学习"
---

# Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees

**会议**: ICML 2025  
**arXiv**: [2412.16441](https://arxiv.org/abs/2412.16441)  
**代码**: [GIT](https://github.com/Zehong-Wang/GIT)  
**领域**: 图学习  
**关键词**: 图基础模型, Task-Tree, 跨任务泛化, GNN预训练, 零样本学习

## 一句话总结

提出 Task-Tree 作为统一学习实例，通过引入虚拟任务节点将节点/边/图级任务对齐到同一表示空间，配合重构目标预训练 GNN，构建图基础模型 GIT，在 32 个图、5 个领域上实现微调/上下文学习/零样本三种范式的跨域跨任务泛化。

## 研究背景与动机

**领域现状**：基础模型在 NLP（LLM）和 CV（LVM）中已高度成功，它们通过在大规模数据上预训练来捕获可迁移的模式（如图像中的纹理轮廓、文本中的 token 语义）。然而，图结构数据的基础模型仍处于早期阶段。

**现有痛点**：图数据的核心难点在于两方面的异质性：(1) **特征/结构异质性**——不同领域的图编码完全不同的现象（社交网络 vs 分子图）；(2) **任务异质性**——图任务作用于不同层级的学习单元（节点、边、整图），难以在统一模型中兼容。现有方法要么基于 graphon 理论（假设过强、计算不可行），要么基于子图/子结构提取（MP-GNN 无法有效编码子结构，且计算开销大）。

**核心矛盾**：子图方法需要额外存储和编码诱导子图，增加时间和内存成本；同时 MP-GNN 在子结构学习上表达力有限，导致跨任务泛化效果不佳。

**本文目标** 如何找到一种统一的学习实例，能够对齐节点/边/图级任务，使得 GNN 能够高效地在其上预训练并迁移到下游任务？

**切入角度**：作者从 MP-GNN 的学习动态出发——在任意图任务中，GNN 的预测都依赖于"任务相关节点"的嵌入（节点任务→目标节点，边任务→端点，图任务→所有节点）。可以引入一个虚拟任务节点连接所有任务相关节点，以此节点为根的计算树即为 Task-Tree。

**核心 idea**：用 Task-Tree（虚拟任务节点+计算树）替代子图作为统一的跨任务学习实例，高效且理论可证地实现图基础模型的预训练与迁移。

## 方法详解

### 整体框架

输入是来自多个领域（学术网络、电商、知识图谱、分子图、时序图）的文本属性图；通过 Sentence-BERT 将所有节点特征编码到共享的 768 维空间；为每个学习实例（节点/边/图）构造 Task-Tree；用重构目标在多域 Task-Tree 上预训练 GNN 编码器（GIT-G）；可选地通过指令微调进行领域特化（GIT-S）；最后在下游任务上通过微调/上下文学习/零样本进行评估。

### 关键设计

1. **Task-Tree 构造与编码**:

    - 功能：为任意图任务实例构造统一的学习单元
    - 核心思路：对于节点/边/图级任务，分别确定任务相关节点集合，引入虚拟任务节点连接所有任务相关节点，形成 Task-Tree。编码时使用 MEAN 聚合：$\mathbf{z}^t = \frac{1}{n}\sum_{i=1}^{n}\phi(T_i)$，其中 $T_i$ 是第 $i$ 个任务相关节点的计算树。操作上只需在原图中添加虚拟节点和边，然后正常做消息传递
    - 设计动机：相比子图方法，Task-Tree 有三大优势——(1) **可学习性**：树结构天然可被 MP-GNN 有效编码；(2) **统一性**：无缝适用于不同级别任务；(3) **高效性**：只需在原图上添加虚拟节点，避免子图提取和存储的开销

2. **Task-Tree 重构预训练（GIT-G）**:

    - 功能：通过自监督重构目标在多域 Task-Tree 上预训练 GNN
    - 核心思路：对每个 Task-Tree 施加两种数据增强（随机边掩码 + 属性掩码）生成两个视图 $\hat{T}$ 和 $\tilde{T}$，用编码器 $\phi$ 分别编码，然后通过 stop-gradient 的对称重构损失让两个视图互相预测，同时加 KL 正则化使嵌入投影到共享空间：$\mathcal{L} = \frac{1}{2n}\sum_i [\|\rho(g(\hat{z}_i)) - \text{sg}[\rho(\tilde{z}_i)]\|^2 + \|\rho(g(\tilde{z}_i)) - \text{sg}[\rho(\hat{z}_i)]\|^2] + \sum_i D_{KL}(h \| z_i)$
    - 设计动机：重构目标能捕获 Task-Tree 中的腐蚀不变语义，KL 正则化确保不同 Task-Tree 的嵌入被映射到共享空间

3. **领域特化指令微调（GIT-S）**:

    - 功能：将通用模型适配到特定领域
    - 核心思路：在目标领域的 Task-Tree 上用监督微调损失进行后训练：$\mathcal{L}_{SFT} = \frac{1}{n}\sum_i \kappa(\phi^*(T_i), \psi(T_i))$，其中 $\psi(T_i)$ 是由 LLM 编码的标签描述嵌入作为指令
    - 设计动机：理论（泛化界）表明减小预训练和微调分布之间的差距可以提升泛化性；同领域图的 Task-Tree 分布相似，因此领域特化能有效缩小分布差距

### 理论分析

论文提供了三个核心定理：(1) **稳定性**——子树结构相似的 Task-Tree 会产生相似的嵌入，且 Task-Tree 的宽度对表示距离影响不大；(2) **可迁移性**——预训练中学到的知识可以按 $O(1)$ 常数比例迁移到下游任务；(3) **泛化界**——下游风险受预训练质量、分布差距和微调样本数共同约束，支持少样本微调即可获得良好泛化。

## 实验关键数据

### 主实验

| 领域 | 设置 | GIT-G | GIT-S | OFA | GraphMAE | Sup. GNN |
|------|------|-------|-------|-----|----------|----------|
| 学术网络 | 零样本 | 14.88 | **23.45** | 13.98 | 15.42 | - |
| 学术网络 | 3-shot | 54.00 | **55.18** | 45.93 | 49.25 | - |
| 学术网络 | 微调 | 75.82 | **75.88** | 72.18 | 73.81 | 73.57 |
| 分子图 | 零样本 | 53.34 | **62.83** | 50.49 | 47.19 | - |
| 全部平均 | 微调 | **75.37** | **75.72** | 73.08 | 72.79 | 72.25 |

### 与 SOTA 图基础模型对比

| 方法 | 学术网络 | 知识图谱 | 分子图 |
|------|---------|---------|--------|
| GraphPrompt+ | 74.80 | 74.78 | 72.99 |
| All in One | 75.25 | 74.92 | 71.87 |
| OpenGraph | 74.64 | 71.38 | 72.84 |
| AnyGraph | 75.01 | 74.30 | 72.49 |
| **GIT-G** | **75.82** | **75.73** | **74.57** |

### 消融实验

| 训练策略 | 零样本 | 3-shot | 微调 |
|---------|--------|--------|------|
| Base Model (GIT) | 15.36 | 53.31 | 75.53 |
| Expert Model (GIT) | 18.38 | 55.10 | 75.47 |
| General Model (GIT) | 14.88 | 54.00 | 75.82 |
| Specialized Model (GIT) | **23.45** | **55.18** | **75.88** |
| General Model (OFA) | 13.98 | 45.93 | 72.18 |
| Specialized Model (OFA) | 20.05 | 46.87 | 73.04 |

### 关键发现

- GIT 的通用模型（General Model）性能保持稳定，不像 GraphMAE/OFA 从 Base → General 时会显著下降，说明 Task-Tree 能有效缓解负迁移
- 领域特化（GIT-S）在零样本和少样本设置下提升尤为明显（零样本从 14.88 → 23.45），但对微调影响较小
- GIT-S 在分子图领域接近领域专家 GIMLET（62.83 vs 64.15），在知识图谱上接近 Ultra（67.80 vs 68.53）
- Task-Tree 在所有评估中一致优于子图方法，同时计算效率更高

## 亮点与洞察

- **Task-Tree 的优雅设计**：通过虚拟节点将跨任务异质性问题转化为"在扩展图上做消息传递"，实现了理论上的统一性和工程上的高效性。核心洞察是 GNN 的预测本质上依赖于计算树，而 Task-Tree 正是这一计算过程的自然抽象
- **理论驱动的框架设计**：稳定性、可迁移性、泛化性三个定理不是事后验证，而是真正指导了模型设计——例如泛化界中的分布差距项直接启发了领域特化策略
- **指令微调思路可迁移**：用 LLM 编码标签描述作为指令来做图领域的 SFT，这个思路可以直接迁移到其他结构化数据（如知识图谱补全、蛋白质功能预测）

## 局限与展望

- Task-Tree 的有效性依赖于文本属性图的假设——所有节点特征必须先通过 Sentence-BERT 对齐到共享空间，但许多实际图数据没有文本属性（如纯数值特征的分子图），特征对齐问题未真正解决
- 预训练数据规模相对有限（约 30 个图），与 NLP/CV 的基础模型相比差距巨大，scaling law 在图领域是否成立尚未验证
- GIT-S 的领域特化对标注数据有一定依赖，在完全无标注的新领域中如何适配是开放问题

## 相关工作与启发

- **vs OFA (Liu et al., 2024a)**：OFA 也追求跨域图学习，但通过子图提取和统一的 prompt 机制，计算开销大。GIT 用 Task-Tree 直接在原图上操作，效率更高，且有理论支撑
- **vs GFT (Wang et al., 2024b)**：GFT 也引入计算树来对齐异质图任务，但采用模型驱动设计（可学习词汇表+多面重构目标）。GIT 是理论驱动，两者互补
- **vs AnyGraph (Xia & Huang, 2024)**：AnyGraph 通过统一特征空间处理跨域问题，GIT 则聚焦于任务对齐。GIT 在实验中全面超越 AnyGraph

## 评分

- 新颖性: ⭐⭐⭐⭐ Task-Tree 概念虽然和 GFT 的计算树有关联，但理论框架和简洁的实现是新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 32 个图、5 个领域、3 种评估范式，覆盖面足够广
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但符号较多需要反复对照
- 价值: ⭐⭐⭐⭐ 为图基础模型提供了理论基础和实用框架，但特征对齐假设限制了通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Mixed-Curvature Decision Trees and Random Forests](mixed-curvature_decision_trees_and_random_forests.md)
- [\[ICML 2025\] WILTing Trees: Interpreting the Distance Between MPNN Embeddings](wilting_trees_interpreting_the_distance_between_mpnn_embeddings.md)
- [\[ICML 2025\] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](from_rag_to_memory_non-parametric_continual_learning_for_large_language_models.md)
- [\[NeurIPS 2025\] Reasoning Meets Representation: Envisioning Neuro-Symbolic Wireless Foundation Models](../../NeurIPS2025/graph_learning/reasoning_meets_representation_envisioning_neuro-symbolic_wireless_foundation_mo.md)
- [\[ICML 2025\] TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration](toping_topologically_interpretable_graph_learning_via_persistent_rationale_filtr.md)

</div>

<!-- RELATED:END -->
