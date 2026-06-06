---
title: >-
  [论文解读] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning
description: >-
  [NeurIPS 2025][图学习][知识图谱] 提出MoEMeta框架，通过混合专家模型学习全局共享的关系原型实现跨任务泛化，结合任务定制的投影适应机制捕获局部上下文，在三个KG基准上达到SOTA。
tags:
  - "NeurIPS 2025"
  - "图学习"
  - "知识图谱"
  - "少样本关系学习"
  - "元学习"
  - "混合专家"
  - "任务适应"
---

# MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.23013](https://arxiv.org/abs/2510.23013)  
**代码**: [GitHub](https://github.com/alexhw15/MoEMeta)  
**领域**: 图学习  
**关键词**: 知识图谱, 少样本关系学习, 元学习, 混合专家, 任务适应

## 一句话总结

提出MoEMeta框架，通过混合专家模型学习全局共享的关系原型实现跨任务泛化，结合任务定制的投影适应机制捕获局部上下文，在三个KG基准上达到SOTA。

## 研究背景与动机

少样本知识图谱关系学习(FSRL)旨在仅凭少量训练三元组推理新关系。现有基于MAML的方法存在两个关键缺陷：

**忽视跨任务共享模式**：假设元训练任务独立同分布(i.i.d.)，逐任务孤立学习元知识。但KG中的关系天然存在语义聚类——例如FatherOfPerson和BrotherOf共享"家庭纽带"主题，而ColorOf属于"物理属性"。忽略这种跨任务共性阻碍泛化

**全局参数缺乏灵活性**：使用单一全局初始化加梯度更新进行适应，而KG中的实体和关系展示多样化交互模式（1-1、1-N、N-1、N-N）。例如Elon Musk作为CeoOf和FatherOfPerson呈现完全不同的方面，共享初始化难以捕获如此不同的局部上下文

**核心挑战**：如何将全局共享知识与任务特定上下文解耦，同时实现有效泛化和快速适应？

## 方法详解

### 整体框架

MoEMeta包含三个核心组件：(1) 注意力邻居聚合增强实体表示；(2) MoE元知识学习器动态选择专家生成关系元；(3) 任务定制投影机制实现局部适应。全局参数 $\bm{\Phi}$ 在外循环优化，局部参数 $\bm{\eta}$ 在内循环每个任务上独立优化。

### 关键设计

1. **注意力邻居聚合**：利用一阶邻居的关系和实体信息增强目标实体表示

   对实体 $e$ 的邻居元组 $(r_i, e'_i)$，先拼接并变换：
    $\mathbf{c}'_i = \text{ReLU}(\mathbf{W} \mathbf{c}_i)$
   
   通过sigmoid门控选择信息性邻居：
    $g_i = \sigma(\bm{\beta}^T \mathbf{c}'_i)$
   
   加权聚合后与实体自身嵌入相加：
    $\mathbf{e} = \text{Aggregate}(\{g_i \cdot \mathbf{c}'_i\}_{i=1}^n) + \mathbf{e}$
   
   设计动机：KG中实体的含义高度依赖其关系上下文，邻居聚合可以提供更丰富的实体表示。

2. **MoE元知识学习**：通过全局共享的稀疏MoE学习可组合的关系原型

   对每个支持三元组，门控网络计算每个专家的相关性分数：
    $s_{i,j} = \text{softmax}(\text{Gate}(\mathbf{h}_i, \mathbf{t}_i; \bm{\theta}_g))$
   
   选择Top-N个专家加权组合生成关系表示：
    $\mathbf{r}_i = \frac{1}{N} \sum_{j=1}^{M} g_{i,j} f_j(\mathbf{h}_i, \mathbf{t}_i; \bm{\theta}_j)$
   
   最终关系元为所有支持三元组关系表示的均值：
    $\mathbf{R}_{\mathcal{T}_r} = \frac{1}{K} \sum_{i=1}^{K} \mathbf{r}_i$
   
   设计动机：不同关系可由相同"关系构建块"（专家）的不同组合表示，稀疏激活使专家专门化，类似关系自然选择相似的专家子集。

3. **任务定制局部适应**：通过可学习投影向量将嵌入投射到任务特定子空间

   为每个任务维护三个投影向量 $\mathbf{p}_h, \mathbf{p}_r, \mathbf{p}_t$，利用关系元进行调制：
    $\mathbf{h}'_i = \mathbf{h}_i + (\mathbf{p}_h^\top \mathbf{R}_{\mathcal{T}_r}) \cdot \mathbf{R}_{\mathcal{T}_r}$
    $\mathbf{R}'_{\mathcal{T}_r} = \mathbf{R}_{\mathcal{T}_r} + (\mathbf{p}_r^\top \mathbf{R}_{\mathcal{T}_r}) \cdot \mathbf{R}_{\mathcal{T}_r}$
    $\mathbf{t}'_i = \mathbf{t}_i + (\mathbf{p}_t^\top \mathbf{R}_{\mathcal{T}_r}) \cdot \mathbf{R}_{\mathcal{T}_r}$
   
   设计动机：受TransD启发，每个任务的关系约束不同，投影向量以极低参数量实现嵌入空间的任务特定偏移，避免过参数化。

### 损失函数 / 训练策略

评分函数采用 $\ell_2$ 距离：$\text{score}(h_i, t_i) = \|\mathbf{h}'_i + \mathbf{R}'_{\mathcal{T}_r} - \mathbf{t}'_i\|_2$

内循环（支持集上的适应）: margin-based损失更新 $\bm{\eta}$ 和 $\mathbf{R}_{\mathcal{T}_r}$：
$$\mathcal{L}(\mathcal{S}_r) = \sum_{(h_i,r,t_i) \in \mathcal{S}_r} \max\{0, \text{score}(h_i, t_i) + \gamma - \text{score}(h_i, t'_i)\}$$

外循环（查询集上的元更新）：查询损失反向传播更新全局参数 $\bm{\Phi}$。

## 实验关键数据

### 主实验 (Nell-One)

| 方法 | 1-shot MRR | 1-shot Hits@1 | 5-shot MRR | 5-shot Hits@1 |
|------|-----------|--------------|-----------|--------------|
| MetaR-P | 0.164 | 0.093 | 0.209 | 0.141 |
| GANA | 0.236 | 0.173 | 0.245 | 0.166 |
| HiRe | 0.288 | 0.184 | 0.306 | 0.207 |
| **MoEMeta** | **0.322** | **0.228** | **0.339** | **0.236** |
| 提升 | +11.8% | +23.9% | +10.8% | +14.0% |

### FB15K-One (3-shot)

| 方法 | MRR | Hits@1 | Hits@10 |
|------|-----|--------|---------|
| RelAdapter | 0.405 | 0.297 | 0.575 |
| **MoEMeta** | **0.423** | **0.302** | **0.651** |
| 提升 | +4.4% | +0.3% | +13.2% |

### 消融实验 (Nell-One)

| 配置 | 1-shot MRR | 5-shot MRR | 说明 |
|------|-----------|-----------|------|
| MoEMeta | 0.322 | 0.339 | 完整模型 |
| w/o N.A | 0.311 | 0.328 | 去除邻居聚合 |
| w/o MoE | 0.291 (↓9.6%) | 0.293 (↓13.6%) | 用MLP替换MoE |
| w/o L.A | 0.301 | 0.315 | 去除局部适应 |

### 关键发现

1. MoE模块是最关键组件——去除后MRR下降高达13.6%，证明跨任务共享模式学习的重要性
2. 门控值可视化显示：相似关系（如家庭类关系）激活相似的专家子集，不同类关系激活不同专家
3. 局部适应对N-1和N-N关系类型改善最显著，MRR分别提升8.1%和8.6%
4. 超参数分析表明32个专家、选择Top-5最优

## 亮点与洞察

1. **全局-局部解耦**：MoE负责"学什么可泛化"，投影负责"怎么适应"，两者互补
2. **关系原型概念**：将关系视为基本模式的组合而非独立学习，更符合KG的语义结构
3. **轻量适应**：仅3个投影向量（标量级参数）即实现有效任务适应，避免少样本过拟合

## 局限与展望

- 仅预测尾实体 $(h, r, ?)$，未考虑头实体预测
- TransE初始化的依赖可能限制对某些复杂关系模式的覆盖
- 专家网络结构较简单（两层MLP），更复杂的专家可能进一步提升
- 未探索跨数据集的迁移学习

## 相关工作与启发

- **元学习FSRL**: MetaR, HiRe, GANA, RelAdapter
- **度量学习FSRL**: GMatching, FAAN, NP-FKGC
- **MoE架构**: 稀疏门控MoE, Switch Transformer
- **KG嵌入**: TransE, TransD, DistMult, ComplEx

## 评分

- 新颖性: ⭐⭐⭐⭐ — MoE作为元学习器用于关系原型学习是新颖组合
- 实验充分度: ⭐⭐⭐⭐⭐ — 三数据集+16个基线+消融+可视化+关系类型分析
- 写作质量: ⭐⭐⭐⭐ — 动机阐述清晰，方法设计有理有据
- 价值: ⭐⭐⭐⭐ — 少样本KG推理的实用性强，框架可迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[NeurIPS 2025\] Moscat: Mixture of Scope Experts at Test for Generalizing Deeper GNNs](mixture_of_scope_experts_at_test_generalizing_deeper_graph_neural_networks_with_.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [\[ICLR 2026\] Relatron: Automating Relational Machine Learning over Relational Databases](../../ICLR2026/graph_learning/relatron_automating_relational_machine_learning_over_relational_databases.md)
- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](../../AAAI2026/graph_learning/commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)

</div>

<!-- RELATED:END -->
