---
title: >-
  [论文解读] MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation
description: >-
  [AAAI 2026][图学习][冷启动推荐] 提出 MoToRec，将多模态推荐重新定义为离散语义分词任务，通过稀疏正则化的残差量化VAE（RQ-VAE）将原始多模态特征转化为可组合的离散语义编码，结合自适应稀有度放大和层级多源图编码器，有效解决物品冷启动问题。
tags:
  - AAAI 2026
  - 图学习
  - 冷启动推荐
  - 多模态推荐
  - 离散语义分词
  - 残差量化VAE
  - 图神经网络
---

# MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation

**会议**: AAAI 2026  
**arXiv**: [2602.11062](https://arxiv.org/abs/2602.11062)  
**代码**: 无  
**领域**: 图学习 / 推荐系统  
**关键词**: 冷启动推荐, 多模态推荐, 离散语义分词, 残差量化VAE, 图神经网络

## 一句话总结

提出 MoToRec，将多模态推荐重新定义为离散语义分词任务，通过稀疏正则化的残差量化VAE（RQ-VAE）将原始多模态特征转化为可组合的离散语义编码，结合自适应稀有度放大和层级多源图编码器，有效解决物品冷启动问题。

## 研究背景与动机

### 问题背景
图神经网络（GNN）已成为现代推荐系统的基石，但其成功依赖于密集的历史交互数据。在数据稀疏场景下，尤其是**物品冷启动问题**（新物品缺乏或没有交互历史）中，GNN性能急剧下降。

### 现有方法的局限
多模态信息（视觉、文本）为缓解冷启动提供了希望，但现有方法存在共性缺陷：

**语义迷雾（Semantic Fog）问题**：现有方法在高维连续空间中进行多模态对齐，本质上是将"红色T恤"这样的概念从像素向量和文本向量映射到高维空间中单一连贯的点，这个过程对噪声非常敏感且不可靠

**传统方法的演进**：从VBPR的简单拼接、MMGCN的模态专用图、LATTICE的物品-物品语义图，到FREEDOM/BM3的对比学习，尽管架构多样化，但**本质仍是在连续空间做噪声对齐**

**OOD表示问题**：即使利用LLM作为特征提取器，对齐这些带噪声的连续嵌入仍会产生次优的分布外（OOD）表示，冷启动物品尤为严重

### 核心洞察

**离散表示优于连续对齐**。作者提出将多模态特征转化为结构化的离散token序列，每个token代表一个解耦的语义概念（如风格：极简，颜色：红色），从根本上避免连续空间的对齐噪声问题。

## 方法详解

### 整体框架

MoToRec 包含三个核心组件：
1. **自适应稀有度放大（Adaptive Rarity Amplification）**：动态加权学习信号，优先关注冷启动物品
2. **稀疏正则化多模态分词器（Sparsely-Regularized Multimodal Tokenizer）**：基于RQ-VAE将原始多模态特征转化为离散语义编码
3. **层级多源图编码器（Hierarchical Multi-Source Graph Encoder）**：融合语义编码与协同信号

输入方面，每个物品具有视觉特征 $\mathbf{f}_i^v$（来自BEiT）和文本特征 $\mathbf{f}_i^t$（来自BGE），目标是学习用户嵌入 $\mathbf{e}_u$ 和物品嵌入 $\mathbf{e}_i$，通过点积预测相关性分数。

### 关键设计

#### 1. **自适应稀有度放大（ARA）**：解决流行度偏差，放大冷启动学习信号

推荐数据集天然存在流行度偏差，模型常忽视长尾稀有物品。ARA通过度感知的动态加权方案解决此问题。

**步骤**：
- 计算每个物品的交互度 $d_i = \sum_{u \in \mathcal{U}} R_{ui}$
- 设置领域阈值 $\tau$，将 $d_i < \tau$ 的物品标记为冷启动物品
- 定义物品权重（反对数加权）：

$$w_i = \begin{cases} (\log_2(d_i + 2))^{-1} & \text{if } c_i = 1 \text{ and } d_i > 0 \\ 1.0 & \text{otherwise} \end{cases}$$

**设计动机**：反对数加权压缩度范围，+2偏移稳定小值。这使得交互越少的物品权重越大，但完全无交互的物品（零样本）不额外加权（它们依赖内容特征的整体学习质量）。

#### 2. **稀疏正则化RQ-VAE分词器**：将连续特征转化为可解释的离散编码

这是MoToRec的核心模块。

**残差量化过程**：
- 对每种模态 $m \in \{v, t\}$，编码器 $E_m$（MLP）将原始特征投影到潜空间：$\mathbf{z}_{e,i}^m = E_m(\mathbf{f}_i^m)$
- 级联 $N_q$ 个量化器进行逐级残差量化：

$$q_i^{(k)} = \arg\min_{c \in C_m^{(k)}} \|r_i^{(k-1)} - c\|_2^2, \quad r_i^{(k)} = r_i^{(k-1)} - q_i^{(k)}$$

- 最终量化表示 $\mathbf{z}_{q,i}^m = \sum_{k=1}^{N_q} \mathbf{q}_i^{(k)}$ 是所有量化码本向量的组合

**稀疏诱导正则化（关键创新）**：
- 为避免码本产生纠缠表示，引入KL散度惩罚，让码本使用的聚合后验分布趋近于均值为 $\rho$ 的稀疏Bernoulli先验：

$$\mathcal{L}_{\text{sparse}} = \sum_{j=1}^{K} \text{KL}(\rho \| \hat{\rho}_j) = \sum_{j=1}^{K} \left(\rho \log \frac{\rho}{\hat{\rho}_j} + (1-\rho) \log \frac{1-\rho}{1-\hat{\rho}_j}\right)$$

- 理论基础：KL惩罚通过最小化码本激活之间的互信息来促进解耦表示，类似于离散潜空间中的非线性独立分量分析

**分词器训练目标**：

$$\mathcal{L}_{\text{RQ-VAE}}^m = \underbrace{\|\mathbf{f}_i^m - D_m(\mathbf{z}_{q,i}^m)\|_2^2}_{\text{重建}} + \beta \underbrace{\|\mathbf{z}_{e,i}^m - \text{sg}(\mathbf{z}_{q,i}^m)\|_2^2}_{\text{承诺}} + \gamma \underbrace{\mathcal{L}_{\text{sparse}}}_{\text{稀疏}}$$

#### 3. **层级多源图编码器**：对齐语义编码与协同偏好

**模态内解耦传播**：维护三个并行的解耦传播通道：
- 视觉通道：用量化视觉嵌入 $\{\mathbf{z}_{q,i}^v\}$ 初始化，捕捉审美偏好
- 文本通道：用量化文本嵌入 $\{\mathbf{z}_{q,i}^t\}$ 初始化，学习物品属性
- 协同通道：用标准可学习ID嵌入初始化，专门建模纯协同信号

每个通道内使用 LightGCN 传播规则进行 $L$ 层嵌入精炼：$\mathbf{E}^{(l+1)} = (\mathbf{D}^{-1/2}\tilde{\mathbf{A}}\mathbf{D}^{-1/2})\mathbf{E}^{(l)}$

**跨源融合**：采用混合融合策略：

$$\mathbf{e}_i^m = \alpha \cdot \text{CONCAT}(\mathbf{i}_v, \mathbf{i}_t) + (1-\alpha) \cdot \text{Attention}(\mathbf{i}_v, \mathbf{i}_t)$$

超参数 $\alpha$ 平衡静态特征保留与动态上下文感知重加权，然后通过门控残差连接整合协同嵌入。

### 损失函数 / 训练策略

最终损失函数整合四部分：

$$\mathcal{L} = \mathcal{L}_{\text{BPR}} + \lambda_{cl} \mathcal{L}_{\text{CL}} + \lambda_{rq} \sum_{m \in \{v,t\}} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} w_i \cdot \mathcal{L}_{\text{RQ-VAE},i}^m + \lambda_{reg} \|\Theta\|_2^2$$

- **BPR排序损失**：优化用户对正负物品的相对排序
- **InfoNCE对比损失**：拉近同节点的增强视图，推开负样本
- **加权RQ-VAE损失**：冷启动物品权重 $w_i$ 更大，确保其分词质量
- **L2正则**：防止过拟合

## 实验关键数据

### 实验设置
- **数据集**：Amazon Baby、Sports、Clothing（稀疏度均>99.88%）
- **评估协议**：8:1:1训练/验证/测试划分，冷启动组包含训练集中交互少于10次的测试物品
- **指标**：Recall@N 和 NDCG@N（N=10, 20）

### 主实验

| 数据集 | 指标 | MoToRec | LGMRec (SOTA) | LPIC (SOTA) | 最大提升 |
|--------|------|---------|---------------|-------------|---------|
| Baby | R@20 | **0.1077** | 0.0989 | 0.0977 | +8.57% |
| Baby | N@20 | **0.0473** | 0.0430 | 0.0422 | +10.00% |
| Sports | R@20 | **0.1163** | 0.1068 | 0.1113 | +4.49% |
| Sports | N@20 | **0.0529** | 0.0477 | 0.0485 | +9.07% |
| Clothing | R@20 | **0.1014** | 0.0828 | 0.0928 | +7.76% |
| Clothing | N@20 | **0.0456** | 0.0371 | 0.0405 | +8.57% |

相比ID-only模型（LightGCN），提升高达**88%**。在冷启动场景下 N@20 提升**12.58%**。

### 消融实验

| 配置 | Baby N@20 | Baby Cold N@20 | Sports N@20 | Clothing N@20 | 说明 |
|------|-----------|---------------|-------------|---------------|------|
| MoToRec (full) | 0.0473 | 0.0147 | 0.0529 | 0.0456 | 完整模型 |
| w/o RQ-VAE | 0.0398 | 0.0092 | 0.0422 | 0.0362 | **降幅最大**，验证离散分词核心价值 |
| w/o ARA | 0.0437 | 0.0111 | 0.0466 | 0.0397 | 冷启动性能显著下降 |
| w/o Sparsity | 0.0430 | 0.0109 | 0.0455 | 0.0389 | 稀疏约束对解耦表示至关重要 |
| w/o CL | 0.0455 | 0.0118 | 0.0515 | 0.0438 | 对比损失改善嵌入空间 |
| w/o HF | 0.0449 | 0.0120 | 0.0468 | 0.0401 | 混合融合优于单一策略 |

### 关键发现

1. **移除RQ-VAE导致最严重性能退化**（冷启动 N@20 从0.0147降至0.0092），直接验证了离散语义分词优于连续特征映射的核心论点
2. **超参数灵敏度因数据集而异**：稀疏的Baby偏好中等稀疏度（γ=0.05）和紧凑码本（K=512），视觉丰富的Clothing需更低稀疏度（γ=0.01）和更大码本（K=1024）
3. **t-SNE可视化**证实完整模型学到了更有组织的语义流形，冷启动物品不再是孤立的异常点，而是无缝融入结构中
4. **案例研究**验证了码本学到了人类可解释的概念，如代码 `<c_121>` 对应"红色"，`<a_34>` 对应"T恤"，新物品可组合这些编码进行表示

## 亮点与洞察

1. **范式转换**：将推荐从"连续空间对齐"转换为"离散语义分词"，这个视角非常新颖且直觉清晰。离散化天然具有去噪和可解释性优势
2. **稀疏正则化促进解耦**：通过KL散度惩罚迫使码本使用趋近稀疏先验，实现了离散潜空间中的独立分量分析效果
3. **三通道解耦传播**：避免过早的模态干扰，分别保留视觉偏好、文本属性和纯协同信号的语义纯度
4. **效率可接受**：训练时间11.33s/epoch，仅比LightGCN多74%开销，推理端效率与其他高性能模型相当

## 局限与展望

1. **冷启动阈值 τ=10 是硬设定**，不同数据集可能需要不同阈值，缺乏自适应调节机制
2. **只处理物品冷启动**，未涉及用户冷启动问题
3. **码本大小和量化级数需大量超参调优**，实际部署中调参成本较高
4. **仅在Amazon数据集上验证**，未在更多样化的推荐场景（如新闻推荐、短视频推荐）中测试泛化能力
5. **可探索方向**：将离散分词与LLM推荐系统结合、探索多码本共享机制、引入用户画像的离散化表示

## 相关工作与启发

- **VQ-Rec**率先在推荐中使用向量量化但用于序列推荐，MoToRec首次将其用于学习解耦的多模态组合表示以解决冷启动
- **MeLU等元学习方法**仅适用于few-shot场景，无法处理零交互的zero-shot冷启动
- 离散分词的思路可迁移到**多模态检索、跨模态生成**等任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 离散化视角在推荐冷启动中属首创
- **实验充分度**: ⭐⭐⭐⭐ — 三数据集+全面消融+可视化+效率分析，但数据集类型单一
- **写作质量**: ⭐⭐⭐⭐⭐ — 动机清晰，"语义迷雾"比喻直觉且准确
- **实用价值**: ⭐⭐⭐⭐ — 训练效率可接受，但超参调优成本较高

<!-- RELATED:START -->

## 相关论文

- [NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)
- [Graph Tokenization for Bridging Graphs and Transformers](../../ICLR2026/graph_learning/graph_tokenization_for_bridging_graphs_and_transformers.md)
- [Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)

<!-- RELATED:END -->
