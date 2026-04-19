---
title: >-
  [论文解读] Griffin: Towards a Graph-Centric Relational Database Foundation Model
description: >-
  [ICML 2025][自监督学习][关系数据库基础模型] Griffin 是首个面向关系数据库（RDB）的基础模型，通过将多表结构转化为异构图，结合统一编码器/解码器、交叉注意力和层级聚合的 MPNN，在 150M+ 行数据上进行自监督掩码补全预训练 + 联合 SFT，实现跨数据库、跨域、跨任务的泛化预测。
tags:
  - ICML 2025
  - 自监督学习
  - 关系数据库基础模型
  - 图神经网络
  - 自监督预训练
  - 跨表推理
  - 迁移学习
---

# Griffin: Towards a Graph-Centric Relational Database Foundation Model

**会议**: ICML 2025  
**arXiv**: [2505.05568](https://arxiv.org/abs/2505.05568)  
**代码**: [github.com/yanxwb/Griffin](https://github.com/yanxwb/Griffin)  
**领域**: 自监督学习  
**关键词**: 关系数据库基础模型, 图神经网络, 自监督预训练, 跨表推理, 迁移学习

## 一句话总结

Griffin 是首个面向关系数据库（RDB）的基础模型，通过将多表结构转化为异构图，结合统一编码器/解码器、交叉注意力和层级聚合的 MPNN，在 150M+ 行数据上进行自监督掩码补全预训练 + 联合 SFT，实现跨数据库、跨域、跨任务的泛化预测。

## 研究背景与动机

基础模型已在 NLP、CV、表格数据和图数据领域取得巨大成功，但**关系数据库（RDB）** 领域仍缺乏通用基础模型。RDB 中多表通过主键-外键（PK-FK）关联，具有以下挑战：

**结构复杂性**：多表间关系交错，简单的单表展平（flatten）会丢失大量关系信息

**特征空间异构**：不同 RDB 的分类特征、数值特征、文本特征语义各不相同，难以统一

**任务多样性**：分类和回归任务共存，不同任务的标签空间和数值范围差异巨大

**大规模时序约束**：实际 RDB 常含时间戳，预测须满足因果约束（仅用历史数据）

现有 GNN 方法（如 RDBTOGRAPH、ATJ-Net、GFS）虽将 RDB 转化为图进行建模，但都是针对特定任务训练的小模型，无法跨数据库泛化。Griffin 的目标是构建一个**一次预训练、多任务通用的 RDB 基础模型**。

## 方法详解

### 整体框架

Griffin 的流程分为三步：

1. **图构建**：将 RDB 每行映射为节点，PK-FK 关系映射为边，构建异构时序图
2. **子图采样**：给定目标列，采样以目标行为根的时序约束子图 $\mathcal{T}^{(L)}$，仅包含时间戳早于目标行的节点
3. **编码-MPNN-解码**：统一编码器处理异构特征 → MPNN 聚合多跳邻居信息 → 统一解码器输出预测

### 关键设计

#### 1. 统一数据编码器（Unified Data Encoder）

**分类/文本特征**：使用预训练文本编码器（Nomic Embed）将所有分类值和文本统一编码为 $d$ 维向量。与传统 one-hot 或单一 embedding 层不同，这种方式保留了丰富的语义信息，且通过余弦相似度可以度量不同类别间的语义距离。

**数值特征**：先用分位数归一化（Quantile Normalization）将数值映射为正态分布，再用预训练的 MLP 编码器将标量映射为 $d$ 维向量：

$$w = \text{ENC}(x) \in \mathbb{R}^d, \quad y = \text{DEC}(w) \in \mathbb{R}$$

训练时用 L1 损失 $|y - x|$ 最小化重构误差，并用 LayerNorm（无仿射参数）防止表示塌缩。预训练完成后 ENC 和 DEC 固定不参与后续训练。

**元数据编码**：表名、列名、边类型等元信息也通过文本编码器编码，作为额外的节点和边特征。

**任务表示**：用目标列的列名经文本编码器生成任务嵌入 $t \in \mathbb{R}^d$，使模型能区分同一行上不同预测任务。

编码输出：每个节点 $i$ 关联特征张量 $x_i \in \mathbb{R}^{L_i \times d}$、元数据张量 $m_i \in \mathbb{R}^{L_i \times d}$，每条边携带关系向量 $e_r \in \mathbb{R}^d$。

#### 2. 改进的 MPNN 架构

**交叉注意力模块（Cross-Attention）**：传统 GNN 对节点内多列特征做均值聚合会丢失信息。Griffin 引入交叉注意力，以任务嵌入和节点表示生成 Query，列名元数据为 Key，列值为 Value：

$$v_i^l = \text{Attention}^l(\text{QMLP}^l(u_i, t),\ m_i,\ x_i)$$

其中 $u_i$ 为节点中间表示，$t$ 为任务嵌入。这使模型能根据任务需要选择性关注相关列，而非平等对待所有列。

**关键改进**：第一层交叉注意力实际退化为均值聚合（因为此时缺乏足够任务信息），因此将第一层改为**自注意力**（Self-Attention），让列名和列值先进行列间交互，后续层再做任务条件聚合。

**层级聚合（Hierarchical Aggregation）**：与标准 MPNN 统一聚合所有邻居不同，Griffin 采用两级聚合：

$$h_i^{r,l} = \text{Mean}^l(\text{AMLP}^l(u_j) \mid (i,j) \in \mathcal{E}^r)$$
$$h_i^l = \text{Max}^l(h_i^{r,l} \odot e_r \mid r \in R)$$

先在**同关系类型内**做 Mean 聚合，再跨不同关系类型做 Max 聚合。这防止了某类邻居数量过多时淹没其他关系的信息，保持了关系结构。

#### 3. 统一任务解码器（Unified Task Decoder）

**分类任务**：直接用目标类别标签的文本嵌入作为分类头。设所有候选类别嵌入为 $z_1, \ldots, z_c$，模型输出 $z$，预测概率为：

$$P = \text{softmax}([\langle z, z_i \rangle \mid i=1,\ldots,c])$$

这让模型无需为每个任务维护独立分类头，可处理不同类别数的分类任务。

**回归任务**：将输出向量通过预训练数值解码器 DEC 还原为标量，再反归一化得到最终预测值。

### 损失函数 / 训练策略

Griffin 采用**三阶段训练流程**：

**第一阶段：掩码补全预训练（Completion Pretraining）**
在 200+ 单表数据集（约 1000 万行）上，随机遮掩一行中的某列值，用剩余列预测被遮掩值。损失为预测嵌入与真实嵌入的余弦距离：

$$\mathcal{L} = 1 - \cos(\text{Model}_\theta(T_{i,:\setminus j'}^k),\ \text{Encoder}(T_{i,j'}^k))$$

这一阶段**不需要标签数据**，属于自监督预训练。

**第二阶段：联合监督微调（Joint SFT）**
在单表 + RDB 数据集上用真实标签做有监督训练。分类任务用交叉熵，回归任务用 L2 损失。SFT 数据与下游评估数据严格隔离，防止数据泄漏。

**第三阶段：下游任务微调**
在具体下游任务上做 task-specific fine-tuning。

模型三个变体：
- **Griffin-unpretrained**：无预训练，仅靠架构优势
- **Griffin-pretrained**：仅在单表数据上预训练
- **Griffin-RDB-SFT**：单表 + RDB 联合 SFT

## 实验关键数据

### 主实验

在 4DBInfer 和 RelBench 两大基准上评估，覆盖 24 个任务（分类 + 回归），涉及电商、体育、社交、医疗等多个领域。

| 模型 | 平均排名↓ | ROC-AUC代表任务 | MAE代表任务 | 说明 |
|------|----------|----------------|------------|------|
| SAGE | 4.79 | 0.792 (Seznam) | 1.357 (Avito) | GNN 基线 |
| GAT | 5.21 | 0.805 | 1.370 | GNN 基线 |
| PNA | 5.33 | 0.800 | 0.894 | GNN 基线 |
| HGT | 5.33 | 0.797 | 0.669 | 异构图基线 |
| DFS+FTTransformer | 5.92 | 0.747 | 0.634 | 单表+DFS |
| DFS+XGB | 7.08 | 0.760 | 0.674 | 单表+DFS |
| **Griffin-unpretrained** | **3.71** | 0.800 | 0.659 | 仅架构优势 |
| **Griffin-pretrained** | **3.04** | **0.813** | **0.659** | +单表预训练 |

Griffin-unpretrained 仅凭架构改进已超越所有基线的平均排名；加入预训练后进一步提升。

### 消融实验

| 配置 | 平均排名↓ | ROC-AUC (Retailrocket) | MAE (Avito) | 说明 |
|------|----------|------------------------|-------------|------|
| Griffin（完整） | **1.4** | 0.716 | -0.659 | 交叉注意力+Max聚合 |
| Griffin-avg-attention | 2.7 | 0.692 | -0.760 | 交叉注意力→均值聚合 |
| Griffin-mean-GNN | 1.9 | 0.708 | -0.670 | Max聚合→Mean聚合 |

交叉注意力模块对性能影响最大，移除后平均排名从 1.4 降至 2.7。层级 Max 聚合也有明显贡献。

### 关键发现

1. **架构本身即有优势**：即使不预训练，Griffin 凭借交叉注意力 + 层级聚合就已超越所有 GNN 和 DFS 基线
2. **单表预训练通用有效**：仅在单表数据上预训练（不含任何 RDB），就能在 RDB 下游任务上普遍提升性能
3. **相似性和多样性驱动迁移**：
    - Commerce→Commerce 迁移效果最好（相似性假说）
    - Others→Commerce 也有正向迁移，有时甚至超过同域迁移（多样性假说）
    - Commerce→Others 迁移效果较差（缺乏相似性）
4. **低数据场景优势显著**：在限样本微调中，预训练的 Griffin 相对无预训练版本提升幅度更大
5. **与 TabPFNv2+DFS 互补**：Griffin 在 Commerce-2 和 Others-2 任务表现更好，TabPFNv2 在 Commerce-1 和 Others-1 更好

## 亮点与洞察

1. **统一编码器设计巧妙**：用预训练文本/数值编码器将异构特征映射到同一空间，使模型能在不同 schema 的 RDB 间泛化，这是 RDB 基础模型的核心需求
2. **分类头使用标签文本嵌入**：不设固定分类层，而是用标签名称的文本嵌入做内积分类，使分类器天然适配不同类别数的任务
3. **第一层自注意力 → 后续交叉注意力**：观察到第一层交叉注意力退化为均值聚合后，针对性改为自注意力，体现了对模型行为的深入分析
4. **先关系内 Mean 后跨关系 Max 的层级聚合**：解决了异构图中不同关系类型邻居数量不平衡的问题，Max 保留了各关系类型的显著信号
5. **预训练数据量达 1.5 亿行**：规模达到了基础模型级别，覆盖多领域多场景

## 局限与展望

1. **预训练数据质量依赖**：某些大规模单表数据因与下游 RDB 分布差距过大而被排除在预训练之外，如何自动选择有效预训练数据仍需探索
2. **迁移方向不对称**：Others-1→Others-2 有效但反向无效，说明迁移成功高度依赖预训练数据的多样性组成
3. **计算开销较大**：4 层 MPNN + 交叉注意力模块在大规模图上的训练成本（AWS g6.48x 实例）不可忽视
4. **未探索文本密集场景**：当 RDB 中文本列占主导时，当前 truncate 到 512 维的文本嵌入可能不够
5. **回归任务需额外反归一化**：分位数归一化 + 预训练解码器的流程较复杂，对极端值或重尾分布的适应性有待验证
6. **仅限预测任务**：未涉及 SQL 生成、表问答等其他 RDB 相关任务

## 相关工作与启发

- **单表基础模型**（TransTab、XTab、UniTabE、TabPFN）：这些方法限于单表，无法建模多表关系。Griffin 将其扩展到 RDB
- **RDB 图方法**（RDBTOGRAPH、GFS、4DBInfer、RelBench）：提供了 RDB→Graph 的建模范式，但缺乏预训练泛化能力
- **图基础模型**（OFA、UniGraph）：虽是图基础模型但面向通用图，未针对 RDB 的表格特性做优化
- 启发：**将结构化数据（表格/关系库）纳入基础模型范式**是一个有前景的方向；统一编码器 + 元数据嵌入的设计思路可推广到其他异构数据源

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首个 RDB 基础模型，问题定义有开创性 |
| 技术深度 | 4 | 编码器/MPNN/解码器各有精细设计 |
| 实验充分性 | 5 | 24 个任务 + 消融 + 迁移分析 + 原始数据 |
| 写作质量 | 4 | 结构清晰，图表丰富 |
| 实用价值 | 4 | 对企业级 RDB 预测场景有直接应用价值 |
| **综合** | **4.2** | 扎实的系统性工作，RDB 基础模型方向的重要基线 |

<!-- RELATED:START -->

## 相关论文

- [Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)
- [What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [InfMAE: A Foundation Model in the Infrared Modality](../../ECCV2024/self_supervised/infmae_a_foundation_model_in_the_infrared_modality.md)
- [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](../../NeurIPS2025/self_supervised/tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](../../AAAI2026/self_supervised/explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)

<!-- RELATED:END -->
