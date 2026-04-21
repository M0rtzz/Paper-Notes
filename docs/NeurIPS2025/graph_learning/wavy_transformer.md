---
title: >-
  [论文解读] Wavy Transformer
description: >-
  [NeurIPS 2025][图学习][Transformer] 揭示了Transformer注意力层本质上等价于完全图上的图神经扩散过程，并基于二阶波动方程提出Wavy Transformer，通过能量守恒特性缓解深层Transformer的过平滑问题，在NLP、CV和稀疏图任务上均取得一致性提升。
tags:
  - NeurIPS 2025
  - 图学习
  - Transformer
  - 过平滑
  - 波动方程
  - 图神经扩散
  - 注意力机制
  - 物理启发
---

# Wavy Transformer

**会议**: NeurIPS 2025  
**arXiv**: [2508.12787](https://arxiv.org/abs/2508.12787)  
**作者**: Satoshi Noguchi (JAMSTEC/RIKEN), Yoshinobu Kawahara (Osaka Univ/RIKEN)  
**代码**: [GitHub](https://github.com/noguchisatoshi/Wavy-Transformer)  
**领域**: graph_learning  
**关键词**: Transformer, 过平滑, 波动方程, 图神经扩散, 注意力机制, 物理启发  

## 一句话总结

揭示了Transformer注意力层本质上等价于完全图上的图神经扩散过程，并基于二阶波动方程提出Wavy Transformer，通过能量守恒特性缓解深层Transformer的过平滑问题，在NLP、CV和稀疏图任务上均取得一致性提升。

## 研究背景与动机

### 问题背景
深层Transformer模型普遍存在**过平滑（over-smoothing）**问题：随着网络层数增加，所有token表示趋于一致，导致更深的Transformer不一定优于浅层模型。该问题在GNN领域已被广泛研究，但在Transformer中尚未得到充分讨论。

### 已有工作的不足
- 现有缓解Transformer过平滑的方法（如FeatScale）主要属于"外部注入高频扰动"的策略，即向隐状态中添加高频信号防止收敛
- 缺乏对Transformer过平滑现象的**内在动力学机制**分析
- GNN领域已有基于波动方程的方法（Graph-CON、PDE-GCN），但尚未迁移到Transformer架构中

### 核心动机
从物理动力系统视角出发，将注意力层的隐状态动态解释为完全图上的扩散过程，利用波动方程的能量守恒和振荡特性，从根本上改变Transformer的内在动力学以缓解过平滑。

## 方法详解

### 关键洞察：注意力即图神经扩散

在完全图上定义图神经扩散方程：

$$\frac{\partial \mathbf{X}}{\partial t} = (\mathbf{A} - \mathbf{I})\mathbf{X}$$

其中$\mathbf{A}$为注意力矩阵（右随机矩阵），$\mathbf{I} - \mathbf{A}$可视为归一化图拉普拉斯算子。对时间离散化得到：

$$\mathbf{X}^{l+1} = \tau \mathbf{A}\mathbf{X}^l + (1-\tau)\mathbf{X}^l$$

当$\tau = 1/2$时，忽略层归一化的缩放效应，该式与标准注意力残差更新$\mathbf{X}^{l+1} = \mathbf{A}\mathbf{X}^l + \mathbf{X}^l$本质等价。这意味着**传统Transformer隐式执行扩散过程**，其耗散特性是过平滑的根源。

### 波动动力学注意力

基于完全图上的波动方程，引入速度变量$\mathbf{Y} = \frac{\partial \mathbf{X}}{\partial t}$，将二阶方程改写为一阶系统：

$$\mathbf{Y}^{l+1} = \tau(\mathbf{A} - \mathbf{I})\mathbf{X}^l + \mathbf{Y}^l, \quad \mathbf{X}^{l+1} = \tau \mathbf{Y}^{l+1} + \mathbf{X}^l$$

该离散系统是**辛（symplectic）**的，保持系统能量守恒。相比纯扩散更新，波动更新额外包含动量项$(\mathbf{X}^l - \mathbf{X}^{l-1})$，防止特征过度平滑。

### 混合残差连接

支持扩散与波动的可学习混合：$\mathbf{X}^{l+1} = \boldsymbol{\lambda} \mathbf{X}_{\text{wave}}^{l+1} + (1-\boldsymbol{\lambda}) \mathbf{X}_{\text{diffuse}}^{l+1}$，其中$\boldsymbol{\lambda} = \text{sigmoid}(\boldsymbol{\theta}) \in [0,1]^d$为可训练参数。

### 物理一致的层归一化与FFN

为保持状态-速度关系$\mathbf{Y} = \frac{\partial \mathbf{X}}{\partial t}$在链式法则下的一致性：

- **速度层归一化 $\text{LN}_v$**：仅保留缩放参数（$\sigma^2, \gamma$），去除平移参数（$\mu, \beta$），并使用状态$\mathbf{X}$的均值方差来归一化速度$\mathbf{Y}$
- **速度FFN**：$\text{FFN}_v(\mathbf{Y}^l) = \phi'(\mathbf{X}^l \mathbf{W}_1 + \mathbf{b}_1) \mathbf{Y}^l \mathbf{W}_1 \mathbf{W}_2$，使用激活函数的导数$\phi'$进行缩放

### 两种变体
- **Full Wave**：包含完整的速度分支（FFN + LN），物理约束更强但计算开销稍大
- **Light Wave**：仅保留动量项$\boldsymbol{\lambda}(\mathbf{X}^l - \mathbf{X}^{l-1})$，无额外FFN/LN，开销几乎可忽略

## 实验关键数据

### 实验1：NLP任务（BERT预训练 + GLUE微调）

| 残差类型 | PPL (↓) | MLM Acc (↑) | GLUE Avg (↑) | STS-B |
|---------|---------|-------------|-------------|-------|
| Diffusion | 31.76 | 44.39% | 64.13 | 52.11 |
| Full Wave | 31.99 | 44.52% | 62.27 | 32.91 |
| Mix (+Full) | 29.00 | **45.56%** | 62.44 | 29.40 |
| Mix (+Light) | 32.29 | 44.53% | **66.12** | **64.76** |

混合残差在PPL和GLUE平均分上均优于纯扩散基线。Mix (+Light) GLUE平均提升+1.99，STS-B提升+12.65。

### 实验2：CV任务（ImageNet分类）与稀疏图任务

**ImageNet分类（DeiT/CaiT）：**

| 方法 | 残差 | 层数 | 参数量 | Top-1 Acc (%) |
|------|------|------|--------|--------------|
| DeiT-Ti | Diffusion | 12 | 5.7M | 72.17 |
| DeiT-Ti | + Full Wave | 12 | 5.7M | 72.33 (↑0.16) |
| DeiT-Ti | + Light Wave | 12 | 5.7M | **73.09** (↑0.92) |
| DeiT-Ti + FeatScale | Diffusion | 12 | 5.7M | 72.35 |
| DeiT-Ti + FeatScale | + Full Wave | 12 | 5.7M | 72.62 (↑0.26) |
| CaiT-XXS-24 | Diffusion | 24 | 12.0M | 77.6 |
| CaiT-XXS-24 | + Full Wave | 24 | 11.1M | **78.6** (↑1.0) |

**稀疏图任务（DIFFormer）：**

| 数据集 | 指标 | 层数 | Diffusion | + Light Wave | Δ |
|--------|------|------|-----------|-------------|---|
| OGBN-Arxiv | Acc | 7 | 24.44±4.51 | **66.73±0.33** | **+42.29** |
| OGBN-Proteins | ROC-AUC | 5 | 69.42±2.31 | **80.14±0.67** | **+10.72** |

在稀疏图任务上改善尤为显著：OGBN-Arxiv 7层时准确率从24.44%提升至66.73%，说明波动残差有效缓解了深层的崩溃问题。

### 过平滑诊断

| 动力学 | 谱隙 (↓) | 节点特征方差 (↑) | 类间方差 |
|--------|---------|----------------|---------|
| Diffusion | 0.836±0.003 | 2.480±0.078 | 0.195 |
| + Full Wave | 0.629±0.009 | 2.609±0.090 | 0.211 |
| + Light Wave | 0.730±0.008 | 2.109±0.070 | **0.308** |

### 计算效率

| 模型 | 变体 | 推理 | 训练 | 峰值GPU内存 |
|------|------|------|------|-----------|
| BERT | Diffusion | 101.6 | 415.6 | 18.31 |
| BERT | Light Wave | 101.3 | 436.2 | 18.69 |
| DeiT-Tiny | Diffusion | 2631.1 | 618.6 | 8.25 |
| DeiT-Tiny | Light Wave | 2644.2 | 617.6 | 9.14 |

Light Wave变体的推理速度、训练吞吐和内存开销与基线几乎相同（差异在几个百分点以内）。

## 亮点

- **优雅的理论洞察**：首次严格建立注意力层与完全图上图神经扩散的等价关系，为Transformer过平滑提供了清晰的物理解释（扩散的耗散特性）
- **即插即用**：Wavy Transformer block可无缝集成到现有Transformer架构（BERT、DeiT、CaiT、DIFFormer），无需额外超参数调优，且几乎不增加参数
- **跨领域一致提升**：在NLP、CV和稀疏图三类任务上均取得改善，验证了方法的通用性
- **Light Wave极致轻量**：仅需一个可学习向量$\boldsymbol{\lambda} \in \mathbb{R}^d$，通过动量项即可获得显著收益
- **物理一致设计**：速度专用的LN和FFN基于链式法则推导，保持了状态-速度关系的物理自洽性

## 局限与展望

- **理论与实践的gap**：虽然波动方程理论上能量守恒，但混合残差中$\boldsymbol{\lambda}$的引入破坏了严格的辛结构，物理意义有所削弱
- **实验规模有限**：NLP实验使用比标准BERT更小的预训练配置（10k步，batch 64），未验证大规模预训练效果
- **Full Wave不稳定**：Full Wave在部分任务上（如STS-B）出现显著退化，可能因速度分支的梯度传播不稳定
- **仅验证分类任务**：未涉及生成任务（如语言生成、图像生成），波动动力学对decoder架构的影响未知
- **扩散-波动等价的假设较强**：忽略了$\mathbf{W}_V$的特征变换和层归一化的非线性效应，实际等价性是近似的
- **未与更多过平滑缓解方法对比**：如SkipInit、ReZero等简单残差缩放方法

## 与相关工作的对比

- **Graph-CON / PDE-GCN**：在稀疏图GNN上引入振荡/PDE动力学缓解过平滑；本文将思路拓展到完全图注意力（Transformer），是互补而非竞争关系
- **FeatScale (Wang et al. 2022)**：通过重加权特征增强高频信号（外部扰动策略）；本文属于内在动力学修改策略，且可与FeatScale叠加使用
- **Deng et al. (Denoising Hamiltonian Network)**：通过辅助损失引入哈密顿结构；本文直接替换残差动力学，无需额外损失
- **GRAND (Chamberlain et al. 2021)**：提出图神经扩散框架；本文建立注意力与该框架在完全图上的等价性，并进一步推广到波动方程
- **DIFFormer (Wu et al. 2023)**：基于扩散的图Transformer；本文在其基础上加入波动残差，在深层场景下显著改善性能崩溃
- **Dong et al. 2021**：理论证明纯注意力的秩随深度指数衰减；本文从扩散角度提供了互补的过平滑解释

## 评分

- 新颖性: ⭐⭐⭐⭐ — 注意力即扩散的等价性洞察新颖且优雅，但波动方程在GNN中已有先例
- 实验充分度: ⭐⭐⭐⭐ — 覆盖NLP/CV/Graph三类任务，但NLP规模较小，缺乏生成任务验证
- 写作质量: ⭐⭐⭐⭐⭐ — 物理直觉与数学推导结合紧密，从PDE到离散化到架构设计的逻辑链非常清晰
- 价值: ⭐⭐⭐⭐ — 提供了理解Transformer过平滑的新视角和轻量级通用解决方案，对深层Transformer设计有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [Relational Graph Transformer](../../ICLR2026/graph_learning/relational_graph_transformer.md)
- [Relieving the Over-Aggregating Effect in Graph Transformers](relieving_the_over-aggregating_effect_in_graph_transformers.md)
- [NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](../../AAAI2026/graph_learning/ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](../../AAAI2026/graph_learning/mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)
- [GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](../../AAAI2026/graph_learning/gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)

<!-- RELATED:END -->
