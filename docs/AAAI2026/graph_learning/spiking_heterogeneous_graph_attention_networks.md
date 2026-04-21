---
title: >-
  [论文解读] Spiking Heterogeneous Graph Attention Networks
description: >-
  [AAAI 2026][图学习][异构图] 提出 SpikingHAN，首次将**脉冲神经网络（SNN）**引入异构图学习，通过共享参数的单层图卷积聚合元路径邻居信息，语义级注意力融合多元路径，再编码为脉冲序列的 1-bit 二值表征，在三个数据集上以更少参数、更快推理和更低能耗达到有竞争力的分类性能。
tags:
  - AAAI 2026
  - 图学习
  - 异构图
  - 脉冲神经网络
  - 元路径
  - 图注意力
  - 能效计算
---

# Spiking Heterogeneous Graph Attention Networks

**会议**: AAAI 2026  
**arXiv**: [2601.02401](https://arxiv.org/abs/2601.02401)  
**代码**: https://github.com/QianPeng369/SpikingHAN  
**领域**: 图学习 / 脉冲神经网络  
**关键词**: 异构图, 脉冲神经网络, 元路径, 图注意力, 能效计算

## 一句话总结

提出 SpikingHAN，首次将**脉冲神经网络（SNN）**引入异构图学习，通过共享参数的单层图卷积聚合元路径邻居信息，语义级注意力融合多元路径，再编码为脉冲序列的 1-bit 二值表征，在三个数据集上以更少参数、更快推理和更低能耗达到有竞争力的分类性能。

## 研究背景与动机

1. **领域现状**：异构图神经网络（HGNNs）如 HAN 能有效处理多类型节点和边，捕获丰富的结构和语义信息。SNN 已在计算机视觉和同构图（SpikingGCN、SpikeGCL）中取得成功。
2. **现有痛点**：HGNNs 通常结构复杂——HAN 为每条元路径分配独立的注意力模块，导致参数量、内存和计算资源随元路径数增长显著增加，难以部署到资源受限设备。
3. **核心矛盾**：异构图学习需要有效的异构信息建模能力，但现有方法的计算开销过大。SNN 的二值稀疏通信天然适合低能耗场景，但尚未在异构图上被探索。
4. **本文目标**：设计一种低能耗、高效率的异构图神经网络，能在资源受限环境下运行。
5. **切入角度**：用共享参数的单层 GCN 替代 HAN 的多独立聚合模块以简化模型，再用 SNN 将异构信息编码为脉冲序列实现二值化。
6. **核心 idea**：共享参数的简化异构图卷积 + SNN 的脉冲编码 = 低能耗的异构图学习。

## 方法详解

### 整体框架

三个组件：(1) **共享卷积与注意力**——单层共享参数 GCN 对每条元路径聚合邻居信息，语义级注意力融合多元路径语义；(2) **全连接层与 SNN**——线性变换后作为 SNN 输入电流，执行积分-发射-重置生成脉冲序列；(3) **脉冲统计与预测**——对多时间步的脉冲输出做平均池化得到类别发射率作为分类预测。

### 关键设计

1. **共享参数单层图卷积**

    - 功能：用最小复杂度的共享卷积聚合每条元路径的邻居信息。
    - 核心思路：不同于 HAN 为每条元路径 $\Phi_p$ 分配独立聚合模块，SpikingHAN 使用同一个权重矩阵 $W_1$ 对所有元路径执行标准 GCN 卷积：$h_i^{\Phi_p} = \sigma(\sum_{j \in N_i^{\Phi_p}} h_j^0 \cdot W_1 / \sqrt{\tilde{D}_i \cdot \tilde{D}_j})$。生成 $P$ 组元路径特定的节点嵌入 $\{h^{\Phi_1}, ..., h^{\Phi_P}\}$。
    - 设计动机：独立模块导致参数量与元路径数线性增长且易过拟合。共享参数大幅减少复杂度同时保持表达能力——不同元路径的语义差异通过后续注意力机制区分。

2. **语义级注意力与 SNN 编码**

    - 功能：自适应学习不同元路径的重要性并融合，然后编码为脉冲表征。
    - 核心思路：语义注意力权重 $\beta_{\Phi_p} = \text{softmax}(\frac{1}{|V|}\sum_i q^T \cdot \tanh(h_i^{\Phi_p} \cdot W_2 + b))$，加权聚合 $H = \sum_p \beta_{\Phi_p} \cdot h^{\Phi_p}$。然后送入去除非线性激活和偏置的简化全连接层 $H \cdot W_3$ 作为 SNN 输入电流。SNN 使用 PLIF（参数化 LIF）模型，$\tau_m$ 可学习，通过积分-发射-重置生成脉冲。最终分类通过多时间步脉冲平均 $\hat{y} = \frac{1}{T}\sum_t \Theta(V^t)$。
    - 设计动机：SNN 的二值通信（0/1 脉冲）相比浮点值大幅降低计算和存储开销。简化全连接层（无非线性+无偏置）基于 LightGCN 的发现——图上深度不是关键因素。

### 损失函数 / 训练策略

交叉熵损失的半监督学习，surrogat gradient 反向传播。

## 实验关键数据

### 主实验

三个异构图数据集上的节点分类（Mi-F1, 20% 训练比例）：

| 方法 | 类型 | DBLP | ACM | IMDB |
|------|------|------|-----|------|
| GAT | 同构 GNN | 91.4 | 91.1 | 58.5 |
| SpikingGCN | 同构 SNN | 88.7 | 91.2 | 51.3 |
| HAN | 异构 GNN | 93.1 | 92.8 | 61.3 |
| PHGT | 异构 GNN | **93.7** | **93.4** | **63.2** |
| **SpikingHAN** | 异构 SNN | **93.7** | 93.3 | 62.9 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 独立 GCN vs 共享 GCN | 共享更优 | 参数更少且避免过拟合 |
| IF vs LIF vs PLIF | PLIF 最优 | 可学习时间常数更灵活 |
| 时间步 T=2/4/8/16 | T=4 最优 | 太少信息不足、太多计算浪费 |
| 硬重置 vs 减阈值重置 | 减阈值更好 | 保留更多膜电位信息 |

### 关键发现

- SpikingHAN 在 DBLP 上与 PHGT 持平（93.7%），在 ACM 上接近（93.3% vs 93.4%），证明 SNN 不牺牲性能。
- 参数量仅为 HAN 的约 1/3，推理速度更快，能耗显著降低。
- PLIF 的可学习 $\tau_m$ 对性能贡献显著——不同数据集需要不同的时间动态。
- IMDB 数据集上性能较弱（62.9% vs 63.2%），可能因为 IMDB 的标签噪声更严重，SNN 的 binary 表征损失了部分辨别信息。

## 亮点与洞察

- **首次将 SNN 引入异构图**：填补了 SNN 在异构图学习中的空白，证明了可行性和效率优势。
- **共享参数的极简设计**：单层共享 GCN + 语义注意力就足以达到多层独立模块的效果，体现了"less is more"的设计哲学。
- **1-bit 表征的实用价值**：脉冲编码将浮点嵌入压缩为 1-bit，对边缘设备部署有重大意义。

## 局限与展望

- 仅验证节点分类任务，链接预测、图分类等其他任务未涉及。
- 共享参数可能在元路径语义差异极大时不够表达力。
- 未在真实神经形态芯片上验证实际能耗收益。
- IMDB 上的性能差距说明 SNN 的二值化表示对噪声标签更敏感。

## 相关工作与启发

- **vs HAN**：HAN 每元路径独立聚合，SpikingHAN 共享参数+SNN，参数更少能耗更低。
- **vs SpikingGCN/SpikeGCL**：仅处理同构图，SpikingHAN 首次支持异构图的多类型节点和边。
- **vs PHGT**：Transformer 方法性能最优但计算量大；SpikingHAN 以极小性能代价换取巨大效率提升。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次 SNN + 异构图的组合
- 实验充分度: ⭐⭐⭐⭐ 三个数据集+多维度效率分析+消融完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，与 HAN 的对比图示直观
- 价值: ⭐⭐⭐⭐ 对边缘部署异构图学习有实际意义

<!-- RELATED:START -->

## 相关论文

- [GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)
- [NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification](../../ICLR2026/graph_learning/neurocircuitry-inspired_hierarchical_graph_causal_attention_networks_for_explain.md)
- [S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)

<!-- RELATED:END -->
