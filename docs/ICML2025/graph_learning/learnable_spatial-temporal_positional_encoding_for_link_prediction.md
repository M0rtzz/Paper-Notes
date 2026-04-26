---
title: >-
  [论文解读] L-STEP: Learnable Spatial-Temporal Positional Encoding for Link Prediction
description: >-
  [ICML 2025][图学习][时序图] 提出 L-STEP，一种基于可学习时空位置编码的轻量级时序链接预测模型，通过离散傅里叶变换捕获位置编码的时序演化，用 MLP 替代 Transformer 注意力机制达到 SOTA 效果且运行更快。
tags:
  - ICML 2025
  - 图学习
  - 时序图
  - 位置编码
  - 链接预测
  - 离散傅里叶变换
  - 轻量级架构
---

# L-STEP: Learnable Spatial-Temporal Positional Encoding for Link Prediction

**会议**: ICML 2025  
**arXiv**: [2506.08309](https://arxiv.org/abs/2506.08309)  
**代码**: https://github.com/kthrn22/L-STEP  
**领域**: 图学习 / 时序链接预测  
**关键词**: 时序图, 位置编码, 链接预测, 离散傅里叶变换, 轻量级架构

## 一句话总结
提出 L-STEP，一种基于可学习时空位置编码的轻量级时序链接预测模型，通过离散傅里叶变换捕获位置编码的时序演化，用 MLP 替代 Transformer 注意力机制达到 SOTA 效果且运行更快。

## 研究背景与动机

**领域现状**：时序链接预测（判断两节点在特定时刻是否存在连接）是图学习核心任务。Graph Transformer 配合位置编码（PE）已成为 SOTA 方案，但依赖 $O(n^2)$ 或 $O(n^3)$ 的注意力机制。

**现有痛点**：(1) 大多数 PE 方法使用预定义的固定函数（如 Laplacian 特征向量），无法适应复杂属性图；(2) 唯一的可学习 PE（Dwivedi et al.）仅考虑静态结构信息，不处理时序演化；(3) PE 与 Transformer 注意力绑定，大规模图上计算代价极高。

**核心矛盾**：需要 PE 的表达能力来编码图的结构位置信息，但又要避免 Transformer 注意力的高复杂度——现有方案两者不可兼得。

**本文目标**：设计一种可学习的时空位置编码，使简单的 MLP 架构也能达到甚至超过 Graph Transformer 的时序链接预测性能。

**切入角度**：将节点的位置编码视为随时间演化的信号序列，利用离散傅里叶变换（DFT）在频域中捕获其时序依赖性，从而"预测"当前时刻的位置编码。

**核心 idea**：用 DFT + 可学习频域滤波器从历史位置编码序列推断当前位置编码，再用 MLP 聚合节点/边/位置信息进行链接预测，彻底摆脱对注意力机制的依赖。

## 方法详解

### 整体框架
L-STEP 在每个时刻 $t$ 需要预测节点 $u$ 和 $v$ 之间是否存在链接。流程为：(1) 用 LPE 模块从历史位置编码估计当前位置编码 $\tilde{p}_u^t$；(2) 用 Node-Link-Positional Encoder 聚合节点特征、边特征和位置编码得到节点表示；(3) 两节点表示送入 MLP 预测链接存在性。

### 关键设计

1. **可学习位置编码模块 (LPE)**:

    - 功能：从节点的历史位置编码序列推断当前时刻的位置编码
    - 核心思路：取节点 $u$ 最近 $L$ 个时刻的位置编码 $\{p_u^{t'_1}, \ldots, p_u^{t'_L}\}$，先做 DFT 转到频域，用可学习复值滤波器 $W_{filter} \in \mathbb{C}^{d_P \times L}$ 进行频域滤波（去噪+特征提取），再用 IDFT 转回时域，最后用加权求和池化得到估计的 $\tilde{p}_u^t$。预测完成后，用当前时刻的真实拓扑更新存储的位置编码
    - 设计动机：避免每个时刻重新计算 Laplacian 特征分解（$O(|V|^3)$），同时保留时序演化信息。从谱域角度证明了该方案能保持图的时空拓扑性质

2. **Node-Link-Positional Encoder**:

    - 功能：将节点特征、1-hop 邻居边特征和位置编码融合为紧凑的节点表示
    - 核心思路：分别聚合 (a) 时间窗内 1-hop 邻居的节点特征均值 $h_{u,N}^t$，(b) 最近 $K$ 个交互的边特征+时间编码 $h_{u,E}^t$，(c) 邻居的位置编码 $h_{u,P}^t$。三者经 MLP 变换后拼接，再过一层 MLP 得到最终表示 $h_u^t$
    - 设计动机：完全用 MLP 替代 Transformer 的注意力机制，验证了当位置编码足够表达时，MLP 可以达到相同效果

3. **时间编码器 (Time Encoder)**:

    - 功能：将时间戳映射为向量表示
    - 核心思路：使用余弦函数 $f_T(t) = \cos(t \cdot \omega)$，其中 $\omega_i = \alpha^{-(i-1)/\beta}$ 提供多尺度时间感知。参数 $\alpha, \beta$ 固定不训练
    - 设计动机：提供连续的时间感知能力，区分不同时间间隔

### 损失函数 / 训练策略
二元交叉熵损失，正样本为实际链接，负样本通过随机/历史/归纳三种采样策略获取。

## 实验关键数据

### 主实验

| 采样策略 | 数据集数 | L-STEP 排名 | 对比方法数 |
|---------|---------|------------|----------|
| Random Negative | 13 | 总体最优 | 10 |
| Historical Negative | 13 | 总体最优 | 10 |
| Inductive Negative | 13 | 总体最优 | 10 |
| TGB Benchmark | 大规模 | 领先 | 多个 SOTA |

在 transductive 和 inductive 两种设置下均表现最优。

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| LPE + MLP vs Transformer | 性能相当 | 验证 MLP 可替代注意力 |
| 不同初始化 PE | 鲁棒 | 即使用随机初始化也能学习有效 PE |
| 去掉 LPE | 性能显著下降 | 可学习 PE 是关键组件 |
| 运行时间对比 | 比 SOTA 快 | 避免了 $O(n^2)$ 注意力 |

### 关键发现
- MLP 配合足够表达的可学习位置编码，可以完全替代 Graph Transformer 的注意力机制，且推理速度更快
- DFT 频域滤波有效捕捉位置编码的时序模式，理论上证明了保持时空图谱性质
- 在 TGB 大规模基准上取得领先性能，验证了方法的实际可扩展性

## 亮点与洞察
- **位置编码即信号**的视角很新颖——将 PE 的时序演化类比为信号处理问题，用 DFT 在频域中学习清洁且预测性的表示
- **MLP vs Transformer** 的实验非常有说服力——证明了表达力瓶颈不在聚合架构，而在位置信息的质量
- 从理论（时空谱保持性）和实践（13 数据集全面领先）两方面验证了可学习 PE 的价值

## 局限与展望
- 初始化位置编码仍需要计算初始图的 Laplacian 特征分解
- 固定的超参数 $L$（历史窗口长度）和 $K$（邻居采样数）需要对每个数据集调优
- 当图结构变化剧烈时（大量新节点加入），历史 PE 的外推能力有待验证
- 目前仅处理无向/有向交互图，未考虑异构图设置

## 相关工作与启发
- **vs DyRep/TGAT/TGN**: 这些方法用 RNN 或注意力捕捉时序信息，L-STEP 用 DFT+PE 更轻量且效果更好
- **vs GraphGPS (Rampásek et al.)**: GraphGPS 用 Laplacian PE + Transformer 注意力，L-STEP 证明用可学习 PE + MLP 即可
- **vs Dwivedi et al. (2022)**: 唯一之前的可学习 PE 工作，但仅限静态图；L-STEP 扩展到时序设置

## 评分
- 新颖性: ⭐⭐⭐⭐ PE 作为时序信号+DFT 处理的视角新颖，MLP 替代 Transformer 的验证有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 13 数据集、10 算法、3 种采样策略、TGB 基准，极为全面
- 写作质量: ⭐⭐⭐⭐ 理论推导和实验组织清晰
- 价值: ⭐⭐⭐⭐ 为时序图学习提供了更高效的范式

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Positional Encoding meets Persistent Homology on Graphs](positional_encoding_meets_persistent_homology_on_graphs.md)
- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](../../NeurIPS2025/graph_learning/tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [\[ICML 2025\] Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction](open_your_eyes_vision_enhances_message_passing_neural_networks_in_link_predictio.md)
- [\[NeurIPS 2025\] OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](../../NeurIPS2025/graph_learning/ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)
- [\[ICLR 2026\] Revisiting Node Affinity Prediction in Temporal Graphs](../../ICLR2026/graph_learning/revisiting_node_affinity_prediction_in_temporal_graphs.md)

<!-- RELATED:END -->
