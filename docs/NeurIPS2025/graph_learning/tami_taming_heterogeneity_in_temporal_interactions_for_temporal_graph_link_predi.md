---
title: >-
  [论文解读] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction
description: >-
  [NeurIPS 2025][图学习][时序图] 首次系统识别时序图交互中的异质性问题（交互间隔呈幂律分布），提出TAMI框架包含对数时间编码(LTE)和链接历史聚合(LHA)两个模块，可无缝集成到现有TGNN中，在16个数据集上持续提升链接预测性能，最高提升87.05%。 领域现状：时序图链接预测是预测连续时间动态图上未来…
tags:
  - "NeurIPS 2025"
  - "图学习"
  - "时序图"
  - "链接预测"
  - "时间编码"
  - "异质性"
  - "交互频率"
---

# TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction

**会议**: NeurIPS 2025  
**arXiv**: [2510.23577](https://arxiv.org/abs/2510.23577)  
**代码**: [GitHub](https://github.com/Alleinx/TAMI_temporal_graph)  
**领域**: 图学习  
**关键词**: 时序图, 链接预测, 时间编码, 异质性, 交互频率

## 一句话总结

首次系统识别时序图交互中的异质性问题（交互间隔呈幂律分布），提出TAMI框架包含对数时间编码(LTE)和链接历史聚合(LHA)两个模块，可无缝集成到现有TGNN中，在16个数据集上持续提升链接预测性能，最高提升87.05%。

## 研究背景与动机

**领域现状**：时序图链接预测是预测连续时间动态图上未来节点交互的基础任务。主流方法包括基于时序邻居聚合的TGNN（DyGFormer, GraphMixer, TGAT, TGN）和基于随机游走的方法（CAWN）。

**现有痛点**：
   - 时序交互天然存在**异质性**：少数节点对贡献大部分交互事件，交互间隔呈幂律/高偏度分布（UCI数据集偏度2.385）
   - 现有时间编码函数使用正弦函数 $\bm{z}(t) = \cos(\Delta t \times \bm{\omega})$，当 $\Delta t$ 分布高度右偏时频率参数 $\bm{\omega}$ 难以学习
   - 节点嵌入仅依赖最近 $m$ 个交互，被高频交互邻居主导，低频交互信息被遗忘

**核心矛盾**：高频交互主导了模型学习，但低频交互的预测同样重要（如感恩节吃火鸡 vs. 日常吃汉堡）。现有方法在高频-低频交互间的预测质量差距巨大。

**本文目标**：(1) 缓解时间编码中的偏度问题；(2) 防止节点对的历史交互被遗忘。

**切入角度**：观察到交互间隔服从幂律分布，用对数变换压缩偏度；额外维护节点对级别的交互历史。

**核心 idea**：用对数变换"驯服"时间编码的偏度，用显式的链接历史聚合防止低频交互被遗忘。

## 方法详解

### 整体框架

TAMI = LTE (Log Time Encoding) + 任意现有TGNN + LHA (Link History Aggregation)

可无缝替换任何TGNN的时间编码和链接预测模块。

### 关键设计

#### LTE: 对数时间编码函数

- **功能**：在时间编码前对时间差进行对数变换，压缩右偏分布
- **核心公式**：
$$\bm{z}_l(t) = \cos(\Delta t_l \times \bm{\omega}), \quad \Delta t_l = \ln(1 + \Delta t)$$
  对比原始编码 $\bm{z}(t) = \cos(\Delta t \times \bm{\omega})$
- **理论基础**（Proposition 1）：若 $\Delta t \sim \text{Pareto}(\alpha, x_m)$，则 $\ln(1+\Delta t)$ 的偏度由 $\alpha$ 唯一确定。对数变换有效降低偏度（UCI数据集从2.385降至-1.14）
- **设计动机**：高偏度导致频率参数 $\bm{\omega}$ 对高频交互过拟合、对低频交互欠拟合。LTE使时间差分布更均衡，频率参数更容易学习

#### LHA: 链接历史聚合

- **功能**：显式维护每对节点最近 $k$ 次交互的嵌入历史
- **核心公式**：
$$\bm{r}_{uv}^\tau = \gamma \times \bm{c}_{uv} + (1-\gamma) \times \bm{r}_{uv}^{t_1}$$
  其中 $\bm{c}_{uv} = \text{MLP}([\bm{h}_u; \bm{h}_v])$ 编码当前状态，$\bm{r}_{uv}^{t_1}$ 为最近历史边嵌入，$\gamma$ 控制遗忘率
- **最终链接预测**：
$$p_{uv} = \text{MLP}([\bm{h}_u; \bm{h}_v; \bm{h}_{uv}])$$
  将历史嵌入与节点嵌入拼接输入预测器
- **设计动机**：当节点对不在彼此最近邻中时（"Exclusive"情况），其历史交互完全丢失。LHA通过额外存储和利用这些历史来补偿
- **空间复杂度**：GPU额外开销仅 $O(bd_r)$（$b$ 为batch大小），历史存CPU按需加载

### 训练策略

- LTE替换原有时间编码，LHA在链接预测器外层包装
- 使用most-recent aggregator简化 $\bm{h}_{uv} = \bm{r}_{uv}^{t_1}$
- 交互发生后更新 $M_{uv}(\cdot)$，维持最多 $k$ 条历史

## 实验关键数据

### 主实验：13个经典数据集 + 3个TGB数据集

**随机负采样下的AP提升**（选取代表性数据集）：

| 基线 → +TAMI | Enron | Lastfm | UCI | UN Vote |
|-------------|-------|--------|-----|---------|
| GraphMixer → | 82.26→90.97 (+10.59%) | 75.56→88.13 (+16.64%) | 93.38→96.20 (+3.02%) | 52.20→57.74 (+10.61%) |
| DyGFormer → | 92.46→92.66 (+0.22%) | 93.01→94.03 (+1.10%) | 95.66→96.72 (+1.11%) | 55.88→56.02 (+0.25%) |

**不同负采样策略最大提升**：
- 随机采样：最高16.64%（Lastfm）
- 历史采样：最高38.48%（UN Vote）
- 归纳采样：最高54.13%（UN Vote）

### TGB数据集表现

| 数据集 | DyGFormer MRR | +TAMI MRR | 提升 |
|--------|--------------|-----------|------|
| tgbl-wiki | 0.798 (rank 1) | 0.815 (rank 1) | +2.13% |
| tgbl-review | 0.224 (rank 6) | 0.419 (rank 2) | **+87.05%** |
| tgbl-coin | 0.752 (rank 2) | 0.794 (rank 1) | +5.59% |

### 消融实验

| 模块 | GraphMixer平均提升 | DyGFormer平均提升 |
|------|-------------------|-------------------|
| 仅LTE | ~0.5% | ~0.7% |
| 仅LHA | ~4.2% | ~0.2% |
| LTE+LHA (TAMI) | **~4.5%** | **~0.8%** |

LHA对GraphMixer贡献更大（因GraphMixer邻居聚合较弱），LTE对DyGFormer贡献更大（因DyGFormer已有强聚合但时间编码仍受偏度影响）。

### 按交互频率分组的性能

UCI数据集上GraphMixer按平均交互间隔分组：
- 间隔<100的节点对：原始AP高，TAMI提升小
- 间隔>10000的节点对：原始AP极低，**LHA提升高达25%+**，LTE进一步提升

### 关键发现

1. TAMI在所有13个经典数据集和3个TGB数据集上均提升性能，**无一下降**
2. 归纳采样下提升最大（最高54.13%），因更多未见节点对需要历史信息
3. LTE不仅提升精度还加速训练——总训练时间最多减少76.7%
4. "Exclusive"节点对（互不在对方最近邻中）的预测提升最大，验证了LHA的设计动机
5. tgbl-review上87.05%的MRR提升极为显著，表明该数据集存在严重的长尾交互问题

## 亮点与洞察

1. **问题定义精准**：首次系统化地识别并量化时序图交互的异质性及其对TGNN的影响
2. **解决方案极简但有效**：LTE仅是一个 $\ln(1+\cdot)$ 变换，LHA是简单的嵌入历史维护
3. **即插即用**：可直接集成到任何TGNN框架，无需修改底层架构
4. **训练效率提升**：LTE使频率参数更易学习，加速收敛，最多减少76.7%训练时间
5. **理论支撑**：Pareto分布下对数变换降偏度的理论分析简洁有力

## 局限与展望

1. LHA维护所有节点对的历史嵌入，总空间 $O(Nd_r)$ 对超大规模图可能是瓶颈
2. $\gamma$ 和 $k$ 为手动超参数，自适应学习可能更优
3. most-recent aggregator可能过于简单——attention-based aggregator或能捕获更复杂的历史模式
4. 仅分析了Pareto分布下的理论性质，其他重尾分布的适用性未讨论
5. 未与更新的TGNN方法（如2024年的频率感知方法）进行对比

## 相关工作与启发

- **DyGFormer** [Yu et al.]：通过注意力聚合大量单跳时序邻居，TAMI在其基础上进一步提升
- **GraphMixer** [Cong et al.]：MLP-Mixer + 邻居均值池化，TAMI的LHA对其改善最大
- **CAWN** [Wang et al.]：因果匿名游走，随机采样可能错过低频交互
- **TGB** [Huang et al.]：标准化时序图benchmark，TAMI将DyGFormer推至多数据集第一

## 评分

⭐⭐⭐⭐

方法简洁有效，实验极为扎实（16个数据集、3种负采样策略、2种TGNN集成）。对时序图交互异质性的分析填补了重要gap。对数变换和链接历史两个模块虽简单但直觉清晰、效果显著。主要不足是理论分析相对简单，且方法的技术创新性有限（本质是log变换+pair-level记忆）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] L-STEP: Learnable Spatial-Temporal Positional Encoding for Link Prediction](../../ICML2025/graph_learning/learnable_spatial-temporal_positional_encoding_for_link_prediction.md)
- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[NeurIPS 2025\] OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)
- [\[ICLR 2026\] Revisiting Node Affinity Prediction in Temporal Graphs](../../ICLR2026/graph_learning/revisiting_node_affinity_prediction_in_temporal_graphs.md)
- [\[NeurIPS 2025\] Spatio-Temporal Directed Graph Learning for Account Takeover Fraud Detection](spatio-temporal_directed_graph_learning_for_account_takeover_fraud_detection.md)

</div>

<!-- RELATED:END -->
