# Learnable Spatial-Temporal Positional Encoding for Link Prediction

**会议**: ICML2025  
**arXiv**: [2506.08309](https://arxiv.org/abs/2506.08309)  
**代码**: [L-STEP](https://github.com/kthrn22/L-STEP)  
**领域**: graph_learning  
**关键词**: 位置编码, 时序链接预测, 可学习编码, 时空谱, MLP

## 一句话总结

提出 L-STEP，一种可学习的时空位置编码方法，从时空谱角度证明可保持图属性，仅用 MLP 即可达到 Transformer 性能，在 13 个数据集和 TGB 基准上取得领先表现，且计算复杂度更优。

## 研究背景与动机

- 位置编码对图 Transformer 至关重要
- 现有方法局限：(1) 预定义固定函数不适应复杂图 (2) 仅考虑静态结构 (3) 依赖 $O(n^2)$ 注意力
- 目标：可学习的时空位置编码 + 简单高效架构

## 方法详解

### Learnable Positional Encoding (LPE)

- 从时空谱角度证明可保持时序图拓扑属性
- 基于 DFT 的 Node-Link-Positional Encoder
- 时间编码器：$f_T(t) = \cos(t \cdot \omega)$，频率参数自适应

### L-STEP 整体架构

1. **LPE 模块**：可学习的位置编码捕获时空结构
2. **Node-Link-Positional Encoder**：基于离散傅里叶变换
3. **MLP 预测器**：替代 Transformer 的注意力机制
4. 端到端训练

### 理论分析

- **定理**：LPE 保持时序图拓扑的时空谱属性
- **复杂度**：$O(n \cdot d)$ vs Graph Transformer 的 $O(n^2 \cdot d)$ 或 $O(n^3)$

### 关键发现

- MLP 可充分利用 LPE 的表达力，达到 Transformer 性能
- 不同初始位置编码输入均保持鲁棒性
- 运行时间减少 2-5x vs SOTA

## 实验关键数据

### 全面对比（13 数据集，10 算法）

| 设置 | L-STEP 排名 |
|---|---|
| 转导 + random neg | 最优 |
| 转导 + historical neg | 最优 |
| 转导 + inductive neg | 最优 |
| 归纳 + random neg | 最优 |
| 归纳 + historical neg | 最优 |
| 归纳 + inductive neg | 最优 |

### TGB 大规模基准

- 在 Temporal Graph Benchmark 上取得领先性能
- 在工业级大规模图上验证可扩展性

### 消融实验

- MLP vs Transformer：MLP 性能相当但快 3-5x
- 不同初始 PE：Random walk PE / Laplacian PE / 随机初始化 → 均有效
- LPE 模块移除 → 性能显著下降

### 运行时间对比

| 方法 | 相对时间 |
|---|---|
| DyGFormer | 3x |
| CAWN | 5x |
| **L-STEP** | **1x** |

## 亮点与洞察

1. 可学习位置编码的时空谱理论保证
2. MLP 替代 Transformer 的计算效率
3. 全面的实验覆盖（数据集/算法/设置）

## 局限与展望

- 位置编码维度选择需调参
- 超大规模图的可扩展性待验证

## 相关工作与启发

- Dwivedi et al. (2022) LSPE
- Cong et al. (2023) GraphMixer

## 评分

⭐⭐⭐⭐ — 理论与实践结合好，效率优势明显

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Positional Encoding meets Persistent Homology on Graphs](positional_encoding_meets_persistent_homology_on_graphs.md)
- [TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](../../NeurIPS2025/graph_learning/tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction](open_your_eyes_vision_enhances_message_passing_neural_networks_in_link_predictio.md)
- [OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](../../NeurIPS2025/graph_learning/ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)
- [Revisiting Node Affinity Prediction in Temporal Graphs](../../ICLR2026/graph_learning/revisiting_node_affinity_prediction_in_temporal_graphs.md)

<!-- RELATED:END -->
