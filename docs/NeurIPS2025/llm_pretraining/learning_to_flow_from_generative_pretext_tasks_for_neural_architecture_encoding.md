---
title: >-
  [论文解读] Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding
description: >-
  [NeurIPS 2025][神经架构编码] 提出 FGP（Flow-based Generative Pre-training），通过让编码器重建"流代理"（flow surrogate）这一架构信息流的简化表征，使任意结构的编码器无需专用的异步消息传递设计即可捕获信息流，在性能预测中 Precision@1% 最高提升 106%。
tags:
  - NeurIPS 2025
  - 神经架构编码
  - 信息流
  - 生成式预训练
  - NAS
  - 图神经网络
---

# Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding

**会议**: NeurIPS 2025  
**arXiv**: [2510.18360](https://arxiv.org/abs/2510.18360)  
**代码**: [GitHub](https://github.com/kswoo97/FGPAnom)  
**领域**: 神经架构搜索 / 表示学习  
**关键词**: 神经架构编码, 信息流, 生成式预训练, NAS, 图神经网络

## 一句话总结

提出 FGP（Flow-based Generative Pre-training），通过让编码器重建"流代理"（flow surrogate）这一架构信息流的简化表征，使任意结构的编码器无需专用的异步消息传递设计即可捕获信息流，在性能预测中 Precision@1% 最高提升 106%。

## 研究背景与动机

神经架构编码器是 NAS 的核心组件，它将架构映射为向量表征，再用于预测架构在目标任务上的性能。当前最好的编码器（如 FlowerFormer）通过专门的异步消息传递结构来模拟前向传播和反向传播中的信息流，效果虽好，但处理速度比简单 GNN 编码器慢高达 **57 倍**，构成严重的效率瓶颈。

同时，已有的生成式预训练方法（如 Arch2vec 重建边、GMAE 预测被掩码的操作）从其他领域直接迁移而来，但在架构域中缺乏清晰的学习信号——因为架构中不存在类似化学键的组合规则，几乎所有操作都是掩码位的合理候选，模型难以从中获得有效指导。

因此，本文解决两个挑战：

**效率问题**：让简单编码器也能学到信息流特征，避免使用昂贵的流式编码器结构。

**预训练目标问题**：设计面向架构信息流的新型生成式预训练任务，提供比操作掩码更有效的学习信号。

## 方法详解

### 整体框架

FGP 分为两个阶段：

1. **构建预训练目标（Flow Surrogate）**：对每个架构图，通过一次性的随机向量传播生成一个代表信息流的向量，无需任何训练。
2. **生成式预训练**：训练任意结构的编码器重建这个 flow surrogate，使编码器内化信息流知识；预训练后在下游任务（性能预测、NAS）上微调。

### 关键设计

#### 1. **拓扑排序分配**

将架构的有向无环图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 按拓扑序将节点分为 $\mathcal{V}^{(1)}, \mathcal{V}^{(2)}, \ldots, \mathcal{V}^{(T)}$，其中 $\mathcal{V}^{(1)}$ 是没有入边的节点（对应输入），$\mathcal{V}^{(T)}$ 是没有出边的节点（对应输出）。拓扑序决定了消息传播的顺序。

#### 2. **流代理（Flow Surrogate）生成**

核心思路是用随机向量在架构图上做一次简化的前向传播 + 反向传播，得到的最终向量就是该架构的"信息流指纹"。

**模拟前向传播**：按 $\mathcal{V}^{(1)} \to \mathcal{V}^{(T)}$ 的顺序传播 fp-message。每个节点 $v_i$ 的操作嵌入为 $\mathbf{h}_i = \mathbf{P} \mathbf{X}_{i,:}$（$\mathbf{P}$ 是随机矩阵）。节点先聚合入边消息：

$$\mathbf{m}_i = \sum_{v_j \in \mathcal{N}^{(i)}} \mathbf{f}_j$$

再通过操作相关的转换得到 fp-message：

$$\mathbf{f}_i = \alpha \mathbf{m}_i + (1 - \alpha) \text{ReLU}([\mathbf{h}_i \| \mathbf{m}_i] \mathbf{W})$$

其中 $\mathbf{W} \in \mathbb{R}^{2k \times k}$ 是固定投影矩阵，$\alpha$ 是权重超参数。

**模拟反向传播**：按 $\mathcal{V}^{(T)} \to \mathcal{V}^{(1)}$ 的反序传播 bp-message，初始化为对应节点的 fp-message（模拟梯度对前向输出的依赖），使用相同的聚合与转换机制但沿出边反向传播。

最终，将所有 order-1 节点的 bp-message 求和得到 flow surrogate $\mathbf{s} = \sum_{v_i \in \mathcal{V}^{(1)}} \mathbf{b}_i$。

设计动机：这个过程模拟了真实神经网络中输入如何经过各层操作、梯度如何回传的完整链路。不同架构因拓扑结构和操作类型不同，产生不同的 flow surrogate，从而天然区分了架构的信息流特征。

#### 3. **生成式预训练损失**

编码器 $f_\theta$ 将架构图编码为 $\mathbf{z} \in \mathbb{R}^d$，MLP 解码器 $g_\phi$ 将其映射回流代理空间：

$$\mathcal{L}_{rec} = \| \mathbf{s} - g_\phi(f_\theta(\mathbf{X}, \mathcal{E})) \|_2^2$$

总训练目标结合辅助损失（如零成本代理预测）：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{rec} + \lambda_2 \mathcal{L}_{aux}$$

设计动机：重建流代理迫使编码器在嵌入中保留信息流特征，而辅助目标提供与性能相关的额外监督信号。

### 损失函数 / 训练策略

- **预训练阶段**：在全部架构上（不使用性能标签）最小化 $\mathcal{L}_{rec} + \mathcal{L}_{aux}$
- **微调阶段**：在有标签的少量训练集上最小化性能预测的监督损失
- 预训练可应用于任意 GNN 编码器（ResGatedGCN、GIN、FlowerFormer 等），无需修改编码器结构

## 实验关键数据

### 主实验 — 性能预测

使用 1% 训练集微调，在 NAS-Bench-101/201/301 上测试三种编码器 × 六种方法。

| 编码器 | 预训练方法 | NB-101 Kendall τ | NB-201 Kendall τ | NB-101 Prec@1% | NB-201 Prec@1% |
|---|---|---|---|---|---|
| ResGatedGCN | N/A (无预训练) | 65.0 | 73.4 | 18.2 | 29.7 |
| ResGatedGCN | ZC-Proxy | 68.3 | 79.9 | 26.2 | 44.3 |
| ResGatedGCN | **FGP** | **74.8** | **82.2** | **37.5** | **48.9** |
| GIN | N/A | 62.8 | 65.7 | 26.9 | 25.0 |
| GIN | **FGP** | **67.8** | **79.2** | **33.2** | **35.6** |
| FlowerFormer | N/A | 74.0 | 77.3 | 35.3 | 35.6 |
| FlowerFormer | **FGP** | **76.3** | **83.6** | **40.6** | **48.3** |

FGP 在 **27 个设置中的 23 个**胜出。ResGatedGCN + FGP 在 NB-101 上 Precision@1% 相比无预训练提升 **106%**。

### 消融实验

| 配置 | NB-101 Kendall τ | NB-201 Kendall τ | NB-101 Prec@1% | 说明 |
|---|---|---|---|---|
| w/o $\mathcal{L}_{rec}$ | 68.3 | 79.9 | 26.2 | 去掉流重建 → 退化为 ZC-Proxy |
| w/o $\mathcal{L}_{aux}$ | 71.5 | 74.5 | 35.6 | 去掉辅助损失 → Kendall τ 下降 |
| w/o Forward | 73.5 | 81.4 | 36.3 | 只做反向 → 仍有效但不完整 |
| w/o Backward | 72.4 | 81.4 | 31.0 | 只做前向 → 反向更重要 |
| **FGP（完整）** | **74.8** | **82.2** | **37.5** | 前向+反向+辅助损失最优 |

### 关键发现

1. **非流编码器 + FGP 可超越流编码器**：ResGatedGCN + FGP 在多数指标上接近或超过 FlowerFormer（无预训练），而速度快数十倍。
2. **NAS 实验**：FGP 预训练的预测器在 NPENAS 搜索中在每一步都找到更好的架构。
3. **PCA 可视化**显示 flow surrogate 能有效区分高性能架构和低性能架构。
4. 反向传播模拟比前向传播模拟对性能贡献更大。

## 亮点与洞察

- 巧妙地将"学习信息流"从模型结构设计问题转化为预训练目标设计问题，用一次性计算的 flow surrogate 替代了昂贵的异步消息传递。
- Flow surrogate 不需要任何标签即可计算，天然适合无监督预训练。
- 方法具有很好的通用性，可插入任意 GNN 编码器作为预训练步骤。

## 局限与展望

- Flow surrogate 使用固定随机矩阵 $\mathbf{P}$ 和 $\mathbf{W}$，可能不是最优选择，可以探索可学习的初始化或多组随机采样的集成。
- 目前只在 NAS-Bench 系列（主要是 CV 领域）上充分验证，NLP 和图学习等领域的结果放在附录中。
- 对于非常大的搜索空间（如 NB-301 的连续空间），FGP 的提升相对较小，可能需要更丰富的流表征。

## 相关工作与启发

- **FlowerFormer**：SOTA 流式编码器，本文的主要对标方法，结合 FGP 还能进一步提升。
- **Arch2vec / GMAE**：已有生成式预训练方法，重建边或预测操作，本文指出其在架构域中缺乏有效学习信号。
- **ZC-Proxy**：利用零成本代理作为预训练目标，FGP 的辅助损失中也使用了零成本代理。
- 启发：对于结构化数据的表示学习，设计领域特定的预训练目标比直接迁移通用方法更有效。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 用信息流模拟生成预训练目标的思路新颖，但核心技术（消息传递、图重建）相对标准
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个 NAS-Bench 数据集 × 三种编码器 × 五个基线，消融、可视化、NAS 应用俱全
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图示直观，挑战-方案的叙事流畅
- **价值**: ⭐⭐⭐⭐ 为 NAS 领域提供了实用的预训练方法，兼顾效率与效果

<!-- RELATED:START -->

## 相关论文

- [Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)
- [Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)
- [Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)

<!-- RELATED:END -->
