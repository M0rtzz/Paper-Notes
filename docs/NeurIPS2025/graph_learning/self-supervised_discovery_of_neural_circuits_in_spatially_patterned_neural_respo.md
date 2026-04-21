---
title: >-
  [论文解读] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks
description: >-
  [NeurIPS 2025][图学习][图神经网络] 提出基于GNN的自监督框架，通过结构学习模块推断潜在突触连接、同时用脉冲预测模块预测未来发放活动，在环形吸引子网络仿真数据和真实小鼠头方向细胞记录上均显著优于统计推断基线。
tags:
  - NeurIPS 2025
  - 图学习
  - 图神经网络
  - 神经回路推断
  - 自监督学习
  - 连续吸引子网络
  - 突触连接推断
---

# Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks

**会议**: NeurIPS 2025  
**arXiv**: [2509.17174](https://arxiv.org/abs/2509.17174)  
**代码**: 暂无  
**领域**: 图学习  
**关键词**: 图神经网络, 神经回路推断, 自监督学习, 连续吸引子网络, 突触连接推断

## 一句话总结

提出基于GNN的自监督框架，通过结构学习模块推断潜在突触连接、同时用脉冲预测模块预测未来发放活动，在环形吸引子网络仿真数据和真实小鼠头方向细胞记录上均显著优于统计推断基线。

## 研究背景与动机

推断神经群体活动之间的突触连接强度是计算神经科学的核心挑战。当前统计推断方法面临两个根本困难：

**部分可观测性**：不可能记录到回路中所有神经元的活动，未观测的神经元会引入虚假相关

**模型不匹配**：推断模型（如GLM、Ising模型）可能无法准确表征底层生成动力学系统

在强递归网络中，弱连接甚至不连接的神经元因"共激活"会呈现出强活动相关性——这是pattern formation原理的体现。现有方法（GLM、最大熵模型等）难以"解释掉"这些虚假相关，导致推断出偏。

**核心洞察**：神经元可以自然建模为图中的节点，突触连接为边。GNN的消息传递机制天然适合捕获交互神经元的动力学，通过预测未来脉冲这个代理任务，可以自监督地从潜在表示中提取连接结构。

## 方法详解

### 整体框架

模型由两个功能分离的模块组成：结构学习模块推断连接矩阵 $\mathbf{w}$，脉冲预测模块利用 $\mathbf{w}$ 预测未来发放。两者通过自监督目标联合优化，无需真实连接标签。

### 关键设计

1. **结构学习模块**：从每个神经元的脉冲序列中学习成对连接强度

   对每个神经元 $i$ 的脉冲序列 $\mathbf{x}_i$ 施加1D卷积提取时间特征，经全连接层得到嵌入 $\mathbf{z}_i$：
   $$\mathbf{z}_i = f_{\text{out}}(\text{vec}(f_{\text{Conv1D}}(\mathbf{x}_i)))$$
   
   将每对神经元的嵌入拼接后送入两层MLP估计连接强度：
   $$w_{ij} = \text{MLP}([\mathbf{z}_i, \mathbf{z}_j])$$
   
   设计动机：通过在核函数空间中学习特征表示来区分真实连接与虚假相关，而非直接从相关矩阵推断。

2. **脉冲预测模块（GNN消息传递）**：将推断的连接作为边权重，通过GNN预测未来脉冲

   编码器对近期脉冲历史窗口编码：$\mathbf{h}_i^t = f_{\text{enc}}(\mathbf{x}_i^{t-\ell+1:t})$
   
   消息计算与聚合：
   $$\mathbf{m}_{ij}^t = \phi([\mathbf{h}_i^t, \mathbf{h}_j^t])$$
   $$\mathbf{h}_i^{t+1} = \psi\left(\sum_{j \in \mathcal{N}(i)} w_{ij} \cdot \mathbf{m}_{ij}^t, \mathbf{h}_i^t\right)$$
   
   解码器输出发放率参数：$\log(\lambda_i^{t+1}) = f_{\text{dec}}(\mathbf{h}_i^{t+1})$，服从泊松过程生成脉冲。
   
   设计动机：GRU门控机制保留长时间依赖，连接权重 $w_{ij}$ 直接作为消息加权因子，使预测精度与连接推断质量紧密耦合。

3. **隐藏神经元处理**：添加辅助节点表示未观测神经元，通过插值初始化嵌入，在transductive设置下参与消息传递，隐式利用未观测成分的信息。

### 损失函数 / 训练策略

最小化泊松负对数似然：

$$\Theta^* = \arg\min_\Theta \sum_{i=1}^{N} \sum_{t=1}^{T} \left(\lambda_i^t - \mathbf{x}_i^t \log \lambda_i^t\right)$$

注意 $\mathbf{w}$ 并非自由参数，而是通过结构学习模块的确定性变换从观测数据中导出，随 $\Theta$ 优化间接更新。使用Adam优化器，学习率 $5 \times 10^{-4}$ 加指数衰减。

## 实验关键数据

### 主实验：全观测网络连接推断

| 方法 | Δ↓ (Thresh.) | $\mathcal{L}_{\text{bps}}$↑ (Thresh.) | Δ↓ (LNP) | $\mathcal{L}_{\text{bps}}$↑ (LNP) |
|------|-------------|-------------|----------|-------------|
| **GNN (本文)** | **0.061** | **0.882** | **0.049** | **0.876** |
| GLM | 0.244 | 0.695 | 0.238 | 0.712 |
| seqNMF | 0.789 | – | 0.796 | – |
| TCA | 0.762 | – | 0.761 | – |

GNN推断误差比GLM低70%以上。

### 外部输入驱动下的全观测网络

| 方法 | Δ↓ (Thresh.) | $\mathcal{L}_{\text{bps}}$↑ (Thresh.) | Δ↓ (LNP) | $\mathcal{L}_{\text{bps}}$↑ (LNP) |
|------|-------------|-------------|----------|-------------|
| **GNN** | **0.073** | **0.916** | **0.058** | **0.924** |
| GLM | 0.259 | 0.724 | 0.245 | 0.748 |

### 弱相关网络（单bump活动模式）

| 方法 | Δ↓ (Thresh.) | $\mathcal{L}_{\text{bps}}$↑ (Thresh.) | Δ↓ (LNP) | $\mathcal{L}_{\text{bps}}$↑ (LNP) |
|------|-------------|-------------|----------|-------------|
| **GNN** | **0.048** | **2.652** | **0.043** | **2.668** |
| GLM | 0.125 | 2.534 | 0.117 | 2.576 |

### 关键发现

1. 基线方法普遍在权重轮廓中出现"侧沟"伪影，即对不连接的远端神经元也推断出较强连接，而GNN有效抑制这些伪影
2. 增加隐藏神经元先改善后饱和——过多隐藏节点引入结构模糊性
3. 在真实小鼠HD细胞数据上，推断出的连接模式与连续吸引子模型理论预测一致

## 亮点与洞察

1. **自监督范式**：不需要真实连接标签，仅通过脉冲预测任务即可间接推断连接结构
2. **双模块解耦设计**：结构学习与动力学预测的分离使得连接矩阵成为可解释的潜在表示
3. **生物可验证**：在真实HD细胞数据上验证了环形吸引子假说

## 局限与展望

- 假设连接强度在记录期间不变，无法处理突触可塑性
- 当前仅验证了环形网络拓扑，对任意拓扑的泛化性待验证
- 同一观测到的动力学可能对应多种回路配置（非唯一性问题）
- 可扩展到2D连续吸引子网络（如网格细胞系统）

## 相关工作与启发

- **统计推断方法**: GLM, 最大熵Ising模型, 最小概率流(MPF)
- **GNN物理推断**: Neural Relational Inference (NRI) 系列
- **神经系统模型**: 环形吸引子、头方向细胞模型

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将GNN自监督框架系统应用于神经回路推断
- 实验充分度: ⭐⭐⭐⭐⭐ — 合成数据多条件消融+真实数据验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，实验递进式展开
- 价值: ⭐⭐⭐⭐ — 计算神经科学和GNN交叉领域有意义的贡献

<!-- RELATED:START -->

## 相关论文

- [Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [HGOT: Self-supervised Heterogeneous Graph Neural Network with Optimal Transport](../../ICML2025/graph_learning/hgot_self-supervised_heterogeneous_graph_neural_network_with_optimal_transport.md)
- [Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [Graph Neural Networks for Interferometer Simulations](graph_neural_networks_for_interferometer_simulations.md)
- [Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)

<!-- RELATED:END -->
