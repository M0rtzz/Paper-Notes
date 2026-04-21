---
title: >-
  [论文解读] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions
description: >-
  [自动驾驶] 模拟人类绘图过程，将车道拓扑建模为逐步图扩展的链式序列，通过自回归变换器增量构建有向车道图，克服 DAG 方法无法表达环路和双向车道的局限。
tags:
  - 自动驾驶
---

# SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2507.04822](https://arxiv.org/abs/2507.04822)
- **代码**: 未开源
- **领域**: 自动驾驶
- **关键词**: 车道拓扑, 图生成, 序列建模, 自回归, 邻接矩阵增量扩展

## 一句话总结

模拟人类绘图过程，将车道拓扑建模为逐步图扩展的链式序列，通过自回归变换器增量构建有向车道图，克服 DAG 方法无法表达环路和双向车道的局限。

## 研究背景与动机

精确的车道拓扑对自动驾驶至关重要，现有方法存在三类局限：

**检测式方法的碎片化**：像素级方法（HDMapNet）无法推断全局拓扑；片段级方法（TopoNet）分段检测中心线导致不连续和错位；路径级方法（LaneGAP）引入冗余和依赖大量后处理。

**生成式方法的结构破坏**：RNTR 将车道图转为 DAG，序列冗余且无法表达环路和双向车道。LaneGraph2Seq 分别生成节点位置和中心线形状，缺乏整体拓扑理解。

**核心洞察——人类如何画图**：人类从一个节点开始逐步添加新元素，持续扩展图并与已有节点建立连接。这启示将车道图构建建模为逐步扩展过程。

## 方法详解

### 车道图数学建模

将车道图表示为有向图 $G = (V, E)$，$V$ 为路口节点 $v_n = (x_n, y_n)$，$E$ 为中心线。使用：

- **邻接矩阵 $A$**：$A(i,j) = 1$ 表示 $v_i$ 到 $v_j$ 存在直接有向中心线
- **几何矩阵 $M$**：$M(i,j) = (\sigma_x^{ij}, \sigma_y^{ij})$ 为二次贝塞尔曲线控制点

### 图的序列化

定义 $S_n$ 为前 $n$ 个节点构成的子图序列。从已有节点来的中心线 $F_n = \sum_{k=0}^{n-1} M(k,n)$（"from" 关系），去往已有节点的中心线 $T_n = \sum_{k=0}^{n-1} M(n,k)$（"to" 关系）：

$$S_n = S_{n-1} + (v_n + F_n + T_n)$$

每引入新节点 $n$，邻接矩阵从 $n \times n$ 扩展到 $(n+1) \times (n+1)$，仅关注新节点与已有节点的拓扑关系。节点遍历顺序采用深度优先搜索。

**关键优势**：与 DAG 方法不同，SeqGrowGraph 可灵活建模环路、双向车道和非平凡拓扑。

### 离散化

坐标量化为离散 bin（分辨率 0.5m）：节点坐标范围 0-200，节点索引范围 200-350，贝塞尔控制点范围 350-570。特殊 token：`<to>` = 571，`<SEP>` = 572 等。

### 模型架构与目标

- **BEV 编码器**：基于 ResNet50 的 LSS 编码器，先在中心线分割任务上预训练
- **序列解码器**：6 层 Transformer，自回归生成 token 序列
- **加权损失**：节点位置 token 权重为 2，其余权重为 1

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log p(\hat{x}_t | x_1, x_2, \ldots, x_{t-1})$$

## 实验

### nuScenes 数据集主实验

| 方法 | L-P | L-R | L-F | R-P | R-R | R-F |
|------|-----|-----|-----|-----|-----|-----|
| TopoNet | 52.5 | 47.1 | 49.6 | 46.9 | 10.8 | 17.5 |
| LaneGAP | 49.9 | 57.0 | 53.2 | 74.1 | 34.9 | 47.5 |
| RNTR | 57.1 | 42.7 | 48.9 | 63.7 | 45.2 | 52.8 |
| LaneGraph2Seq | 46.9 | 43.7 | 45.2 | 63.7 | 36.3 | 46.2 |
| **SeqGrowGraph** | **63.6** | **50.8** | **56.4** | **75.5** | **61.4** | **67.8** |

### Argoverse 2 数据集

| 方法 | L-P | L-R | L-F | R-P | R-R | R-F |
|------|-----|-----|-----|-----|-----|-----|
| RNTR* | 50.7 | 29.4 | 37.2 | 68.1 | 29.6 | 41.3 |
| **SeqGrowGraph** | SOTA | — | — | — | — | — |

### 关键发现

1. **可达性指标大幅领先**：R-F 从 RNTR 的 52.8 提升到 67.8（+15.0），增量图扩展更好地保持拓扑连通性
2. **Landmark 精度高**：L-P 63.6 大幅超越所有基线，节点位置预测准确
3. **端到端无后处理**：一步生成连续的中心线结果
4. **可表达复杂拓扑**：DAG 方法缺失的环路和双向车道可被正确建模

## 亮点与洞察

1. **仿人绘图直觉**：逐步扩展图的方式与人类绘制地图的直觉高度一致
2. **邻接矩阵增量扩展**：每步仅需描述新节点与已有节点的关系，序列简洁
3. **DFS 序列化**：深度优先遍历确保相邻节点在序列中位置接近
4. **贝塞尔曲线几何表示**：二次贝塞尔仅需一个控制点即可描述中心线弯曲
5. **加权损失设计**：节点位置权重更高反映"正确的边依赖正确的点"

## 局限性

1. 节点数量增加时序列长度呈二次增长
2. 仅使用二次贝塞尔曲线，对高曲率道路表达能力有限
3. 离散化分辨率 0.5m 限制了精细位置精度

## 相关工作

- **在线高精地图**：HDMapNet、VectorMapNet、MapTR
- **车道拓扑**：STSU、TopoNet、RNTR、LaneGraph2Seq
- **语言模型用于图**：将图结构编码为可学习序列的研究

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 图增量扩展序列化方法是全新范式
- **技术深度**: ⭐⭐⭐⭐ — 数学建模严谨，序列化设计精巧
- **实验充分性**: ⭐⭐⭐⭐ — 两个大规模数据集，两种数据划分
- **写作质量**: ⭐⭐⭐⭐⭐ — 图示清晰，逐步扩展过程易懂

<!-- RELATED:START -->

## 相关论文

- [Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations](resonance_learning_to_predict_social-aware_pedestrian_trajectories_as_co-vibrati.md)
- [RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters](rescue_crowd_evacuation_simulation_via_controlling_sdm-united_characters.md)
- [World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving](../../CVPR2025/autonomous_driving/t2sg_traffic_topology_scene_graph_for_topology_reasoning_in_autonomous_driving.md)
- [Fine-Grained Representation for Lane Topology Reasoning](../../AAAI2026/autonomous_driving/fine-grained_representation_for_lane_topology_reasoning.md)

<!-- RELATED:END -->
