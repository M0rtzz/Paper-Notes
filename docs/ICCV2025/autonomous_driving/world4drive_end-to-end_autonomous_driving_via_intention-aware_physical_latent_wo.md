---
title: >-
  [论文解读] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model
description: >-
  [自动驾驶] 构建意图感知的潜在世界模型 World4Drive，利用视觉基础模型的空间-语义先验，在无感知标注条件下实现端到端规划，L2误差降低18.1%，碰撞率降低46.7%。
tags:
  - 自动驾驶
---

# World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2507.00603](https://arxiv.org/abs/2507.00603)
- **代码**: [GitHub](https://github.com/ucaszyp/World4Drive)
- **领域**: 自动驾驶
- **关键词**: 端到端自动驾驶, 世界模型, 多模态意图, 视觉基础模型, 自监督学习

## 一句话总结

构建意图感知的潜在世界模型 World4Drive，利用视觉基础模型的空间-语义先验，在无感知标注条件下实现端到端规划，L2误差降低18.1%，碰撞率降低46.7%。

## 研究背景与动机

端到端自动驾驶直接从原始传感器数据生成规划轨迹，但面临以下挑战：

**感知标注成本高**：UniAD、VAD、SparseDrive 等方法依赖 3D 边界框和高精地图等昂贵的感知标注，限制了可扩展性。

**单模态潜在特征不足**：LAW 等先行方法从原始图像提取单模态潜在特征进行自监督学习，但难以捕获物理世界的空间-语义信息和多模态驾驶意图，导致训练收敛慢、性能次优。World4Drive 相比 LAW 实现了 3.75 倍更快的训练收敛。

**多模态意图的不确定性**：在路口等复杂场景中，多种驾驶行为（左转、直行、右转）均为合理选择，需要建模这种意图不确定性并评估不同意图下的未来状态。

## 方法详解

### 整体框架

World4Drive 包含两个核心模块：(1) 驾驶世界编码模块，提取驾驶意图和物理世界潜在表示；(2) 意图感知世界模型，根据多模态意图预测未来潜在表示并评分多模态规划轨迹。

### 意图编码器

以轨迹词汇表 $\mathcal{V} \in \mathbb{R}^{N \times S \times 2}$（N=8192 条轨迹，S 个路点）为输入，通过 K-means 聚类获取意图点 $P_I \in \mathbb{R}^{3 \times K \times 2}$（3种指令类型 × K=6 个意图），经正弦位置编码后通过自注意力得到意图感知规划查询：

$$Q_{plan} = \text{SelfAttention}(Q_{ego} + Q_I)$$

### 物理世界潜在编码

**上下文编码器**包含两个关键组件：

- **语义理解**：使用 Grounded-SAM 生成伪语义标签 $S_t = \text{GroundedSAM}(F_t)$，通过交叉熵损失 $\mathcal{L}_{sem}$ 增强语义理解
- **3D空间编码**：使用度量深度模型 Metric3D v2 估计多视图深度图 $D_t$，通过前向投影获取 3D 位置图 $P_t$，经正弦编码和 MLP 得到位置嵌入：$E_t = \text{MLP}(\text{SPE}(P_t))$

**时间聚合**：通过交叉注意力整合历史信息：$L_t = \text{CrossAttention}(\hat{F_t}, \hat{F}_{t-1})$

### 意图感知世界模型

**动作编码**：规划查询聚合场景上下文后生成多模态轨迹 $T = \{T^1, \ldots, T^K\} \in \mathbb{R}^{K \times S \times 2}$，通过 MLP 获取意图感知动作 token $A \in \mathbb{R}^{K \times D}$。

**世界模型预测**：预测不同意图下的未来潜在表示，使用可学习查询和多层交叉注意力：

$$L_{t+n} = \text{CrossAttention}(Q_{future}, \text{Concat}(A, L))$$

### 世界模型选择器

计算每个意图下预测潜在表示与实际未来潜在表示的特征距离，选择距离最小的意图 $j$。对应距离作为重建损失 $\mathcal{L}_{recon}$，轨迹 $T^j$ 作为最终规划。同时训练 ScoreNet 通过 Focal Loss 预测评分 $\mathbb{S} = \text{Softmax}(\mathcal{C}(L_{t+n}))$。

### 总损失

$$\mathcal{L} = 0.2\mathcal{L}_{sem} + 0.2\mathcal{L}_{recon} + 0.5\mathcal{L}_{score} + 1.0\mathcal{L}_{traj}$$

## 实验

### nuScenes 开环规划

| 方法 | Avg L2(m)↓ | Avg 碰撞率(%)↓ | 需要标注 |
|------|-----------|---------------|---------|
| UniAD | 1.03 | 0.31 | ✓ |
| VAD | 0.72 | 0.23 | ✓ |
| GenAD | 0.52 | 0.19 | ✓ |
| LAW (Perception-based) | 0.49 | 0.19 | ✓ |
| BEV-Planner | 0.55 | 0.59 | ✗ |
| LAW (Perception-free) | 0.61 | 0.30 | ✗ |
| **World4Drive** | **0.50** | **0.16** | **✗** |

### NavSim 闭环规划

| 方法 | PDMS↑ | NC↑ | DAC↑ | EP↑ |
|------|-------|-----|------|-----|
| UniAD | 83.4 | 97.8 | 91.9 | 78.8 |
| LAW (Perception-free) | 83.8 | 97.2 | 93.3 | 78.8 |
| DiffusionDrive | 88.1 | 98.2 | 96.2 | 82.2 |
| **World4Drive** | **85.1** | 97.4 | **94.3** | **79.9** |

### 消融实验

| 深度 | 语义 | 世界模型 | 意图 | Avg L2 | 碰撞率 |
|------|------|---------|------|--------|--------|
| ✗ | ✗ | ✓ | ✗ | 0.61 | 0.30 |
| ✗ | ✗ | ✓ | ✓ | 0.55 | 0.25 |
| ✓ | ✗ | ✓ | ✓ | 0.51 | 0.29 |
| ✓ | ✓ | ✗ | ✗ | 0.49 | 0.26 |
| ✓ | ✓ | ✗ | ✓ | 0.61 | 0.36 |
| ✓ | ✓ | ✓ | ✓ | **0.50** | **0.16** |

### 关键发现

1. **无标注SOTA**：在无感知标注条件下达到最优性能，碰撞率甚至低于需标注方法
2. **意图+世界模型缺一不可**：仅有意图无世界模型反而退化（消融行5），世界模型提供了评估不同意图下规划合理性的能力
3. **语义先验降碰撞**：语义先验显著降低碰撞率，表明对障碍物的理解增强
4. **恶劣条件鲁棒**：夜间碰撞率降低63.7%，雨天降低68.8%，归因于视觉基础模型提供的高维语义信息对光度不一致性的鲁棒性

## 亮点与洞察

1. **模拟人类决策过程**：通过在不同驾驶意图下"想象"未来世界来选择最优行动，类似人类驾驶员的决策过程
2. **视觉基础模型的巧妙利用**：Grounded-SAM 提供语义先验、Metric3D 提供空间先验，无需手动标注
3. **意图-世界模型耦合设计**：意图提供多模态规划可能性，世界模型评估每种可能性的合理性
4. **收敛加速 3.75 倍**：得益于空间-语义先验的引入

## 局限性

1. 依赖预训练视觉基础模型（Grounded-SAM、Metric3D），增加了预处理开销
2. 闭环性能仍落后于使用 LiDAR 的 DiffusionDrive
3. 未来世界预测仅在潜在空间进行，缺乏可解释性

## 相关工作

- **端到端驾驶**：UniAD、VAD、SparseDrive、GenAD、DiffusionDrive
- **驾驶世界模型**：DriveDreamer、Drive-WM、LAW、VaVAM
- **意图建模**：VADv2、Hydra-MDP 概率规划

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 意图感知世界模型选择器设计新颖
- **技术深度**: ⭐⭐⭐⭐⭐ — 物理潜在编码、意图编码、世界模型选择器设计精密
- **实验充分性**: ⭐⭐⭐⭐⭐ — 开环+闭环，多维度消融，天气/光照/驾驶操作分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，消融设计合理

<!-- RELATED:START -->

## 相关论文

- [Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations](resonance_learning_to_predict_social-aware_pedestrian_trajectories_as_co-vibrati.md)
- [RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters](rescue_crowd_evacuation_simulation_via_controlling_sdm-united_characters.md)
- [SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](../../CVPR2025/autonomous_driving/diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)
- [Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)

<!-- RELATED:END -->
