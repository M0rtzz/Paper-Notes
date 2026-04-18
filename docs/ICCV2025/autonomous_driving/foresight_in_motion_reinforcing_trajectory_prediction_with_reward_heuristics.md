---
title: >-
  [论文解读] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics
description: >-
  [ICCV 2025][自动驾驶][轨迹预测] 提出"先推理，后预测"（First Reasoning, Then Forecasting）策略，通过基于查询中心的逆强化学习（QIRL）推断驾驶意图的奖励分布，并结合 Bi-Mamba 增强的 DETR 式轨迹解码器，显著提升轨迹预测的置信度和准确性。
tags:
  - ICCV 2025
  - 自动驾驶
  - 轨迹预测
  - 逆强化学习
  - 意图推理
  - Mamba
---

# Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics

**会议**: ICCV 2025  
**arXiv**: [2507.12083](https://arxiv.org/abs/2507.12083)  
**代码**: 无  
**领域**: Autonomous Driving  
**关键词**: 轨迹预测, 逆强化学习, 意图推理, 自动驾驶, Mamba

## 一句话总结

提出"先推理，后预测"（First Reasoning, Then Forecasting）策略，通过基于查询中心的逆强化学习（QIRL）推断驾驶意图的奖励分布，并结合 Bi-Mamba 增强的 DETR 式轨迹解码器，显著提升轨迹预测的置信度和准确性。

## 研究背景与动机

轨迹预测是自动驾驶系统中连接感知和规划的关键模块。现有数据驱动方法主要通过直接回归轨迹或分类终点来预测未来运动，但存在以下问题：

**缺乏意图推理**：直接预测轨迹而不显式推理行为意图，导致可解释性和可靠性不足

**预测置信度低**：没有意图先验指导，多模态预测的概率分配不够准确

**忽视规划视角**：人类驾驶是层次化的（先做高层决策如换道，再执行具体操作），但预测模型很少借鉴这一思路

核心思想：将轨迹预测视为"为其他智能体做规划"，通过强化学习范式推理智能体的行为意图，为轨迹生成提供先验指导。

## 方法详解

### 整体框架

FiM（Foresight in Motion）采用编码器-解码器结构，包含：(1) 查询中心场景上下文编码器；(2) 基于奖励的意图推理器（QIRL）；(3) Bi-Mamba 增强的层次化 DETR 式轨迹解码器。

### 关键设计

1. **查询中心上下文编码（Query-Centric Context Encoding）**: 

    - 使用 1D CNN 编码智能体特征 $F_a \in \mathbb{R}^{N_a \times C}$，PointNet-like 网络编码地图特征 $F_m \in \mathbb{R}^{N_m \times C}$
    - 引入可学习网格查询 $Q_G \in \mathbb{R}^{H \times W \times C}$，通过交叉注意力机制将场景特征聚合到空间网格 token 中
    - 每个网格查询 $Q_G^{s_i}$ 对应 BEV 平面中特定分辨率 $d$ 的区域

2. **QIRL（Query-centric Inverse Reinforcement Learning）**: 

    - **核心思想**：将每个网格 $s_i$ 作为状态，对应的 query $Q_G^{s_i}$ 作为上下文特征
    - **奖励推断**：通过 1×1 CNN 层将网格 token 映射为奖励分布 $R \in \mathbb{R}^{H \times W \times 1}$
    - **专家示范**：将未来轨迹量化到分辨率 $d$ 的网格上作为示范状态
    - **MaxEnt IRL**：最大化示范数据的对数似然概率，同时遵循最大熵原则，迭代收敛得到奖励分布和最优策略
    - **策略展开（Rollout）**：基于学到的策略执行 $L$ 次展开（$L \gg K$），产生多条可行的 GRT（Grid-based Reasoning Traversal） $\Upsilon \in \mathbb{R}^{L \times \mathcal{H} \times 2}$
    - 根据 GRT 从网格 token 中提取推理 token $Q_G^\Upsilon \in \mathbb{R}^{L \times \mathcal{H} \times C}$

3. **辅助时空 OGM 预测头（Spatial-Temporal Occupancy Grid Map）**: 

    - 利用网格 token $Q_G$ 和奖励 $R$ 通过 U-Net 架构预测未来 $T_f$ 个时间步的占据网格图
    - 建模智能体间的未来交互，增强特征融合

4. **Bi-Mamba 增强的层次化轨迹解码器**: 

    - **轨迹提案生成**：anchor-free query $Q_P$ 通过交叉注意力编码 GRT 推理特征，生成 $L$ 条轨迹提案
    - **聚类**：K-means 聚类为 $K$ 条多模态提案 $\bar{Y}$
    - **轨迹精炼**：每条提案作为锚点，re-encode 为 $Q_T$，通过 DETR 式架构检索上下文特征
    - **Bi-Mamba 解码**：设计双模式查询 $Q_M \in \mathbb{R}^{K \times 2 \times C}$，包含两个 CLS token 分别附在轨迹 query 前后，通过双向扫描捕获前向和后向特征，最后通过模式自注意力模块增强多模态性
    - 最终轨迹：$Y = \bar{Y} + \Delta Y$

### 损失函数 / 训练策略

总损失为四项加权和：

$$\mathcal{L} = \mathcal{L}_{IRL} + \alpha \mathcal{L}_{OGM} + \beta \mathcal{L}_{REG} + \gamma \mathcal{L}_{CLS}$$

- $\mathcal{L}_{IRL}$：MaxEnt IRL 收敛损失
- $\mathcal{L}_{OGM}$：focal BCE 损失（占据预测）
- $\mathcal{L}_{REG}$：Huber 损失（轨迹回归），采用 winner-takes-all 策略
- $\mathcal{L}_{CLS}$：max-margin 损失（模式分类）

## 实验关键数据

### 主实验

**Argoverse 1 测试集（单模型）**:

| 方法 | MR₆↓ | minADE₆↓ | minFDE₆↓ | brier-minFDE₆↓ | Brier score↓ |
|------|------|----------|----------|----------------|-------------|
| DenseTNT | 0.1258 | 0.8817 | 1.2815 | 1.9759 | 0.6944 |
| HiVT | 0.1267 | 0.7735 | 1.1693 | 1.8422 | 0.6729 |
| DSP | 0.1303 | 0.8194 | 1.2186 | 1.8584 | 0.6398 |
| **FiM** | **0.1250** | 0.8296 | **1.2048** | **1.8266** | **0.6218** |

**nuScenes 预测排行榜**:

| 方法 | minADE₅↓ | MR₅↓ | minADE₁₀↓ | MR₁₀↓ |
|------|----------|------|----------|-------|
| DeMo | 1.22 | 0.43 | 0.89 | 0.34 |
| Goal-LBP | 1.02 | 0.32 | 0.93 | 0.27 |
| UniTraj | 0.96 | 0.43 | 0.84 | 0.41 |
| **FiM** | **0.88** | **0.31** | **0.78** | **0.23** |

### 消融实验

**奖励驱动推理策略的消融（Argoverse 验证集）**:

| 策略 | minFDE₆↓ | brier-minFDE₆↓ | Brier score↓ |
|------|----------|----------------|-------------|
| Vanilla（无推理） | 2.185 | 2.879 | 0.694 |
| Cross-Attention 替代 QIRL | 1.490 | 2.132 | 0.642 |
| **Ours (QIRL)** | **1.008** | **1.602** | **0.594** |

**Bi-Mamba 解码器组件消融**:

| MLP | Bi-Mamba | Self-Attn | brier-minFDE₆↓ | Brier score↓ |
|-----|---------|-----------|----------------|-------------|
| ✓ | | | 1.720 | 0.622 |
| ✓ | ✓ | | 1.649 | 0.605 |
| ✓ | | ✓ | 1.682 | 0.617 |
| ✓ | ✓ | ✓ | **1.602** | **0.594** |

**Uni-Mamba vs Bi-Mamba**:

| Mamba 类型 | minFDE₆↓ | brier-minFDE₆↓ | Brier score↓ |
|-----------|----------|----------------|-------------|
| Unidirectional | 1.034 | 1.636 | 0.603 |
| **Bidirectional** | **1.008** | **1.602** | **0.594** |

**长期意图监督的效果（Argoverse 2 验证集）**:

| 变体 | GRT 监督时间步 | minFDE₆↓ | brier-minFDE₆↓ | Brier↓ |
|------|-------------|----------|----------------|--------|
| GRT-S | 30 | 0.529 | 1.147 | 0.617 |
| GRT-M | 45 | 0.530 | 1.134 | 0.604 |
| GRT-L | 60 | **0.528** | **1.131** | **0.603** |

### 关键发现

- **QIRL 比 Cross-Attention 替代方案在 brier-minFDE 上低 0.53**（2.132 vs 1.602），验证了意图推理的关键作用
- **更长时间跨度的意图监督显著提升预测置信度**：GRT-S→GRT-L，Brier score 从 0.617 降至 0.603
- **Bi-Mamba 比 Uni-Mamba 更有效**：双向扫描机制更好地融合轨迹特征
- **FiM 在 nuScenes 上全面超越现有方法**：minADE₅ 达 0.88，大幅领先 UniTraj（0.96）
- OGM 辅助头和轨迹精炼模块各自独立贡献显著，组合效果最佳

## 亮点与洞察

- **"先推理后预测"范式**：首次将规划思想系统性引入轨迹预测，通过奖励驱动的意图推理提供先验
- **QIRL 的创新**：将传统需要结构化网格环境的 MaxEnt IRL 通过查询中心范式扩展到向量化表示，增强了灵活性
- **Bi-Mamba 在轨迹解码中的应用**：利用双向状态空间模型捕获轨迹的时空序列依赖
- **Brier score 的显著优势**：在预测置信度（而非仅距离误差）上的大幅提升，对下游规划模块更有价值

## 局限与展望

- MaxEnt IRL 的内循环迭代增加了计算开销
- 网格分辨率 $d$ 的选择影响推理精度，过粗会丢失信息
- GRT 离散化可能无法完美描述连续的驾驶意图
- 未与 LRM（如 OpenAI-o1）进行意图推理能力的直接对比
- 可进一步探索 reward function 的可迁移性

## 相关工作与启发

- QCNet、DeMo：当前 Argoverse 2 排行榜上的顶级方法
- MaxEnt IRL：经典的逆强化学习方法，从专家示范中学习奖励
- Mamba：选择性状态空间模型，在序列建模中表现出色
- 启发：轨迹预测不应只关注距离误差，预测置信度（Brier score）同样关键

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将 IRL 与查询中心编码结合用于意图推理，"先推理后预测"范式新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个大规模数据集、多项消融实验、定性可视化
- **写作质量**: ⭐⭐⭐⭐ 方法阐述详细，公式推导完整
- **价值**: ⭐⭐⭐⭐⭐ nuScenes 排行榜第一，方法可推广到其他运动预测场景
