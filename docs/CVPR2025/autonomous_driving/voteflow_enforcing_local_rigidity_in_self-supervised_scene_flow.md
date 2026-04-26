---
title: >-
  [论文解读] VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow
description: >-
  [CVPR 2025][自动驾驶][场景流估计] VoteFlow 通过在网络架构中引入一个基于可微投票的轻量级模块，将局部刚性运动约束作为归纳偏置融入自监督场景流估计模型，在 Argoverse 2 和 Waymo 数据集上以极低计算开销超越了此前最优的自监督方法。
tags:
  - CVPR 2025
  - 自动驾驶
  - 场景流估计
  - 局部刚性
  - 投票机制
  - 自监督学习
  - LiDAR点云
---

# VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow

**会议**: CVPR 2025  
**arXiv**: [2503.22328](https://arxiv.org/abs/2503.22328)  
**代码**: https://github.com/tudelft-iv/VoteFlow  
**领域**: 自动驾驶 / 3D视觉  
**关键词**: 场景流估计, 局部刚性, 投票机制, 自监督学习, LiDAR点云

## 一句话总结

VoteFlow 通过在网络架构中引入一个基于可微投票的轻量级模块，将局部刚性运动约束作为归纳偏置融入自监督场景流估计模型，在 Argoverse 2 和 Waymo 数据集上以极低计算开销超越了此前最优的自监督方法。

## 研究背景与动机

**领域现状**：场景流估计旨在从两帧连续的 LiDAR 扫描中恢复每点运动向量。在自动驾驶中，这是自监督场景理解的基石，可用于运动物体关联、伪标签生成等下游任务。目前的前馈式方法（如 ZeroFlow、SeFlow）在推理速度和泛化性上有优势，但仍面临精度瓶颈。

**现有痛点**：现实世界中，同一刚体上的邻近点应共享相同运动。现有方法通过额外损失函数（如 SeFlow 的 cluster loss）或后处理（如 ICP-Flow 的聚类对齐）来施加刚性约束，但这些做法缺乏结构性的归纳偏置——模型本身并不具备"编码局部刚性"的能力。ICP-Flow 使用预聚类点进行 ICP 对齐，一旦聚类过分割或欠分割，就会产生显著误差。

**核心矛盾**：刚性约束要么通过不可微的后处理引入（无法端到端优化），要么通过额外正则项引入（可能与主损失冲突、训练效率低），始终没有在网络架构层面直接编码 "邻近点共享运动" 这一先验。

**本文目标**：设计一个轻量级、可微分、可即插即用的网络模块，使模型在前向传播过程中就能识别并利用局部共享运动。

**切入角度**：作者观察到，在自动驾驶短时间间隔（~0.1 秒）中，物体运动主要由平移主导。因此可以将问题简化为：在一个离散化的平移空间中，通过邻近 pillar 的投票来识别主导平移方向。

**核心 idea**：构建一个离散投票空间覆盖所有可能的平移，让邻近 pillar 根据特征相似度对各方向进行投票，用 CNN 汇总投票信息为连续特征，实现端到端学习。

## 方法详解

### 整体框架

VoteFlow 的输入是两帧连续 LiDAR 点云 $X^t$ 和 $X^{t+\Delta t}$（已补偿自车运动）。流程如下：
1. **Pillarization**：通过 Pillar Feature Net 将点云转为鸟瞰图伪图像，每个网格（pillar）大小为 0.2m × 0.2m，附带嵌入特征
2. **U-Net Backbone**：拼接两帧伪图像，通过 U-Net 提取融合特征 G
3. **Voting Module（核心）**：为每个非空 pillar 构建投票空间，进行可微投票，输出投票特征 H
4. **Decoder**：将伪图像特征、融合特征、投票特征和点相对于 pillar 中心的偏移量拼接，通过 4 层全连接层预测逐点场景流

### 关键设计

1. **离散投票空间 (Discretized Voting Space)**:

    - 功能：为每个 pillar 构建一个离散网格，覆盖所有可能的 2D 平移方向
    - 核心思路：设最大平移范围为 $\pm 2$ 米，pillar 大小为 0.2m，每个 pillar 的投票空间 $V_k^t$ 是一个 $20 \times 20$ 的离散网格。对于 pillar $k$，选择 $M=8$ 个时间 $t$ 的最近邻 pillar，每个邻近 pillar 在 $t+\Delta t$ 时刻搜索 $N=128$ 个候选 pillar，通过余弦特征相似度计算投票分数 $s_{k,m,n} \in [-1,+1]$，将分数累加到对应的平移方向 bin 中：$V_k^t(\vec{T}_{k,m,n}) \leftarrow V_k^t(\vec{T}_{k,m,n}) + s_{k,m,n}$
    - 设计动机：直接 argmax 只能得到单一粗糙方向且训练早期不稳定。因此用两层 CNN + ReLU 将投票空间压缩为连续特征向量，让解码器获得完整的投票分布信息

2. **Pillar 级操作与稀疏性利用**:

    - 功能：在 pillar 而非 point 级别执行投票，大幅降低计算量
    - 核心思路：自动驾驶场景中 90%+ 的 pillar 是空的（尤其去除地面点后）。投票模块只对非空 pillar 操作，利用 ball query 函数在 $t+\Delta t$ 时刻搜索邻近 pillar，避免了全量计算
    - 设计动机：逐点操作在大规模点云上计算量不可接受。pillar 化后不仅减少计算，还天然提供了空间聚合

3. **轻量级解码器设计**:

    - 功能：将多源特征融合为逐点场景流
    - 核心思路：与 SeFlow 使用 GRU 层不同，VoteFlow 使用 4 层全连接层 + ReLU 进行解码。通过 point-to-pillar 索引从伪图像 $I^t$、$I^{t+\Delta t}$、融合特征 $G$、投票特征 $H$ 中检索逐点特征，并附加每点相对于 pillar 中心的偏移量
    - 设计动机：简化设计降低训练和推理的计算成本，用 Voting Module 的结构化先验替代 GRU 的序列建模能力

### 损失函数 / 训练策略

沿用 SeFlow 的自监督损失体系，总损失 $\mathcal{L}_{total} = \mathcal{L}_{chamfer} + \mathcal{L}_{dynamic} + \mathcal{L}_{static} + \mathcal{L}_{cluster}$：

- **双向 Chamfer Loss** $\mathcal{L}_{chamfer}$：最小化变形后源点云 $\hat{X}^t = X^t + F^t$ 与目标点云 $X^{t+\Delta t}$ 之间的距离
- **Dynamic Loss** $\mathcal{L}_{dynamic}$：仅对动态点施加 Chamfer loss，解决类别不平衡（动态点是少数）。动态点由离线方法 DUFOMap 预定义
- **Static Loss** $\mathcal{L}_{static}$：鼓励静态点的流为零
- **Cluster Loss** $\mathcal{L}_{cluster}$：对 HDBSCAN 聚类的点施加流一致性约束

训练使用 Adam 优化器，初始学习率 $2 \times 10^{-4}$，12 个 epoch，第 6 个 epoch 后学习率降 10 倍。

## 实验关键数据

### 主实验

**Argoverse 2 测试集（Bucketed Normalized EPE）：**

| 方法 | 标签 | 动态 Norm. EPE ↓ (avg) | Car | Pedestrian | Wheeled VRU | 静态 EPE ↓ (avg) |
|------|------|------------------------|-----|------------|-------------|------------------|
| Flow4D | ✓ | 0.174 | 0.096 | 0.278 | 0.155 | 0.012 |
| SeFlow | ✗ | 0.309 | 0.214 | 0.463 | 0.267 | 0.014 |
| ICP-Flow | ✗ | 0.331 | 0.195 | 0.435 | 0.363 | 0.027 |
| **VoteFlow** | **✗** | **0.289** | **0.202** | **0.417** | **0.249** | **0.014** |

**Waymo Open 验证集（跨数据集零样本迁移）：**

| 方法 | 同域训练 | FD EPE ↓ | FS EPE ↓ | BS EPE ↓ |
|------|---------|----------|----------|----------|
| SeFlow | ✓ | 0.151 | 0.018 | 0.011 |
| SeFlow | ✗ | 0.155 | 0.018 | 0.013 |
| **VoteFlow** | **✗** | **0.142** | **0.014** | **0.012** |

### 消融实验

| 配置 | 动态 Norm. EPE ↓ | 说明 |
|------|------------------|------|
| SeFlow baseline | 0.309 | 基线 |
| + Voting Module (VoteFlow) | 0.289 | 加入投票模块，提升 2.0%pt |
| VoteFlow w/ GRU decoder | - | 比 FC decoder 计算量更大 |
| VoteFlow (FC decoder) | 0.289 | 更轻量，速度更快 |

### 关键发现

- VoteFlow 在所有自监督方法中取得最优的动态平均 Normalized EPE (0.289)，超越 SeFlow 2.0 个百分点
- 在行人类别上提升最大 (+4.6%pt)，说明投票机制对非车辆类的小物体运动建模更有效
- ICP-Flow 在 Car 类别上最优（0.195），但在 Wheeled VRU 上大幅落后（0.363），说明其聚类策略不稳定
- 跨数据集泛化能力优秀：VoteFlow 在 Argoverse 2 上训练，零样本迁移到 Waymo 后 FD EPE 0.142，超越在 Waymo 上训练过的 SeFlow (0.151)
- 推理速度约 25.6ms/sample（A100 GPU），满足实时需求

## 亮点与洞察

- **架构级归纳偏置替代损失级约束**：将刚性先验从损失函数"搬进"网络结构，是一种更优雅、更高效的方式。投票模块作为即插即用组件，可适配多种基线
- **投票→CNN→连续特征**的设计巧妙避免了 argmax 的不可微问题，同时保留了完整的投票分布信息，让解码器灵活利用
- **稀疏性利用**：自动驾驶 LiDAR 点云的高稀疏性（90%+ 空 pillar）天然适合投票机制，计算开销极低

## 局限与展望

- 投票空间仅建模 2D 平移，对 3D 旋转运动（如转弯车辆）的建模能力有限
- 依赖 DUFOMap 预定义动态/静态点和 HDBSCAN 聚类，引入了外部依赖
- 在 Car 类别上未超越 ICP-Flow，可能是因为车辆运动更符合 ICP 的刚体假设
- 未来可考虑扩展到 3D 投票空间或引入旋转投票维度

## 相关工作与启发

- **vs ICP-Flow**: ICP-Flow 使用手工特征 + 预聚类进行 ICP 对齐，VoteFlow 使用学习特征 + 架构级投票。ICP-Flow 在 Car 上更优，但泛化性和鲁棒性不如 VoteFlow
- **vs SeFlow**: SeFlow 通过 cluster loss 正则来施加刚性约束，VoteFlow 将其内化为架构先验。VoteFlow 用更轻的 FC decoder 替代了 SeFlow 的 GRU decoder
- **vs 测试时优化方法 (NSFP, FastNSF)**: 这些方法精度高但推理时间长（数分钟），VoteFlow 实现了实时推理 (~25ms)

## 评分

- 新颖性: ⭐⭐⭐⭐ 将投票机制作为架构归纳偏置引入场景流是新颖的，但整体框架仍是增量改进
- 实验充分度: ⭐⭐⭐⭐⭐ 两大主流数据集、跨数据集泛化、定性定量分析齐全
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，Figure 1 的可视化直观
- 价值: ⭐⭐⭐⭐ 对自监督场景流的实用价值高，代码已开源

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](../../ECCV2024/autonomous_driving/seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)
- [\[CVPR 2025\] PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds](psa-ssl_pose_and_size-aware_self-supervised_learning_on_lidar_point_clouds.md)
- [\[CVPR 2025\] Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)
- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](../../NeurIPS2025/autonomous_driving/self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)
- [\[CVPR 2025\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)

<!-- RELATED:END -->
