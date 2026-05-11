---
title: >-
  [论文解读] Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction
description: >-
  [CVPR 2026][自动驾驶][轨迹预测] 从数据中心视角出发，提出 Den-TP 框架通过密度感知的数据集筛选和评估协议来解决轨迹预测数据集中场景密度的长尾不平衡问题，仅用 50% 数据就能保持整体性能并显著改善高密度场景的鲁棒性。
tags:
  - CVPR 2026
  - 自动驾驶
  - 轨迹预测
  - 数据中心
  - 密度平衡
  - 子模优化
  - 长尾分布
---

# Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction

**会议**: CVPR 2026  
**arXiv**: [2409.17385](https://arxiv.org/abs/2409.17385)  
**代码**: 无  
**领域**: 自动驾驶 / 轨迹预测  
**关键词**: 轨迹预测, 数据中心, 密度平衡, 子模优化, 长尾分布

## 一句话总结

从数据中心视角出发，提出 Den-TP 框架通过密度感知的数据集筛选和评估协议来解决轨迹预测数据集中场景密度的长尾不平衡问题，仅用 50% 数据就能保持整体性能并显著改善高密度场景的鲁棒性。

## 研究背景与动机

轨迹预测是自动驾驶中的关键任务，近年来基于 Transformer 和 GNN 的方法在标准基准上取得了很好的性能。然而，从数据角度审视会发现一个被忽视的严重问题：现有轨迹预测数据集存在显著的长尾密度不平衡。

在 Argoverse 1 和 2 等主流数据集中，低密度场景（少量交互者）占据绝大多数样本，而安全关键的高密度场景（60+ 个智能体的复杂多体交互）仅占不到 2.5%。这导致两个核心痛点：(1) 训练信号被低密度样本主导，模型在高密度场景中严重欠训练；(2) 标准评估协议使用全数据集平均误差，掩盖了高密度场景中的性能退化——模型在整体指标上看起来不错，但在最危险的密集交互场景中可能严重失效。

这个问题的根本矛盾在于：高密度场景最为安全关键（涉及复杂多体交互，预测误差可能直接危及驾驶安全），但在训练和评估中的影响力最小。现有的应对策略（重采样、重加权、数据增强）虽然改变了采样频率，但并未以原则性方式重塑数据分布。

核心 idea：将场景密度作为条件变量，通过密度感知的分区-选择策略构建紧凑但平衡的子集，利用基于梯度的子模优化在每个密度区间内选择代表性样本，同时跨区间实施偏向高密度的动态分配。

## 方法详解

### 整体框架

Den-TP 包含两个主要阶段：提取阶段和选择阶段。提取阶段通过轻量级预训练获取稳定的梯度估计，并按智能体数量将数据集划分为密度区间。选择阶段在每个区间内用子模优化选择代表性样本，跨区间用偏向采样显式上加权稀有的高密度场景。最终产出紧凑且密度平衡的训练子集。

### 关键设计

1. **密度分区 (Data Partitioning)**:

    - 功能：将数据集按场景复杂度划分为可解释的子集
    - 核心思路：计算每个样本的密度级别 $\rho(S_j)$（基于场景中的智能体数量），用固定间隔 $\tau$ 将数据集划分为 $K$ 个互不相交的子集 $\mathcal{D}_k$，其中 $S \in \mathcal{D}_k$ 当且仅当 $\rho(S) \in [\rho_{\min}+(k-1)\tau, \rho_{\min}+k\tau)$。虽然智能体数量不能完全捕捉场景复杂度，但它是一个数据集无关的代理指标，能进行一致的跨数据集分析
    - 设计动机：不做密度分区时，梯度更新被大量低密度样本主导，导致高密度场景系统性欠训练。显式分区是后续平衡采样的前提

2. **梯度特征提取与子模选择 (Gradient Extraction & Submodular Selection)**:

    - 功能：在每个密度区间内选择最具代表性且冗余最少的样本
    - 核心思路：对每个样本，通过反向传播提取预测输出的梯度特征 $\mathbf{G} = \nabla_{\hat{\mathbf{Y}}} \mathcal{L}$，与解码器嵌入通过元素乘融合为 $\mathbf{g} = \phi(\mathbf{G}) \odot \phi(\mathbf{E})$。定义子模评分函数 $P(S_j) = \sum_{S_i \in \mathcal{C}_k} \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\|\|\mathbf{g}_j\|} - \sum_{S_i \in \mathcal{D}_k \setminus \mathcal{C}_k} \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\|\|\mathbf{g}_j\|}$，贪心选择最小化 $P$ 的样本——既与已选集合冗余最少又与未选部分最具代表性
    - 设计动机：梯度特征融合了损失敏感度和解码器表示信息，比纯特征空间的聚类更能反映样本对训练的实际贡献

3. **动态分配策略 (Dynamic Allocation)**:

    - 功能：在密度区间间公平分配选择预算，优先保障高密度场景
    - 核心思路：给定总预算 $B = \lfloor \alpha |\mathcal{D}| \rfloor$，按从高密度到低密度的逆序处理各区间。每个区间 $\mathcal{D}_k$ 的预算为 $n_k = \min(|\mathcal{D}_k|, \lfloor B/k \rfloor)$。若区间样本数不超过预算则全部保留，无需梯度选择。剩余预算依次分配给低密度区间
    - 设计动机：高密度区间天然样本稀少，逆序处理确保它们不会在早期就被预算不足所淘汰。一个关键发现是高密度场景学到的能力可以向低密度场景迁移，反向迁移则很弱

### 损失函数 / 训练策略

梯度提取使用的训练损失为 $\mathcal{L} = \mathcal{L}_{\text{reg}} + \mathcal{L}_{\text{cls}}$，其中 $\mathcal{L}_{\text{reg}}$ 是最佳匹配预测模式上的负对数似然回归损失，$\mathcal{L}_{\text{cls}}$ 是优化预测模式概率的交叉熵损失。梯度提取需要轻量级预训练步骤获得稳定的梯度估计。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 保留比例 | minADE↓ | minFDE↓ | MR↓ |
|--------|------|------|----------|------|------|
| Argoverse 1 | Full (HiVT-64) | 100% | 0.695 | 1.037 | 0.109 |
| Argoverse 1 | Random | 50% | 0.750 | 1.175 | 0.137 |
| Argoverse 1 | Herding | 50% | 0.728 | 1.107 | 0.126 |
| Argoverse 1 | **Den-TP** | **50%** | **0.706** | **1.074** | **0.110** |
| Argoverse 1 | Full (HPNet) | 100% | 0.647 | 0.871 | 0.070 |
| Argoverse 1 | **Den-TP** (HPNet) | **50%** | **0.661** | **0.913** | **0.074** |

### 消融实验

| 策略 | 数据量 | minADE↓ | minFDE↓ | MR↓ | 说明 |
|------|---------|------|------|------|------|
| Augmenting | 220k | 0.718 | 1.106 | 0.115 | 复制高密度样本 |
| Weighting | 190k | 0.715 | 1.108 | 0.114 | 重加权高密度 |
| Epoch-wise | 95k | 0.752 | 1.189 | 0.130 | 每轮重采样 |
| High-density+Random | 95k | 0.724 | 1.111 | 0.117 | 保留高密度+随机填充 |
| **Den-TP** | **95k** | **0.706** | **1.074** | **0.110** | 密度平衡选择 |

### 关键发现

- **50% 数据即可达到甚至超越全量性能**：Den-TP 在 HiVT-64 上用 50% 数据 (minADE 0.706) 超过了 100% 全量数据 (0.695) 在 MR 指标上的表现，且 minADE 仅差 1.6%
- **高密度场景的能力可迁移**：密集交互场景学到的交互模式能够泛化到简单场景，但反向迁移很弱——这是密度优先分配策略的理论依据
- **跨模型泛化**：用 HiVT-64 做选择的子集在 HiVT-128 和 HPNet 上也保持优势，说明选择策略不依赖特定模型架构
- **朴素策略均不足**：重加权和数据增强虽改变了采样频率但增加了冗余而非覆盖度，epoch-wise 重采样引入不稳定性

## 亮点与洞察

- **数据中心视角切入轨迹预测**是最大亮点——此前该领域几乎完全是模型中心的，本文首次系统揭示了密度不平衡对训练和评估的严重影响。巧妙之处在于用简单的智能体计数作为密度代理就获得了足够的分析力
- **密度条件评估协议**的引入同样有价值——标准的聚合指标掩盖了长尾失效模式，按密度区间报告性能能暴露真正的安全风险
- **子模优化 + 梯度特征的组合**可以迁移到其他存在长尾分布问题的领域（如目标检测中的小目标、医学影像中的罕见病变）

## 局限与展望

- 智能体数量作为密度代理过于简化，不能区分同一密度下不同复杂度的交互模式（如 20 个智能体各自独立行驶 vs 20 个智能体密集交汇）
- 梯度提取需要预训练步骤，增加了额外的计算开销
- 仅在 Argoverse 1 和 2 上验证，未在 nuScenes、INTERACTION 等其他数据集上测试
- 动态分配策略中 $\lfloor B/k \rfloor$ 的设计缺乏理论证明其最优性

## 相关工作与启发

- **vs Random Selection**: 在 50% 预算下，Den-TP 的 minADE (0.706) 比随机选择 (0.750) 低 5.9%
- **vs Herding**: Herding 基于特征均值做贪心选择，但不考虑密度分布，Den-TP 在所有指标上均优
- **vs K-Means Clustering**: 基于轨迹特征聚类的方法忽略了梯度信息和密度平衡，性能介于随机和 Den-TP 之间

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次从数据中心视角系统分析轨迹预测中的密度不平衡，密度条件评估协议有方法论贡献
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多模型、多策略对比、多保留比例，非常全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，可视化丰富（密度分布图、性能曲线），算法描述规范
- 价值: ⭐⭐⭐⭐ 揭示了领域中被忽视的数据层面问题，实用价值高——可以直接用于减少训练成本

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](metadat_generalizable_trajectory_prediction_via_meta_pre-training_and_data-adapt.md)
- [\[CVPR 2026\] A Prediction-as-Perception Framework for 3D Object Detection](a_prediction-as-perception_framework_for_3d_object_detection.md)
- [\[ECCV 2024\] UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](../../ECCV2024/autonomous_driving/unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)
- [\[CVPR 2026\] FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration](foss_modeling_long_range_dependencies_and_multimodal_uncertainty_in_trajectory_p.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
