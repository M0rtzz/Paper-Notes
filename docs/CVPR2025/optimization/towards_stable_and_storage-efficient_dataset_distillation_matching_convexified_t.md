---
title: >-
  [论文解读] Towards Stable and Storage-efficient Dataset Distillation: Matching Convexified Trajectory
description: >-
  [CVPR 2025][优化][数据集蒸馏] 提出 MCT (Matching Convexified Trajectory) 方法，通过将 SGD 专家轨迹替换为从随机初始化到最优点的凸组合线性轨迹，同时解决了传统 MTT 方法的轨迹不稳定、收敛慢和存储消耗高三大问题。
tags:
  - CVPR 2025
  - 优化
  - 数据集蒸馏
  - 轨迹匹配
  - 凸化轨迹
  - 神经正切核
  - 存储高效
---

# Towards Stable and Storage-efficient Dataset Distillation: Matching Convexified Trajectory

**会议**: CVPR 2025  
**arXiv**: [2406.19827](https://arxiv.org/abs/2406.19827)  
**代码**: 无  
**领域**: 优化 / 数据集蒸馏 (Optimization / Dataset Distillation)  
**关键词**: 数据集蒸馏, 轨迹匹配, 凸化轨迹, 神经正切核, 存储高效

## 一句话总结

提出 MCT (Matching Convexified Trajectory) 方法，通过将 SGD 专家轨迹替换为从随机初始化到最优点的凸组合线性轨迹，同时解决了传统 MTT 方法的轨迹不稳定、收敛慢和存储消耗高三大问题。

## 研究背景与动机

数据集蒸馏（Dataset Distillation, DD）将大规模真实数据集压缩为小型合成数据集，使在合成数据上训练的模型达到接近真实数据集训练的性能。在DD方法中，多步轨迹匹配（MTT）是一个重要分支——让学生网络在合成数据上的训练轨迹接近专家网络在真实数据上的训练轨迹。

作者发现 MTT 方法存在三个严重局限：（1）**专家轨迹不稳定**：由 SGD 训练的专家轨迹存在剧烈震荡，验证精度波动不定，导致学生网络学到的训练动态也不稳定。（2）**蒸馏收敛速度低**：需要大量迭代才能生成有效的合成数据集。（3）**存储消耗高**：需要保存轨迹上所有中间点的模型参数（约50个模型），存储开销巨大。

作者提出了一个新视角理解 DD 和 MTT 的本质：通过重新表述 MTT 的损失函数，DD 的目标可以看作是获得一组参数（合成数据集），使其能对参数空间中任意点准确预测下一步更新的方向和幅度。基于此，解决方案是找到一条稳定、易拟合、易存储的专家轨迹。

## 方法详解

### 整体框架

MCT 的流程：（1）在真实数据上训练专家网络获得标准 SGD 轨迹 $\tau_{mtt}$；（2）用轨迹的起点 $\theta_\mathcal{T}^{(0)}$ 和终点 $\theta_\mathcal{T}^{(K)}$ 构造凸化轨迹 $\tau_{conv}$；（3）在凸化轨迹上使用连续采样策略进行蒸馏；（4）仅需存储 2 个模型 + 一组常数即可恢复完整轨迹。

### 关键设计

1. **凸化专家轨迹（Convexified Expert Trajectory）**:
    - 功能：构造稳定、单调递增的专家轨迹替代震荡的 SGD 轨迹
    - 核心思路：受 NTK 线性化动力学启发 $f_\theta(x) \approx f_{\theta_0}(x) + (\theta - \theta_0)^\mathsf{T} \nabla_\theta f_{\theta_0}(x)$，用起点和终点的凸组合（线性插值）构建轨迹 $\hat{\theta}_\mathcal{T}^{(t)} = (1 - \lambda_t) \theta_\mathcal{T}^{(0)} + \lambda_t \theta_\mathcal{T}^{(K)}$，其中 $\lambda_t$ 为插值系数。该轨迹上每个点的更新方向 $\vec{V}_\mathcal{T}^{(t)}$ 始终指向最优点，验证精度单调递增
    - 设计动机：SGD 轨迹的震荡使采样得到的更新方向不稳定；线性轨迹保证更新方向始终指向最终收敛点，极大简化了合成数据需要拟合的模式

2. **连续采样策略（Continuous Sampling Strategy）**:
    - 功能：从凸化轨迹上连续采样训练点，确保全面学习
    - 核心思路：由于凸化轨迹是连续的（不再是离散的 waypoint 集合），可以通过连续选择 $\lambda_t \in [0, 1]$ 来获取任意时间点的模型参数作为采样起点。这极大丰富了训练"数据集"（即参数空间中的采样点），使合成数据能更全面地学习更新规则
    - 设计动机：原始 MTT 只能从有限的预存 waypoint 中离散采样，遗漏了轨迹间的信息

3. **新视角下的损失函数重构**:
    - 功能：提供理解 DD 本质的新框架
    - 核心思路：将 MTT 损失函数 $\mathcal{L}(\mathcal{S},\mathcal{T}) = \frac{\|\theta_\mathcal{S}^{(t+N)} - \theta_\mathcal{T}^{(t+M)}\|_2^2}{\|\theta_\mathcal{T}^{(t)} - \theta_\mathcal{T}^{(t+M)}\|_2^2}$ 简单重写为 $\frac{\|\vec{V}_\mathcal{S} - \vec{V}_\mathcal{T}\|_2^2}{\|\vec{V}_\mathcal{T}\|_2^2}$，将专家轨迹 waypoint 看作"训练数据"$\{(\theta_\mathcal{T}^{(t)}, \vec{V}_\mathcal{T}^{(t)})\}$，DD 本质是学习一个在参数空间中预测更新向量的函数
    - 设计动机：这一视角直接揭示了三个问题的根源——V不稳定、训练集小且难拟合、训练集大需存储

### 损失函数 / 训练策略

- **损失函数**: 与 MTT 相同的归一化 L2 轨迹匹配损失
- **存储优化**: 仅需存储 $\theta_\mathcal{T}^{(0)}$ 和 $\theta_\mathcal{T}^{(K)}$ 两个模型 + 插值系数，相比 MTT 存储约 50 个模型大幅节省
- **蒸馏加速**: 凸化轨迹的稳定性使收敛所需迭代大幅减少

## 实验关键数据

### 主实验

CIFAR-10 数据集蒸馏性能（IPC=合成图片/类）：

| 方法 | IPC=1 | IPC=10 | IPC=50 | 存储 | 收敛迭代↓ |
|------|-------|--------|--------|------|---------|
| MTT | — | — | — | ~50 模型 | 高 |
| **MCT** | **更高** | **更高** | **更高** | **2 模型** | **显著降低** |

### 消融实验

| 配置 | 说明 |
|------|------|
| MTT 原始轨迹 | 震荡导致不稳定，收敛慢 |
| 凸化轨迹（无连续采样） | 稳定性提升，但采样点有限 |
| 凸化轨迹 + 连续采样 | 最佳效果，全面学习轨迹 |
| 不同 $\lambda_t$ 分布 | 均匀分布最有效 |

### 关键发现

- MCT 的凸化轨迹上的模型验证精度单调递增，完全消除了 SGD 轨迹的震荡
- 存储从 ~50 个模型减少到 2 个模型，存储效率提升约 25 倍
- 收敛速度显著加快——在相同精度阈值下，MCT 所需蒸馏迭代远少于 MTT
- 连续采样策略提供了更丰富的训练点，对小 IPC 设置尤为重要
- 在 CIFAR-10、CIFAR-100 和 Tiny ImageNet 三个数据集上均优于传统 MTT

## 亮点与洞察

- **重新理解 DD 本质的视角极具启发性**：将蒸馏看作"学习在参数空间中预测最优更新向量的函数"，自然地将三个看似独立的问题统一到同一框架下解释
- **用线性插值替代 SGD 轨迹的想法看似简单却极其有效**：受 NTK 理论启发，在参数空间中线性插值近似了 linearized dynamics，既保证了方向正确性又大幅简化了轨迹结构

## 局限与展望

- NTK 线性化假设在深度/宽度不足的网络上可能不成立
- 凸化轨迹假设起点和终点之间存在接近线性的路径，过于复杂的损失景观可能违反这一假设
- 未验证在更大规模模型（如 ResNet-101）和更大数据集（如 ImageNet-1K）上的效果
- 连续采样策略的最优分布仍需进一步研究

## 相关工作与启发

- **vs MTT**: MTT 是本方法直接改进的基础；MCT 保持相同的总体蒸馏框架但替换了专家轨迹，同时解决了稳定性、速度和存储三个问题
- **vs Distribution Matching (DM)**: DM 通过分布匹配而非轨迹匹配做蒸馏，避免了轨迹存储问题但通常性能不如轨迹匹配方法
- **vs FRePo/TESLA**: 这些方法优化 MTT 的计算效率和目标函数，但未解决轨迹震荡问题

## 评分

- 新颖性: ⭐⭐⭐⭐ NTK 启发的凸化轨迹简洁有效，新视角对理解 DD 有价值
- 实验充分度: ⭐⭐⭐⭐ 三个数据集，多种 IPC 设置，收敛+稳定性+存储多维度对比
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，从动机到方法的推导自然
- 价值: ⭐⭐⭐⭐ 25 倍存储节省+更快收敛+更好性能，解决了 MTT 的实际痛点

<!-- RELATED:START -->

## 相关论文

- [AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)
- [Asymptotically Stable Quaternionic Hopfield Structured Neural Network with Supervised Projection-based Manifold Learning](../../NeurIPS2025/optimization/asymptotically_stable_quaternion-valued_hopfield-structured_neural_network_with_.md)
- [Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)
- [Efficient Adaptive Federated Optimization](../../NeurIPS2025/optimization/efficient_adaptive_federated_optimization.md)
- [Oracle-Efficient Combinatorial Semi-Bandits](../../NeurIPS2025/optimization/oracle-efficient_combinatorial_semi-bandits.md)

<!-- RELATED:END -->
