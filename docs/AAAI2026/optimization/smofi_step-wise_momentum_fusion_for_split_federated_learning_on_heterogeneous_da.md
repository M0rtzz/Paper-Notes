---
title: >-
  [论文解读] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data
description: >-
  [AAAI 2026][优化/理论][联邦学习] 提出 SMoFi 框架，通过在 Split FL 的 server 端每步同步各 surrogate 模型的 momentum buffer，有效缓解 non-IID 数据导致的梯度分歧，在精度最高提升 7.1%、收敛速度最高加速 10.25 倍。
tags:
  - "AAAI 2026"
  - "优化/理论"
  - "联邦学习"
  - "数据异构"
  - "动量对齐"
  - "非IID"
  - "收敛加速"
---

# SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data

**会议**: AAAI 2026  
**arXiv**: [2511.09828](https://arxiv.org/abs/2511.09828)  
**代码**: 无  
**领域**: 联邦学习 / 分布式优化  
**关键词**: Split Federated Learning, 数据异构, 动量对齐, 非IID, 收敛加速  

## 一句话总结
提出 SMoFi 框架，通过在 Split FL 的 server 端每步同步各 surrogate 模型的 momentum buffer，有效缓解 non-IID 数据导致的梯度分歧，在精度最高提升 7.1%、收敛速度最高加速 10.25 倍。

## 研究背景与动机

### 领域现状

**领域现状**：Split Federated Learning (Split FL) 将模型切分为 client 端和 server 端两部分，利用 server 的强大算力分担训练负载，特别适合资源受限的边缘设备。在 SFLV1 框架中，server 维护多个 surrogate 模型并行训练，通信轮结束后聚合。已有方法如 FedAvg、FedProx、FedAvgM、SlowMo 等通过修改损失函数或改进聚合策略来缓解数据异构问题。

### 现有痛点

**现有痛点**：数据异构（non-IID）是 FL 面临的核心挑战。各 client 本地数据分布不一致导致 server 端各 surrogate 模型更新方向分歧，聚合后全局模型精度下降、收敛变慢。已有方法（FedAvgM、SlowMo）仅在通信轮间操作 momentum，粒度粗；FedNAG 的周期性聚合在部分场景反而降低性能。

### 核心矛盾

**核心矛盾**：Momentum（SGDM）能提升模型最终精度，但在 non-IID 数据下反而减慢收敛——因为 momentum 会让各本地模型更好地收敛到各自的局部最优，使更新方向更加分歧。如何将 momentum 这一"减速因素"转化为加速收敛的工具是一个关键挑战。

### 解决思路

**本文目标**：利用 Split FL 中 server 直接控制 surrogate 模型的天然优势，在每个 SGD 步骤（而非通信轮间）施加一致性约束。**切入角度**：在 server 端每步同步各 surrogate 的 momentum buffer，用全局一致的 momentum 引导所有模型朝同一方向更新。**核心idea**：Step-wise momentum fusion 将各模型的局部 momentum 替换为全局对齐的平均 momentum，零 client 端改动、零额外通信开销。

## 方法详解

### 整体框架
SMoFi 基于 SFLV1 的并行更新框架。Server 维护 $|\mathcal{J}^n|$ 个 surrogate server-side 模型并行训练。在每个 SGD step 后，SMoFi 在 server 端同步所有 optimizer 的 momentum buffer，将各自的局部 momentum 替换为全局平均 momentum。训练结束后按加权平均聚合模型参数。

### 关键设计

1. **Step-wise Momentum Alignment**:

    - 功能：在每一步 SGD 更新中对齐各 surrogate 的 momentum 方向
    - 核心思路：在每步 $\tau$，将各 surrogate optimizer 的 momentum 替换为全局对齐的 momentum $\bar{m}_s^{(n,\tau)}$：$m_{s,j}^{(n,\tau+1)} = \beta \bar{m}_s^{(n,\tau)} + \nabla \mathcal{L}_{\mathcal{B}_j^\tau}(\mathcal{W}_{s,j}^{(n,\tau)})$。每步利用统一的 momentum 约束各模型更新方向，使其趋向全局最优而非各自的局部最优
    - 设计动机：相比 FedAvgM/SlowMo 仅在通信轮间操作 momentum（粒度为多个 epoch），SMoFi 在每个 batch 步级别操作，约束更紧、对梯度分歧的响应更快

2. **Staleness Factor**:

    - 功能：处理各 client 训练进度不一致的问题
    - 核心思路：由于各 client 数据量不同导致本地步数 $T_j$ 各异，提前完成训练的 client 不再贡献 momentum。SMoFi 记录已完成 client 的最终 momentum 并以多项式衰减权重 $s_\alpha = (\tau - |T_j| + 1)^\alpha, \alpha < 0$ 加入对齐计算，保证参与对齐的 momentum 数量始终为 $|\mathcal{J}^n|$
    - 设计动机：如果忽略已完成 client 的 momentum，参与对齐的信号数量会逐步减少，约束强度下降。staleness factor 用衰减权重保留这些信号，同时降低过时信息的影响

3. **Client-Transparent Plug-in 设计**:

    - 功能：零 client 端改动即可使用
    - 核心思路：SMoFi 仅在 server 端操作，不修改 client 端代码、不增加通信开销或隐私风险。可作为 plug-in 叠加到 FedAvg、FedProx、FedNAR 等任何 FL 方法上
    - 设计动机：Split FL 的独特属性是 server 直接控制 surrogate 模型的训练过程，这意味着可以在 server 端做细粒度约束而无需客户端配合

### 理论保证
论文提供了基于强凸假设的 $\mathcal{O}(1/N)$ 收敛保证，证明了 momentum alignment 不会破坏收敛性，且每步的对齐操作可以有效减小各模型更新的方差。

## 实验关键数据

### 主实验

在 CIFAR-10、CIFAR-100、Tiny-ImageNet 上评测，non-IID 设置使用 Dirichlet 分布 ($\alpha=0.1$)。

| 方法 | CIFAR-10 Acc. | CIFAR-100 Acc. | Tiny-ImageNet Acc. |
|------|:---:|:---:|:---:|
| FedAvg | 77.16% | 48.10% | 33.43% |
| + FedAvgM | 79.19% | 50.28% | 33.58% |
| + SlowMo | 76.54% | 50.96% | 33.82% |
| + **SMoFi** | **81.82%** | **53.83%** | **39.73%** |

收敛加速倍数（达到目标精度所需轮数 vs FedAvg）：CIFAR-10 4.61×~5.54×，Tiny-ImageNet **最高 10.25×**。

### 消融实验

| 配置 | Tiny-ImageNet Acc. | 说明 |
|------|:---:|------|
| SFLV1 (baseline) | 33.43% | 无 momentum 对齐 |
| SFLV2 (串行) | 34.72% | 串行训练避免分歧但延迟高 |
| FedNAR + SMoFi | 39.73% | SMoFi 叠加后最优 |
| SMoFi (16 轮) | 达到目标精度 | SFLV1 需 >400 轮 |

### 关键发现
- SMoFi 对更大模型和更多 client 场景效果更好，符合实际部署需求的 scaling 特性
- 跨 optimizer（SGDM/NAG/Adam/AdamW）和跨架构（VGG/MobileNet/ResNet/DenseNet）均有效
- 在 Tiny-ImageNet 上仅 16 轮即达目标精度，而 SFLV1 需要超过 400 轮
- Plug-in 叠加方式：FedAvg+SMoFi、FedProx+SMoFi、FedNAR+SMoFi 均比原方法提升 3%~7%

## 亮点与洞察
- 设计极简但效果显著：仅同步 server 端 momentum buffer 这一个操作，就实现了最高 10.25× 的收敛加速
- 发现了 non-IID + momentum 的"减速悖论"并将其转化为加速工具，这一洞察本身对分布式优化有指导意义
- Plug-in 方式可与多种 FL 方法叠加使用，具有极强的通用性和实用性
- Split FL 中 server 直接控制 surrogate 模型的特性被充分利用，这种细粒度优化思路可推广到其他 split learning 变体

## 局限与展望
- 收敛分析基于强凸假设，实际深度网络为非凸，理论保证的适用范围有限
- Staleness factor $\alpha$ 需手动设定（默认 -0.1），自适应调节可能更优
- 仅在图像分类任务上充分验证，NLP/语音等场景评估有限
- 未考虑 client-side momentum 对齐的潜在收益

## 相关工作与启发
- **vs FedAvgM/SlowMo**: 仅在通信轮间操作 momentum，粒度粗；SMoFi 在每步操作，约束更紧，Tiny-ImageNet 上提升 5.91%
- **vs SFLV2**: 串行训练避免分歧但延迟高；SMoFi 保持并行训练的同时通过 momentum 对齐控制分歧
- **vs MergeSFL**: 自适应 batch size 加速时间效率，但通信轮数多于 SMoFi
- Staleness factor 的设计思路可借鉴到异步分布式训练中处理 stale gradient 的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 简单但有效的 step-wise momentum 对齐思路
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多 FL 方法、多 optimizer、多架构全面验证
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论与实验结合好
- 价值: ⭐⭐⭐⭐ 对 Split FL 社区有即时可用的价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[ICML 2025\] FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging](../../ICML2025/optimization/fedswa_improving_generalization_in_federated_learning_with_highly_heterogeneous_.md)
- [\[ICLR 2026\] Incentives in Federated Learning with Heterogeneous Agents](../../ICLR2026/optimization/incentives_in_federated_learning_with_heterogeneous_agents.md)
- [\[ICLR 2026\] FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](../../ICLR2026/optimization/feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)

</div>

<!-- RELATED:END -->
