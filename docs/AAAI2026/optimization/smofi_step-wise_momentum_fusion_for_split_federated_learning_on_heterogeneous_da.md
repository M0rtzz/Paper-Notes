---
title: >-
  [论文解读] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data
description: >-
  [AAAI 2026][优化][联邦学习] 提出 SMoFi 框架，通过在 Split FL 的 server 端每步同步各 surrogate 模型的 momentum buffer，有效缓解 non-IID 数据导致的梯度分歧，在精度（最高+7.1%）和收敛速度（最高10.25×）上均显著优于现有方法。
tags:
  - AAAI 2026
  - 优化
  - 联邦学习
  - 数据异构
  - 动量对齐
  - 非IID
  - 收敛加速
---

# SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data

**会议**: AAAI 2026  
**arXiv**: [2511.09828](https://arxiv.org/abs/2511.09828)  
**代码**: 待确认  
**领域**: 联邦学习 / 分布式优化  
**关键词**: Split Federated Learning, 数据异构, 动量对齐, 非IID, 收敛加速  

## 一句话总结
提出 SMoFi 框架，通过在 Split FL 的 server 端每步同步各 surrogate 模型的 momentum buffer，有效缓解 non-IID 数据导致的梯度分歧，在精度（最高+7.1%）和收敛速度（最高10.25×）上均显著优于现有方法。

## 背景与动机
Split Federated Learning (Split FL) 将模型切分为 client 端和 server 端两部分，利用 server 的强大算力分担训练负载，特别适合资源受限的边缘设备。然而，数据异构（non-IID）是 FL 面临的核心挑战：各 client 本地数据分布不一致，导致 server 端各 surrogate 模型更新方向分歧，聚合后全局模型精度下降、收敛变慢。

已有方法通常修改损失函数（FedProx）或改进聚合策略（FedAvgM、SlowMo），但忽略了 Split FL 的一个独特属性：server 在每步都直接控制多个 surrogate 模型的训练过程。这意味着可以在更细粒度（step-wise）上施加约束来对齐模型更新，无需额外的通信或隐私开销。

作者还发现，虽然 momentum（SGDM）能提升模型最终精度，但在 non-IID 数据下反而减慢收敛——因为 momentum 会让各本地模型更好地收敛到各自的局部最优，使更新更加分歧。SMoFi 正是利用同步 momentum buffer 将这一"减速因素"转化为加速收敛的机制。

## 核心问题
如何利用 Split FL 中 server 直接控制 surrogate 模型的天然优势，在每个 SGD 步骤施加一致性约束以缓解 non-IID 数据引起的梯度分歧？

## 方法详解

### 整体框架
SMoFi 基于 SFLV1 的并行更新框架，server 维护 $|\mathcal{J}^n|$ 个 surrogate server-side 模型并行训练，在每个 SGD step 后同步所有 optimizer 的 momentum buffer，最终按加权平均聚合。

### 关键设计
**Momentum Alignment**: 在每步 $\tau$，将各 surrogate optimizer 的 momentum 替换为全局对齐的 momentum $\bar{m}_s^{(n,\tau)}$：

$$m_{s,j}^{(n,\tau+1)} = \beta \bar{m}_s^{(n,\tau)} + \nabla \mathcal{L}_{\mathcal{B}_j^\tau}(\mathcal{W}_{s,j}^{(n,\tau)})$$

每步利用统一的 momentum 约束各模型的更新方向，使其趋向全局最优而非各自的局部最优。

**Staleness Factor**: 由于各 client 数据量不同，本地步数 $T_j$ 各异，部分 client 提前完成训练后不再贡献 momentum。SMoFi 记录已完成 client 的最终 momentum 并以多项式衰减权重 $s_\alpha = (\tau - |T_j| + 1)^\alpha, \alpha < 0$ 加入对齐计算：

$$\bar{m}_s^{(n,\tau+1)} = \frac{\sum_{j \in \mathcal{J}^{(n,\tau)}} m_{s,j}^{(n,\tau+1)} + \sum_{j \in \mathcal{H}^n} s_\alpha(\tau) m_{s,j}^{(n,|T_j|+1)}}{|\mathcal{J}^{(n,\tau)}| + |\mathcal{H}^n|}$$

这保证了参与对齐的 momentum 数量始终为 $|\mathcal{J}^n|$，维持约束强度。

**Client-Transparent 设计**: SMoFi 仅在 server 端操作，无需修改 client 端，不增加通信开销或隐私风险。

## 实验关键数据

| 方法 | CIFAR-10 Acc.(%) | CIFAR-100 Acc.(%) | Tiny-ImageNet Acc.(%) |
|------|:---:|:---:|:---:|
| FedAvg | 77.16 | 48.10 | 33.43 |
| + FedAvgM | 79.19 | 50.28 | 33.58 |
| + SlowMo | 76.54 | 50.96 | 33.82 |
| + **SMoFi** | **81.82** | **53.83** | **39.73** |

收敛加速（达到目标精度所需轮数对比 FedAvg）：
- CIFAR-10: 4.61× ~ 5.54×
- Tiny-ImageNet: **最高 10.25×**（FedNAR + SMoFi）

与 Split FL 方法对比，SMoFi 在 Tiny-ImageNet 上 16 轮即达目标精度（SFLV1 >400 轮），同时精度 39.73% 远超 SFLV2 的 34.72%。

## 亮点
- 设计极简但效果显著：仅同步 server 端 momentum buffer，零 client 端改动
- Plug-in 方式可与 FedAvg/FedProx/FedNAR 等方法叠加使用
- 对更大模型和更多 client 场景效果更好，符合实际部署需求
- 提供了 $\mathcal{O}(1/N)$ 收敛保证的理论分析
- 跨 optimizer（SGDM/NAG/Adam/AdamW）和跨架构（VGG/MobileNet/ResNet/DenseNet）均有效

## 局限与展望
- 收敛分析基于强凸假设，实际深度网络为非凸
- staleness factor $\alpha$ 需手动设定（默认 -0.1），自适应调节可能更优
- 仅在图像分类任务上充分验证，NLP/语音等场景评估有限
- 未考虑 client-side momentum 对齐的潜在收益

## 与相关工作的对比
- **FedAvgM/SlowMo**: 仅在通信轮间操作 momentum，粒度粗；SMoFi 在每步操作，约束更紧
- **FedNAG**: 周期性聚合 NAG momentum，在部分场景反而降低性能；SMoFi 则稳定提升
- **SFLV2**: 串行训练避免分歧但延迟高；SMoFi 保持并行训练的同时通过 momentum 对齐控制分歧
- **MergeSFL**: 自适应 batch size 加速时间，但通信轮数多于 SMoFi

## 启发与关联
- Split FL 中 server 直接控制 surrogate 模型的特性为细粒度优化提供了独特机会，这一思路可推广到其他 split learning 变体
- Staleness factor 的设计思路可借鉴到异步分布式训练中处理 stale gradient 的场景
- Momentum 对齐的思路可能对 mixture of experts 中不同 expert 的训练同步也有启发

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Data Heterogeneity and Forgotten Labels in Split Federated Learning](data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging](../../ICML2025/optimization/fedswa_improving_generalization_in_federated_learning_with_highly_heterogeneous_.md)
- [FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](../../ICLR2026/optimization/feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)
- [Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](../../ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)
- [FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)

<!-- RELATED:END -->
