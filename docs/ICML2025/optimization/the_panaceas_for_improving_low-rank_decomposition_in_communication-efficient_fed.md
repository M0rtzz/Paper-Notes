---
title: >-
  [论文解读] The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning
description: >-
  [ICML2025][优化][联邦学习] 针对联邦学习中低秩分解的三个核心问题（分解什么、怎么分解、怎么聚合），分别提出 MUD（模型更新分解）、BKD（分块 Kronecker 分解）和 AAD（聚合感知分解）三种互补技术，在保持低通信开销的同时实现更快收敛和更高精度。
tags:
  - ICML2025
  - 优化
  - 联邦学习
  - low-rank decomposition
  - communication efficiency
  - Kronecker decomposition
  - model aggregation
---

# The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning

**会议**: ICML2025  
**arXiv**: [2505.23176](https://arxiv.org/abs/2505.23176)  
**作者**: Shiwei Li, Xiandi Luo, Haozhao Wang, Xing Tang, Shijie Xu, Weihong Luo, Yuhua Li, Xiuqiang He, Ruixuan Li
**代码**: [GitHub](https://github.com/Leopold1423/fedmud-icml25)  
**领域**: optimization  
**关键词**: federated learning, low-rank decomposition, communication efficiency, Kronecker decomposition, model aggregation

## 一句话总结

针对联邦学习中低秩分解的三个核心问题（分解什么、怎么分解、怎么聚合），分别提出 MUD（模型更新分解）、BKD（分块 Kronecker 分解）和 AAD（聚合感知分解）三种互补技术，在保持低通信开销的同时实现更快收敛和更高精度。

## 研究背景与动机

联邦学习（FL）需要多轮通信来训练全局模型，通信开销是核心瓶颈。低秩分解是一种有效的参数压缩技术，已被广泛用于 FL 中的双向通信压缩（如 FedHM、FedLMT、FedPara）。然而，现有方法面临三个关键问题：

**分解什么（What to decompose）**: 现有方法（如 FedLMT）直接对整个模型参数进行低秩分解，但模型参数本身可能是满秩的，强制降秩会导致严重的信息丢失。实际上 FL 中通信传递的是模型更新（model update），而非完整参数。

**怎么分解（How to decompose）**: 标准低秩分解 $W = UV^\top$ 的秩上界为 $\text{rank}(W) \leq r \ll \min\{m, n\}$，参数量为 $(m+n)r$，表达能力受限。FedPara 通过 Hadamard 乘积提升秩，但参数量加倍。

**怎么聚合（How to aggregate）**: 如果直接聚合子矩阵 $\bar{U}$ 和 $\bar{V}$，恢复出的矩阵 $\bar{U}\bar{V}^\top$ 与真实聚合矩阵 $\bar{W}$ 存在隐式偏差。FedHM 通过 SVD 重分解引入显式误差，FedLMT 直接聚合子矩阵则引入隐式误差。

这三个问题相互独立，本文分别提出对应的解决方案。

## 方法详解

### 1. Model Update Decomposition (MUD) — 分解什么

核心思想类似 LoRA：冻结原始模型参数 $W_0$，仅学习低秩的模型更新 $\Delta W$。

- 在每轮本地训练中，客户端冻结全局模型参数 $W_0$，学习低秩子矩阵 $U, V$ 作为模型更新
- 模型参数表示为 $W = W_0 + \Delta W$，其中 $\Delta W = UV^\top$
- 通信时只传递低秩子矩阵 $U, V$，服务器聚合后更新全局模型
- 与直接分解整个参数相比，模型更新通常具有更低的秩，因此低秩近似更加准确
- 每轮开始时重新初始化 $U, V$（$V$ 初始化为零），确保 $\Delta W$ 初始为零

**理论分析**: 作者提供了 MUD 的收敛性证明。在标准假设（L-光滑、有界方差、有界异质性）下，MUD 的收敛上界比 FedLMT 更紧，因为 MUD 的压缩误差仅与模型更新有关，而 FedLMT 的误差与整个模型参数有关。

### 2. Block-wise Kronecker Decomposition (BKD) — 怎么分解

用分块 Kronecker 乘积替代标准矩阵乘法来提升恢复矩阵的秩上界。

- 将权重矩阵 $W \in \mathbb{R}^{m \times n}$ 分为 $k$ 个块，每个块大小为 $\frac{m}{k} \times \frac{n}{k}$
- 每个块用 Kronecker 乘积分解：$W_j = A_j \otimes B_j$，其中 $A_j \in \mathbb{R}^{p \times q}$，$B_j \in \mathbb{R}^{\frac{m}{kp} \times \frac{n}{kq}}$
- 秩上界：$\text{rank}(W_j) = \text{rank}(A_j) \cdot \text{rank}(B_j) \leq pq \cdot \frac{m}{kp} \cdot \frac{n}{kq} = \frac{mn}{k^2}$
- 总参数量为 $k(pq + \frac{mn}{k^2 pq})$，通过选择 $pq = \frac{\sqrt{mn}}{k}$ 可得最优参数量 $2\sqrt{mn}$
- 当 $k = 1$ 时，单块 Kronecker 分解的秩上界为 $\min\{m, n\}$，即满秩，远优于标准低秩分解

### 3. Aggregation-Aware Decomposition (AAD) — 怎么聚合

解决子矩阵直接聚合导致的隐式误差问题。

**问题分析**: 设 $N$ 个客户端分别有 $U_i, V_i$，直接聚合 $\bar{U} = \frac{1}{N}\sum U_i$，$\bar{V} = \frac{1}{N}\sum V_i$，但 $\bar{U}\bar{V}^\top \neq \frac{1}{N}\sum U_i V_i^\top = \bar{W}$。

**解决方案**: 将乘积中的一方固定为所有客户端共享的参数 $\tilde{V}$（上一轮聚合结果），只训练 $U_i$：

- $\Delta W_i = U_i \tilde{V}^\top$（仅 $U_i$ 可训练，$\tilde{V}$ 固定共享）
- 聚合后 $\bar{U}\tilde{V}^\top = \frac{1}{N}\sum U_i \tilde{V}^\top = \overline{\Delta W}$，无隐式误差
- 为保持双向训练灵活性，采用交替训练：奇数轮训练 $U$（固定 $V$），偶数轮训练 $V$（固定 $U$）
- 该方法同样适用于 Kronecker 乘积的聚合

### 三种技术的结合

MUD + BKD + AAD 可以同时应用：

- MUD 决定"分解模型更新而非整个参数"
- BKD 决定"用分块 Kronecker 乘积分解"
- AAD 决定"交替训练以避免聚合误差"

## 实验关键数据

### 实验设置

- **数据集**: CIFAR-10, CIFAR-100, Tiny-ImageNet, CINIC-10
- **模型**: ResNet-18, VGG-11
- **Non-IID 设置**: Dirichlet 分布 $\alpha = 0.3$ 划分数据
- **客户端数**: $N = 10$，每轮全部参与
- **本地训练**: SGD, 学习率 0.01, 5 epochs/round
- **通信轮次**: 200 rounds
- **基线方法**: FedAvg（全量通信）、FedHM、FedLMT、FedPara、FedKSeed、FedRolex

### 表1: 主要结果 — 各方法在不同数据集上的测试准确率 (%)

| 方法 | 压缩率 | CIFAR-10 | CIFAR-100 | Tiny-ImageNet | CINIC-10 |
|------|--------|----------|-----------|---------------|----------|
| FedAvg | 1× | 87.72 | 58.64 | 40.80 | 79.42 |
| FedHM | ~10× | 79.45 | 44.84 | 27.74 | 72.07 |
| FedLMT | ~10× | 83.03 | 47.84 | 27.60 | 74.51 |
| FedPara | ~5× | 84.60 | 53.94 | 32.26 | 76.04 |
| **Ours (MUD+BKD+AAD)** | **~10×** | **87.33** | **57.00** | **38.42** | **78.34** |

在 ~10× 压缩率下，本文方法在 CIFAR-100 上比 FedLMT 提升 **+9.16%**，在 Tiny-ImageNet 上提升 **+10.82%**，接近无压缩的 FedAvg 性能。

### 表2: 消融实验 — 各技术的独立贡献 (CIFAR-100, ResNet-18)

| 配置 | MUD | BKD | AAD | 准确率 (%) |
|------|-----|-----|-----|-----------|
| Baseline (FedLMT) | ✗ | ✗ | ✗ | 47.84 |
| +MUD | ✓ | ✗ | ✗ | 52.31 |
| +BKD | ✗ | ✓ | ✗ | 51.67 |
| +AAD | ✗ | ✗ | ✓ | 49.12 |
| +MUD+BKD | ✓ | ✓ | ✗ | 54.89 |
| +MUD+AAD | ✓ | ✗ | ✓ | 53.54 |
| **+MUD+BKD+AAD** | ✓ | ✓ | ✓ | **57.00** |

三种技术互补，每种都有独立正贡献，组合使用效果最佳。MUD 贡献最大（+4.47%），BKD 次之（+3.83%），AAD 提供额外增益（+1.28%）。

### 表3: 不同压缩率下的性能 (CIFAR-100)

| 压缩率 | FedLMT | FedPara | Ours |
|--------|--------|---------|------|
| 5× | 52.40 | 53.94 | 57.85 |
| 10× | 47.84 | — | 57.00 |
| 20× | 42.31 | — | 53.12 |

在各压缩率下本文方法均显著优于基线，且在 20× 高压缩率下仍保持较好性能。

## 亮点

1. **问题分解清晰**: 将联邦学习中低秩分解的挑战拆解为三个正交问题（分解什么/怎么分解/怎么聚合），每个配以独立技术，逻辑清晰
2. **MUD 思想与 LoRA 的巧妙联系**: 将 LoRA 的"冻结+低秩更新"思想引入联邦学习通信压缩场景，使低秩近似更合理（更新比参数本身更容易低秩）
3. **AAD 揭示了被忽视的隐式聚合误差**: 指出直接聚合子矩阵的隐式问题是之前工作未注意到的，交替训练方案简洁有效
4. **BKD 用 Kronecker 乘积实现满秩上界**: 在相同参数量下获得远高于标准分解的秩上界
5. **最高 12% 的准确率提升**: 在多数据集上显著优于 FedHM、FedLMT 等基线，接近无压缩的 FedAvg

## 局限性

1. **实验规模有限**: 仅在 10 个客户端的小规模场景下验证，未测试大规模（100+ 客户端）或部分参与的设置
2. **模型限制**: 实验仅使用 ResNet-18 和 VGG-11 等传统 CNN，未验证在 Transformer 架构或大语言模型上的效果
3. **AAD 的交替训练代价**: 交替固定 U/V 可能导致训练不稳定或收敛速度变慢，每轮只能更新一半参数
4. **BKD 的分块约束**: 要求矩阵维度能被 $k$ 整除，对非标准尺寸的层需要额外的 padding 处理
5. **缺少异构设备场景**: 未讨论不同客户端使用不同压缩率的场景

## 相关工作

- **联邦学习通信压缩**: FedAvg (McMahan et al., 2017), 梯度量化 (Reisizadeh et al., 2020), 梯度稀疏化 (Li et al., 2024)
- **低秩分解方法**: FedHM (Yao et al., 2021) 使用截断 SVD, FedLMT (Liu et al., 2024) 直接训练预分解模型, FedPara (Hyeon-Woo et al., 2022) 用 Hadamard 乘积提升秩
- **LoRA 与参数高效微调**: LoRA (Hu et al., 2022) 冻结预训练参数、学习低秩更新，本文的 MUD 将此思路推广到联邦学习
- **Kronecker 分解**: Kronecker 乘积在模型压缩中的应用 (Thakker et al., 2019)

## 评分 ⭐

| 维度 | 评分 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 三个正交技术的组合有新意，AAD 揭示了被忽视的聚合偏差问题 |
| 理论性 | ⭐⭐⭐⭐ | 提供了 MUD 的收敛性分析，证明比 FedLMT 收敛更快 |
| 实验性 | ⭐⭐⭐ | 实验全面但规模偏小，缺少大模型和大规模客户端验证 |
| 实用性 | ⭐⭐⭐⭐ | 方法简洁可落地，三种技术可灵活组合，代码已开源 |
| 写作质量 | ⭐⭐⭐⭐ | 问题拆解清晰，图示直观，逻辑通顺 |
| 综合 | ⭐⭐⭐⭐ | 联邦学习通信效率领域的扎实工作，思路清晰，实验有说服力 |

<!-- RELATED:START -->

## 相关论文

- [Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](../../NeurIPS2025/optimization/layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [MAR-FL: A Communication Efficient Peer-to-Peer Federated Learning System](../../NeurIPS2025/optimization/mar-fl_a_communication_efficient_peer-to-peer_federated_learning_system.md)
- [FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging](fedswa_improving_generalization_in_federated_learning_with_highly_heterogeneous_.md)
- [Training-Free Bayesianization for Low-Rank Adapters of Large Language Models](../../NeurIPS2025/optimization/training-free_bayesianization_for_low-rank_adapters_of_large_language_models.md)
- [Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](../../NeurIPS2025/optimization/multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)

<!-- RELATED:END -->
