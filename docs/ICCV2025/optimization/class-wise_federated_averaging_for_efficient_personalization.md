---
title: >-
  [论文解读] Class-Wise Federated Averaging for Efficient Personalization
description: >-
  [ICCV 2025][优化/理论][个性化联邦学习] cwFedAvg 将 FedAvg 从"按客户端聚合"扩展为"按类别聚合"，为每个类别创建专属全局模型，再根据各客户端的类别分布加权组合成个性化模型，配合权重分布正则化（WDR）增强类别分布与权重范数的关联，在保持 FedAvg 通信开销的同时显著提升非 IID 场景下的个性化性能。
tags:
  - "ICCV 2025"
  - "优化/理论"
  - "个性化联邦学习"
  - "类别级聚合"
  - "权重分布正则化"
  - "数据异质性"
  - "隐私保护"
---

# Class-Wise Federated Averaging for Efficient Personalization

**会议**: ICCV 2025  
**arXiv**: [2406.07800](https://arxiv.org/abs/2406.07800)  
**代码**: [github.com/regulationLee/cwFedAvg](https://github.com/regulationLee/cwFedAvg)  
**领域**: 联邦学习 / 优化  
**关键词**: 个性化联邦学习, 类别级聚合, 权重分布正则化, 数据异质性, 隐私保护

## 一句话总结

cwFedAvg 将 FedAvg 从"按客户端聚合"扩展为"按类别聚合"，为每个类别创建专属全局模型，再根据各客户端的类别分布加权组合成个性化模型，配合权重分布正则化（WDR）增强类别分布与权重范数的关联，在保持 FedAvg 通信开销的同时显著提升非 IID 场景下的个性化性能。

## 研究背景与动机

联邦学习（FL）通过模型聚合实现分布式协作训练，但 FedAvg 在数据异质（non-IID）场景下表现不佳。核心原因在于：

**类别特定路径（Class-specific Pathways）**：深度网络通过权重路径编码类别信息，不同类别的关键路径（大权重组成的路径）呈现不同模式；

**FedAvg 的局限**：其聚合权重仅考虑客户端样本总量 $p_i = n_i/n$，无法反映类别特定路径的差异。一个全局模型无法同时捕捉所有客户端的独特模式。

已有个性化联邦学习（PFL）方法的问题：
- FedFomo/FedAMP：需要下载其他客户端模型或进行成对计算，通信/计算开销大；
- CFL/IFCA：依赖聚类假设（客户端可以分成离散组）；
- FedNH/FedUV：正则化方法改进有限。

## 方法详解

### 整体框架

cwFedAvg 的两步聚合过程：
1. **类别级本地模型聚合**：为每个类别 $j$ 创建专属全局模型；
2. **类别级全局模型聚合**：根据各客户端的类别分布，将 $K$ 个类别全局模型加权合并为个性化本地模型。

### 关键设计

1. **类别级聚合（Class-Wise Aggregation）**：对于 $K$ 类分类任务中的第 $j$ 个类别，类别全局模型通过加权聚合各客户端本地模型获得：

$$\boldsymbol{w}_j^G = \sum_{i=1}^{M} q_{i,j} \boldsymbol{w}_i^L, \quad q_{i,j} = \frac{p_i \cdot p_{i,j}}{\sum_{i=1}^{M} p_i \cdot p_{i,j}} = \frac{n_{i,j}}{\sum_{i=1}^M n_{i,j}}$$

其中 $q_{i,j}$ 表示客户端 $i$ 对类别 $j$ 在系统中的贡献比例。这等价于对每个类别分别执行 FedAvg。个性化本地模型则通过类别分布加权组合：$\boldsymbol{w}_i^L = \sum_{j=1}^K p_{i,j} \boldsymbol{w}_j^G$。

2. **权重分布正则化（WDR）**：为使 cwFedAvg 有效工作，需要模型权重与类别分布强相关。基于 Anand et al. 的理论发现——输出层权重的 $\ell_2$ 范数与对应类别的样本量正相关，定义近似类别分布：

$$\tilde{p}_{i,j} = \frac{\|\mathbf{w}_{i,j}\|_2}{\sum_{k=1}^K \|\mathbf{w}_{i,k}\|_2}$$

WDR 通过最小化 $\tilde{p}_{i,j}$ 与经验类别分布 $p_{i,j}$ 的差距来强化这一关联：

$$\mathcal{R}_i = \|\boldsymbol{p}_i - \tilde{\boldsymbol{p}}_i\|_2$$

总损失为 $\tilde{\mathcal{L}}_i = \mathcal{L}_i + \lambda \mathcal{R}_i$。这同时解决了两个问题：(a) 增强权重与类别分布的关联以提升聚合效果；(b) 可用 $\tilde{p}_{i,j}$ 替代真实 $p_{i,j}$ 发送给服务器，保护隐私（不直接暴露 $n_{i,j}$）。

3. **选择性层应用**：由于深度网络低层学习通用特征、高层学习类别特定特征，cwFedAvg 可仅对输出层（或上层）执行类别级聚合，低层仍用 FedAvg，从而减少服务器端存储 $K$ 个全局模型的内存需求。实验表明仅对输出层应用 cwFedAvg 就能获得大部分性能增益。

### 训练策略

- 通信轮次：1000 轮；
- 本地训练：1 个 epoch，学习率 0.005，batch size 10；
- 正则化系数 $\lambda$：MNIST/CIFAR-10 设为 10，CIFAR-100 设为 1000，Tiny ImageNet 设为 2000；
- 所有类别级聚合在服务端完成，通信成本与 FedAvg 完全相同。

## 实验关键数据

### 主实验

**Pathological setting（每客户端仅有 2 类数据）**：

| 方法 | CIFAR-10 | CIFAR-100 | MNIST |
|------|----------|-----------|-------|
| FedAvg | 60.68 | 28.22 | 98.70 |
| FedFomo | 90.76 | 63.12 | 99.13 |
| FedAMP | 88.82 | 63.29 | 99.26 |
| FedUV | 88.11 | 62.72 | 99.25 |
| **cwFedAvg (Output)** | **91.23** | **67.50** | **99.52** |

**Practical setting（α=0.1 Dirichlet 分布）**：

| 方法 | CIFAR-10 | CIFAR-100 | Tiny ImageNet | Tiny ImageNet* (ResNet-18) |
|------|----------|-----------|---------------|---------------------------|
| FedAvg | 61.94 | 32.44 | 21.35 | 24.71 |
| FedAMP | 89.46 | 47.65 | 29.95 | 31.38 |
| CFL | 61.40 | 44.19 | 29.62 | 33.47 |
| **cwFedAvg (Output)** | **88.65** | **56.29** | **41.38** | **43.51** |

在 CIFAR-100 和 Tiny ImageNet 上的提升尤为显著（+8.64 和 +10.13）。

### 消融实验

| 配置 | CIFAR-100 (α=0.1) | 说明 |
|------|-------------------|------|
| FedAvg 基线 | 32.44 | 无个性化 |
| cwFedAvg (无 WDR) | ~45 | 权重-类别关联不强 |
| cwFedAvg (全层) | ~55 | 全层聚合 |
| cwFedAvg (仅输出层) + WDR | **56.29** | 最佳效率-性能平衡 |

**不同客户端数量和异质程度的鲁棒性**（CIFAR-100）：

| 方法 | 50 clients | 100 clients | α=0.01 | α=0.5 | α=1.0 |
|------|-----------|-------------|--------|-------|-------|
| FedAvg | 32.63 | 32.32 | 28.00 | 36.18 | 36.75 |
| FedAMP | 44.97 | 41.37 | 73.46 | 25.41 | 21.23 |
| cwFedAvg | 需查看原文 | 需查看原文 | 优 | 优 | 优 |

### 关键发现

- IID 极限下 cwFedAvg 退化为 FedAvg（理论证明）；极端 non-IID 下退化为类内 FedAvg（Eq. 9）；
- 权重范数热力图直观显示了 cwFedAvg+WDR 的个性化效果：每个客户端模型的输出层权重模式与其数据分布高度一致；
- FedAMP 在 α 较大（接近 IID）时性能急剧下降，而 cwFedAvg 保持稳定。

## 亮点与洞察

1. **极其简洁的思路**：仅修改 FedAvg 的聚合权重——从 $p_i$ 变为 $p_i \cdot p_{i,j}$——就实现了有效的个性化，算法复杂度极低；
2. **零额外通信开销**：所有类别级聚合在服务端完成，客户端仍只上传/下载一个模型，通信量与 FedAvg 完全相同；
3. **隐私保护设计**：WDR 使得服务器可从模型权重推断类别分布（$\tilde{p}_{i,j}$），无需客户端直接发送敏感的 $n_{i,j}$；
4. **理论-实践闭环**：从神经网络路径理论出发，到权重-类别分布关联的经验观察，再到 WDR 的实践设计，逻辑链完整。

## 局限与展望

- 服务端需存储 $K$ 个全局模型（虽可仅对输出层应用以缓解）；
- $\lambda$ 在不同数据集上需手动调整（从 10 到 2000），缺乏自适应策略；
- 实验仅使用 4 层 CNN 和 ResNet-18，在更大模型上的效果需验证；
- WDR 的理论保证仅限于输出层权重，对中间层的效果是基于反向传播级联效应的经验假设。

## 相关工作与启发

- 与 FedFomo 的区别：FedFomo 需要客户端下载其他客户端模型来学习组合权重（$O(M)$ 通信开销），cwFedAvg 不需要；
- WDR 的类别分布估计可独立用于联邦学习中的客户端选择（client selection），具有超越个性化的通用价值；
- Anand et al. 的梯度-类别大小关系是本文的理论基础。

## 评分

- **新颖性**: ⭐⭐⭐ 思路简洁但并非深度创新，主要是 FedAvg 的自然扩展
- **实验充分度**: ⭐⭐⭐⭐ 4 数据集、多异质度、多客户端数、通路可视化
- **写作质量**: ⭐⭐⭐⭐ 理论动机清晰，可视化有说服力
- **价值**: ⭐⭐⭐⭐ 实用性强——简单、高效、无额外通信开销

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Layer-wise Quantization for Quantized Optimistic Dual Averaging](../../ICML2025/optimization/layer-wise_quantization_for_quantized_optimistic_dual_averaging.md)
- [\[NeurIPS 2025\] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](../../NeurIPS2025/optimization/layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](../../NeurIPS2025/optimization/efficient_adaptive_federated_optimization.md)
- [\[ICML 2025\] FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging](../../ICML2025/optimization/fedswa_improving_generalization_in_federated_learning_with_highly_heterogeneous_.md)
- [\[ICCV 2025\] Memory-Efficient 4-bit Preconditioned Stochastic Optimization](memory-efficient_4-bit_preconditioned_stochastic_optimization.md)

</div>

<!-- RELATED:END -->
