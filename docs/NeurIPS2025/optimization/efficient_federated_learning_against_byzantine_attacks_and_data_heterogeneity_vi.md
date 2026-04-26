---
title: >-
  [论文解读] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients
description: >-
  [NeurIPS 2025][优化][联邦学习] 提出 Fed-NGA 算法，通过对客户端上传的梯度做归一化后加权平均来实现聚合，以 $\mathcal{O}(pM)$ 的极低时间复杂度同时抵御 Byzantine 攻击与数据异质性，并在非凸损失函数下首次证明了特定温和条件下的零最优性间隙收敛。
tags:
  - NeurIPS 2025
  - 优化
  - 联邦学习
  - Byzantine Robustness
  - Gradient Normalization
  - Non-IID Data
  - Non-Convex Optimization
---

# Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients

**会议**: NeurIPS 2025  
**arXiv**: [2408.09539](https://arxiv.org/abs/2408.09539)  
**代码**: 未提供  
**领域**: optimization  
**关键词**: federated learning, Byzantine Robustness, Gradient Normalization, Non-IID Data, Non-Convex Optimization

## 一句话总结

提出 Fed-NGA 算法，通过对客户端上传的梯度做归一化后加权平均来实现聚合，以 $\mathcal{O}(pM)$ 的极低时间复杂度同时抵御 Byzantine 攻击与数据异质性，并在非凸损失函数下首次证明了特定温和条件下的零最优性间隙收敛。

## 背景与动机

联邦学习（Federated Learning, FL）允许多个客户端在不共享原始数据的前提下协作训练模型，但面临两大核心挑战：

1. **Byzantine 攻击**：恶意客户端可以向服务器发送任意伪造的梯度向量，严重破坏全局模型的训练过程。服务器既不知道 Byzantine 客户端的数量，也无法识别其身份。
2. **数据异质性（Non-IID）**：各客户端本地数据分布差异巨大，导致局部最优解存在偏差，阻碍全局模型收敛。

现有 Byzantine 鲁棒聚合方法（Median、Krum、Geometric Median、CClip 等）虽然在一定程度上解决了上述问题，但**聚合阶段的计算开销很高**——时间复杂度普遍为 $\mathcal{O}(pM\log M)$、$\mathcal{O}(pM^2)$ 甚至 $\mathcal{O}(Mp^{3.5})$，严重拖慢训练速度。同时，现有理论分析大多只能保证收敛到驻点的邻域，无法实现零最优性间隙。

## 核心问题

如何在保持 Byzantine 鲁棒性和适应数据异质性的同时，将聚合复杂度降至最低？具体而言：

- 能否设计一种聚合算法使时间复杂度降至 $\mathcal{O}(pM)$（线性于模型维度和客户端数）？
- 在非凸损失函数下，能否在理论上保证收敛到真正的驻点（零最优性间隙）？

## 方法详解

### Fed-NGA 算法框架

Fed-NGA 的核心思想极其简洁：**对每个客户端上传的梯度向量做 $\ell_2$ 归一化，然后按权重求和更新全局模型**。

具体流程如下：

1. **本地更新**：每个诚实客户端 $m$ 根据本地子数据集 $\xi_m^t$ 计算局部梯度 $g_m^t = \nabla F_m(w^t, \xi_m^t)$；Byzantine 客户端可发送任意向量。
2. **归一化聚合**：服务器收到所有向量后，对每个向量除以其 $\ell_2$ 范数进行归一化，再按权重 $\alpha_m$ 加权求和：

$$w^{t+1} = w^t - \eta^t \sum_{m \in \mathcal{M}} \alpha_m \cdot \frac{g_m^t}{\|g_m^t\|}$$

3. **广播**：将更新后的全局参数广播给所有客户端。

### 归一化的关键作用

归一化操作将所有客户端的梯度映射到单位球面上，从而：

- **限制 Byzantine 攻击的影响**：无论恶意客户端发送多大量级的梯度，归一化后其贡献被限制在单位向量范围内，无法通过放大梯度来主导聚合结果。
- **缓解数据异质性**：归一化消除了梯度幅值差异，使聚合更关注梯度方向而非大小，降低了 non-IID 数据导致的偏差。
- **计算高效**：归一化仅需 $\mathcal{O}(p)$ 的范数计算和除法操作，$M$ 个客户端共 $\mathcal{O}(pM)$。

### 理论收敛保证

论文在非凸损失函数下给出两组收敛结论：

**定理 3.7（一般假设下）**：在标准的 Lipschitz 连续（假设 3.1）、无偏梯度（假设 3.2）、有界内部误差（假设 3.3）和有界数据异质性（假设 3.4）条件下，当 Byzantine 比例满足约束时，Fed-NGA 以 $\mathcal{O}(1/T^{1/2-\delta})$ 的速率收敛到驻点的 $\mathcal{O}(\sigma + \theta)$ 邻域。

**定理 3.9（温和假设下零间隙）**：在类型 II 的有界误差假设（假设 3.5、3.6）下，当内部误差和数据异质性足够小时，Fed-NGA 可实现零最优性间隙——即真正收敛到驻点。收敛速率同为 $\mathcal{O}(1/T^{1/2-\delta})$。这在 Byzantine 鲁棒 FL 的文献中是首次被证明。

关键约束条件为 Byzantine 客户端的权重比例 $\bar{C}_\alpha < 0.5$，即恶意客户端的数据量不超过总量的一半。

## 实验关键数据

在 TinyImageNet（MobileNetV3）、CIFAR10（LeNet）、MNIST（MLP）三个数据集上，Dirichlet($\beta=0.6$) non-IID 设置下，测试了 Gaussian、Same-value、Sign-flip、LIE、FoE 五种 Byzantine 攻击。

### 测试准确率

| 数据集 | 无攻击 | Gaussian | Same-value | Sign-flip | LIE | FoE |
|---|---|---|---|---|---|---|
| TinyImageNet (Fed-NGA) | **56.95** | **55.77** | **49.31** | **45.38** | **55.82** | **45.23** |
| TinyImageNet (最佳基线) | 55.68 | 54.22 | 48.98 | 43.14 | 54.34 | 42.88 |
| CIFAR10 (Fed-NGA) | **54.48** | **52.07** | **29.16** | **51.16** | **51.82** | **51.16** |
| MNIST (Fed-NGA) | **96.72** | **94.98** | **83.66** | **94.71** | 94.92 | **94.71** |

### 聚合时间

Fed-NGA 聚合耗时远低于其他鲁棒基线：

- TinyImageNet 上约 0.19-0.34 秒，而 Krum 约 50 秒（>100 倍差距）
- CIFAR10 上约 0.04-0.06 秒，而 Median/Krum 约 22-32 秒（>100 倍差距）
- MNIST 上约 0.04-0.05 秒，而其他方法约 1.2-324 秒

Fed-NGA 的聚合时间仅约为无鲁棒保障的 FedAvg 的 3 倍。

## 亮点

- **方法极简但有效**：仅需对梯度做归一化再加权求和，实现了 $\mathcal{O}(pM)$ 的最优聚合复杂度，远低于现有方法
- **首次证明零最优性间隙**：在温和条件下首次证明 Byzantine 鲁棒 FL 可收敛到真正驻点，而非驻点邻域
- **同时解决两个挑战**：归一化天然地同时抵御 Byzantine 攻击和缓解数据异质性
- **全面实验验证**：覆盖三个数据集、三种模型、五种攻击类型，在准确率和效率上均有优势

## 局限与展望

- 零最优性间隙的理论保证依赖于类型 II 假设（假设 3.5、3.6），要求梯度归一化后的误差和异质性有界，在实际训练后期梯度接近零时可能不满足
- 收敛速率 $\mathcal{O}(1/T^{1/2-\delta})$ 略慢于标准 SGD 的 $\mathcal{O}(1/\sqrt{T})$，其中 $\delta \in (0, 1/2)$ 引入了额外的次优因子
- 仅考虑单步本地更新（每轮每客户端算一次梯度），未扩展到多步本地更新（如 FedAvg 的多轮本地 SGD）
- Same-value 攻击下 CIFAR10 准确率仅 29.16%，虽优于基线但仍有较大下降，表明特定攻击模式仍具挑战

## 与相关工作的对比

| 方法 | 时间复杂度 | 非凸损失 | non-IID | 零间隙 |
|---|---|---|---|---|
| Median | $\mathcal{O}(pM\log M)$ | ✗ | ✗ | ✗ |
| Krum | $\mathcal{O}(pM^2)$ | ✓ | ✗ | ✗ |
| GM (RFA) | $\mathcal{O}(pM\log^3(M\epsilon^{-1}))$ | ✗ | ✓ | ✗ |
| CClip | $\mathcal{O}(\tau Mp)$ | ✓ | ✗ | ✗ |
| MCA | $\mathcal{O}(pM\log^3(M\epsilon^{-1}))$ | ✗ | ✓ | ✗ |
| **Fed-NGA** | $\mathcal{O}(pM)$ | **✓** | **✓** | **✓** |

Fed-NGA 是唯一同时满足非凸损失支持、non-IID 鲁棒性、线性聚合复杂度和零最优性间隙保证的方法。

## 启发与关联

- 梯度归一化作为一种通用技术，在对抗 heavy-tailed noise、sharpness-aware minimization 等场景中已有应用，本文将其引入 Byzantine 鲁棒 FL 展示了跨领域迁移的价值
- 暗示了一种设计哲学：**极简操作+严格理论分析**往往比复杂的聚合机制更具实用价值，尤其在边缘计算资源受限的场景下
- 可进一步探索将归一化与其他鲁棒技术（如 clipping、momentum）结合，或扩展到异步 FL、去中心化 FL 等更复杂的场景

## 评分
- 新颖性: 3.5/5 — 归一化本身不新，但将其作为唯一鲁棒聚合手段并给出完整理论分析是新颖的
- 实验充分度: 4/5 — 三数据集五攻击全面覆盖，但缺少大规模模型和更多 non-IID 设置的实验
- 写作质量: 4/5 — 结构清晰，理论推导规范，对比表格直观
- 价值: 4/5 — 实用性强，尤其适合资源受限的联邦学习部署场景

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[CVPR 2025\] Model Poisoning Attacks to Federated Learning via Multi-Round Consistency](../../CVPR2025/optimization/model_poisoning_attacks_to_federated_learning_via_multi-round_consistency.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](efficient_adaptive_federated_optimization.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)

<!-- RELATED:END -->
