---
title: >-
  [论文解读] Distill Gold from Massive Ores: Bi-level Data Pruning towards Efficient Dataset Distillation
description: >-
  [ECCV 2024][Dataset Distillation] 提出双层数据剪枝策略 BiLP，通过经验损失静态剪枝和基于因果效应 (ITE) 的动态剪枝，高效选择对数据集蒸馏最有价值的真实样本，以即插即用方式一致性提升现有蒸馏方法性能并降低计算开销。
tags:
  - ECCV 2024
  - Dataset Distillation
  - 剪枝
  - Causal Inference
  - Individual Treatment Effect
  - Neural Tangent Kernel
---

# Distill Gold from Massive Ores: Bi-level Data Pruning towards Efficient Dataset Distillation

**会议**: ECCV 2024  
**arXiv**: [2305.18381](https://arxiv.org/abs/2305.18381)  
**代码**: [silicx/GoldFromOres-BiLP](https://github.com/silicx/GoldFromOres-BiLP)  
**领域**: causal_inference  
**关键词**: Dataset Distillation, Data Pruning, Causal Inference, Individual Treatment Effect, Neural Tangent Kernel

## 一句话总结

提出双层数据剪枝策略 BiLP，通过经验损失静态剪枝和基于因果效应 (ITE) 的动态剪枝，高效选择对数据集蒸馏最有价值的真实样本，以即插即用方式一致性提升现有蒸馏方法性能并降低计算开销。

## 研究背景与动机

**数据集蒸馏的核心目标**：从大规模真实数据集中学习一个小型合成数据集，使在合成数据上训练的模型达到接近全量数据的性能，是数据高效学习的重要方向。

**真实数据存在严重冗余**：实验发现在 CIFAR10 上使用 DC 方法 (IPC=1) 时，剪掉 90% 的真实数据不会降低蒸馏性能，甚至只需 0.04% 的样本即可完成蒸馏，说明传统蒸馏算法浪费了大量计算在冗余样本上。

**现有工作忽视真实数据选择**：已有蒸馏方法主要关注匹配策略（梯度匹配、分布匹配、轨迹匹配等）或优化加速，很少分析哪些真实样本对蒸馏过程本身是有价值的。

**合成数据容量有限**：由于合成数据规模远小于真实数据，NTK 矩阵存在秩亏缺，小规模合成数据无法记忆所有真实样本的模式，选择"重要且容易"的样本才能帮助蒸馏。

**静态剪枝不够充分**：单纯的预处理剪枝在某些高 IPC 场景下失效，说明样本的价值在训练过程中会动态变化，需要自适应的动态选择机制。

**大规模数据集蒸馏瓶颈**：ImageNet-1K、Kinetics-400 等大规模数据集的蒸馏极其耗时，亟需高效的数据利用策略来降低计算成本。

## 方法详解

### 整体框架：双层数据剪枝 (BiLP)

BiLP 包含两个层级的数据选择机制，以即插即用的方式嵌入现有蒸馏算法：

- **Level 1 — 预先剪枝 (Preemptive Pruning)**：蒸馏开始前，基于经验损失值静态裁剪冗余样本
- **Level 2 — 自适应剪枝 (Adaptive Pruning)**：蒸馏过程中，基于因果效应 (ITE) 动态选择当前最有贡献的样本

### 关键设计 1：基于经验损失的静态剪枝

通过 Neural Tangent Kernel (NTK) 分析蒸馏动力学，论文发现：由于合成数据规模 $M_s \ll M_r$，NTK 矩阵 $\boldsymbol{K}_r$ 存在秩亏缺，导致合成数据的训练动态难以逼近全量真实数据的动态。梯度值 $\|\frac{\partial \ell}{\partial u_r}\|$ 大的样本（即经验损失大的"难样本"）对蒸馏贡献小，可以安全剪掉。又因为宽神经网络的 NTK 近似恒定，该准则在训练初就有效，支持一次性预先剪枝。

具体操作：先在全量数据上训练网络若干 epoch，记录每个样本的平均损失值，按阈值 $\tau$ 保留损失值最小的 $(1-\alpha)$ 比例的样本。

### 关键设计 2：基于因果效应 (ITE) 的动态剪枝

将每个真实样本的"存在与否"视为二元处理 (treatment)，合成数据的 meta 梯度视为效应 (effect)，通过个体处理效应 (Individual Treatment Effect) 衡量样本贡献：

$$ITE(x_r) = \frac{\partial \mathcal{L}_{meta}(\mathcal{D}_r, \mathcal{D}_s)}{\partial \mathcal{D}_s} - \frac{\partial \mathcal{L}_{meta}(\mathcal{D}_r \setminus \{x_r\}, \mathcal{D}_s)}{\partial \mathcal{D}_s}$$

使用 $\|ITE(x_r)\|_2$ 作为选择准则，**同时剪掉 ITE 值最大和最小的样本**：最小 ITE 样本对合成数据学习贡献小，最大 ITE 样本可能是异常值会增加训练不稳定性，保留中间 ITE 值的样本最优。

### 关键设计 3：ITE 计算加速（三重优化）

原始 ITE 计算需要对每个样本额外计算一次 meta 梯度，开销巨大。论文提出三项加速：

1. **Taylor 近似**：对可加分离的 meta loss 函数，利用一阶 Taylor 展开将 ITE 计算简化为一次 Hessian 矩阵乘法 + 逐样本特征向量，时间减少 98.8%
2. **全局 ITE 分布估计**：维护 ITE 的 running mean $\hat{\mu}$ 和 variance $\hat{\sigma}^2$，假设正态分布，利用分位点确定剪枝阈值，无需遍历全数据集
3. **Lazy Selection**：每隔若干迭代才更新一次剪枝决策

三项优化将 ITE 计算开销压缩到原始的 1/60,000，总额外训练时间仅增加 7%。

## 损失函数与训练策略

BiLP 不引入新的损失函数，而是作为即插即用模块嵌入现有蒸馏方法（如 DC、IDC）的 meta loss 之上。训练流程：先预训练分类网络得到经验损失 → 按比例 $\alpha$ 预先剪枝 → 蒸馏过程中在每个 mini-batch 内计算 ITE → 按比例 $\beta$ 自适应剪枝 → 用剪枝后的样本更新合成数据。

## 实验

### 主实验：蒸馏性能对比 (Table 4)

| 方法 | CIFAR10 IPC=1 | CIFAR10 IPC=10 | CIFAR10 IPC=50 | CIFAR100 IPC=1 | CIFAR100 IPC=10 |
|------|:---:|:---:|:---:|:---:|:---:|
| DC | 28.3 | 44.9 | 53.9 | 12.8 | 25.2 |
| IDC | 50.6 | 67.5 | 74.5 | — | 45.1 |
| MTT | 46.3 | 65.6 | 71.6 | 24.3 | 40.1 |
| DREAM | 51.1 | 69.4 | 74.8 | 29.5 | 46.8 |
| **BiLP+DC** | **30.5** | **45.2** | **54.9** | **13.7** | **26.0** |
| **BiLP+IDC** | **51.5** | **69.4** | **75.4** | **30.1** | **47.2** |
| **BiLP+IDC (x3)** | **55.9** | **69.8** | **76.9** | **34.0** | **48.0** |

BiLP 一致性提升 DC (+0.8%) 和 IDC (+1.2%)，在 CIFAR100 等多样化数据集上提升更显著；BiLP+IDC (x3) 超越大多数 SOTA 方法。

### 大规模数据集蒸馏 (Table 7)

| 数据集 | IPC | 方法 | Full Data Acc | BiLP Acc | BiLP 训练时间 |
|--------|:---:|------|:---:|:---:|:---:|
| ImageNet-1K | 1 | DC | 1.79 | **2.02** | 17.3h (↓27%) |
| ImageNet-1K | 10 | DM | 3.86 | **5.21** | 15.3h (↓38%) |
| ImageNet-1K | 50 | DM | 8.22 | **9.41** | 20.6h (↓41%) |
| Kinetics-400 | 1 | DM | 2.78 | **2.92** | 29.6h (↓21%) |
| Kinetics-400 | 10 | DM | 9.48 | **9.70** | 32.2h (↓26%) |

在 ImageNet-1K 和 Kinetics-400 上，BiLP 同时提升精度并减少最多 60% 的训练时间，尤其适合大规模场景。

### 消融实验要点

- 预先剪枝和自适应剪枝各自独立贡献性能增益，二者结合效果最佳
- 同时剪掉大 ITE 和小 ITE 的样本优于单方向剪枝
- Taylor 近似引入的样本排序偏差仅约 3%，可忽略不计
- 经验损失只需少量 epoch（5 epoch 即可）和少量训练次数即可准确估计
- ITE 三重优化将额外计算开销压缩到仅 7%

## 亮点

- **视角新颖**：首次系统分析数据集蒸馏中真实数据的冗余性，从 NTK 理论和因果推断两个角度提供理论支撑
- **即插即用**：BiLP 可直接嵌入 DC、IDC 等多种蒸馏方法，通用性强
- **效率与效果兼得**：在提升蒸馏精度的同时大幅减少训练时间（大规模数据集减少 27%–60%）
- **理论与实验双重验证**：NTK 秩亏缺理论解释了冗余性，ITE 因果框架刻画了样本动态贡献，实验覆盖 6 个数据集 + 7 种蒸馏方法

## 局限性

- **高阶样本交互未建模**：当前方法基于逐样本独立评估，未考虑样本间的交互作用（如多样性），在低多样性数据集（如 MNIST）上表现不佳
- **对蒸馏方法的提升幅度有限**：平均提升 0.8%–1.2%，绝对增益不大
- **Taylor 近似依赖可加分离假设**：仅适用于 meta loss 可分解为逐样本贡献之和的蒸馏方法，对 MTT 等轨迹匹配方法不直接适用
- **预先剪枝需要额外预训练**：虽然几个 epoch 即可，但仍有一定前置开销

## 相关工作

- **数据集蒸馏方法**：DC (梯度匹配)、DM (分布匹配)、MTT (轨迹匹配)、IDC (多形成因子)、DREAM (特征分布剪枝) 等，BiLP 与之互补
- **数据选择/剪枝**：传统方法基于预定义标量分数筛选样本，BiLP 首次将因果推断引入蒸馏场景的数据选择
- **Neural Tangent Kernel**：用于建模训练动力学，BiLP 利用 NTK 的稳定性支持静态早期剪枝
- **因果推断中的 ITE**：个体处理效应原用于观察性研究，BiLP 创新性地将其应用于衡量真实样本对合成数据梯度的影响

## 评分

- 新颖性: ⭐⭐⭐⭐ — 因果推断 + NTK 双重视角分析蒸馏数据冗余，框架新颖
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 6 数据集、7 蒸馏方法，含大规模 ImageNet-1K 和视频 Kinetics-400，消融详细
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，从观察到理论到方法层层递进
- 价值: ⭐⭐⭐⭐ — 即插即用的通用框架，对大规模蒸馏场景的效率提升实用价值高

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](../../NeurIPS2025/causal_inference/bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/causal_inference/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [\[ICLR 2026\] Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol](../../ICLR2026/causal_inference/validating_interpretability_in_sirna_efficacy_prediction_a_perturbation-based_da.md)
- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](../../ICLR2026/causal_inference/efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)

<!-- RELATED:END -->
