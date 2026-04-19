---
title: >-
  [论文解读] The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions
description: >-
  [优化] 通过"产卵-扰动"实验范式，系统研究神经网络训练轨迹对初始条件的敏感性，发现训练初期极微小的扰动（甚至单个权重）就能导致完全不同的收敛结果——即"蝴蝶效应"，且这种不稳定性与训练噪声无关，随训练进展迅速消减。
tags:
  - 优化
---

# The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions

- **会议**: ICML 2025
- **arXiv**: [2506.13234](https://arxiv.org/abs/2506.13234)
- **代码**: [gsaltintas/lmc](https://github.com/gsaltintas/lmc)
- **领域**: 训练动力学 / 优化理论 / 损失景观
- **关键词**: 蝴蝶效应, 训练稳定性, 损失屏障, 线性模式连通性, 模型合并, 扰动实验

## 一句话总结

通过"产卵-扰动"实验范式，系统研究神经网络训练轨迹对初始条件的敏感性，发现训练初期极微小的扰动（甚至单个权重）就能导致完全不同的收敛结果——即"蝴蝶效应"，且这种不稳定性与训练噪声无关，随训练进展迅速消减。

## 研究背景与动机

神经网络训练已知对初始化和 SGD 随机性敏感。但一个关键未解问题是：**这种敏感性在多大程度上导致实质性不同的网络**——无论是权重还是学习到的函数？

现有研究将训练分为"混沌"（早期）和"稳定"（后期）两个阶段，但存在以下局限：
1. 先前工作仅测量训练噪声（batch 采样、数据增强、GPU 非确定性）引起的不稳定性
2. 无法区分不稳定性来自噪声还是训练过程本身
3. 无法精确量化在损失景观的不同位置需要多大扰动才能引起分歧

本文从**确定性动力系统**视角出发，通过控制扰动消除所有训练噪声，精确刻画训练何时、以多大程度对初始条件敏感。这对模型合并、微调和集成学习等实践场景具有直接指导意义。

## 方法详解

### 整体框架：产卵-扰动实验（Spawn-And-Perturb）

1. 选择初始参数 $\theta_0$ 并训练到扰动时间 $t$，得到 $\theta_t = \mathcal{T}^t(\theta_0; \xi_{1:t})$
2. 复制两份网络，对其中一份添加大小为 $\sigma$ 的扰动：$\theta'_t = \theta_t + \sigma \varepsilon$
3. **使用完全相同的训练噪声** $\xi_{t:T}$ 训练两份网络至收敛
4. 度量分歧 $d(\theta_T, \theta'_T)$

关键区别于 Frankle et al. (2020) 的方法：后者从时间 $t$ 开始使用独立训练噪声，而本文消除了所有训练噪声的差异，仅隔离扰动本身的效果。

### 扰动类型

**Batch 扰动**（沿训练方向）：

$$\hat{\varepsilon}_{\text{Batch}} = \frac{1}{n} \sum_{i=1}^{b} \nabla \ell(x_i, y_i; \theta_t)$$

**高斯扰动**（各向均匀）：

$$\hat{\varepsilon}_{\text{Gaussian}} = [\varepsilon_i^{(l)}], \quad \varepsilon_i^{(l)} \sim \mathcal{N}\left(0, \frac{2}{n_{l-1}}\right)$$

两种扰动都按初始化尺度归一化，确保不会对某些层产生不成比例的影响。

### 功能相似性评估指标

1. **$L^2$ 距离**：参数空间中权重向量的欧氏距离 $\|\theta_T - \theta'_T\|_2$
2. **损失屏障（Barrier）**：线性插值路径上的最大损失增量

$$\sup_{\alpha \in (0,1)} \ell(\alpha \theta_T + (1-\alpha)\theta'_T) - \alpha \ell(\theta_T) - (1-\alpha)\ell(\theta'_T)$$

3. **排列对齐后屏障**：使用权重匹配算法找到排列 $P$ 后计算 $\theta_T$ 与 $P[\theta'_T]$ 之间的屏障
4. **表示相似度**：Angular CKA 测量倒数第二层表示的相似性

## 实验关键数据

### 核心发现一：蝴蝶效应

| 扰动设置 | ResNet-20 on CIFAR-10 | |
|---|---|---|
| 初始化时扰动 1 个权重 | 产生显著损失屏障 | |
| 初始化时扰动 0.01% 权重 | 产生大屏障 | |
| 0.5% 训练进度后同等扰动 | 屏障显著降低 | |
| 50% 训练进度后扰动 | 仅超大扰动(10% 初始化尺度)才产生非零屏障 | |

**关键结论**：训练初始阶段（约前 0.5% 步）极其敏感，单个权重的改变即可导致收敛至不同损失盆地。

### 核心发现二：排列不是根因

比较排列对齐前后的屏障，发现**排列匹配无法降低屏障**。这表明训练不稳定性产生的是真实的功能差异，而非仅仅是等效权重的排列变换。

### 核心发现三：超参数影响

| 设置 | 对稳定性的影响 |
|------|--------------|
| 更宽/更浅的网络（ResNet-8） | 最稳定 |
| 10x 学习率预热 | 显著提升 |
| 4x batch size | 略微提升 |
| Adam 优化器 | 降低稳定性 |
| Weight decay | 降低稳定性 |
| 宽架构 + 长预热 | 进一步提升但不能消除初始化屏障 |

### 核心发现四：预训练与微调稳定性

| 模型 | 场景 | 关键发现 |
|------|------|---------|
| ResNet-50 | CIFAR-100→10 | 比 CIFAR-10→100 更稳定；更长预训练提升稳定性 |
| Multi-BERT | GLUE 任务 | 更长预训练不一定更稳定，2000k checkpoint 在 QNLI/RTE 上反而最不稳定 |
| OLMo | GSM8K | 更长预训练也可能降低微调稳定性 |

### 核心发现五：$L^2$ 与屏障的关系

- 在视觉模型（ResNet）中，**屏障与 $L^2$ 距离呈强 log-线性关系**
- 在语言模型（BERT 微调）中，**二者几乎无相关性**
- $L^2$ 和屏障的增长**不符合线性化动力系统的指数增长预测**

### 集成性能实验

Angular CKA 不相似度与集成性能正相关——有意扰动可以增加模型多样性，提升集成效果。但 ViT 在 CIFAR-100 上不遵循此趋势。

## 亮点与洞察

1. **极简的实验设计揭示深刻规律**：通过消除训练噪声隔离扰动效果，实验设计非常干净
2. **"一个权重就够了"**：单个权重的初始化扰动即可导致分歧，这一发现极具震撼力
3. **实践指导价值明确**：
    - 模型合并：需要确保网络来自同一训练盆地（避免早期分歧）
    - 集成学习：可以通过有意扰动增加多样性
    - 微调：预训练越久不一定越稳定，需要注意过拟合导致的脆弱性
4. **语言模型与视觉模型的本质差异**：$L^2$-屏障关系在两个领域表现完全不同，暗示损失景观的结构性差异

## 局限性

1. **缺乏理论解释**：发现了蝴蝶效应但未给出为何存在的理论分析
2. **$L^2$/屏障增长速率不符合动力系统预测**：说明线性化动力系统模型不适用于神经网络训练，但未提出替代理论
3. **实验规模受限**：OLMo 是最大模型但仅做了初步实验；缺少在更大 LLM 上的系统验证
4. **某些趋势不一致**：如更长预训练对稳定性的影响因任务而异，缺乏统一解释
5. **排列对齐方法的局限**：无法排除更好的排列算法可能降低屏障的可能性

## 相关工作

- **训练稳定性与优化**：Edge of Stability（Cohen et al., 2021）、SGD 噪声分析（Wu et al., 2018）
- **线性模式连通性**：Lottery Ticket 假说衍生的 LMC 研究（Frankle et al., 2020a）
- **模型合并/集成**：Git Re-Basin（Ainsworth et al., 2023）、权重平均（Wortsman et al., 2021）
- **Spawning 实验**：Frankle et al. (2020)、Fort et al. (2020) 的训练噪声稳定性实验
- **微调稳定性**：Juneja et al. (2023) 发现语言模型微调不稳定

## 评分

⭐⭐⭐⭐⭐ (5/5)

这是一项出色的实证研究。实验设计精妙（消除噪声隔离扰动）、发现极具震撼力（单权重蝴蝶效应）、覆盖全面（视觉/语言、预训练/微调、多种架构）、实践价值明确（模型合并与集成指导）。尽管缺乏理论解释，但对训练动力学的经验理解推进巨大。

<!-- RELATED:START -->

## 相关论文

- [Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg](widening_the_network_mitigates_the_impact_of_data_heterogeneity_on_fedavg.md)
- [Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [Federated Continual Instruction Tuning](../../ICCV2025/optimization/federated_continual_instruction_tuning.md)
- [Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](../../CVPR2025/optimization/automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)
- [Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification](interior-point_vanishing_problem_in_semidefinite_relaxations_for_neural_network_.md)

<!-- RELATED:END -->
