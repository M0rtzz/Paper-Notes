---
title: >-
  [论文解读] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks
description: >-
  [CVPR2026][层选择] 提出 AdaBet，一种基于代数拓扑（第一 Betti 数 $b_1$）的无梯度层选择方法，仅通过前向传播计算每层激活空间的拓扑复杂度来决定哪些层需要微调，无需标签、梯度或反向传播，在 ResNet50/VGG16/MobileNetV2/ViT-B16 上以仅 10% 层微调达到优于全量训练的准确率，同时峰值内存降低约 40%。
tags:
  - CVPR2026
  - 层选择
  - Betti数
  - 拓扑数据分析
  - 无梯度微调
  - 边缘设备
  - 迁移学习
---

# AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks

**会议**: CVPR2026  
**arXiv**: [2510.03101](https://arxiv.org/abs/2510.03101)  
**代码**: https://github.com/Nokia-Bell-Labs/efficient_layer_selection  
**领域**: 高效训练 / 边缘计算  
**关键词**: 层选择, Betti数, 拓扑数据分析, 无梯度微调, 边缘设备, 迁移学习

## 一句话总结
提出 AdaBet，一种基于代数拓扑（第一 Betti 数 $b_1$）的无梯度层选择方法，仅通过前向传播计算每层激活空间的拓扑复杂度来决定哪些层需要微调，无需标签、梯度或反向传播，在 ResNet50/VGG16/MobileNetV2/ViT-B16 上以仅 10% 层微调达到优于全量训练的准确率，同时峰值内存降低约 40%。

## 研究背景与动机

**领域现状**：深度神经网络在边缘设备（手机、IoT、嵌入式系统）上的微调需求日益增长，但边缘设备内存和计算资源极度有限，全量微调不可行。

**现有痛点**：(1) 传统迁移学习冻结大部分层、只训练最后几层，但这种启发式选择忽略了中间层可能也需要适配新任务；(2) 基于 Fisher Information 的层选择方法需要反向传播和标签数据，在无标签或隐私敏感场景下不适用；(3) 结构化剪枝（PruneTrain）和弹性训练（ElasticTrainer）虽降低计算量，但仍依赖梯度信息。

**核心矛盾**：层选择需要衡量每层对新任务的"重要性"，但现有度量（Fisher、梯度范数）都需要反向传播，这本身就是计算瓶颈——用昂贵操作来决定如何省钱，逻辑上矛盾。

**本文目标**：如何在不需要标签、梯度、反向传播的情况下，仅用前向传播来判断哪些层最需要更新？

**切入角度**：从拓扑数据分析出发，用每层激活空间的拓扑结构（特别是 1 维环结构）衡量该层特征的复杂度/纠缠程度。

**核心 idea**：高 $b_1$（多 1 维环）= 激活空间流形纠缠 = 预训练特征与新任务不对齐 → 需要更新；低 $b_1$（少环）= 近似线性可分 = 可直接复用 → 冻结。

## 方法详解

### 整体框架
AdaBet 是一个两阶段流程：(1) 层重要性评估——对未标注数据做一次前向传播，计算每层激活的归一化第一 Betti 数 $\hat{b}_1$；(2) 层选择与微调——选择 top-$\rho$ 比例（默认 10%）的层进行微调，其余冻结。整个过程不需要服务器端 meta-training，可完全在边缘设备上完成。

### 关键设计

1. **第一 Betti 数 $b_1$ 的计算**:

    - 功能：量化每层激活空间的 1 维拓扑特征（环/loops 的数量）
    - 核心思路：对第 $i$ 层的激活张量 $a_i \in \mathbb{R}^{B \times C \times H \times W}$，先展平为 $\mathbb{R}^{B \times d}$（$d = C \times H \times W$），构建 Vietoris-Rips 复形，计算持续同调（persistent homology）中 1 维特征的 Betti 数 $b_1^{(i)}$——即在不同尺度下持续存在的 1 维环结构数量
    - 设计动机：$b_1$ 捕获的是流形的"纠缠程度"。若激活空间存在大量 1 维环，说明不同类别的特征流形相互缠绕，线性分类器无法分开 → 该层需要调整以"解缠"

2. **归一化 Betti 数 $\hat{b}_1$**:

    - 功能：平衡拓扑重要性与计算成本
    - 核心思路：$\hat{b}_1^{(i)} = b_1^{(i)} / |a_i|$，其中 $|a_i|$ 是第 $i$ 层激活张量的参数量（$C \times H \times W$）。更大的层自然倾向于有更高的 $b_1$，归一化消除这种尺度偏差
    - 设计动机：直接用 $b_1$ 会偏向选择大层（参数多的层），而大层微调的内存/计算代价也更高。归一化后选择"单位参数量拓扑复杂度最高"的层，实现性能与资源的最优权衡

3. **层选择策略**:

    - 功能：按 $\hat{b}_1$ 降序排列，选 top-$\rho$ 比例的层微调
    - 核心思路：给定微调比例 $\rho$（默认 10%），选择 $\hat{b}_1$ 最高的 $\lceil \rho \times L \rceil$ 层启用梯度更新，其余层关闭梯度。分类头始终参与训练
    - 设计动机：$\rho$ 是用户可调的旋钮，在极端资源受限场景可设为 5%，资源充足时可设为 20%。默认 10% 在多个架构上表现最优

4. **Channel 级扩展 $\rho_{ch}$**:

    - 功能：在选中层内进一步选择最重要的通道微调
    - 核心思路：对被选中层的每个通道分别计算 $b_1$，再选 top-$\rho_{ch}$ 比例的通道微调。实现更细粒度的资源节省
    - 设计动机：某些层可能只有部分通道需要adapt，channel 级选择进一步减少可训练参数

### 与 Fisher Information 的关键对比
- **FI 的问题**：(1) 需要反向传播 + 标签数据；(2) 对 batch 大小和 backprop 次数高度敏感——不同 batch 给出不同层排序（论文 Fig.3 展示了 FI 在不同 backprop 次数下的剧烈变化）；(3) 在 batch 有限的边缘设备上不可靠
- **$b_1$ 的优势**：(1) 仅需前向传播，无需标签；(2) 对不同 batch 高度一致（论文验证了跨 batch 的 $b_1$ 排序稳定性 > 0.95 Kendall-$\tau$）；(3) 计算复杂度与 batch 大小线性相关

### 损失函数 / 训练策略
- 微调阶段使用标准交叉熵损失，冻结层不计算梯度
- 层选择在训练前一次性完成，训练过程中不再改变

## 实验关键数据

### 主实验（准确率 %）

| 方法 | ResNet50 Flowers | ResNet50 CIFAR-100 | MobileNetV2 Cars | ViT-B16 CIFAR-100 |
|------|-----------------|-------------------|-----------------|-------------------|
| Full Training | 82.3 | 75.8 | 80.1 | 85.2 |
| Transfer Learning (last layers) | 79.5 | 72.1 | 76.8 | 82.7 |
| PruneTrain | 80.1 | 73.6 | 77.4 | 83.5 |
| ElasticTrainer | 81.2 | 74.3 | 78.9 | 84.1 |
| Fisher-based Selection | 81.8 | 74.9 | 79.2 | 84.6 |
| **AdaBet (ours, ρ=10%)** | **84.5** | **77.4** | **82.8** | **87.1** |

### 资源效率

| 方法 | 可训练参数 (%) | 峰值内存 (相对) | 训练时间 (相对) |
|------|--------------|----------------|----------------|
| Full Training | 100% | 1.00× | 1.00× |
| Transfer Learning | ~20% | 0.72× | 0.45× |
| Fisher Selection (10%) | 10% | 0.62× | 0.88× (含FI计算) |
| **AdaBet (ρ=10%)** | **10%** | **0.60×** | **0.52×** |

### 消融实验

| 配置 | ResNet50 Flowers | CIFAR-100 |
|------|-----------------|-----------|
| 随机层选择 (10%) | 80.8 | 73.2 |
| 按参数量选择 | 81.1 | 73.8 |
| $b_1$ 不归一化 | 82.9 | 75.6 |
| $b_1$ 归一化 (AdaBet) | **84.5** | **77.4** |
| AdaBet + Channel (ρ_ch=50%) | 84.2 | 77.1 |

### 关键发现
- **$b_1$ 归一化至关重要**：不归一化时准确率下降 1.6-1.8%，因为会偏向选择大层而忽略关键的小层
- **远优于 Fisher**：AdaBet 比 Fisher 高 2.5%+，且训练总时间更少（Fisher 需要额外反向传播来估计信息量）
- **10% 层微调 > 100% 全量训练**：这看似反直觉，但合理——冻结已良好对齐的层避免了灾难性遗忘和过拟合
- **$b_1$ 排序跨 batch 高度稳定**：Kendall-τ > 0.95，而 Fisher 在不同 batch 下 τ < 0.7
- **架构泛化性强**：在 CNN（ResNet50, VGG16, MobileNetV2）和 Transformer（ViT-B16）上均有效

## 亮点与洞察
- **拓扑视角解读层适配**：将"哪些层需要微调"问题转化为拓扑特征分析，这是一个全新视角。高 $b_1$ = 流形纠缠 = 需要解缠的直觉非常优雅
- **不需要标签和梯度**：这对隐私保护场景（联邦学习、边缘设备个性化）价值巨大——设备无需将数据/梯度发送到服务器
- **一次性前向传播决策**：层选择只需一次前向传播，后续训练过程与标准微调完全相同，实现简单
- **归一化设计精巧**：$\hat{b}_1 = b_1/|a_i|$ 同时平衡了重要性和计算成本，一个除法解决两个问题

## 局限与展望
- Betti 数的计算本身有开销（Vietoris-Rips 复形构建），对高维激活空间可能需要降维采样，论文未充分讨论大规模模型（如 LLaMA）的可行性
- $\rho$ 比例是手动设定的超参数，不同任务最优 $\rho$ 可能不同，缺少自适应选择策略
- 只在分类任务上验证，检测/分割/生成等任务的效果未知
- Channel 级选择 $\rho_{ch}$ 带来的额外提升有限（仅 ~0.3%），但实现复杂度显著增加
- 未与 LoRA/Adapter 等参数高效微调方法对比，这些方法也是边缘微调的候选方案

## 相关工作与启发
- **vs Fisher Information Selection**: FI 需要反向传播 + 标签，且对 batch 敏感，AdaBet 完全不依赖梯度且稳定性显著更强
- **vs ElasticTrainer**: ElasticTrainer 用弹性搜索策略选择层，仍需梯度信息，AdaBet 前向传播即可
- **vs PruneTrain**: PruneTrain 在训练过程中动态剪枝，AdaBet 在训练前一次性决策，更简单高效
- **vs LoRA/Adapter**: LoRA 在每层插入低秩矩阵，AdaBet 选择性冻结整层；两者可结合——对 AdaBet 选中的层用 LoRA 微调
- **拓扑数据分析在 DL 中的应用**：之前主要用于分析训练动态和数据复杂度，AdaBet 首次将 TDA 用于指导训练策略

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将代数拓扑（Betti 数）用于层选择，视角全新且理论直觉优雅
- 实验充分度: ⭐⭐⭐⭐ 4 个架构 × 4 个数据集，消融完整，但缺少与 LoRA 等 PEFT 方法对比
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，FI vs Betti 的对比可视化有说服力
- 价值: ⭐⭐⭐⭐ 对边缘设备微调场景有实际意义，且方法简单易用

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](../../ICLR2026/llm_evaluation/improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)
- [\[ICML 2025\] Improving the Effective Receptive Field of Message-Passing Neural Networks](../../ICML2025/llm_evaluation/improving_the_effective_receptive_field_of_message-passing_neural_networks.md)
- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)
- [\[ICML 2025\] Fully Heteroscedastic Count Regression with Deep Double Poisson Networks](../../ICML2025/llm_evaluation/fully_heteroscedastic_count_regression_with_deep_double_poisson_networks.md)
- [\[NeurIPS 2025\] HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization](../../NeurIPS2025/llm_evaluation/hybridnorm_towards_stable_and_efficient_transformer_training_via_hybrid_normaliz.md)

<!-- RELATED:END -->
