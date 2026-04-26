---
title: >-
  [论文解读] Dataset Distillation via the Wasserstein Metric
description: >-
  [ICCV 2025][模型压缩][数据集蒸馏] 提出 WMDD（Wasserstein Metric-based Dataset Distillation），使用 Wasserstein 重心替代 MMD 进行分布匹配，结合逐类 BatchNorm 正则化，在 ImageNet-1K 等大规模数据集上达到 SOTA 数据集蒸馏性能。
tags:
  - ICCV 2025
  - 模型压缩
  - 数据集蒸馏
  - Wasserstein距离
  - 最优传输
  - 分布匹配
  - BatchNorm正则化
---

# Dataset Distillation via the Wasserstein Metric

**会议**: ICCV 2025  
**arXiv**: [2311.18531](https://arxiv.org/abs/2311.18531)  
**代码**: [https://github.com/Liu-Hy/WMDD](https://github.com/Liu-Hy/WMDD)  
**领域**: 数据集蒸馏 / 模型压缩  
**关键词**: 数据集蒸馏, Wasserstein距离, 最优传输, 分布匹配, BatchNorm正则化

## 一句话总结

提出 WMDD（Wasserstein Metric-based Dataset Distillation），使用 Wasserstein 重心替代 MMD 进行分布匹配，结合逐类 BatchNorm 正则化，在 ImageNet-1K 等大规模数据集上达到 SOTA 数据集蒸馏性能。

## 研究背景与动机

数据集蒸馏旨在生成一个紧凑的合成数据集，使得在其上训练的模型能达到接近全量数据训练的性能，从而大幅降低计算开销。现有方法可分为三类：

**性能匹配类**（如 DD、KIP）：双层优化，计算代价高，难以扩展到大数据集

**参数匹配类**（如 DC、MTT）：需要二阶导数计算，内存需求大

**分布匹配类**（如 DM）：计算高效但性能通常不如前两者

**核心矛盾**：分布匹配方法计算效率高但精度不足，其瓶颈在于 MMD（最大均值差异）作为分布度量存在以下问题：(1) 实际实现中通常只匹配一阶矩（均值），等价于线性核 MMD，无法区分高阶矩差异；(2) 使用更复杂的核（如 RBF）则计算代价激增，无法扩展到大数据集。

**切入角度**：最优传输理论中的 Wasserstein 距离天然考虑分布的几何结构，其重心（barycenter）能保留原始分布的结构特征。论文在预训练分类器的特征空间中计算 Wasserstein 重心作为每类数据的紧凑摘要，从而实现高效且精确的分布匹配。

## 方法详解

### 整体框架

WMDD 的流程分为三步：(1) 用预训练分类器提取全量数据的特征；(2) 对每类特征计算 Wasserstein 重心，得到代表性特征点和权重；(3) 通过特征匹配损失和逐类 BN 正则化优化合成图像，使其特征对齐到重心位置。

### 关键设计

1. **Wasserstein 重心计算**：对每类 $n_k$ 个特征点，计算支撑在 $m_k$ 个原子上的 Wasserstein 重心。采用 [Cuturi & Doucet, 2014] 的交替优化算法：

    - **权重优化**：固定位置，求解线性规划获得最优传输方案 $\mathbf{T}$，利用对偶变量 $\boldsymbol{\beta}$ 作为关于权重的次梯度，执行投影次梯度下降
    - **位置优化**：固定权重，目标关于每个合成点位置是二次的（Hessian 为 $2w_j \mathbf{I}$），一步 Newton 更新即可：$\tilde{\mathbf{x}}_j \leftarrow \tilde{\mathbf{x}}_j - \frac{1}{w_j}\sum_i t_{ij}(\tilde{\mathbf{x}}_j - \mathbf{x}_i)$
    - 实验表明仅需 $K=10$ 次交替迭代即可获得高质量合成数据

2. **逐类 BatchNorm 正则化（PCBN）**：传统方法（如 SRe2L）使用全局 BN 统计量对齐合成数据与真实数据的均值/方差。但不同类别的特征分布可能差异较大，全局 BN 无法为不同类的合成样本提供差异化指导。PCBN 独立计算并匹配每个类别在每个 BN 层的均值和方差，且引入 Wasserstein 重心的权重 $w_{k,j}$ 来加权统计量计算。

3. **联合优化目标**：
    $\mathcal{L}(\tilde{\mathbf{X}}) = \mathcal{L}_{\text{feature}}(\tilde{\mathbf{X}}) + \lambda \mathcal{L}_{\text{BN}}(\tilde{\mathbf{X}})$
   其中特征损失是每个合成图像特征到对应重心点的 L2 距离之和，$\lambda$ 为正则化系数。

### 损失函数 / 训练策略

训练分为 squeeze（预训练分类器）和 recover（优化合成图像）两阶段。Recover 阶段使用 Adam 优化器，在 ImageNet-1K 上仅需约 2000 次迭代。合成数据附带的权重用于后续的 FKD（Fast Knowledge Distillation）阶段。

## 实验关键数据

### 主实验

| 方法 | ImageNette 1IPC | ImageNette 10IPC | Tiny-IN 50IPC | ImageNet-1K 10IPC | ImageNet-1K 50IPC |
|------|---------|----------|---------|-----------|-----------|
| Random | 23.5 | 47.7 | 16.8 | 3.6 | 15.3 |
| DM | 32.8 | 58.1 | 24.1 | - | - |
| SRe2L | 20.6 | 54.2 | 41.1 | 21.3 | 46.8 |
| G-VBSM | - | - | 47.6 | 31.4 | 51.8 |
| SCDD | - | - | 45.9 | 32.1 | 53.1 |
| **WMDD** | **40.2** | **64.8** | **59.4** | **38.2** | **57.6** |

在 100 IPC 设置下，WMDD 在三个数据集上分别达到 87.1%、61.0%、60.7%，接近全量数据训练性能（89.9%、63.5%、63.1%）。

### 消融实验

| 特征损失 | 正则化 | ImageNette | Tiny-IN | ImageNet-1K |
|---------|--------|------------|---------|-------------|
| Wasserstein | PCBN | **64.7** | **41.8** | **38.1** |
| CE | PCBN | 63.5 | 41.0 | 36.4 |
| Wasserstein | BN | 60.7 | 36.6 | 26.8 |
| CE | BN | 54.2 | 38.0 | 35.9 |

PCBN + Wasserstein 的组合在所有数据集上显著优于其他组合，说明两个设计缺一不可。直接用 MMD 替代 Wasserstein 度量在 Tiny-IN 和 ImageNet-1K 上接近随机性能。

### 关键发现

- **跨架构泛化**：用 ResNet-18 蒸馏的数据在 ResNet-50/101 和 ViT-Tiny/Small 上均有良好表现（ViT 略弱）
- **计算效率**：WMDD 的 per-iteration 时间仅 0.013s，与 SRe2L（0.015s）相当，但远快于 DC（2.154s）和 DM（1.965s）
- **Wasserstein vs MMD 的理论解释**：Wasserstein 的误差上界仅依赖 Lipschitz 常数，而 MMD 的上界依赖 RKHS 范数，后者在实践中难以精确控制

## 亮点与洞察

- 巧妙地将最优传输理论引入数据集蒸馏领域，用 Wasserstein 重心替代简单的均值匹配
- PCBN 的设计思路简洁有效——不同类的 BN 统计量不应混在一起
- 保持了分布匹配方法的计算效率优势，同时性能追平甚至超过双层优化方法
- 嵌入空间中计算重心带来的额外开销极小（整体仅增加约 10s）

## 局限与展望

- 依赖预训练分类器的质量，分类器本身的偏见可能传递到合成数据
- Wasserstein 重心的最优传输求解在超大类别数时可能成为瓶颈
- 论文未探索与生成模型（如 GAN/Diffusion）结合的可能性
- ViT 等 data-hungry 架构上的跨架构泛化仍有较大提升空间

## 相关工作与启发

- **SRe2L** 系列（squeeze-recover-relabel）是最直接的基线，WMDD 在其 recover 阶段引入 Wasserstein 匹配
- Sliced Wasserstein 距离可些微加速但性能略降（Table 5），说明完整的 OT 计算对于质量至关重要
- 方法可能扩展到其他需要分布摘要的场景，如联邦学习中的数据共享

## 评分

- 新颖性：⭐⭐⭐⭐ — 将 OT 理论系统引入 DD 是新颖视角
- 理论深度：⭐⭐⭐⭐ — 提供了 Wasserstein vs MMD 的误差上界分析
- 实验充分度：⭐⭐⭐⭐⭐ — 三个数据集、多种 IPC、跨架构、效率分析齐全
- 实用性：⭐⭐⭐⭐⭐ — 计算效率优秀，可扩展到 ImageNet-1K

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Heavy Labels Out! Dataset Distillation with Label Space Lightening](heavy_labels_out_dataset_distillation_with_label_space_lightening.md)
- [\[CVPR 2025\] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](../../CVPR2025/model_compression/dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](../../NeurIPS2025/model_compression/hyperbolic_dataset_distillation.md)
- [\[NeurIPS 2025\] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](../../NeurIPS2025/model_compression/optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](../../AAAI2026/model_compression/tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)

<!-- RELATED:END -->
