---
title: >-
  [论文解读] Recurrent Memory for Online Interdomain Gaussian Processes
description: >-
  [NeurIPS 2025][图像生成][高斯过程] 提出 OHSVGP（Online HiPPO Sparse Variational Gaussian Process），将深度学习中的 HiPPO（高阶多项式投影算子）框架引入稀疏变分高斯过程中作为跨域诱导变量，利用时变正交多项式基函数实现在线学习中的长期记忆保持，核矩阵可通过 ODE 递推高效更新。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "高斯过程"
  - "HiPPO"
  - "在线学习"
  - "长期记忆"
  - "状态空间模型"
---

# Recurrent Memory for Online Interdomain Gaussian Processes

**会议**: NeurIPS 2025  
**arXiv**: [2502.08736](https://arxiv.org/abs/2502.08736)  
**代码**: [GitHub](https://github.com/harrisonzhu508/HIPPOSVGP)  
**领域**: 高斯过程 / 在线学习  
**关键词**: 高斯过程, HiPPO, 在线学习, 长期记忆, 状态空间模型

## 一句话总结

提出 OHSVGP（Online HiPPO Sparse Variational Gaussian Process），将深度学习中的 HiPPO（高阶多项式投影算子）框架引入稀疏变分高斯过程中作为跨域诱导变量，利用时变正交多项式基函数实现在线学习中的长期记忆保持，核矩阵可通过 ODE 递推高效更新。

## 研究背景与动机

**领域现状**：高斯过程（GP）因其函数表达能力和不确定性量化能力，是时间序列建模的经典选择。但 GP 面临 $O(n^3)$ 计算和 $O(n^2)$ 存储的复杂度瓶颈。稀疏变分高斯过程（SVGP）通过引入诱导点降低复杂度，OSVGP 进一步使其支持在线学习。

**核心痛点**：OSVGP 在在线学习中存在**灾难性遗忘**问题。随着新数据到来，诱导点会不可避免地漂移到最新任务的数据区域，导致早期任务的记忆丧失。除非不断增加诱导点数量，否则无法维持长期记忆。

**本文切入角度**：HiPPO 框架在深度学习中因其出色的长程记忆能力（S4、Mamba 的基础）而闻名。本文将 HiPPO 的时变正交多项式投影解释为跨域 GP 的诱导变量，以此在固定数量的诱导变量下实现长期记忆的有效保持。

## 方法详解

### 整体框架

OHSVGP 的核心思路：
1. 将 HiPPO 的多项式投影系数解释为跨域稀疏变分 GP 的诱导变量
2. 利用 HiPPO 的 ODE 递推更新核矩阵，避免重新计算
3. 结合在线变分推断框架，在新数据到达时高效更新后验

### 关键设计

1. **HiPPO 作为跨域诱导变量（Section 3.1）**：在标准跨域 GP 中，诱导变量定义为 $u_m = \int f(x) \phi_m(x) dx$，其中 $\phi_m$ 是基函数。OHSVGP 使用 HiPPO 的时变基函数 $\phi_m^{(t)}(x) = g_m^{(t)}(x) \omega^{(t)}(x)$，其中 $g_m^{(t)}$ 是时变正交多项式（如 Legendre 多项式），$\omega^{(t)}$ 是时变度量函数。

    - **设计动机**：传统跨域 GP 使用固定度量的基函数（如固定区间的均匀度量），新任务的时间索引可能超出预定义范围。HiPPO 的自适应基函数随时间扩展，自然覆盖新到达的数据区域。
    - 诱导变量 $u_m^{(t)} = \int f(x) \phi_m^{(t)}(x) dx$ 不再是固定随机变量，而是随时间演化的随机过程。

2. **核矩阵的 ODE 递推更新（Section 3.2）**：核矩阵可以通过 HiPPO 的 ODE 参数高效更新：

    - 先验交叉协方差：$\frac{d}{dt}[\mathbf{K}_{\mathbf{fu}}^{(t)}]_{n,:} = \mathbf{A}(t)[\mathbf{K}_{\mathbf{fu}}^{(t)}]_{n,:} + \mathbf{B}(t)k(x_n, t)$
    - 诱导变量协方差 $\mathbf{K}_{\mathbf{uu}}^{(t)}$ 涉及双重积分，利用 Bochner 定理和随机傅里叶特征（RFF）分解为两个单积分的乘积之和，每个单积分都可以通过 HiPPO ODE 递推更新。使用 1000 个 RFF 样本近似。
    - **设计动机**：避免每次新数据到来时重新计算核矩阵，计算量从 $O(NM^2)$ 降低到增量更新。

3. **多维输入扩展（Section 3.3）**：对于非时间序列数据（如 UCI 数据集的持续学习），需要对训练样本排序以创建伪时间顺序。提出两种排序策略：

    - OHSVGP-o：使用与任务划分一致的oracle排序
    - OHSVGP-k：基于核相似度的启发式排序，$\mathbf{x}_i^{(j)} = \arg\max k(\mathbf{x}, \mathbf{x}_{i-1}^{(j)})$

### 训练策略

- 使用在线 ELBO（Eq. 3）进行变分更新，前一任务的后验作为下一任务的先验
- 核超参数仅在初始任务中训练，之后固定（避免在线更新时的不稳定性）
- 对于共轭高斯似然，后验有闭式解（OHSGPR），完全不需要训练迭代

## 实验关键数据

### 主实验1：时间序列预测（NLPD↓）

| 数据集 | 方法 | M=50 (任务10后) | M=150 (任务10后) |
|--------|------|---------|---------|
| Solar | OSGPR | ~2.5（灾难性遗忘） | ~1.8 |
| Solar | OVC | ~1.2 | ~0.9 |
| Solar | OVFF | ~1.0 | ~0.8 |
| Solar | **OHSGPR** | **~0.8** | **~0.7** |
| Audio | OSGPR | 严重遗忘 | 中度遗忘 |
| Audio | **OHSGPR** | **最优** | **最优** |

OSGPR 从约第5个任务开始出现灾难性遗忘，OHSGPR 在整个学习过程中保持一致性能。

### 主实验2：运行时间比较（秒）

| 方法 | Solar M=50 | Solar M=150 | Audio M=100 | Audio M=200 |
|------|-----------|-------------|-------------|-------------|
| OSGPR | 140 | 149 | 144 | 199 |
| OVC | 0.450 | 0.620 | 0.558 | 0.863 |
| OVFF | 0.327 | 0.354 | 0.295 | 0.356 |
| **OHSGPR** | **0.297** | **0.394** | **0.392** | **0.655** |

OHSGPR 比 OSGPR 快 **300-450倍**，因为不需要优化诱导点位置。

### 消融实验：持续学习排序策略影响

| 数据集(排序方式) | OSVGP | OVC | OHSVGP-k | OHSVGP-o |
|-----------------|-------|-----|----------|----------|
| Skillcraft(1st dim) | 最差 | 中等 | 类似OSVGP | **最优** |
| Skillcraft(L2) | 最差 | 中等 | 类似OSVGP | **最优** |
| Powerplant(1st dim) | 遗忘 | 遗忘 | 中等 | **最优** |
| Powerplant(L2) | 遗忘 | 中等 | 中等 | **最优** |

### 关键发现

- OHSVGP-o（使用oracle排序）在所有场景下都是最优的，说明排序策略对多维输入至关重要
- OVFF 在早期任务中严重欠拟合，因为其诱导变量在预定义的全时间区间上积分，早期信息被稀释
- 即使使用非共轭似然（如 COVID 数据的负二项分布），OHSVGP 仍优于基线
- OVC-optZ（进一步优化诱导点）在后期反而性能下降，说明在线ELBO目标无法保证诱导点的最优在线更新

## 亮点与洞察

- 将 HiPPO 从 RNN/SSM 领域引入 GP 领域是非常自然且优雅的跨领域迁移
- 核矩阵的 ODE 递推更新避免了重新计算，使 OHSGPR 在共轭情形下完全无需训练
- 有限基重建（$f = \sum_{m=1}^M u_m^{(t)} g_m^{(t)}(x)$）作为副产品提供了可解释的函数近似
- 与 SVGP 中其他近似推断框架（EP、Laplace）兼容，只改变核矩阵计算方式

## 局限与展望

- RFF 近似 $\mathbf{K}_{\mathbf{uu}}^{(t)}$ 可能在长时间序列上累积误差
- 多维输入需要排序，排序策略对性能影响大，OHSVGP-k 在某些场景下不优于 OSVGP
- 核超参数固定在初始任务后，无法适应分布变化
- 尚未在超大规模时间序列或高维输出（如视频生成）上充分验证

## 相关工作与启发

- 与 Markovian GP（也有递推结构）不同，OHSVGP 专为在线学习设计
- VFF（Variational Fourier Features）需要预定义时间区间，不适合在线场景
- 可将此方法与更先进的 SSM 变体（如 Mamba 的选择性机制）结合
- GP-VAE 的持续学习实验（ERA5 气候数据）展示了在深度生成模型中的潜力

## 评分

- **新颖性**: ⭐⭐⭐⭐ HiPPO→GP 的跨领域迁移新颖，但本质上是已知方法的组合
- **实验充分度**: ⭐⭐⭐⭐ 涵盖时间序列、持续学习、GP-VAE 三类任务，基线完整
- **写作质量**: ⭐⭐⭐⭐ 数学推导清楚，图示直观，但跨域 GP 背景知识要求较高
- **价值**: ⭐⭐⭐⭐ 解决了在线 GP 的核心痛点（灾难性遗忘），具有较好的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Flow Matching Neural Processes](flow_matching_neural_processes.md)
- [\[ICML 2025\] Gaussian Mixture Flow Matching Models](../../ICML2025/image_generation/gaussian_mixture_flow_matching_models.md)
- [\[ICML 2025\] Quantum Algorithms for Finite-horizon Markov Decision Processes](../../ICML2025/image_generation/quantum_algorithms_for_finite-horizon_markov_decision_processes.md)
- [\[CVPR 2025\] Nonisotropic Gaussian Diffusion for Realistic 3D Human Motion Prediction](../../CVPR2025/image_generation/nonisotropic_gaussian_diffusion_for_realistic_3d_human_motion_prediction.md)
- [\[CVPR 2025\] Improving Editability in Image Generation with Layer-wise Memory](../../CVPR2025/image_generation/improving_editability_in_image_generation_with_layer-wise_memory.md)

</div>

<!-- RELATED:END -->
