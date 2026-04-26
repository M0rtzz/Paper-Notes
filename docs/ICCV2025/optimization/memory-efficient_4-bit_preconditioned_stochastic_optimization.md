---
title: >-
  [论文解读] Memory-Efficient 4-bit Preconditioned Stochastic Optimization
description: >-
  [ICCV 2025][优化][Shampoo] 提出基于 Cholesky 分解 + 误差反馈的 4-bit 量化方案，将 Shampoo 优化器的预条件矩阵压缩至 4-bit 精度，在大幅降低 GPU 显存的同时保持与 32-bit Shampoo 接近的训练性能，并给出了光滑与非光滑两种场景下的收敛性证明。
tags:
  - ICCV 2025
  - 优化
  - Shampoo
  - 量化
  - Cholesky 分解
  - 误差反馈
  - 低精度优化器
  - 内存高效训练
---

# Memory-Efficient 4-bit Preconditioned Stochastic Optimization

**会议**: ICCV 2025  
**arXiv**: [2412.10663](https://arxiv.org/abs/2412.10663)  
**代码**: 待确认  
**领域**: 优化 / 预条件随机优化  
**关键词**: Shampoo, 量化, Cholesky 分解, 误差反馈, 低精度优化器, 内存高效训练

## 一句话总结

提出基于 Cholesky 分解 + 误差反馈的 4-bit 量化方案，将 Shampoo 优化器的预条件矩阵压缩至 4-bit 精度，在大幅降低 GPU 显存的同时保持与 32-bit Shampoo 接近的训练性能，并给出了光滑与非光滑两种场景下的收敛性证明。

## 研究背景与动机

预条件随机优化算法（如 Shampoo）通过利用非对角预条件矩阵捕获参数间的相关性，在理论收敛速度和实际训练效果上均优于一阶优化器（SGD、Adam 等）。然而，Shampoo 需要额外存储四个预条件矩阵 $(L_k, R_k, L_k^{-1/4}, R_k^{-1/4})$，带来了巨大的 GPU 显存开销，严重限制了其在大规模模型训练中的可扩展性。

一种直观的解决方案是将预条件矩阵从 32-bit 量化为 4-bit，但直接量化会导致严重的信息损失和性能退化。例如在 ViT-Small/CIFAR-100 上，32-bit Shampoo 达到 77.95% 测试精度，而直接 4-bit 量化仅有 74.56%，差距高达 3.39%。因此，核心挑战在于：**如何在 4-bit 极低精度下有效压缩预条件矩阵，同时保留其光谱特性、避免优化质量下降？**

## 方法详解

### 整体框架

本文提出的 4-bit Shampoo 包含三个层次递进的技术：

1. **基线方案：直接量化（VQ）**——对预条件矩阵进行 block-wise 4-bit 量化
2. **核心创新一：Cholesky 量化（CQ）**——先做 Cholesky 分解再量化下三角因子
3. **核心创新二：补偿 Cholesky 量化（CQ+EF）**——在 Cholesky 量化基础上引入误差反馈

### 关键设计一：Cholesky 量化（CQ）

不直接量化对称正定的预条件矩阵 $L_k$ 和 $R_k$，而是先做 Cholesky 分解：

$$C_k^L = \text{Cholesky}(L_k + \epsilon I), \quad C_k^R = \text{Cholesky}(R_k + \epsilon I)$$

然后量化下三角 Cholesky 因子 $\bar{C}_k^L = \mathcal{Q}(C_k^L)$。恢复预条件矩阵时用 $L_k = \mathcal{D}(\bar{C}_k^L)\mathcal{D}(\bar{C}_k^L)^T$。

**两大优势：**

- **内存减半**：Cholesky 因子为下三角矩阵，存储量接近全矩阵的一半
- **光谱保持**：从量化后的 Cholesky 因子恢复的矩阵自动保持对称正定性（SPD），其逆 1/4 次根与原 32-bit 版本更接近。定量验证：在合成 PD 矩阵上，NRE 从 46.14（直接量化）降至 9.19（Cholesky 量化），AE 从 27.19 降至 9.20

**量化策略**：对角元素保留 32-bit 精度保数值稳定，非对角量化为 4-bit。采用 block-wise linear-2 映射。

### 关键设计二：误差反馈（EF）

受分布式训练中梯度压缩的误差反馈思想启发，将其适配至预条件矩阵的压缩：

- 维护一个 4-bit 误差状态 $\bar{E}_k^L$，记录 Cholesky 因子的量化误差
- 每次量化前先用上一步的误差补偿：$\bar{C}_k^L = \mathcal{Q}(C_k^L + E_{k-1}^L)$
- 误差状态通过指数移动平均更新：$E_k^L = \beta_e E_{k-1}^L + (1-\beta_e)(C_k^L + E_{k-1}^L - \mathcal{D}(\bar{C}_k^L))$

**关键存储技巧**：Cholesky 因子为下三角、误差状态也为严格下三角（对角为零），两者可分别存储在同一矩阵的下三角和上三角部分，不增加额外显存。

### 损失函数与更新规则

模型更新遵循标准 Shampoo 流程，使用量化后的逆 1/4 次根对梯度做预条件：

$$W_{k+1} = W_k - \eta_k \mathcal{D}(\hat{L}_k) G_k \mathcal{D}(\hat{R}_k)$$

学习率通过 grafting trick 缩放。支持 SGD 和 AdamW 作为 base optimizer。

### 收敛性理论

- **光滑非凸情形**：4-bit Shampoo 达到最优收敛率 $\mathcal{O}(1/\sqrt{T})$
- **非光滑非凸情形（如 ReLU 网络）**：证明全局收敛至稳定点集合，这是预条件梯度下降在非光滑情形下的**首个收敛性证明**

## 实验关键数据

### 主实验：CIFAR-100 测试精度 (%) / 峰值显存 (MB)

| 模型 | 优化器 | 精度 | 显存 |
|------|--------|------|------|
| VGG-19 | SGDM | 74.43 | 597.3 |
| VGG-19 | SGDM + 32-bit Shampoo | 75.02 | 1065.2 |
| VGG-19 | SGDM + 4-bit VQ | 74.36 | 662.2 |
| VGG-19 | SGDM + 4-bit CQ+EF | **75.21** | 662.2 |
| ViT-Small | AdamW | 73.00 | 2930.0 |
| ViT-Small | AdamW + 32-bit Shampoo | 77.95 | 3448.9 |
| ViT-Small | AdamW + 4-bit VQ | 74.56 | 3001.7 |
| ViT-Small | AdamW + 4-bit CQ+EF | **77.74** | 3001.7 |

### ImageNet 测试精度 (%) / 峰值显存 (MB)

| 模型 | 优化器 | 精度 | 显存 |
|------|--------|------|------|
| ResNet-50 | AdamW Base | 77.56 | 11356 |
| ResNet-50 | 32-bit Shampoo | 78.06 | 11986 |
| ResNet-50 | 4-bit CQ+EF | **78.00** | 11445 |
| ViT-Base | AdamW Base | 72.59 | 11840 |
| ViT-Base | 32-bit Shampoo | 75.47 | 13319 |
| ViT-Base | 4-bit CQ+EF | **75.01** | 12052 |

### LLM 预训练：LLaMA on C4 困惑度 (PPL，越低越好)

| 模型 | 优化器 | PPL | 显存 (GB) |
|------|--------|-----|-----------|
| LLaMA-130M | AdamW Base | 27.32 | 45.9 |
| LLaMA-130M | 32-bit Shampoo | 26.93 | 48.2 |
| LLaMA-130M | 4-bit CQ+EF | **26.98** | 46.2 |

### 消融实验：组件逐层叠加

| 方法 | ResNet-34/CIFAR-100 | ViT-Small/CIFAR-100 |
|------|---------------------|---------------------|
| 4-bit VQ | 79.45 | 74.56 |
| 4-bit CQ | 80.27 (+0.82) | 77.34 (+2.78) |
| 4-bit CQ+EF | **80.52** (+1.07) | **77.74** (+3.18) |

### 光谱保持度量（NRE / AE，越低越好）

| 方法 | 合成 NRE | 合成 AE | Epoch100 NRE | Epoch100 AE |
|------|----------|---------|-------------|-------------|
| 直接量化 VQ | 46.14 | 27.19 | 25.71 | 18.51 |
| Cholesky CQ | **9.19** | **9.20** | **4.85** | **4.85** |

### 关键发现

- CQ+EF 在 ViT-Small/CIFAR-100 上将 4-bit 精度损失从 3.39% 缩小至 0.21%
- ImageNet/ViT-Base 上，4-bit CQ+EF 仅比 32-bit 低 0.46%，显存节省 1267 MB
- Cholesky 分解计算开销可忽略：ResNet-50/ImageNet 时间 1899 min vs VQ 1883 min

## 亮点与洞察

1. **量化对象的创新选择**：不直接量化预条件矩阵，而是量化其 Cholesky 因子——首个此类尝试。利用三角结构和 SPD 恢复特性，一箭双雕解决内存和信息保持
2. **误差反馈与三角结构的巧妙结合**：误差状态可无额外开销存于同一矩阵的上三角部分
3. **首个非光滑收敛证明**：为预条件梯度方法在 ReLU 等非光滑激活下提供理论保证
4. **全面验证**：涵盖 VGG、ResNet、Swin、ViT 和 LLaMA 等多种架构

## 局限性

1. Cholesky 分解要求预条件矩阵严格正定，需依赖 $\epsilon I$ 正则化
2. 对角元素仍保留 32-bit，总体压缩比低于纯 4-bit
3. 仅在图像分类和 LLM 预训练上验证，未扩展至检测/分割等
4. 误差反馈引入额外超参 $\beta_e$，敏感性分析不够充分

## 相关工作

- **预条件优化器**：Shampoo、K-FAC、K-BFGS、AdaBK
- **优化器量化**：4-bit Adam 等，但仅针对一阶优化器状态
- **误差反馈**：源自分布式梯度压缩，本文首次用于预条件矩阵量化补偿

## 评分

| 维度 | 分数 (1-10) |
|------|-------------|
| 新颖性 | 8 |
| 理论深度 | 8 |
| 实验充分性 | 8 |
| 实用价值 | 8 |
| 写作质量 | 7 |
| **总评** | **8** |

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis](../../NeurIPS2025/optimization/nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](../../NeurIPS2025/optimization/efficient_adaptive_federated_optimization.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](../../NeurIPS2025/optimization/isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[ICCV 2025\] Class-Wise Federated Averaging for Efficient Personalization](class-wise_federated_averaging_for_efficient_personalization.md)
- [\[ICML 2025\] Efficient Curvature-Aware Hypergradient Approximation for Bilevel Optimization](../../ICML2025/optimization/efficient_curvature-aware_hypergradient_approximation_for_bilevel_optimization.md)

<!-- RELATED:END -->
