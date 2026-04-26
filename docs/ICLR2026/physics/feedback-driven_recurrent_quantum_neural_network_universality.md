---
title: >-
  [论文解读] Feedback-driven Recurrent Quantum Neural Network Universality
description: >-
  [ICLR2026][quantum reservoir computing] 本文首次为基于反馈的循环量子神经网络 (RQNN) 建立了定量逼近误差界和普适性证明，表明 RQNN 可在 qubit 数仅以 $\lceil\log_2(\varepsilon^{-1})\rceil$ 对数增长的条件下，以线性读出层逼近任意 fading memory 滤波器，且不受维度灾难影响。
tags:
  - ICLR2026
  - quantum reservoir computing
  - recurrent quantum neural network
  - universal approximation
  - fading memory filter
  - NISQ
---

# Feedback-driven Recurrent Quantum Neural Network Universality

**会议**: ICLR2026  
**arXiv**: [2506.16332](https://arxiv.org/abs/2506.16332)  
**代码**: 无  
**领域**: physics  
**关键词**: quantum reservoir computing, recurrent quantum neural network, universal approximation, fading memory filter, NISQ  

## 一句话总结

本文首次为基于反馈的循环量子神经网络 (RQNN) 建立了定量逼近误差界和普适性证明，表明 RQNN 可在 qubit 数仅以 $\lceil\log_2(\varepsilon^{-1})\rceil$ 对数增长的条件下，以线性读出层逼近任意 fading memory 滤波器，且不受维度灾难影响。

## 背景与动机

- **量子储层计算 (QRC)** 利用量子系统动力学处理时序数据，尤其适合 NISQ 设备；已有大量实证成功，但理论基础薄弱
- 经典循环神经网络 (RNN) 的万能逼近定理已有深入研究 (Hornik 1991, Barron 1993, Grigoryeva & Ortega 2018)，但 **量子 RNN 缺乏定量逼近界**
- 此前 QRC 的万能性证明依赖 **多项式读出层**（利用 Stone-Weierstrass 定理），但实际系统普遍使用 **线性读出层**，因其训练简单快速
- 反馈协议 (feedback protocol) 通过将输出状态反馈到下一时刻输入，使系统能用更少组件保留输入历史，支持实时计算，但其逼近能力此前缺乏理论保障

## 核心问题

1. 反馈驱动的 RQNN 能否以 **可控的量子资源**（qubit 数、电路大小）逼近一般性状态空间系统？
2. RQNN 族是否对 **任意因果、时不变、fading memory 滤波器** 具有万能逼近性质？
3. 能否在仅使用 **线性读出层** 的条件下保持万能性？

## 方法详解

### RQNN 架构

网络由 $N$ 个并行量子电路组成，每个电路对应状态向量的一个分量：

- **量子门 $\mathtt{U}$**：由 $n$ 个参数化旋转块 $\bar{\mathtt{U}}^{(i)}$ 以块对角形式构成的 uniformly controlled quantum gate，每块是输入编码门 $\mathtt{U}_1^{(i)}$（依赖状态 $\bm{x}$ 和输入 $\bm{z}$）与偏置门 $\mathtt{U}_2^{(i)}$ 的张量积
- **初始化门 $\mathtt{V}$**：将 $|0\rangle^{\otimes \mathfrak{n}}$ 映射到均匀叠加态 $|\psi\rangle = \frac{1}{\sqrt{n}}\sum_{i=0}^{n-1}|i\rangle \otimes |00\rangle$
- **测量**：计算目标 qubit 在特定状态下的概率 $\mathbb{P}_m^{n,\bm{\theta}}$
- **状态映射**：$\bar{F}_{R,j}^{n,\bm{\theta}}(\bm{x},\bm{z}) = R - 2R[\mathbb{P}_1^{n,\bm{\theta}^j} + \mathbb{P}_2^{n,\bm{\theta}^j}]$，等价于 $\frac{1}{n}\sum_{i=1}^n R\cos(\gamma^{i,j})\cos(b^{i,j} + \bm{a}^{i,j}\cdot(\bm{x},\bm{z}))$

整个系统通过反馈 $\hat{\bm{x}}_t = \bar{F}_R^{n,\bm{\theta}}(\hat{\bm{x}}_{t-1}, \bm{z}_t)$ 形成循环结构。

### 量子资源分析

- 电路作用于 $\mathfrak{n} = \lceil\log_2(2n)\rceil$ 个 qubit
- $n$ 为精度参数（决定块数），qubit 数仅对数增长
- 达到逼近精度 $\varepsilon$ 需要 $\mathcal{O}(\varepsilon^{-2})$ 个权重和 $\mathcal{O}(\lceil\log_2(\varepsilon^{-1})\rceil)$ 个 qubit

### 主要理论结果

**定理 4.6（状态空间系统逼近界）**：对满足 Barron 型可积条件且收缩系数 $\lambda < 1$ 的状态映射 $F$，RQNN 滤波器均匀逼近误差满足：

$$\sup_{\bm{z}}\sup_t \|U^F(\bm{z})_t - \bar{U}(\bm{z})_t\| \leq \frac{1}{1-\lambda}\frac{\sqrt{N}\max_j C_j^\infty}{\sqrt{n}}$$

- 误差以 $1/\sqrt{n}$ 衰减，**与输入维度 $d$ 和状态维度 $N$ 无关**（无维度灾难）
- 关键技术：QNN 同时逼近函数及其导数（Proposition 4.4），使反馈回路的误差传播可控

**定理 4.8（万能逼近）**：对 **任意因果、时不变、fading memory 滤波器** $U$，存在 RQNN 参数和线性读出 $W$ 使得：

$$\sup_{\bm{z}}\sup_t \|U(\bm{z})_t - \bar{U}_W(\bm{z})_t\| \leq \varepsilon$$

- 此处无需 Barron 可积条件，也无需收缩性假设
- 通过引入线性预处理矩阵 $P_j$ 和有限步记忆分区确保 echo state property

### 证明策略

采用 **internal approximation approach**：先建立 QNN 对静态函数及其导数的逼近界 → 利用导数控制反馈回路中的误差累积 → 从状态映射逼近推导滤波器逼近

## 实验关键数据

本文为纯理论工作，无数值实验。核心定量结果为逼近误差界中的常数与收敛速率。

## 亮点

- **首个 RQNN 定量逼近界**：填补了量子 RNN 逼近理论的空白
- **无维度灾难**：误差率 $1/\sqrt{n}$ 与输入/状态维度无关，qubit 数仅对数增长
- **线性读出即万能**：不需要多项式读出层，大幅降低实验实现难度
- **比经典 RNN 条件更弱**：所需 Sobolev 光滑性条件 $s > \frac{N+d}{2}+4$，弱于经典 RNN 所需的 $s > N+d+3$
- **与硬件兼容**：基于 uniformly controlled gates 的电路已有高效分解方案和 Rydberg 原子实现

## 局限与展望

- **仅限理论分析**：缺少数值验证，无法评估实际 NISQ 设备上的表现
- **Barron 型条件限制**：定量界仅适用于傅里叶变换充分可积的函数，对粗糙或非收缩动力学尚无误差率
- **Barren plateau 问题未讨论**：变分量子电路训练的梯度消失问题可能影响实际可训练性
- **Monte Carlo 采样误差**：量子测量的有限采样误差仅在附录中简要讨论，未纳入主逼近界
- **未与随机储层比较**：结论仅适用于全参数可训练的变分设置，对参数部分随机的真正 QRC 尚未推广

## 与相关工作的对比

| 方法 | 线性读出万能性 | 定量误差界 | 无维度灾难 | 反馈/时序 |
|------|:---:|:---:|:---:|:---:|
| 经典 ESN (Grigoryeva & Ortega 2018) | ✗ | ✗ | - | ✓ |
| 经典 RNN (Gonon et al. 2023) | ✓ | ✓ | ✓ | ✓ |
| 前馈 QNN (Gonon & Jacquier 2025) | - | ✓ | ✓ | ✗ |
| QRC 多项式读出 (Sannia et al. 2024) | ✗ | ✗ | - | ✓ |
| **本文 RQNN** | **✓** | **✓** | **✓** | **✓** |

相比经典 RNN，RQNN 对目标函数的光滑性要求更低；相比前馈 QNN，本文处理了反馈回路带来的额外分析难度；相比已有 QRC 万能性结果，本文首次实现线性读出 + 定量界。

## 启发与关联

- 为量子储层计算提供了坚实的理论基础，未来可结合泛化误差界 (generalization bound) 建立完整的学习理论
- RQNN 的余弦基展开形式 (Proposition 4.1) 暗示了与经典随机特征方法的深层联系
- 线性预处理 + 有限记忆分区保证 echo state property 的技巧可能对设计实用 QRC 架构有指导意义
- Barren plateau 与表达能力之间的权衡是落地的关键瓶颈，值得后续深入研究

## 评分

- 新颖性: ⭐⭐⭐⭐☆ — 首次为反馈驱动 RQNN 建立完整逼近理论
- 实验充分度: ⭐⭐☆☆☆ — 纯理论工作，无实验验证
- 写作质量: ⭐⭐⭐⭐☆ — 结构清晰，主要结果陈述精确
- 价值: ⭐⭐⭐⭐☆ — 为量子机器学习时序任务奠定理论基础

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Compact Matrix Quantum Group Equivariant Neural Networks](../../ICML2025/physics/compact_matrix_quantum_group_equivariant_neural_networks.md)
- [\[ICLR 2026\] Sublinear Time Quantum Algorithm for Attention Approximation](sublinear_time_quantum_algorithm_for_attention_approximation.md)
- [\[CVPR 2025\] ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks](../../CVPR2025/physics/atp_adaptive_threshold_pruning_for_efficient_data_encoding_in_quantum_neural_net.md)
- [\[ICCV 2025\] ResQ: A Novel Framework to Implement Residual Neural Networks on Analog Rydberg Atom Quantum Computers](../../ICCV2025/physics/resq_a_novel_framework_to_implement_residual_neural_networks_on_analog_rydberg_a.md)
- [\[NeurIPS 2025\] Neural Deprojection of Galaxy Stellar Mass Profiles](../../NeurIPS2025/physics/neural_deprojection_of_galaxy_stellar_mass_profiles.md)

<!-- RELATED:END -->
