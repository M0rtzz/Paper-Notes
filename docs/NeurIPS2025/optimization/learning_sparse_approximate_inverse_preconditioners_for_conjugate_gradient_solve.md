---
title: >-
  [论文解读] Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs
description: >-
  [NeurIPS 2025][优化][预条件子] 提出一种基于图神经网络（GNN）的稀疏近似逆（SPAI）预条件子学习方法，利用 SPAI 的局部性与 GNN 消息传递的天然兼容性，并引入尺度不变损失函数（SAI loss），在 GPU 上实现 40%-53% 的求解时间缩减（68%-113% 加速）。
tags:
  - NeurIPS 2025
  - 优化
  - 预条件子
  - 共轭梯度法
  - 图神经网络
  - GPU加速
  - 稀疏近似逆
---

# Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs

**会议**: NeurIPS 2025  
**arXiv**: [2510.27517](https://arxiv.org/abs/2510.27517)  
**代码**: 无  
**领域**: Optimization  
**关键词**: 预条件子, 共轭梯度法, 图神经网络, GPU加速, 稀疏近似逆

## 一句话总结

提出一种基于图神经网络（GNN）的稀疏近似逆（SPAI）预条件子学习方法，利用 SPAI 的局部性与 GNN 消息传递的天然兼容性，并引入尺度不变损失函数（SAI loss），在 GPU 上实现 40%-53% 的求解时间缩减（68%-113% 加速）。

## 研究背景与动机

**领域现状**：求解对称正定（SPD）稀疏线性系统 $\mathbf{Ax} = \mathbf{b}$ 是科学计算的核心问题，共轭梯度法（CG）是主流迭代求解器，但收敛速度严重依赖预条件技术。

**现有痛点**：
   - 对角预条件子：GPU 友好但收敛改善有限
   - 不完全 Cholesky (IC)：CPU 性能好但需要三角求解，GPU 难以并行
   - 现有基于学习的方法（GNN→IC）：继承了三角求解的 GPU 并行性问题，且 GNN 难以建模沿消除树的长程依赖

**核心矛盾**：IC 类预条件子在迭代次数上表现好但 GPU 执行效率差（三角求解瓶颈），对角预条件子 GPU 效率高但迭代次数多。

**本文目标**：如何构建既能有效降低条件数又能充分利用 GPU 并行性的预条件子。

**切入角度**：从 SPAI（稀疏近似逆）入手，将 $\mathbf{M}^{-1} = \mathbf{GG}^\top + \varepsilon\mathbf{I}$ 直接作为预条件子，每步 CG 只需两次稀疏矩阵向量乘（SpMV），天然 GPU 友好。

**核心idea**：SPAI 的局部性（输出节点依赖两跳邻域）与 GNN 的局部消息传递天然相容，用 GNN 学习 SPAI 比学习 IC 更适合。

## 方法详解

### 整体框架

输入矩阵 $\mathbf{A}$ 的非零元素和节点特征 → GNN 编码器-处理器-解码器 → 输出稀疏矩阵 $\mathbf{G}$ → 组装 $\mathbf{M}^{-1} = \mathbf{GG}^\top + \varepsilon\mathbf{I}$ → 用于预条件 CG 求解。

### 关键设计

1. **SPAI 预条件子的局部性分析**：假设 $\mathbf{G}$ 与 $\mathbf{A}$ 共享稀疏模式，则预条件子的输出 $\mathbf{s}_j = (\mathbf{M}^{-1}\mathbf{r})_j$ 仅依赖节点 $j$ 的两跳邻域：
$$\mathbf{s}_j = \varepsilon\mathbf{r}_j + \sum_{l: \mathbf{A}_{jl}\neq 0} \mathbf{G}_{jl} \sum_{k: \mathbf{A}_{kl}\neq 0} \mathbf{G}_{kl}\mathbf{r}_k$$
这与 IC 沿消除树的全局依赖形成鲜明对比，使 SPAI 天然适配 GNN 的局部传播机制。

2. **GNN 架构**：采用编码器-处理器-解码器结构，包括：

    - 编码器：两个 MLP 分别编码节点特征 $\mathbf{v}_i$ 和边特征 $\mathbf{e}_{ij}$（即 $\mathbf{A}_{ij}$）
    - 处理器：$L=4$ 层消息传递，每层包含消息函数 $f_m^{(t)}$、节点更新 $f_v^{(t)}$、边更新 $f_e^{(t)}$，均使用残差连接
    - 解码器：MLP 将最终边特征 $\mathbf{h}_{ij}^{(L)}$ 映射为 $\mathbf{G}_{ij} \in \mathbb{R}^{b \times b}$

   总参数量仅约 24k。

3. **尺度不变对齐恒等损失函数（SAI Loss）**：

    - 传统 SPAI 损失 $\|\mathbf{AM}^{-1} - \mathbf{I}\|_F^2$ 的两个问题：(1) 直接计算 $\mathbf{AM}^{-1}$ 对大矩阵代价过高；(2) 依赖 $\mathbf{A}$ 的绝对量级，而 CG 收敛率是尺度不变的
    - **随机估计**：用随机向量 $\mathbf{w} \sim \mathcal{N}(0,1)$ 近似 Frobenius 范数，将矩阵乘法简化为矩阵-向量乘法
    - **尺度归一化**：
$$\mathcal{L}_{\text{SAI}}(\mathbf{A}, \mathbf{M}^{-1}, \mathbf{w}) = \left\|\left(\frac{1}{\|\mathbf{A}\|}\mathbf{AM}^{-1} - \mathbf{I}\right)\mathbf{w}\right\|_2^2$$
   满足 $\mathcal{L}_{\text{SAI}}(\mathbf{A}, \mathbf{M}^{-1}) = \mathcal{L}_{\text{SAI}}(\alpha\mathbf{A}, \mathbf{M}^{-1})$ 对任意 $\alpha > 0$ 成立
    - 范数定义为 $\|\mathbf{A}\| = \text{mean}_{\mathbf{A}_{ij}\neq 0}|\mathbf{A}_{ij}|$，对矩阵维度和异常值更鲁棒
    - 理论推导：该损失直接约束条件数 $\kappa(\mathbf{AM}^{-1}) \leq 1 + 2\|\mathbf{E}\|_2$

### 训练策略

- 所有实验固定 $L=4$、隐藏维度 $d=24$、$\varepsilon = 10^{-4}$
- 500 epochs、batch size 4、AdamW 优化器、指数衰减学习率（衰减率 0.99）
- 单 NVIDIA A100 GPU 训练

## 实验关键数据

### 主实验——GPU 求解时间（$\text{rtol} = 10^{-8}$）

| 测试问题 | Diag | IC | AINV | **Ours** | 相对加速 |
|---------|------|-----|------|----------|---------|
| Heat Equation | 77ms (520) | 167ms (204) | 78ms (330) | **36ms (197)** | **113%** |
| Poisson Equation | 45ms (320) | 101ms (128) | 58ms (217) | **26ms (128)** | **73%** |
| Hyperelasticity | 86ms (464) | 202ms (117) | 247ms (266) | **51ms (175)** | **68%** |
| Synthetic System | 445ms (2775) | 1399ms (1808) | 5024ms (10896) | **253ms (1122)** | **75%** |

括号内为 CG 迭代次数。本方法在所有数据集上均取得最快求解时间。

### 消融实验——不同收敛阈值下的性能（Heat 问题，GPU）

| rtol | Diag | IC | AINV | **Ours** |
|------|------|-----|------|----------|
| $10^{-2}$ | 43ms (309) | 92ms (115) | 52ms (199) | **22ms (124)** |
| $10^{-4}$ | 53ms (384) | 114ms (143) | 61ms (246) | **27ms (154)** |
| $10^{-6}$ | 63ms (442) | 132ms (164) | 67ms (280) | **32ms (176)** |
| $10^{-8}$ | 77ms (520) | 167ms (204) | 78ms (330) | **36ms (197)** |

### 关键发现

- 在 GPU 上，本方法实现 68%-113% 的速度提升
- 迭代次数与 IC 相当（说明预条件质量好），但每步应用代价远低于 IC 的三角求解
- 构建时间（0.181ms）与对角预条件子（0.196ms）在同一量级，远低于 IC（1.866ms）和 AINV（18.924ms）
- SAI 损失相比非尺度不变的传统损失带来明显的条件数改善

## 亮点与洞察

- **SPAI-GNN 的自然兼容性**：SPAI 的两跳局部性与 GNN 的消息传递完美匹配，是比 IC-GNN 更合理的组合
- **SAI 损失的设计精妙**：尺度不变性对齐了预条件优化目标与 CG 收敛特性，且不需要预计算 $\mathbf{A}^{-1}$
- **极轻量模型**：GNN 仅 24k 参数，说明局部结构信息足以指导高质量预条件子的构建
- **全 GPU 流水线**：从构建（GNN 前向）到应用（SpMV），整个流程均在 GPU 上执行

## 局限与展望

- 仅针对 SPD 矩阵，不适用于非对称或不定系统
- 需要为不同类型的 PDE 问题分别训练 GNN
- 稀疏模式固定为与 $\mathbf{A}$ 相同，未探索自适应稀疏模式选择
- 与 AMG 等多重网格方法的对比有限
- 在极大规模问题（百万级自由度）上的可扩展性有待验证

## 相关工作与启发

- **Li et al., Häusner et al. (2023-2024)**：GNN→IC 预条件子的先驱工作，本文指出其三角求解与 GNN 局部性之间的根本矛盾
- **传统 SPAI (Kolotilina et al.)**：本文用数据驱动方法解决了传统 SPAI 构建依赖顺序算法、性能次优的问题
- 启发：科学计算中"GPU 友好性"可能比"理论最优性"更重要——IC 理论上更优但 GPU 上反而更慢

## 评分

- 新颖性: ⭐⭐⭐⭐ GNN+SPAI的组合新颖，SAI损失设计有理论深度
- 实验充分度: ⭐⭐⭐⭐ 多种PDE问题+合成数据集，与传统和学习方法全面对比
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，局部性分析的对比论证尤为精彩
- 价值: ⭐⭐⭐⭐ 在GPU科学计算领域实用价值高，68%-113%的加速非常显著

<!-- RELATED:START -->

## 相关论文

- [Natural Gradient VI: Guarantees for Non-Conjugate Models](natural_gradient_vi_guarantees_for_non-conjugate_models.md)
- [Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry](projecting_assumptions_the_duality_between_sparse_autoencoders_and_concept_geome.md)
- [Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning](fedqs_optimizing_gradient_and_model_aggregation_for_semi-asynchronous_federated_.md)

<!-- RELATED:END -->
