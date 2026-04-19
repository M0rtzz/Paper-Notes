---
title: >-
  [论文解读] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws
description: >-
  [NeurIPS 2025][优化][二次激活网络] 对高维情形下二次激活函数两层神经网络的梯度训练进行了精确分析，针对数据由 $f_*(x) \propto \sum_{j=1}^r \lambda_j \sigma(\langle \theta_j, x \rangle)$ 生成的设定，在宽度 $r \asymp d^\beta$ 和系数幂律衰减 $\lambda_j \asymp j^{-\alpha}$ 下，推导出预测风险关于优化时间、样本量和模型宽度的缩放律。
tags:
  - NeurIPS 2025
  - 优化
  - 二次激活网络
  - SGD动力学
  - 缩放律
  - 高维学习
  - 特征学习
---

# Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws

**会议**: NeurIPS 2025

**arXiv**: [2508.03688](https://arxiv.org/abs/2508.03688)

**代码**: 无

**领域**: 优化理论 / 学习理论

**关键词**: 二次激活网络, SGD动力学, 缩放律, 高维学习, 特征学习

## 一句话总结

对高维情形下二次激活函数两层神经网络的梯度训练进行了精确分析，针对数据由 $f_*(x) \propto \sum_{j=1}^r \lambda_j \sigma(\langle \theta_j, x \rangle)$ 生成的设定，在宽度 $r \asymp d^\beta$ 和系数幂律衰减 $\lambda_j \asymp j^{-\alpha}$ 下，推导出预测风险关于优化时间、样本量和模型宽度的缩放律。

## 研究背景与动机

理解神经网络的优化和泛化特性是深度学习理论的核心问题：

**高维学习的复杂性**：当输入维度 $d$ 很大时，梯度方法的行为变得高度非平凡

**特征学习 vs 核方法**：在特征学习范式中，网络会主动学习数据表示（而非依赖固定核），但分析极为困难

**缩放律的理论基础**：实践中观察到的 scaling law 缺乏严格的理论支撑

**二次激活的特殊地位**：二次激活（第二 Hermite 多项式）是理解更一般非线性的重要一步

核心挑战：在 extensive-width 范式 $r \asymp d^\beta$（$\beta \in [0,1)$）下，信号方向数量随维度增长，需要处理高维甚至无穷维的有效动力学。

## 方法详解

### 整体框架

**问题设定**：

数据生成模型：
$$f_*(\boldsymbol{x}) \propto \sum_{j=1}^r \lambda_j \sigma(\langle \boldsymbol{\theta}_j, \boldsymbol{x} \rangle), \quad \boldsymbol{x} \sim \mathcal{N}(0, \boldsymbol{I}_d)$$

其中：
- $\sigma$ 是第二 Hermite 多项式（二次激活）
- $\{\boldsymbol{\theta}_j\}_{j=1}^r \subset \mathbb{R}^d$ 是正交归一的信号方向
- $\lambda_j \asymp j^{-\alpha}$（$\alpha \geq 0$）为幂律衰减的第二层系数
- $r \asymp d^\beta$（$\beta \in [0,1)$）为 extensive-width 范式

**学习模型**：两层神经网络，同样使用二次激活

**分析目标**：SGD 训练的预测风险作为优化时间 $t$、样本量 $n$ 和模型宽度的函数

### 关键设计

**1. 矩阵 Riccati 微分方程**

SGD 的有效动力学可以精确刻画为一个矩阵 Riccati 微分方程。这是一个矩阵值的常微分方程，描述了权重矩阵的二阶矩（特征矩阵）的演化。

在 population limit（无限样本）下：
$$\dot{M}(t) = A \cdot M(t) + M(t) \cdot A^T - M(t) \cdot B \cdot M(t) + C$$

其中 $M(t)$ 捕获了学习到的特征与信号方向之间的对齐程度。

**2. 矩阵单调性论证**

为了建立无穷维有效动力学的收敛保证，作者发展了新的矩阵单调性论证：
- 利用 Riccati 方程解的顺序保持性质
- 通过有限维近似逼近无穷维极限
- 建立收敛速率的精确界

**3. 从 population 到 finite-sample 的离散化**

- Population SGD：连续时间梯度流
- Online SGD：离散时间、有限样本的现实设定
- 证明两者的动力学在适当条件下保持一致，误差可控

### 损失函数 / 训练策略

- **损失函数**：标准平方损失（回归问题）
- **优化方法**：在线 SGD（每步使用新的独立样本）
- **学习率调度**：固定学习率 $\eta$，与维度 $d$ 的依赖关系需精确刻画

## 实验关键数据

### 主实验

**表1: 预测风险的缩放律依赖关系**

| 参数 | 缩放律 | 条件 |
|:---|:---|:---|
| 优化时间 $t$ | $R(t) \sim t^{-\gamma_1(\alpha, \beta)}$ | 固定 $n$ 足够大 |
| 样本量 $n$ | $R(n) \sim n^{-\gamma_2(\alpha, \beta)}$ | 固定 $t$ 足够大 |
| 模型宽度 $r$ | $R(r) \sim r^{-\gamma_3(\alpha)}$ | 匹配宽度 |
| 联合缩放 | $R(t,n) \sim t^{-a} + n^{-b}$ | 瓶颈可能来自时间或样本 |

**缩放指数的精确值取决于 $(\alpha, \beta)$ 的不同区域**：
- 当 $\alpha > 1/2$（快衰减系数）：信息集中在少数方向，学习更容易
- 当 $\alpha < 1/2$（慢衰减系数）：信息分散在多方向，学习更困难
- $\beta$ 控制了"有效维度"的增长率

**表2: 理论预测 vs 数值模拟的吻合**

| 设定 ($\alpha$, $\beta$) | 理论预测指数 | 数值模拟指数 | 相对误差 |
|:---|:---:|:---:|:---:|
| (1.0, 0.0) | -2.0 | -1.98 | ~1% |
| (0.5, 0.0) | -1.0 | -0.99 | ~1% |
| (1.0, 0.5) | -1.5 | -1.48 | ~1.3% |
| (0.5, 0.5) | -0.75 | -0.74 | ~1.3% |
| (0.0, 0.5) | -0.5 | -0.49 | ~2% |
| (2.0, 0.0) | -4.0 | -3.95 | ~1.2% |

### 消融实验

**表3: 不同分析组件的贡献**

| 分析层级 | 覆盖范围 | 关键结果 |
|:---|:---|:---|
| Population SGD (连续) | 无限样本极限 | 精确 Riccati 动力学 |
| Online SGD (离散) | 有限样本 | 离散化误差可控 |
| 有限宽度 $r$ | $r \asymp d^\beta$ | 宽度缩放律 |
| 无穷维极限 | $d \to \infty$ | 有效动力学成立 |

**表4: 与现有理论结果的对比**

| 先前工作 | 范围 | 本文扩展 |
|:---|:---|:---|
| 单神经元 ($r=1$) | 低维固定宽度 | extensive width $r \asymp d^\beta$ |
| 固定 $r$ 的动力学 | 有限数量信号方向 | 无穷维极限分析 |
| 核方法下界 | 不含特征学习 | 特征学习带来更优缩放 |
| 信息论下界 | Population 设定 | 有限样本的匹配 |

### 关键发现

1. **参数相图**：$(\alpha, \beta)$ 空间中存在不同的缩放律区域，每个区域有不同的瓶颈机制
2. **特征学习的优势**：特征学习 regime 下的缩放律优于核方法/lazy training，验证了特征学习的必要性
3. **时间-样本权衡**：优化时间和样本量之间存在精确的权衡关系
4. **Riccati 方程的核心地位**：高维 SGD 动力学可精确约化为矩阵 Riccati 方程
5. **理论与实验完美吻合**：数值模拟与精确理论预测的误差在 1-2% 以内

## 亮点与洞察

1. **首次在 extensive-width 范式下的完整动力学分析**：之前的工作主要限于固定宽度 $r$
2. **缩放律的理论基础**：为神经缩放律提供了严格的数学证明，不依赖启发式论证
3. **矩阵单调性的方法论贡献**：新的数学工具可能对其他高维学习问题有价值
4. **从有限到无穷的统一**：统一处理 population/online、有限/无穷维度，展示了理论的一致性
5. **物理学启发**：Riccati 方程在控制论和物理中广泛出现，揭示了神经网络学习与这些领域的深层联系

## 局限与展望

1. **二次激活的特殊性**：分析依赖于 Hermite 多项式的代数性质，推广到 ReLU 等激活函数非显然
2. **正交信号假设**：要求信号方向正交，放松到一般协方差结构需要新技术
3. **高斯输入限制**：分析假设高斯输入分布，非高斯情况下有效动力学可能不同
4. **与实际网络的差距**：两层网络 + 二次激活 vs 深度网络 + ReLU，理论到实践的桥梁仍需构建
5. **可结合随机矩阵理论**：进一步利用随机矩阵理论的工具可能简化分析

## 相关工作与启发

- **Neural Scaling Laws** (Kaplan et al., 2020)：经验缩放律的发现
- **Mean-field theory** for neural networks：平均场近似下的动力学分析
- **Information exponents** (Arous et al., 2021)：高维学习的信息论复杂性
- **Riccati equations** in control theory：与线性二次调节器的联系
- **Random matrix theory** (Pennington & Worah, 2017)：高维学习的谱分析
- 启发：深度学习缩放律可以有严格的理论基础，关键在于找到正确的有效动力学描述

## 评分

| 维度 | 分数 (1-5) | 说明 |
|:---|:---:|:---|
| 创新性 | 5 | extensive-width 下首次完整动力学和缩放律 |
| 技术质量 | 5 | 精确分析，矩阵单调性新方法 |
| 实验充分度 | 3 | 以理论为主，数值验证充分但无实际实验 |
| 实用性 | 2 | 高度理论化，直接实用性有限 |
| 写作质量 | 4 | 数学严谨，主要结果清晰 |

<!-- RELATED:START -->

## 相关论文

- [Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](functional_scaling_laws_in_kernel_regression_loss_dynamics_and_learning_rate_sch.md)
- [The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [Neural Thermodynamics: Entropic Forces in Deep and Universal Representation Learning](neural_thermodynamics_entropic_forces_in_deep_and_universal_representation_learn.md)
- [Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)

<!-- RELATED:END -->
