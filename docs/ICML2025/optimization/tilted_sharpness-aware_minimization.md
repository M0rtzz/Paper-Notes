---
title: >-
  [论文解读] Tilted Sharpness-Aware Minimization
description: >-
  [ICML2025][优化][SAM] 提出 Tilted SAM (TSAM)，利用指数倾斜 (exponential tilting) 将 SAM 的 min-max 目标平滑化为对邻域内多个局部解按损失值加权的软优化，理论上更平滑、更偏好平坦极小值，实验在图像和文本任务上一致优于 SAM 及其变体。
tags:
  - ICML2025
  - 优化
  - SAM
  - 指数倾斜
  - 平坦极小值
  - 泛化
  - Hamiltonian Monte Carlo
---

# Tilted Sharpness-Aware Minimization

**会议**: ICML2025  
**arXiv**: [2410.22656](https://arxiv.org/abs/2410.22656)  
**代码**: [github.com/litian96/TSAM](https://github.com/litian96/TSAM)  
**领域**: 优化 / Sharpness-Aware Minimization  
**关键词**: SAM, 指数倾斜, 平坦极小值, 泛化, Hamiltonian Monte Carlo

## 一句话总结

提出 Tilted SAM (TSAM)，利用指数倾斜 (exponential tilting) 将 SAM 的 min-max 目标平滑化为对邻域内多个局部解按损失值加权的软优化，理论上更平滑、更偏好平坦极小值，实验在图像和文本任务上一致优于 SAM 及其变体。

## 研究背景与动机

SAM 通过优化邻域内最坏情况损失来寻找平坦极小值，提升过参数化模型的泛化性能。其标准 min-max 目标为：

$$\min_{\theta} L^s(\theta) := \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$$

然而 SAM 存在两个核心问题：

**优化困难**：在高度非凸的损失面上，单步梯度上升难以找到真正的最大损失扰动 $\epsilon$

**信息浪费**：仅关注最坏情况的单一局部解，忽略了邻域内许多同样导致较大损失的其他方向，使得局部极小值附近的损失面仍然可能是尖锐的

作者发现即使 SAM 找到的解在最坏方向上损失较低，其邻域的**平均损失**仍然高于 TSAM 的解，说明仅优化最坏情况是次优的。

## 方法详解

### 核心目标：Tilted SAM

TSAM 引入倾斜参数 $t \geq 0$，通过 LogSumExp 算子对邻域损失做指数加权聚合：

$$\min_{\theta} L^t(\theta) := \frac{1}{t} \log \left( \mathbb{E}_{\mu(\epsilon)} \left[ e^{t L(\theta + \epsilon)} \right] \right)$$

其中 $\mu(\epsilon)$ 表示扰动的概率分布（如均匀球 $\|\epsilon\| \leq \rho$）。TSAM 统一了三种优化范式：

| 参数 $t$ | 退化形式 | 含义 |
|:---:|:---:|:---|
| $t \to \infty$ | SAM | 仅关注最坏情况扰动 |
| $t = 0$ | 平均扰动损失 | 均匀加权邻域所有方向 |
| $0 < t < \infty$ | **TSAM** | 按损失值软加权，高损失方向权重更大 |

### 倾斜权重与梯度

定义 $t$-倾斜权重为：

$$w^t(\theta + \epsilon) := \frac{e^{t L(\theta + \epsilon)}}{\mathbb{E}[e^{t L(\theta + \epsilon)}]}$$

TSAM 的梯度即为用倾斜权重对邻域梯度做加权平均：

$$\nabla L^t(\theta) = \mathbb{E}\left[ w^t(\theta + \epsilon) \nabla L(\theta + \epsilon) \right]$$

这是 SAM （将全部权重分配给单一最坏方向）的**软化版本**。

### 理论性质

1. **平滑性** (Lemma 3.3)：TSAM 的平滑性参数 $\beta(t) = O(t)$，对有限 $t$ 有界；而 SAM ($t \to \infty$) 平滑性参数无界，因此 TSAM 更易优化
2. **偏好平坦性** (Theorem 3.6)：对 GLM 类损失，若 $\theta_1$ 比 $\theta_2$ 更尖锐，则 $L^t(\theta_1) - L^t(\theta_2)$ 关于 $t$ 单调递增——$t$ 越大，TSAM 越"惩罚"尖锐解
3. **小 $t$ 近似**：当 $t$ 较小时，$L^t(\theta) \approx \mathbb{E}[L(\theta+\epsilon)] + \frac{t}{2} \mathrm{var}(L(\theta+\epsilon))$，即同时最小化邻域的均值和方差
4. **泛化界** (Theorem 3.7)：存在有限的最优 $t^*$ 使泛化上界最紧

### 优化算法：基于 HMC 的采样

TSAM 的核心挑战是从分布 $p(\epsilon) \propto e^{\delta L(\theta + \epsilon)}$ 高效采样 $\epsilon$。作者采用基于 Hamiltonian Monte Carlo 的 Euler 离散化：

1. 随机初始化 $s$ 个扰动 $\epsilon_j$ 和动量 $p_j$
2. 对每个样本做一步梯度上升：$\epsilon \leftarrow \epsilon + \beta' \nabla L(\theta + \epsilon)$
3. 用倾斜权重聚合梯度更新模型：

$$\theta^{i+1} \leftarrow \theta^i - \eta \frac{\sum_{j} e^{(t-\delta) L(\theta^i + \epsilon_j)} \nabla L(\theta^i + \epsilon_j)}{\sum_{j} e^{(t-\delta) L(\theta^i + \epsilon_j)}}$$

实验中仅需 $s = 3 \sim 5$ 个采样即可获得显著提升。

## 实验关键数据

### 图像分类 (Test Accuracy %)

| 方法 | CIFAR100 (ResNet18) | CIFAR100 (WRN) | DTD (ViT) | Noisy CIFAR100 (ResNet18) | TinyImageNet (ResNet18) |
|:---|:---:|:---:|:---:|:---:|:---:|
| ERM | 71.39 | 73.22 | 66.38 | 61.01 | 71.10 |
| SAM | 76.52 | 78.44 | 67.87 | 69.00 | 72.43 |
| ESAM1 | 77.40 | 80.22 | 68.18 | 69.20 | 73.24 |
| RSAM | 77.35 | 79.02 | 68.35 | 69.31 | 73.57 |
| **TSAM** | **77.78** | **80.85** | **68.82** | **69.98** | **73.55** |

### GLUE 基准 (DistilBERT 微调)

| 方法 | CoLA | SST-2 | MNLI | QNLI | AVG |
|:---|:---:|:---:|:---:|:---:|:---:|
| ERM | 80.34 | 90.48 | 79.6 | 87.72 | 77.15 |
| SAM | 80.48 | 91.74 | 81.1 | 86.42 | 77.56 |
| **TSAM** | **80.81** | **91.86** | 81.1 | **87.81** | **78.01** |

### 平坦性验证 (CIFAR100, ResNet18 Top-5 Hessian 特征值)

| 方法 | Top-5 Eigenvalues |
|:---|:---|
| ERM | 342, 305, 261, 253, 211 |
| SAM | 233, 198, 183, 154, 146 |
| **TSAM** (t=20) | **141, 113, 106, 93, 90** |

TSAM 的 Hessian 最大特征值仅为 ERM 的 41%，远比 SAM 更平坦。

## 亮点与洞察

1. **统一框架**：TSAM 通过一个参数 $t$ 平滑插值于 min-avg 和 min-max 之间，提供了 SAM 家族的理论统一视角
2. **理论严谨**：证明了 TSAM 保持凸性/Lipschitz 性质、平滑性可控、偏好平坦解、存在最优 $t$
3. **采样高效**：仅需 3-5 个扰动样本即可有效逼近倾斜梯度，额外计算开销可接受
4. **广泛适用**：在 CNN、ViT、BERT 三类架构和图像/文本两类任务上一致优于 SAM 及其变体
5. **噪声鲁棒**：在标签噪声设置下提升尤为显著（+8.97% over ERM），表明 TSAM 对分布偏移有更强鲁棒性

## 局限与展望

1. **计算开销**：每步需 $s$ 次额外前向+反向传播，训练时间约为 SAM 的 $s$ 倍，对大规模模型可能不够友好
2. **超参数 $t$ 敏感**：需要通过验证集从 $\{0,1,5,20,100\}$ 中搜索最优 $t$，增加调参成本
3. **理论-实践差距**：平坦性偏好的理论仅对 GLM 类损失严格成立，对深度网络是经验性结论
4. **HMC 近似质量**：为效率妥协只做 1 步 Euler 离散化并直接接受样本，不保证采样一致性
5. **未结合自适应优化器**：实验中未与 Adam/AdaSAM 结合，留给了 future work

## 相关工作与启发

- **SAM 家族**：SAM (Foret 2020)、RSAM (Liu 2022)、PGN (Zhao 2022)、VASSO (Li & Giannakis 2023) 等
- **指数倾斜**：Tilted ERM (Li et al., 2023) 在数据空间做指数加权，TSAM 将其推广到参数空间
- **平均扰动损失**：与 average-perturbed sharpness (Wen 2022)、noise-perturbed loss (Zhang 2024) 相关，TSAM 是它们和 SAM 之间的插值

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将指数倾斜从数据空间推广到参数空间的思路简洁优雅
- 实验充分度: ⭐⭐⭐⭐ — CNN/ViT/BERT 全覆盖，含噪声/OOD 设置，对比了多个 SAM 变体
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导清晰，实验设计严谨，消融实验完整
- 价值: ⭐⭐⭐⭐ — 为 SAM 提供了更平滑的替代方案，理论洞察有启发意义

<!-- RELATED:START -->

## 相关论文

- [Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](../../ICLR2026/optimization/minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [Generalization and Robustness of the Tilted Empirical Risk](generalization_and_robustness_of_the_tilted_empirical_risk.md)
- [Unveiling m-Sharpness Through the Structure of Stochastic Gradient Noise](../../NeurIPS2025/optimization/unveiling_m-sharpness_through_the_structure_of_stochastic_gradient_noise.md)
- [Efficient Curvature-Aware Hypergradient Approximation for Bilevel Optimization](efficient_curvature-aware_hypergradient_approximation_for_bilevel_optimization.md)
- [Understanding Sharpness Dynamics in NN Training with a Minimalist Example: The Effects of Dataset Difficulty, Depth, Stochasticity, and More](understanding_sharpness_dynamics_in_nn_training_with_a_minimalist_example_the_ef.md)

<!-- RELATED:END -->
