---
title: >-
  [论文解读] Function Spaces Without Kernels: Learning Compact Hilbert Space Representations
description: >-
  [ICLR 2026][人体理解][函数编码器] 证明函数编码器（Function Encoders）通过学习神经网络基函数定义了一个有效的核，建立了神经特征学习与RKHS理论的桥梁，并提出PCA引导的紧凑基选择算法和有限样本泛化界。
tags:
  - ICLR 2026
  - 人体理解
  - 函数编码器
  - 核方法
  - Hilbert空间
  - 泛化界
  - PCA基选择
---

# Function Spaces Without Kernels: Learning Compact Hilbert Space Representations

**会议**: ICLR 2026  
**arXiv**: [2509.20605](https://arxiv.org/abs/2509.20605)  
**代码**: [有](https://github.com/suann124/function_encoder_kernels)  
**领域**: 人体理解  
**关键词**: 函数编码器, 核方法, Hilbert空间, 泛化界, PCA基选择

## 一句话总结

证明函数编码器（Function Encoders）通过学习神经网络基函数定义了一个有效的核，建立了神经特征学习与RKHS理论的桥梁，并提出PCA引导的紧凑基选择算法和有限样本泛化界。

## 研究背景与动机

机器学习方法长期面临**计算效率与理论保证之间的权衡**：

- **神经网络**：灵活、可扩展，但理论保证有限（现有的界往往空洞或依赖限制性假设）
- **核方法**：理论完善（RKHS理论、精确的统计保证），但可扩展性差——推理成本与训练集大小 m 成正比（对偶表示中需要计算 m×m 的Gram矩阵）

函数编码器（Function Encoders）是一种近期提出的迁移/表示学习技术：学习一组神经网络基函数 $\{\psi_j\}_{j=1}^n$ 作为显式特征映射 $\phi(x) = [\psi_1(x), \ldots, \psi_n(x)]^\top$，推理成本仅依赖于基函数个数 n 而非数据集大小 m。但其理论基础尚不完善：

1. 缺乏与Hilbert空间技术的形式化联系
2. 没有选择基函数个数 n 的原则性方法
3. 缺乏有限样本的泛化保证

本文旨在填补这三个理论缺口。

## 方法详解

### 整体框架

核心观点：函数编码器可以从原始空间（primal）和对偶空间（dual）两个视角理解：

- **原始空间**：提供显式特征映射 $\phi(x)$，支持 $\mathcal{O}(n)$ 的高效线性预测
- **对偶空间**：内积 $\langle\phi(x), \phi(x')\rangle$ 对应核评估，可用核方法的理论工具分析

函数编码器分两阶段工作：
1. **离线阶段**：在函数集合 $\{D_1, \ldots, D_N\}$ 上联合训练基函数，最小化多任务重建损失
2. **在线推理**：固定基函数，通过正则化最小二乘求解系数 c

### 关键设计

**1. 函数编码器即可学习核**

**核心定理（Proposition 1）**：基函数定义的内积自动构成有效核：

$$k(x, x') = \langle\phi(x), \phi(x')\rangle = \sum_{j=1}^n \psi_j(x)\psi_j(x')$$

该核是对称正半定的（因为 $\|\sum_i \alpha_i \phi(x_i)\|^2 \geq 0$），因此是合法的再生核。

这一联系意味着：函数编码器不仅在原始空间提供高效预测，在对偶空间也实例化了有效的核——可以用核方法的全套理论工具来分析。

**2. 渐进式训练（Progressive Training）**

类比PCA逐步添加主成分的过程，逐个训练基函数：

- 步骤 b=1：训练第一个基 $\psi_1$ 后冻结
- 步骤 b=2：添加 $\psi_2$，仅训练 $\psi_2$ 拟合残差
- 每步 b：计算系数协方差矩阵 $\Sigma_b$，取特征值 $\lambda_1 \geq \ldots \geq \lambda_b$
- 停止条件：累积解释方差 $\text{CEV}_r = \sum_{i=1}^r \lambda_i / \sum_{j=1}^b \lambda_j \geq \tau$（如99%）

优势：有序、可解释的基函数，清晰的停止规则。
缺点：训练本质上是串行的，无法GPU并行。

**3. 训练后剪枝（Train-Then-Prune）**

- 用 B 个基函数（B≫r）联合训练
- 计算系数协方差矩阵的特征分解
- 有效秩 $r = \min\{n : \sum_{i=1}^n \lambda_i / \sum_{j=1}^B \lambda_j \geq \tau\}$
- 评分每个基的重要性：$s_p = \sum_{i=1}^r \lambda_i U_{pi}^2$
- 保留 top-r 个基，剪枝其余部分
- 短期微调恢复性能

优势：可并行训练，计算效率高。

**4. 泛化界（理论核心）**

**Rademacher复杂度界**：

$$L(f_{\hat{c}_\lambda}) \lesssim \hat{L}_m(f_{\hat{c}_\lambda}) + \tilde{\mathcal{O}}\left(Y^2 R^2 \frac{n}{\lambda\sqrt{m}}\right)$$

关键scaling：复杂度随基函数数 n 增长，随数据量 m 和正则化 λ 衰减。更多基提升表达力但增加过拟合风险。

**PAC-Bayes界**：

$$L(f_{\hat{c}_\lambda}) \lesssim \hat{L}_m(f_{\hat{c}_\lambda}) + \tilde{\mathcal{O}}\left(Y^2 R^2 \frac{n^{3/2}}{\lambda\sqrt{m}}\right)$$

技术贡献：使用截断高斯分布处理无界损失函数，扩展了PAC-Bayes分析到多变量输出和学习特征映射的场景。

### 损失函数 / 训练策略

- **离线训练损失**：多任务正则化最小二乘
  $$\frac{1}{Nm}\sum_{j=1}^N \sum_{i=1}^m \|f_j(x_i) - \hat{f}_j(x_i)\|^2 + \lambda\|\hat{f}_j\|^2$$
- **在线推理**：固定基后求解系数的正则化最小二乘

## 实验关键数据

### 主实验

**多项式空间基准**（已知内在维度 d+1）：

| 多项式阶数 | 内在维度 | Progressive恢复 | Train-Prune恢复 |
|-----------|---------|----------------|----------------|
| d=3 | 4 | 4 ✓ | 4 ✓ |
| d=4 | 5 | 5 ✓ | 5 ✓ |
| d=5 | 6 | 6 ✓ | 6 ✓ |

两种算法均精确恢复已知的内在维度，剪枝后模型精度与过参数化原始模型一致。

**Van der Pol振荡器**：

| 方法 | 基函数数量 | 预测精度 |
|------|-----------|---------|
| 先前工作 (Ingebrand et al.) | 100 | 基准精度 |
| Progressive/Prune | **2** | **相同精度** |

从100个基降至2个，减少**50倍**，精度不变。

**二体轨道问题**：

| 方法 | 基函数数量 | 解释方差>99% |
|------|-----------|-------------|
| 过参数化 | 大量 | — |
| Progressive/Prune | **5-6** | ✓ |

5-6个基捕获>99%方差，与轨道参数的5维空间一致。

### 消融实验

**与深度核学习的对比**（degree-3多项式）：
- 训练时间：函数编码器比深度核（RBF）快约**10倍**（m=20时）
- 原因：深度核需要 $\mathcal{O}(m^3)$ 的Gram矩阵求逆
- Gram矩阵几何结构几乎一致，验证函数编码器可视为原始空间的自适应核

### 关键发现

1. 两种PCA引导算法均能精确识别函数空间的内在维度
2. 紧凑基表示可大幅减少基函数数量（50倍），无精度损失
3. 函数编码器在训练效率上显著优于深度核方法
4. 理论界提供了非空洞的推理时保证
5. 学习到的基函数在动力学系统中展现出可解释的物理结构

## 亮点与洞察

1. **统一了两个世界**：将神经网络的可扩展性与核方法的理论保证桥接，函数编码器恰好处于交汇点
2. **PCA与基选择的类比精妙**：将函数空间的维度选择问题转化为系数空间的PCA截断问题
3. **理论贡献扎实**：截断高斯PAC-Bayes技术本身具有独立价值，可用于其他无界损失场景
4. **实验设计有层次**：从已知维度的多项式验证到实际动力学系统，逐步展示方法的有效性

## 局限与展望

1. 停止阈值 τ 仍是**启发式**的，缺乏非启发式的基选择准则
2. 渐进式训练本质上是串行的，在大规模问题上效率受限
3. 实验仅在低维动力学系统上验证，高维复杂系统（PDE、视觉任务）的表现未知
4. 泛化界中的常数可能过大，虽然非空洞但实际松紧度有待检验
5. 未探索与具体应用（如机器人控制、轨道预测）的端到端集成

## 相关工作与启发

- **核方法（GP/KRR/SVM）**：理论完善但扩展性差，函数编码器在保留理论性质的同时解决这一问题
- **核近似（Nyström/随机Fourier特征）**：缓解计算成本但固定核无法适应数据
- **深度核学习**：参数化核提升表达力但保留 $\mathcal{O}(m)$ 推理成本
- **字典学习/Koopman/SINDy**：也学习基函数但目标不同（稀疏恢复/动力学线性化）
- 启发点：函数编码器的核视角可能为其他迁移学习方法提供理论分析工具

## 评分

- **新颖性**: ★★★★☆ — 统一视角有理论新意，但函数编码器本身并非全新
- **技术深度**: ★★★★★ — Rademacher+PAC-Bayes双重理论分析，截断高斯技术有独立价值
- **实验说服力**: ★★★★☆ — 多项式验证充分，动力学系统展示有力，但缺乏大规模实验
- **实用价值**: ★★★☆☆ — 理论贡献大于实际应用，紧凑基在嵌入式系统中有潜力
- **表达清晰度**: ★★★★★ — 理论推导层次分明，实验设计循序渐进

<!-- RELATED:START -->

## 相关论文

- [The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)
- [Searching Latent Program Spaces](../../NeurIPS2025/human_understanding/searching_latent_program_spaces.md)
- [One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)
- [Seeing without Pixels: Perception from Camera Trajectories](../../CVPR2026/human_understanding/seeing_without_pixels_perception_from_camera_trajectories.md)
- [Distributionally Robust Online Markov Game with Linear Function Approximation](../../AAAI2026/human_understanding/distributionally_robust_online_markov_game_with_linear_function_approximation.md)

<!-- RELATED:END -->
