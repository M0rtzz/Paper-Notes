---
title: >-
  [论文解读] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?
description: >-
  [ICLR 2026][优化][SignSGD] 在幂律随机特征（Power-Law Random Features）模型下，系统分析了 SignSGD 的缩放定律，揭示了 SignSGD 相对于 SGD 的两个独特效应——漂移归一化和噪声重塑，并证明在噪声主导的情形下 SignSGD 的计算最优斜率可以超过 SGD。
tags:
  - ICLR 2026
  - 优化
  - SignSGD
  - 缩放定律
  - 线性回归
  - 随机特征
  - 学习率调度
---

# Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?

**会议**: ICLR 2026  
**arXiv**: [2603.02069](https://arxiv.org/abs/2603.02069)  
**代码**: 无  
**领域**: 优化理论  
**关键词**: SignSGD, 缩放定律, 线性回归, 随机特征, 学习率调度

## 一句话总结

在幂律随机特征（Power-Law Random Features）模型下，系统分析了 SignSGD 的缩放定律，揭示了 SignSGD 相对于 SGD 的两个独特效应——漂移归一化和噪声重塑，并证明在噪声主导的情形下 SignSGD 的计算最优斜率可以超过 SGD。

## 研究背景与动机

SignSGD 是 Adam 等自适应优化器的核心组件之一——它使用梯度的符号而非梯度本身来更新参数。在大规模语言模型训练中，Adam/AdamW 几乎是事实标准，但对 SignSGD 为什么以及何时优于 SGD 的理论理解仍然有限。

**现有理论的不足**：
1. 传统的 SignSGD 分析大多关注凸优化或简单设定，未考虑现代深度学习中观察到的缩放定律现象
2. Paquette et al. (2024) 分析了 SGD 在幂律随机特征模型下的缩放定律，但未涉及 SignSGD
3. 缺乏对 SignSGD 在什么条件下优于 SGD 的精确刻画

**核心问题**：
- SignSGD 的 population risk 如何随模型大小、训练步数、学习率缩放？
- 计算最优配置下，SignSGD 和 SGD 的缩放行为何时不同？
- WSD（warmup-stable-decay）学习率调度对 SignSGD 有何独特影响？

作者选择幂律随机特征模型作为分析框架，这是因为它能同时捕获特征衰减和目标衰减两个关键维度，而这两个维度正是理解缩放定律的核心。

## 方法详解

### 整体框架

分析框架：
- **模型**：线性模型 $f(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x})$
- **特征**：Gaussian-sketched 随机特征，具有幂律衰减谱 $\lambda_k \propto k^{-\alpha}$（特征衰减参数 $\alpha$）
- **目标**：幂律衰减目标 $\beta_k \propto k^{-\gamma}$（目标衰减参数 $\gamma$）
- **优化器**：单遍（one-pass）SignSGD
- **度量**：population risk（泛化误差）

### 关键设计

1. **Population Risk 的闭合表达式**：作者推导出 SignSGD 训练的线性模型 population risk 作为模型大小 $d$、训练步数 $T$、学习率 $\eta$、特征衰减 $\alpha$ 和目标衰减 $\gamma$ 的函数。

   → 核心思路：将 SignSGD 的非线性更新规则在随机特征模型下线性化分析  
   → 设计动机：获得可解释的闭合形式，便于与 SGD 的已知结果直接比较

2. **漂移归一化效应（Drift-Normalization Effect）**：SignSGD 对梯度取符号，相当于对每个坐标的更新步长进行归一化。在期望层面，这改变了有效漂移项（bias term）的缩放行为。

   → 核心思路：符号操作消除了梯度幅度信息，使得沿所有方向的更新步长相等  
   → 设计动机：这解释了为什么 SignSGD/Adam 在某些场景下收敛更快——它自动平衡了不同特征方向的学习速度

3. **噪声重塑效应（Noise-Reshaping Effect）**：SignSGD 不仅改变漂移项，还改变了噪声项的结构。具体来说，符号操作将随机梯度噪声从依赖于特征缩放的非均匀分布重塑为更均匀的分布。

   → 核心思路：取符号后，大梯度和小梯度的噪声贡献被拉平  
   → 设计动机：噪声重塑是 SignSGD 可以超越 SGD 的关键机制——当噪声主导时，重塑后的噪声可以具有更好的衰减行为

4. **计算最优缩放定律**：在最优学习率选择下，推导计算最优配置（compute-optimal scaling），即固定总计算量 $C = d \times T$ 时如何分配模型大小和训练步数。

   → 核心思路：类似 Chinchilla 定律的分析，但针对 SignSGD  
   → 设计动机：实际训练中最关心的是"给定计算预算，怎么配置最好"

5. **WSD 调度的分析**：分析了 warmup-stable-decay 学习率调度在 SignSGD 下的效果。当特征衰减快（大 $\alpha$）但目标衰减慢（小 $\gamma$）时，WSD 进一步降低噪声项并锐化计算最优斜率。

   → 核心思路：分段分析不同学习率阶段的贡献  
   → 设计动机：WSD 是现代 LLM 训练的标准调度策略，理解其与 SignSGD 的交互至关重要

### 损失函数 / 训练策略

理论分析框架，主要工具包括：
- 随机矩阵理论（Random Matrix Theory）
- 确定性等价（Deterministic Equivalents）
- 幂律渐近展开

所有结果都有严格的数学证明（论文89页，包含25个图）。

## 实验关键数据

### 主实验

**SignSGD vs SGD 的计算最优斜率比较**

| 参数区域 | SignSGD 斜率 | SGD 斜率 | 胜者 | 说明 |
|---------|------------|---------|------|------|
| 噪声主导区（高 $T$ 相对于 $d$）| 更陡 | 较缓 | SignSGD | 噪声重塑效应使 SignSGD 受益 |
| 偏差主导区（低 $T$ 相对于 $d$）| 相近或较缓 | 相近 | SGD 或持平 | 漂移归一化可能不利 |
| 平衡区 | 过渡行为 | 过渡行为 | 取决于具体 $\alpha, \gamma$ | 两个效应竞争 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 固定 $\alpha$，变化 $\gamma$ | 斜率变化 | 目标衰减越慢，SignSGD 优势越大 |
| 固定 $\gamma$，变化 $\alpha$ | 斜率变化 | 特征衰减对两者影响类似 |
| 有/无 WSD 调度 | 噪声项减少 | WSD 在特征衰减快 + 目标衰减慢时效果显著 |
| 不同学习率 | 最优配置 | 最优学习率的选择方式与 SGD 不同 |

### 关键发现

1. **SignSGD 优于 SGD 的条件明确**：当训练处于噪声主导区域（高步数/低模型大小比）时，SignSGD 的计算最优缩放更优。这与实践中观察到的"Adam 比 SGD 好"的现象一致——大规模训练通常处于这个区域
2. **两个效应可以分离**：漂移归一化和噪声重塑效应有明确的数学表达，可以独立分析其对缩放律的贡献
3. **WSD 调度的理论支撑**：首次为 WSD 调度在 SignSGD 上的有效性提供了理论解释——它通过降低噪声项来锐化缩放斜率
4. **幂律参数决定一切**：$\alpha$（特征衰减）和 $\gamma$（目标衰减）两个参数完全决定了 SignSGD 和 SGD 的相对表现

## 亮点与洞察

1. **填补重要理论空白**：首次为 SignSGD 建立了完整的缩放定律理论，与 SGD 的已有分析形成对比
2. **两个独特效应的发现**：漂移归一化和噪声重塑是理解 Adam 类优化器为何有效的关键概念，有望推广到更一般的设定
3. **实践指导意义**：明确了 SignSGD/Adam 超越 SGD 的条件——噪声主导区域，这与大规模 LLM 训练的实际场景吻合
4. **WSD 调度的理论基础**：为目前广泛使用但缺乏理论支撑的 WSD 调度提供了解释
5. **分析的彻底性**：89 页的论文、25 个图，覆盖了参数空间的各种区域，是一项极其细致的理论工作

## 局限与展望

1. **模型假设的简化性**：线性回归 + 随机特征模型与实际的深度网络训练有较大差距。虽然幂律特征模型能捕获一些核心现象，但非线性效应、参数间耦合等被忽略
2. **单遍训练假设**：one-pass SGD/SignSGD 意味着每个样本只见一次，而实际训练通常是多 epoch 的
3. **未考虑动量**：实际的 Adam 包含一阶和二阶动量，而 SignSGD 只是其简化。动量的加入可能改变缩放行为
4. **Gaussian-sketched 特征**：特征的 Gaussian 假设可能无法捕获真实数据中的结构
5. **向非线性扩展**：将分析扩展到两层或多层网络是重要但困难的方向
6. **实验验证**：理论预测与实际深度学习训练中的缩放行为是否一致，需要实证验证

## 相关工作与启发

- **Paquette et al. (2024)**：SGD 在幂律随机特征模型下的缩放定律分析，是本文最直接的比较对象
- **Chinchilla 缩放定律**（Hoffmann et al., 2022）：计算最优配置的原始工作，本文在 SignSGD 下得到类似的分析
- **Neural Scaling Laws**（Kaplan et al., 2020）：关于模型大小和数据量的缩放关系，本文从优化器角度补充
- 启发：理解优化器对缩放定律的影响与理解模型架构和数据同样重要；签名操作这样简单的非线性能产生复杂而有益的效应

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统分析 SignSGD 缩放定律，漂移归一化和噪声重塑是新概念
- 实验充分度: ⭐⭐⭐⭐ — 极其彻底的理论分析+数值验证（89页），但缺少深度学习实验
- 写作质量: ⭐⭐⭐⭐ — 理论严谨，但篇幅极长可能影响可读性
- 价值: ⭐⭐⭐⭐ — 为理解自适应优化器的理论提供了重要洞察

<!-- RELATED:START -->

## 相关论文

- [Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](../../NeurIPS2025/optimization/functional_scaling_laws_in_kernel_regression_loss_dynamics_and_learning_rate_sch.md)
- [When to Restart? Exploring Escalating Restarts on Convergence](when_to_restart_exploring_escalating_restarts_on_convergence.md)
- [Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)

<!-- RELATED:END -->
