---
title: >-
  [论文解读] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?
description: >-
  [ICLR 2026][优化/理论][SignSGD] 在幂律随机特征（Power-Law Random Features）模型下，系统分析了 SignSGD 的缩放定律，揭示了 SignSGD 相对于 SGD 的两个独特效应——漂移归一化和噪声重塑…
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "SignSGD"
  - "缩放定律"
  - "线性回归"
  - "随机特征"
  - "学习率调度"
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

全文搭建在幂律随机特征（Power-Law Random Features）这一可解析的玩具模型上：用线性模型 $f(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x})$ 拟合 Gaussian-sketched 随机特征，特征谱按 $\lambda_k \propto k^{-\alpha}$ 衰减、目标系数按 $\beta_k \propto k^{-\gamma}$ 衰减，再用单遍（one-pass）SignSGD 优化、以 population risk 衡量泛化。这套设定的好处是 $\alpha$（特征衰减）和 $\gamma$（目标衰减）两个标量就能同时刻画现代缩放定律里的"特征有多容易学"与"目标有多复杂"，从而把 SignSGD 的整套缩放行为压缩成 $(d, T, \eta, \alpha, \gamma)$ 的解析函数，与 Paquette et al. (2024) 对 SGD 的已知结果一一对照。

### 关键设计

**1. Population Risk 的闭合表达式：把符号更新变成可解析的对象。**

SignSGD 的更新规则 $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\,\mathrm{sign}(\nabla \ell_t)$ 因为取符号而高度非线性，直接分析无从下手。作者借助随机矩阵理论与确定性等价（Deterministic Equivalents），在随机特征模型下把这条非线性递推渐近线性化，最终给出 population risk 作为模型大小 $d$、训练步数 $T$、学习率 $\eta$ 以及衰减指数 $\alpha, \gamma$ 的闭合形式。有了这个解析式，SignSGD 才能和 SGD 的已有结果摆在同一坐标系里逐项比较，后续所有结论都是从它推出来的。

**2. 漂移归一化效应（Drift-Normalization）：符号操作如何改写偏差项。**

取符号会丢掉梯度的幅度，只保留方向，等价于把每个坐标的更新步长强行归一化到同一量级。在期望层面，这让 risk 中的漂移项（bias term，由初始权重沿各特征方向收敛的快慢决定）的缩放方式发生改变：不同特征方向不再按各自梯度大小、而是按统一步长前进。这正是 SignSGD/Adam"自动平衡各方向学习速度"直觉的数学体现——它在偏差主导的区域可能反而吃亏，因为均匀步长牺牲了对大特征方向的快速收敛。

**3. 噪声重塑效应（Noise-Reshaping）：SignSGD 能赢 SGD 的真正机制。**

符号操作不只动了漂移项，还重塑了噪声项的结构。随机梯度噪声原本依赖特征缩放、是高度非均匀的，取符号后大梯度方向与小梯度方向的噪声贡献被拉平、趋于均匀分布。关键在于：当训练处在噪声主导（而非偏差主导）的区域时，这种被重塑、更均匀的噪声拥有更好的衰减行为，于是 SignSGD 的 risk 比 SGD 下降得更快。漂移归一化和噪声重塑这两个效应在解析式里有各自独立的表达，可以分开追踪谁在哪个区域占上风。

**4. 计算最优缩放定律：给定算力预算该怎么配 $d$ 和 $T$。**

把学习率取到最优后，作者固定总计算量 $C = d \times T$，求解如何在模型大小 $d$ 与训练步数 $T$ 之间分配才能最小化 risk，得到 SignSGD 版本的计算最优（compute-optimal）斜率。这与 Chinchilla 定律的思路一致，但首次落到 SignSGD 上。比较两条计算最优曲线就能回答全文的核心问题：在噪声主导区，SignSGD 的最优斜率比 SGD 更陡（下降更快）；在偏差主导区则相近甚至更缓。

**5. WSD 调度的分析：为什么 warmup-stable-decay 配 SignSGD 特别有效。**

WSD（warmup-stable-decay）是当下 LLM 训练的标准学习率调度，但缺乏理论支撑。作者把它分成 warmup、stable、decay 三段分别代入解析式累加各阶段贡献，发现当特征衰减快（大 $\alpha$）而目标衰减慢（小 $\gamma$）时，decay 阶段会进一步压低噪声项，从而锐化计算最优斜率。这给了 WSD 在 SignSGD 下有效性的首个解释：它本质上是借末段降学习率来收割噪声重塑带来的红利。

整个分析是纯理论框架，核心工具是随机矩阵理论、确定性等价与幂律渐近展开，全部结论都附严格证明（正文 89 页、25 个图覆盖参数空间各区域）。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[NeurIPS 2025\] Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](../../NeurIPS2025/optimization/functional_scaling_laws_in_kernel_regression_loss_dynamics_and_learning_rate_sch.md)
- [\[ICLR 2026\] When to Restart? Exploring Escalating Restarts on Convergence](when_to_restart_exploring_escalating_restarts_on_convergence.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)

</div>

<!-- RELATED:END -->
