---
title: >-
  [论文解读] Semi-infinite Nonconvex Constrained Min-Max Optimization
description: >-
  [NeurIPS 2025][半无穷规划] 针对带有无穷多非凸约束的非凸 min-max 优化问题，提出 iDB-PD（不精确动态障碍原始-对偶）算法，在 Łojasiewicz 正则条件下建立了首个全局非渐近收敛保证，稳定性 $\mathcal{O}(\epsilon^{-3})$、可行性 $\mathcal{O}(\epsilon^{-6\theta})$、互补松弛 $\mathcal{O}(\epsilon^{-3\theta/(1-\theta)})$。
tags:
  - NeurIPS 2025
  - 半无穷规划
  - 非凸约束
  - Min-Max优化
  - 动态障碍法
  - 收敛复杂度
---

# Semi-infinite Nonconvex Constrained Min-Max Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2510.12007](https://arxiv.org/abs/2510.12007)  
**代码**: 暂无  
**领域**: 优化理论 / 鲁棒优化  
**关键词**: 半无穷规划, 非凸约束, Min-Max优化, 动态障碍法, 收敛复杂度

## 一句话总结

针对带有无穷多非凸约束的非凸 min-max 优化问题，提出 iDB-PD（不精确动态障碍原始-对偶）算法，在 Łojasiewicz 正则条件下建立了首个全局非渐近收敛保证，稳定性 $\mathcal{O}(\epsilon^{-3})$、可行性 $\mathcal{O}(\epsilon^{-6\theta})$、互补松弛 $\mathcal{O}(\epsilon^{-3\theta/(1-\theta)})$。

## 研究背景与动机

现代 AI 中的许多问题可以建模为带约束的 min-max 优化，例如：
- **鲁棒多任务学习**：优化优先任务的最坏 case 损失，同时约束其他任务的损失不超过阈值，且约束中的权重分布属于一个不确定性集合，形成无穷多约束。
- **鲁棒能量约束 DNN 训练**：在分布不确定性下最小化最坏 case 损失，同时满足能量和稀疏性的非凸约束。

这类问题的形式化表述为：

$$\min_{x \in \mathbb{R}^n} \max_{y \in Y} \phi(x, y), \quad \text{s.t.} \quad \psi(x, w) \leq 0, \quad \forall w \in W$$

其中 $Y, W$ 是连续紧集，$\phi$ 和 $\psi$ 均可以在 $x$ 上非凸。这同时涉及三大挑战：

**非凸性**：目标和约束在 $x$ 上非凸，无法求全局解

**无穷约束**：$w \in W$ 是连续集，约束数量无穷

**嵌套 max 结构**：目标和约束中都有内层最大化

现有方法的不足：
- 标准 min-max 算法假设有限维约束且约束凸
- 半无穷规划（SIP）方法通常假设凸性或缺乏目标中的 max 结构
- 非凸约束优化需要特殊正则条件来避免收敛到不可行驻点
- 最接近的工作 Yao et al. 仅处理凸情形

## 方法详解

### 整体框架

iDB-PD 算法是一个原始-对偶方法：
1. **原始更新**（$x$）：通过求解一个 QP 子问题得到搜索方向，同时减少目标函数和改善可行性
2. **对偶更新**（$y, w$）：通过梯度上升近似求解内层最大化问题

### 关键设计

#### 1. **QP 搜索方向**

在每步 $k$，搜索方向 $d_k$ 通过求解以下 QP 获得：

$$d_k = \arg\min_d \| d + \nabla_x \phi(x_k, y_k) \|^2 \quad \text{s.t.} \quad \nabla_x \psi(x_k, w_k)^\top d + \alpha_k \rho(x_k, w_k) \leq 0$$

其中 $\rho(x, w) = \|\nabla_x \psi(x, w)\|$ 是动态障碍函数。这个 QP 有闭式解：

$$d_k = -\nabla_x \phi(x_k, y_k) - \lambda_k \nabla_x \psi(x_k, w_k)$$

$$\lambda_k = \frac{1}{\|\nabla_x \psi(x_k, w_k)\|^2} [-\nabla_x \psi(x_k, w_k)^\top \nabla_x \phi(x_k, y_k) + \alpha_k \rho(x_k, w_k)]_+$$

设计动机：目标部分 $-\nabla_x \phi$ 驱动减小损失，约束部分 $-\lambda_k \nabla_x \psi$ 驱动改善可行性，QP 在两者之间找到最优平衡。

#### 2. **指示函数与对偶稳定性**

$\lambda_k$ 的更新在 $\|\nabla_x \psi\| \to 0$ 时会发散。为解决此问题，引入指示函数：

$$\zeta(x, w) = [\psi(x, w)]_+ \|\nabla_x \psi(x, w)\|$$

当 $\zeta(x_k, w_k) = 0$ 时，表示当前点可行或在约束的临界点上，此时直接设 $\lambda_k = 0$，搜索方向退化为纯目标函数梯度 $d_k = -\nabla_x \phi(x_k, y_k)$。

这一设计解决了 Hinder & Sidford (2020) 的动态障碍方法中需要假设对偶迭代有界的限制。

#### 3. **Łojasiewicz 正则条件**

对约束函数的平方不可行残差 $[\psi(x, w)]_+^2$ 施加 Łojasiewicz 不等式：

$$[\psi(x, w)]_+^{2\theta} \leq \mu \|\nabla_x \psi(x, w) [\psi(x, w)]_+\|, \quad \forall x \in \mathbb{R}^n, \forall w \in W$$

其中 $\theta \in (0, 1)$。当 $\theta = 1/2$ 时退化为经典的 Polyak-Łojasiewicz 条件。

设计动机：这个条件确保约束的梯度在临界点附近不会衰减得太快，从而保证算法能从不可行区域"逃逸"到可行区域。这对于使用光滑激活函数（如 tanh、softplus）的 DNN 是自然满足的。

### 损失函数 / 训练策略

完整算法每步执行：
1. 根据 $\zeta(x_k, w_k)$ 决定 $\lambda_k$（约束活跃则按 QP 解，否则设为 0）
2. 计算搜索方向 $d_k$ 并更新 $x_{k+1} = x_k + \gamma_k d_k$
3. 用 $N_k$ 步（加速）梯度上升近似求解 $y_{k+1} \approx \arg\max_{y \in Y} \phi(x_{k+1}, y)$
4. 用 $M_k$ 步（加速）梯度上升近似求解 $w_{k+1} \approx \arg\max_{w \in W} \psi(x_{k+1}, w)$

参数选择：$\gamma_k = \mathcal{O}(T^{-1/3})$，$\alpha_k = T^{1/3} / (k+2)^{1+\omega}$，$N_k = \mathcal{O}(\log(k+1))$，$M_k$ 依赖于 $\theta$ 和当前不可行残差。

## 实验关键数据

### 主实验 — 鲁棒多任务学习

在 Multi-MNIST、CHD49、Multi-Fashion-MNIST、Yeast、20NG 五个数据集上，将两个分类任务中的一个作为优先任务（目标），另一个作为约束任务。

| 方法 | 稳定性收敛 | 可行性收敛 | 互补松弛收敛 | 说明 |
|---|---|---|---|---|
| Adaptive Discretization + COOPER | ✗ 发散 | ✗ 不可行 | ✗ 发散 | 经典离散化方法失败 |
| **iDB-PD** | ✓ 收敛 | ✓ 收敛 | ✓ 收敛 | 三项指标均一致收敛 |

### 与加权聚合方法 (GDMA) 的比较

| 方法 | ρ值 | 稳定性 | 可行性 | 说明 |
|---|---|---|---|---|
| GDMA | ρ=1 | 低 | 不收敛 | 权重太小无法满足约束 |
| GDMA | ρ=10 | 不稳定 | 部分收敛 | 权重太大导致震荡 |
| **iDB-PD** | 自适应 | **收敛** | **收敛** | 无需手动调权重 |

### 关键发现

1. **离散化方法在此问题上失败**：自适应离散化 + COOPER 在可行性上无法收敛，在稳定性上发散，说明经典 SIP 方法不适用于非凸 min-max 设定。
2. **加权聚合的 ρ 选择困难**：GDMA 对 ρ 极度敏感，小 ρ 不满足约束、大 ρ 导致不稳定，而 iDB-PD 通过对偶变量自动平衡目标和约束。
3. **$\theta = 1/2$ 的特殊情况**下所有三项复杂度统一为 $\mathcal{O}(\epsilon^{-3})$，与有限约束非凸优化的已知最优结果一致。

## 亮点与洞察

- 首次为非凸半无穷 min-max 优化建立非渐近收敛保证，填补了 SIP 与 min-max 优化交叉领域的理论空白。
- 指示函数 $\zeta(x, w)$ 的引入巧妙解决了对偶发散问题，使算法无需假设对偶迭代有界。
- QP 子问题有闭式解，每步计算开销低，适用于大规模问题。
- 收敛保证依赖于 Łojasiewicz 指数 $\theta$，为实际问题提供了灵活的理论框架（不同 $\theta$ 对应不同收敛速率）。

## 局限与展望

- **ReLU 等非光滑激活函数不满足 Łojasiewicz 条件**，需要使用 softplus 等光滑替代——这在实际深度学习中是一个限制。
- 实验仅在小规模网络（单隐层 + tanh）上验证，未展示在现代深度网络上的可扩展性。
- $M_k$ 的选择依赖于未知的 $\theta$ 值，实际中需要估计或使用保守选择。
- 缺乏随机优化（SGD）版本的分析，目前要求精确梯度或近似精确的内层求解。

## 相关工作与启发

- **DBGD (Hinder & Sidford 2020)**：动态障碍梯度下降，本文的重要基础，但需要对偶有界假设且 $\tau = 1$ 时无收敛保证。
- **COOPER**：PyTorch 约束优化库，使用拉格朗日方法求解离散化问题，实验中作为对比。
- **GDMA**：梯度下降多上升方法，用于非凸-强凹 min-max，本文通过 constrained 版本与之对比。
- 启发：对于约束优化，自适应对偶调节（通过指示函数、动态障碍等）比固定惩罚/权重更稳健。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次处理非凸半无穷 min-max 设定，理论贡献显著
- **实验充分度**: ⭐⭐⭐ 实验规模偏小（单隐层网络），仅展示收敛曲线的定性结果，缺乏定量精度对比
- **写作质量**: ⭐⭐⭐⭐ 数学推导严谨，符号一致，但高度理论化的内容可读性一般
- **价值**: ⭐⭐⭐⭐ 为鲁棒优化和安全约束学习提供了理论工具，但实际适用性有待进一步验证

<!-- RELATED:START -->

## 相关论文

- [Learning (Approximately) Equivariant Networks via Constrained Optimization](learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees](fsnet_feasibility-seeking_neural_network_for_constrained_optimization_with_guara.md)
- [Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games](../../ICML2025/others/continuous-time_analysis_of_heavy_ball_momentum_in_min-max_games.md)
- [Optimized Learned Count-Min Sketch](optimized_learned_count-min_sketch.md)
- [Prediction-Powered Semi-Supervised Learning with Online Power Tuning](prediction-powered_semi-supervised_learning_with_online_power_tuning.md)

<!-- RELATED:END -->
