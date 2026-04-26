---
title: >-
  [论文解读] Constant Stepsize Local GD for Logistic Regression: Acceleration by Instability
description: >-
  [ICML2025][优化][Local GD] 证明了 Local GD 在分布式逻辑回归问题上可以使用**任意正步长** $\eta > 0$ 收敛，且通过允许初始不稳定阶段的非单调目标下降，可实现比现有凸优化最坏情况下界更快的 $\widetilde{\mathcal{O}}(M/(\gamma^5 R^2))$ 收敛速率。
tags:
  - ICML2025
  - 优化
  - Local GD
  - 逻辑回归
  - 分布式优化
  - 大步长
  - 稳定性边缘
  - 联邦学习
---

# Constant Stepsize Local GD for Logistic Regression: Acceleration by Instability

**会议**: ICML2025  
**arXiv**: [2506.13974](https://arxiv.org/abs/2506.13974)  
**代码**: 无  
**领域**: optimization  
**关键词**: Local GD, 逻辑回归, 分布式优化, 大步长, 稳定性边缘, 联邦学习

## 一句话总结
证明了 Local GD 在分布式逻辑回归问题上可以使用**任意正步长** $\eta > 0$ 收敛，且通过允许初始不稳定阶段的非单调目标下降，可实现比现有凸优化最坏情况下界更快的 $\widetilde{\mathcal{O}}(M/(\gamma^5 R^2))$ 收敛速率。

## 研究背景与动机
- **核心矛盾**：Local SGD/GD 在实践中（联邦学习、大规模分布式训练）广泛使用且表现优异，但现有理论分析要求步长 $\eta \leq \mathcal{O}(1/K)$（$K$ 为通信间隔），这保证了目标函数的单调下降，却与实际中常见的非单调下降现象严重脱节。
- **理论空白**：单机 GD 已被证明在逻辑回归中可以用任意步长收敛（Wu et al., 2024），且大步长可带来加速收敛。但分布式 Local GD 是否也有类似性质，此前没有理论回答。先前工作 (Crawshaw et al., 2025) 只分析了一个**两阶段**变体（先小步长后大步长），原始的常数步长 Local GD 仍是开放问题。
- **最坏情况下界的局限**：现有 Local GD 的最坏情况下界 $\Omega(R^{-2/3})$ 针对一般凸+光滑目标，而逻辑回归（线性可分数据）不属于该问题类（因为不存在有界最优解 $\mathbf{w}_*$），提示问题特定分析可能突破这些下界。

## 方法详解

### 问题设定
- $M$ 个客户端，每个客户端有 $n$ 个数据点，数据维度 $d$
- 全局数据集线性可分，最大间隔为 $\gamma$，最大间隔分类器为 $\mathbf{w}_*$
- 目标：最小化全局逻辑损失 $F(\mathbf{w}) = \frac{1}{M}\sum_{m=1}^{M} F_m(\mathbf{w})$，其中 $F_m(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n} \ell(\langle \mathbf{w}, \mathbf{x}_i^m \rangle)$
- 算法：标准 Local GD（Algorithm 1），每轮在各客户端做 $K$ 步本地梯度下降后平均

### 核心理论结果

**定理 4.1（不稳定阶段，平均损失上界）**：对任意 $r \geq 0$，
$$\frac{1}{r}\sum_{s=0}^{r-1} F(\mathbf{w}_s) \leq 26 \cdot \frac{\|\mathbf{w}_0\|^2 + 1 + \log^2(K + \eta K \gamma^2 r) + \eta^2 K^2}{\eta \gamma^4 r}$$
对**任意** $\eta > 0$ 和 $K \geq 1$ 成立。右侧关于 $\eta$ 至多线性增长、关于 $K$ 至多二次增长。

**定理 4.2（稳定阶段，末端迭代上界）**：定义过渡时间 $\tau$，对 $r \geq \tau$：
$$F(\mathbf{w}_r) \leq \frac{16}{\eta \gamma^2 K (r - \tau)}$$
过渡时间 $\tau = \widetilde{\mathcal{O}}(\eta K M / \gamma^3)$。

**推论 4.3（最优参数选择）**：取 $\eta K = \widetilde{\Theta}(\gamma^3 R / M)$，当 $R \geq \widetilde{\Omega}(\max(Mn/\gamma^2, KM/\gamma^3))$ 时：
$$F(\mathbf{w}_R) \leq \widetilde{\mathcal{O}}\left(\frac{M}{\gamma^5 R^2}\right)$$
这是 $R^{-2}$ 速率，**突破**了一般分布式凸优化的 $R^{-2/3}$ 最坏情况下界。

### 关键技术手段

1. **轨迹分解与 $\beta$ 系数**：将 Local GD 的一轮更新分解为各数据点贡献的线性组合，引入系数 $\beta_{r,i}^m$ 来关联 Local GD 与 GD 的轨迹。通过上下界 $1/K \leq \beta_{r,i}^m \leq 1 + \exp(\|\mathbf{w}_r\|)$ 进行分析。
2. **Split Comparator 技术**（改编自 Wu et al., 2024a）：使用分裂比较器 $\mathbf{u} = \mathbf{u}_1 + \mathbf{u}_2$ 分析不稳定阶段，建立 $\|\mathbf{w}_r\|$ 的递推界（Lemma 4.4），证明 $\|\mathbf{w}_r\|$ 仅对数增长。
3. **自适应光滑性**：利用逻辑损失的特殊性质 $\|\nabla^2 F(\mathbf{w})\| \leq F(\mathbf{w})$ 和 $\|\nabla F(\mathbf{w})\| \leq F(\mathbf{w})$，证明当目标值足够小时，局部光滑常数也小，从而大步长也能保证单调下降（Lemma 4.5 修正下降不等式）。
4. **偏差控制（Lemma 4.8）**：证明当 $F(\mathbf{w}_r) \leq \gamma/(70\eta K M)$ 时，更新偏差 $\|\mathbf{b}_r\| \leq \frac{1}{5}\|\nabla F(\mathbf{w}_r)\|$，确保稳定阶段的下降递推。

## 实验关键数据

### 实验设置

| 数据集 | 客户端数 $M$ | 每客户端数据 $n$ | 通信轮数 $R$ |
|--------|:---:|:---:|:---:|
| 合成数据 | 2 | 1 | 2048 |
| MNIST (二分类) | 5 | 200 | 2048 |

参数搜索范围：$\eta \in \{2^{-2}, 2^0, 2^2, 2^4, 2^6, 2^8, 2^{10}\}$，$K \in \{2^0, 2^2, 2^4, 2^6\}$

### 主要发现

| 问题 | 结论 | 细节 |
|------|------|------|
| Q1: 大 $\eta, K$ 能否加速？ | ✅ 是 | 增大 $\eta$ 或 $K$ 后最终误差更小，尽管初始不稳定；$\eta=2^{10}$ 时发散 |
| Q2: 固定 $K$ 调 $\eta$ 有益？ | ✅ 是 | 大 $K$ 使更大 $\eta$ 成为可能（如 MNIST 上 $K=1$ 时 $\eta=256$ 发散，$K=16$ 时收敛） |
| Q3: 固定 $\eta K$ 变 $K$ 有益？ | ❌ 基本无 | 最终误差几乎相同，但大 $K$ 减少过渡到稳定阶段的轮数 |

### 关键观察
- 增大 $K$ **不增加**（甚至减少）达到稳定阶段的通信轮数，这比理论预测（$\tau \propto \eta K$）更强
- 大 $K$ 对 $\eta$ 有"稳定化"效果：允许使用原本会导致发散的更大步长
- $\eta = 2^{10}$ 的所有实验均发散，说明步长仍有有效上界

## 亮点与洞察
- **首个任意步长收敛保证**：完全去掉了 $\eta \leq \mathcal{O}(1/K)$ 的限制，是 Local GD 理论的重要突破
- **"不稳定带来加速"的范式**：与 Edge of Stability 现象呼应——非单调下降不是缺陷而是加速的关键机制
- **问题特定分析的价值**：用窄而精确的分析（逻辑回归）揭示通用分析遗漏的现象，为理论与实践对齐提供了范例
- **优雅的分解技术**：$\beta$ 系数将 Local GD 轨迹与 GD 轨迹联系起来，上下界分析巧妙（下界 $1/K$ 虽简单但紧致到对数因子）

## 局限与展望
1. **Local GD 效果不如 GD**：当前保证 $M/(\gamma^5 R^2)$ 比单机 GD 的 $1/(\gamma^4 R^2)$ 差，未能证明 $K > 1$ 的收益
2. **$\eta$ 和 $K$ 仅以乘积 $\eta K$ 出现**：理论不能区分不同 $(η, K)$ 组合（只要 $\eta K$ 相同），与实验中观察到的 $K$ 独立效益矛盾
3. **$\gamma$ 依赖性次优**：源于控制偏差项 $\|\mathbf{b}_r\|$ 时需要额外的 $\gamma$ 因子（Lemma B.2），改进需要 Local GD 隐式偏差的分析
4. **问题设定窄**：仅限逻辑回归+线性可分，未扩展至 Local SGD 或神经网络训练
5. **无隐式偏差结论**：未证明 Local GD 收敛到最大间隔解

## 相关工作与启发
- **Wu et al. (2024a/b)**：单机 GD 对逻辑回归的任意步长收敛和加速分析，本文将其核心技术（split comparator、gradient potential）推广到分布式场景
- **Crawshaw et al. (2025)**：两阶段 Local GD 分析（先小步长后大步长），本文统一为常数步长一阶段
- **Cohen et al. (2021)**：Edge of Stability 现象的实证发现，本文为其提供了分布式优化中的理论支撑
- **Patel et al. (2024)**：Local GD 对一般凸优化的下界，本文的加速率突破了该下界
- 启发：问题特定分析可能是弥合分布式优化理论-实践差距的关键路径，后续可扩展至近似齐次激活的两层网络

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次证明 Local GD 任意步长收敛，"不稳定加速"视角新颖
- 实验充分度: ⭐⭐⭐ — 合成+MNIST 验证充分但规模有限，CIFAR-10 仅在附录
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，证明概述精炼，定理-推论-实验布局典范
- 价值: ⭐⭐⭐⭐ — 理论贡献扎实，但 Local GD 未比 GD 更优限制了实际影响

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression](benefits_of_early_stopping_in_gradient_descent_for_overparameterized_logistic_re.md)
- [\[NeurIPS 2025\] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](../../NeurIPS2025/optimization/large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)
- [\[AAAI 2026\] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](../../AAAI2026/optimization/fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)
- [\[ICML 2025\] In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention](in-context_linear_regression_demystified_training_dynamics_and_mechanistic_inter.md)
- [\[ICML 2025\] The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning](the_panaceas_for_improving_low-rank_decomposition_in_communication-efficient_fed.md)

<!-- RELATED:END -->
