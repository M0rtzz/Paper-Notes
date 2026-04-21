---
title: >-
  [论文解读] Tight Lower Bounds and Improved Convergence in Performative Prediction
description: >-
  [NeurIPS 2025][Performative Prediction] 在 performative prediction 框架下，首次证明了 Repeated Risk Minimization (RRM) 收敛率的紧致性，并提出 Affine Risk Minimizers (ARM) 算法类，通过利用历史训练快照的数据实现更广泛问题类上的收敛。
tags:
  - NeurIPS 2025
  - Performative Prediction
  - 收敛率
  - 下界
  - Repeated Risk Minimization
  - 决策依赖分布
---

# Tight Lower Bounds and Improved Convergence in Performative Prediction

**会议**: NeurIPS 2025  
**arXiv**: [2412.03671](https://arxiv.org/abs/2412.03671)  
**代码**: 暂无  
**领域**: 机器学习理论 / 决策依赖学习  
**关键词**: Performative Prediction, 收敛率, 下界, Repeated Risk Minimization, 决策依赖分布

## 一句话总结

在 performative prediction 框架下，首次证明了 Repeated Risk Minimization (RRM) 收敛率的紧致性，并提出 Affine Risk Minimizers (ARM) 算法类，通过利用历史训练快照的数据实现更广泛问题类上的收敛。

## 研究背景与动机

Performative prediction 研究的是模型部署后改变数据分布的场景——模型的预测会影响被预测对象的行为，从而改变未来数据的分布。例如：

- **信用评分**：信用模型部署后，申请人会策略性地调整行为以获得更好的评分
- **环境监管**：企业可能操纵排放数据以满足监管目标
- **教育**：学校会针对标准化测试进行应试教育

核心目标是收敛到**performatively stable point** $\theta_{\text{PS}}$：在该点上，模型在其自身诱导的分布上最小化风险，进一步的重训练不会改变模型参数。

**现有工作的局限**：

- Perdomo et al. (2020) 在假设 $\frac{\beta\epsilon}{\gamma} < 1$ 下证明了 RRM 以速率 $(\frac{\beta\epsilon}{\gamma})^t$ 线性收敛，但**从未分析过该上界是否紧致**
- Mofakhami et al. (2023) 在 Pearson $\chi^2$ 散度下给出了收敛条件，但上界含有多余常数 $C$
- 两个框架都**只使用最近一次迭代的数据**，丢弃了历史信息

**核心问题**：(1) RRM 的收敛率是否已经是最优的？(2) 能否通过利用历史快照数据打破 RRM 的收敛下界？

## 方法详解

### 整体框架

Performative prediction 的 RRM 更新为 $\theta^{t+1} = \arg\min_\theta \mathbb{E}_{z \sim \mathcal{D}(f_{\theta^t})}[\ell(f_\theta(x), y)]$，即在当前模型诱导的分布上重训练。本文的框架扩展为 ARM：在历史分布的仿射组合上重训练。

### 关键设计

1. **紧致下界（Theorem 2 & 3）**：首次证明两个主流框架下 RRM 的收敛率都是紧致的。

   **Mofakhami 框架下**（Theorem 2）：存在问题实例使得

   $$\|f_{\theta^t} - f_{\theta_{\text{PS}}}\|_{f_{\theta_{\text{PS}}}} = \Omega\left(\left(\frac{\sqrt{\epsilon}M}{\gamma}\right)^t\right)$$

   **Perdomo 框架下**（Theorem 3）：存在问题实例使得

   $$\|\theta^t - \theta_{\text{PS}}\| = \Omega\left(\left(\frac{\epsilon\beta}{\gamma}\right)^t \|\theta_0 - \theta_{\text{PS}}\|\right)$$

   **证明技术**：受 Nesterov 凸优化下界证明的启发，构造一个特殊的矩阵 $A$（下三角全 1 矩阵），每次 RRM 迭代"发现"一个新维度。损失函数取 $\ell(f_\theta(x),y) = \frac{\gamma}{2}\|\theta - \frac{\beta}{\gamma}z\|^2$，分布映射取 $\mathcal{D}(\theta) = \mathcal{N}(\frac{\epsilon}{2}A\theta + e_1, \sigma^2)$。这确保 RRM 更新遵循 $\theta^{t+1} = \frac{\beta}{\gamma}(\frac{\epsilon}{2}A\theta^t + e_1)$，通过未发现维度的贡献累积得到下界。

2. **改进的上界（Theorem 1）**：在 Mofakhami 框架下，RRM 收敛率从 $(\frac{\sqrt{\epsilon}CM}{\gamma})^t$ 改进为 $(\frac{\sqrt{\epsilon}M}{\gamma})^t$，消除了常数 $C$。改进来自于修改 Assumption 1 中的 $\epsilon$-sensitivity 定义，使用 $\|f_\theta - f_{\theta'}\|_{f_\theta}^2$ 而非 $\|f_\theta - f_{\theta'}\|^2$。

3. **Affine Risk Minimizers (ARM)（Section 5）**：核心创新——利用历史快照数据的仿射组合：

   $$D_t = \sum_{i=0}^{t-1} \alpha_i^{(t)} D(f_{\theta^i}), \quad \sum_{i=0}^{t-1}\alpha_i^{(t)} = 1$$

   **Lemma 1（2-Snapshots ARM）**：仅使用最近两次快照的平均 $D_t = \frac{1}{2}D(f_{\theta^t}) + \frac{1}{2}D(f_{\theta^{t-1}})$，收敛条件从 $\frac{\sqrt{\epsilon}M}{\gamma} < 1$ 放宽到 $\frac{\sqrt{3}}{2}\frac{\sqrt{\epsilon}M}{\gamma} < 1$，等价于 $\frac{\sqrt{\epsilon}M}{\gamma} < \frac{2}{\sqrt{3}} \approx 1.155$。

   **Theorem 4**：证明 2-Snapshots ARM 生成柯西序列并收敛到稳定点。这突破了 Theorem 2 的 RRM 下界，证明利用历史数据确实能在更广的问题类上收敛。

4. **ARM 下界（Theorem 5 & 6）**：给出了 ARM 类算法可实现的收敛率下界。

   Perdomo 框架下（Theorem 6）：$\|\theta^t - \theta_{\text{PS}}\| = \Omega((\frac{\epsilon\beta}{2\gamma})^t)$。这意味着 RRM 的上界 $(\frac{\epsilon\beta}{\gamma})^t$ 与 ARM 的下界 $(\frac{\epsilon\beta}{2\gamma})^t$ 仅差**常数因子 2**，精确量化了利用历史数据的最大潜在收益。

### 训练策略

ARM 的实际实现是在每步 $t$ 对最近 $\tau$ 个快照的数据做均匀混合：$D_t = \frac{1}{\tau}\sum_{i=t-\tau+1}^{t} D(f_{\theta^i})$。窗口大小 $\tau$ 控制着收敛速度与计算开销的权衡。

## 实验关键数据

### 主实验：信用评分 RIR 环境

| 窗口大小 $\tau$ | 早期收敛速度 | 50 步后 $\Delta\mathcal{R}_t$ | 说明 |
|---|---|---|---|
| $\tau=1$（RRM） | 基线 | ~0.02 | 标准 RRM |
| $\tau=2$ | 快约 2x | ~0.01 | 损失偏移减半 |
| $\tau=4$ | 快约 3x | ~0.005 | 继续改善 |
| $\tau=t/2$ | 最快 | ~0.003 | 接近饱和 |
| $\tau=\text{all}$ | ≈ $t/2$ | ~0.003 | 边际收益递减 |

### 消融实验：下界验证

| 框架 | 理论下界 | 实验观察 | 说明 |
|---|---|---|---|
| Perdomo | $(\frac{\epsilon\beta}{\gamma})^t$ | 所有 $\tau$ 值均不突破 | 验证了 Theorem 3 的紧致性 |
| ARM 下界 | $(\frac{\epsilon\beta}{2\gamma})^t$ | $\tau \geq 2$ 突破 RRM 上界但不突破 ARM 下界 | 验证了 Theorem 6 |
| 改进的 Mofakhami | $(\frac{\sqrt{\epsilon}M}{\gamma})^t$ | 匹配上界 | 验证了 Theorem 1 的紧致性 |

### 关键发现

- $\tau$ 从 1 增加到 2 带来的改善最大（损失偏移减半），之后边际收益递减
- 所有实验中收敛轨迹均不突破理论下界，验证了紧致性分析
- 更大的 $\tau$ 意味着更高的计算和存储成本（线性增长），需要在速度与资源间权衡
- ARM 的稳定点集合与 RRM 完全一致（Lemma 9），利用历史数据不改变均衡，只加速收敛

## 亮点与洞察

- **首次紧致性分析**：证明现有 RRM 上界无法在不引入额外假设的情况下进一步改进，这是一个清晰的不可能性结果
- **简洁的下界构造**：受 Nesterov 下界启发的维度增长矩阵 $A$ 是一个巧妙的构造，既满足所有假设又显示收敛率不可改进
- **打破下界的条件明确**：ARM 通过使用两个快照将收敛阈值从 1 放宽到 1.155，虽然改进幅度不大但意义在于突破了 RRM 下界
- 下界分析技术可推广到其他迭代决策依赖优化（如 proximal ARM, Theorem 7）

## 局限与展望

- ARM 的收敛阈值改进有限（1 → 1.155），实际收益主要体现在收敛速度而非问题类的扩展
- RRM 与 ARM 的下界之间仍有 2 倍常数因子的间隙，不清楚是否能紧致
- 实验仅在两个半合成环境上进行（信用评分和拼车市场），缺少大规模真实场景
- 仿射组合中权重选择目前采用均匀分配，自适应权重可能带来更好的收敛
- 未讨论有限样本下的收敛保证（当前假设每步可获得精确的分布期望）
- 实际部署中存储所有历史快照的成本可能不可接受

## 相关工作与启发

- 与 Brown et al. (2022) 的 stateful performative prediction 的关系：后者考虑一般的历史依赖，ARM 是其特殊情况（仿射组合）
- 与 Nesterov 加速的类比：ARM 利用历史信息加速收敛，类似于动量方法利用历史梯度
- 下界构造技术可用于分析其他迭代学习框架（如联邦学习中的分布偏移问题）
- Theorem 8 展示了新的 $\epsilon$-sensitivity 定义在 RIR 机制下可以得到更紧的常数（$\delta^{-3/2}$ vs $\delta^{-2}$）

## 评分

- 新颖性: ⭐⭐⭐⭐ ARM 思路自然但首次形式化，紧致性分析是重要的理论贡献
- 实验充分度: ⭐⭐⭐ 实验设置较简单（两个半合成环境），但理论结果足够支撑结论
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，Figure 1-3 直观展示了核心思想
- 价值: ⭐⭐⭐⭐ 对 performative prediction 理论社区有显著贡献，下界技术有方法论价值

<!-- RELATED:START -->

## 相关论文

- [Movie101v2: Improved Movie Narration Benchmark](../../ACL2025/llm_evaluation/movie101v2_improved_movie_narration_benchmark.md)
- [Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization](conformal_prediction_in_the_loop_a_feedback-based_uncertainty_model_for_trajecto.md)
- [Improved and Oracle-Efficient Online $\ell_1$-Multicalibration](../../ICML2025/llm_evaluation/improved_and_oracle-efficient_online_ell_1-multicalibration.md)
- [Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads](let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e.md)
- [Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds](../../ICLR2026/llm_evaluation/non-clashing_teaching_in_graphs_algorithms_complexity_and_bounds.md)

<!-- RELATED:END -->
