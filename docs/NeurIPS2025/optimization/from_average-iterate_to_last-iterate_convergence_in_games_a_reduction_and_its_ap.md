---
title: >-
  [论文解读] From Average-Iterate to Last-Iterate Convergence in Games: A Reduction and Its Applications
description: >-
  [NeurIPS 2025][优化][博弈学习] 提出 A2L (Average to Last-iterate) 黑箱规约，对效用函数关于自身策略和对手联合策略均线性的博弈，能将任意非耦合学习动力学的平均迭代转换为新动力学的末迭代，由此在多人零和多矩阵博弈中取得 $O(\log d / T)$ 梯度反馈和 $\tilde{O}(d^{1/5}T^{-1/5})$ bandit 反馈的 SOTA last-iterate 收敛率。
tags:
  - NeurIPS 2025
  - 优化
  - 博弈学习
  - last-iterate 收敛
  - OMWU
  - 零和博弈
  - black-box reduction
---

# From Average-Iterate to Last-Iterate Convergence in Games: A Reduction and Its Applications

**会议**: NeurIPS 2025  
**arXiv**: [2506.03464](https://arxiv.org/abs/2506.03464)  
**代码**: 无  
**领域**: optimization  
**关键词**: 博弈学习, last-iterate 收敛, OMWU, 零和博弈, black-box reduction

## 一句话总结
提出 A2L (Average to Last-iterate) 黑箱规约，对效用函数关于自身策略和对手联合策略均线性的博弈，能将任意非耦合学习动力学的平均迭代转换为新动力学的末迭代，由此在多人零和多矩阵博弈中取得 $O(\log d / T)$ 梯度反馈和 $\tilde{O}(d^{1/5}T^{-1/5})$ bandit 反馈的 SOTA last-iterate 收敛率。

## 研究背景与动机

**领域现状**：博弈中在线学习算法的收敛性是博弈论和机器学习的基础问题。经典的 no-regret 算法（如 MWU）保证平均迭代收敛到 Nash 均衡，但实际策略（last-iterate）可能发散、循环或混沌。

**现有痛点**：
   - Last-iterate 收敛比 average-iterate 收敛困难得多，算法设计需要 optimism/regularization，分析需要定制 Lyapunov 函数
   - 现有最优 last-iterate 率 $O(\text{poly}(d)/T)$（如 AOG 算法），维度依赖为多项式
   - Bandit 反馈下更困难，先前最优为 $\tilde{O}(T^{-1/8})$ 高概率 / $\tilde{O}(T^{-1/6})$ 期望

**核心矛盾**：average-iterate 收敛仅需 no-regret 性质即可（简单直接），而 last-iterate 收敛要求专门设计——能否弥合这一鸿沟？

**切入角度**：效用函数的线性性意味着 $u_i(\cdot, \bar{x}_{-i}) = \frac{1}{t}\sum_{k=1}^t u_i(\cdot, x_{-i}^k)$，因此从观测到的混合策略反馈可以恢复原始策略的反馈

**核心idea**：设计一个通用规约 A2L，让每个玩家输出内部策略的运行均值作为实际策略，通过线性性恢复内部策略的效用反馈，从而将 last-iterate 问题转化为 average-iterate 问题

## 方法详解

### 整体框架
A2L 是一个黑箱规约：输入任意在线学习算法 $\mathcal{R}$，输出新算法 $\text{A2L}(\mathcal{R})$。每轮 $t$，从 $\mathcal{R}$ 获取策略 $x^t$（内部不实际执行），实际执行运行均值 $\bar{x}^t = \frac{1}{t}\sum_{k=1}^t x^k$，收到反馈 $\bar{u}^t$ 后恢复 $u^t = t \cdot \bar{u}^t - \sum_{k=1}^{t-1} u^k$ 并传给 $\mathcal{R}$。

### 关键设计

1. **A2L 黑箱规约 (Algorithm 1)**:

    - 功能：将任意在线学习算法的平均迭代转化为新算法的末迭代
    - 核心思路：玩家 $i$ 执行 $\bar{x}_i^t = \frac{1}{t}\sum_{k=1}^t x_i^k$，观测 $\bar{u}_i^t = u_i(\cdot, \bar{x}_{-i}^t)$。由效用线性性，$\bar{u}_i^t = \frac{1}{t}\sum_{k=1}^t u_i(\cdot, x_{-i}^k)$，所以 $u_i^t = t \cdot \bar{u}_i^t - (t-1)\bar{u}_i^{t-1}$。将恢复的 $u^t$ 传给子程序 $\mathcal{R}$
    - 设计动机：规约保持非耦合性（每个玩家仅观测自身效用），不需要共享随机性或通信

2. **A2L-OMWU (梯度反馈)**:

    - 功能：将 A2L 应用于 OMWU 算法实现最优 last-iterate 收敛
    - 核心思路：OMWU 的 RVU 性质保证 $\sum_i \text{Reg}_i(T) \leq \frac{\sum_i \log d_i}{\eta}$（当 $\eta \leq \frac{1}{2(n-1)}$ 时，variation 项被 stability 项抵消）。由 Lemma 1，$\text{TGap}(\bar{x}^T) = \frac{1}{T}\sum_i \text{Reg}_i(T)$。由 Theorem 2，$\bar{x}^T$ 就是 A2L-OMWU 的 last-iterate
    - 设计动机：OMWU 已有 $O(\log d/T)$ average-iterate 收敛率，通过 A2L 直接获得相同率的 last-iterate 保证

3. **A2L-OMWU-Bandit (bandit 反馈, Algorithm 2)**:

    - 功能：在仅观测单动作收益（非全效用向量）时实现 last-iterate 收敛
    - 核心思路：每轮 $t$，混合 $\epsilon_t$-exploration：$\bar{x}_{i,\epsilon}^t = (1-\epsilon_t)\bar{x}_i^t + \epsilon_t \text{Uniform}(d_i)$。执行 $B_t$ 轮采样估计效用向量 $\hat{U}_{i,\epsilon}^t$，恢复 $\hat{u}_{i,\epsilon}^t = t\hat{U}_{i,\epsilon}^t - (t-1)\hat{U}_{i,\epsilon}^{t-1}$，传给 OMWU
    - 设计动机：bandit 反馈无法直接提供效用向量，需要通过重复采样估计；估计误差分析的关键技术在于利用 RVU 不等式的负项吸收二阶误差项

### 训练策略
- 梯度反馈：$\eta \leq \frac{1}{2(n-1)}$，无需知道时间视界 $T$（anytime 保证）
- Bandit 反馈：$\epsilon_t = t^{-1}$，$B_t = d \cdot t^4$，$\eta \leq \frac{1}{6n}$
- 对抗鲁棒性：监控累积遗憾，若超过阈值则切换到标准 no-regret 算法

## 实验关键数据

### 主要理论结果

| 反馈类型 | 算法 | last-iterate 收敛率 | 维度依赖 | 此前 SOTA |
|---------|------|-------------------|---------|----------|
| 梯度 | A2L-OMWU | $O(\frac{\log d}{T})$ | $\log d$ | $O(\frac{\text{poly}(d)}{T})$ (AOG) |
| Bandit | A2L-OMWU-Bandit | $\tilde{O}(d^{1/5}T^{-1/5})$ | $d^{1/5}$ | $\tilde{O}(\sqrt{d}T^{-1/8})$ (高概率) |

### 推论：动态遗憾

| 算法 | 动态遗憾 | 此前 SOTA |
|------|---------|----------|
| A2L-OMWU | $O(\log d \cdot \log T)$ | $O(\text{poly}(d) \cdot \log T)$ |

### 关键发现
- 梯度反馈下维度依赖从 $\text{poly}(d)$ 指数级改善到 $\log d$
- Bandit 反馈率从 $\tilde{O}(T^{-1/8})$ 改善到 $\tilde{O}(T^{-1/5})$，关键在于二阶误差分析技巧（利用 RVU 负项 $-\frac{1}{8\eta}\|x_i^t - x_i^{t-1}\|_1^2$ 吸收 $\|\delta_i^t\|_\infty^2$）
- A2L 适用于多人零和多矩阵博弈（不仅是二人），且可推广到 Fisher 市场的 PRD 动力学
- 无需正则化（不同于 Cen 等的方法），无需知道 $T$（真正的 anytime 保证）
- 动态遗憾仅 $O(\log d \cdot \log T)$，与 OMWU 的 $O(\log T)$ 静态遗憾接近

## 亮点与洞察
- **概念优雅**：A2L 规约极其简洁（Algorithm 1 仅几行），却产生了强大的理论结果。核心观察——效用线性性允许从混合反馈恢复纯反馈——虽然事后看来简单，但此前未被发现
- **Anytime 保证**：不同于正则化方法需预知 $T$，A2L-OMWU 对每个 $T \geq 1$ 都保证 $O(\log d/T)$-近似 Nash 均衡
- **Bandit 分析的技术创新**：利用 RVU 负项吸收估计误差的二阶项，从朴素的 $\tilde{O}(T^{-1/7})$ 改善到 $\tilde{O}(T^{-1/5})$
- **非线性效用的扩展**：对 Fisher 市场的 PRD 也适用，因为其反馈具有类似可恢复结构

## 局限与展望
- 依赖效用线性假设（Assumption 1），不适用于凹凸 min-max 优化等非线性效用设置——EG/OG 在此类问题中仍有优势
- 需要计算运行均值并存储历史效用向量 $\sum_{k=1}^{t-1} u^k$，空间开销为 $O(d \cdot T)$，不同于 EG/OG 等无需均值化的算法
- Bandit 反馈的 $\tilde{O}(T^{-1/5})$ 是否最优仍为开放问题；$B_t = d \cdot t^4$ 的总样本复杂度为 $\sum_{t=1}^T dt^4 \approx dT^5/5$，实际资源需求大
- 能否推广到 Markov 博弈和时变博弈是重要的后续方向
- 加权版本 A2L 可用非均匀权重 $\alpha_t$（如线性权重 $\alpha_t = t$），对 regret matching 等算法可能有实际价值，但理论分析尚未展开
- 对抗鲁棒性的切换机制（检测遗憾超过阈值后切换到标准no-regret）在实践中的检测效率和延迟影响未深入讨论

## 相关工作与启发
- **vs AOG (Cai et al. 2023)**：AOG 用加速 optimistic gradient 获得 $O(\text{poly}(d)/T)$ last-iterate 率，A2L-OMWU 维度依赖指数级改善至 $\log d$。Bandit 下 AOG 因 potential function 对误差敏感只获得 $T^{-1/8}$，而 A2L 通过将问题转化为遗憾分析避免了这一困难
- **vs 正则化方法 (Cen et al. 2021/2024)**：需要预知 $T$ 调正则化参数，仅保证最后一步（非 anytime），率为 $\tilde{O}(\log T/T)$ 有额外 $\log$ 因子。而 A2L-OMWU 无需正则化、不需预知 $T$、消除 $\log T$ 因子
- **vs OMWU 原始版本**：Cai & Zheng (2024) 证明 OMWU 的 last-iterate 可任意慢（$\Omega(1)$ 下界），A2L 规约将这一"不可能"结果巧妙绕过
- **vs 递减正则化 (Park et al. 2023)**：递减正则化给出 $\tilde{O}(T^{-1/4})$ anytime 率，远慢于 A2L-OMWU 的 $O(\log d/T)$
- **vs Cutkosky (2019) AO2B reduction**：AO2B 针对静态优化问题，不保证新旧算法迭代等价；A2L 针对动态博弈，严格保证 last-iterate = average-iterate
- **Fisher 市场的 PRD 应用**：展示了 A2L 不限于 Assumption 1，只要反馈满足可恢复结构（如 PRD 的比例结构）即可适用，获得 $O(1/T)$ last-iterate 率改善 Cheung (2025) 的仅 average-iterate 结果

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 黑箱规约思路极为优雅，首次将 last-iterate 问题系统性地归约为 average-iterate
- 实验充分度: ⭐⭐⭐ 纯理论工作无实验，但理论结果已非常充分
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰，核心思路一目了然，证明sketch信息量大
- 价值: ⭐⭐⭐⭐⭐ 解决了博弈学习的核心开放问题，结果同时改善梯度和bandit两种设置

<!-- RELATED:START -->

## 相关论文

- [Improved Last-Iterate Convergence of Shuffling Gradient Methods for Nonsmooth Convex Optimization](../../ICML2025/optimization/improved_last-iterate_convergence_of_shuffling_gradient_methods_for_nonsmooth_co.md)
- [Sharper Convergence Rates for Nonconvex Optimisation via Reduction Mappings](sharper_convergence_rates_for_nonconvex_optimisation_via_reduction_mappings.md)
- [Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner](purifying_shampoo_investigating_shampoos_heuristics_by_decomposing_its_precondit.md)
- [Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [Optimistic Online-to-Batch Conversions for Accelerated Convergence and Universality](optimistic_online-to-batch_conversions_for_accelerated_convergence_and_universal.md)

<!-- RELATED:END -->
