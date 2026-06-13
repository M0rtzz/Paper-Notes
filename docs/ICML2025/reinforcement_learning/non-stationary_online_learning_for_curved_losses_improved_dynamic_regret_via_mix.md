---
title: >-
  [论文解读] Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability
description: >-
  [ICML2025][强化学习][动态遗憾] 利用 mixability（可混合性）概念替代传统 KKT 分析，提出基于指数权重+fixed-share更新的连续空间在线学习框架，将弯曲损失函数（squared/logistic loss）的动态遗憾中对维度 $d$ 的依赖从 $O(d^{10/3})$ 大幅改进至 $O(d)$。
tags:
  - "ICML2025"
  - "强化学习"
  - "动态遗憾"
  - "在线凸优化"
  - "mixability"
  - "exp-concavity"
  - "非平稳在线学习"
  - "指数权重方法"
  - "fixed-share"
---

# Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability

**会议**: ICML2025  
**arXiv**: [2506.10616](https://arxiv.org/abs/2506.10616)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 动态遗憾, 在线凸优化, mixability, exp-concavity, 非平稳在线学习, 指数权重方法, fixed-share

## 一句话总结

利用 mixability（可混合性）概念替代传统 KKT 分析，提出基于指数权重+fixed-share更新的连续空间在线学习框架，将弯曲损失函数（squared/logistic loss）的动态遗憾中对维度 $d$ 的依赖从 $O(d^{10/3})$ 大幅改进至 $O(d)$。

## 研究背景与动机

**非平稳在线学习**是近年的热门方向。标准的在线凸优化（OCO）框架中，学习器在 $T$ 轮博弈中与环境交互，每轮提交预测 $\mathbf{w}_t$，环境生成凸损失函数 $f_t$。在非平稳环境中，通常用**动态遗憾**（dynamic regret）衡量算法性能：

$$\text{D-Reg}_T = \sum_{t=1}^T f_t(\mathbf{w}_t) - \sum_{t=1}^T f_t(\mathbf{u}_t)$$

其中 $\{\mathbf{u}_t\}$ 是时变的比较器序列，$P_T = \sum_{t=2}^T \|\mathbf{u}_t - \mathbf{u}_{t-1}\|_2$ 为路径长度，反映环境非平稳程度。

**现有问题**：

- 一般凸损失的动态遗憾已被充分研究，最优界为 $O(\sqrt{T(1+P_T)})$
- 对于具有**更强曲率**的损失（如 squared loss、logistic loss），即 exp-concave 损失，Baby & Wang (2021, 2022a) 通过 KKT 条件分析获得了 $\widetilde{O}(d^{10/3} T^{1/3} P_T^{2/3})$ 的动态遗憾
- 但已知下界为 $\Omega(d^{1/3} T^{1/3} P_T^{2/3})$，维度依赖 $d^{10/3}$ vs $d^{1/3}$ 存在巨大 gap
- KKT 分析技术复杂，难以推广到更一般的场景

## 方法详解

### 核心思想：从 exp-concavity 到 mixability

本文的关键创新是用 **mixability**（可混合性）替代 exp-concavity 来刻画损失曲率。

**Exp-concavity**：函数 $f$ 是 $\eta$-exp-concave，当 $e^{-\eta f(\mathbf{w})}$ 是凹函数，即对任意分布 $P$：

$$f(\mathbb{E}_P[\mathbf{w}]) \leq -\frac{1}{\eta} \ln \mathbb{E}_{\mathbf{w}\sim P}[e^{-\eta f(\mathbf{w})}]$$

**Mixability（更弱条件）**：仅要求**存在**某个 $\mathbf{w}_{\text{mix}}$（不一定是均值）满足上述不等式。Exp-concavity $\Rightarrow$ mixability，反之不成立。

重要实例：

| 损失函数 | mixability 系数 | exp-concavity 系数 |
|---------|----------------|-------------------|
| Squared loss $(z-y)^2$ | $\frac{1}{2B^2}$（全空间） | $\frac{1}{2(B+D)^2}$（有界域） |
| Logistic loss $\log(1+e^{-yz})$ | $1$（全空间） | $e^{-D}$（有界域） |

Mixability 在全空间上可获得更大的曲率系数 $\eta$，这是改进维度依赖的关键。

### 算法框架：Fixed-share for Continuous Space（Algorithm 1）

算法维护参数空间 $\mathbb{R}^d$ 上的**分布** $P_t$（而非单个参数），包含两步更新：

**Step 1 — 指数权重更新**：

$$\widetilde{P}_{t+1}(\mathbf{u}) \propto P_t(\mathbf{u}) \exp(-\eta f_t(\mathbf{u}))$$

**Step 2 — Fixed-share 混合**：

$$P_{t+1}(\mathbf{u}) = (1-\mu) \widetilde{P}_{t+1}(\mathbf{u}) + \mu \mathcal{N}(\mathbf{w}_0, I_d)$$

其中 $\mu = 1/T$。每轮将一小部分概率质量"重置"为初始高斯分布，使算法能适应环境变化。

**预测**通过 mixability 条件构造 $z_t$，使 $\ell(z_t, y_t) \leq m_t(P_t)$（mix loss）。

### 三段式分析框架

动态遗憾被分解为三项：

$$\text{D-Reg}_T \leq \underbrace{\sum \ell(z_t,y_t) - \sum m_t(P_t)}_{\text{(A) mixability gap} \leq 0} + \underbrace{\sum m_t(P_t) - \sum \mathbb{E}_{Q_t}[f_t]}_{\text{(B) mixability regret}} + \underbrace{\sum \mathbb{E}_{Q_t}[f_t] - \sum f_t(\mathbf{u}_t)}_{\text{(C) comparator gap}}$$

- **(A)** 由 mixability 条件保证非正
- **(B)** 引入分析用高斯分布 $Q_t = \mathcal{N}(\mathbf{u}_t, \sigma^2 I_d)$，通过 KL 散度界定
- **(C)** 由损失平滑性 + 高斯期望得到 $\leq \beta d \sigma^2 T / 2$

选择 $\sigma = \Theta(P_T^{1/3} T^{-1/3})$ 平衡 (B)(C)，得到最终界。

### 等价实现：Follow-the-Leading-History（Algorithm 2）

Fixed-share 等价于一个**两层在线集成**：

- **基学习器层**：每轮新增一个基学习器 $\mathcal{B}_i$，从高斯初始化，按指数权重更新
- **元学习器层**：用 Hedge 方法按 mix loss 分配权重，新学习器权重为 $\mu$

### 扩展到一般 OCO（Algorithm 3）

对于一般 exp-concave 损失（不一定在全空间 mixable），引入两个改进：

1. **代理损失**：$\widetilde{f}_t(\mathbf{w}) = \mathbf{g}_t^\top(\mathbf{w}-\mathbf{w}_t) + \frac{\gamma}{2}(\mathbf{g}_t^\top(\mathbf{w}-\mathbf{w}_t))^2$，消除平滑性假设
2. **投影步骤**：将分布投影到约束集 $\mathscr{M}$（均值在 $\mathcal{W}$ 内、协方差有界的高斯混合族），确保 proper learning

## 实验关键数据

本文为**纯理论贡献**，无实验部分。主要理论结果如下：

| 损失类型 | 方法 | 动态遗憾界 | Proper | 复杂度 |
|---------|------|-----------|--------|-------|
| 1-d squared | Baby & Wang 2021 | $\widetilde{O}(T^{1/3}P_T^{2/3})$ | ✓ | $O(T)$ |
| 1-d squared | **本文 Corollary 1** | $\widetilde{O}(T^{1/3}P_T^{2/3})$ | ✓ | $O(T)$ |
| Least-squares | Baby & Wang 2022b | $\widetilde{O}(d^{10/3}T^{1/3}P_T^{2/3})$ | ✓ | $O(T)$ |
| Least-squares | **本文 Corollary 2** | $\widetilde{O}(d \cdot T^{1/3}P_T^{2/3})$ | ✗ | $O(T)$ |
| Logistic | Baby et al. 2023 | $\widetilde{O}(d^{10/3}T^{1/3}P_T^{2/3})$ | ✓ | $O(T)$ |
| Logistic | **本文 Corollary 3** | $\widetilde{O}(d \cdot T^{1/3}P_T^{2/3})$ | ✗ | poly$(T)$ |
| General exp-concave | Baby & Wang 2022a | $\widetilde{O}(d^{10/3}T^{1/3}P_T^{2/3})$ | ✗* | $O(T)$ |
| General exp-concave | **本文 Theorem 3** | $\widetilde{O}(d \cdot T^{1/3}P_T^{2/3})$ | ✓ | — |

**关键发现**：维度依赖从 $d^{10/3}$ 改进至 $d$，缩小了与下界 $\Omega(d^{1/3})$ 之间的 gap。

## 亮点与洞察

1. **概念性突破**：首次将 mixability 引入动态遗憾分析，提供了比 KKT 分析更简洁、更具一般性的理论工具
2. **维度改进显著**：$d^{10/3} \to d$，在高维问题中差异巨大（如 $d=100$ 时差 $\sim 10^4$ 倍）
3. **分析框架优雅**：三段分解（mixability gap / regret / comparator gap）给出了清晰的遗憾来源刻画
4. **无需先验知识**：算法不需要提前知道路径长度 $P_T$，通过维护分布自动适应非平稳性
5. **Algorithm 3 的 proper learning**：通过代理损失+投影步骤在一般 exp-concave 场景下实现 proper 预测，解决了已有方法无法在任意凸域上实现 proper 的问题
6. **等价性定理（Theorem 2）**：揭示了 fixed-share 与 FLH 型算法在连续空间中的深层联系

## 局限与展望

1. **Improper learning**：Algorithm 1 在最小二乘和逻辑回归中是 improper 的（预测可能超出线性假设空间），不如 Baby & Wang (2022b) 和 Baby et al. (2023) 的 proper 方法
2. **计算效率**：logistic loss 缺乏闭式更新，需要采样近似；一般 OCO 的投影步骤计算开销不明确
3. **仍有维度 gap**：最优界为 $\widetilde{O}(d \cdot T^{1/3} P_T^{2/3})$，下界为 $\Omega(d^{1/3})$，中间差距 $d$ vs $d^{1/3}$ 尚未完全弥合
4. **无法加速到 $O(\log T)$**：KKT 方法可通过 geometric covering 加速到每轮 $O(\log T)$，但固定指数权重框架下尚不清楚如何实现
5. **纯理论工作**：缺乏实验验证，未展示在实际非平稳学习任务上的表现

## 相关工作与启发

- **Baby & Wang (2021, 2022a, b), Baby et al. (2023)**：exp-concave 损失动态遗憾的 KKT 分析路线，是本文直接改进的基准
- **Vovk (2001), van Erven et al. (2012)**：mixability 概念的奠基工作，本文将其从静态遗憾推广到动态遗憾
- **Cesa-Bianchi et al. (2012b)**：有限专家环境下 fixed-share 的动态遗憾分析，本文推广到连续空间
- **Hazan & Seshadhri (2009)**：FLH 算法，本文证明了其与 fixed-share 在连续空间的等价性
- **Zhang et al. (2018), Zhao et al. (2024)**：凸损失动态遗憾最优方法，本文框架可视为弯曲损失的对应推广

**启发**：mixability 作为比 exp-concavity 更弱的条件却能获得更好的维度依赖，提示在线学习中"合适的抽象层次"对获得更优理论保证的重要性。分布维护（而非点维护）+ fixed-share 的范式或许可推广到其他非平稳学习问题。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (首次用 mixability 分析动态遗憾，概念性贡献显著)
- 实验充分度: ⭐⭐ (纯理论，无实验)
- 写作质量: ⭐⭐⭐⭐⭐ (分析清晰，证明sketch完整，对比详尽)
- 价值: ⭐⭐⭐⭐ (大幅缩小理论gap，但计算瓶颈和实际应用价值待验证)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] On the Dynamic Regret of Following the Regularized Leader: Optimism with History Pruning](on_the_dynamic_regret_of_following_the_regularized_leader_optimism_with_history_.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](../../NeurIPS2025/reinforcement_learning/forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[NeurIPS 2025\] Establishing Linear Surrogate Regret Bounds for Convex Smooth Losses via Convolutional Fenchel–Young Losses](../../NeurIPS2025/reinforcement_learning/establishing_linear_surrogate_regret_bounds_for_convex_smooth_losses_via_convolu.md)
- [\[NeurIPS 2025\] Dynamic Regret Reduces to Kernelized Static Regret](../../NeurIPS2025/reinforcement_learning/dynamic_regret_reduces_to_kernelized_static_regret.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](../../NeurIPS2025/reinforcement_learning/improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)

</div>

<!-- RELATED:END -->
