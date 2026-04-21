---
title: >-
  [论文解读] Dynamic Regret Reduces to Kernelized Static Regret
description: >-
  [NeurIPS 2025][dynamic regret] 将动态遗憾最小化问题重新建模为再生核希尔伯特空间(RKHS)中的静态遗憾问题，通过精心设计平移不变核实现最优路径长度依赖 $\widetilde{\mathcal{O}}(\sqrt{MP_TT})$，且天然不需要时间范围先验知识。
tags:
  - NeurIPS 2025
  - dynamic regret
  - online convex optimization
  - RKHS
  - kernel methods
  - path-length
---

# Dynamic Regret Reduces to Kernelized Static Regret

**会议**: NeurIPS 2025  
**arXiv**: [2507.05478](https://arxiv.org/abs/2507.05478)  
**代码**: 无  
**领域**: reinforcement_learning / 在线学习  
**关键词**: dynamic regret, online convex optimization, RKHS, kernel methods, path-length

## 一句话总结
将动态遗憾最小化问题重新建模为再生核希尔伯特空间(RKHS)中的静态遗憾问题，通过精心设计平移不变核实现最优路径长度依赖 $\widetilde{\mathcal{O}}(\sqrt{MP_TT})$，且天然不需要时间范围先验知识。

## 研究背景与动机

**领域现状**：在线凸优化(OCO)中，学习者每轮选择 $w_t \in \mathcal{W}$，环境揭示凸损失 $\ell_t$。经典目标是最小化相对于固定基准的**静态遗憾**(static regret)。当环境非平稳时，需要最小化相对于时变基准 $u_1,\dots,u_T$ 的**动态遗憾**(dynamic regret)。

**现有痛点**：现有动态遗憾算法面临三个问题：(a) 最优 $\sqrt{P_T}$ 依赖需要时间范围 $T$ 的先验知识，或借助 doubling trick 但实际效果差；(b) 已有的动态到静态的 reduction 仅对线性损失有效；(c) 无法利用损失函数的曲率获得更紧的 bound。

**核心矛盾**：将比较器序列叠成高维向量的方法（Jacobsen & Orabona 2024）受限于有限维特征空间和线性损失假设，无法自然地处理无限维设计和非线性损失。

**本文目标**：(a) 不依赖 $T$ 先验实现最优 $\sqrt{P_T}$ 依赖；(b) 将 reduction 推广到任意凸损失；(c) 利用损失曲率获得 $\mathcal{O}(\|u\|_{\mathcal{H}}^2 + d_{\text{eff}}(\lambda)\ln T)$ 的 bound。

**切入角度**：观察到与任意比较器序列竞争等价于与一个固定函数 $u(\cdot)$ 竞争，只要 $u(t) = u_t$。将这个函数嵌入 RKHS 中，就把动态遗憾降维为函数空间中的静态遗憾。

**核心 idea**：动态遗憾 = RKHS 中的核化静态遗憾，通过设计合适的核函数控制 RKHS 范数与路径长度的关系。

## 方法详解

### 整体框架
输入为在线凸损失序列 $\ell_1,\dots,\ell_T$，输出为学习者每轮决策 $w_t$。核心 pipeline：(1) 选择 RKHS $\mathcal{H}$ 及其特征映射 $\phi$；(2) 将原始损失 $\ell_t$ 转化为算子空间 $L(\mathcal{H}, \mathcal{W})$ 上的辅助损失 $\tilde{\ell}_t(W) = \ell_t(W\phi(t))$；(3) 在算子空间上运行任意静态遗憾算法 $\mathcal{A}$；(4) 每轮通过 $w_t = W_t\phi(t)$ 得到决策。

### 关键设计

1. **动态-静态等价定理 (Theorem 1)**:

    - 功能：证明动态遗憾与核化静态遗憾严格相等
    - 核心思路：对任意满足 $u_t = U\phi(t)$ 的算子 $U$，有 $R_T(u_1,\dots,u_T) = \sum_t (\tilde{\ell}_t(W_t) - \tilde{\ell}_t(U)) = \tilde{R}_T(U)$。这是一个恒等式而非上界，对任意损失成立
    - 设计动机：之前的 reduction 只对线性损失有效（因为需要将梯度分解为 $g_t \otimes \phi(t)$），本文的 reduction 直接操作损失函数本身，保留了曲率信息

2. **核化 FTRL 与 Kernel Trick (Example 1)**:

    - 功能：展示如何在无限维空间中高效计算
    - 核心思路：利用再生性质，FTRL 的决策可表示为 $w_t = -\frac{(\Psi_t^*)'}{\|\theta_t\|_{\text{HS}}} \sum_{s=1}^{t-1} k(s,t)g_s$，只需核函数值 $k(s,t)$ 而无需显式计算 $\phi(t)$
    - 设计动机：虽然 RKHS 是无限维空间，但核技巧使得算法实际可计算

3. **平移不变核的精心设计 (Proposition 2)**:

    - 功能：构造谱密度 $Q(\omega)$ 使 RKHS 范数与路径长度成正比
    - 核心思路：希望 $Q(\omega) \approx 1/|\omega|$（因为由 Parseval 恒等式可得 $\|u\|_{\mathcal{H}}^2 \approx \sup_t|u(t)| \cdot \|\nabla u\|_{L^1}$），但 $1/|\omega|$ 不可积。通过添加微量正则化得到具体谱密度 $Q(\omega)$
    - 关键性质：$k(t,t) \leq 8\pi^2$ 对所有 $t$ 成立（不依赖 $T$），且 $\|f\|_{\mathcal{H}}^2 = \widetilde{\mathcal{O}}(\|\nabla f\|_{L^1}\|f - \nabla^2 f\|_{L^\infty})$
    - 设计动机：标准核（如高斯核、Matérn 核）的 RKHS 范数包含 $\|u\|_{L^2}^2 = \Omega(T)$ 项导致次优 bound；样条核虽然 $\|u\|_{\mathcal{H}}^2 = \|\nabla u\|_{L^2}^2$ 但 $k(t,t) = t$ 也是次优的

4. **连续路径长度到离散路径长度的桥接 (Theorem 4)**:

    - 功能：证明存在 RKHS 函数 $u$ 满足 $u(t)=u_t$ 且 $\|\nabla u\|_{L^1} = \mathcal{O}(P_T)$
    - 核心思路：构造有限支撑的光滑插值函数，利用 RKHS 包含有界导数的有限支撑函数的条件
    - 设计动机：连续路径长度和离散路径长度可能差很大（函数在插值点之间可能剧烈振荡），需要证明存在"好的"插值

### 损失函数 / 训练策略
对于线性损失：使用 parameter-free 算法，结合设计的核函数直接得到最优 bound $\widetilde{\mathcal{O}}(\sqrt{(M^2+MP_T)\sum_t\|g_t\|^2})$。

对于曲线损失：
- **exp-concave 损失**：reduction 保留 exp-concavity，可直接应用 Kernelized ONS (KONS)，得到 $R_T = \mathcal{O}(\lambda\|U\|_{\text{HS}}^2 + d_{\text{eff}}(\lambda)\ln T / \beta)$
- **强凸损失**：利用特征协方差矩阵的加权范数获得类似对数 bound
- **在线线性回归**：恢复核化 Vovk-Azoury-Warmuth 预测器的标准 bound

## 实验关键数据

### 主要理论结果对比

| 损失类型 | 已有最优 bound | 本文 bound | 改进 |
|----------|---------------|-----------|------|
| 线性 (bounded domain) | $\mathcal{O}(\sqrt{DP_TT})$ 需知 $T$ | $\widetilde{\mathcal{O}}(\sqrt{(M^2+MP_T)T})$ 不需知 $T$ | 首个不需 $T$ 先验的最优 bound |
| 线性 (scale-free) | 无已有结果 | $\widetilde{\mathcal{O}}(\sqrt{(M^2+MP_T)\sum_t\|g_t\|^2})$ | 首个 scale-free 动态遗憾 |
| exp-concave | 无动态遗憾结果 | $\mathcal{O}(\lambda\|U\|_{\text{HS}}^2 + d_{\text{eff}}(\lambda)\ln T)$ | 首次利用曲率获得对数 bound |
| 强凸 | 无动态遗憾结果 | 类似对数 bound | 同上 |

### 与先前 reduction 的对比

| 特性 | Jacobsen & Orabona (2024) | 本文 |
|------|--------------------------|------|
| 适用损失 | 仅线性损失 | 任意凸损失 |
| 特征维度 | 有限维 ($\mathbb{R}^{dT}$) | 无限维 RKHS |
| 需要 $T$ 先验 | 是 | 否 |
| Scale-free | 否 | 是 |
| 利用损失曲率 | 否 | 是 |
| 方向自适应 | 否 | 是 |

### 关键发现
- 选择 Dirac 核可以精确恢复 Jacobsen & Orabona 的 reduction，说明他们的方法是本框架的特例
- 核函数 $Q(\omega)$ 的设计是技术核心：标准核（高斯、Matérn）都给出次优 bound，需要非标准的"接近 $1/|\omega|$ 但可积"的谱密度
- 方向自适应保证 $\widetilde{\mathcal{O}}(\sqrt{d_{\text{eff}}(\lambda)(\|u\|_{\mathcal{H}^d}^2 + \sum_t \langle g_t, u_t \rangle^2)})$ 在某些方向梯度小时可以更紧

## 亮点与洞察
- **视角转换的优雅性**：将比较器序列视为函数的采样值这一观察极其自然，但打开了整个 RKHS 工具箱的大门。类似"升维解题"的思路——在更高维空间中问题反而更简单
- **Horizon independence**：这是第一个不需要 $T$ 先验知识就能实现最优 $\sqrt{P_T}$ 依赖的动态遗憾算法，避免了 doubling trick 的实际性能损失
- **统一框架**：一个框架同时处理线性、exp-concave、强凸损失，且每种情况都给出（接近）最优保证。这种统一性暗示了动态遗憾与核方法之间的深层联系
- 核设计的具体 trick（$Q(\omega)$ 的构造）可能启发信号处理和非参数统计中类似的权衡问题

## 局限与展望
- **计算复杂度**：朴素实现需要 $\mathcal{O}(t)$ 时间和内存，虽然可用核近似方法加速但未给出实验
- **纯理论贡献**：论文没有任何实验验证，不清楚在实际非平稳在线学习场景中的表现
- **对数因子**：所有 bound 都隐藏了 polylogarithmic 因子，能否去除？
- **核选择的自动化**：当前核的选择依赖对问题结构的先验知识，能否自动选择最优核？

## 相关工作与启发
- **vs Zhang et al. (2023, 信号处理方法)**：他们将比较器序列叠为高维信号用字典分解，本文将其嵌入 RKHS。本文更通用（不限线性损失），但他们的方法计算上可能更高效
- **vs Jacobsen & Orabona (2024)**：作为本文框架的特例（Dirac 核），但仅处理线性损失和有限维特征
- **vs Online Newton Step**：本文的 RKHS reduction 天然兼容 ONS，首次给出动态遗憾的对数 bound

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 视角转换极其优雅，将动态遗憾问题全新地重建为核方法问题
- 实验充分度: ⭐⭐ 纯理论论文，缺乏实验验证
- 写作质量: ⭐⭐⭐⭐ 数学严谨，结构清晰，直觉解释到位
- 价值: ⭐⭐⭐⭐ 建立了动态遗憾与核方法的深层联系，统一了多种损失类型下的最优 bound

<!-- RELATED:START -->

## 相关论文

- [Simultaneous Swap Regret Minimization via KL-Calibration](simultaneous_swap_regret_minimization_via_kl-calibration.md)
- [On the Dynamic Regret of Following the Regularized Leader: Optimism with History Pruning](../../ICML2025/reinforcement_learning/on_the_dynamic_regret_of_following_the_regularized_leader_optimism_with_history_.md)
- [Improved Regret Bounds for GP-UCB in Bayesian Optimization](improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability](../../ICML2025/reinforcement_learning/non-stationary_online_learning_for_curved_losses_improved_dynamic_regret_via_mix.md)
- [Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)

<!-- RELATED:END -->
