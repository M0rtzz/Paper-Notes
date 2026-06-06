---
title: >-
  [论文解读] Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning
description: >-
  [ICML 2026][强化学习][两时间尺度] 本文首次在 Markov 噪声且不依赖任何投影算子的条件下，建立了一般两时间尺度随机逼近 (SA) 的稳定性与几乎必然收敛性，并据此给出离策略线性函数逼近下 TDC($\lambda$) 算法的首个几乎必然收敛结果。
tags:
  - "ICML 2026"
  - "强化学习"
  - "两时间尺度"
  - "随机逼近"
  - "ODE 方法"
  - "Markov 噪声"
  - "TDC"
---

# Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning

**会议**: ICML 2026  
**arXiv**: [2605.31172](https://arxiv.org/abs/2605.31172)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 两时间尺度, 随机逼近, ODE 方法, Markov 噪声, TDC

## 一句话总结
本文首次在 Markov 噪声且不依赖任何投影算子的条件下，建立了一般两时间尺度随机逼近 (SA) 的稳定性与几乎必然收敛性，并据此给出离策略线性函数逼近下 TDC($\lambda$) 算法的首个几乎必然收敛结果。

## 研究背景与动机

**领域现状**：强化学习中 actor-critic、TDC、目标网络等大量算法都属于两时间尺度 SA：快慢两组参数分别以大、小步长更新，渐近上快尺度仿佛在静态慢参数下收敛，从而形成"嵌套循环"的随机版本。理论分析依赖 Borkar 提出的 ODE 方法 (Borkar 1997；Borkar & Meyn 2000)，其核心前提是迭代有界 (稳定性)，进而由 ODE 轨迹刻画离散迭代的渐近行为。

**现有痛点**：经典两时间尺度收敛结论几乎都假设 (i) 噪声 i.i.d.，(ii) 稳定性预先成立。然而 RL 中噪声序列是 Markov 链 (状态-动作-资格迹联合)，且资格迹在离策略情况下可无界，使得 i.i.d. 与有界噪声假设均不再适用。先前工作或退而强制添加投影 (Yu 2017；Panda & Bhatnagar 2025) 来人为保证有界，或假设两尺度参数解耦、噪声落在紧空间 (Karmakar & Bhatnagar 2021)，从而都无法直接覆盖带资格迹的 TDC($\lambda$)。

**核心矛盾**：在两尺度耦合动力学中，如何只用快慢步长之比 $\beta(n)/\alpha(n)\to 0$ 与温和的 Lipschitz 假设，把快尺度迭代的范数与慢尺度迭代的范数联系起来，从而在 Markov 噪声、非紧噪声空间、不投影、不预设稳定性的"全开"设定下证明几乎必然收敛？这正是文献多年未填的缺口 (见正文 Table 1)。

**本文目标**：在统一框架内同时去除上述全部限制，覆盖 TDC($\lambda$)、actor-critic 等真实算法。

**切入角度**：作者注意到 Lakshminarayanan & Bhatnagar (2017) 用"重缩放迭代 + ODE@$\infty$"在 i.i.d. 噪声下证两尺度稳定性，而 Liu et al. (2025b) 在单尺度 Markov 噪声下完成了类似论证。要把两者拼起来，关键障碍是：现有两尺度证明都需"同步 (same-step) 地"用慢参数 $\|y_n\|$ 控制快参数 $\|x_n\|$，这种同步界在重缩放过程中天然失效。

**核心 idea**：放宽为"用迄今为止最大的慢参数 $y_n^{\max}$ 控制当前快参数 $x_n$"——即把 $\|x_n\|\le K(1+\|y_n^{\max}\|)$ 作为新的桥接不等式 (Lemma 3.1)。这一"running max"提法恰好和 Lakshminarayanan 的"维持单调的缩放因子"对齐，使两套工具得以无缝拼接。

## 方法详解

本文不是算法论文，而是 SA 理论文章，"方法"指的是一套围绕新 Lemma 3.1 展开的证明体系，最终把结论应用到 RL 算法 TDC($\lambda$)。

### 整体框架

研究对象是一般两时间尺度递推 (快尺度 $x\in\mathbb{R}^{d_1}$，慢尺度 $y\in\mathbb{R}^{d_2}$)：

$$
x_{n+1}=x_n+\alpha(n)\,H(x_n,y_n,W_{n+1}),\quad y_{n+1}=y_n+\beta(n)\,G(x_n,y_n,W_{n+1}),
$$

其中噪声 $\{W_n\}$ 是空间 $\mathcal{W}$ (允许非紧、不可数) 上的 Markov 链，步长满足 $\lim_n \beta(n)/\alpha(n)=0$。论文的证明 roadmap 是：① 在快尺度证明 Lemma 3.1 (running-max 桥接界)，② 在慢尺度证明 Theorem 3.2 (整体稳定性)，③ 据此证明 Theorem 3.3 (几乎必然收敛)，④ 验证 TDC($\lambda$) 满足全部假设，得 Theorem 7.2。

### 关键设计

1. **Running-max 桥接 (Lemma 3.1)**:

    - 功能：把两个时间尺度的迭代范数串起来，证明存在样本路径相关常数 $K$ 使得 $\|x_n\|\le K(1+\|y_n^{\max}\|)$ a.s.，其中 $y_n^{\max}=\arg\max_{i\le n}\|y_i\|$ 对应的慢迭代。
    - 核心思路：把时间轴 $[0,\infty)$ 按快步长切成长度近似 $T$ 的区间 $[T_n,T_{n+1})$，在每段上用 $r_n\doteq\max\{1,r_{n-1},\|\bar z(T_n)\|\}$ 单调放缩 $z=(x,y)$ 得到 $\tilde z_n(t)=\bar z(T_n+t)/r_n$；选 Arzelà–Ascoli 子列得到极限轨迹 $z^{\lim}$ 满足 ODE@$\infty$ $\dot x=h_\infty(x,y), \dot y=0$；用 Lemma 4.7 证明只要 $\|\bar x(T_n)\|>C_1(1+\|\bar y(T_n)\|)$ 就有 $\|\bar z(T_{n+1})\|\le\|\bar z(T_n)\|$ (即 $r_{n+1}=r_n$)，归纳出 $\|\bar z(T_n)\|\le C_1C_2(\max_{m\le n}\|\bar y(T_m)\|+1)$，再用 Lemma 4.9 把界推广到所有 $n$。
    - 设计动机：与先前工作 (Kushner & Yin 2003；Mokkadem & Pelletier 2006；Dalal et al. 2018；Yaji & Bhatnagar 2020；Zeng et al. 2024) 不同，先前证明都尝试用同步慢参数 $\|y_n\|$ 控制 $\|x_n\|$；当噪声非 i.i.d. 时，这种同步界无法成立。Running max 比当前值更"宽松"，正是能容纳 Lakshminarayanan 的单调缩放方案与 Liu 的 Markov 噪声平均技巧的最弱条件。

2. **慢尺度稳定性证明 (Theorem 3.2)**:

    - 功能：在 Lemma 3.1 基础上把"整体迭代有界" $\sup_n\|z_n\|<\infty$ a.s. 由反证得出。
    - 核心思路：在慢尺度上重做一遍重缩放，但缩放因子改为 $r_n\doteq\max\{1,\|z_{m(T_n)}^{\max}\|\}$，使其至少与历史最大迭代同阶；随后证明 Lemma 5.3——即使在慢尺度上观察，快迭代仍能近似追上 $\lambda_\infty(\tilde y_n(t))$，从而把"快尺度等价于已收敛"这一启发式严格化。结合 ODE@$\infty$ 的零吸引子性，假设 $\sup_n r_n=\infty$ 会与极限 ODE 收敛到 0 矛盾，故 $r_n$ 有界，稳定性成立。
    - 设计动机：直接套 Liu et al. (2025b) 的单尺度论证会失败，因为快迭代步长大、增长更快；通过把缩放因子定义成"历史最大 $\|z\|$"，可以保证快慢两个尺度的缩放速率自动同步，从而让 Markov 噪声下的平均化估计 (Kushner & Yin 2003 的技术) 仍然适用。

3. **TDC($\lambda$) 离策略收敛应用 (Theorem 7.2)**:

    - 功能：给出第一个 (无投影、线性函数逼近、离策略、带资格迹) TDC 几乎必然收敛证明。算法定义为 $e_t=\lambda\gamma\rho_{t-1}e_{t-1}+\phi_t$，$\delta_t=R_{t+1}+\gamma\phi_{t+1}^\top\theta_t-\phi_t^\top\theta_t$，$\nu_{t+1}=\nu_t+\alpha_t(\rho_t\delta_t e_t-\phi_t\phi_t^\top\nu_t)$，$\theta_{t+1}=\theta_t+\beta_t(\rho_t\delta_t e_t-\rho_t(1-\lambda)\gamma\phi_{t+1}e_t^\top\nu_t)$。
    - 核心思路：把 $\nu_t$ 视作快尺度、$\theta_t$ 视作慢尺度，状态扩展为 $(S_t,A_t,e_t)$；由于离策略下重要性采样比 $\rho_t$ 的连乘使 $e_t$ 可无界，$\mathcal{W}$ 取在非紧空间上。验证 Appendix B 中 B.1–B.7 假设 (Markov 链遍历、步长条件、$H/G$ 的齐次极限、Lipschitz 性、平均化条件) 在 TDC($\lambda$) 上均自然成立，从而 Theorem 3.3 直接给出几乎必然收敛。
    - 设计动机：先前 Yu (2017) 必须额外加投影以强制有界，Panda & Bhatnagar (2025) 同样依赖投影；本文的非投影框架是第一个能"原样"刻画实践算法的工具。这也展示了 Lemma 3.1 的实际威力——它把"理论上不投影"从口号变为可验证条件。

### 损失函数 / 训练策略
理论论文，无训练。需要满足的关键假设包括：步长 $\sum\alpha=\sum\beta=\infty$、$\sum\alpha^2,\sum\beta^2<\infty$、$\lim\beta/\alpha=0$；$H,G$ 的齐次缩放极限 $h_\infty,g_\infty$ 存在且 Lipschitz；对应 ODE 有唯一全局渐近稳定平衡 $\lambda_\infty(y),0$；以及 Kushner & Yin 风格的长时平均正则性条件 (B.7)。

## 实验关键数据

### 主实验 (理论结果对比)

| 工作 | 单/双时间尺度 | 噪声 | 是否需投影 | 噪声空间 | 覆盖 TDC($\lambda$) |
|------|--------------|------|-----------|----------|----------------------|
| Borkar (2009) | 双 | i.i.d. | 否 | 紧 | 否 |
| Lakshminarayanan & Bhatnagar (2017) | 双 | i.i.d. | 否 | 紧 | 否 |
| Karmakar & Bhatnagar (2021) | 双 (解耦) | Markov | 否 | 紧 | 否 |
| Liu et al. (2025b) | 单 | Markov | 否 | 非紧 | — |
| Panda & Bhatnagar (2025) | 双 | Markov | **是** | 非紧 | 否 |
| **本文** | 双 (耦合) | Markov | 否 | 非紧 | **是** |

### 主要定理一览

| 结果 | 类型 | 关键陈述 |
|------|------|----------|
| Lemma 3.1 | 桥接界 | $\|x_n\|\le K(1+\|y_n^{\max}\|)$ a.s. |
| Theorem 3.2 | 稳定性 | $\sup_n\|z_n\|<\infty$ a.s. |
| Theorem 3.3 | 收敛 | $\|z_n-(\lambda(y^*),y^*)\|\to 0$ a.s. |
| Theorem 7.2 | RL 应用 | TDC($\lambda$) 在离策略线性逼近下 a.s. 收敛 |

### 关键发现
- 把"同步控制"换成"历史最大慢迭代控制"是把 Lakshminarayanan & Bhatnagar (2017) 与 Liu et al. (2025b) 拼起来的最弱条件，没有它则两套工具互不兼容。
- 论文同时指出 Chandak et al. (2025) 关于两尺度 Markov SA 几乎必然收敛的论证存疑：他们用"期望有界 $\Rightarrow$ 几乎必然有界"，但 $x_n=\sqrt n$ 概率 $1/n$、否则为 0 这一反例直接证伪 (第二 Borel–Cantelli)。这说明本文论证不仅补缺，而且纠错。
- 离策略 + 资格迹 + 线性逼近 (即 "deadly triad") 的可证收敛结果仍极少，本文给出首个完全无投影的方案，为后续 actor-critic 全分析铺路。

## 亮点与洞察
- "用 running max 桥接两个时间尺度"是一个极简却极锋利的技巧：与其要求 $\|x_n\|$ 被同步 $\|y_n\|$ 压住，不如允许它被历史最高的 $\|y\|$ 压住——这与重缩放方法天然单调的尺度因子高度一致，迁移到 actor-critic、target-network 等其他两尺度算法只需重新验证 $H,G$ 的齐次极限。
- 全文把 ODE@$\infty$ 框架推到"非紧 Markov 噪声 + 不投影"的边界，给出关于 Lemma 4.7 (生长抑制) 与 Lemma 5.3 (快尺度尾随平衡) 的明确技术接口，可直接复用在其他需要从离散迭代过渡到极限 ODE 的证明中。
- 文献对照表 (Table 1) 与对 Chandak et al. (2025) 反例的精确反驳，提醒该领域研究者：在 Markov 噪声下，"期望有界" 与 "几乎必然有界" 之间存在本质鸿沟，不能混用。

## 局限与展望
- 仅给出渐近 a.s. 收敛，没有有限时间速率；已有的两尺度 $L^2$ 速率 (Doan 2021a/b/2022；Chandak et al. 2025) 暂未与本文 a.s. 路径结果统一。
- 当前理论需 ODE 拥有唯一全局渐近稳定平衡——actor-critic 中策略 ODE 可能有多个不动点 (例如局部最优策略)，此时本文结论需要进一步弱化为"收敛到不变集"。
- 离散 Markov 链假设排除了连续状态空间 RL，需要把 Markov 噪声从一般状态空间推广 (Borkar 2009 第 6 章框架)。
- 形式化方面，Zhang (2025) 已在 Lean 中验证单尺度 Markov SA，把本文两尺度框架机器化验证是值得做的下一步。

## 相关工作与启发
- **vs Lakshminarayanan & Bhatnagar (2017)**: 同为两尺度稳定性 + ODE@$\infty$，但他们假设 i.i.d. 噪声且证明依赖快慢参数同步界；本文用 running max 把同步界换成更宽松的历史最大界，从而把噪声放到 Markov 与非紧空间。
- **vs Karmakar & Bhatnagar (2021)**: 他们假设两尺度参数解耦、噪声在紧空间且不需投影；本文允许耦合 + 非紧 Markov 噪声，因而能覆盖 TDC、actor-critic 这类真正耦合算法。
- **vs Panda & Bhatnagar (2025)**: 二者同样面对 Markov 噪声非紧空间，但 Panda 用额外投影强制有界；本文用 Lemma 3.1 替代投影，避免引入算法层修改、保持理论与实践一致。
- **vs Chandak et al. (2025)**: Chandak 给出两尺度 $L^2$ 速率并附带 a.s. 收敛声明，但该声明的关键步骤 (期望有界 ⇒ a.s. 有界) 有反例；本文不仅提供更完整的 a.s. 路径论证，还显式指出该错误。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning](stochastic_minimum-cost_reach-avoid_reinforcement_learning.md)
- [\[ICML 2026\] Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games](revisiting_regularized_policy_optimization_for_stable_and_efficient_reinforcemen.md)
- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](global_policy-space_response_oracles_for_two-player_zero-sum_games.md)
- [\[ICML 2026\] Convergence of Steepest Descent and Adam under Non-Uniform Smoothness](convergence_of_steepest_descent_and_adam_under_non-uniform_smoothness.md)
- [\[NeurIPS 2025\] Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/convergence_theorems_for_entropy-regularized_and_distributional_reinforcement_le.md)

</div>

<!-- RELATED:END -->
