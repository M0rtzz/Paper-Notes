---
title: >-
  [论文解读] Follow-the-Perturbed-Leader for Decoupled Bandits: Best-of-Both-Worlds and Practicality
description: >-
  [ICML 2026][优化/理论][Follow-the-Perturbed-Leader] 本文给 decoupled multi-armed bandit 问题（每轮分别选一个臂"利用"、一个臂"探索"）设计了首个 Best-of-Both-Worlds (BOBW) FTPL 算法：用 Pareto…
tags:
  - "ICML 2026"
  - "优化/理论"
  - "Follow-the-Perturbed-Leader"
  - "解耦 bandits"
  - "Best-of-Both-Worlds"
  - "Pareto 扰动"
  - "无凸优化"
---

# Follow-the-Perturbed-Leader for Decoupled Bandits: Best-of-Both-Worlds and Practicality

**会议**: ICML 2026  
**arXiv**: [2510.12152](https://arxiv.org/abs/2510.12152)  
**代码**: 未公开  
**领域**: 在线学习 / Bandits / 优化  
**关键词**: Follow-the-Perturbed-Leader, 解耦 bandits, Best-of-Both-Worlds, Pareto 扰动, 无凸优化

## 一句话总结
本文给 decoupled multi-armed bandit 问题（每轮分别选一个臂"利用"、一个臂"探索"）设计了首个 Best-of-Both-Worlds (BOBW) FTPL 算法：用 Pareto 扰动做利用、用一个仅依赖累积估损排名的代理量 $q_{t,i}$ 直接定义探索分布——既不需要 FTRL 的每步凸优化，也不需要 FTPL 标准做法中的几何重采样，对抗与随机两种环境下均达到与现有最优 FTRL 算法同阶的 $\mathcal{O}(\sqrt{KT})$ / $\mathcal{O}(K/\Delta_{\min})$ 后悔界，实测对 $K=2$ 比基线快约 130×。

## 研究背景与动机

**领域现状**：标准 multi-armed bandit (MAB) 每轮必须用同一个臂同时承担探索和利用。Avner 等 2012 提出 decoupled bandit：每轮选 $i_t$ 承担损失（不观测）、再选 $j_t$ 观测损失（不承担），探索和利用解耦。该设定起源于超宽带通信（感知与传输可用不同信道）、sim-to-real 机器人（仿真器探索、真实系统利用）、推荐系统（一小撮用户探索、其他用户利用）。Rouyer & Seldin 2020 用 Decoupled-Tsallis-INF 拿到了 BOBW 保证：对抗 $\mathcal{O}(\sqrt{KT})$、随机 $\mathcal{O}(K/\Delta_{\min})$，是当前理论 SOTA。

**现有痛点**：Decoupled-Tsallis-INF 属于 FTRL 框架，每轮要在 $K-1$ 维单纯形上解一个 Tsallis 熵正则化的凸优化得到利用概率向量 $w_t$，再用 $w_t$ 算探索概率 $p_t$。即便 Newton 迭代+热启动，凸优化对超宽带通信这种"每步必须毫秒级响应"的实际场景仍是负担，实测 $K=2$ 时已比 FTPL 慢约 130×。

**核心矛盾**：FTPL（一种用随机扰动代替正则化、纯 argmin 即出动作的轻量框架）天然便宜，但所有现有 decoupled bandit 算法都规定"探索概率 $p_{t,i}$ 必须由利用概率 $w_{t,i}$ 算出"——而 FTPL 一般没有 $w_t$ 的闭式表达，要通过几何重采样 (GR) 估它，单步代价 $\mathcal{O}(K^2)$ 或 $\mathcal{O}(K\log K)$；如果要估整个向量 $w_t$（不只是被选中的那一个）再灌进探索分布，代价升到 $\mathcal{O}(K^2\log K)$，反而比 FTRL 还慢，FTPL 的速度优势全失。

**本文目标**：在 decoupled 设定下设计一个 FTPL 算法，既保留 FTPL 的 $\mathcal{O}(K\log K)$ 量级单步开销，又拿到与 Decoupled-Tsallis-INF 同阶的 BOBW 后悔界。

**切入角度**：先前的探索分布"$p_t$ 是 $w_t$ 的函数"只是一个充分条件，不是必要条件——只要能找到一个**仅用当前已有量**（累积估损 $\hat L_t$、学习率 $\eta_t$、排名 $\sigma_{t,i}$）就能算出的代理向量 $q_t$，使得它在分析上能与 $w_t$ 建立紧的不等式联系，就可以绕开 $w_t$。

**核心 idea**：用 Pareto($\alpha$) 扰动做 FTPL 利用（对应 $\beta=1-1/\alpha$ 的 Tsallis 熵 FTRL），同时用 $q_{t,i}=\big(\min\{1/(1+\eta_t\hat{\underline L}_{t,i}),\,1/\sigma_{t,i}^{1/\alpha}\}\big)^{(\alpha+1)/2}$ 作为 $w_{t,i}^{1/2+1/(2\alpha)}$ 的可计算上界代理，再归一化得到探索分布 $p_{t,i}=q_{t,i}/\sum_j q_{t,j}$——纯排序+赋值，无凸优化、无重采样。

## 方法详解

### 整体框架
算法每轮做三件事：① 从 Pareto 分布 $\mathcal{P}_\alpha$ 采 $K$ 个扰动 $r_{t,i}$，挑利用臂 $i_t=\arg\min_i\{\hat L_{t,i}-r_{t,i}/\eta_t\}$；② 用排名直接算探索分布 $p_t$ 并采样 $j_t\sim p_t$，观测 $\ell_{t,j_t}$；③ 用 IW 估计更新累积损失 $\hat L_{t+1}=\hat L_t+\ell_{t,j_t}p_{t,j_t}^{-1}e_{j_t}$。整个过程没有任何凸优化、没有任何重采样，单步开销主导项是维护排序 $\mathcal{O}(K)$（增量 + 二分插入 $\mathcal{O}(\log K)$）。

注意 $p_t$ 与 $w_t$ 互不依赖：利用臂的选择只看 $\hat L_t$ 加噪扰动后取 argmin，探索臂的采样只看 $\hat L_t$ 的排名和损失差，二者共享同一份累积估损但走完全独立的两条计算路径——这是绕开"$p_t$ 必须由 $w_t$ 算出"这一历史约定的核心结构。

### 关键设计

**1. Pareto 扰动 FTPL 做利用：不解优化就拿到 Tsallis-INF 同阶的利用策略**

FTPL 的利用臂只要一次 argmin 就能选出，问题是用什么扰动。早期 FTPL bandit 用 Gumbel 扰动（对应 Exp3），利用概率有 softmax 闭式但在随机环境下后悔界次优。作者改用 shape $\alpha>1$ 的 Pareto 分布 $f(x)=\alpha/x^{\alpha+1}$ 采扰动 $r_{t,i}$，利用臂取 $i_t=\arg\min_i\{\hat L_{t,i}-r_{t,i}/\eta_t\}$。Kim & Tewari 2019 与 Lee 2025 早已证明 Pareto 扰动 FTPL 在隐含利用概率上与 $\beta=1-1/\alpha$ 的 Tsallis 熵 FTRL 完全对应，所以 BOBW 所需的"利用概率衰减率"自动具备。换 Pareto 是为了拿 BOBW，但副作用是利用概率 $w_{t,i}=\phi_i(\eta_t\hat L_t)$ 没有闭式表达——这正是下一步必须绕开 $w_t$ 的来由。

**2. 基于排名的代理向量 $q_t$ 替代 $w_t$ 定义探索分布：彻底绕过几何重采样**

所有现有 decoupled 算法都规定"探索概率必须由利用概率 $w_t$ 算出"，可 FTPL 没有 $w_t$ 闭式，估它要靠几何重采样，整向量代价升到 $\mathcal{O}(K^2\log K)$，速度优势全失。作者指出"$p_t$ 是 $w_t$ 的函数"只是充分而非必要条件：只要找到一个仅用已有量、又能与 $w_t$ 建立紧不等式的代理就行。于是定义损失差 $\hat{\underline L}_{t,i}=\hat L_{t,i}-\min_j\hat L_{t,j}$ 和排名 $\sigma_{t,i}$，令

$$q_{t,i}=\Big(\min\Big\{\tfrac{1}{1+\eta_t\hat{\underline L}_{t,i}},\ \tfrac{1}{\sigma_{t,i}^{1/\alpha}}\Big\}\Big)^{(\alpha+1)/2},\qquad p_{t,i}=\frac{q_{t,i}}{\sum_j q_{t,j}}.$$

$\min$ 里前一项按损失大小衰减、后一项按排名衰减，取较小者再做 $(\alpha+1)/2$ 次方，恰好近似 $w_{t,i}^{1/2+1/(2\alpha)}$，对应 Decoupled-Tsallis-INF 中的 $w_{t,i}^{1-\beta/2}$，关键紧不等式 $q_{t,i}\le w_{t,i}^{1/2+1/(2\alpha)}\lesssim w_{t,i}^{1-1/\alpha}$ 由 Lemma D.2 给出。纯排序加赋值，无凸优化、无重采样。

**3. 增量排名维护 + 自界后悔分析：实现与证明两侧都不依赖优化**

代理量解决了"怎么定义 $p_t$"，还要让它在工程和理论上都便宜。实现侧，IW 估计每轮只更新被选臂 $j_t$ 的 $\hat L$，其余臂排名最多变 1，所以对 $j_t$ 二分定位 $\mathcal{O}(\log K)$、对受影响区间做 $\mathcal{O}(K)$ 修正，把整向量重排序的 $\mathcal{O}(K\log K)$ 压到平均 $\mathcal{O}(K)$。分析侧，Lemma 3.4 把后悔拆成 stability + penalty 两项（省掉 Honda 2023/Lee 2024 的额外 $\log T$ 因子），Lemma 3.5/3.6 分别界住两项，再借辅助事件 $D_t$ 在好事件上把 $q_{t,i}$ 换成 $w_{t,i}^{1-1/\alpha}$，最后用 self-bounding $\max_w\{Aw^{1-1/\alpha}/\sqrt t-\Delta_i w\}\le\mathcal{O}(A^\alpha\Delta_i^{1-\alpha}t^{-\alpha/2})$ 收敛到对抗 $\mathcal{O}(\sqrt{KT})$、随机 $\mathcal{O}(K/\Delta_{\min})$。$\alpha=3$（即 $\beta=2/3$）恰是 Decoupled-Tsallis-INF 的最优配置，于是 FTPL 在同阶 BOBW 下把实现路径整体换成了轻量版。

## 实验关键数据

### 主实验

| 实验设定 | 指标 | EXP3 (β=1) | FTRL (β=2/3, Decoupled-Tsallis-INF) | FTPL (本文, α=3) |
|--------|------|------|----------|------|
| 对抗 8 臂, $\Delta=0.125$ | 累积后悔 (越低越好) | 最高 | 中 | **最低** |
| 随机 5 臂, 简单 $\mu_1$, $\Delta_{\min}=0.05$ | 累积后悔 | 较高 | 较低 | **最低** |
| 随机 5 臂, 困难 $\mu_2$, $\Delta_{\min}=0.002$ | 累积后悔 | 高 | 中 | **最低** |
| SCS 凸求解器, $K\in\{2,\ldots,64\}$ | 单步耗时 (ms) | — | 显著高 | **↓约 130× ($K=2$)** |
| Newton+warm start, $K$ 增大 | 单步耗时斜率 | — | 斜率最大 | **斜率最小** |

### 消融 / 实现对比

| 维度 | FTRL (Newton + warm start) | FTPL(sorting) | FTPL(improved, 本文实现) |
|------|---------|---------|---------|
| 每轮算法依赖 | 凸优化迭代 | 整向量重排序 | 增量二分定位 + 块修正 |
| 平均单步复杂度 | 无形式化保证 (≥ $\mathcal{O}(K)$ × #iter) | $\mathcal{O}(K\log K)$ | **$\mathcal{O}(K)$ 平均** |
| 需要 $w_t$ 估计 | 是 (优化解) | 否 | **否** |
| 需要几何重采样 | 否 | 否 (本文绕开) | **否** |
| 对抗后悔界 | $\mathcal{O}(\sqrt{KT})$ | $\mathcal{O}(\sqrt{KT})$ | $\mathcal{O}(\sqrt{KT})$ |
| SCA / 随机后悔界 | $\mathcal{O}(K/\Delta_{\min})$ | $\mathcal{O}(K/\Delta_{\min})$ | $\mathcal{O}(K/\Delta_{\min})$ |

### 关键发现
- "FTPL 比 FTRL 慢"的直觉只在不优化实现的情况下成立；本文证明只要绕开 $w_t$ 估计这一步，FTPL 的实现复杂度（$\mathcal{O}(K)$ 增量）本来就低于 FTRL（凸优化迭代）；实测 $K=2$ 已快约 130×，且随 $K$ 增长斜率更小，规模化优势随 $K$ 放大。
- 用排名而不是用估损值本身做代理量是设计精髓：排名是个鲁棒统计量，对损失噪声/量级不敏感，且增量维护代价低，使理论紧界与高效实现能同时达成。
- $\alpha\in(1,3]$ 时分析最紧，$\alpha=3$ 等价 $\beta=2/3$，与 Rouyer & Seldin 2020 的最优 Tsallis 配置吻合，说明 FTPL 和 FTRL 在这个问题上的"理论最优形状"是对应的，本文只是把实现路径换了。
- 在"困难"随机实例（$\mu_2$，所有臂均值都接近 0.9、$\Delta_{\min}=0.002$）下，所有 BOBW 算法都需更多轮才收敛，但 FTPL 在每个时间点都领先 FTRL/EXP3——经验上 self-bounding 推出的常数也最优。
- 在对抗设定的 16 个 $(K,\Delta)$ 网格扫描（附录 E.2）中，FTPL 的领先稳定地出现在各种 $K$ 和各种 gap 上，不存在某个区域被基线反超的现象，说明改进是结构性的而不是某种特殊调参的副产物。

## 亮点与洞察
- "用代理量代替真实概率"是一个可外推的方法论：FTPL 的 $w_t$ 难算的问题在很多 bandit 变体里（contextual、组合臂、半 bandit）都出现，本文证明只要能找到一个有"对的紧不等式"的可计算上界，BOBW 都可以照搬，这给后续工作提供了一个明确的设计范式。
- 把"排名"当作合法的状态量利用，是论文里最便宜也最聪明的一步——它把一个本来要凸优化才能算清楚的几何对象（概率单纯形上的最优解）简化成 $\sigma_{t,i}\in\{1,\ldots,K\}$，分析和实现都顺势变简单。
- 增量排名 + 二分定位的实现把 $\mathcal{O}(K\log K)$ 排序压成 $\mathcal{O}(K)$ 平均，是一个"理论上量级、实现上常数"都吃干净的工程优化，很值得复用到其他 FTPL 类算法。
- 作者指出可以把同样的代理思路推广到 Prod family (Cesa-Bianchi 2007) 等其他在线学习框架——一个本来为 decoupled bandit 量身定制的技巧，其抽象层次足以撑起一类新工具，论文做了开放式留白。

## 局限与展望
- SCA 随机后悔界的第二项 $K/\Delta_{\min}$ 比 FTRL 用 log-barrier + arm-dependent learning rate (Jin 2023) 的 $\sqrt{K}/\Delta_{\min}$ 松一个 $\sqrt{K}$ 因子；论文指出对 FTPL 加 arm-dependent learning rate 等价于用 arm-dependent 扰动 + non-i.i.d. perturbation，分析非常复杂，留作 future work。
- $\alpha>3$ 时 $K$ 依赖会从 $\sqrt{K}$ 退化到 $K^{(\alpha-2)/(\alpha-1)}$，所以 $\alpha$ 不能任意大，需要在 $(1,3]$ 内调，调节灵活度有限。
- 算法依赖 unique best arm 假设，对存在多个等优臂的极端实例没有保证；SCA 假设每对臂的均值差恒定，对非平稳环境（drift）需要进一步分析。
- 实证只评了 $T=10^4$ 量级和 $K\le 256$，超大规模 bandit（推荐系统场景常见 $K\gtrsim 10^4$）下的实际效益还需补充；对超大 $K$ 时 PRG 重生成扰动+增量维护排名的常数项是否仍小于 FTRL Newton 迭代的常数项，需要新一轮 benchmark。
- 代理量 $q_t$ 的具体形式（双约束 $\min$ + $(\alpha+1)/2$ 次方）是为现有分析路径定制的，对 stability-penalty matching 等更细的分析技巧能否同样兼容，目前不明。

## 相关工作与启发
- **vs Decoupled-Tsallis-INF (Rouyer & Seldin 2020)**：同阶 BOBW 后悔界，但本文绕开了凸优化求 $w_t$；用 Pareto 扰动 + 排名代理是关键差异。
- **vs Avner et al. 2012 (decoupled Exp3)**：Avner 是 Exp3 + 方差最优探索，本文也是 FTPL 但用 Pareto 扰动（更强的尾），并把 BOBW 第一次带到 FTPL 框架下。
- **vs Honda 2023 / Lee 2024 (标准 MAB 的 FTPL-BOBW)**：他们在标准 MAB 下证明了 FTPL 的 BOBW，但用 GR 估 $1/w_{t,i_t}$；本文扩展到 decoupled 设定，但要估**整个**向量 $w_t$ 时 GR 不再划算，于是用 $q_t$ 代理跳过 GR，是一个非平凡的推广。
- **vs Chen et al. 2025 (近线性时间 GR)**：把 GR 单步成本从 $\mathcal{O}(K^2)$ 压到 $\mathcal{O}(K\log K)$，但仍是 GR；本文完全不需要 GR，结构上更轻。
- **vs Jin et al. 2023 (log-barrier + arm-dependent LR FTRL)**：在 SCA 第二项常数上更紧（$\sqrt{K}/\Delta_{\min}$ vs 本文 $K/\Delta_{\min}$），但每轮要解带 log-barrier 的凸优化，实际开销显著更高；本文走的是"在算法侧让步一点常数、在系统侧换回大量延迟"的权衡。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Oracle-Efficient Combinatorial Semi-Bandits](../../NeurIPS2025/optimization/oracle-efficient_combinatorial_semi-bandits.md)
- [\[ICML 2025\] BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization](../../ICML2025/optimization/bopo_neural_combinatorial_optimization_via_best-anchored_and_objective-guided_pr.md)
- [\[ICML 2026\] URS：统一的神经路由求解器](urs_a_unified_neural_routing_solver_for_cross-problem_zero-shot_generalization.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)

</div>

<!-- RELATED:END -->
