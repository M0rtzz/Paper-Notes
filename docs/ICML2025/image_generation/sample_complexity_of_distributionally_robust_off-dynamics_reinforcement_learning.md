---
title: >-
  [论文解读] Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction
description: >-
  [ICML2025][图像生成][Robust MDP] 提出 supremal visitation ratio $C_{vr}$ 度量在线鲁棒 MDP 的探索难度，设计首个支持一般 $f$-散度（TV/KL/$\chi^2$）的高效在线算法 ORBIT，并给出匹配的上下界，证明 $C_{vr}$ 是刻画 off-dynamics RL 在线可学习性的紧致度量。
tags:
  - ICML2025
  - 图像生成
  - Robust MDP
  - Off-dynamics RL
  - Sample Complexity
  - online learning
  - f-divergence
  - Regret Bounds
---

# Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction

**会议**: ICML2025  
**arXiv**: [2511.05396](https://arxiv.org/abs/2511.05396)  
**代码**: [panxulab/Online-Robust-Bellman-Iteration](https://github.com/panxulab/Online-Robust-Bellman-Iteration)  
**领域**: 分布鲁棒强化学习  
**关键词**: Robust MDP, Off-dynamics RL, Sample Complexity, online learning, f-divergence, Regret Bounds

## 一句话总结

提出 supremal visitation ratio $C_{vr}$ 度量在线鲁棒 MDP 的探索难度，设计首个支持一般 $f$-散度（TV/KL/$\chi^2$）的高效在线算法 ORBIT，并给出匹配的上下界，证明 $C_{vr}$ 是刻画 off-dynamics RL 在线可学习性的紧致度量。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：Off-dynamics RL 处理训练环境与部署环境转移动力学不同的问题，通常建模为鲁棒马尔可夫决策过程（RMDP）。现有工作主要依赖两类设定：

**生成模型设定**：允许对任意 $(s,a)$ 查询转移，回避了探索困难

**离线数据集设定**：假设预采集数据对部署环境有良好覆盖

这两类设定在实际中都过于理想。更现实的场景是智能体只能与训练环境**在线交互**，但需要学到在部署环境（可能发生分布偏移）下也表现良好的策略。

**核心挑战——信息赤字（Information Deficit）**：在名义环境中极少访问的状态 $s$，可能因部署环境的动力学偏移而变得至关重要。在标准 MDP 中这些稀有状态影响可忽略，但在 RMDP 中它们可能是最坏情况下决策的关键因素。

已有在线 RMDP 工作仅限于 **TV 距离 + fail-state 假设**这一特殊情况，对更一般的 $f$-散度（KL/$\chi^2$）没有可行的在线学习理论。本文首次系统回答：**在什么条件下可以在一般 $f$-散度 RMDP 上实现可证明高效的在线学习？**

## 方法详解

### 框架：CRMDP 与 RRMDP

论文统一处理两种鲁棒 MDP 框架：

| 框架 | 目标 | 不确定性建模 |
|------|------|-------------|
| **CRMDP**（约束鲁棒 MDP） | $\max_\pi \min_{P \in \mathcal{U}^\rho} V^\pi$ | 不确定集 $D_f(P \| P^o) \le \rho$ |
| **RRMDP**（正则鲁棒 MDP） | $\max_\pi \min_P V^\pi + \beta \cdot D_f(P, P^o)$ | 正则化惩罚偏移 |

两者均采用 $(s,a)$-rectangular 不确定集结构，考虑 TV、KL、$\chi^2$ 三种 $f$-散度。

### Supremal Visitation Ratio $C_{vr}$

论文的核心贡献是定义了衡量在线 RMDP 探索难度的新指标：

$$C_{vr} := \sup_{\pi, h, s} \frac{q_h^\pi(s)}{d_h^\pi(s)}$$

其中 $d_h^\pi(s)$ 是策略 $\pi$ 在名义环境下第 $h$ 步访问状态 $s$ 的概率，$q_h^\pi(s)$ 是在最坏情况转移下的对应概率。

- 当 $C_{vr} = 1$ 时退化为标准（非鲁棒）MDP
- 当 $C_{vr}$ 有界（关于 $S, A, H$ 多项式量级）时，在线学习可行
- 当 $C_{vr}$ 无界时，存在困难实例使任何算法指数级困难

### ORBIT 算法

**Online Robust Bellman Iteration (ORBIT)** 是本文提出的元算法，核心步骤：

1. **反向价值迭代**：从 $h = H$ 到 $1$，利用鲁棒 Bellman 方程估计 $Q$ 函数
2. **乐观探索**：加入 bonus 项 $b_h^k(s,a)$ 实现 optimism in the face of uncertainty
3. **执行贪心策略**收集轨迹，更新经验转移核 $\hat{P}_h^{k+1}$

$Q$ 函数估计的统一形式：

$$\hat{Q}_h^k(s,a) = \min\{RB_h^k(s,a) + b_h^k(s,a),\; H - h + 1\}$$

其中 $RB_h^k$ 是鲁棒 Bellman 估计器（通过对偶转换求解内部优化），$b_h^k$ 是探索奖励。

**不同散度下的对偶形式各异**：

- **TV**：内层优化转化为关于 $\eta$ 的一维搜索
- **KL**：利用指数函数的对偶，RRMDP-KL 有闭式解
- **$\chi^2$**：转化为关于 $\lambda$ 的方差优化

## 理论结果

### 遗憾上界（Theorem 5.9 & 5.16）

| 设定 | CRMDP 遗憾上界 | RRMDP 遗憾上界 |
|------|---------------|---------------|
| **TV** | $\tilde{O}(C_{vr}^{1/2} S^{3/2} A^{1/2} H^2 \sqrt{K})$ | $\tilde{O}(C_{vr}^{1/2} S A^{1/2} H^2 \sqrt{K})$ |
| **KL** | $\tilde{O}((1 + \frac{H\sqrt{S}}{\rho C_{MP}}) C_{vr}^{1/2} S^{1/2} A^{1/2} H \sqrt{K})$ | $\tilde{O}((1+\beta e^{\beta^{-1}H}\sqrt{S}) C_{vr}^{1/2} S^{1/2} A^{1/2} H \sqrt{K})$ |
| **$\chi^2$** | $\tilde{O}((1+\sqrt{\rho}) C_{vr}^{1/2} S^{3/2} A^{1/2} H^2 \sqrt{K})$ | $\tilde{O}((1+\frac{H}{\beta}) C_{vr}^{1/2} S^{3/2} A^{1/2} H^2 \sqrt{K})$ |

### 遗憾下界（Theorem 5.12 & 5.18）

对所有三种散度（TV/KL/$\chi^2$），CRMDP 和 RRMDP 均有：

$$\mathbb{E}[\text{Regret}(K)] = \Omega(C_{vr}^{1/2} \sqrt{K})$$

**关键结论**：上界主项中 $C_{vr}$ 和 $K$ 的阶与下界完全匹配，说明 $C_{vr}$ 是在线 RMDP 探索难度的紧致度量。

### 困难实例（Lemma 5.14）

当 $C_{vr} = 2^{2A}$ 时，任何算法的遗憾均为 $\Omega(2^A \sqrt{K})$，指数级困难。这证明了 $C_{vr}$ 有界是在线可学习的必要条件。

### 损失函数 / 训练策略
模型采用端到端训练，优化目标综合考虑任务损失和正则化项。


## 实验关键数据

1. **$C_{vr}$ 影响验证**（合成 MDP，$H=3, S=6, A=10$）：随 $C_{vr}$ 增大，学到的策略与最优策略的差距持续扩大，与理论一致
2. **模拟 RMDP**（$H=3, S=5, A=5$）：当扰动 > 0.6 时，所有鲁棒策略（CRMDP/RRMDP × TV/KL/$\chi^2$）均优于非鲁棒算法
3. **Frozen Lake**：ORBIT 在训练过程中稳定收敛；RRMDP-TV/KL 有闭式对偶解，训练最快；CRMDP-$\chi^2$ 计算代价最高

### 关键发现
- 主要组件/模块贡献了最关键的性能提升


## 亮点与洞察

- **理论完整性极高**：同一框架下统一处理 6 种设定（2 框架 × 3 散度），每种都有匹配的上下界
- **$C_{vr}$ 概念优雅**：将 fail-state 等特殊假设统一为一个标量度量，且证明了其紧致性
- **信息赤字视角新颖**：从名义环境与最坏环境的访问分布错配出发理解问题，解释了为何 fail-state 假设有效（Proposition 5.4）
- **RRMDP 优于 CRMDP 的证据**：RRMDP-TV 的 regret 比 CRMDP-TV 少 $\sqrt{S}$ 因子；RRMDP-KL 无需额外假设且有闭式解

## 局限与展望

1. **仅限表格型**：算法和理论均限于有限状态动作空间，未扩展到函数逼近
2. **$C_{vr}$ 实际不可知**：部署前无法计算 $C_{vr}$（取决于未知的最坏转移），难以预判算法效果
3. **CRMDP-KL 需额外假设**（Assumption 5.7）：要求转移概率有下界 $C_{MP}$，限制了适用范围
4. **RRMDP-KL 的 $\beta e^{\beta^{-1}H}$ 因子**可能较大，KL 正则在长 horizon 下 regret 退化
5. **上下界在 $S, A, H$ 维度未完全匹配**，仅在 $C_{vr}$ 和 $K$ 上最优

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首次在一般 f-散度在线 RMDP 上给出匹配上下界)
- 实验充分度: ⭐⭐⭐ (实验规模偏小，仅有表格型环境验证)
- 写作质量: ⭐⭐⭐⭐ (理论严谨，符号较多但逻辑清晰)
- 价值: ⭐⭐⭐⭐⭐ (为 off-dynamics RL 在线学习的理论基础做出关键贡献)


## 相关工作与启发
- **vs 同领域代表性方法**：本文在方法设计上有独特贡献，与现有方法形成互补
- **vs 传统方法**：相比传统方案，本文方法在关键指标上取得了显著提升
- **启发**：本文的技术路线对后续相关工作有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] ORVIT: Near-Optimal Online Distributionally Robust Reinforcement Learning](../../AAAI2026/image_generation/orvit_near-optimal_online_distributionally_robust_reinforcement_learning.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](../../NeurIPS2025/image_generation/towards_robust_zero-shot_reinforcement_learning.md)
- [\[NeurIPS 2025\] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](../../NeurIPS2025/image_generation/composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)
- [\[ICML 2025\] PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs](pak-ucb_contextual_bandit_an_online_learning_approach_to_prompt-aware_selection_.md)
- [\[ICML 2025\] Hierarchical Reinforcement Learning with Uncertainty-Guided Diffusional Subgoals](hierarchical_reinforcement_learning_with_uncertainty-guided_diffusional_subgoals.md)

<!-- RELATED:END -->
