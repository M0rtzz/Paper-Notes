---
title: >-
  [论文解读] Data- and Variance-dependent Regret Bounds for Online Tabular MDPs
description: >-
  [ICML 2026][强化学习][在线 MDP] 本文针对已知转移的在线 episodic tabular MDP，基于带 log-barrier 的乐观 follow-the-regularized-leader（OFTRL）设计了单一 best-of-both-worlds 算法…
tags:
  - "ICML 2026"
  - "强化学习"
  - "在线 MDP"
  - "best-of-both-worlds"
  - "OFTRL"
  - "log-barrier"
  - "数据依赖 regret"
  - "方差依赖 regret"
---

# Data- and Variance-dependent Regret Bounds for Online Tabular MDPs

**会议**: ICML 2026  
**arXiv**: [2602.01903](https://arxiv.org/abs/2602.01903)  
**代码**: 无  
**领域**: 强化学习 / 在线学习 / Bandits 理论  
**关键词**: 在线 MDP, best-of-both-worlds, OFTRL, log-barrier, 数据依赖 regret, 方差依赖 regret  

## 一句话总结
本文针对已知转移的在线 episodic tabular MDP，基于带 log-barrier 的乐观 follow-the-regularized-leader（OFTRL）设计了单一 best-of-both-worlds 算法，同时给出对抗 regime 下的一阶/二阶/路径长度三种数据依赖 regret 上界，以及随机 regime 下的方差感知 gap-无关和 gap-相关 polylog 界，并配套证明匹配的下界。

## 研究背景与动机

**领域现状**：在线 episodic tabular MDP 是 RL 理论的标准抽象——学习者在 $T$ 个 episode 里反复与一个 $S$ 状态、$A$ 动作、$H$ 层的 MDP 交互，每个 episode 由环境给出一个 loss 函数，学习者只能沿采样轨迹观测 bandit 反馈。主流求解器有两条线：一是直接在所有 occupancy measure 集合 $\Omega(P)$ 上做全局优化（minimax 最优但计算重），二是在每个 state 上做策略优化（每个 state 看作一个 multi-armed bandit，更实用但 regret 多一个 $H$ 因子）。在对抗 regime 已知 minimax 速率是 $\tilde{O}(\sqrt{HSAT})$，在随机 regime 则能达到 gap 相关的 $O(\log T)$。

**现有痛点**：现有结果碎成三块且互不兼容。第一，best-of-both-worlds 算法（在两个 regime 上都接近最优）和精细数据依赖界（如一阶 small-loss $L^\star$）通常由不同算法分别给出，实际环境未知时无法选；第二，对抗 regime 下唯一的数据依赖结果只有一阶界，bandit 文献里早已成熟的二阶界和路径长度界在 MDP 里是空白；第三，随机 regime 的 gap 相关界（如 Jin et al. 2021）多了一个 $1/\min_{s,a}\Delta(s,a)$ 因子，且没有方差感知版本。

**核心矛盾**：把这些精细界统一到一个算法里之所以困难，是因为 MDP 的 bandit 反馈下 loss 估计误差会沿动力学向下游 value 估计传播——和 multi-armed bandit 不同，每个 state-action 对的估计误差不能被独立控制，必须设计出 bias 与精细复杂度量"对齐"的 loss 估计器和 Q 估计器，才能让 self-bounding 分析跑通。

**本文目标**：构造单一算法，在已知转移设定下同时达到：(1) 对抗 regime 的一阶、二阶、路径长度三种数据依赖界；(2) 随机 regime 的方差感知 gap-无关界与 polylog 的 gap-相关界；(3) 全局优化版本与策略优化版本两条路线都覆盖；(4) 通过下界证明上界是 minimax 最优。

**切入角度**：以 OFTRL + log-barrier + 自适应学习率为骨架，把 Jin et al. (2021) 在 FTRL 上的 loss-shifting 技巧迁移到 OFTRL 框架，并据需要切换两种 loss prediction（梯度下降式 vs 经验均值式）来分别撬动路径长度界和方差感知 gap-相关界。

**核心 idea**：用 OFTRL 同时承载多种数据依赖性——OFTRL 的 stability 由"移位 loss" $\tilde{\ell}_t = \hat{\ell}_t - m_t$ 控制，恰当选择 $m_t$ 就能让 stability 项收敛到所需的复杂度量。

## 方法详解

### 整体框架
固定一个已知转移的 $H$ 层 tabular MDP $M=(\mathcal{S},\mathcal{A},P,H,s_0)$。学习者每个 episode $t$ 选一个策略 $\pi_t$，与环境交互一条轨迹，目标是最小化对所有平稳策略的 regret：
$\mathrm{Reg}_T = \max_{\pi \in \Pi} \mathbb{E}\bigl[\sum_{t=1}^T V^{\pi_t}(s_0; \ell_t) - V^{\pi}(s_0; \ell_t)\bigr]$。

整个工作由三部分构成：(i) 第 3 节定义新的复杂度量；(ii) 第 4–5 节分别给出全局优化与策略优化算法，所有算法共享"OFTRL + log-barrier + 自适应学习率 + loss-shifting"模板，差异只在 loss / Q 估计器和 loss prediction 的选择；(iii) 第 6 节通过经典硬实例构造证明 $\Omega(\sqrt{SAL^\star})$、$\Omega(\sqrt{SAQ_\infty})$、$\Omega(\sqrt{HV_1})$、$\Omega(\sqrt{SAV_T})$ 四个下界，匹配全局优化上界。

### 关键设计

1. **新数据依赖复杂度量**:

    - 功能：把"对抗 loss 序列有多容易"和"随机 loss 噪声有多小"这两件事用可计算量精确刻画，再由后续 OFTRL 自动适配这些量。
    - 核心思路：在对抗 regime 引入三个量——一阶 small-loss $L^\star = \min_{\pi} \mathbb{E}[\sum_t V^\pi(s_0;\ell_t)]$；二阶 $Q_\infty = \min_{\ell^\star} \mathbb{E}[\sum_t \sum_h \|\ell_t(h)-\ell^\star(h)\|_\infty^2]$，当 loss 围绕某个基线波动时变小；路径长度 $V_1 = \mathbb{E}[\sum_t \|\ell_{t+1}-\ell_t\|_1]$，当 loss 序列变化缓慢时变小。在随机 regime 引入 occupancy 加权方差 $V = \max_\pi \sum_{s,a} q^\pi(s,a)\sigma^2(s,a)$ 和条件占用加权方差 $V_c(s)$，刻画"已经走到 $s$ 之后剩下多少噪声"。
    - 设计动机：$Q_\infty$ 和 $V_1$ 是 bandit 文献的标配但 MDP 里此前是空白；$V$ 和 $V_c$ 相比已有的 $\mathrm{Var}_{\max}$、$\mathrm{Var}^c_{\max}$ 去掉了多余的 $V^{\pi^\star}(s')$ 方差项，得益于已知转移设定，结果比已有方差度量"$H^2$ 倍更紧"。

2. **全局优化算法（Algorithm 1，定理 4.1 / 4.2）**:

    - 功能：在 occupancy 集合 $\Omega(P)$ 上直接用 OFTRL 滚动，得到 $\tilde{O}(\sqrt{SA \cdot \min\{L^\star, HT-L^\star, Q_\infty, V_1\}})$ 的对抗界和 $\tilde{O}(\sqrt{SAV_T})$、$\mathrm{polylog}(T)$ 的随机界。
    - 核心思路：每个 episode 解 $q^{\pi_t} = \arg\min_{q\in\Omega(P)}\{\langle q, \sum_{\tau<t}\hat\ell_\tau + m_t\rangle + \psi_t(q)\}$，其中 $\psi_t(q) = \sum_{s,a} \tfrac{1}{\eta_t(s,a)}\log(1/q(s,a))$ 是 per-coord log-barrier，学习率 $1/\eta_{t+1} = 1/\eta_t + \eta_t \zeta_t/\log T$ 按 stability 项 $\zeta_t$ 自适应增长。loss 估计器用乐观 IW 形式 $\hat\ell_t(s,a) = m_t(s,a) + I_t(s,a)(\ell_t - m_t)/q^{\pi_t}(s,a)$，关键的 loss-shifting 函数 $g_t(s,a) = Q^{\pi_t}(s,a;\tilde\ell_t) - V^{\pi_t}(s;\tilde\ell_t) - \tilde\ell_t(s,a)$ 把 OFTRL 等价改成在 advantage 上跑，stability 项天然由 advantage 的二阶矩界住，进而走 self-bounding 得到 polylog 的 gap-相关界。两种 $m_t$ 各司其职：梯度下降式 $m_{t+1}=(1-\xi)m_t+\xi\ell_t$（$\xi=1/4$）适合路径长度 $V_1$，经验均值式 $m_t = \sum_\tau I_\tau \ell_\tau / N_{t-1}$ 适合方差感知 gap-相关界（让 stability 收敛到 $V_c$）。
    - 设计动机：把 Jin et al. (2021) 的 FTRL+log-barrier 路线整体搬到 OFTRL，多出来的 $m_t$ 项正好提供"乐观"——当预测准时 stability 几乎为零，自然得到 $V_1$ 和 $V_c$ 这类细界，预测不准时退化为 FTRL 的 $\sqrt{HSAT}$ worst case。

3. **策略优化算法（定理 5.2 / 5.3）+ 更乐观的 Q 估计器**:

    - 功能：以每个 state 上的 OFTRL 作为局部 bandit 求解器，得到比全局优化只多一个 $H$ 因子的同款数据/方差依赖界 $\tilde{O}(\sqrt{H^2 SA \cdot \min\{L^\star,\ldots,V_1\}})$，胜在每步只需 per-state 闭式更新，计算友好。
    - 核心思路：在每个 $s$ 上跑 OFTRL，$\pi_t(\cdot|s) = \arg\min_{p\in\Delta(A)} \{\langle p, \sum_{\tau<t}(\hat Q_\tau(s,\cdot) - B_\tau(s,\cdot)) + m_t(s,\cdot)\rangle + \psi_t(p)\}$，其中 $\hat Q_t$ 是新的"更乐观"的 Q 估计器：不仅对当前 loss 用 IW，还把对未来 value 的预测也注入到估计里，从而抵消 OFTRL loss prediction 引入的 bias。Dann et al. (2023a) 在 FTRL 策略优化里用过一阶 Q 估计器，但搬到 OFTRL 时会因 $m_t \neq 0$ 留下额外 bias 项，本文的新估计器构造让 $\mathbb{E}_t[\hat Q_t - B_t]$ 恰好等于真 advantage，从而 stability 分析复用全局优化路线。
    - 设计动机：策略优化更实用，但已有结果只能做到 first-order；要把二阶、路径长度、方差感知 gap-相关全做出来，必须解决"乐观预测如何与 per-state 局部更新和 layer-by-layer value 估计同时兼容"这件事，新 Q 估计器正是为此设计。

### 损失函数 / 训练策略
本文是纯理论工作，"训练"指 OFTRL 的迭代更新。共享超参：$H \le S$ 假设，初始学习率 $1/\eta_1 = 2H$，loss prediction 步长 $\xi = 1/4$，log-barrier 系数随 stability 项 $\zeta_t = q^{\pi_t}(s,a)^2 \cdot \min\{(\hat\ell_t-m_t)^2, (\hat\ell_t+g_t-m_t)^2\}$ 自适应增长。所有 regret 上界对未知的复杂度量是"参数无关"的——算法不需要预先知道 $L^\star$、$Q_\infty$、$V_1$、$V$、$V_c$。

## 实验关键数据

本文是纯理论工作，无实验数据，所有结果以定理 + 表格形式给出。下面整理两张关键对比表（数字为 leading term，省略对数因子；$U = \sum_{s,a\neq\pi^\star(s)} H^2\log(T)/\Delta(s,a)$、$U_{\mathrm{Var}} = \sum_{s,a\neq\pi^\star(s)} HV_c(s)\log(T)/\Delta(s,a)$、$C$ 为对抗 corruption 总量）。

### 主实验：全局优化 regret 上界对比
| 方法 | 对抗 regime | 随机 + 对抗 corruption regime |
|------|------|------|
| Zimin & Neu (2013) | $\sqrt{HSAT}$ | $\sqrt{HSAT}$ |
| Lee et al. (2020) | $\sqrt{SAL^\star}$ | $\sqrt{SAL^\star}$ |
| Jin et al. (2021) | $\sqrt{HSAT}$ | $U_{\mathrm{Jin}} + \sqrt{U_{\mathrm{Jin}}C}$，含 $1/\min\Delta$ |
| **本文 Thm 4.1** | $\sqrt{SA\min\{L^\star, HT{-}L^\star, Q_\infty, V_1\}}$ | $\min\{\sqrt{SA(V_T+C)},\ U+\sqrt{UC}\}$ |
| **本文 Thm 4.2** | $\sqrt{SA\min\{L^\star, HT{-}L^\star, Q_\infty\}}$ | $\min\{\sqrt{SA(V_T+C)},\ U_{\mathrm{Var}}+\sqrt{U_{\mathrm{Var}}C}\}$ |

### 关键消融：策略优化 vs 全局优化（同款适配性，多一个 $H$）
| 方法 | 对抗 regime | 随机 + corruption regime |
|------|------|------|
| Luo et al. (2021) | $\sqrt{H^3 SAT}$ | $\sqrt{H^3 SAT}$ |
| Dann et al. (2023a) | $\sqrt{H^2 SAL^\star}$ | $U + \sqrt{UC}$ |
| **本文 Thm 5.2** | $\sqrt{H^2 SA \min\{L^\star, HT{-}L^\star, Q_\infty, V_1\}}$ | $\min\{\sqrt{H^2 SA(V_T+C)},\ U+\sqrt{UC}\}$ |
| **本文 Thm 5.3** | $\sqrt{H^2 SA \min\{L^\star, HT{-}L^\star, Q_\infty\}}$ | $\min\{\sqrt{H^2 SA(V_T+C)},\ U_{\mathrm{Var}}+\sqrt{U_{\mathrm{Var}}C}\}$ |

### 关键发现
- 第 6 节构造硬实例证 $\Omega(\sqrt{SAL^\star})$、$\Omega(\sqrt{SA Q_\infty})$、$\Omega(\sqrt{H V_1})$、$\Omega(\sqrt{SA V_T})$ 下界，意味着全局优化版本在 $L^\star$、$Q_\infty$、$V_1$ 上 minimax 最优，方差感知 gap-无关界也只差对数因子；策略优化版本相比下界差一个 $H$ 因子，作者把这归为已知的 $H$-gap 现象。
- 两种 loss prediction 必须按目标量分开用：经验均值预测 $m_t = \sum_\tau I_\tau\ell_\tau / N_{t-1}$ 能撬动 $V_c$，但不能给出 $V_1$ 界；梯度下降式预测 $m_{t+1}=(1-\xi)m_t+\xi\ell_t$ 能给 $V_1$，但 $V_c$ 退化为更松的 $V$；因此即使在同一算法骨架下，"具体能拿到哪种数据依赖性"取决于 $m_t$ 的选择。
- gap-相关 polylog 界（Thm 4.2）比 Jin et al. (2021) 干净——后者额外有 $H^3 S \log T / \min_{s,a\neq\pi^\star(s)} \Delta(s,a)$ 项，本文用方差 $V_c$ 替换 $H^2$ 后这一项被吸收，相当于"方差小的 MDP"会有真实可见的 polylog 改进。

## 亮点与洞察
- OFTRL + log-barrier + 自适应学习率组合是把多种数据依赖性塞进同一算法的"瑞士军刀"，关键是 stability 项的二次形式 $\zeta_t \propto q^{\pi_t}^2(\hat\ell_t-m_t)^2$ 天然兼容 $L^\star$、$Q_\infty$、$V_1$、$V_c$ 四种复杂度量——只要换 $m_t$ 就换一种适配性，不必重写整套算法。这种"骨架 + 插件"思想可直接迁移到其他 bandit-style RL 问题（如线性 MDP、contextual bandits with structure）。
- "更乐观的 Q 估计器"是策略优化版本的灵魂：FTRL 路线的 Q 估计器搬到 OFTRL 会留下 $m_t$ 引入的不可消 bias，本文在估计器里再加一项预测项把 bias 精确抵消，这种"为乐观算法专门重写估计器"的思路对未来想把 OFTRL 引入更复杂 RL 设定（unknown transition、function approximation）极具参考价值。
- 已知转移设定下重新定义方差为 $V$、$V_c$（去掉 $\mathrm{Var}_{s'\sim P}[V^{\pi^\star}(s')]$ 项）让方差界天然紧了 $H^2$，启示我们：在做精细 regret 分析时，"假设强度"和"度量定义"是耦合的——已知转移就该用更细的方差度量，不能盲套未知转移文献的 $\mathrm{Var}^\star$。

## 局限与展望
- 设定限于已知转移。作者明确把未知转移留作 open problem——unknown $P$ 下要同时控制 loss 估计误差和 transition 估计误差，二阶/路径长度/方差感知的全套适配性目前对全局和策略优化都无解，Dann et al. (2023a) 仅做到一阶。
- 策略优化版本所有上界比全局优化多一个 $H$ 因子，下界没能匹配，作者把这归到已知的 $H$-gap 现象，但是否能去掉这个 $H$ 仍是 open。
- 算法计算复杂度未明确给出，全局优化每步要在 $\Omega(P)$ 上解带 log-barrier 的凸优化，实际部署成本可能不低；理论上 minimax 最优，工程上是否值得替换现有 OMD/FTRL 路线还需要后续 empirical 验证。
- 结果只覆盖 tabular MDP，未涉及函数逼近——把 OFTRL + log-barrier 与线性 MDP / general function approximation 结合，可能是延伸方向。
- 实证空白：作为纯理论论文，没有任何 simulation 或 benchmark 实验比较"理论上更紧的常数与对数因子"在中等规模 $T$ 下是否真能转化为更小的实际 regret，工程师视角的实用价值还需要后续 empirical 研究来验证。

## 相关工作与启发
- **vs Jin et al. (2021)**：同样基于 occupancy + log-barrier，但他们用 FTRL，只能给出一阶 small-loss 界和带 $1/\min\Delta$ 因子的 gap-相关界；本文换成 OFTRL 并配合两种 loss prediction，把二阶、路径长度、方差感知 gap-相关全部纳入，并去掉了 $1/\min\Delta$ 项。
- **vs Lee et al. (2020)**：他们在 unknown transition 下做出一阶 $\sqrt{SAL^\star}$ 界，但仅限对抗 regime；本文是 known transition + best-of-both-worlds + 更多数据依赖性，覆盖面广得多但设定更窄。
- **vs Dann et al. (2023a)**：他们是策略优化路线下首个 best-of-both-worlds 算法（一阶 $\sqrt{H^2 SAL^\star}$），本文把同款架构升级到 OFTRL 并设计更乐观 Q 估计器，得到二阶、路径长度、方差感知三件套。
- **vs Simchowitz & Jamieson (2019) / Chen et al. (2025)**：随机 regime 的方差感知 gap-相关界传统上用 value-based 方法做，本文是首次用 occupancy/policy 优化路线得到同款 polylog 界且方差度量更紧（$V_c$ vs $\mathrm{Var}^c_{\max}$）。
- **vs Wei & Luo (2018) / Ito et al. (2022a)**：他们在 multi-armed bandit 设定下用 OFTRL + log-barrier 得到二阶/路径长度界，本文首次把这条技术路线推到 tabular MDP，需要额外处理 MDP 动力学带来的 loss 估计误差传播——这正是新 loss-shifting 函数 $g_t$ 和更乐观 Q 估计器要解决的核心障碍。
- **启示**：把数据依赖 bandit 理论搬到 MDP 不是套公式那么简单，而是需要重新设计与 MDP 结构对齐的估计器；这种基础 bandit 技术升级再做 MDP 适配的工作模式，对未来研究 partial observability、constraint MDP、multi-agent MDP 的精细 regret 仍有方法论参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Online Prediction of Stochastic Sequences with High Probability Regret Bounds](../../ICLR2026/reinforcement_learning/online_prediction_of_stochastic_sequences_with_high_probability_regret_bounds.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](../../NeurIPS2025/reinforcement_learning/improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[ICML 2026\] ALSO: Adversarial Online Strategy Optimization for Social Agents](also_adversarial_online_strategy_optimization_for_social_agents.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)

</div>

<!-- RELATED:END -->
