---
title: >-
  [论文解读] Learning to Play Multi-Follower Bayesian Stackelberg Games
description: >-
  [ICLR 2026][强化学习][Stackelberg博弈] 首次系统研究多追随者贝叶斯Stackelberg博弈（BSG）的在线学习问题，通过对领导者策略空间进行"最佳响应区域"几何分割，在类型反馈下实现 $\tilde{O}(\sqrt{\min\{L…
tags:
  - "ICLR 2026"
  - "强化学习"
  - "Stackelberg博弈"
  - "贝叶斯博弈"
  - "在线学习"
  - "最佳响应区域"
  - "遗憾界"
---

# Learning to Play Multi-Follower Bayesian Stackelberg Games

**会议**: ICLR 2026  
**arXiv**: [2510.01387](https://arxiv.org/abs/2510.01387)  
**代码**: 无  
**领域**: 博弈论/在线学习  
**关键词**: Stackelberg博弈, 贝叶斯博弈, 在线学习, 最佳响应区域, 遗憾界

## 一句话总结

首次系统研究多追随者贝叶斯Stackelberg博弈（BSG）的在线学习问题，通过对领导者策略空间进行"最佳响应区域"几何分割，在类型反馈下实现 $\tilde{O}(\sqrt{\min\{L, nK\} \cdot T})$ 的遗憾界——该界不随追随者数 $n$ 呈多项式增长，并证明了几乎匹配的 $\Omega(\sqrt{\min\{L, nK\}T})$ 下界。

## 研究背景与动机

**领域现状**：Stackelberg博弈是多智能体系统中建模「领导者先承诺策略、追随者后最佳响应」这一非对称交互的基础框架。其应用覆盖安全博弈（如机场巡逻资源分配）、平台经济（平台发布功能影响消费者）、信息设计（贝叶斯说服）和合同设计等场景。在贝叶斯版本中，追随者持有私有类型，领导者需要知道类型分布才能计算最优策略（Stackelberg均衡）。现有的在线学习工作（Balcan et al. 2015, 2025; Bollini et al. 2026）仅处理**单追随者**情形。

**现有痛点**：当存在 $n$ 个追随者、每人 $K$ 种类型时，联合类型空间大小为 $K^n$，呈指数爆炸。直接估计联合分布需要 $\Omega(\sqrt{K^n/t})$ 的估计误差，导致 $\Omega(\sqrt{K^n T})$ 的遗憾界，在 $n$ 较大时完全不可接受。此外，追随者的最佳响应是对领导者混合策略的分段常数函数，使得领导者效用在策略空间上**不连续且非凸**。即便在离线、单追随者版本中，当领导者动作数 $L$ 增长时，计算最优策略也是NP-Hard的（Conitzer & Sandholm, 2006）。

**核心矛盾**：多追随者引入联合类型空间的指数爆炸，如何设计学习算法使遗憾界避免对 $n$ 的指数依赖？直觉上，$K^n$ 大小的联合分布难以学习，但领导者真正关心的是**效用函数**而非分布本身——效用函数的复杂度是否远小于分布的复杂度？

**本文目标** (1) 多追随者BSG的在线学习是否可行？(2) 在类型反馈和动作反馈两种设置下，最优遗憾界分别是多少？(3) 遗憾界能否避免对 $n$ 的指数依赖？

**切入角度**：作者观察到，虽然联合类型空间 $K^n$ 指数大，但领导者策略空间 $\Delta(\mathcal{L})$ 仅有 $L-1$ 维。利用计算几何中关于超平面排列的经典结果，可以证明策略空间被分割为**至多 $O(n^L K^L A^{2L})$ 个非空的最佳响应区域**——这个数量关于 $n$ 仅多项式增长（当 $L$ 固定时）。更关键的是，领导者效用在每个区域内是**线性的**，可以用线性规划高效求解。

**核心 idea**：将领导者策略空间按追随者最佳响应几何分割为多面体区域，利用区域数关于 $n$ 多项式可控的性质，设计基于UCB和集中不等式的学习算法，实现遗憾界不随 $n$ 指数增长。

## 方法详解

### 整体框架

本文的问题设定如下：一个领导者有 $L$ 个动作，$n$ 个追随者各有 $K$ 种类型和 $A$ 种动作。每轮领导者先选混合策略 $x \in \Delta(\mathcal{L})$，追随者类型从未知分布 $\mathcal{D}$ 中采样，每个追随者观察 $x$ 后选最佳响应动作。领导者目标是通过 $T$ 轮交互最小化累积遗憾。关键假设包括：(1) 追随者短视最佳响应；(2) 无外部性——每个追随者的效用只依赖自身动作、类型和领导者动作，不受其他追随者动作影响。

整体算法思路分三层：**底层**——将策略空间 $\Delta(\mathcal{L})$ 几何分割为最佳响应区域，每个区域内效用线性；**中层**——利用区域结构进行集中不等式分析，证明经验效用可以均匀地逼近真实效用；**顶层**——在类型反馈下用经验最优策略算法，在动作反馈下用UCB算法在区域间探索。

### 关键设计

**1. 最佳响应区域的几何刻画：把追随者引起的效用不连续性变成超平面排列的区域计数。**

领导者效用之所以非凸又不连续，是因为追随者的最佳响应是领导者混合策略的分段常数函数——策略稍微一动，某个追随者就可能跳到另一个动作。本文的破局思路是把这种"分段"显式刻画出来：定义一个响应模式 $W = (w_1, \ldots, w_n)$，其中 $w_i: \Theta \to \mathcal{A}$ 指定追随者 $i$ 在各类型下选哪个动作，对应的**最佳响应区域** $R(W) = \{x \in \Delta(\mathcal{L}) \mid \mathbf{br}(\theta, x) = W(\theta), \forall \theta\}$ 就是所有让追随者恰好按 $W$ 响应的领导者策略集合。

每个区域都能写成 $O(nKA)$ 个半空间的交集 $R(W) = \bigcap_{i, \theta_i, a_i} H(d_{\theta_i, w_i(\theta_i), a_i})$，其中 $H(d) = \{x: \langle x, d \rangle \geq 0\}$ 编码了"追随者 $i$ 类型 $\theta_i$ 偏好 $w_i(\theta_i)$ 而非 $a_i$"这条线性约束。于是整个策略单纯形被一组超平面切成若干多面体。借助计算几何中超平面排列的计数结果，非空区域总数 $|\mathcal{W}| = O(n^L K^L A^{2L})$——当 $L$ 固定时只对 $n$ 多项式增长，而非随 $K^n$ 爆炸；这些区域还能用 BFS 图遍历在 $\mathrm{poly}(n^L, K^L, A^L, L)$ 时间内枚举出来。这一步的价值在于：区域内追随者响应固定，领导者效用退化成线性函数，每个区域的最优策略一个 LP 就能解，全局最优只需在 $|\mathcal{W}|$ 个区域里取最大值。作为副产品，这也证明了 BSG 在 $L$ 为常数时可多项式时间离线求解，补上了此前只知"$L$ 增长时 NP-Hard"的复杂度图谱缺口。

**2. 类型反馈下的学习：绕开 $K^n$ 的联合分布，直接让经验效用均匀收敛。**

当每轮能观测到全部追随者类型时，最朴素的 Algorithm 1（一般类型分布）就是每轮选经验效用最大的策略 $x^t = \arg\max_x \sum_{s<t} u(x, \mathbf{br}(\theta^s, x))$。关键不在于估计需要 $K^n$ 个参数的联合分布 $\mathcal{D}$，而是借区域结构证明经验效用的**均匀集中**：每个区域内效用是 $L$ 维线性函数、伪维度至多 $L$，对 $|\mathcal{W}|$ 个区域取 union bound 即得 $|U_\mathcal{D}(x) - \hat{U}^t(x)| \leq O(\sqrt{L \log(nKAT)/t})$，从而拿到 $\tilde{O}(\sqrt{L \cdot T})$ 遗憾——完全避开了分布的指数复杂度。

当类型在追随者间独立时，Algorithm 2 给出另一条更优路径：分别估计每个追随者的边际分布 $\hat{\mathcal{D}}_i$，再取乘积 $\hat{\mathcal{D}} = \prod_i \hat{\mathcal{D}}_i$。独立性让联合分布的 TV 距离可拆成各边际的 Hellinger 距离之和，把估计误差从 $O(\sqrt{K^n/t})$ 一举压到 $O(\sqrt{nK/t})$，对应 $O(\sqrt{nK \cdot T})$ 遗憾。两套分析互补：Algo 1 在 $n$ 大、$L$ 小时占优（$\sqrt{L}$ 对 $\sqrt{nK}$ 或 $\sqrt{K^n}$），Algo 2 在 $K$ 小时占优，最终遗憾界取两者的最小值。

**3. 动作反馈下的 UCB：把连续策略空间的探索转成区域上的多臂 bandit。**

实际场景里领导者往往只能看到追随者动作、看不到类型（如平台只观测到用户行为而非偏好），信息量更少。本文的观察是：只要领导者一直在同一区域 $R(W)$ 内下注，观测到的追随者动作就来自同一条件分布 $\mathcal{P}(\cdot \mid R(W))$，因而能估计该区域内任意策略的效用。于是把每个区域当成一条"臂"，维护 UCB 分数 $\mathrm{UCB}^t(W) = \hat{u}^*_{R(W)} + \sqrt{4(L+1)\log(3T)/N^t(W)}$，每轮挑分数最高的区域、执行其经验最优策略——一个多臂 bandit 外壳里套着每臂自己的线性优化子问题。这条路线给出 $O(\sqrt{n^L K^L A^{2L} L \cdot T \log T})$ 的遗憾，对 $L$ 呈指数，但这是 BSG 对 $L$ 本就 NP-Hard 所决定的、计算上无法回避。作为互补，作者还借 Bernasconi et al. (2023) 的技术把 BSG 重述成一个线性规划、再用 OFUL 求解，得到 $O(K^n \sqrt{T} \log T)$ 遗憾，在 $L$ 大、$n$ 小时反而更好。

### 损失函数 / 训练策略

本文是纯在线学习理论工作，无训练损失函数。领导者的目标函数是最大化累积效用（等价于最小化遗憾 $\mathrm{Reg}(T) = \sum_{t=1}^T [U_\mathcal{D}(x^*) - U_\mathcal{D}(x^t)]$）。类型反馈算法基于经验效用最大化（ERM原则），动作反馈算法基于乐观面对不确定性（UCB原则）。关键的理论工具是基于最佳响应区域伪维度的均匀集中不等式，以及独立分布下TV距离与Hellinger距离的关系。

## 实验关键数据

### 主结果：遗憾界汇总

| 反馈设置 | 类型分布假设 | 遗憾上界 | 遗憾下界 | 间隙分析 |
|---------|------------|---------|---------|---------|
| 类型反馈 | 独立类型 | $\tilde{O}(\sqrt{\min\{L, nK\} \cdot T})$ | $\Omega(\sqrt{\min\{L, nK\} \cdot T})$ | 仅差对数因子，几乎最优 |
| 类型反馈 | 一般类型 | $\tilde{O}(\sqrt{\min\{L, K^n\} \cdot T})$ | $\Omega(\sqrt{\min\{L, nK\} \cdot T})$ | 一般类型上界中 $K^n$ vs 下界中 $nK$，存在间隙 |
| 动作反馈 | 任意 | $\tilde{O}(\min\{\sqrt{n^L K^L A^{2L} L}, K^n\} \cdot \sqrt{T})$ | $\Omega(\sqrt{\min\{L, nK\} \cdot T})$ | 上下界间隙显著，是开放问题 |

### 计算复杂度对比

| 算法 | 反馈类型 | 每轮时间复杂度 | 适用条件 |
|------|---------|-------------|---------|
| Algorithm 1（一般类型ERM） | 类型反馈 | $\mathrm{poly}((nKA)^L \cdot L \cdot T)$ | $L$ 小，$n$ 可大 |
| Algorithm 2（独立类型ERM） | 类型反馈 | $\mathrm{poly}((nKA)^L \cdot L \cdot T \cdot K^n)$ | 独立类型，$n$ 和 $K$ 均较小 |
| Algorithm 3（区域UCB） | 动作反馈 | $\mathrm{poly}((nKA)^L \cdot L \cdot T)$ | $L$ 小，$n$ 大 |
| 线性Bandit（OFUL） | 动作反馈 | $\mathrm{poly}(K^n \cdot T)$ | $n$ 小，$K$ 小 |

### 关键发现

- **遗憾界不随 $n$ 指数增长**：在类型反馈+独立类型下，上下界均为 $\Theta(\sqrt{\min\{L, nK\} \cdot T})$，对 $n$ 仅线性依赖。这是最出人意料的结果——联合类型空间 $K^n$ 指数大，但通过学习 $nK$ 个边际分布参数即可重建所需信息
- **$L$ 的角色转换**：传统观点认为 $L$ 增长导致计算困难（NP-Hard），但在统计层面 $L$ 小反而有利——最佳响应区域数对 $L$ 呈指数依赖，$L$ 小意味着区域少，集中不等式的union bound更紧
- **反馈模型的信息鸿沟**：类型反馈下可达近乎最优的 $\tilde{O}(\sqrt{L \cdot T})$，但动作反馈下最优遗憾率目前未确定，存在计算-统计权衡——即使遗憾能做到 $\mathrm{poly}(n, K, L)\sqrt{T}$，运行时间也必须对 $L$ 呈指数（除非P=NP）
- **离线求解的新结果**：作为副产品，证明了BSG在 $L$ 为常数时可多项式时间求解，填补了此前度图谱中仅知NP-Hard（$L$ 增长时）的空白

## 亮点与洞察

- **计算几何与博弈论的桥接**：将追随者最佳响应引起的效用不连续性处理为超平面排列中的区域计数问题，是本文最核心的技术贡献。这个视角使得原本难以处理的非凸非连续优化问题变成了在有限多个多面体区域上各自求解线性规划的标准问题。这种"几何化→离散化→分区域优化"的范式可能对其他效用不连续的在线学习问题也有启发
- **学分布 vs 学效用的本质区别**：虽然联合类型分布 $\mathcal{D}$ 需要 $K^n$ 个参数来描述，但领导者效用函数在每个区域内仅是 $L$ 维线性函数。因此学习效用函数远比学习分布容易——这一洞察打破了"分布估计难度决定学习难度"的直觉，可迁移到其他在线优化场景
- **下界证明的双重归约技巧**：先将分布学习归约为单追随者BSG（得到 $\Omega(\sqrt{\min\{L, K\}T})$），再将单追随者 $nK$ 类型博弈归约为 $n$ 追随者各 $K$ 类型博弈（得到 $\Omega(\sqrt{\min\{L, nK\}T})$），这种两步归约的构造技巧值得借鉴

## 局限与展望

- **无外部性假设的限制**：追随者效用仅依赖自身动作和类型、不受其他追随者影响的假设在某些场景下不成立（如拍卖、拥堵博弈）。引入外部性后，追随者间存在纳什均衡计算问题，最佳响应区域的结构将大幅复杂化
- **追随者短视假设**：假设追随者每轮独立最佳响应而非战略性地考虑长期收益。在实际中训练有素的对手可能故意"隐藏"类型信息以获取长期优势
- **动作反馈的间隙未闭合**：动作反馈的最优遗憾率是本文留下的主要开放问题。是否存在 $\mathrm{poly}(n, K, L)\sqrt{T}$ 的遗憾算法仍未知
- **对抗性类型的扩展**：本文仅考虑随机类型。作者在讨论中猜测"区域集中"技术可推广到对抗性设置（如用EXP3+FTRL替代ERM），但未正式验证

## 相关工作与启发

- **vs Balcan et al. (2015, 2025)**：他们设计了单追随者BSG的在线学习算法，遗憾 $\mathrm{poly}(K)\sqrt{T}$。本文将问题推广到多追随者，核心挑战在于联合类型空间从 $K$ 变为 $K^n$。本文利用最佳响应区域几何避免了指数困难
- **vs Bernasconi et al. (2023)**：他们用对抗线性bandit技术处理多接收者贝叶斯说服（结构上类似多追随者BSG），获得 $\tilde{O}(K^{3n/2}\sqrt{T})$ 遗憾，对 $n$ 指数增长。本文用随机线性bandit（OFUL）优化到 $O(K^n \sqrt{T})$，进一步用区域UCB方法在 $L$ 小时得到 $O(\sqrt{n^L K^L A^{2L} L \cdot T})$
- **vs 分段线性Bandit（Bacchiocchi et al. 2025）**：他们研究一维分段线性bandit且区间位置未知，而本文是多维、区域位置已知的情形，技术和结果互补

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统处理多追随者BSG的在线学习，几何化最佳响应区域的技术框架是全新的
- 实验充分度: ⭐⭐⭐ 纯理论工作，附录有简单模拟验证但无大规模实验；不过理论结果（上下界几乎匹配）本身已具有强说服力
- 写作质量: ⭐⭐⭐⭐ 从单追随者逐步推进到多追随者的叙述逻辑清晰，关键技术思路解释充分
- 价值: ⭐⭐⭐⭐ 对博弈论与在线学习交叉领域有重要理论贡献，几何分割方法具有可迁移性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICML 2026\] Learning in Structured Stackelberg Games](../../ICML2026/reinforcement_learning/learning_in_structured_stackelberg_games.md)
- [\[ACL 2026\] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](../../ACL2026/reinforcement_learning/the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
