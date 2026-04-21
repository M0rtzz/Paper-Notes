---
title: >-
  [论文解读] Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits
description: >-
  [AAAI 2026][多目标赌博机] 提出两种消除式算法 LexElim-Out 和 LexElim-In，首次在词典序多目标赌博机中同时解决遗憾最小化（RM）和最优臂识别（BAI）问题，其中 LexElim-In 通过跨目标信息共享突破了单目标问题的已知下界。
tags:
  - AAAI 2026
  - 多目标赌博机
  - 词典序偏好
  - 遗憾最小化
  - 最优臂识别
  - 消除算法
---

# Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits

**会议**: AAAI 2026  
**arXiv**: [2511.05802](https://arxiv.org/abs/2511.05802)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 多目标赌博机, 词典序偏好, 遗憾最小化, 最优臂识别, 消除算法

## 一句话总结

提出两种消除式算法 LexElim-Out 和 LexElim-In，首次在词典序多目标赌博机中同时解决遗憾最小化（RM）和最优臂识别（BAI）问题，其中 LexElim-In 通过跨目标信息共享突破了单目标问题的已知下界。

## 研究背景与动机

多臂赌博机（MAB）是顺序决策的基础框架，主要有两大任务范式：遗憾最小化（RM，最小化累积遗憾）和最优臂识别（BAI，用尽量少的样本找到最优臂）。传统研究聚焦单目标标量奖励，但实际应用中常涉及多个目标且目标间存在层次优先级。例如临床试验中患者安全优先于成本、推荐系统中公平性优先于用户参与度。

词典序赌博机（Lexicographic Bandits）正是为这种层次化决策场景设计的框架：agent 按优先级顺序优化多个目标，高优先级目标必须先满足。现有词典序赌博机研究主要关注 RM 任务，而 BAI 在此框架下尚未被探索。然而在很多场景中（如临床试验），需要在学习过程中降低遗憾的同时准确识别最优臂。这引出了核心研究问题：**能否设计同时统一 RM 和 BAI 的词典序赌博机算法？**

## 方法详解

### 整体框架

论文提出基于消除的算法框架，核心思想是利用置信上界逐步淘汰次优臂。关键定义包括：

- **词典序主导**：臂 $a_1$ 词典序主导 $a_2$，当且仅当存在目标 $i$ 使得前 $i-1$ 个目标上二者相同，第 $i$ 个目标上 $a_1$ 更优
- **词典序最优臂** $a_*$：不被任何其他臂词典序主导的唯一臂
- **遗憾**：$R^i(T) = T \cdot \mu^i(a_*) - \sum_{t=1}^T \mu^i(a_t)$，对每个目标 $i$ 分别定义

### 关键设计

#### 算法一：LexElim-Out（外层消除）

LexElim-Out 采用逐层消除策略，按目标优先级从高到低处理：

1. 对每个目标 $i = 1, \ldots, m$，重复淘汰直到活跃臂集合缩小到已知最优集大小 $|\mathcal{O}_*(i)|$
2. 每轮选择不确定性最大的臂（置信宽度最大者）进行拉取
3. 淘汰条件：经验最优臂的均值与某臂均值之差超过 $2c(a_t)$（两倍置信宽度）
4. 拉取后更新所有目标的经验均值

置信宽度定义为：$c(a) = \sqrt{\frac{4}{n(a)} \log\frac{6Km \cdot n(a)}{\delta}}$

**局限**：逐层处理使得优化低优先级目标时无法利用高优先级目标的信息，早期对低优先级目标的探索是随机的。

#### 算法二：LexElim-In（内层消除）

LexElim-In 采用创新的跨目标消除策略，在每一轮中同时利用所有目标的信息：

1. 每轮在所有 $m$ 个目标上进行嵌套过滤：$\mathcal{A}_t^0 = \mathcal{A}_t$，对 $i = 1, \ldots, m$
2. 淘汰阈值随目标层级几何增长：$(2 + 4\lambda + \cdots + 4\lambda^{i-1}) \cdot c(a_t)$
3. 参数 $\lambda \geq 0$ 刻画目标间的冲突程度
4. 最终活跃集 $\mathcal{A}_{t+1} = \mathcal{A}_t^m$

缩放因子的设计确保低优先级目标在提供更强消除信号时可以加速淘汰，同时不影响高优先级目标的优化。

### 损失函数 / 训练策略

本文为理论分析驱动的算法设计，无传统意义的损失函数。核心理论保证包括：

**LexElim-Out 的遗憾界**（Theorem 1）：对任意目标 $i$，
$$R^i(t) \leq \sum_{j=1}^{i} \sum_{a \in \mathcal{S}(j)} \frac{\gamma^j(\delta) \cdot \Delta^i(a)}{(\Delta^j(a))^2}$$

**LexElim-In 的遗憾界**（Theorem 3）：
$$R^i(t) \leq \sum_{\Delta^i(a) > 0} \min_{j \in [m]} \left\{ \frac{(\Lambda^j(\lambda))^2 \cdot \Delta^i(a) \cdot \gamma^j(\delta)}{(\Delta^j(a))^2} \right\}$$

其中 $\min_{j \in [m]}$ 的存在使得算法可以自适应地利用辅助目标加速学习，突破单目标下界。

**LexElim-In 的极小极大遗憾界**（Corollary 1）：$R^i(t) \leq \widetilde{O}(\Lambda^i(\lambda) \cdot \sqrt{Kt})$，匹配单目标最优速率。

## 实验关键数据

### 主实验（RM 结果）

| 算法 | 目标1遗憾 | 目标2遗憾 | 目标3遗憾 | 增长趋势 |
|------|----------|----------|----------|---------|
| LexElim-In | 最低 | 最低 | 最低 | 次线性 |
| LexElim-Out | 低 | 低 | 低 | 次线性 |
| PF-LEX | 中 | 中 | 中 | $\widetilde{O}(T^{2/3})$ |
| UCBα | 低 | 线性增长 | 线性增长 | 仅目标1次线性 |

设置：$K=10$ 臂，$m=3$ 目标，$T=10000$ 轮，10 次独立运行取平均。

### BAI 样本复杂度

| 算法 | K=10 | K=20 | K=30 | 趋势 |
|------|------|------|------|------|
| LexElim-In | 最优 | 最优（优势增大） | 最优（优势显著） | 随K增大优势扩大 |
| LexElim-Out | 次优 | 次优 | 次优 | — |
| UCBα | 中等 | 中等 | 中等 | — |
| EGE | 最差 | 最差 | 最差 | — |

### 消融实验

**跨目标加速分析**：当目标间冲突程度 $\lambda$ 不同时：
- $\lambda = 0$（无冲突）：第二目标的大间隔可以高效淘汰次优臂
- $\lambda = 1$（有冲突）：仅距最优臂较远的臂能被快速淘汰，因为第二目标的置信阈值被缩放为 $2 + 4\lambda = 6$

### 关键发现

1. LexElim-Out 和 LexElim-In 在所有目标上均实现次线性遗憾，而单目标方法 UCBα 仅在第一目标上表现良好
2. LexElim-In 的 BAI 性能随臂数增加优势更加显著，因为低优先级目标的大间隔提供了更强的消除信号
3. PF-LEX 虽为多目标方法但收敛速度慢（$O(T^{2/3})$ vs $O(\sqrt{T})$）

## 亮点与洞察

1. **突破单目标下界**：LexElim-In 通过跨目标信息共享，在 RM 和 BAI 上均可突破经典单目标问题的下界 $\Omega(\sum 1/\Delta(a))$，这是该领域首次展示多目标结构的正向效益
2. **理论优雅性**：遗憾界中的 $\min_{j \in [m]}$ 自然地捕获了跨目标加速机制，某个目标的大间隔可以"帮助"其他目标加速淘汰
3. **首次统一框架**：首个在词典序赌博机中同时处理 RM 和 BAI 的算法框架，填补了理论空白

## 局限与展望

1. LexElim-In 需要先验知识 $\lambda$（刻画目标间冲突程度），消除此依赖是重要的未来方向
2. LexElim-Out 需要已知 $|\mathcal{O}_*(i)|$（各层最优臂集大小），实际中可能难以获取
3. 实验仅在合成数据上进行，缺乏真实应用场景的验证
4. 尚未建立词典序 RM 和 BAI 的紧下界以完全刻画目标间交互

## 相关工作与启发

- **与 UCBα 的关系**：UCBα 统一了单目标 RM 和 BAI，本文将此推广到多目标词典序设置
- **与 PF-LEX 的关系**：PF-LEX 仅处理词典序 RM 且遗憾为 $O(T^{2/3})$，本文实现 $O(\sqrt{T})$
- **启发**：跨目标加速机制可推广到其他多目标在线学习问题，如 Pareto 赌博机或上下文赌博机

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次统一词典序 RM+BAI，突破单目标下界
- 实验充分度: ⭐⭐⭐ — 仅合成实验，缺乏真实场景
- 写作质量: ⭐⭐⭐⭐ — 理论严谨，结构清晰
- 价值: ⭐⭐⭐⭐ — 理论贡献突出，填补重要空白

<!-- RELATED:START -->

## 相关论文

- [Deep (Predictive) Discounted Counterfactual Regret Minimization](deep_predictive_discounted_counterfactual_regret_minimization.md)
- [Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits](../../ICLR2026/reinforcement_learning/online_minimization_of_polarization_and_disagreement_via_low-rank_matrix_bandits.md)
- [Perturbing Best Responses in Zero-Sum Games](perturbing_best_responses_in_zero-sum_games.md)
- [Simultaneous Swap Regret Minimization via KL-Calibration](../../NeurIPS2025/reinforcement_learning/simultaneous_swap_regret_minimization_via_kl-calibration.md)
- [Comparing Uniform Price and Discriminatory Multi-Unit Auctions through Regret Minimization](../../NeurIPS2025/reinforcement_learning/comparing_uniform_price_and_discriminatory_multi-unit_auctions_through_regret_mi.md)

<!-- RELATED:END -->
