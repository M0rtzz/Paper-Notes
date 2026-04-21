---
title: >-
  [论文解读] Bandit Learning in Housing Markets
description: >-
  [AAAI 2026][multi-armed-bandit] 本文首次将多臂老虎机（MAB）框架引入住房市场（单边匹配市场），定义了基于核（core）概念的遗憾值，并分别提出去中心化 ETC 和中心化 UCB 两种算法，证明了 $\mathcal{O}(N\log T / \Delta_{\min}^2)$ 的去中心化遗憾上界与匹配的下界，建立了阶最优性。
tags:
  - AAAI 2026
  - multi-armed-bandit
  - housing-market
  - matching
  - online-learning
  - game-theory
---

# Bandit Learning in Housing Markets

**会议**: AAAI 2026  
**arXiv**: [2511.12629](https://arxiv.org/abs/2511.12629)  
**代码**: 无  
**领域**: 其他  
**关键词**: multi-armed-bandit, housing-market, matching, online-learning, game-theory  

## 一句话总结

本文首次将多臂老虎机（MAB）框架引入住房市场（单边匹配市场），定义了基于核（core）概念的遗憾值，并分别提出去中心化 ETC 和中心化 UCB 两种算法，证明了 $\mathcal{O}(N\log T / \Delta_{\min}^2)$ 的去中心化遗憾上界与匹配的下界，建立了阶最优性。

## 研究背景与动机

### 住房市场模型

住房市场（housing market）是经典的单边匹配市场模型，由 Shapley 和 Scarf（1974）提出。在该模型中，$N$ 个参与者各自拥有一个不可分割的物品（"房子"），并对所有物品有自己的偏好排序。目标是找到"核稳定分配"（core-stable allocation），即不存在任何子群体可以通过内部交换让所有成员都更满意。著名的 Top Trading Cycle（TTC）算法可以在已知偏好时高效计算唯一的核匹配。

### 核心痛点：偏好未知

传统住房市场研究假设每个参与者完全了解自己对所有物品的偏好，但现实中这往往不成立：

- **租房交换平台**（如 LeaseSwap NYC）：租客未亲自体验过所有房源
- **肾脏交换**：器官匹配的兼容性需要通过医学测试逐步确认
- **NFT 交易**：交易者对未体验过的数字资产价值不确定

这些场景的共同特点是参与者可以通过**多轮短期交互**（如看房、兼容性测试、临时许可）获得即时反馈来逐步学习偏好，天然适合 bandit 学习范式。

### 与现有工作的差异

已有的在线匹配学习主要关注**双边匹配市场**（bipartite matching，如 Gale-Shapley 框架下的稳定匹配），其中双方都有偏好且存在多个稳定匹配。本文研究的是**单边匹配市场**：

1. 只有参与者对物品有偏好，物品没有偏好
2. 碰撞规则不同：多人竞争同一物品时**所有人**都收到零奖励（而非按物品偏好选择最优者）
3. 核匹配是**唯一的**，遗憾定义天然无歧义
4. 基准是博弈论中的"核"而非最大权重匹配

## 方法详解

### 问题形式化

**模型设定：**  $N$ 个玩家 $\mathcal{N} = \{p_1, \ldots, p_N\}$，$N$ 个臂 $\mathcal{A} = \{a_1, \ldots, a_N\}$。玩家 $p_i$ 初始拥有臂 $a_i$。效用矩阵 $\bm{U}(i,j) \in [0,1]$ 表示玩家 $p_i$ 对臂 $a_j$ 的偏好（假设严格不同）。

**交互规则：** 每轮 $t$，每个玩家选择一个臂。若臂 $a_j$ 只收到一个玩家的选择，该玩家获得以 $\bm{U}(i,j)$ 为均值的 1-subgaussian 随机奖励；若多个玩家选择同一个臂，发生碰撞，所有人奖励为 0。

**核遗憾定义：** 以 TTC 算法的唯一输出 $\mu^*$ 为基准，每个玩家 $p_i$ 的核遗憾为：

$$Reg_i(T) = T \cdot \bm{U}(i, \mu^*(p_i)) - \mathbb{E}\left[\sum_{t=1}^T X_i(t)\right]$$

### 整体框架

本文提出两种算法，分别对应去中心化和中心化两种市场设定：

| 维度 | 去中心化 ETC（Algorithm 1） | 中心化 UCB（Algorithm 2） |
|------|---------------------------|--------------------------|
| 是否需要平台 | 否 | 是（聚合偏好排序并计算匹配） |
| 是否需要知道 $T$ | 是 | 否（anytime 算法） |
| 遗憾上界 | $\mathcal{O}(N\log T / \Delta_{\min}^2)$ | $\mathcal{O}(N^2\log T / \Delta_{\min}^2)$ |
| 阶最优性 | 是（匹配下界） | 松一个 $N$ 因子 |

### Algorithm 1：去中心化 Explore-then-YRMH-IGYT

该算法分两个阶段：

**Phase 1 — 探索与偏好学习：** 包含多个子阶段 $\ell = 1, 2, \ldots$，每个子阶段有 $2^\ell + N$ 轮。

- **探索阶段**（$2^\ell$ 轮）：玩家按 round-robin 策略轮流选臂。由于每个玩家有唯一索引，round-robin 保证**零碰撞**。玩家用收集的样本更新效用估计值 $\hat{\bm{U}}(i,j)$ 和 UCB/LCB 置信区间。
- **通信阶段**（$N$ 轮）：利用初始拥有的臂作为广播通道，实现隐式信息传递。玩家 $p_j$ 若已学好偏好排序则向 $p_i$ 的臂提出申请，否则弃权。当 $p_i$ 观察到所有 $N$ 个玩家都提出申请时，推断所有人都已完成偏好学习，触发进入 Phase 2。
- **终止条件：** 对所有臂对 $(a_j, a_{j'})$，满足 $\text{LCB}_{i,\sigma_k} > \text{UCB}_{i,\sigma_{k+1}}$，即置信区间不重叠可以确定完整排序。

**Phase 2 — YRMH-IGYT 去中心化核匹配：** 玩家使用学到的偏好排序 $\sigma_i$ 执行 You Request My House, I Get Your Turn 机制的去中心化版本：

1. 未分配的最小索引玩家发起提案链，向自己最偏好的可用臂提出申请
2. 收到提案的玩家转而向自己的最优可用臂提案，形成链式传递
3. 当提案"绕回"到链中已有的玩家时，识别出一个 top trading cycle
4. 环内所有玩家锁定分配，其拥有的臂从市场移除
5. 重复直到所有臂分配完毕

Phase 2 最多需要 $N^2$ 轮完成（最多 $N$ 个 epoch，每个 epoch 最多 $N$ 轮识别一个 cycle）。

### Algorithm 2：中心化 Anytime UCB

无需知道时间范围 $T$，每轮操作：

1. 每个玩家 $p_i$ 基于 UCB 指标 $u^{(t)}(i,j) = \hat{\bm{U}}^{(t)}(i,j) + \sqrt{3\log t / (2T_{i,j}(t-1))}$ 计算偏好排序
2. 将排序提交给中心平台
3. 平台用离线 TTC 算法计算当前核匹配 $\mu_t$
4. 玩家拉取分配到的臂并更新估计

核心优势是**自适应**且**不需要碰撞处理**（平台协调保证零碰撞），代价是遗憾多一个 $N$ 因子。

### 遗憾下界构造

针对**单一 Top Trading Cycle Bandit（STTCB）**实例类（所有 $N$ 个玩家形成一个大环），通过 KL 散度方法证明：

- 任一均匀一致策略 $\pi$，任一玩家 $p_i$ 满足 $\liminf_{T\to\infty} Reg_i(T;\bm{\nu},\pi) / \log T \geq \Omega(N/\Delta_{\min}^2)$
- 核心构造思路：偏好间距为 $\Delta_{\min}$ 的玩家至少需要 $\Omega(\log T / \Delta_{\min}^2)$ 轮探索才能识别最优臂；而 $N-1$ 个这样的玩家探索时不可避免地给特定玩家造成碰撞，累积导致 $\Omega(N\log T / \Delta_{\min}^2)$ 的遗憾

## 理论结果关键数据

本文为纯理论工作，无实验数据。以下汇总核心理论结果：

**表 1：遗憾界总结**

| 设定 | 算法 | 遗憾上界 | 是否 anytime | 需要知道 $T$ |
|------|------|---------|-------------|-------------|
| 去中心化 | ETC + YRMH-IGYT | $\mathcal{O}(N\log T / \Delta_{\min}^2)$ | 否 | 是 |
| 中心化 | UCB + TTC | $\mathcal{O}(N^2\log T / \Delta_{\min}^2)$ | 是 | 否 |
| 去中心化下界 | 任意一致策略 | $\Omega(N\log T / \Delta_{\min}^2)$ | — | — |

**表 2：技术引理汇总**

| 引理 | 内容 | 关键结论 |
|------|------|---------|
| Lemma 1 | 坏集中事件概率 | $\mathbb{P}(\mathcal{F}) \leq 2N^2 / T$ |
| Lemma 2 | Phase 2 收敛速度 | 最多 $N^2$ 轮找到核匹配 |
| Lemma 3 | Phase 1 持续时间 | 最多 $\ell_{\max} = \log(192N\log T / \Delta_{\min}^2)$ 个子阶段 |
| Lemma 4 | STTCB 遗憾分解 | 将遗憾分解为碰撞项和次优臂拉取项 |

## 亮点与洞察

1. **问题定义的自然性**：将核的博弈论概念与 bandit 的在线学习框架优雅地结合。核匹配的唯一性保证了遗憾定义的无歧义性，避免了双边市场中多个稳定匹配带来的基准选择问题。

2. **Round-robin 零碰撞探索**：利用初始禀赋的唯一索引设计 round-robin 策略，在探索阶段完全避免碰撞，是一个简洁有效的设计。

3. **隐式通信机制**：通过"是否向某人的臂提出申请"传递"我是否学好了偏好"的一比特信息，无需显式通信通道，实现了去中心化下的全局同步。

4. **匹配下界的构造**：巧妙利用单一 TTC 结构——每个玩家的探索必然导致其他特定玩家的碰撞——证明了碰撞的不可避免性，这是多人 bandit 下界证明中的新颖元素。

5. **中心化 vs 去中心化的精确 tradeoff**：去中心化需要知道 $T$ 但阶最优；中心化不需要知道 $T$ 但多一个 $N$ 因子。这个 tradeoff 的根源在于平台可以消除碰撞但引入了额外的"blocking triplet"导致的松弛。

## 局限性

1. **纯理论工作，缺少实验验证**：没有仿真实验验证算法在有限样本下的实际表现，无法直观了解常数因子对性能的影响。

2. **严格偏好假设过强**：要求 $\bm{U}(i,j) \neq \bm{U}(i,j')$ 对所有 $j \neq j'$；当偏好间距 $\Delta_{\min} = o(1/\sqrt{T})$ 时遗憾界退化为平凡。作者在讨论中也承认在有无差异（indifference）时需要新的方法。

3. **去中心化算法需要知道 $T$**：虽然可以用倍增技巧处理，但会导致"非单调的瞬时遗憾"，削弱参与者对算法的信任。

4. **中心化算法遗憾比下界松 $N$ 倍**：$\mathcal{O}(N^2)$ vs $\Omega(N)$，是否能关闭这个 gap 是开放问题。

5. **模型假设较强**：要求全局已知的臂 ID、共享时钟、所有玩家知道可用臂集合等，在实际去中心化系统中可能难以满足。

6. **未讨论策略操纵**：在偏好需要学习的设定下，TTC 的策略防护性（strategy-proofness）是否仍然成立未被分析。

## 相关工作与启发

- **多人 bandit 学习**：Tibrewal et al. (2019)、Mehrabian et al. (2020)、Shi et al. (2021) 等工作关注以最大权重匹配为基准的系统遗憾，本文的核心区别是使用"核"作为基准并关注个体遗憾。
- **双边匹配市场中的 bandit**：Liu et al. (2020, 2021)、Kong and Li (2023) 等研究两方市场中学习稳定匹配，碰撞规则和基准定义均不同。
- **启发**：将博弈论解概念（核、稳定匹配等）作为 bandit 学习的基准是一个有前景的方向，可以推广到其他合作博弈场景，如联盟形成、成本分担等。

## 评分

⭐⭐⭐⭐ (4/5)

**理由：** 问题定义新颖，将住房市场的核概念与 bandit 框架结合具有理论创新性。去中心化算法的上下界完全匹配是强结果。但缺少任何实验验证、严格偏好假设限制了适用范围、中心化算法存在 $N$ 因子的 gap，整体偏重理论而实际影响有待观察。

<!-- RELATED:START -->

## 相关论文

- [Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors](../../ICML2025/others/learning-augmented_algorithms_for_mts_with_bandit_access_to_multiple_predictors.md)
- [How to Marginalize in Causal Structure Learning?](how_to_marginalize_in_causal_structure_learning.md)
- [Why Isn't Relational Learning Taking Over the World?](why_isnt_relational_learning_taking_over_the_world.md)
- [Learning Fair Representations with Kolmogorov-Arnold Networks](learning_fair_representations_with_kolmogorov-arnold_networks.md)
- [Learning Network Dismantling Without Handcrafted Inputs](learning_network_dismantling_without_handcrafted_inputs.md)

<!-- RELATED:END -->
