---
title: >-
  [论文解读] Stable Matching with Ties: Approximation Ratios and Learning
description: >-
  [NeurIPS 2025][稳定匹配] 研究有并列偏好的双边匹配市场，提出最优稳定份额(OSS)比率概念衡量公平性，证明稳定匹配分布下OSS-ratio为$\Omega(N)$但一般匹配分布下可达$O(\log N)$（渐近紧），并将离线近似结果扩展到bandit学习场景。
tags:
  - NeurIPS 2025
  - 稳定匹配
  - 偏好并列
  - 近似比
  - OSS-ratio
  - 多臂老虎机
---

# Stable Matching with Ties: Approximation Ratios and Learning

**会议**: NeurIPS 2025  
**arXiv**: [2411.03270](https://arxiv.org/abs/2411.03270)  
**代码**: 暂无  
**领域**: 算法博弈论 / 双边匹配 / Bandit学习  
**关键词**: 稳定匹配, 偏好并列, 近似比, OSS-ratio, 多臂老虎机

## 一句话总结

研究有并列偏好的双边匹配市场，提出最优稳定份额(OSS)比率概念衡量公平性，证明稳定匹配分布下OSS-ratio为$\Omega(N)$但一般匹配分布下可达$O(\log N)$（渐近紧），并将离线近似结果扩展到bandit学习场景。

## 研究背景与动机

双边匹配市场(学生-学校、医生-医院、工人-工作)是经济学和算法设计中的重要问题。经典Gale-Shapley算法在严格偏好下给出工人最优稳定匹配。但现实中**工人可能对某些工作无差异（并列偏好）**——例如会议稿件分配系统TPMS中审稿人用离散等级("Eager"/"Willing"等)评价论文时自然产生并列。

并列偏好带来的核心困难：不同打破并列的方式产生不同的稳定匹配，不同匹配对不同工人最优。**不存在一个匹配同时保证所有工人获得其最优稳定效用**。这在bandit学习场景中更加棘手——当效用估计在统计上不可区分时，现有regret界$O(\ln T/\Delta^2)$中的$\Delta \to 0$导致界发散。

本文的核心问题：对于有并列偏好的市场，**能否通过匹配分布来近似每个工人的最优稳定份额(OSS)，同时保证所有工人至少获得一定比例的效用？**

## 方法详解

### 整体框架

三层递进：(1) 离线已知偏好——分析不同匹配类别(稳定/内部稳定/一般)上的OSS-ratio紧界；(2) 偏好不确定时的鲁棒性——面对效用估计误差的保证；(3) Bandit学习——在线学习偏好的同时追求低regret。

### 关键设计

1. **OSS-ratio定义与下界 (Section 3.1)**: 定义工人$w$的最优稳定份额 $\mathbf{U}^*(w) = \max_{\mu \in \mathcal{S}} \mathbf{U}(w, \mu(w))$，OSS-ratio为 $R_\mathcal{C} = \min_{D \in \Delta(\mathcal{C})} \max_{w} \frac{\mathbf{U}^*(w)}{\mathbf{U}_D(w)}$。**Theorem 1**：存在实例使得稳定匹配分布的$R_\mathcal{S} = \Omega(N)$——构造$N/2$个高技能工人和$N/2$个普通工人，每个稳定匹配只能满足一个普通工人。**Theorem 2**：即使允许非稳定匹配，$R_\mathcal{M} = \Omega(\log N)$——用递归构造使工人数以对数速度快于有价值工作数。

2. **离线近似Oracle——Algorithm 1 (Section 3.2)**: **核心技术贡献**。算法步骤：(a) 将每个工作复制$m$份(duplication index)；(b) 工人按效用排序，同效用时优先选择低复制索引；(c) 在扩展市场上运行GS算法获得稳定匹配$\tilde{\mu}$；(d) 将$\tilde{\mu}$分解为$m$个原始匹配$\tilde{\mu}_1,\ldots,\tilde{\mu}_m$，均匀随机选择。**Theorem 3**：取$m = \lfloor\log_2 N + 2\rfloor$，保证每个工人效用$\geq \mathbf{U}^*(w)/m$，即$R_\mathcal{I} = O(\log N)$。关键证明技巧：构造有向森林——节点代表更偏好稳定匹配的工人，边表示竞争同一工作副本的冲突——利用树结构和稳定性约束归纳证明。**Theorem 4**：算法满足策略防操纵(DSIC)。

3. **$\epsilon$-稳定性与鲁棒性 (Section 4)**: 定义$\epsilon$-稳定：阻塞对的效用增量需超过$\epsilon$。Algorithm 2推广Algorithm 1，保证 $\mathbf{U}_D(w) \geq \mathbf{U}_\epsilon^*(w)/m - \epsilon$。**Theorem 6**：对于矩形不确定集$\mathcal{U}$，取中心点$\hat{\mathbf{U}}$和宽度$\epsilon$运行Algorithm 2，可保证所有工人效用$\geq \mathcal{U}^*(w)/m - \epsilon$。这直接适用于从数据估计效用矩阵的场景(Example 2)。

4. **Bandit学习——ETCO算法 (Section 5)**: 定义$\alpha$-近似稳定regret $Reg_i^\alpha(T) = \alpha T \cdot \mathbf{U}^*(w_i) - \mathbb{E}[\sum_t X_i(t)]$。**Algorithm 3 (ETCO=Explore-Then-Choose-Oracle)**：探索阶段round-robin分配建立置信区间$[LCB_{i,j}, UCB_{i,j}]$；如果所有top-$N$工作的CI不重叠→GS oracle（无并列）；否则→近似oracle（有并列）。**Theorem 7**：大$\Delta_{\min}$时 $Reg_i = O(K\ln T/\Delta_{\min}^2)$（匹配下界）；小$\Delta_{\min}$时 $Reg_i^\alpha = O(\alpha T_0 + T\sqrt{K\ln T/T_0})$。**Theorem 8** (trade-off下界)：如果算法在$\Delta_{\text{rel}} \geq cT^{-1/2+\delta}$时保证次线性regret，则存在$\Delta_{\text{rel}}=0$的实例使近似regret为$\Omega(T^{1-2\delta})$。

### 损失函数 / 训练策略

Bandit设定下每轮观察1-sub-Gaussian奖励$X_i(t)$，均值$\mathbf{U}(w_i, \mu_t(w_i))$。置信界 $UCB_{i,j} = \hat{\mathbf{U}}(i,j) + \sqrt{6\ln T/T_{i,j}}$。探索长度$T_0$是核心超参数：$T_0 = T^{2/3}(K\ln T)^{1/3}$给最优近似regret；$T_0 = T/(2\ln T)$给最优standard regret。

## 实验关键数据

### 理论结果总结

| 匹配类别 $\mathcal{C}$ | OSS-ratio下界 | OSS-ratio上界 | 紧致性 |
|------------------------|--------------|--------------|--------|
| 稳定匹配 $\mathcal{S}$ | $\Omega(N)$ (Thm 1) | $O(N)$ (trivial) | 紧 |
| 内部稳定 $\mathcal{I}$ | $\Omega(\log N)$ (Thm 2) | $O(\log N)$ (Thm 3) | 紧 |
| 一般匹配 $\mathcal{M}$ | $\Omega(\log N)$ (Thm 2) | $O(\log N)$ (Thm 3) | 紧 |

### Bandit学习上下界

| 情景 | 上界(ETCO) | 下界 | 说明 |
|------|----------|------|------|
| $\Delta_{\min} = \tilde{\Omega}(T^{-1/3})$ | $O(K\ln T/\Delta_{\min}^2)$ | $\Omega(N\ln T/\Delta_{\min}^2)$ [52] | 匹配(差常数) |
| $\Delta_{\min} = \tilde{O}(T^{-1/3})$, $T_0=T^{2/3}(\cdot)^{1/3}$ | $O((K\ln T)^{1/3}T^{2/3})$ | — | 最优ETC rate |
| Trade-off: 次线性regret当$\Delta_{\text{rel}} \geq cT^{-1/2+\delta}$ | — | $\exists$ 实例$\Delta_{\text{rel}}=0$: $\Omega(T^{1-2\delta})$ approx-regret | 根本性trade-off |

### 关键发现

- **稳定匹配分布的$\Omega(N)$下界揭示了"单匹配不够"**：需要考虑更广泛的匹配类来保证公平性。
- **$O(\log N)$是跨匹配类别的普遍上界**：duplication index技巧使得GS算法可以产生公平的匹配分布。
- **bandit下界的trade-off**：不仅在计算上无法同时优化两种regret，在统计上也不行——这是匹配bandit领域的新发现。
- **DSIC性质**：Algorithm 1不可被工人通过谎报偏好来操纵，保证了机制设计的正确性。

## 亮点与洞察

- OSS-ratio概念优雅地量化了"每个工人距离其最优有多远"，比最大匹配大小更好地捕捉了公平性
- duplication index是核心技术创新——通过复制工作并让工人按索引打破并列，将并列偏好问题化归为严格偏好
- 有向森林的构造和归纳证明是$O(\log N)$上界的关键，利用稳定性约束限制了冲突树的深度
- Theorem 8的trade-off下界具有独立于匹配问题的bandit理论意义，表明这不是计算限制而是统计本质

## 局限与展望

- 仅考虑单边并列(工人侧)，工作侧也有并列偏好的双边情形更困难
- Algorithm 1中每个工人在$m$个匹配中只被分配一个工作（概率$1/m$），大部分时间闲置
- bandit学习部分仅考虑集中式分配，去中心化设定(工人自行提议)是重要的开放问题
- 未考虑many-to-one或many-to-many匹配的推广
- 实验验证缺失，所有结果均为理论性的

## 相关工作与启发

- 与Irving(1994)的弱/强/超稳定性概念互补——本文关注弱稳定下的公平近似
- 与Freeman et al.(2021)的DEF1/DMMS公平概念正交——本文的OSS-ratio适用于one-to-one市场
- 与Kong et al.(2025)在ICLR的工作直接比较——他们用悲观regret，本文用近似regret更灵活
- bandit trade-off下界的证明技巧(通过单entry差异构造双实例)可能启发其他组合bandit问题

## 核心结果速查

- $R_\mathcal{S} = \Theta(N)$：稳定匹配分布下OSS-ratio是线性的
- $R_\mathcal{M} = R_\mathcal{I} = \Theta(\log N)$：一般/内部稳定匹配下是对数的
- Algorithm 1: 复制工作$m$份 + GS → 均匀分布over内部稳定匹配，DSIC
- ETCO: 大gap时$O(K\ln T/\Delta^2)$, 小gap时$O(T^{2/3})$ approx-regret
- Trade-off: 保证大gap次线性regret → 必有$\Delta_{\text{rel}}=0$实例suffer $\Omega(T^{1-2\delta})$

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ OSS-ratio概念和duplication index技巧均为原创；trade-off下界是独立的理论贡献
- 实验充分度: ⭐⭐⭐ 纯理论论文，无实验验证
- 写作质量: ⭐⭐⭐⭐ 定义和定理陈述清楚，但证明主要在附录，正文对intuition解释充分
- 价值: ⭐⭐⭐⭐⭐ 完整解决了并列偏好匹配的近似比问题，建立了bandit设定的基础框架

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)
- [\[ICML 2025\] Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation](../../ICML2025/others/sassha_sharpness-aware_adaptive_second-order_optimization_with_stable_hessian_ap.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[ICML 2025\] Learning Distances from Data with Normalizing Flows and Score Matching](../../ICML2025/others/learning_distances_from_data_with_normalizing_flows_and_score_matching.md)
- [\[ICML 2025\] Score Matching with Missing Data](../../ICML2025/others/score_matching_with_missing_data.md)

<!-- RELATED:END -->
