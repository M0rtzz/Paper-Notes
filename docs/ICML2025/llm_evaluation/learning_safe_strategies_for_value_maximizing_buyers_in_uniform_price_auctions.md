---
title: >-
  [论文解读] Learning Safe Strategies for Value Maximizing Buyers in Uniform Price Auctions
description: >-
  [ICML2025][LLM评测][统一价格拍卖] 针对重复统一价格多物品拍卖中带有RoI约束的价值最大化买家，提出"安全竞标策略"概念，证明其仅需满足温和的不超出竞价条件，并设计多项式时间在线学习算法实现 $\widetilde{O}(M\sqrt{mT})$ 的遗憾界。
tags:
  - "ICML2025"
  - "LLM评测"
  - "统一价格拍卖"
  - "安全竞标策略"
  - "价值最大化"
  - "RoI约束"
  - "在线学习"
  - "遗憾界"
---

# Learning Safe Strategies for Value Maximizing Buyers in Uniform Price Auctions

**会议**: ICML2025  
**arXiv**: [2406.03674](https://arxiv.org/abs/2406.03674)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: 统一价格拍卖, 安全竞标策略, 价值最大化, RoI约束, 在线学习, 遗憾界

## 一句话总结
针对重复统一价格多物品拍卖中带有RoI约束的价值最大化买家，提出"安全竞标策略"概念，证明其仅需满足温和的不超出竞价条件，并设计多项式时间在线学习算法实现 $\widetilde{O}(M\sqrt{mT})$ 的遗憾界。

## 研究背景与动机

### 问题场景
统一价格多物品拍卖（uniform price multi-unit auction）广泛应用于碳排交易（EU ETS）、国债拍卖和电力市场。拍卖中 $K$ 个相同物品出售给 $n$ 个买家，每单位价格统一设定为第 $K$ 高的出价。

### 买家行为模型
与传统拟线性效用（quasilinear utility）不同，本文建模为**价值最大化买家**（value maximizer），其目标是最大化获得的总价值，同时满足每轮**投资回报率（RoI）约束**：

$$V(\mathbf{b};\boldsymbol{\beta}_{-}) \geq (1+\gamma) P(\mathbf{b};\boldsymbol{\beta}_{-})$$

即每轮获得的价值不低于支付的固定倍数。这一模型反映了实际中代理机构（如广告自动竞价系统、碳排中介公司）为客户管理竞标的场景。

### 竞标语言
采用 **$m$-uniform bidding** 格式：买家提交 $m$ 个（出价, 数量）对 $\mathbf{b} = \langle(b_1,q_1),\dots,(b_m,q_m)\rangle$，其中 $b_1 > b_2 > \cdots > b_m > 0$。实际中 $m \ll M$（$M$ 为最大需求量），例如 EU ETS 2023 年拍卖中平均每人提交 ~4.35 对。

### 核心挑战
在重复拍卖的在线设置中，买家在提交策略时不知道竞争对手的出价（可能为对抗性），但仍需在**每轮**都满足 RoI 约束。一个在某些竞争出价下可行的策略，在竞争出价略有变化时可能立即违反约束。

## 方法详解

### 1. 安全竞标策略（Safe Bidding Strategies）

**核心定义**：若一个 $m$-uniform 策略 $\mathbf{b}$ 在**任意**竞争出价 $\boldsymbol{\beta}_{-}$ 下都满足 RoI 约束，则称其为安全策略。

**Theorem 3.1**（安全策略刻画）：安全策略类 $\mathscr{S}_m$ 中不允许超出竞价（overbidding），即：

$$\mathscr{S}_m = \{\mathbf{b} = \langle(b_1,q_1),\dots,(b_m,q_m)\rangle : b_\ell \leq w_{Q_\ell}, \forall \ell \in [m]\}$$

其中 $w_j = \frac{1}{j}\sum_{\ell \leq j} v_\ell$ 为累积平均估值，$Q_j = \sum_{\ell=1}^{j} q_\ell$。

**Theorem 3.2**（安全无支配策略）：消除弱支配策略后，有限策略子集为：

$$\mathscr{S}_m^{\star} = \{\mathbf{b} : b_\ell = w_{Q_\ell}, \forall \ell \in [m]\}$$

这些策略具有"嵌套"结构——第 $j$ 高的出价恰好等于估值曲线前 $Q_j$ 项的均值。关键性质：**仅依赖买家自身估值曲线**，与竞争出价无关。

### 2. 离线最优策略的DAG规约

虽然 $\mathscr{S}_m^{\star}$ 是有限集，但包含 $O(M^m)$ 个策略。作者利用**价值分解引理**（Lemma 4.1），将目标函数分解到各出价-数量对上，并构造有向无环图（DAG）：

- **顶点**：源点 $s$、汇点 $d$，加上 $m$ 层中间节点 $(\ell, j)$
- **边权**：编码策略在各竞争出价下获得的价值
- **关键结论（Theorem 4.2）**：$s$-$d$ 路径与 $\mathscr{U}_m^{\star}$ 中的策略一一对应，求最大权路径即为最优安全策略，可在 $\text{poly}(m, M)$ 时间内完成

### 3. 在线学习算法

基于 DAG 的路径-策略映射，采用 Takimoto & Warmuth (2003) 的**权重推送（weight-pushing）**动态规划方法：

**算法框架**（Algorithm 1）：
1. **UPDATE**：根据上一轮反馈更新边概率，利用底至顶递归计算归一化因子 $\Gamma^{t-1}(u)$，然后按指数加权更新边概率 $\varphi^t(e)$
2. **SAMPLE**：从源点出发，按马尔可夫链采样 $s$-$d$ 路径
3. **MAP**：将路径映射为安全策略并提交

**全信息设置**：直接观察所有竞争出价，按公式设定边权

**Bandit设置**：仅观察分配数量和支付。使用无偏估计器 $\hat{\mathsf{w}}^t(e)$ 替代真实边权，利用半bandit反馈结构（可观测路径上各边奖励）

### 4. 遗憾界

| 设置 | 遗憾上界 | 时间复杂度/轮 |
|------|---------|-------------|
| 全信息 | $\widetilde{O}(M\sqrt{mT})$ | $\text{poly}(m, M)$ |
| Bandit | $\widetilde{O}(M^2 m^{3/2}\sqrt{T})$ | $\text{poly}(m, M)$ |
| **下界** | $\Omega(M\sqrt{T})$ | — |

与此前拟线性效用买家的工作相比，本文方法：(a) 无需离散化出价空间，时间复杂度与 $T$ 无关；(b) bandit 设置下遗憾依赖从 $T^{2/3}$ 改进为 $\sqrt{T}$。

### 5. 对抗更强基准：丰富度比（Richness Ratio）

定义丰富度比 $\alpha \in (0,1]$ 衡量安全策略相对于更丰富策略类的近似程度：

| 基准策略类 | 丰富度比 $\alpha$ | 说明 |
|-----------|------------------|------|
| RoI可行 + $\leq m$ 对 | $1/2$ | 安全性代价有界且独立于 $m$ |
| 安全 + $\leq m'$ 对 ($m' \geq m$) | $m/m'$ | 表达力代价线性增长 |
| RoI可行 + $\leq m'$ 对 | $m/(2m')$ | 两维度代价叠乘 |

所有界均构造了匹配的困难实例证明紧性。

## 实验关键数据

### EU ETS 排放权拍卖半合成数据

使用 2022-2023 年欧盟碳排交易系统拍卖的公开汇总统计数据，合成个体出价进行实验：

- **经验丰富度比 vs RoI可行策略**：当 $m \geq 4$ 时，$\alpha$ 下界超过 **0.95**，远优于理论worst-case的 $1/2$
- **经验丰富度比 vs 更多出价对的安全策略**：即使 $m=10$，1-uniform安全策略所得值与最优10-uniform安全策略之比约为 **0.8**，远优于worst-case的 $0.1$
- 增加出价对数的边际收益在 $m=4$ 后基本平坦

### 关键发现
worst-case紧界出现在高度病态实例中（需指数级大小的精心构造），实际场景中安全策略的表现接近最优。

## 亮点与洞察

1. **安全策略的简洁性**：仅由估值曲线决定，无需知道竞争者信息，具有直观的"不超出竞价"条件和嵌套结构
2. **DAG规约的巧妙性**：将指数大小的策略空间压缩为多项式大小的DAG，使得高效在线学习成为可能
3. **鲁棒性量化**：通过丰富度比系统地评估安全策略对抗更强基准的代价，并证明所有界都是紧的
4. **$\sqrt{T}$ 遗憾**：相比拟线性效用设置下的 $T^{2/3}$，价值最大化设置反而更容易学习
5. **实践适用性**：算法复杂度仅依赖 $m$ 和 $M$（不依赖 $T$），且实验表明很小的 $m$ 即可达到接近最优表现

## 局限与展望

1. **RoI约束的粒度**：主要考虑每轮RoI约束，虽然 Section 6.2 给出了滑动窗口RoI的启发式方法，但缺乏理论保证
2. **遗憾上下界的gap**：全信息下上界 $O(M\sqrt{mT\log M})$ vs 下界 $\Omega(M\sqrt{T})$，差距为 $\sqrt{m\log M}$
3. **自适应对手**：Section 6.1 扩展到自适应对手，但遗憾界退化为高概率界
4. **单一商品假设**：仅考虑多份相同商品的拍卖，未覆盖更一般的组合拍卖场景
5. **估值曲线固定假设**：主模型假设估值固定，时变估值仅作为扩展简要讨论
6. **半合成实验**：由于隐私限制只能使用合成的个体出价数据，无法在真实个体数据上验证

## 相关工作与启发

- **价值最大化文献**：Wilkens et al. (2016) 最早研究价值最大化代理的truthful拍卖设计；Golrezaei et al. (2021c) 在在线广告中验证了RoI软约束的存在
- **多物品拍卖学习**：Brânzei et al. (2023), Galgana & Golrezaei (2024), Potfer et al. (2024) 研究拟线性效用下的在线竞标，本文为首个针对RoI约束价值最大化的系统性工作
- **技术工具**：DAG上的在线学习借鉴了 Takimoto & Warmuth (2003) 的权重推送算法；丰富度比概念与 Streeter & Golovin (2008), Niazadeh et al. (2022) 的近似遗憾框架密切相关
- **实际关联**：EU ETS拍卖（Regulations, 2019）和国债拍卖是本文模型的直接应用场景

## 评分
- 新颖性: ⭐⭐⭐⭐ — 安全策略概念自然但非平凡，DAG规约和丰富度比分析体系完整
- 实验充分度: ⭐⭐⭐ — 半合成实验有效支持理论，但缺少真实数据和在线算法的运行时实验
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，动机、理论、实验环环相扣，管理启示部分有实践思维
- 价值: ⭐⭐⭐⭐ — 为大规模重复拍卖中的自动竞标提供了可落地的理论框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation](../../NeurIPS2025/llm_evaluation/beyond_the_singular_revealing_the_value_of_multiple_generations_in_benchmark_eva.md)
- [\[ACL 2026\] Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies](../../ACL2026/llm_evaluation/teaching_language_models_to_check_grounded_claim_factuality_with_human_test-taki.md)
- [\[ICML 2025\] Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces](function_encoders_a_principled_approach_to_transfer_learning_in_hilbert_spaces.md)
- [\[ICML 2025\] Sample Efficient Demonstration Selection for In-Context Learning](sample_efficient_demonstration_selection_for_in-context_learning.md)
- [\[ICML 2025\] Random Registers for Cross-Domain Few-Shot Learning](random_registers_for_cross-domain_few-shot_learning.md)

</div>

<!-- RELATED:END -->
