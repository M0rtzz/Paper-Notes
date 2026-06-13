---
title: >-
  [论文解读] Multimodal Bandits: Regret Lower Bounds and Optimal Algorithms
description: >-
  [NeurIPS 2025][多模态VLM][多模态赌博机] 针对奖励函数至多有 $m$ 个极值的多模态多臂赌博机问题，提出首个计算可行的算法求解 Graves-Lai 优化问题，实现渐近最优的遗憾界，并证明局部搜索策略是次优的。 1. 领域现状 随机多臂赌博机（MAB）问题是在线决策的基础模型。经典设置中…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "多模态赌博机"
  - "Graves-Lai界"
  - "动态规划"
  - "渐近最优"
  - "多臂老虎机"
---

# Multimodal Bandits: Regret Lower Bounds and Optimal Algorithms

**会议**: NeurIPS 2025  
**arXiv**: [2510.25811](https://arxiv.org/abs/2510.25811)  
**代码**: [GitHub](https://github.com/wilrev/MultimodalBandits)  
**领域**: 多模态VLM  
**关键词**: 多模态赌博机, Graves-Lai界, 动态规划, 渐近最优, 多臂老虎机

## 一句话总结

针对奖励函数至多有 $m$ 个极值的多模态多臂赌博机问题，提出首个计算可行的算法求解 Graves-Lai 优化问题，实现渐近最优的遗憾界，并证明局部搜索策略是次优的。

## 研究背景与动机

### 1. 领域现状

随机多臂赌博机（MAB）问题是在线决策的基础模型。经典设置中，学习者对奖励分布无结构化假设。当加入结构化约束（如奖励函数的单峰性），可显著降低探索代价。单峰赌博机（$m=1$）已有多种渐近最优算法（KL-UCB、Thompson Sampling、DMED 等）。

### 2. 现有痛点

**多峰场景**（$m \geq 2$, 即多模态赌博机）远比单峰复杂：
- 混淆参数集 $\mathcal{B}(m, \boldsymbol{\mu})$ 非凸且不连通，不能简单通过凸优化求解
- 将 $\mathcal{B}$ 分解为凸子集的数量指数级增长（如 $K=100, m=5$ 需要 $>10^5$ 个凸分量）
- 先前工作（Saber & Maillard 2024）采用局部搜索策略但未证明其最优性

### 3. 核心矛盾

Graves-Lai 优化问题提供了信息论层面的遗憾下界和最优探索策略，但多模态情况下其约束集结构极其复杂，无法高效求解。没有可行的求解算法，就无法实现渐近最优的赌博机策略。

### 4. 本文目标

设计首个**计算可行**的算法来求解多模态赌博机的 Graves-Lai 优化问题，从而实现渐近最优的在线决策。

### 5. 切入角度

通过系列子问题分解、极值位置的结构性质（Proposition 6）、离散化近似（Proposition 7）和动态规划（Proposition 8）逐步化解非凸约束的计算难题，最终用惩罚次梯度下降求解原问题。

### 6. 核心 idea

利用"混淆参数的极值必然在原始极值集合 $\mathcal{M}(\boldsymbol{\mu}) \cup \{k\}$ 中"这一关键结构性质，将指数级复杂度的搜索转化为树上的多项式时间动态规划。

## 方法详解

### 整体框架

算法流程（如 Figure 1 所示）：
1. **约束分解**：将全局约束分解为 $(k, k')$ 子问题
2. **离散化**：将连续搜索空间离散化为精度 $n$ 的网格
3. **动态规划**：在树上用 DP 求解每个离散化子问题
4. **惩罚次梯度下降**：迭代求解原始 $P_{GL}$

### 关键设计

#### 模块1：约束集分解与简化

**功能**：将复杂非凸约束分解为可求解的子问题。

**核心思路**：

**Proposition 3**：当 $k \in \mathcal{N}(\boldsymbol{\mu})$（极值邻域内）或 $|\mathcal{M}(\boldsymbol{\mu})| < m$ 时，约束有闭式解：
$$\inf_{\boldsymbol{\lambda} \in \mathcal{B}_k(m, \boldsymbol{\mu})} \boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \boldsymbol{\lambda}) = \eta_k d_k(\mu_k, \mu^\star)$$

**Proposition 4**：另一情况可限制在紧集 $[\mu_\star, \mu^\star]^K$ 上搜索。

**Proposition 5**：$\boldsymbol{\eta}^\star \in [0, \mathfrak{B}(\boldsymbol{\mu})]^K$，搜索空间有界。

**设计动机**：逐步限制搜索空间维度和范围，使问题可计算。

#### 模块2：极值位置结构定理

**功能**：确定最优混淆参数的极值位置。

**Proposition 6（关键结构结果）**：
$$\mathcal{M}(\boldsymbol{\lambda}^\star) \subset \mathcal{M}(\boldsymbol{\mu}) \cup \{k\}$$

即最优混淆参数的极值**只能出现在原始奖励函数的极值位置或目标臂 $k$**。

**意义**：当 $|\mathcal{M}(\boldsymbol{\mu})| = m$ 时，必有一个原始极值 $k' \neq k^\star(\boldsymbol{\mu})$ 不再是极值，将 $P_{GL}(k)$ 进一步细分为 $P_{GL}(k, k')$。

**设计动机**：知道极值位置后搜索空间急剧缩小。

#### 模块3：离散化与动态规划

**功能**：在离散化空间上用 DP 精确求解子问题。

**Proposition 7（离散化精度保证）**：
$$\boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \tilde{\boldsymbol{\lambda}}) - \frac{\mathfrak{C}(\boldsymbol{\mu})}{n} \leq \min_{\boldsymbol{\lambda} \in \mathcal{B}_{k,k'}} \boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \boldsymbol{\lambda}) \leq \boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \tilde{\boldsymbol{\lambda}})$$

误差随 $1/n$ 衰减。

**Proposition 8（DP 递推）**：将树 $G$ 以 $k$ 为根形成有向树，定义：
$$f_\ell(z, u) = \text{min obtainable value of } \sum_{j \in \mathcal{D}(\ell) \cup \{\ell\}} \eta_j d_j(\mu_j, \lambda_j)$$

递推公式对极值节点和非极值节点分别处理，单次 DP 复杂度 $O(nK)$。

**设计动机**：树结构使 DP 可行，离散化使搜索有限。

#### 模块4：惩罚次梯度下降

**功能**：求解原始 Graves-Lai 优化问题。

**核心思路**：将硬约束转为惩罚：
$$h(\boldsymbol{\eta}) = \boldsymbol{\eta}^\top \boldsymbol{\Delta} + \gamma \max\left[1 - \min_{\boldsymbol{\lambda}} \boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \boldsymbol{\lambda}), 0\right]$$

投影次梯度下降：
$$\boldsymbol{\eta}(s+1) = \Pi[\boldsymbol{\eta}(s) - \delta(\boldsymbol{\Delta} - \gamma d(\boldsymbol{\mu}, \boldsymbol{\lambda}(s)) \mathbb{1}\{\cdot < 1\})]$$

**Proposition 9**：$t$ 步迭代后误差 $O(1/\sqrt{t})$，离散化误差 $O(1/n)$。

### 损失函数/训练策略（总复杂度）

**定理 2**：算法在 $t$ 次迭代、$n$ 个离散点下：
- 时间复杂度：$O(K^2 m n t)$（可改进到 $O(Knt)$）
- 空间复杂度：$O(Knt)$
- 近似误差：$O(1/n + 1/\sqrt{t})$

给定时间预算 $nt = a$，选 $n = a^{1/3}, t = a^{2/3}$ 得误差 $O(a^{-1/3})$。

## 实验关键数据

### 主实验：多模态 OSSB vs 经典 OSSB

设置：二叉树 $G$（$K=7$ 臂），高斯奖励 $\mathcal{N}(\mu_k, 1)$，双峰奖励函数（$m=2$），极值在节点 4 和 6。

| 设置 | 实例 | Multimodal OSSB | Classical OSSB | 改进 |
|------|------|------|------|------|
| 尖峰 ($\sigma=0.5$) | 容易 | 显著低遗憾 | 高遗憾 | **大幅** |
| 平坦 ($\sigma=4$) | 困难 | 显著低遗憾 | 高遗憾 | **大幅** |

两种设置下多模态 OSSB 均显著优于不考虑结构的经典 OSSB，且在 $T=10000$ 圆内优势持续增大。

### 局部搜索次优性

**定理 3**：局部搜索策略的遗憾与全局策略的比值**无上界**：
$$\sup_{\boldsymbol{\mu} \in \mathcal{F}_m} \frac{C_{loc}(m, \boldsymbol{\mu})}{C(m, \boldsymbol{\mu})} = +\infty$$

具体例子（Figure 2）：$\boldsymbol{\mu} = (1, 2, 4, 2, 3)$
- 局部搜索：$\inf_{\boldsymbol{\lambda}} \boldsymbol{\eta}^\top d(\boldsymbol{\mu}, \boldsymbol{\lambda}) = 0.5$
- 全局最优：$\boldsymbol{\lambda}^\star = (4, 2, 4, 2.8, 2.8)$，值 $\approx 0.145 < 0.5$

局部搜索遗憾多出 **3.4 倍**。

### 关键发现

1. **局部搜索的反直觉结果**：在单峰 ($m=1$) 时局部搜索最优，但多峰时可以任意差。原因是平坦极值附近需要大量样本验证极值性质
2. **$\kappa$-尖峰函数的安全区间**：当奖励函数是 $\kappa$-peaked 时，局部搜索最多比全局差 $\kappa$ 倍（Proposition 11）
3. **纠正先前错误**：证明 Saber & Maillard (2024) 的 IMED-MB 算法的渐近最优性声明不成立
4. **结构利用的价值**：即使在小规模问题 ($K=7$) 上，利用多模态结构也带来实质性遗憾降低

## 亮点与洞察

1. **首个计算可行的算法**：解决了多模态赌博机领域长期存在的计算瓶颈——Graves-Lai 问题的高效求解
2. **优美的分解策略**：从 Proposition 3 到 Proposition 9，每一步都精确地限制搜索空间，最终使指数级问题变为多项式级
3. **Proposition 6 是核心突破**：混淆参数极值位置的刻画使 DP 成为可能
4. **局部搜索次优性的严格证明**：纠正了学界对局部搜索策略的过度乐观，说明求解 $P_{GL}$ 是不可避免的
5. **$\kappa$-peaked 概念**：为判断局部策略何时"足够好"提供了量化标准
6. **理论与算法紧密结合**：不仅给出界，还给出可实现的算法并做实验验证

## 局限与展望

1. 仅考虑**树图**结构，一般图上的多模态赌博机尚未解决
2. 算法复杂度 $O(K^2 mnt)$ 对大规模问题（$K$ 很大）可能仍然昂贵
3. 实验规模较小（$K=7, T=10000$），大规模验证缺乏
4. 求解 $P_{GL}$ 需要在每轮（或对数间隔轮次）更新，实时性存在挑战
5. 离散化精度 $n$ 与迭代次数 $t$ 的最优平衡需要事先知道 $\mathfrak{B}(\boldsymbol{\mu})$ 等问题相关常数
6. 未考虑非平稳或对抗性奖励

## 相关工作与启发

- **与单峰赌博机的关系**：$m=1$ 时退化为经典单峰赌博机，Proposition 3 给出闭式解——本文是其严格推广
- **与 OSSB 算法的关系**：OSSB 是通用渐近最优框架，本文补上了多模态赌博机下求解 $P_{GL}$ 的关键一环
- **与组合赌博机的类比**：组合赌博机用迭代算法求解 $P_{GL}$，多模态赌博机用 DP + 次梯度下降
- **对优化实践的启发**：多模态目标函数（如深度网络损失面）的优化可借鉴本文的多峰结构利用思想
- **定价应用**：多模态赌博机自然对应多模态需求曲线下的定价问题

## 评分

⭐⭐⭐⭐ (4/5)

理论贡献突出：首次给出多模态赌博机的计算可行算法和局部搜索次优性的严格证明。分解策略层层递进，数学功力深厚。扣分点在于实验规模偏小且目前限于树图结构。注意：此论文与"视觉-语言多模态"关系较远，"multimodal"指奖励函数的"多峰"性质。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Learning Optimal Multimodal Information Bottleneck Representations](../../ICML2025/multimodal_vlm/learning_optimal_multimodal_information_bottleneck_representations.md)
- [\[ACL 2025\] Inference Compute-Optimal Video Vision Language Models](../../ACL2025/multimodal_vlm/inference_compute_optimal_video_vlm.md)
- [\[AAAI 2026\] To Align or Not to Align: Strategic Multimodal Representation Alignment for Optimal Performance](../../AAAI2026/multimodal_vlm/to_align_or_not_to_align_strategic_multimodal_representation_alignment_for_optim.md)
- [\[AAAI 2026\] Information Theoretic Optimal Surveillance for Epidemic Prevalence in Networks](../../AAAI2026/multimodal_vlm/information_theoretic_optimal_surveillance_for_epidemic_prevalence_in_networks.md)
- [\[NeurIPS 2025\] Multimodal Negative Learning](multimodal_negative_learning.md)

</div>

<!-- RELATED:END -->
