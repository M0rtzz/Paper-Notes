---
title: >-
  [论文解读] Actor-Critics Can Achieve Optimal Sample Efficiency
description: >-
  [ICML 2025][Actor-Critic] 本文首次证明 Actor-Critic 算法在一般函数逼近（general function approximation）和需要策略性探索的设定下可以达到 $O(1/\epsilon^2)$ 的最优样本复杂度，通过整合乐观探索、离策略 Critic 估计和稀疏策略切换，并将结果扩展到混合 RL 设定。
tags:
  - ICML 2025
  - Actor-Critic
  - 样本复杂度
  - Bellman Eluder维度
  - 混合RL
  - 一般函数逼近
---

# Actor-Critics Can Achieve Optimal Sample Efficiency

**会议**: ICML 2025  
**arXiv**: [2505.03710](https://arxiv.org/abs/2505.03710)  
**代码**: 无  
**领域**: 强化学习理论  
**关键词**: Actor-Critic, 样本复杂度, Bellman Eluder维度, 混合RL, 一般函数逼近

## 一句话总结

本文首次证明 Actor-Critic 算法在一般函数逼近（general function approximation）和需要策略性探索的设定下可以达到 $O(1/\epsilon^2)$ 的最优样本复杂度，通过整合乐观探索、离策略 Critic 估计和稀疏策略切换，并将结果扩展到混合 RL 设定。

## 研究背景与动机

**领域现状**：Actor-Critic 算法是 RL 领域的核心方法，融合了基于策略（policy-based）和基于值函数（value-based）两种范式的优势。在实践中（如 PPO、SAC），Actor-Critic 取得了巨大成功。在理论方面，近年来对其统计效率的理解也取得了进展。

**现有痛点**：尽管理论有进展，但一个关键的开放问题始终未解决——**在一般函数逼近和需要策略性探索（strategic exploration）的设定下，没有任何已有 Actor-Critic 算法能以 $O(1/\epsilon^2)$ 条轨迹（trajectory）的样本复杂度学到 $\epsilon$-最优策略**。这是 RL 理论中的最优速率，但已有 Actor-Critic 方法要么只在简单设定（如 tabular）下达到，要么在一般函数逼近下有次优的速率。

**核心矛盾**：Actor-Critic 的理论分析面临独特挑战——Actor（策略）和 Critic（值函数）的耦合更新使得分析困难。特别是：（1）策略更新频率如何影响样本效率？（2）如何在 off-policy 估计中保证 Critic 的准确性？（3）如何在一般函数类上实现有效的乐观探索？

**本文目标**：解决上述开放问题——设计一个 Actor-Critic 算法，在一般函数逼近 + Bellman Eluder 维度框架下，达到 $O(1/\epsilon^2)$ 的最优样本复杂度。同时解决混合 RL（Hybrid RL）中是否可以去掉乐观性的另一个开放问题。

**切入角度**：结合三个关键技术——（1）乐观探索（optimism），（2）针对最优 Q 函数的离策略 Critic 估计（off-policy critic estimation），（3）稀疏策略切换（rare-switching policy resets）。

**核心 idea**：通过让 Critic 直接估计最优 Q 函数（而非当前策略的 Q 函数）并结合乐观上界和稀疏策略切换，打破了 Actor-Critic 分析中的耦合难题，首次实现了最优样本效率。

## 方法详解

### 整体框架

算法框架：

1. **Critic 更新**：使用离策略数据构建最优 Q 函数的乐观估计
2. **Actor 更新**：基于 Critic 的估计，选择使 Q 值最大化的策略
3. **稀疏切换**：Actor 不是每步都更新，而是仅在满足特定条件时才切换策略

关键输出：$\epsilon$-最优策略 $\pi$，满足 $V^* - V^\pi \leq \epsilon$

### 关键设计

1. **乐观 Critic 估计（Optimistic Off-Policy Critic）**:

    - **功能**：Critic 不估计当前策略 $\pi_t$ 的 Q 函数，而是直接估计最优 Q 函数 $Q^*$ 的乐观上界
    - **核心思路**：使用函数类 $\mathcal{F}$ 构建置信集合 $\mathcal{F}_t$，从中选取乐观的 Q 函数估计。关键是——所有历史数据（包括由不同策略收集的数据）都可以用来更新 Critic，因为它估计的是与策略无关的 $Q^*$。形式上：
  
    $\hat{Q}_t = \arg\max_{f \in \mathcal{F}_t} f(s_t, \cdot)$
   
   其中 $\mathcal{F}_t$ 是基于置信集合构造的乐观函数类
    - **设计动机**：传统 Actor-Critic 中 Critic 估计当前策略的 Q 函数，导致策略变化时 Critic 需要重新拟合，产生耦合和偏差问题。直接估计 $Q^*$ 消除了这种耦合——无论 Actor 怎么变，Critic 的目标不变

2. **稀疏策略切换（Rare-Switching Policy Resets）**:

    - **功能**：Actor 不频繁更新策略，而是仅在"足够多新信息"积累后才进行策略切换
    - **核心思路**：定义一个切换条件（通常基于数据覆盖的变化量），只有满足条件时才从 Critic 生成新策略。策略切换次数在整个训练过程中是 $O(\log T)$ 级别
    - **设计动机**：频繁策略切换带来 off-policy 偏差——如果策略每步都变，那么旧数据对新策略的价值降低。稀疏切换保证了策略在大部分时间内是稳定的，使得 off-policy Critic 估计更加准确

3. **Bellman Eluder 维度框架**:

    - **功能**：在 Bellman Eluder (BE) 维度 $d$ 的一般函数逼近框架下进行分析
    - **核心思路**：BE 维度是衡量函数类复杂度的指标，统一了 tabular MDP、线性 MDP、低秩 MDP 等多种设定。算法的样本复杂度以 $d$ 为核心参数：
  
    $N = O\left(\frac{dH^5 \log|\mathcal{A}|}{\epsilon^2} + \frac{dH^4 \log|\mathcal{F}|}{\epsilon^2}\right)$
   
   其中 $H$ 是 episode 长度，$|\mathcal{A}|$ 是动作空间大小，$|\mathcal{F}|$ 是 Critic 函数类大小
    - **设计动机**：BE 维度是目前最通用的 RL 复杂度度量之一，在此框架下证明最优性意味着结果可以自动推广到各种特殊 MDP 类型

4. **混合 RL 扩展（Hybrid RL）**:

    - **功能**：利用离线数据初始化 Critic，获得相比纯在线或纯离线更好的样本效率
    - **核心思路**：离线数据提供了一个"暖启动"——用 $N_\text{off}$ 条离线轨迹初始化 Critic 的置信集合，然后在线阶段只需更少的采样。特别地，如果离线数据量满足 $N_\text{off} \geq c^*_\text{off} d H^4 / \epsilon^2$（其中 $c^*_\text{off}$ 是单策略可达性系数），则可以完全去掉乐观性（optimism），使用**非乐观**的 Actor-Critic 也能达到最优效率
    - **设计动机**：这解决了 Hybrid RL 中的另一个开放问题——在有离线数据的情况下，是否可以避免乐观探索（乐观探索在实践中通常很难实现）

### 理论结果

- **在线 RL**：样本复杂度 $O(dH^5\log|\mathcal{A}|/\epsilon^2 + dH^4\log|\mathcal{F}|/\epsilon^2)$，对应 $\tilde{O}(\sqrt{T})$ 的遗憾（regret）
- **Hybrid RL（乐观版）**：利用离线数据减少在线采样量
- **Hybrid RL（非乐观版）**：以额外的离线数据需求换取去除乐观性

## 实验关键数据

### 主实验（理论对比）

| 算法 | 样本复杂度 | 一般函数逼近 | 策略性探索 | Actor-Critic |
|------|-----------|-------------|-----------|--------------|
| GOLF (value-based) | $O(1/\epsilon^2)$ | ✓ | ✓ | ✗ |
| 已有 AC 方法 | $O(1/\epsilon^3)$ 或更差 | 部分 | 部分 | ✓ |
| **本文 (OPAC)** | $O(1/\epsilon^2)$ | ✓ | ✓ | **✓** |

### 消融/比较实验（数值实验）

| 设定 | 本文方法 | Baseline AC | 说明 |
|------|---------|------------|------|
| Tabular MDP | 最优收敛 | 次优收敛 | 验证理论的 $O(1/\epsilon^2)$ |
| Linear MDP | 最优收敛 | 次优 | 一般函数逼近的特例 |
| Hybrid RL (有离线数据) | 加速收敛 | 无法利用 | 离线数据初始化有效 |
| 非乐观 Hybrid RL | 可行 | 不可行 | 充足离线数据可去除乐观性 |

### 关键发现

1. **首次达到最优速率**：在一般函数逼近 + 策略性探索的完整设定下，Actor-Critic 首次达到了与纯 value-based 方法相同的 $O(1/\epsilon^2)$ 最优样本复杂度
2. **稀疏切换是关键**：$O(\log T)$ 次策略切换平衡了 off-policy 偏差和信息利用效率
3. **离策略 Critic 的解耦效果**：直接估计 $Q^*$ 而非 $Q^\pi$ 是打破 Actor-Critic 耦合分析瓶颈的关键技术
4. **Hybrid RL 的实用意义**：离线数据可以"购买"去掉乐观性的权利，这对实际应用很重要（乐观探索在实践中难以实现）

## 亮点与洞察

- **解决重要开放问题**：Actor-Critic 方法的最优样本效率和 Hybrid RL 中去除乐观性，都是 RL 理论中被多篇论文提及的开放问题
- **技术优雅**：三个关键设计（乐观 Critic、稀疏切换、$Q^*$ 估计）各自解决一个分析瓶颈，组合起来恰好达成最优
- **理论与实践的桥梁**：虽然是理论论文，但稀疏切换和利用离线数据去除乐观性的思想对实际 Actor-Critic 算法设计有指导意义

## 局限与展望

1. 纯理论工作，数值实验仅验证了基本设定，没有在大规模或实际 RL 问题上测试
2. 关于 $H$ 的依赖（$H^5$）可能不是最紧的，存在进一步优化的空间
3. 需要已知函数类 $\mathcal{F}$ 和 $\mathcal{A}$，在实践中寻找合适的函数类仍是挑战
4. 乐观性的构造在实践中难以精确实现
5. 连续动作空间的扩展需要进一步研究

## 相关工作与启发

- **GOLF (Jin et al.)**：在 BE 维度框架下达到了 $O(1/\epsilon^2)$ 但不是 Actor-Critic 方法
- **PEVI (Jin et al.)**：离线 RL 的理论分析，本文的 Hybrid RL 扩展与之相关
- **OPPO / PC-PG**：之前的 Actor-Critic 理论工作，但样本复杂度次优
- **启发**：Actor-Critic 的理论分析中，**Critic 估计什么**（$Q^*$ vs $Q^\pi$）可能比**如何估计**更重要

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 解决了两个明确的开放问题，技术路线有实质创新
- 实验充分度: ⭐⭐⭐ 理论论文的实验规模合适，但缺少大规模验证
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，问题动机和贡献阐述清晰
- 价值: ⭐⭐⭐⭐⭐ 对 RL 理论有重要推动，解决了长期开放问题

<!-- RELATED:START -->

## 相关论文

- [Optimal and Practical Batched Linear Bandit Algorithm](optimal_and_practical_batched_linear_bandit_algorithm.md)
- [Pessimism Principle Can Be Effective: Towards a Framework for Zero-Shot Transfer RL](pessimism_principle_can_be_effective_towards_a_framework_for_zero-shot_transfer_.md)
- [Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)
- [The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability](the_sample_complexity_of_online_strategic_decision_making_with_information_asymm.md)
- [Flow Actor-Critic for Offline Reinforcement Learning (FAC)](../../ICLR2026/reinforcement_learning/flow_actor-critic_for_offline_reinforcement_learning.md)

<!-- RELATED:END -->
