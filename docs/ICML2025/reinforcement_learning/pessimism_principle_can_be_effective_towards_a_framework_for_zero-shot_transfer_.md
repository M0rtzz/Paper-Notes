---
title: >-
  [论文解读] Pessimism Principle Can Be Effective: Towards a Framework for Zero-Shot Transfer RL
description: >-
  [ICML 2025][强化学习][transfer RL] 提出基于悲观主义原则的迁移RL框架：用鲁棒MDP构建目标域性能保守下界作为代理目标优化，设计Averaged Operator和Minimal Pessimism两种代理及分布式算法，确保安全迁移并避免负迁移。 领域现状 领域现状：领域现状: 迁移RL利用源域数据…
tags:
  - "ICML 2025"
  - "强化学习"
  - "transfer RL"
  - "pessimism"
  - "robust MDP"
  - "distributed learning"
  - "negative transfer"
---

# Pessimism Principle Can Be Effective: Towards a Framework for Zero-Shot Transfer RL

**会议**: ICML 2025  
**arXiv**: [2505.18447](https://arxiv.org/abs/2505.18447)  
**代码**: 无  
**领域**: 强化学习 / 迁移学习  
**关键词**: transfer RL, pessimism, robust MDP, distributed learning, negative transfer

## 一句话总结
提出基于悲观主义原则的迁移RL框架：用鲁棒MDP构建目标域性能保守下界作为代理目标优化，设计Averaged Operator和Minimal Pessimism两种代理及分布式算法，确保安全迁移并避免负迁移。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**: 迁移RL利用源域数据为数据匮乏的目标域训练策略。DR是常用方法但优化平均性能代理，多任务学习联合优化多任务。

**现有痛点**: (1) 缺性能保证——DR可能高估目标域性能，部署后产生严重后果；(2) 无法避免负迁移——差异大的源域拉低整体性能。

**核心矛盾**: 代理目标与目标域性能没有理论联系，过于乐观不安全，过于保守性能差。

**本文目标**: 设计保守代理 $f(\pi) \leq V_{P_0}^\pi$ 使迁移策略有性能保证且悲观度可控。

**切入角度**: 鲁棒RL天然悲观——目标域在不确定集内时鲁棒值函数就是下界。

**核心idea**: 悲观原则保证安全+性能单调性；AO代理优于Proximal DR；MP代理避免负迁移。

## 方法详解

### 整体框架
目标域 $\mathcal{M}_0$，$K$ 个源域，$D(P_0, P_k) \leq \Gamma$。对每个源域构建局部不确定集 $\mathcal{P}_k$ 使 $P_0 \in \mathcal{P}_k$，设计保守代理 → 分布式优化 → 可迁移策略。

### 关键设计

1. **悲观原则效果 (Lemma 4.1)**: $V_{P_0}^{\pi^*} - V_{P_0}^{\pi_f} \leq \|\zeta\| = \max_\pi(V_{P_0}^\pi - f(\pi))$。悲观度越小→次优性gap越小→策略越好。改善代理=改善迁移。

2. **Averaged Operator代理 (Sec 5)**: $\mathbf{T}^\pi_{\text{AO}}Q = \frac{1}{K}\sum_k \mathbf{T}^\pi_k Q$。保守性(Thm 5.2)：$V^\pi_{\text{AO}} \leq V^\pi_{P_0}$。优于Proximal DR(Prop 5.4)：$V^\pi_{\bar{\mathcal{P}}} \leq V^\pi_{\text{AO}}$。

3. **MDTL-Avg算法**: 各源域本地鲁棒Bellman更新 $E$ 步→全局平均同步。收敛(Thm 5.7)：$\tilde{O}(1/(TK) + (E-1)\Gamma/T)$，部分线性加速。

4. **Minimal Pessimism代理 (Sec 6)**: $\mathbf{T}^\pi_{\text{MP}}Q = \max_k \mathbf{T}^\pi_k Q$。保守且优于AO(Thm 6.1)：$V^\pi_{\mathcal{P}_k} \leq V^\pi_{\text{AO}} \leq V^\pi_{\text{MP}} \leq V^\pi_{P_0}$。差异大源域自动被覆盖→避免负迁移。

5. **MDTL-Max + MLMC**: $\mathbb{E}[\max_k Q_k] \neq \max_k \mathbb{E}[Q_k]$ 有偏问题用MLMC构造无偏估计(Lemma 6.2)。

## 实验关键数据

### 代理方法对比


### 主实验

| 代理 | 保守 | 效果 | 避免负迁移 | 分布式 |
|------|------|------|-----------|--------|
| 单源域 $V^\pi_{\mathcal{P}_k}$ | ✓ | 最差 | ✗ | ✓ |
| Proximal DR | ✓ | 差 | ✗ | 需共享模型 |
| **AO** | ✓ | 中 | ✗ | ✓ |
| **MP** | ✓ | 最好 | ✓ | ✓(需MLMC) |

### 收敛速率


### 消融实验

| 算法 | 收敛 |
|------|------|
| MDTL-Avg | $\tilde{O}(1/(TK) + (E-1)\Gamma/T)$ |
| MDTL-Max | 同上 |

### 关键发现
- AO > Proximal DR："先平均后悲观"优于"先悲观后平均"
- MP > AO：取max天然筛选最相似源域
- 保守代理同时保证目标域附近扰动环境的表现（鲁棒性bonus）

## 亮点与洞察
- **性能单调性**（Lemma 4.1）是框架基石——改善代理=改善迁移，DR缺乏此保证
- **MP的自动域选择**无需额外域相似性估计，取max本身即筛选
- **分布式隐私保护**：仅共享Q表不暴露原始数据

## 局限与展望
- 需要已知上界 $\Gamma \geq D(P_0, P_k)$，过大的 $\Gamma$ 导致过度保守；极端情况下可设 $\Gamma=1$（TV距离），但策略可能过于保守
- MLMC带来额外计算开销，虽可通过threshold-MLMC缓解但增加了实现复杂性；样本复杂度为价格换取更优效果
- 表格型MDP设置，未扩展到函数逼近/连续状态空间/深度RL场景
- 未与实际的sim-to-real迁移基准做数值对比，缺少机器人/自动驾驶等domain的实证验证
- 假设所有源域共享相同的状态空间、动作空间和奖励函数，仅转移核不同——实际中奖励也可能不同
- $E$ 的选择（通信效率vs收敛）需要根据具体问题调参，理论给出的范围可能较保守
- 当源域数量 $K$ 很大时，MP代理的max聚合可能对个别源域的噪声异常敏感

## 相关工作与启发
- **vs DR (Tobin et al., 2017)**: DR缺理论保证可能过于乐观；本文悲观原则保证安全
- **vs Offline RL悲观 (Jin et al., 2020)**: 思路类似但应用不同（域外迁移 vs 分布外估计）
- **vs 联邦学习**: 借鉴局部更新+全局聚合范式，但MP聚合方式是迁移RL特有的

## 评分
- 新颖性: ⭐⭐⭐⭐ 悲观原则在迁移RL中的系统化应用
- 实验充分度: ⭐⭐ 有实验但规模较小
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论层层递进
- 价值: ⭐⭐⭐⭐ 为迁移RL提供理论化可保证的框架

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Zero-Shot Generalization of Vision-Based RL Without Data Augmentation](zero-shot_generalization_of_vision-based_rl_without_data_augmentation.md)
- [\[ICML 2025\] Actor-Critics Can Achieve Optimal Sample Efficiency](actor-critics_can_achieve_optimal_sample_efficiency.md)
- [\[ICML 2025\] Solving Zero-Sum Convex Markov Games](solving_zero-sum_convex_markov_games.md)
- [\[NeurIPS 2025\] Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization](../../NeurIPS2025/reinforcement_learning/dynamics-aligned_latent_imagination_in_contextual_world_models_for_zero-shot_gen.md)
- [\[ICML 2026\] Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards](../../ICML2026/reinforcement_learning/unlocking_zero-shot_geospatial_reasoning_via_indirect_rewards.md)

</div>

<!-- RELATED:END -->
