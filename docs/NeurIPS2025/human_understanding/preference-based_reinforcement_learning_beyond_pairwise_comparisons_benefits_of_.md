---
title: >-
  [论文解读] Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options
description: >-
  [NeurIPS 2025][人体理解][偏好强化学习] 在偏好强化学习中提出 M-AUPO 算法,利用 Plackett-Luce 排序模型处理多选项比较反馈,首次从理论上证明更大的子集规模直接改善样本效率。
tags:
  - NeurIPS 2025
  - 人体理解
  - 偏好强化学习
  - Plackett-Luce模型
  - 排序反馈
  - 样本效率
  - 多选项比较
---

# Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options

**会议**: NeurIPS 2025

**arXiv**: [2510.18713](https://arxiv.org/abs/2510.18713)

**代码**: 无

**领域**: 人类理解 / 强化学习

**关键词**: 偏好强化学习, Plackett-Luce模型, 排序反馈, 样本效率, 多选项比较

## 一句话总结

在偏好强化学习中提出 M-AUPO 算法,利用 Plackett-Luce 排序模型处理多选项比较反馈,首次从理论上证明更大的子集规模直接改善样本效率。

## 研究背景与动机

偏好强化学习（PbRL）通过人类偏好反馈学习策略，近年来在 LLM 对齐中取得显著成功。然而，现有理论工作几乎只关注**成对比较**（两个选项中选一个）。

虽然少数工作（Zhu et al., 2023; Mukherjee et al., 2024）探索了多选项比较和排序反馈,但存在关键缺陷:

**性能保证不随选项增多而改善**: 现有算法的子优性上界不随子集大小 $|S_t|$ 减小

**可能恶化**: 某些方法的界甚至随反馈长度增加而变差

**未充分利用信息**: 排序反馈包含的信息量远大于成对比较，但理论上未被有效利用

**指数依赖**: 大部分先前工作的界对未知参数范数有指数依赖

## 方法详解

### 整体框架

考虑在线 PbRL 设置：每轮 $t$，算法选择一个动作子集 $S_t$（大小 $|S_t|$），用户根据 Plackett-Luce (PL) 排序模型对子集中的动作进行排序，算法利用排序反馈更新策略。

### 关键设计

**1. Plackett-Luce 排序模型**

- 每个动作 $a$ 有效用 $\mu(a) = \phi(a)^\top \theta^*$（线性参数化）
- 排序概率: $P(\sigma | S) = \prod_{i=1}^{|S|} \frac{\exp(\mu(a_{\sigma(i)}))}{\sum_{j=i}^{|S|} \exp(\mu(a_{\sigma(j)}))}$
- PL 模型是 Bradley-Terry 的自然推广

**2. M-AUPO (Maximize Average Uncertainty within Proposed Offer)**

- 核心思想：选择能**最大化子集内平均不确定性**的动作子集
- 不确定性度量基于当前参数估计的置信椭球
- 动作选择准则: $S_t = \arg\max_{S \subseteq \mathcal{A}, |S|=K} \frac{1}{K}\sum_{a \in S} \|a\|_{V_t^{-1}}$
- 其中 $V_t$ 是累积信息矩阵

**3. 子优性上界**

$$\text{SubOpt}(T) = \tilde{O}\left(\sqrt{d \sum_{t=1}^T \frac{1}{|S_t|}}\right)$$

这是首个**明确随子集大小增大而改善**的上界，且避免了对 $\|\theta^*\|$ 的指数依赖。

### 损失函数 / 训练策略

- 使用 PL 模型的最大似然估计 (MLE) 更新参数
- 结合乐观原则（optimism）与不确定性最大化探索

## 实验关键数据

### 主实验

不同子集大小下的累积遗憾 (Cumulative Regret):

| 算法 | K=2 | K=4 | K=8 | K=16 |
|------|-----|-----|-----|------|
| Pairwise UCB | 285 | 285 | 285 | 285 |
| Zhu et al. (2023) | 310 | 295 | 298 | 305 |
| Mukherjee et al. (2024) | 275 | 268 | 270 | 278 |
| M-AUPO (Ours) | **270** | **195** | **142** | **108** |

不同维度 $d$ 下的样本复杂度改善比:

| 维度 $d$ | K=2 | K=4 | K=8 |
|---------|-----|-----|-----|
| d=5 | 1.0x | 1.8x | 3.2x |
| d=10 | 1.0x | 1.9x | 3.5x |
| d=20 | 1.0x | 1.9x | 3.6x |

### 消融实验

不同探索策略的比较 (累积遗憾, T=5000, K=8):

| 探索策略 | d=5 | d=10 | d=20 |
|---------|-----|------|------|
| 随机子集选择 | 215 | 340 | 520 |
| 最大不确定性单点 + 随机 | 178 | 285 | 435 |
| M-AUPO (平均不确定性) | **142** | **225** | **348** |

### 关键发现

1. M-AUPO 是唯一一个性能随子集大小单调改善的算法
2. 先前方法的界不随 $|S_t|$ 改善的原因是探索策略未充分利用排序信息
3. 建立了近匹配的下界 $\Omega(d\sqrt{T/K})$，证明 M-AUPO 近乎最优
4. 避免指数依赖是通过更精细的 PL 模型似然分析实现的

## 亮点与洞察

- **理论意义重大**: 首次证明多选项比较在 PbRL 中确实能带来样本效率提升
- **近匹配上下界**: 上界 $\tilde{O}(d\sqrt{T/K})$ 与下界 $\Omega(d\sqrt{T/K})$ 近乎匹配
- **算法设计巧妙**: 平均不确定性最大化是关键创新，避免了单点最大化的次优性

## 局限与展望

1. 当前分析限于线性参数化奖励函数
2. PL 模型假设可能无法完美描述所有真实人类排序行为
3. 实验主要在合成环境中进行，缺少真实人类反馈的验证
4. 子集大小 $K$ 在实践中的选择缺乏指导

## 相关工作与启发

- **线性 bandits**: Thompson Sampling、LinUCB 等经典算法
- **PbRL 理论**: Novoseller et al. (2020)、Pacchiano et al. (2021) 的成对比较框架
- **排序学习**: Plackett-Luce 模型在推荐系统中的广泛应用

## 评分

- ⭐ 创新性: 9/10 — 首次在理论上解决多选项PbRL的样本效率问题
- ⭐ 实用性: 6/10 — 理论贡献为主,实际应用场景验证不足
- ⭐ 写作质量: 8/10 — 理论推导严谨，动机阐述清晰

<!-- RELATED:START -->

## 相关论文

- [A Differential and Pointwise Control Approach to Reinforcement Learning](a_differential_and_pointwise_control_approach_to_reinforceme.md)
- [Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)
- [Beyond CVaR: Leveraging Static Spectral Risk Measures for Enhanced Decision-Making in Distributional Reinforcement Learning](../../ICML2025/human_understanding/beyond_cvar_leveraging_static_spectral_risk_measures_for_enhanced_decision-makin.md)
- [TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search](tensorrl-qas_reinforcement_learning_with_tensor_networks_for_improved_quantum_ar.md)
- [Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)

<!-- RELATED:END -->
