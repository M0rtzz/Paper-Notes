---
title: >-
  [论文解读] Human-Inspired Multi-Level Reinforcement Learning
description: >-
  [NeurIPS 2025][rating-based RL] 本文提出 RbRL-KL，在 rating-based RL 基础上增加 KL 散度驱动的策略损失项，利用不同评分等级的失败经验以不同权重推开当前策略，在 6 个 DeepMind Control 环境中超越标准 RbRL。
tags:
  - NeurIPS 2025
  - rating-based RL
  - KL divergence
  - human feedback
  - multi-level learning
  - reward-free RL
---

# Human-Inspired Multi-Level Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2501.07502](https://arxiv.org/abs/2501.07502)  
**代码**: 无  
**领域**: reinforcement_learning  
**关键词**: rating-based RL, KL divergence, human feedback, multi-level learning, reward-free RL

## 一句话总结
本文提出 RbRL-KL，在 rating-based RL 基础上增加 KL 散度驱动的策略损失项，利用不同评分等级的失败经验以不同权重推开当前策略，在 6 个 DeepMind Control 环境中超越标准 RbRL。

## 研究背景与动机

**领域现状**：在无奖励环境中，RLHF 通过人类反馈推断奖励。PbRL 用偏好对比，RbRL 用评分标注学习奖励。

**现有痛点**：RbRL 只将评分用于奖励学习，浪费了不同评级中蕴含的策略方向信息。

**核心矛盾**：不同性能等级的失败经验被一视同仁，但人类会区分——完全miss球与出界是不同严重程度的错误。

**本文目标** 在策略学习中直接利用多级评分信息，让策略按不同程度远离不同性能等级的失败经验。

**切入角度**：用 KL 散度度量当前策略与不同评级经验的分布相似性，以递减权重惩罚。

**核心 idea**：以 KL 散度为基础的分级策略损失，让 RL 智能体像人类一样从多级失败经验中提取方向信息。

## 方法详解

### 整体框架

RbRL-KL 在标准 RbRL 上加入第三个通道：低级信息提取（RbRL 奖励学习）、高级信息提取（KL 散度策略方向）、联合训练。

### 关键设计

1. **评分缓冲区分级存储**:

    - $n$ 个评分类存入独立缓冲区 $R_0, \ldots, R_{n-1}$
    - 最高评级不参与 KL 损失，其余视为性能各异的"失败"

2. **多元高斯表示**:

    - 各评级轨迹集和当前策略轨迹用多元高斯 $\mathcal{N}(\mu, \Sigma)$ 参数化

3. **分级 KL 散度策略损失**:

    - 核心公式：$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log(\pi_\theta) \hat{R}(\sigma_\theta)] - \nabla_\theta \sum_{i=0}^{n-2} \omega_i D_{KL}(D_i \| D_{\pi_\theta})$
    - KL 散度用多元高斯解析式：$D_{KL}(P\|Q) = \frac{1}{2}(\text{Tr}(\Sigma_Q^{-1}\Sigma_P) + (\mu_P-\mu_Q)^T \Sigma_Q^{-1}(\mu_Q-\mu_P) + \ln\frac{\det\Sigma_Q}{\det\Sigma_P})$
    - 权重 $\omega_0 > \omega_1 > \cdots > \omega_{n-2}$，低评级惩罚更大
    - 第一项是标准策略梯度，第二项将策略从不同级别的差行为中"推开"

### 训练策略
- 先用 $M$ 轮收集评分训练奖励预测器，后续联合两项损失更新策略
- KL 损失模块化添加，不修改原有 RbRL 框架

## 实验关键数据

### 主实验（6 个 DeepMind Control 环境）

| 环境 | RbRL(n=4) | RbRL-KL(n=4) | 提升% | RbRL(n=6) | RbRL-KL(n=6) | 提升% |
|------|-----------|-------------|-------|-----------|-------------|-------|
| Cartpole | 402.55 | **417.54** | +3.7 | 306.92 | **381.79** | +24.4 |
| Ball-in-cup | 789.30 | **861.47** | +9.1 | 828.62 | **873.92** | +5.5 |
| Finger-spin | 511.55 | **579.27** | +13.2 | 559.73 | **646.37** | +15.5 |
| HalfCheetah | 238.99 | **337.04** | +41.0 | 235.46 | **303.88** | +29.1 |
| Walker | 606.14 | **742.05** | +22.4 | 797.90 | **825.18** | +3.4 |
| Quadruped | 308.48 | **477.29** | +54.7 | 199.83 | **306.78** | +53.5 |

### 不同评分类数的提升百分比

| 环境 | n=3 | n=4 | n=5 | n=6 |
|------|-----|-----|-----|-----|
| Cartpole | +15.5% | +3.7% | +22.5% | +24.4% |
| HalfCheetah | +60.0% | +41.0% | +45.2% | +29.1% |
| Quadruped | -7.5% | +54.7% | +226.0% | +53.5% |

### 关键发现
- 高复杂度环境（HalfCheetah, Walker, Quadruped）提升显著
- 低评分类数（n=3）偶有负增益：失败经验分组太粗，KL 惩罚过于均匀
- 统一超参（$\omega_i$ 按 $2^{-i}$ 递减）所有环境通用

## 亮点与洞察
- **人类学习类比**：分级 KL 惩罚形式化了"从不同错误中学不同教训"的直觉
- **模块化设计**：即插即用，可与 PPO/DDPG/SAC 结合
- **多元高斯近似**简洁有效，避免复杂分布估计
- 可迁移到 preference-based RL 中的分级惩罚

## 局限与展望
- $\omega_i$ 手动设定，缺乏自适应机制
- 多元高斯假设对高维多模态分布可能不准
- 评分类数 $n$ 的最优选择依赖环境
- 实验环境相对简单

## 相关工作与启发
- **vs RbRL (White et al. 2024)**：原始 RbRL 仅用评分学奖励，本文增加策略方向学习通道，两者互补
- **vs PbRL (Christiano et al. 2017)**：PbRL 用偏好对不能评价单个样本绝对质量，RbRL-KL 利用绝对评分的多级信息
- **vs Wu et al. (2024) 负面经验 RL**：他们只用统一的失败经验惩罚，本文区分不同级别的失败，更精细
- **vs DQfD/DDPGfD**：它们用专家示范参与 replay，本文用非专家的多级失败经验做策略塑形
- **vs NAC (Gao et al. 2018)**：用噪声示范初始化策略后微调，本文持续利用多级信息

### 超参数设置

| 参数 | 值 | 说明 |
|------|------|------|
| Clip $\epsilon$ | 0.4 | PPO clip 参数 |
| Learning rate $\alpha$ | 5e-5 | 所有环境统一 |
| Batch size | 128 | 所有环境统一 |
| Hidden layers | 3 | 所有环境统一 |
| $\omega_0$ | 1.0 | 最低评级权重 |
| $\omega_1$ | 0.5 | 指数递减 |
| $\omega_2$ | 0.25 | 指数递减 |

## 评分
- 新颖性: ⭐⭐⭐⭐ 分级 KL 损失直觉新颖，但技术上是已有组件组合
- 实验充分度: ⭐⭐⭐ 6 环境 10 seeds 尚可，缺消融和更复杂 benchmark
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式完整，动机形象
- 价值: ⭐⭐⭐⭐ 简洁有效地利用多级反馈，对 RLHF 有参考意义

<!-- RELATED:START -->

## 相关论文

- [Learning Human-Like RL Agents through Trajectory Optimization with Action Quantization](learning_human-like_rl_agents_through_trajectory_optimization_with_action_quanti.md)
- [EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)
- [Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [Visual Grounding for Object-Level Generalization in Reinforcement Learning](../../ECCV2024/reinforcement_learning/visual_grounding_for_object-level_generalization_in_reinforcement_learning.md)
- [Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)

<!-- RELATED:END -->
