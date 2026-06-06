---
title: >-
  [论文解读] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs
description: >-
  [NeurIPS 2025][强化学习][POMDP] 提出基于非马尔可夫 Feynman-Kac 模型的嵌套 SMC（Sequential Monte Carlo）算法，在连续 POMDP 中实现策略优化，天然捕获信息收集价值而无需手工启发式。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "POMDP"
  - "sequential Monte Carlo"
  - "policy gradient"
  - "partial observability"
  - "Feynman-Kac"
---

# Sequential Monte Carlo for Policy Optimization in Continuous POMDPs

**会议**: NeurIPS 2025  
**arXiv**: [2505.16732](https://arxiv.org/abs/2505.16732)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: POMDP, sequential Monte Carlo, policy gradient, partial observability, Feynman-Kac

## 一句话总结
提出基于非马尔可夫 Feynman-Kac 模型的嵌套 SMC（Sequential Monte Carlo）算法，在连续 POMDP 中实现策略优化，天然捕获信息收集价值而无需手工启发式。

## 研究背景与动机
**领域现状**：部分可观测马尔可夫决策过程（POMDP）中的最优决策要求智能体平衡减少不确定性（探索）与追求即时目标（利用）。

**现有痛点**：现有连续 POMDP 策略优化方法要么使用次优近似（如信念点方法），要么依赖手工启发式奖励塑造来鼓励探索。

**核心矛盾**：POMDP 的信念空间是无限维的，直接优化计算不可行；但简化近似会丢失信息收集的价值。

**切入角度**：将策略学习转化为概率推断问题，通过 Feynman-Kac 模型自然编码信息价值。

## 方法详解

### 整体框架
将 POMDP 策略优化映射到非马尔可夫 Feynman-Kac 模型中的概率推断：最优轨迹分布由 POMDP 的奖励结构定义，策略梯度在该分布下通过嵌套 SMC 估计。

### 关键设计
1. **Feynman-Kac 模型构建**

    - 功能：将 POMDP 的值函数编码为 Feynman-Kac 路径积分
    - 核心思路：$\mathcal{Z}_\theta = \mathbb{E}_{\pi_\theta}\left[\prod_{t=0}^T G_t(s_{0:t}, o_{0:t})\right]$
    - 设计动机：FK 模型天然通过预期未来观测来编码信息收集的价值

2. **嵌套 SMC 算法**

    - 外层 SMC：在历史空间中采样轨迹
    - 内层 SMC：对给定历史进行信念更新
    - 核心思路：外层粒子表示不同行为轨迹，内层粒子跟踪信念状态
    - 设计动机：嵌套结构解耦策略评估和信念维护

3. **历史依赖策略梯度**

    - 功能：计算关于策略参数的梯度
    - 核心思路：通过 SMC 采样的轨迹估计 $\nabla_\theta \log \pi_\theta$ 的期望
    - 支持非马尔可夫策略（依赖完整观测历史）

### 训练策略
- 粒子数逐步增加的退火策略
- RNN/Transformer 参数化策略网络以处理历史序列
- 自然梯度版本用于改善收敛

## 实验关键数据

### 主实验：连续 POMDP 基准（累积奖励↑）

| 环境 | QMDP | POMCP | Belief-PPO | RNN-PPO | **FK-SMC** |
|------|------|-------|------------|---------|-----------|
| Tiger | -12.3 | -5.7 | -8.2 | -6.1 | **-3.4** |
| LightDark | 45.2 | 78.3 | 62.1 | 71.5 | **85.7** |
| Navigation | 32.1 | 58.6 | 48.3 | 55.2 | **67.3** |
| Active Sensing | 18.7 | 42.3 | 31.5 | 38.9 | **52.1** |

### 消融实验：SMC 粒子数影响

| 粒子数 (外/内) | LightDark 奖励 | 计算时间(s/iter) |
|----------------|---------------|-----------------|
| 16/16 | 72.3 | 0.8 |
| 32/32 | 79.5 | 2.1 |
| 64/64 | 83.1 | 5.6 |
| **128/128** | **85.7** | **14.2** |
| 256/256 | 86.1 | 38.5 |

### 策略类型消融

| 策略参数化 | LightDark 奖励 | Navigation 奖励 |
|-----------|---------------|-----------------|
| 线性策略 | 61.2 | 38.7 |
| RNN 策略 | 79.8 | 58.3 |
| GRU 策略 | 82.4 | 62.1 |
| **Transformer 策略** | **85.7** | **67.3** |

### 关键发现
- FK-SMC 在所有需要主动信息收集的环境中大幅领先
- Tiger 环境中改进最显著——该环境最需要探索-利用平衡
- 128 粒子在性能与计算间取得最佳平衡
- 非马尔可夫策略（Transformer）的优势在长 horizon 任务中更明显

## 亮点与洞察
- **理论优雅**：用概率推断框架解决 RL 问题，FK 模型天然编码信息价值
- **无需人工启发式**：不需要好奇心奖励、信息增益奖励等手工设计
- 嵌套 SMC 算法在理论上可证明收敛
- 首次将 FK 路径积分引入 POMDP 策略优化

## 局限与展望
- 嵌套 SMC 的计算成本随粒子数平方增长
- 高维观测空间（如图像）的扩展需要结合深度信念模型
- 与 Transformer-based memory 方法的比较不充分
- 连续动作空间的重采样步骤需要特殊处理
- 长 horizon（>100 步）下 SMC 的粒子退化问题

## 相关工作与启发
- POMCP (Silver & Veness 2010) 蒙特卡洛树搜索
- SMC² (Chopin et al. 2013) 嵌套 SMC
- 控制即推断框架 (Levine 2018)
- Dreamer 系列 (Hafner et al. 2020) 世界模型
- 启发：FK 模型可推广到 risk-sensitive RL 和多智能体 Dec-POMDP

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ FK+嵌套SMC的策略优化范式新颖
- 实验充分度: ⭐⭐⭐⭐ 多环境+粒子数消融+策略消融
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 推进POMDP理论与实践

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)
- [\[ICML 2026\] Reinforced Sequential Monte Carlo for Amortised Sampling](../../ICML2026/reinforcement_learning/reinforced_sequential_monte_carlo_for_amortised_sampling.md)
- [\[NeurIPS 2025\] Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown](feel-good_thompson_sampling_for_contextual_bandits_a_markov_chain_monte_carlo_sh.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[NeurIPS 2025\] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)

</div>

<!-- RELATED:END -->
