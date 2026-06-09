---
title: >-
  [论文解读] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning
description: >-
  [NeurIPS 2025][强化学习][multi-agent RL] 通过 82,620 次大规模实验系统性研究合作多智能体 RL 中的鲁棒性和弹性，揭示超参数调优比算法选择更重要，并发现参数共享、GAE、PopArt 等常见做法在不确定性下反而有害，提出一套实用的超参数建议。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "multi-agent RL"
  - "robustness"
  - "resilience"
  - "hyperparameter tuning"
  - "cooperative MARL"
---

# Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.11824](https://arxiv.org/abs/2510.11824)  
**代码**: [github.com/BUAA-TrustworthyMARL/adv_marl_benchmark](https://github.com/BUAA-TrustworthyMARL/adv_marl_benchmark)  
**领域**: 强化学习  
**关键词**: multi-agent RL, robustness, resilience, hyperparameter tuning, cooperative MARL

## 一句话总结
通过 82,620 次大规模实验系统性研究合作多智能体 RL 中的鲁棒性和弹性，揭示超参数调优比算法选择更重要，并发现参数共享、GAE、PopArt 等常见做法在不确定性下反而有害，提出一套实用的超参数建议。

## 研究背景与动机

**领域现状**：合作 MARL 算法（MADDPG、MAPPO、HAPPO）在模拟器中通过调优超参数最大化合作性能取得了显著成功。部署时面临观测噪声、动作扰动、环境不确定性等真实世界问题。

**现有痛点**：(a) 鲁棒性(robustness)和弹性(resilience)在 MARL 文献中经常被混淆——前者是在持续不确定性下维持性能，后者是从扰动中恢复的能力；(b) 超参数对鲁棒性/弹性的影响几乎未被研究；(c) 现有鲁棒 MARL 研究局限于简单模拟环境，缺乏真实世界验证。

**核心矛盾**：为合作性能调优的超参数可能严重损害鲁棒性和弹性（如 Figure 1 所示），但研究者通常不知道哪些超参数有害。

**本文目标**：(a) 正式区分 MARL 中的鲁棒性和弹性；(b) 系统评估合作、鲁棒性、弹性三者的关系；(c) 评估 15 种超参数对三者的影响并给出实用建议。

**切入角度**：控制变量的大规模实证研究——4 个真实世界环境、13 种不确定性类型、15 种超参数、3 种算法、18 个任务、5 个随机种子。

**核心 idea**：通过迄今最大规模的 MARL 鲁棒性实证研究，证明超参数调优比算法选择更重要，并揭示多个违背直觉的发现。

## 方法详解

### 整体框架
基于 Dec-POMDP $\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \mathcal{O}, O, \mathcal{A}, \mathcal{P}, R, \gamma \rangle$，系统性评估合作→鲁棒性→弹性的三阶段流程：(1) 训练合作模型；(2) 在不确定性下评估鲁棒性 $J^{\text{robust}}(\pi)$；(3) 从扰动状态出发评估恢复能力 $J^{\text{resilience}}(\pi)$。

### 关键设计

1. **鲁棒性的形式化定义**:

    - 功能：度量固定策略在持续不确定性下的期望回报
    - 核心思路：$J^{\text{robust}}(\pi) = \mathbb{E}_{u\sim\mathcal{U}}[\mathbb{E}_{s_0\sim\rho_0}\mathbb{E}_{\pi,u}[\sum_{t=0}^{\infty}\gamma^t r_t | s_0]]$，其中 $\mathcal{U}$ 是不确定性集合
    - 设计动机：不确定性在整个 episode 中持续作用，策略 $\pi$ 固定不变

2. **弹性的形式化定义**:

    - 功能：度量策略从扰动状态恢复正常功能的能力
    - 核心思路：$J^{\text{resilience}}(\pi) = \mathbb{E}_{u\sim\mathcal{U}}[\mathbb{E}_{s_u\sim\rho_u}\mathbb{E}_\pi[\sum_{t=t_u}^{\infty}\gamma^t r_t | s_0=s_u]]$——从扰动后的状态 $s_u$ 开始，此后不再有不确定性
    - 设计动机：区别于鲁棒性——弹性关注的是"冲击后恢复"而非"持续抵抗"。例如地震后电网恢复供电的能力
    - 与鲁棒性的区别：一个系统可以弹性好但不鲁棒（从扰动恢复快但持续噪声下性能差），也可以鲁棒但弹性差（能抵抗持续噪声但从异常初始状态恢复慢）

3. **13 种不确定性的系统覆盖**:

    - 观测不确定性(6种)：高斯噪声/贪婪攻击/学习攻击 × 全体智能体($\epsilon=0.1$)/单个智能体($\epsilon=0.2$)
    - 动作不确定性(6种)：随机策略/贪婪策略/学习策略 × 全体/单个，建模为 $\epsilon\hat{\pi} + (1-\epsilon)\pi$
    - 环境不确定性(1种)：从环境参数不确定性集中均匀采样 50 次 rollout 取最差结果

4. **15 种超参数的系统评估**:

    - 通用超参数(10种)：隐藏层大小、折扣因子、激活函数、初始化方法、网络类型、学习率、critic 学习率、特征归一化、参数共享、早停
    - MADDPG 特定(2种)：N-step TD、探索噪声
    - MAPPO/HAPPO 特定(3种)：entropy 系数、GAE、PopArt
    - 每次变一个超参数，默认值固定，产生 34 种模型配置

### 实验环境
4 个真实世界环境：DexHand（灵巧手操作，Isaac Gym）、Quads（四旋翼编队，连续控制）、Traffic（交通信号控制，离散动作）、Voltage（电压控制，连续控制）。共 18 个任务。

### 实验流程
5 随机种子 × 27 不确定性设置（1合作 + 13鲁棒 + 13弹性）× 18 任务 × 34 超参数配置 = **82,620 次实验**，总计约 230K GPU 小时（GTX 4090）。

## 实验关键数据

### 核心发现 1：合作-鲁棒-弹性的相关性

| 不确定性严重程度 | 合作↔鲁棒相关性 | 合作↔弹性相关性 |
|-----------------|----------------|----------------|
| 轻微扰动 | 高 ($r > 0.8$) | 高 ($r > 0.8$) |
| 中等扰动 | 中等 ($r \approx 0.6$) | 中等 |
| 严重扰动 | 低 ($r < 0.4$) | 低 |

线性回归：合作-鲁棒 $r=0.85, p<.001$；合作-弹性 $r=0.76, p<.001$（随攻击严重程度线性下降）

### 核心发现 2：超参数推荐

| 超参数 | 推荐设置 | 合作改善 | 鲁棒改善 | 弹性改善 | 统计检验 |
|--------|---------|---------|---------|---------|---------|
| 早停 | 开启 | ✓ | ✓ | ✓ | $t(161)=6.16, p<.001$ |
| Critic LR | > Actor LR | ✓ | ✓ | ✓ | $t(161)=10.02, p<.001$ |
| 激活函数 | Leaky ReLU | ✓ | ✓ | ✓ | $t(161)=6.31, p<.001$ |
| GAE | 关闭 | ✓ | ✓ | ✓ | $t(107)=7.44, p<.001$ |
| PopArt | 关闭 | ✓ | ✓ | ✓ | $t(107)=4.84, p<.001$ |
| 参数共享 | 关闭（异质环境）| ✓ | ✓ | ✓ | $t(161)=7.01, p<.001$ |
| 探索 (PPO) | 高 entropy | ✓ | ✓ | ✓ | $t(107)=6.64, p<.001$ |

### 综合提升效果

| 评估维度 | 默认超参数 → 最优超参数提升 | 鲁棒方法 + 最优超参数提升 |
|----------|--------------------------|------------------------|
| 合作 | +52.60% | +89.43% |
| 鲁棒性 | +34.78% | +65.83% |
| 弹性 | +60.34% | +82.96% |

### 关键发现
- **超参数比算法更重要**：双因素 ANOVA 显示在 18 个任务中有 9 个超参数效应显著强于算法效应 ($p<0.001$)
- **不确定性类型不可互换**：观测/动作/环境不确定性之间相关性低（单因素 ANOVA 鲁棒性 $F(2,153)=9.53, p<.001$），对一种鲁棒不意味着对另一种鲁棒
- **智能体范围不可互换**：针对所有智能体的防御在单个智能体攻击下可能失败（$F(1,142)=4.36, p<.05$）
- **算法各有所长**：MADDPG 对动作不确定性最鲁棒（持续探索噪声），MAPPO/HAPPO 对观测不确定性最好（中心化 critic）
- **GAE 为何有害**：在真实世界环境中奖励稀疏且高方差，GAE 的 bootstrapping 误差被放大；仅在 MuJoCo 等奖励密集环境有效

## 亮点与洞察
- **规模空前的实证研究**：82,620 次实验、~230K GPU 小时,在 MARL 领域前所未有。每个结论都有系统的统计检验支撑
- **弹性的正式引入**：首次从控制论/生态学引入弹性概念到 MARL，并给出清晰的形式化定义，区分"持续抵抗"和"冲击后恢复"两种能力
- **多个违背直觉的发现**：参数共享、GAE、PopArt 在 MAPPO 论文中被推荐为关键技巧，本文在更广泛的真实环境中证明它们可能有害——这挑战了 MARL 社区的共识
- **开源 benchmark**：提供模块化代码库，支持自定义算法和环境集成，可直接用于评估新 MARL 方法的可信度

## 局限与展望
- **仅覆盖策略梯度方法**：评估的三个算法（MADDPG、MAPPO、HAPPO）都基于策略梯度，未包含值分解方法（QMIX、QPLEX 等）
- **每次只变一个超参数**：未全面探索超参数组合的交互效应，虽然后验分析部分弥补
- **不确定性模型受限**：观测/动作噪声模型相对简单（高斯噪声、$\epsilon$-替换策略），未覆盖更复杂的对抗场景
- **弹性从扰动状态开始的假设**：实际中扰动可能是渐进的而非突然的，"从扰动状态开始无扰动 rollout"是简化

## 相关工作与启发
- **vs Engstrom et al. (2020, RL Implementation Matters)**：类似精神——证明超参数比算法本身更重要，但本文将此扩展到 MARL 的鲁棒性/弹性维度，规模更大
- **vs MAPPO (Yu et al. 2021)**：MAPPO 推荐的参数共享、GAE 等在 SMAC/MuJoCo 上有效，但本文在更多样的真实世界环境中证明它们的局限性
- **vs RRLS/Robust Gymnasium**：这些工作提供基准代码库但主要聚焦单智能体或小规模评估，本文在真实世界 MARL 中给出系统性的关系分析

## 评分
- 新颖性: ⭐⭐⭐ 弹性概念的引入和大规模实证研究有价值，但方法论贡献相对有限
- 实验充分度: ⭐⭐⭐⭐⭐ 82,620 次实验，严格的统计检验，四个真实世界环境，极其充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，发现总结为实用建议表格，便于从业者参考
- 价值: ⭐⭐⭐⭐ 对 MARL 社区有重要指导意义，挑战了多个普遍假设

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] Extending NGU to Multi-Agent RL: A Preliminary Study](extending_ngu_to_multi-agent_rl_a_preliminary_study.md)
- [\[ICML 2025\] Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](../../ICML2025/reinforcement_learning/enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[ICML 2026\] LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](../../ICML2026/reinforcement_learning/llm-guided_communication_for_cooperative_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
