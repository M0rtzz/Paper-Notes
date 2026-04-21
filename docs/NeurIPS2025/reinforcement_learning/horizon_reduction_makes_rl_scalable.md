---
title: >-
  [论文解读] Horizon Reduction Makes RL Scalable
description: >-
  [NeurIPS 2025][offline RL] 本文通过大规模实验（最高 10 亿转移数据）揭示离线 RL 的可扩展性瓶颈源于决策时域过长（curse of horizon），并证明通过 n-step 回报和层次策略等时域缩减技术可显著提升扩展性，进而提出了简洁有效的 SHARSA 方法。
tags:
  - NeurIPS 2025
  - offline RL
  - scalability
  - horizon reduction
  - hierarchical RL
  - goal-conditioned RL
---

# Horizon Reduction Makes RL Scalable

**会议**: NeurIPS 2025  
**arXiv**: [2506.04168](https://arxiv.org/abs/2506.04168)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: offline RL, scalability, horizon reduction, hierarchical RL, goal-conditioned RL

## 一句话总结

本文通过大规模实验（最高 10 亿转移数据）揭示离线 RL 的可扩展性瓶颈源于决策时域过长（curse of horizon），并证明通过 n-step 回报和层次策略等时域缩减技术可显著提升扩展性，进而提出了简洁有效的 SHARSA 方法。

## 背景与动机

1. **可扩展性是现代 ML 的核心**：NLP 和 CV 领域已验证数据+计算规模带来持续性能提升，但离线 RL 在复杂任务上的可扩展性仍未被充分研究——之前的工作主要关注"更多任务"（宽度扩展），而非"更难任务"（深度扩展）。

2. **标准离线 RL 在复杂任务上失败**：在 OGBench 的极难任务（cube-octuple、puzzle-4x6、humanoidmaze-giant）上，IQL、CRL、SAC+BC 等 SOTA 方法即使使用 10 亿数据仍然无法解决任务，性能远低于最优值。

3. **增大模型无法根本解决问题**：将 SAC+BC 模型扩大到 5.91 亿参数（35 倍），性能有限提升甚至退化，表明问题不在模型容量不足。

4. **TD 学习中的偏差累积**：时序差分（TD）学习的预测目标本身有偏，偏差随时域累积。这与监督学习（如 next-token prediction）形成鲜明对比——后者的目标无偏，因此可无限扩展。

5. **策略学习的时域诅咒**：即使价值函数完美，长时域意味着状态-最优动作映射极度复杂，类似于大模型直接回答复杂问题（无 chain-of-thought）的困难。

6. **缺乏面向可扩展性的评估范式**：现有离线 RL 基准（D4RL 等）任务相对简单，数据集较小（~1M），无法暴露扩展性瓶颈。需要专门的大规模、长时域基准来评估算法是否"准备好被扩展"。

## 方法详解

### 问题设定

聚焦离线目标条件 RL（goal-conditioned RL），使用 OGBench 中四个极难任务：
- **cube-octuple**：顺序拾放 8 个方块
- **puzzle-4x5/4x6**：用机器臂解组合谜题"Lights Out"
- **humanoidmaze-giant**：人形机器人在大型迷宫中导航

数据集最大为 **10 亿转移**（约 100 万条轨迹），比标准离线 RL 数据集大 1000 倍。

### 时域诅咒的诊断

**价值学习诊断**：在 combination-lock 教学任务上对比 1-step DQN 和 64-step DQN：
- 两者 TD 误差相近，但 1-step DQN 的 Q 误差（相对真实 $Q^*$）随时域急剧增大
- Q 误差在距离终点越远的状态越大，直接验证了偏差累积假说
- 调整模型大小、学习率、目标网络更新率均无法修复

**策略学习诊断**：长时域使得状态到最优动作的映射极为复杂，类似于不用 chain-of-thought 直接回答复杂问题。层次化策略通过分解为子目标策略降低每层复杂度。

### SHARSA 方法

SHARSA 同时缩减价值时域和策略时域：

**高层策略提取**（拒绝采样）：
$$\pi^h(s,g) \stackrel{d}{=} \arg\max_{w_1,...,w_N: w_i \sim \pi_\beta^h(w|s,g)} Q^h(s, w_i, g)$$

从高层 flow BC 策略采样 $N$ 个子目标，用高层价值函数选最优。

**高层 SARSA 价值学习**（n-step）：

$$L^Q(Q^h) = \mathbb{E}\left[D\left(Q^h(s_h, s_{h+n}, g), \sum_{i=0}^{n-1}\gamma^i r(s_{h+i}, g) + \gamma^n V^h(s_{h+n}, g)\right)\right]$$

使用 n-step 回报减少 TD 递归次数，降低偏差累积。

**低层策略**：使用目标条件 flow BC，或再做一轮拒绝采样（double SHARSA 变体）。

SHARSA 的核心优势：(1) 仅依赖行为克隆和 SARSA，无需复杂超参调优；(2) 同时缩减两个维度的时域。

## 实验结果

### 实验一：标准离线 RL 的扩展失败

| 方法 | cube-octuple (1B) | puzzle-4x6 (1B) | humanoidmaze-giant (1B) |
|------|------------------|----------------|----------------------|
| Flow BC | ~0% | ~5% | ~0% |
| IQL | ~0% | ~20% | ~15% |
| CRL | ~0% | ~10% | ~45% |
| SAC+BC | ~0% | ~25% | ~20% |

4 个方法在 cube-octuple 上全部失败（~0%），即使数据从 1M 增至 1B 性能几乎无提升。增大 SAC+BC 模型至 5.91 亿参数（8 天训练）也未能实质性改善。进行了 9 项消融（策略类型、网络架构、集成、正则化、学习率、目标网络更新率、batch size、梯度步数），无一能全面提升扩展性。

### 实验二：时域缩减的效果

| 方法 | 时域缩减类型 | cube-octuple (1B) | puzzle-4x6 (1B) | humanoidmaze-giant (1B) |
|------|------------|------------------|----------------|----------------------|
| SAC+BC | 无 | ~0% | ~25% | ~20% |
| n-step SAC+BC | 价值 | ~0% | ~40% | ~55% |
| Hierarchical FBC | 策略 | ~10% | ~5% | ~0% |
| HIQL | 策略 | ~5% | ~30% | ~35% |
| **SHARSA** | **价值+策略** | **~15%** | **~45%** | **~60%** |
| **Double SHARSA** | **价值+策略** | **~20%** | **~50%** | **~55%** |

关键发现：(1) n-step 回报单独就能大幅改善（puzzle +15pp，humanoidmaze +35pp）；(2) 层次策略在 cube 上是必要的（从 0% 到 10%+）；(3) SHARSA 是唯一在全部四个任务上均取得非平凡性能的方法。值得注意的是，SHARSA 使用标准 [1024]×4 MLP，远小于 5.91 亿参数的消融模型，说明算法改进比模型规模更重要。

## 亮点

- **首次大规模诊断离线 RL 可扩展性**：10 亿数据规模的系统性分析，涵盖 9 项消融，给出了"为何标准 RL 不能类似 LLM 那样通过加数据/模型持续提升"的清晰答案
- **因果归因清晰**：combination-lock 教学实验精确分离了 TD 偏差累积与模型容量的影响，Q 误差随位置的分布图极具说服力
- **方法极简而有效**：SHARSA 仅用 SARSA + flow BC + 拒绝采样三个简单组件，无需复杂目标函数或过多超参
- **开放问题的诚实讨论**：承认即使 SHARSA 也未能在全部任务上达到 100%，呼吁社区关注可扩展性评估

## 局限性

- **数据集假设较强**：SHARSA 隐含假设数据集在短片段内接近最优（near-optimal within short segments），对低质量数据集的鲁棒性未验证
- **仅限状态空间实验**：为隔离核心因素去除了视觉观测，但现实中表征学习与时域诅咒的交互效应可能更复杂
- **理想化过多**：去除了分布外泛化和数据覆盖不足的评估场景，实际部署中这些因素与时域长度交织
- **扩展性未完全解决**：SHARSA 在 cube-octuple 上仅达 ~20% 成功率，且性能随数据增加非单调上升，两层层次结构只是缓解而非根治偏差累积

## 相关工作对比

### vs. IQL（Kostrikov et al., 2022）

IQL 通过 in-sample maximization 避免分布偏移，在标准 D4RL 基准上表现优异。但 IQL 使用 1-step TD 学习，在长时域任务上偏差严重累积。本文的 HIQL（层次化 IQL）通过添加层次策略改善了策略时域，但未解决价值时域问题。SHARSA 通过 n-step SARSA 和层次策略同时缩减两个维度的时域，在全部四个任务上均优于 IQL 和 HIQL。

### vs. CRL（Eysenbach et al., 2022）

CRL 基于对比学习和 one-step RL，不使用 TD 学习因此不受偏差累积影响。CRL 在 humanoidmaze-giant 上确实表现较好（~45%），验证了本文关于 TD 偏差的假说。但 CRL 在 cube-octuple（~0%）和 puzzle（~10%）上仍然失败，因为其平坦策略受策略时域诅咒限制。SHARSA 通过层次化弥补了这一不足。

### vs. Park et al.（2024）的扩展性研究

Park et al. 研究了策略提取和泛化对离线 RL 扩展性的瓶颈，但使用的数据集仅为本文的 1/100，且未深入分析时域的作用。本文使用更具表达力的 flow 策略和 100 倍大数据集，系统性地将时域识别为核心障碍，提出了互补的见解。

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次系统性地将时域诅咒识别为离线 RL 扩展性的核心障碍，并给出清晰因果验证 |
| 技术深度 | ⭐⭐⭐ | SHARSA 方法本身较简单，核心贡献在诊断分析而非算法创新 |
| 实验充分度 | ⭐⭐⭐⭐⭐ | 10 亿规模数据、9 项消融、教学任务归因、4 个极难环境，实验体量和严谨性极高 |
| 实用价值 | ⭐⭐⭐⭐ | 为 RL 社区指明扩展方向，SHARSA 实现简单可直接应用，但仍有理想化限制 |

<!-- RELATED:START -->

## 相关论文

- [Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)
- [When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners](when_less_language_is_more_language-reasoning_disentanglement_makes_llms_better_.md)
- [Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [Scalable Neural Incentive Design with Parameterized Mean-Field Approximation](scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)
- [Oryx: a Scalable Sequence Model for Many-Agent Coordination in Offline MARL](oryx_a_scalable_sequence_model_for_many-agent_coordination_in_offline_marl.md)

<!-- RELATED:END -->
