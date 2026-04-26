---
title: >-
  [论文解读] Improving Planning and MBRL with Temporally-Extended Actions
description: >-
  [NeurIPS 2025][temporally-extended actions] 本文提出在 shooting-based 规划和 MBRL 中将动作持续时间作为额外优化变量，配合 MAB 自动选择持续时间范围，在多个环境中显著加速规划并解决标准方法无法解决的困难任务。
tags:
  - NeurIPS 2025
  - temporally-extended actions
  - model-based RL
  - planning
  - action duration
  - multi-armed bandit
---

# Improving Planning and MBRL with Temporally-Extended Actions

**会议**: NeurIPS 2025  
**arXiv**: [2505.15754](https://arxiv.org/abs/2505.15754)  
**代码**: 无  
**领域**: reinforcement_learning  
**关键词**: temporally-extended actions, model-based RL, planning, action duration, multi-armed bandit

## 一句话总结
本文提出在 shooting-based 规划和 MBRL 中将动作持续时间作为额外优化变量，配合 MAB 自动选择持续时间范围，在多个环境中显著加速规划并解决标准方法无法解决的困难任务。

## 研究背景与动机

**领域现状**：连续系统常用离散时间近似，仿真步长 $\delta_t$ 小导致规划 horizon $D$ 需要很长，给 CEM/MPPI 带来计算负担。

**现有痛点**：frame-skip 最优值因环境而异。Model-free RL 有学习 frame-skip 的工作，但无人在规划/MBRL 中解决。长 rollout 导致 compounding error。

**核心矛盾**：小 $\delta_t$ 保证精度但搜索空间大；大 frame-skip 减少搜索但不灵活。

**本文目标** 让 planner 每步同时优化动作和持续时间 $\delta t_k \in [\delta t_{\min}, \delta t_{\max}]$。

**切入角度**：$\delta t$ 作为连续优化变量 + MAB 自动选择 $\delta t_{\max}$。

**核心 idea**：动作持续时间作为 planner 优化变量 + 学习的 TE 动力学模型 + MAB 自动范围选择。

## 方法详解

### 整体框架

CEM 每决策步输出 $(a_k, \delta t_k)$。回报：$J_2 = \sum_k \gamma^{e_{<k}} \sum_t \gamma^{t-1} \mathcal{R}(s,a)$，其中 $e_k = \lfloor \delta t_k / \delta_t^{\text{env}} \rfloor$。支持双折扣 $\gamma_1, \gamma_2$。

### 关键设计

1. **时间扩展动力学模型 $\hat{F}_{\text{TE}}$**:

    - 输入 $(s, a, \delta t)$，输出下一状态分布+奖励
    - 推理时间恒定（vs 迭代式 $F_{\text{IP}}$ 随 $\delta t$ 线性增长）
    - 更短 rollout 减少 compounding error

2. **MAB 自动选择 $\delta t_{\max}$**:

    - $m = \log_2(T)$ 个指数间隔候选值
    - UCB + EMA：$\arg\max_i (\hat{R}_{i,T} + c\sqrt{2\log T / N(i,T)})$
    - 每个候选维护独立数据集和模型

3. **搜索空间分析**:

    - 原始：搜索 $|\mathcal{A}|^H$，优化 $H|\mathcal{A}|$ 变量
    - TE（$m$ 倍）：搜索 $|\mathcal{A}|^{H/m}$，优化 $(H/m)(|\mathcal{A}|+1)$ 变量

## 实验关键数据

### 规划实验（精确动力学）

| 环境 | 标准 | TE | 改进 |
|------|------|----|----|
| Mountain Car | 需 $D \geq 60$ | $D_{\text{TE}} \geq 4$ 即解 | horizon 缩减 15x |
| Multi-hill MC (5实例) | 仅解 1/5 | 全部解决 | 解决不可行问题 |
| Dubins Car (102维) | 内存不足 4GB | 103MB 成功 | 内存减 40x |

### MBRL 实验

| 环境 | PETS | TE(F) | TE(D) | 最大提升 |
|------|------|-------|-------|---------|
| Reacher | -6.5 | **-4.2** | -4.5 | +35% |
| HalfCheetah | ~4900 | **~6100** | ~5800 | +24% |
| Hopper | ~230 | **~680** | ~350 | +195% |
| Walker | ~420 | **~610** | ~540 | +45% |

### 消融实验

| 配置 | 效果 |
|------|------|
| $\gamma_1$ 固定, $\gamma_2$ 减小 | 决策步增加，总步不变 |
| $\gamma_2$ 固定, $\gamma_1$ 减小 | 总步减少 |
| 共享 vs 独立模型 | 独立模型更优 |
| TE(F) vs TE(D) | MAB 自动选择接近手动最优 |

### 关键发现
- 稀疏奖励环境优势巨大：浅 depth 实现深搜索
- MBRL 优势：短 rollout 减少 compounding error + 更快收敛
- Hopper 改善最大 (+195%)
- MAB 消除手动调参需求

## 亮点与洞察
- **简洁强大**：仅增一个优化维度就大幅改变 planner 能力边界
- **MAB 超参自动化**：EMA + UCB 解决非平稳问题
- **搜索空间分析**：$2^H$ vs $2^{H/m}$ 直观
- 可迁移到"时间分辨率 vs 搜索深度" trade-off 的任何系统

## 局限与展望
- $\hat{F}_{\text{TE}}$ 对长持续时间动作的预测精度有挑战——预测从 $s$ 经过 $e_k$ 步到达的状态比单步预测本质上更难
- 独立维护多个模型（每个 $\delta t_{\max}$ 候选）增加了内存和训练时间开销
- 仅在 shooting-based planners（CEM）上验证，未在 gradient-based 方法或 tree-search（MCTS）中测试
- 缺乏理论分析——时间扩展动作在什么条件下保证不损失最优性？误差传播如何与 $\delta t$ 关系？
- 双折扣因子 $\gamma_1, \gamma_2$ 虽然默认相等即可，但引入了额外自由度

### 两种变体对比

| 特性 | $A_{\text{TE}}$(F) 固定范围 | $A_{\text{TE}}$(D) 动态选择 |
|------|--------------------------|----------------------------|
| $\delta t_{\max}$ | 手动设定 | MAB 自动选择 |
| 模型数量 | 1 个 | $m = \log_2(T)$ 个 |
| 数据集 | 共享 | 每个候选独立 |
| 调参需求 | 需调 $\delta t_{\max}$ | 无（MAB 自动化） |
| 性能 | 手调最优时略好 | 接近手调最优，更鲁棒 |

## 相关工作与启发
- **vs PETS (Chua et al. 2018)**：在 PETS 上增加 TE + MAB，架构变化最小
- **vs Ni & Jang (2022)**：model-free 学 timescale，本文 MBRL 直接优化 duration
- **vs Options (Sutton et al. 1999)**：Options 需学 initiation + termination，本文更轻量

## 评分
- 新颖性: ⭐⭐⭐⭐ 在 MBRL/planning 中首次系统研究动作持续时间优化
- 实验充分度: ⭐⭐⭐⭐⭐ 两大场景 7+ 环境，消融完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，notation 一致
- 价值: ⭐⭐⭐⭐ 实用性强，可直接集成到现有 MBRL 框架

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Parameter-Free Algorithms for the Stochastically Extended Adversarial Model](parameter-free_algorithms_for_the_stochastically_extended_adversarial_model.md)
- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](self-improving_embodied_foundation_models.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured MDPs](retrosynthesis_planning_via_worst-path_policy_optimisation_in_tree-structured_md.md)
- [\[NeurIPS 2025\] FedRAIN-Lite: Federated Reinforcement Algorithms for Improving Idealised Numerical Weather and Climate Models](fedrain-lite_federated_reinforcement_algorithms_for_improving_idealised_numerica.md)

<!-- RELATED:END -->
