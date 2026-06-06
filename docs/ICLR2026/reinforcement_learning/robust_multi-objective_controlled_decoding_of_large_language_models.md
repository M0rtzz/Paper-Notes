---
title: >-
  [论文解读] Robust Multi-Objective Controlled Decoding of Large Language Models
description: >-
  [ICLR 2026][强化学习][多目标对齐] 提出RMOD（Robust Multi-Objective Decoding），一种推理时算法，通过求解最小最大博弈的Nash均衡来动态计算最坏情况目标权重，在无需先验权重信息的情况下实现LLM的鲁棒多目标对齐。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "多目标对齐"
  - "推理时对齐"
  - "控制解码"
  - "鲁棒优化"
  - "最小最大博弈"
---

# Robust Multi-Objective Controlled Decoding of Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2503.08796](https://arxiv.org/abs/2503.08796)  
**代码**: [GitHub](https://github.com/williambankes/robust-multi-objective-decoding)  
**领域**: 强化学习  
**关键词**: 多目标对齐, 推理时对齐, 控制解码, 鲁棒优化, 最小最大博弈

## 一句话总结

提出RMOD（Robust Multi-Objective Decoding），一种推理时算法，通过求解最小最大博弈的Nash均衡来动态计算最坏情况目标权重，在无需先验权重信息的情况下实现LLM的鲁棒多目标对齐。

## 研究背景与动机

LLM需要同时对齐多个目标（如有用性、无害性、安全性、指令遵循等），多目标对齐自然引出一个问题：**如何在推理时平衡多个可能冲突的目标？**

现有方法通常需要手动指定目标权重，但权重选择面临多种困难：
- Shi et al. (2024)通过验证集超参搜索选择权重，但易受分布偏移影响
- 基于用户画像或历史交互的方法需要额外信息，实际中往往不可用
- 当安全性是目标之一时，不能容忍其被忽视，但也不能过度保守

核心动机：**不依赖任何先验权重信息，通过最大化最坏情况目标来实现鲁棒对齐**——让最弱的目标得到最大关注。

## 方法详解

### 整体框架

RMOD在每个解码步骤中，利用为每个目标训练的值函数 $V_g$，通过求解 $\max_\pi \min_{w \in \Delta^{G-1}} \lambda \sum_g w_g V_g(x,y^t;\pi) - D_{KL}(\pi \| \pi_{\text{ref}})$ 的Nash均衡来确定权重和策略。

### 关键设计

1. **最小最大博弈形式化**:

    - 功能：将鲁棒多目标对齐建模为策略 $\pi$ 和权重 $w$ 的两人零和博弈
    - 核心思路：目标对 $w$ 线性、对 $\pi$ 凹，因此Nash均衡存在且minimax定理允许交换max-min顺序，转化为先求解最优策略再优化权重
    - 设计动机：最小最大确保不会有任何单一目标被严重忽视

2. **最优策略的解析解**:

    - 功能：给定权重 $w$，推导最优采样策略
    - 核心思路（Proposition 1）：$\pi(z|[x,y^t];w) = \frac{\pi_{\text{ref}}(z|[x,y^t]) \exp(\lambda \sum_g w_g V_g(x,y^t;z))}{Z(x,y^t,w)}$
    - 设计动机：解析解避免了昂贵的策略搜索，与标准KL-正则化RLHF的Boltzmann形式一致

3. **凸优化求解最坏情况权重**:

    - 功能：将权重搜索简化为LogSumExp形式的凸优化
    - 核心思路：$w^* = \arg\min_{w \in \Delta^{G-1}} \log \mathbb{E}_{z\sim\pi_\text{ref}}[\exp(\sum_g \lambda w_g V_g)]$，使用指数加权梯度下降迭代更新 $w_{g,i+1} = w_{g,i} \cdot \exp(-\eta \cdot \text{gradient})$
    - 设计动机：凸性保证全局最优，维度仅为 $G$（目标数），计算高效

4. **Block-wise解码实现**:

    - 功能：将连续解码分为长度 $B$ 的块，从 $K$ 个候选中选择最优
    - 核心思路：先用 $\pi_{\text{ref}}$ 采样 $K$ 个块候选，计算值函数，迭代更新权重 $I$ 次，选择加权值最高的块
    - 设计动机：相比逐token解码大幅减少值函数评估次数

### 损失函数 / 训练策略

值函数训练使用MSE损失：$\mathbb{E}[\sum_t(V_g(x,y^t;\theta) - r_g(x,y))^2]$，基于参考策略生成的响应和对应奖励。RMOD本身是推理时算法，不需要训练策略网络。

## 实验关键数据

### 主实验（HH数据集，最坏情况奖励）

| 方法 | 最坏情况奖励 | 最坏情况胜率(WCWR) |
|------|------------|-------------------|
| CD-Helpful | 高helpful但低harmless | 较低 |
| CD-Harmless | 高harmless但低helpful | 较低 |
| CD-Uniform | 中等平衡 | 57.6% |
| MO-GRPO | 中等 | 54.6% |
| RS/MOD | 低于Uniform | - |
| Distill-RMOD | - | 57.9% |
| **RMOD** | **最高** | **59.1%** |

### 消融实验

| 参数 | 关键指标 | 说明 |
|------|---------|------|
| $\lambda=0.1$（低） | 接近Uniform | 权重分布均匀 |
| $\lambda=0.5$ | 中等鲁棒 | 平衡权衡 |
| $\lambda=10$（高） | 最集中于最差目标 | 权重高度稀疏 |
| B=16（小块） | 最高胜率 | 更细粒度控制 |
| B=256（大块） | 胜率下降 | 接近参考策略 |
| 目标数=2-10 | RMOD持续优于Uniform | 但随目标增多性能下降 |

### 关键发现
- RMOD比所有基线高出最多20%的最坏情况胜率
- 延迟仅比标准CD增加4.5%，计算效率高
- Distill-RMOD（用RMOD生成的数据做SFT）在不使用解码的情况下也表现出色
- LLM-as-Judge（GPT-4o）评估也确认RMOD的优越性

## 亮点与洞察

- **理论优雅**：将问题形式化为凸凹博弈，有解析解和凸优化，理论保证充分
- **实用性强**：推理时算法可随时切换对齐目标，延迟开销极小
- **权重行为分析深入**：通过KKT条件证明最优权重会均衡化各目标的期望奖励
- **Distill-RMOD**提供了一种将推理时方法蒸馏为普通策略的实用路径

## 局限与展望

- 随目标数增多（>10）性能下降，大规模多目标场景需要进一步研究
- 需要为每个目标训练独立的值函数，准备成本较高
- $\lambda$ 的选择影响鲁棒性偏好（稀疏度），目前需要手动设定
- 当前仅在gemma-2-2b-it上实验，更大模型的效果未验证

## 相关工作与启发

- Mudgal et al. (2023)的Controlled Decoding是直接基础，RMOD扩展为鲁棒版本
- Shi et al. (2024)的MOD方法需要预设权重，RMOD自动寻找
- Yoon et al. (2024)和Ramesh et al. (2024)考虑了鲁棒对齐但非推理时方法
- 启示：推理时算法+鲁棒优化的组合为多目标LLM对齐提供了灵活且有保障的解决方案

## 评分
- 新颖性: ⭐⭐⭐⭐ 最小最大推理时对齐是新组合，但各组件较成熟
- 实验充分度: ⭐⭐⭐⭐ 多数据集、消融、LLM-as-Judge、延迟分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，问题动机直观
- 价值: ⭐⭐⭐⭐ 为多目标LLM对齐提供了原则性的推理时方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICLR 2026\] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICLR 2026\] TROLL: Trust Regions improve Reinforcement Learning for Large Language Models](troll_trust_regions_improve_reinforcement_learning_for_large_language_models.md)
- [\[ICML 2026\] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](../../ICML2026/reinforcement_learning/game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)

</div>

<!-- RELATED:END -->
