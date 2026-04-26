---
title: >-
  [论文解读] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning
description: >-
  [ICLR 2026][策略集成] 从理论上分析了集成策略梯度方法中策略间多样性对学习效率的影响，提出通过KL散度约束调控多样性的Coupled Policy Optimization（CPO），在大规模并行环境中实现高效稳定的探索。
tags:
  - ICLR 2026
  - 策略集成
  - 大规模并行RL
  - 策略多样性
  - KL约束
  - 灵巧操作
---

# Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2603.01741](https://arxiv.org/abs/2603.01741)  
**代码**: [项目页面](https://naoki04.github.io/paper-cpo/)  
**领域**: 强化学习  
**关键词**: 策略集成, 大规模并行RL, 策略多样性, KL约束, 灵巧操作

## 一句话总结

从理论上分析了集成策略梯度方法中策略间多样性对学习效率的影响，提出通过KL散度约束调控多样性的Coupled Policy Optimization（CPO），在大规模并行环境中实现高效稳定的探索。

## 研究背景与动机

GPU基并行物理模拟器（如Isaac Gym、Genesis）使得同时在数万个环境中收集数据成为可能，但**简单增加并行环境数量并不提升学习效率**（Singla et al., 2024）——单一策略在大量并行环境中产生高度相似的轨迹，探索多样性不足。

为此，SAPG方法提出了leader-follower框架：一个leader策略和多个follower策略分别在不同环境块中收集数据，leader通过重要性采样（IS）聚合所有follower数据。然而，**过度的策略多样性反而有害**：当follower策略与leader策略差距过大时，IS比率偏离1，导致有效样本量（ESS）下降和PPO裁剪偏差增大，损害训练稳定性和样本效率。

核心矛盾在于：探索多样性需要策略差异，但过大的差异降低off-policy数据的利用效率。本文提出调控这种"适度多样性"的方法。

## 方法详解

### 整体框架

CPO基于SAPG的leader-follower架构，引入两个关键机制：（1）follower更新时对leader的KL散度约束，保持适度距离；（2）对抗性奖励鼓励follower间的分散，防止策略过度集中。

### 关键设计

1. **KL约束的Follower策略更新**:

    - 功能：将follower策略更新建模为带KL约束的优化问题
    - 核心思路：$\pi_{F_i}^* = \arg\max_{\pi_{F_i}} A_{F_i}(\mathbf{s},\mathbf{a}) \quad \text{s.t.} \quad D_{KL}(\pi_{F_i}(\cdot|\mathbf{s}) \| \pi_L(\cdot|\mathbf{s})) \leq \varepsilon_{KL}$
    - 采用AWAC式闭合解近似，得到参数化目标：$L_{\text{CPO},F_i}(\theta) = -\mathbb{E}_{\mathbf{a},\mathbf{s} \sim \pi_L}[\log \pi_{F_i,\theta}(\mathbf{a}|\mathbf{s}) \exp(\frac{1}{\lambda_f} A^{F_i})] + L_{\text{SAPG},F_i}$
    - 设计动机：根据Pinsker不等式，KL约束直接控制IS比率偏差上界 $\mathbb{E}[|1-\frac{\pi_L}{\pi_F}|] \leq \sqrt{2D_{KL}}$，从而保证ESS和训练稳定性

2. **对抗性奖励促进策略分散**:

    - 功能：训练判别器 $D_\xi(y|\mathbf{s}_t,\mathbf{a}_t)$ 预测策略身份
    - 核心思路：$r_t^{adv} = \lambda_{adv} \log D_\xi(y|\mathbf{s}_t,\mathbf{a}_t)$，奖励每个follower探索可被识别的独特区域
    - 设计动机：KL约束隐式拉近follower间的距离，对抗性奖励提供反向推力防止策略坍塌

### 损失函数 / 训练策略

总目标：$L_{\text{CPO}}(\theta) = L_{\text{SAPG}}(\theta,j) + \beta \sum_{i} L_{\text{CPO},F_i,f}(\theta,\lambda_f)$，其中 $\beta$ 用于平衡PPO目标和KL正则项的尺度。对抗性奖励仅提供给follower，leader更新时只使用真实环境奖励。

## 实验关键数据

### 主实验（灵巧操作任务，$2\times10^{10}$环境步后）

| 任务 | PPO | PBT | SAPG | **CPO** |
|------|-----|-----|------|---------|
| ShadowHand | 10661±1050 | 10294±1728 | 12882±343 | **13762±414** |
| AllegroHand | 10439±1282 | 13239±239 | 11989±817 | **14421±885** |
| Reorientation | 1.04±0.98 | 2.92±4.27 | 38.79±1.66 | **43.75±0.65** |
| Two-Arms | 1.41±0.80 | 26.43±11.12 | 5.11±3.41 | **35.30±2.77** |

### 消融实验（IS比率偏差与ESS，$5\times10^9$步时）

| 方法 | 平均IS比率偏差↓ | ESS率↑ |
|------|----------------|--------|
| SAPG | 0.889 | 0.0223 |
| CPO($\lambda_f$=0.5) | 更低 | 更高 |

### 关键发现
- CPO在多数任务中以约一半环境步数达到SAPG的最终性能
- SAPG在Two-Arms Reorientation任务中完全失败（5.11），CPO成功学习（35.30）
- KL约束使IS比率更接近1，验证了理论分析
- follower策略自然形成围绕leader的结构化分布，展现"涌现的探索行为"

## 亮点与洞察

- **理论分析有力**：从IS比率偏差和PPO裁剪偏差两个角度证明过度多样性的危害
- **方法简洁实用**：仅在SAPG基础上添加KL约束和对抗奖励，实现显著改进
- **对比分析深入**：揭示SAPG的策略错位问题——部分follower策略严重偏离leader

## 局限与展望

- KL约束强度 $\lambda_f$ 虽然鲁棒但仍需设定，自适应调节方案值得探索
- 简单任务（如Locomotion）中改进幅度有限
- 当前使用一维条件向量区分策略，更丰富的条件化方案可能进一步提升多样性

## 相关工作与启发

- SAPG（Singla et al., 2024）是直接的前辈工作，CPO在其基础上解决了策略错位问题
- DIAYN（Eysenbach et al., 2018）的判别器思路被巧妙应用于促进策略间分散
- 启示：在大规模分布式RL中，"有组织的多样性"比"无约束的多样性"更有效

## 评分
- 新颖性: ⭐⭐⭐⭐ 理论驱动的设计，但个别组件（KL约束、判别器）已有先例
- 实验充分度: ⭐⭐⭐⭐⭐ 10个任务、详细消融、ISS/KL分析、5个随机种子
- 写作质量: ⭐⭐⭐⭐ 从问题发现到理论分析到方法设计逻辑连贯
- 价值: ⭐⭐⭐⭐ 为大规模并行RL中的探索策略提供了实用指导

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning](policyflow_policy_optimization_with_continuous_normalizing_flow_in_reinforcement.md)
- [\[ACL 2026\] CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](../../ACL2026/reinforcement_learning/ce-gppo_coordinating_entropy_via_gradient-preserving_clipping_policy_optimizatio.md)
- [\[ICLR 2026\] Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control](towards_bridging_the_gap_between_large-scale_pretraining_and_efficient_finetunin.md)
- [\[ICLR 2026\] TROLL: Trust Regions improve Reinforcement Learning for Large Language Models](troll_trust_regions_improve_reinforcement_learning_for_large_language_models.md)
- [\[ACL 2025\] Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient](../../ACL2025/reinforcement_learning/bypass_back-propagation_optimization-based_structural_pruning_for_large_language.md)

<!-- RELATED:END -->
