---
title: >-
  [论文解读] 跨域离线强化学习中统一值对齐与值分配
description: >-
  [ICML 2026][强化学习][离线强化学习] 本文在异质跨域离线强化学习设定下揭示"值误分配"问题——源数据来自多个域和多个策略时，优势函数评估不准导致数据筛选失效。提出 V2A 框架通过时间一致的模态表示学习与模态感知的优势学习来统一解决值对齐和值分配问题，性能比 DVDF 提升 21.4%。
tags:
  - "ICML 2026"
  - "强化学习"
  - "离线强化学习"
  - "跨域迁移"
  - "值对齐"
  - "异质数据集"
---

# 跨域离线强化学习中统一值对齐与值分配

**会议**: ICML 2026  
**arXiv**: [2605.24862](https://arxiv.org/abs/2605.24862)  
**代码**: https://github.com/zq2r/V2A.git  
**领域**: 强化学习 / 跨域学习  
**关键词**: 离线强化学习, 跨域迁移, 值对齐, 异质数据集

## 一句话总结
本文在异质跨域离线强化学习设定下揭示"值误分配"问题——源数据来自多个域和多个策略时，优势函数评估不准导致数据筛选失效。提出 V2A 框架通过时间一致的模态表示学习与模态感知的优势学习来统一解决值对齐和值分配问题，性能比 DVDF 提升 21.4%。

## 研究背景与动机

**领域现状**：跨域离线 RL 通过结合充足的源域数据和有限的目标域数据来解决数据稀缺问题。最近方法从动态对齐（IGDF/OTDF）或值对齐（DVDF）角度进行数据筛选。

**现有痛点**：这些方法都假设源数据来自单一环境+单一策略（单模态）。但实际场景（机器人学）中源数据常来自多个源域+多个行为策略（多模态混合），现有方法在此设定下性能大幅下降。

**核心矛盾**：当源数据异质时，DVDF 用全局优势函数 $A_{\text{insrc}}^{\star}(s,a)$ 来评估各子数据集的样本质量，但不同动态下同一优势值代表不同的相对质量，导致"值误分配"——误将低质样本评分为高质。

**本文目标**：在异质跨域离线 RL 设定下，既要对齐动态，也要对齐值，还要正确分配值。

**切入角度**：关键洞察是通过聚类区分源数据中的不同动态模态，为每个模态单独学习优势函数，确保值评估准确。

**核心 idea**：用 EM 方法学习时间一致的模态表示，再用模态感知的优势学习，最后结合数据筛选——形成端到端的 V2A 框架。

## 方法详解

### 整体框架
V2A 分三个阶段——（1）提取源数据中的动态模态；（2）基于模态重标记数据和学习准确的优势函数；（3）用模态感知评分筛选源样本。

### 关键设计

1. **时间一致的模态表示学习**:

    - 功能：从异质源数据中提取每条轨迹的隐层动态模态表示 $z$。
    - 核心思路：标准 ELBO 会为同一轨迹的每个转移单独编码 $z$，产生时间不一致。V2A 改进为轨迹级编码、转移级解码，用 EM 交替优化——E 步固定解码器优化编码器，M 步固定编码器优化解码器。损失 $\mathsf{TC\text{-}ELBO} = \mathbb{E}_{\tau,z}[\sum_t \log p_\theta(s_{t+1}|s_t,a_t,z) - D_{KL}(q_\psi(\cdot|\tau),p(\cdot))]$。
    - 设计动机：保证同一轨迹内所有转移共享一个 $z$，反映统一的动态环境，避免随机波动导致模态识别失败。

2. **模态感知的优势学习**:

    - 功能：在学到的模态表示基础上，用重标记数据训练模态条件的优势函数 $A(s,a,z)$。
    - 核心思路：将 $z$ 输入到 Q 和 V 函数，用 Sparse-QL 学习 $Q(s,a,z)$ 和 $V(s,z)$，优势为 $A(s,a,z) = Q(s,a,z) - V(s,z)$。不同模态下同一 $(s,a)$ 对会获得不同优势值，准确反映各模态的相对质量。
    - 设计动机：修正 DVDF 的根本缺陷——全局优势函数无法区分模态差异；模态感知版本能为每个动态环境单独"打分"。

3. **模态感知数据筛选**:

    - 功能：根据模态感知评分进行数据筛选，选取得分前 $\xi$ 分位数的源样本。
    - 核心思路：评分 $f(s,a,s',z) = \lambda \cdot h(s,a,s') + (1-\lambda) \cdot \text{Norm}(A(s,a,z))$，综合动态对齐度 $h$ 和模态感知优势 $A$，权重 $\lambda$（实验固定 0.6）。
    - 设计动机：既要保证选出的样本动态与目标一致，也要保证质量高；模态意识使筛选避免"被骗"。

## 实验关键数据

### 主实验

| 方法 | IGDF | DVDF | V2A | OTDF | DVDF | V2A |
|------|------|------|-----|------|------|-----|
| 总得分 | 1286.7 | 1374.7 | **1562.5** | 1319.5 | 1395.9 | **1612.9** |
| 相对提升 | — | +6.8% | +21.4% | — | +5.5% | +22.2% |

在 4 个任务（HalfCheetah / Hopper / Walker2d / Ant）× 6 个源-目标组合上测试。V2A 在 IGDF 基础上 20/24 任务领先，在 OTDF 基础上 21/24 任务领先。

### 消融与分析

| 分析维度 | 结果 | 说明 |
|---------|------|------|
| 模态表示质量 | 图 2(a) t-SNE | 同源域轨迹聚集紧密，不同 shift 类型分布分离 |
| 优势分布对比 | 图 2(b) 密度图 | V2A 优势分布更尖锐；DVDF 分布平坦说明误估样本质量 |
| 超参数 $\lambda$ | 图 3(a) | $\lambda=0.6$ 最优；太小忽视动态对齐，太大忽视质量差异 |
| 数据选择比 $\xi$ | 图 3(b) | $\xi \in [0.5, 0.75]$ 较好；过小过大都降性能 |

## 亮点与洞察
- **问题定义精准**：值误分配是多模态异质数据下的真实问题，论文用定义 4.6 形式化刻画。
- **方法设计简洁**：核心就是三个模块各司其职——EM 学表示、模态条件优势、数据筛选。
- **实验说服力强**：动机实验直观展示 DVDF 失效现象；定性分析（模态可视化、优势分布）验证方法内在逻辑。

## 局限与展望
- 时间成本——EM 迭代学习模态表示增加训练开销。
- 理论假设——次优性界依赖"温和假设"，某些高度非平稳环境可能不适用。
- 单纯动态异质——论文关注动态 shift，但未探索奖励、初始分布等其他源特异性差异。

## 相关工作与启发
- **vs DVDF**：都考虑值对齐，但 DVDF 用全局优势函数；V2A 加入模态分解。
- **vs IGDF/OTDF**：前两者只考虑动态对齐；V2A 额外考虑质量。
- **启发**：模态分解思路可推广到其他多源迁移任务（如多源 domain adaptation）。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  异质数据设定是重要 gap，值误分配问题前所未见。
- 实验充分度: ⭐⭐⭐⭐⭐  系统的任务设计 + 充分的消融与可视化分析。
- 写作质量: ⭐⭐⭐⭐  逻辑清晰，动机例子有力。
- 价值: ⭐⭐⭐⭐⭐  解决离线 RL 实际难题，框架通用可与多个基础算法结合。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](../../ICLR2026/reinforcement_learning/dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)
- [\[ICML 2026\] 视觉工具使用强化学习究竟学到了什么？](what_does_vision_tool-use_reinforcement_learning_really_learn_disentangling_tool.md)
- [\[ICML 2026\] RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs](rl-sph_learning_to_achieve_feasible_solutions_for_integer_linear_programs.md)
- [\[ICML 2026\] Probing RLVR Training Instability through the Lens of Objective-Level Hacking](probing_rlvr_training_instability_through_the_lens_of_objective-level_hacking.md)
- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](global_policy-space_response_oracles_for_two-player_zero-sum_games.md)

</div>

<!-- RELATED:END -->
