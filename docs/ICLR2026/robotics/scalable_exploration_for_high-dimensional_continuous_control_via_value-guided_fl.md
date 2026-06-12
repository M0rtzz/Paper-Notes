---
title: >-
  [论文解读] Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow
description: >-
  [ICLR 2026][机器人][高维控制] 提出Qflex(Q-guided Flow Exploration)——在高维连续动作空间中实现可扩展探索的RL方法：从可学习源分布沿Q函数诱导的概率流传输动作→探索与任务相关梯度对齐(而非各向同性噪声)→在多种高维基准上超越高斯/扩散RL基线…
tags:
  - "ICLR 2026"
  - "机器人"
  - "高维控制"
  - "价值引导流"
  - "概率流探索"
  - "肌骨模型"
  - "actor-critic"
---

# Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow

**会议**: ICLR 2026  
**arXiv**: [2601.19707](https://arxiv.org/abs/2601.19707)  
**领域**: 强化学习/高维控制  
**关键词**: 高维控制, 价值引导流, 概率流探索, 肌骨模型, actor-critic

## 一句话总结

提出Qflex(Q-guided Flow Exploration)——在高维连续动作空间中实现可扩展探索的RL方法：从可学习源分布沿Q函数诱导的概率流传输动作→探索与任务相关梯度对齐(而非各向同性噪声)→在多种高维基准上超越高斯/扩散RL基线,成功控制700执行器的全身人体肌骨模型执行敏捷复杂动作。

## 研究背景与动机

**领域现状**：高维动力系统控制(全身肌骨/多腿机器人)→RL的核心挑战。动作空间可达数百维→标准高斯探索急剧失效。

**现有痛点**：
   - (1) 高斯噪声探索→维度增长→覆盖率指数级下降→样本效率骤降
   - (2) 降维方法(DynSyn/DEP-RL)→限制策略表达力→牺牲灵活性
   - (3) 扩散/流策略→用于多模态→但isotropic初始分布→高维仍低效
   - (4) 700个肌肉执行器→远超现有方法的成功应用范围

**切入角度**：Q函数引导的概率流→使探索对齐任务相关方向→保持高维原始空间。

## 方法详解

### 整体框架

Qflex 仍是一个标准的 actor-critic 框架，但把"如何采样动作"从一次性的高斯加噪，换成了由 Q 函数引导的多步概率流：动作从一个可学习的源分布出发，沿着学到的速度场 $v_\theta(a, s, t)$ 逐步被"传输"到 Q 值更高的区域，最终落点既是策略采样也是探索方向。Critic 学到的 Q 函数同时扮演评价器和探索指南针两个角色，从而让探索方向与任务回报的梯度对齐，而不是在 700 维空间里盲目撒各向同性噪声。

### 关键设计

**1. Q 引导的概率流传输：让探索方向对齐价值梯度。** 高维连续控制的根本困境是绝大多数动作扰动方向都是无用的——动作空间一旦涨到上百维，各向同性高斯噪声的有效覆盖率随维度指数衰减，样本效率随之崩塌。Qflex 不再从固定噪声直接得到动作，而是把动作沿一条概率流逐步推进，更新形式为 $a \leftarrow a + v_\theta(a, s, t) \cdot dt$，其中 $v_\theta$ 是学到的速度场，被训练成指向 Q 增长的方向。这样每一步传输都把动作推向价值更高的区域，探索从"随机试"变成"按 Q 指引的方向试"，在维度极高、有效方向极稀疏的场景里把采样预算集中到真正有用的子空间上。

**2. 多步流传输取代一步加噪：逐步精化而非单次扰动。** 标准高斯探索（如 SAC）是一步采样，扩散类方法（如 DACER）虽然多步却从各向同性的起点出发、靠后验引导回拉，二者在高维下都受限于起点信息的缺失。Qflex 用多步传输逐步精化动作：沿 $v_\theta$ 积分若干小步 $dt$，每一步都重新利用当前状态与 Q 信息修正方向，使得最终动作是被价值场"塑形"过的结果。相比一步噪声，这种逐步精化能在保留原始全维动作空间的同时，避免单次扰动落入无意义区域。

**3. 可学习的源分布：让初始分布也携带信息。** 探索的起点不再是固定的标准高斯，而是一个可学习的源分布，与速度场一同被优化。固定高斯起点在高维下本身就浪费了大量概率质量在无用方向上；让源分布可学习后，初始采样就已经偏向任务相关区域，相当于"初始分布也在 carrying information"，与后续的 Q 引导流前后衔接，进一步压缩了从源到高价值目标动作之间需要传输的距离。

## 实验关键数据

### 高维基准(MuJoCo/Isaac)

| 环境 | 动作维度 | Qflex vs SAC | vs 扩散 |
|------|---------|-------------|---------|
| Humanoid | ~23 | +15% | +10% |
| 高维变体 | ~100 | +30% | +20% |
| **全身肌骨** | **700** | **成功(SAC失败)** | **成功(扩散失败)** |

### 全身肌骨控制

- 600+肌肉→700维动作空间
- 复杂运动(跑/跳/转)→Qflex成功→基线全部失败
- 无降维→保持全部灵活性

### 关键发现

- Q引导→高维探索非常有效→因为绝大多数方向是无用的→Q引导聚焦有用方向
- 可学习源分布→比固定高斯好→初始分布也carrying information
- 维度越高→Qflex vs 基线差距越大→验证了可扩展性

## 亮点与洞察

- **"700维的'不可能'任务"**：之前没有RL方法在700+维连续空间成功→Qflex突破了这个barrier。
- **"Q函数=探索指南针"**：不是随机试→而是按Q引导方向试→每次探索都有方向。
- **保持原始空间的价值**：降维→牺牲灵活性→可能错过最优解→Qflex证明保持全维度是值得的。
- **生物启发**：人类肌骨控制→大脑通过value-like信号引导探索→Qflex的流与此类似。


## 局限与展望

- In this paper, we introduce Qflex, a scalable online RL method for efficient exploration in high-dimensional continuous control.

- Our method conducts directed exploration by sampling from a Q-guided probability flow with policy-improvement guarantees, yielding superior learning efficiency over representative online RL baselines across benchmarks characterized by high dimensionality and over-actuation.

- Qflex further demonstrates agile, complex motion control on a full-body musculoskeletal model with 700 actuators, achieving high efficiency and strong scalability in truly high-dimensional settings.

- Our analysis shows that value-aligned exploration in Qflex surpasses undirected sampling strategies in high-dimensional regimes, which is readily extensible to a variety of online RL frameworks and exploration settings.

- Acknowledgments

This work is supported by STI 2030-Major Projects 2022ZD0209400, Beijing Academy of Artificial Intelligence and Beijing Municipal Science & Technology Commissi


## 相关工作与启发

- **vs DynSyn**: 本文在此基础上提出了不同的技术路线，在关键指标上取得了改进。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Q引导概率流探索的首次提出+700维成功
- 实验充分度: ⭐⭐⭐⭐⭐ 多维度基准+全身肌骨+与多种基线对比
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰
- 价值: ⭐⭐⭐⭐⭐ 对高维RL有根本性突破

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Test-driven Reinforcement Learning in Continuous Control](../../AAAI2026/robotics/test-driven_reinforcement_learning_in_continuous_control.md)
- [\[AAAI 2026\] Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation](../../AAAI2026/robotics/affordance-guided_coarse-to-fine_exploration_for_base_placem.md)
- [\[CVPR 2026\] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](../../CVPR2026/robotics/como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)
- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)
- [\[ICLR 2026\] On Entropy Control in LLM-RL Algorithms](on_entropy_control_in_llm-rl_algorithms.md)

</div>

<!-- RELATED:END -->
