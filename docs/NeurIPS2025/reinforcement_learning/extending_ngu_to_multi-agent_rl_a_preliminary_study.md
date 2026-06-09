---
title: >-
  [论文解读] Extending NGU to Multi-Agent RL: A Preliminary Study
description: >-
  [NeurIPS 2025][强化学习][Never Give Up] 将单智能体 NGU（Never Give Up）算法扩展至多智能体环境，通过共享回放缓冲区、共享新颖性信号和异构 β 参数三个设计维度的系统消融…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "Never Give Up"
  - "多智能体强化学习"
  - "内在动机"
  - "探索"
  - "稀疏奖励"
---

# Extending NGU to Multi-Agent RL: A Preliminary Study

**会议**: NeurIPS 2025  
**arXiv**: [2512.01321](https://arxiv.org/abs/2512.01321)  
**代码**: [GitHub](https://github.com/juan-hernandez/extending-ngu-marl)  
**领域**: 强化学习  
**关键词**: Never Give Up, 多智能体强化学习, 内在动机, 探索, 稀疏奖励

## 一句话总结
将单智能体 NGU（Never Give Up）算法扩展至多智能体环境，通过共享回放缓冲区、共享新颖性信号和异构 β 参数三个设计维度的系统消融，发现 NGU + 共享经验池组合在 PettingZoo simple_tag 追捕任务中显著优于多智能体 DQN 基线。

## 研究背景与动机

**领域现状**：强化学习在 Atari 等任务上取得了巨大成功，但在稀疏奖励环境（如 Montezuma's Revenge）中 DQN 等方法因探索不足而表现较差。NGU 算法通过情景新颖性（episodic novelty）与内在动机的结合，在单智能体稀疏奖励任务中取得了 SOTA 表现。

**现有痛点**：多智能体强化学习（MARL）中的稀疏奖励问题更加严峻——除了探索困难外，还面临信用分配、环境非平稳性和协调探索等挑战。现有 MARL 探索方法（如 EMC、MACE）往往引入额外的架构复杂性和计算开销。

**核心矛盾**：复杂的探索机制在 MARL 中虽然有效但难以泛化，而简单强力的 NGU 框架尚未被系统地适配到多智能体场景中。

**本文目标**：能否将 NGU 的核心探索机制直接扩展到 MARL？共享经验、共享新颖性和异构探索参数这三个设计选择对性能有怎样的影响？

**切入角度**：保留 NGU 的核心组件（嵌入网络、逆动力学模型、情景记忆），去掉 RND 和 UVFA 以降低计算开销，直接在 MARL 环境中验证。

**核心 idea**：通过消融三个关键设计维度，证明 NGU + 共享回放缓冲区是最有效的多智能体探索组合。

## 方法详解

### 整体框架
每个智能体 $i \in \{1, \dots, N\}$ 拥有独立的 Q 网络、嵌入网络、情景记忆和内在奖励计算模块。输入为各智能体的局部观测，输出为离散动作。系统通过嵌入网络+逆动力学模型学习状态表示，利用情景记忆中的 k 近邻距离计算新颖性奖励。

### 关键设计

1. **内在奖励计算**:

    - 功能：为每个智能体计算基于情景记忆的新颖性奖励
    - 核心思路：嵌入网络 $\phi: \mathcal{S} \to \mathbb{R}^d$ 通过逆动力学损失训练，预测连续嵌入之间的动作 $a_t$。每步计算智能体 $i$ 的内在奖励：$r_{t,i}^{\text{intrinsic}} = f(\phi(s_{t+1}^i), \mathcal{M}_i)$，其中 $f$ 是 k 近邻距离函数，$\mathcal{M}_i$ 是当前 episode 内的嵌入记忆缓冲
    - 总奖励为 $r_{t,i} = r_t^{\text{extrinsic}} + \beta_i \cdot r_{t,i}^{\text{intrinsic}}$
    - 设计动机：保留 NGU 最核心的新颖性驱动探索机制，无需 RND 等额外模块即可鼓励访问未见过的状态

2. **共享回放缓冲区（Shared Replay Buffer）**:

    - 功能：所有智能体的经验汇聚到一个中心化回放缓冲区
    - 核心思路：提高样本效率、减少非平稳性影响，每个智能体都可以从其他智能体的轨迹中学习
    - 设计动机：MARL 中个体经验有限，共享经验相当于扩大了有效样本量，使得更稳定的 Q 函数估计成为可能

3. **共享新颖性（Shared Novelty）**:

    - 功能：当一个状态嵌入被 $k$ 个不同智能体访问过后，标记为"不再新颖"
    - 核心思路：用余弦相似度检测投影嵌入间的相似性，当 $k$ 个智能体都在附近区域出现过时，该状态对所有智能体不再产生内在奖励
    - 设计动机：避免多个智能体重复探索相同区域，但过高的 $k$ 值会使信号过于稀疏

4. **异构 β 参数（Heterogeneous β）**:

    - 功能：为不同智能体分配不同的内在/外在奖励权衡参数，如 $\{0.1, 0.2, 0.4\}$
    - 核心思路：让部分智能体侧重探索（大 β）、部分侧重利用（小 β），实现角色分工
    - 设计动机：受混合策略思想启发，期望通过多样性提升整体性能

### 训练策略
所有配置使用相同超参数：学习率 0.001、缓冲区大小 $10^6$、batch size 128、$\gamma=0.99$、训练频率 16 步/次、梯度步数 4、目标网络更新间隔 2000 步、$\epsilon$ 从 1.0 线性衰减到 0.1。每个配置运行 200K 时间步、15 次独立种子。

## 实验关键数据

### 主实验

实验在 PettingZoo simple_tag_v3 环境中进行，多个追捕者（红色）合作捕获逃避者（蓝色），奖励稀疏且共享。逃避者采用默认启发式策略。

| 配置 | 回放缓冲区 | 平均回报排名 | 学习稳定性 |
|------|-----------|-------------|-----------|
| Multi-DQN | 独立 | 最低 | 不稳定、波动大 |
| Multi-NGU | 独立 | 中等偏上 | 明显更稳定 |
| Multi-NGU + Shared Novelty (k=1) | 独立 | 与 NGU 相当 | 稳定 |
| Multi-NGU + 异构β | 独立 | 略低于 NGU | 一般 |
| Multi-DQN | 共享 | 中等 | 有改善 |
| **Multi-NGU** | **共享** | **最高** | **最稳定** |
| Multi-NGU + Shared Novelty (k=1) | 共享 | 接近最佳 | 稳定 |
| Multi-NGU + 异构β | 共享 | 低于标准 NGU | 一般 |

### 消融实验

| 消融维度 | 最佳配置 | 关键发现 |
|---------|---------|---------|
| 回放缓冲区 | 共享 > 独立 | 共享后所有变体均有提升，NGU 优势更明显 |
| 新颖性共享阈值 k | k=1 最优 | k=2, k=3 性能退化；新颖性信号在多邻居平均后信息量不足 |
| 异构 β 值 | {0.1,0.2,0.4} > {0.2,0.4,0.6} | 较小 β 值组合更稳定，但均不如统一 β=0.1 |
| β 网格搜索 | β=0.1 最优 | 在 {0, 0.1, 0.5, 1.0} 中 β=0.1 最稳定且回报最高 |

### 关键发现
- **共享回放缓冲区是最大增益来源**：NGU 的内在探索 + 经验共享的组合效果最强
- **共享新颖性 k=1 时与标准 NGU 表现相当**，但 k>1 会降低性能——高 k 值使新颖性信号过于稀疏
- **异构 β 未能超越统一小 β**：在该环境中角色分工的收益不如一致的温和探索
- 共享新颖性在训练初期收敛更快但长期被标准 NGU 超越，暗示初始阶段的快速去冗余探索和长期的持续新颖性之间存在权衡

## 亮点与洞察
- **NGU 核心机制的可迁移性**：去掉 RND 和 UVFA 后 NGU 仍然有效，说明情景新颖性是其探索能力的核心驱动力。这一简化思路可以推广到其他复杂算法的多智能体适配
- **共享经验 ≈ 共享新颖性的概念等价性**很有启发：两者都本质上减少了已观察经验对内在奖励的贡献，但实际学习动态不同——前者通过丰富训练数据间接影响，后者直接改变奖励信号
- **系统性消融设计**：三个维度正交且全面，15 次独立运行和 95% 置信区间的报告方式为后续工作提供了可靠参考

## 局限与展望
- **仅在单一环境（simple_tag）上验证**，无法确认结论在竞争性、部分可观测等场景下的泛化性
- **仅用 DQN 作为底层算法**，未尝试 VDN、QMIX、MAPPO 等更强的 MARL 算法
- **去掉了 RND 和 UVFA**，在更复杂/更大规模环境中这些组件可能是关键
- **环境规模较小**（2D 有界竞技场），未在 StarCraft 等大规模基准上测试
- 共享新颖性的机制较为粗糙（硬阈值），可考虑注意力加权或渐进衰减方案

## 相关工作与启发
- **vs EMC**：EMC 用 Q 值预测误差做好奇心信号 + 情景记忆强化信息轨迹，架构更复杂；本文用更简洁的 NGU 机制取得可比效果
- **vs MACE**：MACE 允许分散化智能体共享局部新颖性来近似全局新颖性，而本文发现直接共享回放缓冲区比共享新颖性信号更有效
- 本文的发现暗示在 MARL 中"**经验共享比信号共享更重要**"，对设计新的分布式探索方法有核心启发

## 评分
- 新颖性: ⭐⭐⭐ 方法本身是直接扩展而非全新设计，但系统性消融有研究价值
- 实验充分度: ⭐⭐⭐ 消融全面（15 seeds × 8 配置），但仅一个环境，缺乏与 MARL SOTA 对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰、实验设置严谨、结论明确，自认 preliminary study 很诚实
- 价值: ⭐⭐⭐ 为 NGU→MARL 方向奠定了实证基础，共享回放的结论有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)
- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] A Theory of Multi-Agent Generative Flow Networks](a_theory_of_multi-agent_generative_flow_networks.md)

</div>

<!-- RELATED:END -->
