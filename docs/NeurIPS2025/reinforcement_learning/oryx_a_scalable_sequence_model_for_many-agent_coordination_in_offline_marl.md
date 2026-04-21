---
title: >-
  [论文解读] Oryx: a Scalable Sequence Model for Many-Agent Coordination in Offline MARL
description: >-
  [NeurIPS 2025][离线多智能体强化学习] 本文提出 Oryx，一种面向离线合作 MARL 的可扩展序列模型算法，将基于 Retention 的 Sable 架构与自回归形式的 ICQ 离线正则化结合，通过双解码器输出策略和 Q 值并利用反事实优势估计，在 65 个数据集上超过 80% 达到 SOTA，并展示了在 50 智能体规模下的稳健扩展能力。
tags:
  - NeurIPS 2025
  - 离线多智能体强化学习
  - 序列建模
  - 自回归策略
  - 多智能体协调
  - Retention机制
---

# Oryx: a Scalable Sequence Model for Many-Agent Coordination in Offline MARL

**会议**: NeurIPS 2025  
**arXiv**: [2505.22151](https://arxiv.org/abs/2505.22151)  
**代码**: [https://github.com/instadeepai/og-marl](https://github.com/instadeepai/og-marl)  
**领域**: 强化学习  
**关键词**: 离线多智能体强化学习, 序列建模, 自回归策略, 多智能体协调, Retention机制

## 一句话总结
本文提出 Oryx，一种面向离线合作 MARL 的可扩展序列模型算法，将基于 Retention 的 Sable 架构与自回归形式的 ICQ 离线正则化结合，通过双解码器输出策略和 Q 值并利用反事实优势估计，在 65 个数据集上超过 80% 达到 SOTA，并展示了在 50 智能体规模下的稳健扩展能力。

## 研究背景与动机
离线多智能体强化学习（Offline MARL）旨在仅从预收集数据中训练多智能体策略，无需进一步环境交互。这一设定对安全敏感和成本受限领域（自动驾驶、仓储物流、铁路调度等）至关重要——这些场景中有大量历史日志数据但实时试验成本高昂。

离线 MARL 面临两大核心挑战：

**外推误差（Extrapolation Error）**：智能体在训练中选择了数据分布外的动作，误差随联合动作空间指数增长。先前工作（ICQ, OMAR, CFCQL 等）通过策略约束或保守 Q 值估计来缓解，但通常只在少量智能体的简单场景上测试。

**协调失调（Miscoordination）**：离线训练中智能体无法主动交互，只能依赖历史数据中其他（通常次优的）策略产生的行为，可能发展出不兼容的策略。这个问题在长时间依赖和大量智能体参与时尤为严重。

现有方法要么只解决外推误差（如 ICQ、CFCQL），要么缺乏在大规模智能体和长时间依赖场景下的验证。核心 idea：将具有长上下文建模能力的 Retention 序列模型与自回归形式的 ICQ 离线约束结合，通过顺序策略更新显式解决协调失调问题。

## 方法详解

### 整体框架
Oryx 包含一个编码器和一个双头解码器。编码器使用 Retention 块处理每个智能体的观测序列（时间步 $t$ 到 $t+k$），在每个 Retention 块内同时对智能体 $(a_1, \ldots, a_n)$ 和时间上下文 $(t, \ldots, t+k)$ 进行联合推理。编码后的表征与数据集中的动作一起送入解码器，解码器通过两个头分别输出 Q 值和策略分布。

### 关键设计
1. **基于 Retention 的编码器-解码器架构**:

    - 采用 Sable (Mahjoub et al., 2025) 的 Retention 机制替代标准 softmax attention，使用衰减矩阵实现：训练时用 chunkwise mode 高效并行计算，推理时用 recurrent mode 维持隐状态
    - 相比 MAT（Transformer-based），Sable 在大量智能体场景下效率和稳定性更优
    - 双输出解码器：同时输出策略 logits（动作概率）和 Q 值估计
    - 与原始 Sable 不同，移除了编码器端的 Value head，改为编码器和解码器端到端训练
    - 设计动机：Retention 的线性复杂度保证了对大量智能体和长序列的可扩展性

2. **自回归 ICQ 损失（Autoregressive ICQ Loss）**:

    - 联合策略分解为自回归形式：$\pi(\boldsymbol{a}|\boldsymbol{\tau}) = \prod_{j=1}^n \pi^{i_j}(a^{i_j} | \boldsymbol{\tau}, \mathbf{a}^{i_{1:j-1}})$
    - 利用多智能体优势分解定理（Kuba et al., 2021）：$A(\boldsymbol{\tau}, \mathbf{a}) = \sum_{j=1}^n A^{i_j}(\boldsymbol{\tau}, \mathbf{a}^{i_{1:j-1}}, a^{i_j})$
    - 每个智能体的策略更新在 ICQ 框架下顺序进行，确保单调改进
    - 核心定理（Theorem 1）：自回归策略在 ICQ 正则化下可按智能体顺序优化，每步策略更新为：
    $\pi_*^{i_j} = \arg\max_{\pi^{i_j}} \mathbb{E}\left[-\frac{1}{Z^{i_{1:j}}} \log(\pi^{i_j}(a^{i_j} | \boldsymbol{\tau}, \mathbf{a}^{i_{1:j-1}})) \exp\left(\frac{A^{i_{1:j}}(\boldsymbol{\tau}, \mathbf{a}^{i_{1:j}})}{\alpha}\right)\right]$
    - 设计动机：顺序更新使每个智能体的策略改进条件于其他智能体已选动作，直接缓解协调失调

3. **反事实优势估计（Counterfactual Advantage）**:

    - 标准集中式优势估计的梯度方差上界包含 $(n-1)$ 因子，随智能体数线性增长
    - 采用 COMA 风格的反事实基线消除 $(n-1)$ 项：
    $A^{i_{1:j}}(\boldsymbol{\tau}, \mathbf{a}^{i_{1:j}}) = \sum_{m=1}^j \left[Q(\boldsymbol{\tau}, \mathbf{a}^{i_{1:m}}) - \sum_{a^{i_m}} \pi^{i_m}(a^{i_m} | \boldsymbol{\tau}, \mathbf{a}^{i_{1:m-1}}) Q(\boldsymbol{\tau}, \mathbf{a}^{i_{1:m}})\right]$
    - 设计动机：降低梯度估计方差，使算法在大量智能体下保持稳定训练

### 损失函数 / 训练策略
- **Critic 损失**：自回归形式的 SARSA-like ICQ 更新，从数据集采样目标动作，用隐式重要性权重：
$$J_Q(\phi) = \mathbb{E}_\mathcal{B}\left[\left(r + \gamma \frac{\exp(Q_{\phi^-}^{i_j}/\alpha)}{Z(\boldsymbol{\tau}')} Q_{\phi^-}^{i_j} - Q_\phi^{i_j}\right)^2\right]$$
- **Policy 损失**：最小化 ICQ 最优策略与当前策略之间的 KL 散度
- 每步随机抽取智能体排列 $i_{1:n}$ 以避免固定顺序偏差
- 使用目标网络 $\phi^-$ 稳定训练

## 实验关键数据

### 主实验

| 环境 | 数据集数 | Oryx ≥ SOTA 占比 | 关键对比 |
|------|---------|------------------|---------|
| SMAC | 43 | 34/43 (79%) | vs CFCQL, OMIGA, ICQ-MA |
| MAMuJoCo | 16 | 14/16 (88%) | vs CFCQL, OMIGA |
| RWARE | 6 | 6/6 (100%) | 多个场景提升近20% |
| **总计** | **65** | **54/65 (83%)** | |

| 架构对比 (Oryx vs MAT+ICQ) | SMAC Median↑ | SMAC IQM↑ | RWARE Median↑ |
|---------------------------|-------------|-----------|--------------|
| MAT+ICQ | 0.71 | 0.67 | 0.85 |
| **Oryx** | **0.91** | **0.87** | **0.89** |

### 消融实验

| 配置 | T-Maze Replay | T-Maze Expert | 说明 |
|------|---------------|---------------|------|
| I-ICQ | 0.0±0.0 | 0.0±0.0 | 独立学习完全失败 |
| MAICQ | 0.0±0.0 | 0.0±0.0 | CTDE 也失败 |
| Oryx w/o 自回归动作 | 0.0±0.0 | 0.0±0.0 | 自回归是关键 |
| Oryx w/o 记忆 | 0.58±0.04 | 0.63±0.04 | 记忆对长期协调重要 |
| Oryx w/o ICQ | 0.0±0.0 | 0.0±0.0 | 离线约束不可缺少 |
| **Oryx (完整)** | **0.99±0.01** | **0.94±0.03** | 三者缺一不可 |

### 关键发现
- Oryx 的三个核心组件（自回归动作选择、记忆机制、ICQ 离线正则化）各自不可或缺，去掉任何一个都导致 T-Maze 任务完全失败
- 在 Connector 环境中，23 到 50 个智能体规模下 Oryx 保持接近专家性能，而 MAICQ 性能急剧下降（50 智能体时仅约 25% 专家水平）
- 在 RWARE 的长时间步（500 步）稀疏奖励场景中表现尤为突出，多个数据集提升近 20%

## 亮点与洞察
- **统一解决离线 MARL 的两大核心挑战**：通过自回归策略更新解决协调失调，通过 ICQ 约束解决外推误差，两者在序列建模框架下自然融合。这种设计的优雅之处在于自回归结构同时服务于动作生成和策略约束。
- **50 智能体规模的系统性验证**：当前大多数离线 MARL 工作仅在 3-10 个智能体上测试，Oryx 通过 Connector 环境展示了在 50 智能体下的稳健性能，切实推动了可扩展性边界。

## 局限与展望
- 自回归策略更新的顺序依赖引入额外计算开销，且解码时间与智能体数量线性增长
- 在连续动作空间（MAMuJoCo）上的优势小于离散动作空间（SMAC/RWARE），可能因为连续空间中 ICQ 的重要性采样效率较低
- 数据集质量和覆盖度对性能影响显著，但论文缺少对数据集质量的系统分析
- 随机智能体排列虽然避免了固定顺序偏差，但可能引入额外方差

## 相关工作与启发
- **vs MAT (Multi-Agent Transformer)**: MAT 在在线设定中使用 Transformer 自回归动作选择，Oryx 将其扩展到离线场景，并用 Retention 替代 attention 以获得更好的扩展性
- **vs MAICQ**: MAICQ 用 RNN 记忆 + CTDE 值分解 + 非自回归 ICQ，Oryx 在所有三个维度上做了改进（Retention 记忆、自回归 ICQ、反事实优势）
- **vs CFCQL/OMIGA**: 这些工作通过保守正则化或分布约束处理外推误差，但缺乏对大规模智能体协调的显式处理

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 Retention 序列模型与自回归 ICQ 结合的思路有创新，Theorem 1 提供了理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 65 个数据集、T-Maze 验证、50 智能体扩展性测试、架构消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 方法推导清晰，但个别符号使用不够统一
- 价值: ⭐⭐⭐⭐⭐ 在离线 MARL 领域树立了新的 SOTA，且代码和数据集开源，对后续研究有直接推动作用

<!-- RELATED:START -->

## 相关论文

- [Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models](communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied.md)
- [Partial Action Replacement: Tackling Distribution Shift in Offline MARL](../../AAAI2026/reinforcement_learning/partial_action_replacement_tackling_distribution_shift_in_offline_marl.md)
- [Incremental Sequence Classification with Temporal Consistency](incremental_sequence_classification_with_temporal_consistency.md)
- [Scalable Policy-Based RL Algorithms for POMDPs](scalable_policy-based_rl_algorithms_for_pomdps.md)
- [Horizon Reduction Makes RL Scalable](horizon_reduction_makes_rl_scalable.md)

<!-- RELATED:END -->
