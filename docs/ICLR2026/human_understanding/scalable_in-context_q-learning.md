---
title: >-
  [论文解读] Scalable In-Context Q-Learning
description: >-
  [ICLR 2026][人体理解][情境RL] 提出 S-ICQL——将动态规划（Q-learning）和世界模型引入监督式 ICRL 框架，通过多头 Transformer 同时预测策略和情境值函数，预训练世界模型构建轻量级精确提示，advantage-weighted regression 提取策略，在离散和连续环境中从次优数据学习时一致超越所有基线。
tags:
  - ICLR 2026
  - 人体理解
  - 情境RL
  - Q学习
  - 世界模型
  - 动态规划
  - 高效提示
---

# Scalable In-Context Q-Learning

**会议**: ICLR 2026  
**arXiv**: [2506.01299](https://arxiv.org/abs/2506.01299)  
**代码**: [GitHub](https://github.com/NJU-RL/SICQL)  
**领域**: 强化学习/情境学习  
**关键词**: 情境RL, Q学习, 世界模型, 动态规划, 高效提示

## 一句话总结

提出 S-ICQL——将动态规划（Q-learning）和世界模型引入监督式 ICRL 框架，通过多头 Transformer 同时预测策略和情境值函数，预训练世界模型构建轻量级精确提示，advantage-weighted regression 提取策略，在离散和连续环境中从次优数据学习时一致超越所有基线。

## 背景与动机

**领域现状**：In-context RL（ICRL）将语言模型的情境学习能力扩展到决策领域——在多任务离线数据上预训练 Transformer，测试时通过 prompt 适应新任务而无需更新参数。现有方法分为两大分支：Algorithm Distillation（AD，用学习历史作为上下文自回归预测动作）和 Decision-Pretrained Transformer（DPT，基于交互转移序列预测最优动作）。

**现有痛点**：

- **监督预训练的固有局限**：AD 和 DPT 本质是模仿学习，无法超越收集数据的质量——缺乏 stitching 能力（将次优轨迹片段拼接成全局最优行为的能力）
- **AD 需要长 horizon 上下文**：需要完整的学习历史作为 prompt，且会继承次优行为的梯度更新规则
- **DPT 需要 oracle 最优动作标注**：在实际场景中往往不可行
- **原始轨迹作为提示效率低**：token 数量多且高度冗余，行为策略和任务信息纠缠在一起，导致有偏的任务推断

**核心矛盾**：RL 的精髓在于通过值函数的动态规划更新实现奖励最大化，而现有 ICRL 方法完全停留在监督学习范式中，放弃了 RL 的核心优势。如何在保持监督预训练的可扩展性和稳定性的同时，引入动态规划来释放从次优数据中学习的潜力？

**本文方案**：利用 RL 的两个基本性质——(1) 动态规划（Bellman backup）的 stitching 能力和 (2) 世界模型对环境动力学的精确表征——设计一个可扩展的 ICRL 框架，同时实现高效的奖励最大化和精确的任务泛化。

## 方法详解

### 整体框架

S-ICQL 包含三个核心组件：(a) 预训练的通用世界模型，用于将原始轨迹压缩为轻量级任务提示 $\beta$；(b) 多头 Transformer 网络，同时预测策略 $\pi_\theta(a|s;\beta)$、状态值函数 $V_\theta(s;\beta)$ 和动作值函数 $Q_\theta(s,a;\beta)$；(c) 联合优化目标，结合 Bellman backup（Q-learning）和优势加权回归（策略提取）。

问题设定为多任务离线 RL：任务 $M^i = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}^i, \mathcal{R}^i, \gamma \rangle \sim P(M)$，共享状态-动作空间但奖励函数或转移动力学不同。每个任务的离线数据集 $\mathcal{D}^i$ 由任意行为策略收集。

### 关键设计 1：世界模型 → 轻量级精确提示

核心洞察：环境动力学 $p(s', r | s, a)$ 完整刻画了决策任务，且天然不受行为策略的影响。因此使用世界模型来编码任务信息比直接使用原始轨迹更精确、更紧凑。

**世界模型架构**：包含上下文编码器 $E_\phi$ 和动力学解码器 $D_\varphi$：

- 上下文编码器：将近 $k$ 步的经验 $\eta_t^i = (s_{t-k}, a_{t-k}, r_{t-k}, \ldots, s_t, a_t)^i$ 压缩为任务表征 $z_t^i = E_\phi(\eta_t^i)$
- 动力学解码器：条件于任务表征预测即时奖励和下一状态 $[\hat{r}_t, \hat{s}_{t+1}] = D_\varphi(s_t, a_t; z_t^i)$

**预训练目标**为最小化奖励和状态转移的预测误差：

$$\mathcal{L}(\phi, \varphi) = \mathbb{E}_{\eta_t^i \sim M^i} \left[ \| [r_t, s_{t+1}] - D_\varphi(s_t, a_t; z_t^i) \|_2^2 \mid z_t^i = E_\phi(\eta_t^i) \right]$$

预训练完成后冻结世界模型，将 $h$ 步轨迹转化为轻量级提示：

$$\beta^i := [z_1^i, z_2^i, \ldots, z_h^i] = [E_\phi(\eta_1^i), E_\phi(\eta_2^i), \ldots, E_\phi(\eta_h^i)]$$

相比 AD 需要长学习历史作为 context，此提示结构更紧凑且包含更精确的任务信息。

### 关键设计 2：情境 Q-Learning（Bellman Backup + 期望回归）

**Q 函数训练**——最小化 Bellman 误差，引入 stitching 能力：

$$\mathcal{L}_Q(\theta) = \mathbb{E}_{(s_t^i, a_t^i, s_{t+1}^i) \sim \mathcal{D}^i} \left[ \left( r(s_t^i, a_t^i) + \gamma V_\theta(s_{t+1}^i; \beta^i) - Q_\theta(s_t^i, a_t^i; \beta^i) \right)^2 \right]$$

**状态值函数训练**——使用 expectile regression 拟合 Q 函数的上尾分位数：

$$\mathcal{L}_V(\theta) = \mathbb{E}_{(s_t^i, a_t^i) \sim \mathcal{D}^i} \left[ L_2^\omega \left( Q_{\hat{\theta}}(s_t^i, a_t^i; \beta^i) - V_\theta(s_t^i; \beta^i) \right) \right]$$

其中 $L_2^\omega(u) = |\omega - \mathbb{1}(u < 0)| \cdot u^2$ 是非对称损失函数，$\omega \in (0.5, 1)$。当 $Q > V$ 时赋予更大权重 $\omega$，当 $Q < V$ 时权重仅为 $1-\omega$，从而近似 $\max_a Q(s, a)$。

### 关键设计 3：优势加权回归策略提取

将情境值函数蒸馏到策略提取中，使用 advantage-weighted regression：

$$\mathcal{L}_\pi(\theta) = -\mathbb{E}_{(s_t^i, a_t^i) \sim \mathcal{D}^i} \left[ \exp\left( \frac{1}{\lambda} \left( Q_{\hat{\theta}}(s_t^i, a_t^i; \beta^i) - V_\theta(s_t^i; \beta^i) \right) \right) \cdot \log \pi_\theta(a_t^i | s_t^i; \beta^i) \right]$$

优势值 $A = Q - V$ 越大的动作获得越大的训练权重。这不是简单的行为克隆，而是学习在数据集约束下最大化 Q 值的策略。总损失为三者的加权和：

$$\mathcal{L}(\theta) = \mathsf{c}_1 \mathcal{L}_\pi(\theta) + \mathsf{c}_2 \mathcal{L}_Q(\theta) + \mathsf{c}_3 \mathcal{L}_V(\theta)$$

系数设为 $(1:1:1)$，整个多头 Transformer 端到端联合优化。

## 实验结果

### 主实验：Mixed 数据集上的 Few-shot 评估

| 方法 | DarkRoom | Push | Reach | Cheetah-Vel | Walker-Param | Ant-Dir |
|:-----|:---------|:-----|:------|:------------|:-------------|:--------|
| DPT | 22.12 | 362.74 | 736.72 | -78.35 | 257.11 | 591.31 |
| AD | 42.72 | 604.50 | 738.96 | -67.37 | 424.82 | 215.01 |
| IDT | 40.70 | 621.58 | 790.68 | -59.46 | 343.01 | 631.83 |
| DICP | 59.76 | 487.28 | 706.46 | -66.53 | 403.90 | 745.05 |
| DIT | 30.90 | 633.58 | 758.92 | -74.50 | 253.94 | 723.49 |
| IC-IQL | 60.12 | 646.08 | 773.33 | -56.53 | 391.38 | 713.26 |
| **S-ICQL** | **66.05** | **653.04** | **806.97** | **-35.48** | **466.72** | **813.34** |

S-ICQL 在所有 6 个环境中均取得最佳表现。在复杂环境（Cheetah-Vel、Ant-Dir）中优势尤为明显，分别将误差从 -56.53 降至 -35.48（提升 37%）、从 745.05 提升至 813.34。

### 消融实验：各组件贡献分析

| 消融配置 | Reach | Cheetah-Vel | Ant-Dir |
|:---------|:------|:------------|:--------|
| w/o\_cq（去掉世界模型+Q学习 = DPT） | 736.72 | -78.35 | 591.31 |
| w/o\_c（去掉世界模型） | 792.09 | -56.19 | 693.87 |
| w/o\_q（去掉Q学习） | 752.41 | -63.66 | 784.07 |
| **S-ICQL（完整）** | **806.97** | **-35.48** | **813.34** |

世界模型和 Q-learning 两个组件各自贡献显著增益。去掉任一组件都导致性能下降，去掉两个退化为 DPT 时性能最差。Ant-Dir 上 Q-learning 贡献更大（+29 vs. w/o\_q），体现了 stitching 在复杂任务中的重要性。

### OOD 泛化实验

| 方法 | Cheetah-Vel (OOD) | Ant-Dir (OOD) |
|:-----|:-----------------|:-------------|
| DPT | -137.26 | 205.29 |
| IC-IQL | -101.89 | 540.20 |
| **S-ICQL** | **-83.45** | **664.95** |

在分布外任务上 S-ICQL 同样显著领先，Ant-Dir 上超出第二名 IC-IQL 约 23%，验证了世界模型赋予的 OOD 泛化能力。

## 评价

**评分**: ⭐⭐⭐⭐⭐

**优点**：

- 创新性地将动态规划（Q-learning stitching）和世界模型两大 RL 核心概念引入 ICRL，解决了监督预训练无法超越收集数据质量的根本局限
- 多头 Transformer 架构设计优雅——仅增加两个轻量级头即可同时预测策略和值函数，参数增量可忽略
- 世界模型驱动的提示构建方法精炼且有理论依据——环境动力学天然不受行为策略影响
- 实验极其全面：6 个标准环境 + 2 个复杂环境 + OOD 泛化 + stitching 验证 + 7 个竞争基线

**不足**：

- 提示长度与采样轨迹长度绑定，在长 horizon 交互问题中可能过长
- 世界模型需要额外的预训练阶段，增加了训练流程的复杂度
- 仅在标准 RL benchmark 上验证，未涉及更复杂的实际决策场景（如机器人操作的 sim-to-real）

<!-- RELATED:START -->

## 相关论文

- [EgoHandICL: Egocentric 3D Hand Reconstruction with In-Context Learning](egohandicl_egocentric_3d_hand_reconstruction_with_in-context_learning.md)
- [COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics](cold-steer_steering_large_language_models_via_in-context_one-step_learning_dynam.md)
- [In-Context Compositional Learning via Sparse Coding Transformer](../../NeurIPS2025/human_understanding/in-context_compositional_learning_via_sparse_coding_transformer.md)
- [Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow](scalable_exploration_for_high-dimensional_continuous_control_via_value-guided_fl.md)
- [Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](../../NeurIPS2025/human_understanding/counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)

<!-- RELATED:END -->
