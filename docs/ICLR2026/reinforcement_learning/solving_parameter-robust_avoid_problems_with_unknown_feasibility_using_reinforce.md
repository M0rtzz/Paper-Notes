---
title: >-
  [论文解读] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning
description: >-
   提出 Feasibility-Guided Exploration (FGE)，同时识别可行参数子集并学习在该子集上安全的策略，解决可行性未知的参数鲁棒避障问题，在 MuJoCo 任务中比最佳现有方法多覆盖 50% 以上。
tags:

---

# Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2602.15817](https://arxiv.org/abs/2602.15817)
- **代码**: [https://oswinso.xyz/fge](https://oswinso.xyz/fge)
- **领域**: reinforcement_learning
- **关键词**: safe RL, Hamilton-Jacobi, robust optimization, feasibility, curriculum learning, MuJoCo

## 一句话总结
提出 Feasibility-Guided Exploration (FGE)，同时识别可行参数子集并学习在该子集上安全的策略，解决可行性未知的参数鲁棒避障问题，在 MuJoCo 任务中比最佳现有方法多覆盖 50% 以上。

## 研究背景与动机
- **Hamilton-Jacobi (HJ) 安全控制**是获得最大安全初始状态集合的强大工具，但经典方法受维度灾难限制。
- **深度 RL 逼近 HJ**：用 RL 学习近似最优控制策略，但 RL 优化期望回报 vs. 最坏情况安全形成根本性不匹配——在低概率但仍应安全的状态上表现可能很差。
- **鲁棒优化**方案（如 RARL）对初始条件集合做最坏情况优化，但前提是该集合可行（即存在安全策略）。如果包含不可行参数，所有策略都同样差，导致退化。
- **关键困难**：确定可行参数集合本身就是 HJ 分析的目标——先验未知！
- 例子：自动驾驶中，暴风雪+高速可能无论如何都不安全；用这种不可能的场景训练会阻止模型学好晴天场景。

## 方法详解

### 整体框架
FGE 同时完成两个任务：
1. 识别最大可行参数子集 $\Theta^* \subseteq \Theta$（哪些参数下存在安全策略）
2. 学习一个在 $\Theta^*$ 上安全的鲁棒策略

### 关键设计 1：可行性分类器
核心挑战是标签不对称：观测到安全 = 确定可行；观测到不安全 ≠ 确定不可行（可能只是策略不好）。

构建混合分布训练分类器：

$$p_{\text{mix}}(\mathfrak{f}, \theta) = \alpha \cdot p^*(\mathfrak{f}|\theta) p_{\mathcal{D}_\mathfrak{f}}(\theta) + (1-\alpha) \cdot p^\pi(\mathfrak{f}|\theta) \rho(\theta)$$

- 第一项：已确认安全的参数（可靠正标签）
- 第二项：当前策略下的在线样本（可能含假阴性）
- 通过变分推断拟合 $q_\psi(\mathfrak{f}=1|\theta)$，阈值化得到分类器

**理论保证**：
- 零假阳性：不可行参数永远不会被标为可行
- 可控假阴性率：通过调节 $\alpha$ 和 $\rho$ 控制

### 关键设计 2：鞍点优化
将鲁棒安全控制建模为 maximin 问题，采用在线学习的鞍点方法而非 RARL 的对抗性策略：

$$\pi_{t+1} = \arg\max_\pi \mathbb{E}_{\theta \sim \mathcal{D}_{\theta,t}}[J(\pi, \theta)]$$
$$\theta_{t+1} = \arg\min_{\theta} J(\pi_t, \theta), \quad \theta \sim p(\cdot | \theta \in \Theta^*)$$

使用 Follow-the-Regularized-Leader (FTRL) 结合 PPO 做策略更新，维护 rehearsal buffer 存储历史最坏情况参数。

### 关键设计 3：探索分布扩展
FGE 将采样分布分为三部分（图 1）：
- **基础分布**：原始参数采样
- **探索分布**：提升未被观测为安全的参数的采样概率
- **排练分布**：对先前已解决但可能退化的参数采样（近似最佳响应）

三者组合平衡：最大化安全率增益 + 最小化安全率损失。

### 损失函数
策略训练使用标准 RL 目标（PPO），奖励为负指示器：

$$r_k = -\mathbb{1}\{h_\theta(\bm{s}_k) > 0\}$$

安全 = 奖励 0，进入不安全状态 = 奖励 -1，训练到首次违规即终止。

## 实验关键数据

### 主实验：MuJoCo 安全覆盖率

| 环境 | Domain Rand. | RARL | FGE (Ours) | 提升 |
|------|-------------|------|-----------|------|
| Ant (避障) | ~40% | ~45% | **~70%** | +56% |
| Humanoid (避障) | ~35% | ~40% | **~65%** | +63% |
| HalfCheetah | ~50% | ~55% | **~78%** | +42% |

> FGE 在所有挑战性 MuJoCo 任务中比最佳基线多覆盖 50%+ 的安全参数集合。

### 消融实验：各组件贡献

| 消融设置 | 安全覆盖率 | 说明 |
|---------|----------|------|
| FGE (完整) | ~70% | 基准 |
| 无可行性分类器 | ~50% | 不可行参数干扰训练 |
| 无探索分布 | ~55% | 探索不充分 |
| 无排练分布 | ~60% | 已学技能退化 |
| 用密度模型替代分类器 | ~58% | 不如混合分布分类器 |

### 关键发现
1. 标准域随机化和 RARL 在参数可行性未知时严重受限
2. 可行性分类器的零假阳性保证对训练稳定性至关重要
3. FTRL 比 GDA（RARL 近似的方法）在鞍点问题上收敛更稳定
4. 探索和排练分布的平衡对持续扩展安全集合不可或缺

## 亮点与洞察
- **新问题定义**：参数鲁棒避障 + 未知可行性，填补了安全 RL 与 HJ 分析的重要空白
- **正标签学习**：巧妙处理单侧标签问题（只有正可靠），理论保证零假阳性
- **在线学习视角**：用鞍点方法替代不稳定的对抗性 RL，理论保证更强
- **实用三采样策略**：base + explore + rehearse 的组合灵感来自课程学习和在线学习

## 局限性
- 理论收敛保证依赖凸凹性和精确最佳响应等假设，实际中不完全满足
- 可行性分类器在高维参数空间中的准确性需更多验证
- 仅考虑确定性动力学，随机系统扩展未讨论
- MuJoCo 环境相比真实机器人仍有差距

## 相关工作
- **HJ 安全控制**: DeepReach (Bansal et al., 2021), ISAACS (Hsu et al., 2023), So et al. (2024)
- **鲁棒 RL**: RARL (Pinto et al., 2017), WCSAC (Yang et al., 2021)
- **无监督环境设计 (UED)**: PAIRED (Dennis et al., 2020), PLR (Jiang et al., 2021)
- **安全 RL**: Constrained MDP (Altman, 1999), SauteRL (Sootla et al., 2022)

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 全新问题定义，可行性未知的鲁棒安全控制
- 理论深度: ⭐⭐⭐⭐ — 分类器保证、鞍点收敛分析
- 实验充分性: ⭐⭐⭐⭐ — 多 MuJoCo 环境、详细消融
- 实用价值: ⭐⭐⭐⭐ — 对机器人安全部署有直接意义

<!-- RELATED:START -->

## 相关论文

- [Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)
- [$\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](textbfre2_unlocking_llm_reasoning_via_reinforcement_learning_with_re-solving.md)
- [Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)
- [Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)

<!-- RELATED:END -->
