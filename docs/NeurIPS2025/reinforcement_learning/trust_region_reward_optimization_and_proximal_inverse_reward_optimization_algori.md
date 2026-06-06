---
title: >-
  [论文解读] Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm
description: >-
  [NeurIPS 2025][强化学习][Trust Region] 提出 TRRO 理论框架和 PIRO 实用算法，通过 Minorization-Maximization 过程保证 IRL 中奖励函数更新的单调改进，实现了逆强化学习领域类似于 TRPO/PPO 在正向 RL 中的稳定性保证。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "Trust Region"
  - "Reward Learning"
  - "Non-Adversarial IRL"
  - "Monotonic Improvement"
---

# Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm

**会议**: NeurIPS 2025  
**arXiv**: [2509.23135](https://arxiv.org/abs/2509.23135)  
**代码**: [有](https://github.com/PolynomialTime/PIRO)  
**领域**: Reinforcement Learning  
**关键词**: Inverse Reinforcement Learning, Trust Region, Reward Learning, Non-Adversarial IRL, Monotonic Improvement

## 一句话总结

提出 TRRO 理论框架和 PIRO 实用算法，通过 Minorization-Maximization 过程保证 IRL 中奖励函数更新的单调改进，实现了逆强化学习领域类似于 TRPO/PPO 在正向 RL 中的稳定性保证。

## 研究背景与动机

逆强化学习（IRL）从专家演示中学习奖励函数，现代 IRL 方法主要有两种范式：

**对抗式 IRL**（如 GAIL、AIRL）：将奖励学习建模为极小极大博弈，交替优化奖励和策略。理论上优雅但实践中训练不稳定，对超参数敏感。

**非对抗式 IRL**（如 SQIL、IQ-Learn、ML-IRL）：通过能量模型将奖励和策略耦合，联合更新。经验稳定性更好，但缺乏对奖励更新的原则性控制——无法保证每一步更新都朝正确方向前进。

论文指出一个关键观察：现有非对抗式 IRL 方法本质上都在**最大化专家行为的似然**（等价于最小化模仿差距）。这个统一视角引出核心思路：如果能**保证每步更新都提高似然**，就能实现 IRL 的稳定训练。

这与正向 RL 中 TRPO 的思路完美对称：
- TRPO 保证在固定奖励下策略的单调改进
- TRRO 保证在给定专家行为下奖励的单调改进

论文自称填补了这个"对称图景的右半边"。

## 方法详解

### 整体框架

TRRO/PIRO 采用非对抗式、显式奖励学习（ER）路线：
1. 统一视角：证明 SQIL、IQ-Learn、f-IRL、ML-IRL 都在优化专家行为似然
2. 理论贡献：TRRO 框架通过 MM 算法保证逆奖励优化的单调改进
3. 实用算法：PIRO 通过自适应正则化和近似策略优化实现 TRRO

### 关键设计

1. **似然目标的等价形式（Proposition 1）**：

    - ML-IRL 的对数似然 $\ell(\boldsymbol{\theta}) = \mathbb{E}_{\rho^{\pi_E}}[\log \pi_{\boldsymbol{\theta}}(\mathbf{a}|\mathbf{s})]$
    - 等价于模仿差距：$\ell(\boldsymbol{\theta}) = J(\pi_E, r_{\boldsymbol{\theta}}) - J(\pi_{\boldsymbol{\theta}}, r_{\boldsymbol{\theta}})$
    - 梯度为两个占用度量下的奖励梯度之差：$\nabla_{\boldsymbol{\theta}} \ell = \mathbb{E}_{\rho^{\pi_E}}[\nabla r_{\boldsymbol{\theta}}] - \mathbb{E}_{\rho^{\pi_{\boldsymbol{\theta}}}}[\nabla r_{\boldsymbol{\theta}}]$
    - 这绕过了内层 RL 循环，将嵌套优化简化为单循环

2. **Trust Region Reward Optimization (TRRO, 定理 3)**：

    - 引入代理函数 $\ell_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta})$：用旧策略 $\pi_{\text{old}}$ 替代新策略计算模仿差距
    - Proposition 2 证明代理函数在 $\boldsymbol{\theta}_{\text{old}}$ 处与原目标一阶匹配
    - 定理 3 建立下界：$\ell(\boldsymbol{\theta}_{\text{new}}) \geq \ell_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}_{\text{new}}) - C\epsilon_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}_{\text{new}})$
    - 其中 $\epsilon = \max_{s,a} |r_{\boldsymbol{\theta}_{\text{new}}} - r_{\boldsymbol{\theta}_{\text{old}}}|$ 是奖励变化量
    - 最大化下界保证 $\ell$ 单调不减（推论 4）
    - 这是一个 MM 算法：代理函数 minorize 原目标，在 $\boldsymbol{\theta}_{\text{old}}$ 处相切

3. **Proximal Inverse Reward Optimization (PIRO)**：

    - 理论常数 $C$ 太大，用可调系数 $\mu > 0$ 替代
    - $\epsilon$ 的最大范数不可微，用 $L^2$ 范数在专家数据和策略 rollout 上的估计替代
    - 目标函数：$L_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) = \ell_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta}) - \mu \bar{\epsilon}_{\boldsymbol{\theta}_{\text{old}}}(\boldsymbol{\theta})$
    - $\mu$ 自适应调节：若 $\bar{\epsilon} > \bar{\epsilon}^{\text{target}} \times x$，则 $\mu \leftarrow \mu \times y$（反之亦然）
    - 策略用 SAC 的若干轮迭代近似优化（而非精确求解）

### 损失函数 / 训练策略

PIRO 的交替更新：
- **策略更新**：$k$ 轮 SAC 迭代（基于当前奖励 $r_{\boldsymbol{\theta}_{\text{old}}}$）
- **奖励更新**：$n$ 步梯度上升，梯度为 $\nabla_{\boldsymbol{\theta}} L = \mathbb{E}_{\hat{D}_E}[\nabla r_{\boldsymbol{\theta}}] - \mathbb{E}_{D_S}[\nabla r_{\boldsymbol{\theta}}] - \mu \nabla \bar{\epsilon}$
- 当 $k=n=1, \mu=0$ 时退化为一般非对抗式 IRL

## 实验关键数据

### 主实验：MuJoCo 和 Gym Robotics

| 任务 | Expert | GAIL | AIRL | HyPE | IQ-Learn | ML-IRL | f-IRL | **PIRO** | 提升 |
|------|--------|------|------|------|----------|--------|-------|----------|------|
| Ant-v4 | 5926 | 997 | 991 | 2801 | 3590 | 5383 | 980 | **5967** | +585 |
| Humanoid-v4 | 5501 | 508 | 281 | 718 | 1848 | 5573 | 470 | **5955** | +382 |
| Walker2d-v4 | 5525 | 4158 | 73 | 1479 | 3023 | 4795 | 244 | **5644** | +849 |
| AntMaze-UMaze | 35.6 | 5.2 | 4.5 | 11.9 | 3.9 | 4.2 | 3.6 | **25.7** | +13.8 |
| AntMaze-Large | 11.5 | 0.9 | 3.4 | 1.5 | 0.8 | 0.3 | 0.9 | **8.8** | +5.4 |

### 消融/分析实验

| 分析维度 | 结果 |
|----------|------|
| 学习稳定性 | PIRO 曲线最平滑，其他方法波动大或性能崩溃 |
| 样本效率 | PIRO 收敛速度与最快基线持平，但最终性能更高 |
| State-only 奖励恢复 | 7×7 网格世界中恢复的奖励与 ground truth 高度一致 |
| 奖励迁移 | LunarLander 学到的奖励在添加风力后仍能训练有效策略 |
| 超参数敏感性 | $x, y \in (1, 2)$, $\bar{\epsilon}^{\text{target}} \in (0.1, 1)$ 范围内不敏感 |

### 关键发现

- PIRO 在几乎所有任务上超越或匹配 SOTA，尤其在高难度任务（Humanoid、AntMaze、AdroitHand）上优势显著
- 训练稳定性是最大优势——ML-IRL 等基线在复杂任务上经常出现性能崩溃
- 虽然单步计算开销略高，但稳定收敛导致总计算量并不增加
- 唯一弱於基线的任务是 Hopper-v4（-173.7），说明近端约束可能在简单任务上过于保守

## 亮点与洞察

- **TRPO 的逆对称**这个视角非常优雅：正向 RL 信任域保证策略改进 ↔ 逆向 RL 信任域保证奖励改进
- 将多种非对抗式 IRL 方法统一到似然最大化框架下是重要的理论贡献
- PIRO 的实现非常简洁：在 SAC 基础上只需添加若干奖励梯度步，工程友好
- 奖励迁移实验展示了显式奖励学习相比隐式方法的优势——奖励不与环境动力学耦合

## 局限与展望

- 理论保证假设精确策略优化（实际用有限步 SAC 近似），理论和实践之间有 gap
- On-policy 采样依赖可能限制在样本昂贵任务上的扩展性
- 理论常数 $C$ 太大以至于直接使用不实际，需要自适应 $\mu$ 来"放松"约束
- 可扩展到 RLHF 场景——从人类反馈中学习奖励模型与 IRL 有天然联系

## 相关工作与启发

- GAIL/AIRL 等对抗式方法的不稳定性是 IRL 领域的长期痛点，PIRO 提供了有保证的替代方案
- 与 ML-IRL 的关系密切（PIRO 可视为加了信任域约束的 ML-IRL）
- TRPO→PPO 的简化路径启发了 TRRO→PIRO 的设计
- 对 LLM 对齐中的奖励建模有潜在应用价值——RLHF 中的奖励学习本质上是 IRL

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — IRL 稳定性的首个形式化保证，TRPO 的逆对称视角优雅
- 实验充分度: ⭐⭐⭐⭐⭐ — 9 个任务 + 13 个基线 + 稳定性/效率/迁移/敏感性全面分析
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨，实践算法简洁
- 价值: ⭐⭐⭐⭐⭐ — 对 IRL 理论和实践都有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Inverse Optimization Latent Variable Models for Learning Costs Applied to Route Problems](inverse_optimization_latent_variable_models_for_learning_costs_applied_to_route_.md)
- [\[ICML 2025\] Embedding Safety into RL: A New Take on Trust Region Methods](../../ICML2025/reinforcement_learning/embedding_safety_into_rl_a_new_take_on_trust_region_methods.md)
- [\[NeurIPS 2025\] Risk-Averse Total-Reward Reinforcement Learning](risk-averse_total-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](reward-aware_proto-representations_in_reinforcement_learning.md)
- [\[NeurIPS 2025\] DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
