---
title: >-
  [论文解读] Online Pre-Training for Offline-to-Online Reinforcement Learning
description: >-
  [ICML2025][强化学习][offline-to-online RL] 提出 OPT 方法，在离线预训练和在线微调之间引入"在线预训练"阶段，通过新增一个独立值函数并用元适应目标训练，解决离线预训练智能体因值估计不准而导致在线微调性能下降的问题，在 D4RL 基准上平均提升约 30%。 离线转在线 RL（offline…
tags:
  - "ICML2025"
  - "强化学习"
  - "offline-to-online RL"
  - "值函数估计"
  - "元适应"
  - "在线预训练"
  - "D4RL"
---

# Online Pre-Training for Offline-to-Online Reinforcement Learning

**会议**: ICML2025  
**arXiv**: [2507.08387](https://arxiv.org/abs/2507.08387)  
**代码**: 待确认  
**领域**: 离线转在线强化学习 (Offline-to-Online RL)  
**关键词**: offline-to-online RL, 值函数估计, 元适应, 在线预训练, D4RL

## 一句话总结

提出 OPT 方法，在离线预训练和在线微调之间引入"在线预训练"阶段，通过新增一个独立值函数并用元适应目标训练，解决离线预训练智能体因值估计不准而导致在线微调性能下降的问题，在 D4RL 基准上平均提升约 30%。

## 研究背景与动机

离线转在线 RL（offline-to-online RL）旨在先用离线数据集预训练智能体，再通过在线交互微调以提升性能。然而，近期研究揭示了一个**反直觉现象**：离线预训练的智能体在在线微调阶段往往**不如从零开始训练**的效果好。

核心原因在于**值函数估计不准确**：

- 离线 RL 中值函数仅在固定数据集上训练，对分布外（OOD）动作的估计存在外推误差
- 这种不准确的值估计不仅降低离线性能，还会在后续在线微调中产生负面传导
- 现有方法（如 Cal-QL 的值下界约束、Zhang et al. 的值扰动）均依赖**同一个值函数**贯穿离线和在线阶段，难以根本解决问题

本文提出两个关键研究问题：(1) 新增值函数能否解决性能提升缓慢的问题？(2) 如何最好地利用新值函数？

## 方法详解

OPT（Online Pre-Training for Offline-to-Online RL）的核心思想是引入一个全新值函数 $Q^{\text{on-pt}}$，并设计三阶段训练流程：

### 阶段一：离线预训练

与传统离线 RL 相同，在离线数据集 $\mathcal{B}_{\text{off}}$ 上联合训练值函数 $Q^{\text{off-pt}}$ 和策略 $\pi^{\text{off}}$。使用 TD3+BC 或 SPOT 作为骨干算法，策略损失为：

$$\mathcal{L}_\pi(\phi) = \mathbb{E}_{s \sim B}\left[-Q_\theta(s, \pi_\phi(s)) + \alpha(\pi_\phi(s) - a)^2\right]$$

### 阶段二：在线预训练（核心创新）

初始化一个全新的值函数 $Q^{\text{on-pt}}_\psi$，通过元适应目标在离线数据和少量在线样本上进行预训练：

1. **数据设计**：先用离线策略 $\pi^{\text{off}}$ 收集 $N_\tau$ 个在线样本存入 $\mathcal{B}_{\text{on}}$，同时使用 $\mathcal{B}_{\text{off}}$ 和 $\mathcal{B}_{\text{on}}$ 训练
2. **元适应目标函数**：

$$\mathcal{L}_{Q^{\text{on-pt}}}^{\text{pretrain}}(\psi) = \mathcal{L}_{Q^{\text{on-pt}}}^{\text{off}}(\psi) + \mathcal{L}_{Q^{\text{on-pt}}}^{\text{on}}\left(\psi - \alpha \nabla \mathcal{L}_{Q^{\text{on-pt}}}^{\text{off}}(\psi)\right)$$

第一项在离线数据上学习，第二项确保 $Q^{\text{on-pt}}$ 在离线数据上梯度更新一步后能快速适应在线数据。这种 MAML 风格的元学习使新值函数天然具备对在线样本的快速适应能力。

### 阶段三：在线微调

策略利用两个值函数的加权组合进行改进：

$$\mathcal{L}_\pi^{\text{finetune}}(\phi) = \mathbb{E}_{s \sim B}\left[-\left\{(1-\kappa) Q^{\text{off-pt}}(s, \pi_\phi(s)) + \kappa Q^{\text{on-pt}}(s, \pi_\phi(s))\right\}\right]$$

其中 $\kappa \in (0, 1]$ 是权重系数，**动态调度**：训练初期偏向可靠的 $Q^{\text{off-pt}}$，随着在线微调推进逐步增大 $\kappa$ 以利用适应更快的 $Q^{\text{on-pt}}$。对于低质量数据集（如 random），从一开始就主要依赖 $Q^{\text{on-pt}}$。

## 实验关键数据

在 D4RL 基准（MuJoCo / Antmaze / Adroit）上，离线阶段 1M 步，在线阶段 300k 步（其中 OPT 用前 25k 步做在线预训练）。

### MuJoCo 结果（TD3+OPT vs 基线，10 seeds）

| 环境 | TD3 | Off2On | Cal-QL | **TD3+OPT** |
|------|-----|--------|--------|-------------|
| halfcheetah-r | 94.6 | 92.8 | 32.2 | **90.2** |
| hopper-r | 86.0 | 94.5 | 10.3 | **108.7** |
| walker2d-r | 0.1 | 29.4 | 10.9 | **88.0** |
| halfcheetah-m | 93.4 | 103.3 | 69.9 | **97.0** |
| hopper-m | 89.3 | 108.4 | 102.3 | **112.2** |
| walker2d-m | 103.5 | 112.3 | 96.1 | **116.1** |
| **Total** | 752.4 | 860.9 | 586.8 | **939.1** |

### Antmaze 结果（SPOT+OPT vs 基线）

| 环境 | SPOT | Cal-QL | **SPOT+OPT** |
|------|------|--------|--------------|
| umaze | 98.7 | 90.8 | **99.7** |
| umaze-diverse | 55.9 | 75.2 | **97.7** |
| medium-play | 91.1 | 94.6 | **97.6** |
| large-diverse | 70.0 | 72.9 | **90.1** |
| **Total** | 465.7 | 503.4 | **565.3** |

### Adroit 结果（SPOT+OPT）

| 环境 | SPOT | Cal-QL | **SPOT+OPT** |
|------|------|--------|--------------|
| pen-cloned | 114.8 | 0.21 | **136.2** |
| hammer-cloned | 84.0 | 0.23 | **121.9** |
| door-cloned | 1.6 | -0.33 | **51.1** |
| **Total** | 200.2 | -0.23 | **309.1** |

OPT 在所有三个领域均取得 SOTA，IQM 指标显示 95% 置信区间不重叠，统计显著。

## 亮点与洞察

- **新颖视角**：不修复旧值函数，而是引入全新值函数——简洁但有效地跳出离线值估计偏差的困局
- **三阶段框架**：在传统两阶段之间插入"在线预训练"，用少量在线样本（25k 步）即可让新值函数做好适应准备
- **元适应训练**：MAML 风格的目标函数使 $Q^{\text{on-pt}}$ 不是简单拟合数据，而是学会快速适应在线分布变化
- **强通用性**：OPT 可作为即插即用模块应用于 TD3、SPOT、IQL 等多种骨干算法
- **κ 自适应调度**：根据数据集质量（random vs medium）自动调整两个值函数的权重，体现了对数据分布的敏感设计
- **walker2d-random 提升惊人**：从基线最高 38.8 直接跃升到 88.0，说明新值函数在分布偏移严重时优势巨大

## 局限与展望

- **超参数调节**：$\kappa$ 的调度策略需要根据数据集质量手动设定，缺乏自动化的 $\kappa$ 适应机制
- **额外计算开销**：维护两个值函数 + 元适应梯度计算，增加计算和内存成本
- **在线预训练步数固定**：25k 步的在线预训练对所有环境统一使用，可能对部分场景非最优
- **仅验证基于值函数的方法**：未探索 SAC 等基于策略梯度的算法适配
- **离线数据集假设**：实验仅在 D4RL 上进行，真实场景的复杂离线数据分布有待验证
- **$N_\tau$ 样本效率**：在线预训练阶段收集样本但不更新策略，存在一定样本浪费

## 相关工作与启发

- **Cal-QL** (Nakamoto et al., 2024)：值下界约束方法，但保守性限制了策略改进潜力
- **OEMA** (Guo et al., 2023)：元适应思路的来源，OPT 将其从策略适应推广到值函数适应
- **Off2On** (Lee et al., 2022)：平衡重放策略，OPT 同样采用但配合双值函数更有效
- **PEX** (Zhang et al., 2023)：多策略集成思路，与 OPT 的多值函数思路正交互补
- **IQL** (Kostrikov et al., 2021)：通过扩展验证了 OPT 的通用性

## 评分

- 新颖性: ⭐⭐⭐⭐ — 引入"在线预训练"第三阶段和双值函数设计，视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 三大域全覆盖 + 多骨干验证 + 详细消融 + IQM 统计
- 写作质量: ⭐⭐⭐⭐ — 问题驱动的清晰叙事，图表丰富
- 价值: ⭐⭐⭐⭐ — 即插即用的通用模块，对 offline-to-online RL 有实际推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[ICML 2025\] Reward-free World Models for Online Imitation Learning](reward-free_world_models_for_online_imitation_learning.md)
- [\[ICML 2025\] Continual Reinforcement Learning by Planning with Online World Models](continual_reinforcement_learning_by_planning_with_online_world_models.md)
- [\[ICML 2025\] Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration](leveraging_skills_from_unlabeled_prior_data_for_efficient_online_exploration.md)
- [\[ICML 2025\] Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability](non-stationary_online_learning_for_curved_losses_improved_dynamic_regret_via_mix.md)

</div>

<!-- RELATED:END -->
