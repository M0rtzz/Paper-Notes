---
title: >-
  [论文解读] Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics
description: >-
  [ICLR 2026][人体理解][跨域强化学习] 提出 Q Avatar 框架，通过跨域 Bellman 一致性量化源域模型可迁移性，利用自适应无超参权重函数混合源域和目标域 Q 函数，实现在状态-动作空间不同的跨域 RL 中的可靠知识迁移，无论源域模型质量或域相似性如何都能保证不产生负迁移。
tags:
  - ICLR 2026
  - 人体理解
  - 跨域强化学习
  - Bellman一致性
  - 混合Critic
  - Q函数迁移
  - 负迁移防护
---

# Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics

**会议**: ICLR 2026  
**arXiv**: [2603.12087](https://arxiv.org/abs/2603.12087)  
**代码**: [https://rl-bandits-lab.github.io/Cross-Domain-RL/](https://rl-bandits-lab.github.io/Cross-Domain-RL/)  
**领域**: 强化学习 / 迁移学习  
**关键词**: 跨域强化学习, Bellman一致性, 混合Critic, Q函数迁移, 负迁移防护

## 一句话总结

提出 Q Avatar 框架，通过跨域 Bellman 一致性量化源域模型可迁移性，利用自适应无超参权重函数混合源域和目标域 Q 函数，实现在状态-动作空间不同的跨域 RL 中的可靠知识迁移，无论源域模型质量或域相似性如何都能保证不产生负迁移。

## 研究背景与动机

### 问题背景
跨域强化学习（CDRL）旨在利用源域采集的数据来提升目标域的学习效率。实际场景中（如不同形态的机器人之间），源域和目标域往往具有**不同的状态空间和动作空间**，这使得直接迁移不可行。

### 两大根本挑战

**状态-动作空间不一致**：源域和目标域可能有不同维度的状态和动作表示，需要复杂的域间映射

**可迁移性未知**：源域模型的迁移效果难以事先判断，CDRL 容易产生**负迁移**——即迁移后性能反而不如从头学习

### 现有方法的不足
- 手工设计的潜空间映射方法（Ammar & Taylor, 2012）缺乏灵活性
- 学习型域间映射方法（Zhang et al., 2021; Gui et al., 2023）基于动态对齐，但无性能保证且忽略可迁移性问题
- 所有现有方法都假设域足够相似，未解决负迁移防护问题

## 方法详解

### 整体框架

Q Avatar 包含三个核心组件：
1. **跨域 Bellman 一致性**：量化源域 Q 函数的可迁移性
2. **混合 Critic**：自适应加权组合源域和目标域 Q 函数
3. **Normalizing Flow 域间映射**：学习状态和动作的跨域对应关系

### 关键设计

1. **跨域 Bellman 一致性（Cross-Domain Bellman Consistency）**

   定义跨域 Bellman 误差来衡量源域 Q 函数在目标域中的适用性：

   $$\epsilon_{\text{cd}}(s,a;\phi,\psi,Q_{\text{src}},\pi) = |Q_{\text{src}}(\phi(s),\psi(a)) - r_{\text{tar}}(s,a) - \gamma \mathbb{E}_{s',a'}[Q_{\text{src}}(\phi(s'),\psi(a'))]|$$

   其中 $\phi: \mathcal{S}_{\text{tar}} \to \mathcal{S}_{\text{src}}$ 和 $\psi: \mathcal{A}_{\text{tar}} \to \mathcal{A}_{\text{src}}$ 是域间映射。如果这个误差小，说明源域 Q 函数通过映射后在目标域中满足 Bellman 方程，可迁移性强。

2. **Q Avatar 混合 Critic**

   在每一步策略更新时，使用加权组合：
   $$Q^{(t)}_{\text{avatar}} = (1-\alpha(t)) Q^{(t)}_{\text{tar}} + \alpha(t) Q_{\text{src}}(\phi^{(t)}, \psi^{(t)})$$

   权重 $\alpha(t)$ 是**自适应、无超参**的，由跨域 Bellman 误差和 TD 误差的比率决定：

   $$\alpha(t) = \frac{1/\|\epsilon_{\text{cd}}\|_{d^{\pi^{(t)}}}}{1/\|\epsilon_{\text{td}}^{(t)}\|_{d^{\pi^{(t)}}} + 1/\|\epsilon_{\text{cd}}\|_{d^{\pi^{(t)}}}}$$

   当源域 Q 的 Bellman 误差小（可迁移性强），$\alpha$ 大，多用源域知识；反之，$\alpha$ 小，主要依赖目标域自身学习。这确保了**无论源域模型质量如何都不会负迁移**。

3. **Normalizing Flow 域间映射**

   使用 normalizing flow 模型学习 $\phi$ 和 $\psi$，目标函数最小化跨域 Bellman 损失。Flow 模型保证可逆性，训练稳定。这一设计展示了 Q Avatar 框架与现有域间映射方法的兼容性。

### 收敛性保证

**定理**（非正式）：Q Avatar 的平均次优性差距上界为：
$$O\left(\frac{\log|\mathcal{A}|}{\sqrt{T}(1-\gamma)}\right) + C \cdot \min\{\|\epsilon_{\text{td}}^{(t)}\|, \|\epsilon_{\text{cd}}\|\}$$

即 Q Avatar 自动选择 TD 误差和跨域 Bellman 误差中较小的那个，保证了在任何源域模型质量下都能有效利用。

## 实验关键数据

### 主实验

评估环境覆盖运动控制、机械臂操作和目标导航：

| 环境 | 阈值 | Q Avatar | SAC (从头学) | 比率 |
|------|------|----------|------------|------|
| HalfCheetah | 6000 | 126K步 | 176K步 | 0.71 |
| Ant | 1600 | 206K步 | 346K步 | 0.59 |
| Door Opening | 90 | 48K步 | 98K步 | 0.49 |
| Table Wiping | 45 | 72K步 | 98K步 | 0.73 |
| Navigation | 20 | 218K步 | 490K步 | **0.44** |

最佳情况下仅需 SAC 44% 的环境步数达到阈值。

### 对比方法
Q Avatar 在所有任务上优于 CMD、CAT-SAC、CAT-PPO 和 FT（fine-tuning），IQM 聚合指标显著领先。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 强正迁移（对称Ant） | $\alpha(t)$ 高 | 有效利用源域知识 |
| 强负迁移（目标相反的Ant） | $\alpha(t)$ 低 | 自动防护负迁移 |
| 低质量源模型（return 1000 vs 7000） | $\alpha(t)$ 逐渐降低 | 自适应减少依赖 |
| 无关域迁移（Hopper → Table Wiping） | 不产生负迁移 | 可靠性保证 |
| 非稳态环境（噪声奖励+动作） | 仍有正迁移 | 鲁棒性 |
| $N_\alpha$ 敏感性测试 | 轻微敏感 | 300/1000/3000 均可 |

### 关键发现
- $\alpha(t)$ 能准确反映可迁移性：正迁移时高，负迁移时低
- 即使源域和目标域完全不相关（Hopper vs Table Wiping），Q Avatar 也不会负迁移
- 支持多源域迁移，权重自动分配
- 在基于图像的 DMC 任务上同样有效

## 亮点与洞察

1. **理论驱动的框架设计**：从 Bellman 一致性出发建立可迁移性的形式化定义，理论和算法设计环环相扣
2. **无超参自适应加权**：$\alpha(t)$ 完全由 Bellman 误差比率决定，无需手动调节，这是实际可用性的关键
3. **负迁移保证**：无论源域模型多差、域差异多大，性能至少不低于纯目标域学习——这是已有 CDRL 方法普遍缺乏的
4. **"Avatar"隐喻**：类比电影中人类远程控制工程化身体适应外星环境，形象表达了算法思想

## 局限与展望

- 表格式分析假设有限状态-动作空间和探索性初始分布，与实际连续控制有差距
- Normalizing flow 映射的训练质量直接影响跨域 Bellman 误差的准确估计
- 实验中源域模型都是用 SAC 预训练的，对其他算法（如 PPO）训练的源模型效果未验证
- 高维复杂任务（如灵巧手操作）中的可扩展性有待验证
- 目前仅处理单源→单目标和多源→单目标，未涉及多目标场景

## 相关工作与启发

- **CMD** (Gui et al., 2023)：通过动态循环一致性学习域间映射，但无性能保证
- **CAT** (You et al., 2022)：通过编码器-解码器学习映射，但受限于参数级迁移
- **DARC** (Eysenbach et al., 2021)：奖励增强方法，但假设相同状态-动作空间
- **Task Vectors** (Wang et al., 2020)：使用双 Q 函数进行 Q-learning 更新，但同样限于相同空间
- 启发：Bellman 一致性作为可迁移性度量的思想可推广到模仿学习、离线 RL 等场景

## 评分
- 新颖性: ⭐⭐⭐⭐ — 跨域 Bellman 一致性和自适应混合 Critic 的结合很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 多种环境、正/负迁移、多源迁移、图像任务、敏感性分析，非常全面
- 写作质量: ⭐⭐⭐⭐ — 理论清晰，实验详尽，"Avatar"命名优雅
- 价值: ⭐⭐⭐⭐⭐ — 解决了 CDRL 的负迁移核心痛点，有理论保证且实用

<!-- RELATED:START -->

## 相关论文

- [Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks](../../CVPR2026/human_understanding/stable_spike_dual_consistency_optimization_via_bitwise_and_operations_for_spikin.md)
- [From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](../../AAAI2026/human_understanding/from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks](../../AAAI2026/human_understanding/from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha.md)
- [Statistical Guarantees for Offline Domain Randomization](statistical_guarantees_for_offline_domain_randomization.md)
- [RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)

<!-- RELATED:END -->
