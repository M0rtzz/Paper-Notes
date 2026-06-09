---
title: >-
  [论文解读] Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning
description: >-
  [ICLR 2026][强化学习][continual RL] 受人脑海马体-大脑皮层交互机制启发，提出 FAME 双学习器框架，通过快速学习器进行知识迁移、元学习器进行知识整合，在原则性地最小化灾难性遗忘的前提下实现高效的持续强化学习。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "continual RL"
  - "catastrophic forgetting"
  - "knowledge transfer"
  - "dual-learner"
  - "meta learning"
---

# Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2603.00903](https://arxiv.org/abs/2603.00903)  
**代码**: [GitHub](https://github.com/datake/FAME)  
**领域**: 强化学习  
**关键词**: continual RL, catastrophic forgetting, knowledge transfer, dual-learner, meta learning

## 一句话总结

受人脑海马体-大脑皮层交互机制启发，提出 FAME 双学习器框架，通过快速学习器进行知识迁移、元学习器进行知识整合，在原则性地最小化灾难性遗忘的前提下实现高效的持续强化学习。

## 研究背景与动机

**持续学习的核心挑战**：传统深度 RL 算法针对单任务设计，面对连续到来的任务序列时，需要在可塑性（快速适应新任务）和稳定性（保留旧知识）之间取得平衡。

**现有方法缺乏理论基础**：现有持续 RL 方法多基于启发式或不同角度的独立设计，缺乏统一的理论框架来分析知识迁移何时有效、如何量化遗忘。

**负迁移问题**：直接 finetune 在任务差异大时会导致性能下降（负迁移），而从头训练（reset）又无法利用先前积累的知识。

**脑科学启发**：人脑中海马体负责快速编码新经验，大脑皮层负责渐进式知识整合，这种分工协作机制为算法设计提供了生物学基础。

**多任务 RL 的局限**：传统多任务 RL 通过最大化平均回报来共享知识，无法显式控制灾难性遗忘。

**缺乏 MDP 距离度量**：此前没有原则性的方法来量化不同环境之间的相似度，难以判断知识迁移是否有益。

## 方法详解

### 整体框架

FAME（FAst and MEta knowledge learner）把持续 RL 拆成两个耦合学习器：类比海马体的**快速学习器**在新任务上做快速学习，类比大脑皮层的**元学习器**做渐进式知识整合。新任务到来时，元学习器先为快速学习器挑选热启动策略以迁移先验知识；任务结束后，快速学习器学到的新策略被增量整合回元学习器，使其在原则性最小化灾难性遗忘的同时不断积累跨任务知识。

### 关键设计

**1. MDP 距离度量：让"环境有多像"变成可量化的量。** 持续 RL 中知识迁移是否有益、新任务会干扰旧任务多少，都取决于环境间的相似度，而此前缺乏原则性的度量。FAME 用两个 MDP 对应的最优解之间的差异来定义距离——既可以用最优 $Q$ 函数的 $\ell_2$ 距离，也可以用最优策略之间的 KL 散度。有了这个度量，后续的迁移决策和遗忘分析就有了统一的几何基础：距离近的任务适合复用旧知识，距离远的任务则提示要警惕负迁移。

**2. 灾难性遗忘的形式化定义：把直觉变成可优化的目标。** 遗忘长期停留在"学新忘旧"的直觉层面，难以直接优化。FAME 将其定义为：在**旧任务的状态访问分布**下，新旧策略（或 $Q$ 函数）之间的加权差异。用旧策略的状态访问分布作权重是关键——它让度量聚焦于旧任务中真正高频、重要的状态-动作对，而非整个状态空间的均匀差异。这样遗忘就从一个概念变成了一个明确的数学目标（Eq. 4），算法可以直接对它做优化。

**3. 增量元学习器更新（Proposition 1）：知识整合等价于最大似然。** 直接保存所有历史任务的 $Q$ 函数随任务数线性膨胀、不可扩展，因此需要一个增量规则。FAME 证明：在 KL 散度下最小化策略级遗忘目标，等价于对元学习器做最大似然估计（MLE），即让元学习器去拟合所有已遇环境状态-动作分布的混合。这一等价把持续 RL 与多任务 RL 直接连了起来，也给出了只需维护单个元策略、逐任务更新的可行方案。之所以采用策略级而非 $Q$ 值级定义，是因为前者对不同尺度的奖励更鲁棒、训练也更稳定。

**4. 自适应元热启动（Adaptive Meta Warm-up）：用假设检验回避负迁移。** 当新旧任务差异大时，直接拿旧知识初始化反而拖慢学习（负迁移），但完全放弃先验又浪费了已有积累。FAME 在每个新任务的早期交互阶段，对三种候选初始化——元学习器、上一个快速学习器、随机初始化——分别做策略评估，再用 one-vs-all 假设检验挑出统计上表现最好的那个作为热启动。这等于用统计学的严格框架取代启发式规则：实验中对已知环境约 95% 选择元热启动，对全新环境则更多回退到随机初始化。

**5. Wasserstein 距离下的策略整合（FAME-WD）：连续动作空间的几何感知整合。** 在连续控制中策略常用高斯分布表示，此时 KL 散度对分布的几何结构不敏感。FAME-WD 改用 2-Wasserstein 距离衡量策略差异——高斯之间的 2-Wasserstein 距离有闭式解，既能高效计算元学习器的增量更新，又能利用数据空间的几何结构，对复杂策略分布比 KL 更贴合，实验中也确实略优于 FAME-KL。

### 损失函数 / 训练策略

知识整合阶段最小化策略级灾难性遗忘目标（Eq. 4），等价于最大化元学习器对所有历史状态-动作分布的对数似然。在值函数方法中，元热启动以行为克隆正则化的形式引导早期探索，目标写作 $L(Q_k)=L_0(Q_k)+\lambda\,\mathbb{E}[\mathrm{KL}(\pi_M\,\|\,\pi_Q)]$，把元策略 $\pi_M$ 当作专家约束新 $Q$ 函数 $\pi_Q$。为估计整合目标中旧任务的状态访问权重，FAME 维护一个**元缓冲区**，仅在每个任务最后 $N$ 步收集状态-动作对（约占训练数据 1–2%），开销很小。

## 实验关键数据

### 主实验

**MinAtar 结果（10 sequences × 3 seeds）**：

| 方法 | Avg. Perf ↑ | FT ↑ | Forgetting ↓ |
|------|------------|------|-------------|
| Reset | 6.51 ± 1.67 | 0.74 ± 0.38 | 1.31 ± 0.23 |
| Finetune | 10.62 ± 2.75 | 0.89 ± 0.49 | 1.26 ± 0.32 |
| LargeBuffer | 10.71 ± 2.84 | 1.16 ± 0.59 | 1.65 ± 0.33 |
| **FAME** | **14.54 ± 0.58** | **1.69 ± 0.17** | **0.72 ± 0.13** |

**Meta-World 结果（3 sequences × 10 seeds）**：

| 方法 | Avg. Perf ↑ | FT ↑ | Forgetting ↓ |
|------|------------|------|-------------|
| Reset | 0.093 ± 0.017 | 0.000 | 0.710 ± 0.030 |
| PackNet | 0.491 ± 0.025 | -0.194 | 0.000 |
| FAME-KL | 0.733 ± 0.026 | 0.022 | 0.073 ± 0.019 |
| **FAME-WD** | **0.767 ± 0.024** | -0.003 | **0.023 ± 0.015** |

### 消融实验

**自适应元热启动的选择比例**：

| 到达环境类型 | Meta Warm-up | Reset | Finetune |
|-------------|-------------|-------|----------|
| 已知环境 | 95.1% | 低 | 低 |
| 未知环境 | 低 | 较高 | 低 |

**Atari 游戏结果**：

| 方法 | Freeway Avg. Perf | SpaceInvader Avg. Perf |
|------|------------------|----------------------|
| Reset | 0.16 | 0.10 |
| PackNet | 0.41 | 0.47 |
| ProgressiveNet | 0.39 | 0.61 |
| **FAME** | **0.90** | **0.96** |

### 关键发现

1. FAME 在所有基准上都显著优于基线方法，平均性能最高且方差最小，表现最为稳定。
2. 自适应元热启动能够正确识别已知/未知环境：对已知环境 95.1% 的概率选择元热启动，对新环境更多选择随机初始化。
3. FAME-WD 在连续动作空间中略优于 FAME-KL，验证了 Wasserstein 距离在复杂策略分布上的优势。
4. 虽然 PackNet 通过存储模型参数可以实现零遗忘，但它需要预知任务数量和 task ID，实用性受限。

## 亮点与洞察

1. **理论贡献扎实**：首次为持续 RL 提供了 MDP 距离和灾难性遗忘的形式化定义，将算法设计建立在坚实的理论基础上。
2. **生物学启发自然且有效**：海马体-皮层的双系统类比不是简单的比喻，而是直接映射到了算法的两个组件及其交互方式。
3. **知识整合 = MLE**：最小化策略级灾难性遗忘等价于最大似然估计，建立了持续 RL 与多任务 RL 之间的深刻联系。
4. **假设检验选择热启动**：用统计学的严格框架解决负迁移问题，而非依赖启发式规则。
5. **同时适用于值函数和策略方法**：框架的通用性强，不局限于特定的 RL 算法。

## 局限与展望

1. **假设限制**：要求相同的状态和动作空间、已知任务边界，这在实际应用中不一定成立。
2. **元缓冲区大小**：虽然只存储少量数据，但随任务数增加仍会线性增长。
3. **可扩展性**：假设检验在任务切换频率高时可能增加计算开销。
4. **未来方向**：学习潜在表示替代直接的策略/值函数存储；在线假设检验处理未知任务边界；上下文嵌入增强知识迁移。

## 相关工作与启发

- 与 **PackNet** 和 **ProgressiveNet** 相比：这些方法通过存储参数/网络结构避免遗忘，但不灵活且不具备知识迁移能力；FAME 通过原则性的知识整合同时解决遗忘和迁移。
- 与 **PT-DQN** 的差异：PT-DQN 的永久值函数知识保留能力有限；FAME 的元学习器通过显式优化灾难性遗忘目标更有效。
- **启发**：将理论驱动的方法与脑科学启发结合，可以为 RL 算法设计提供更有原则的指导。灾难性遗忘的形式化可以推广到其他持续学习场景。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 理论贡献（MDP 距离、遗忘度量）有创新性，双学习器框架虽受启发于 CLS 理论但在 RL 中的形式化新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ 涵盖像素级和连续控制两类基准、值函数和策略梯度两类算法，统计结果充分（30 seeds）
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，理论推导严谨，从基础定义到算法设计的逻辑链条完整
- **价值**: ⭐⭐⭐⭐ 为持续 RL 提供了理论基础和实用算法，弥合了理论与实践的差距

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/continual_knowledge_adaptation_for_reinforcement_learning.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICML 2026\] Position: Deployed Reinforcement Learning should be Continual](../../ICML2026/reinforcement_learning/position_deployed_reinforcement_learning_should_be_continual.md)
- [\[ICML 2025\] Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning](../../ICML2025/reinforcement_learning/position_lifetime_tuning_is_incompatible_with_continual_reinforcement_learning.md)
- [\[ICLR 2026\] ReMoT: Reinforcement Learning with Motion Contrast Triplets](remot_reinforcement_learning_with_motion_contrast_triplets.md)

</div>

<!-- RELATED:END -->
