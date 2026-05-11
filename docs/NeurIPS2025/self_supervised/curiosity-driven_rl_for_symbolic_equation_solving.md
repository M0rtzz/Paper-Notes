---
title: >-
  [论文解读] Curiosity-driven RL for Symbolic Equation Solving
description: >-
  [NeurIPS 2025][自监督][强化学习] 将好奇心驱动探索（RND、ICM 等）与基于表达式树的图动作空间结合，使 PPO 智能体能够求解包含根号、指数和三角函数的非线性方程，超越了此前仅限于线性方程的 RL 方法。
tags:
  - NeurIPS 2025
  - 自监督
  - 强化学习
  - 符号数学
  - 好奇心驱动探索
  - PPO
  - 表达式树
---

# Curiosity-driven RL for Symbolic Equation Solving

**会议**: NeurIPS 2025  
**arXiv**: [2510.17022](https://arxiv.org/abs/2510.17022)  
**代码**: 无  
**领域**: 自监督  
**关键词**: 强化学习, 符号数学, 好奇心驱动探索, PPO, 表达式树

## 一句话总结

将好奇心驱动探索（RND、ICM 等）与基于表达式树的图动作空间结合，使 PPO 智能体能够求解包含根号、指数和三角函数的非线性方程，超越了此前仅限于线性方程的 RL 方法。

## 研究背景与动机

符号数学（如解代数方程、计算积分）传统上依赖人工规则系统（Mathematica、Maple 等），RL 有望**自主学习变换策略**并发现新颖解法。然而 RL 应用于符号数学面临两大挑战：

**状态空间组合爆炸**：每个方程可分支为大量子表达式

**动作空间大且动态**：每步可对不同子表达式施加不同代数变换，可用动作随方程形式变化

先前工作（Poesia et al. 2021）仅能解线性方程，使用原始文本作为状态和原子级操作。Dabelow & Ueda (2024) 使用 Q-learning，仍局限于线性方程 $a_0 + a_1 x = a_2 + a_3 x$。本文首次展示 RL 求解非线性方程的可行性。

## 方法详解

### 整体框架

将方程求解建模为 MDP：状态是方程（以表达式树向量化表示），动作是 (operation, term) 对，奖励基于方程复杂度变化，终止条件为 $\text{lhs} = x$。

### 关键设计

**状态表示**：方程被表示为表达式树，通过前序遍历向量化并填充至最大长度 $L=50$。运算符和变量映射为整数：$\{\text{add}:1, \text{sub}:2, \text{mul}:3, \ldots, x:5, a:6, \ldots\}$。

**动作空间**：定义为 $(O \times T)$ 笛卡尔积加上额外操作：

$$O = \{\text{add, sub, mul, div}\} \cup \{\text{square, sqrt, exp, log, sin, cos, asin, acos}\}$$

$$T = \text{SubExpr}(\text{lhs}) \cup \text{SubExpr}(\text{rhs})$$

$$A = (O \times T) \cup \{(\text{Expand}, \text{None}), (\text{collect}, x), (\text{multiply}, -1)\}$$

动作集大小上限 $|A| = 50$，非法动作（如除以零）被 mask。

**奖励函数**：基于方程复杂度（表达式树节点数 + 边数）的变化：

$$R(\text{action}) = C(\text{equation}) - C(\text{equation after action})$$

鼓励智能体采取简化方程的动作。例如 $C(ax+b) = 5 + 4 = 9$，$C(ax^2 + bx + c) = 10 + 9 = 19$。

**好奇心探索**：将四种好奇心方法与 PPO 结合——ICM（Intrinsic Curiosity Module）、RIDE、NGU（Never Give Up）和 RND（Random Network Distillation）。好奇心作为内在奖励补充稀疏的外在奖励。

### 损失函数

使用标准 PPO 损失加上各好奇心方法的内在奖励。以 RND 为例，内在奖励为：

$$r_{\text{int}} = \|f(s; \theta) - \hat{f}(s; \hat{\theta})\|^2$$

其中 $f$ 为固定随机网络，$\hat{f}$ 为可训练预测网络。

## 实验关键数据

### 主实验

**固定方程环境**（成功率，$N_{\text{trial}}=10$，$N_{\text{train}}=3\times10^6$）：

| 方程 | A* | A2C | PPO | PPO-ICM | PPO-RIDE | PPO-NGU | PPO-RND |
|------|-----|-----|-----|---------|----------|---------|---------|
| $ax+b$ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| $a/x+b$ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| $c(ax+b)+d$ | 1.00 | 0.20 | 0.10 | 0.70 | 0.60 | 0.40 | **1.00** |
| $c+d/(ax+b)$ | 0.00 | 0.00 | 0.20 | 0.50 | 0.40 | 0.50 | **0.80** |
| $(ax+b)+e(cx+d)$ | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.20 | **0.40** |
| $e+(ax+b)/(cx+d)$ | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 | 0.30 | **0.50** |

**随机方程环境**（大数据集，depth < 5，$10^4/10^3$ train/test split）：
- PPO-RND：$\text{test}_{10} \approx 0.8$，$\text{test}_{\text{greedy}} \approx 0.25$
- PPO-RIDE：$\text{test}_{10} \approx 0.8$
- 纯 PPO：在大数据集上完全失败
- PPO-NGU：出人意料地表现最差（可能因记忆开销大）

### 消融实验

- 好奇心对比：RND 在固定和随机环境均表现最佳，具有轻量计算优势
- 动作集大小对比：$|A| = 40, 70, 100$ 对性能无显著影响
- 可解方程类型：涵盖根式方程、指数方程、三角方程

### 关键发现

1. 基于表达式树的 MDP 表述有效，好奇心驱动探索是求解非简单方程的**必要条件**
2. RND 在所有好奇心方法中一致最优：轻量高效且固定/随机环境通吃
3. 当前方法仅限于"封闭"方程——求解所需操作全部包含在原表达式中

## 亮点与洞察

1. **首次用 RL 求解非线性符号方程**：包括涉及根号、指数、三角函数的方程
2. 表达式树动作空间设计巧妙——子表达式作为操作对象，使动作空间随方程自适应变化
3. 好奇心探索的必要性实验令人信服：无好奇心的 PPO/A2C 在复杂方程上完全失败
4. 提出了清晰的未来 benchmark：二次方程求解（需 completing the square 这类"开放式"推理）

## 局限性

- 仅处理"封闭"方程——求解不需要引入方程外的新项
- 无法解二次方程（需发现 $(b/2a)^2$ 这种"生成性"推理）
- Workshop paper 篇幅有限，实验规模较小
- $\text{test}_{\text{greedy}}$ 仅约 0.25，greedy policy 下泛化能力不足

## 相关工作与启发

- Poesia et al. (2021)：线性方程 + 原始文本动作空间 + 对比学习，本文拓展到非线性 + 表达式树 + PPO+好奇心
- Dabelow & Ueda (2024)：Q-learning 框架，仍限于线性方程
- 启发思考：是否可以用 AlphaZero 式的规划算法来处理需要长程推理的"开放式"方程

## 评分

- ⭐ 新颖性: 4/5 — 首次展示 RL 非线性方程求解的可行性
- ⭐ 实验充分度: 3/5 — Workshop paper 实验充分但规模有限
- ⭐ 写作质量: 4/5 — MDP 形式化清晰，局限性讨论坦诚
- ⭐ 价值: 3/5 — 探索性工作，打开新方向但距实用仍有距离

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling](../../ICML2025/self_supervised/neighbour-driven_gaussian_process_variational_autoencoders_for_scalable_structur.md)
- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](../../CVPR2026/self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)
- [\[NeurIPS 2025\] TRIDENT: Tri-Modal Molecular Representation Learning with Taxonomic Annotations and Structural Relationships](trident_tri-modal_molecular_representation_learning_with_taxonomic_annotations_a.md)
- [\[NeurIPS 2025\] Manifolds and Modules: How Function Develops in a Neural Foundation Model](manifolds_and_modules_how_function_develops_in_a_neural_foundation_model.md)

</div>

<!-- RELATED:END -->
