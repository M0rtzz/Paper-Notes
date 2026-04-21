---
title: >-
  [论文解读] EFX and PO Allocation Exists for Two Types of Goods
description: >-
  [AAAI2026][AI安全][fair division] 证明了当物品只有两种类型且所有估值为正时，满足 EFX（任意物品无嫉妒）和 Pareto 最优的分配总是存在的，并给出了准线性时间算法。
tags:
  - AAAI2026
  - AI安全
  - fair division
  - EFX
  - Pareto optimality
  - indivisible goods
  - additive valuations
---

# EFX and PO Allocation Exists for Two Types of Goods

**会议**: AAAI2026  
**arXiv**: [2601.03438](https://arxiv.org/abs/2601.03438)  
**代码**: 无（理论工作）  
**领域**: ai_safety  
**关键词**: fair division, EFX, Pareto optimality, indivisible goods, additive valuations  

## 一句话总结

证明了当物品只有两种类型且所有估值为正时，满足 EFX（任意物品无嫉妒）和 Pareto 最优的分配总是存在的，并给出了准线性时间算法。

## 背景与动机

不可分物品的公平分配是算法博弈论和计算社会选择的核心研究问题。经典的无嫉妒性（Envy-Freeness, EF）要求每个代理都不偏好他人的物品束，但对于不可分物品，EF 分配往往不存在，即使存在也是 NP 完全的。

EFX（Envy-Freeness up to Any good）是 EF 的一个重要松弛：允许代理嫉妒他人，但从被嫉妒的物品束中移除**任意一件**物品后嫉妒就会消失。EFX 是否对任意数量代理总是存在，至今仍是公平分配领域的核心开放问题。

已有工作证明了 EFX 在多种受限场景下存在：相同估值、两个代理、有序实例、至多三种估值、两种物品类型等。其中 Garg & Murhekar 证明了两种物品类型时 EFX 存在，但未保证 Pareto 最优性。本文的动机正是填补这一空白——同时保证**公平性**（EFX）和**效率**（PO）。

## 核心问题

**主要研究问题**：当物品只有两种类型（每种有多个副本）、代理具有加性正估值时，是否总存在同时满足 EFX 和 Pareto 最优的分配？

**形式化设定**：

- $n$ 个代理，物品集 $M$ 包含 $m_1$ 个第一类物品和 $m_2$ 个第二类物品
- 每个代理 $i$ 对物品类型 $j$ 的估值为 $v_{i,j} > 0$（正估值假设）
- 分配 $X = (X_1, \ldots, X_n)$ 是 $M$ 的划分
- **EFX**：对任意 $i, j$ 及 $g \in X_j$，$v_i(X_i) + v_i(g) \geq v_i(X_j)$
- **PO**：不存在使某人更好而无人更差的再分配

## 方法详解

### 1. 输入预处理

算法首先对输入进行标准化处理：

- **分组**：将代理分为两组——偏好第一类物品的（$v_{i,2} \leq v_{i,1}$）和偏好第二类的
- **归一化**：将每个代理的估值标准化为 $(1, v_{i,2}/v_{i,1})$，使第一类物品的估值统一为 1
- **排序**：按 $v_{i,2}$ 从小到大排列代理
- **约束保证**：确保 $m_1/n_1 \geq m_2/n_2$，否则交换物品类型

### 2. Proper 分配与 Pareto 最优

作者引入 **proper allocation** 的概念：存在分割点 $t$ 使得 $t$ 之前的代理只收第一类物品，$t$ 之后的代理收到的第一类物品数量有上界。核心定理证明所有 proper allocation 都是 Pareto 最优的，这为后续构造提供了效率保证。

### 3. Split 分配

核心分配构造是 **$(t, k)$-split-allocation**：

- 代理 $t$ 是"分割点"，同时接收两类物品（获得 $k$ 个第二类物品）
- 代理 $1, \ldots, t-1$ 只接收第一类物品（均等分配）
- 代理 $t+1, \ldots, n$ 只接收第二类物品（均等分配）
- 剩余第一类物品在代理 $1, \ldots, t$ 之间分配

分配使用 **Prioritized Equitable Allocation (PEA)** 机制——在指定代理间几乎均等分配，高优先级代理获得更多。每个 split 分配都是 proper 的，因此自动满足 PO。

### 4. 嫉妒方向分析

关键发现：如果 split 分配不是 EFX，嫉妒方向是**单向**的。定义：

- **左嫉妒（LE）**：存在编号较大的代理嫉妒编号较小的代理
- **右嫉妒（RE）**：存在编号较小的代理嫉妒编号较大的代理

**定理 2** 证明任一 split 分配不可能同时左嫉妒和右嫉妒。**定理 3** 进一步证明：最小的 split 分配要么 EFX 要么仅 LE，最大的要么 EFX 要么仅 RE。

### 5. 二分搜索与 Reallocation

如果所有 split 分配都不是 EFX，则嫉妒方向必然在某处从 LE 转为 RE。算法通过**二分搜索**找到这个"嫉妒方向翻转点" $(t, k)$。

在翻转点处，构造 **$(t, k)$-reallocation**：在 split 分配基础上，从左侧代理重新分配部分第一类物品给右侧代理（具体分配量由 $\lceil d \cdot v_{t,2} \rceil$ 控制）。**定理 4** 证明这种 reallocation 同时满足 EFX 和 PO。

### 6. 时间复杂度

整体算法运行时间为 $\mathcal{O}(n \log n + \log m)$；若代理已按相对估值排序，则只需 $\mathcal{O}(\log n + \log m)$。

## 实验关键数据

本文为纯理论工作，无实验数据。主要理论结果：

| 结果 | 说明 |
|------|------|
| EFX+PO 存在性 | 两类物品、正加性估值下总是存在 |
| 时间复杂度 | $\mathcal{O}(n \log n + \log m)$ |
| EFX+fPO 不相容 | 通过反例证明 EFX 与分数 PO 不可兼得 |
| 特殊情形 | $m_1 + m_2 \leq n$ 时有 trivial 解 |

## 亮点

- **填补理论空白**：首次在两类物品设定下同时保证 EFX 和 PO，增强了先前仅保证 EFX 的结果
- **算法优雅简洁**：基于 split 分配 + 嫉妒方向二分搜索 + reallocation 的三步构造非常自然
- **准线性效率**：$\mathcal{O}(n \log n + \log m)$ 的复杂度极为高效，远低于一般公平分配算法
- **Proper allocation 框架**：引入的 proper allocation 概念统一了 PO 的证明，所有中间分配和最终分配都自动满足 PO
- **嫉妒方向单调性**：发现 split 分配族中嫉妒方向的单调性质，这是二分搜索可行的关键洞察

## 局限与展望

- **物品类型受限**：仅适用于两种物品类型，距离一般情形（任意多类型或任意物品）仍有很大距离
- **正估值假设**：要求所有 $v_{i,j} > 0$，不适用于代理对某类物品估值为零的场景（已知零估值下 EFX+PO 可能不相容）
- **EFX+fPO 不可达**：虽然保证了离散 PO，但更强的分数 PO（fPO）与 EFX 仍不相容
- **加性估值限制**：仅处理加性估值，未涉及次模或更一般的估值函数
- **无实验验证**：缺乏在实际场景（如资源分配平台）上的经验性验证

## 与相关工作的对比

| 工作 | 设定 | 保证 |
|------|------|------|
| Plaut & Roughgarden (2020) | 2 个代理 | EFX |
| Chaudhury et al. (2020) | 3 个代理 | EFX |
| Mahara (2023) | 2 种估值类型 | EFX |
| Garg & Murhekar (2024) | 2 种物品类型 | EFX（无 PO 保证）|
| Amanatidis et al. (2021) | bivalued 实例 | EFX+PO |
| **本文** | **2 种物品类型、正估值** | **EFX+PO，$\mathcal{O}(n\log n + \log m)$** |

本文相比 Garg & Murhekar 的核心提升是在同一设定下额外保证了 Pareto 最优性，且算法更高效。

## 启发与关联

- **方法论启发**：嫉妒方向的单调性分析思路值得在其他公平分配问题中推广——当分配族中存在某种单调结构时，二分搜索可能高效定位 EFX 分配
- **Proper allocation 框架**可能对更多类型物品的 EFX+PO 问题有借鉴价值
- **值得关注的后续方向**：三种或更多物品类型的 EFX+PO 存在性、含零估值的松弛版本、非加性估值的推广

## 评分

- 新颖性: ⭐⭐⭐⭐ （在已有 EFX 结果上自然而有意义地添加 PO 保证）
- 实验充分度: N/A （纯理论工作）
- 写作质量: ⭐⭐⭐⭐ （构造清晰、逐步递进，前置定义充分）
- 价值: ⭐⭐⭐⭐ （推进公平分配核心开放问题，但距一般解仍有距离）

<!-- RELATED:START -->

## 相关论文

- [Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation](matrix-free_two-to-infinity_and_one-to-two_norms_estimation.md)
- [Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](../../ICLR2026/ai_safety/beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)
- [Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/ai_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [PubSub-VFL: Towards Efficient Two-Party Split Learning in Heterogeneous Environments via Publisher/Subscriber Architecture](../../NeurIPS2025/ai_safety/pubsub-vfl_towards_efficient_two-party_split_learning_in_heterogeneous_environme.md)
- [LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models](lamp_learning_universal_adversarial_perturbations_for_multi-image_tasks_via_pre-.md)

<!-- RELATED:END -->
