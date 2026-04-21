---
title: >-
  [论文解读] Approximating Shapley Explanations in Reinforcement Learning
description: >-
  [NeurIPS 2025][Shapley值] 提出 FastSVERL，一种可扩展的参数化学习框架，分别近似强化学习中 Shapley 值的两个计算瓶颈（特征函数和 Shapley 求和），支持离策略数据学习和随策略演化持续更新解释。
tags:
  - NeurIPS 2025
  - Shapley值
  - 强化学习可解释性
  - 特征归因
  - 参数化近似
  - 离策略学习
---

# Approximating Shapley Explanations in Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2511.06094](https://arxiv.org/abs/2511.06094)  
**代码**: [有](https://github.com/djeb20/fastsverl)  
**领域**: Reinforcement Learning / Interpretability  
**关键词**: Shapley值, 强化学习可解释性, 特征归因, 参数化近似, 离策略学习

## 一句话总结

提出 FastSVERL，一种可扩展的参数化学习框架，分别近似强化学习中 Shapley 值的两个计算瓶颈（特征函数和 Shapley 求和），支持离策略数据学习和随策略演化持续更新解释。

## 研究背景与动机

Shapley 值为 RL 中的特征归因提供了有理论保障（公平性、一致性）的原则性框架。SVERL 框架定义了三种解释目标：**行为**（特征如何影响动作选择）、**结果**（特征如何影响期望回报）和**预测**（特征如何影响价值预测）。

然而，精确计算 Shapley 值的代价是 $\mathcal{O}(2^{|\mathcal{F}|} \cdot |\mathcal{S}|)$——对每个输入需遍历所有特征子集，每个子集需在状态空间上求期望。在高维RL问题中这完全不可行。

RL 相比监督学习还有额外挑战：(1) 解释需跨多步轨迹的时间依赖，(2) 策略不断演化时解释需同步更新，(3) 环境交互受限时需从离策略数据中学习解释。

## 方法详解

### 整体框架

FastSVERL 将近似问题分解为两个层次：

1. **近似特征函数**（characteristic function）：学习参数化模型预测"给定部分特征时的条件期望"
2. **近似 Shapley 求和**：学习参数化模型直接预测所有特征的 Shapley 值，类比 FastSHAP

两个层次可以分别用于行为/结果/预测三种解释目标，且共享相同的模型架构和训练流程。

### 关键设计

1. **参数化特征函数近似**

    功能：训练参数模型 $\hat{\pi}(s, a | \mathcal{C}; \beta)$ 来近似行为特征函数 $\tilde{\pi}_s^a(\mathcal{C})$（给定特征子集 $\mathcal{C}$ 时的条件动作概率）。

    核心思路：将不在 $\mathcal{C}$ 中的特征替换为支撑集外的值，训练最小化 $\mathcal{L}(\beta) = \mathbb{E}_{p^\pi(s)} \mathbb{E}_{\text{Unif}(a)} \mathbb{E}_{p(\mathcal{C})} |\pi(s,a) - \hat{\pi}(s,a|\mathcal{C};\beta)|^2$。因为不同状态在相同子集 $\mathcal{C}$ 上共享遮蔽表示，模型无法恢复精确目标，而是学习其均值——即特征函数值。结果特征函数在 $p^\pi(s)>0$ 的所有 $(s,a,\mathcal{C})$ 上是精确无偏的。

    设计动机：相比蒙特卡洛采样（不跨状态泛化且需策略变化时重新计算），参数化模型将近似开销摊销到所有状态和特征子集上。

2. **条件策略 + 参数化价值函数近似结果特征函数**

    功能：定义条件策略 $\hat{\pi}(a|s; s_e, \mathcal{C})$（在待解释状态 $s_e$ 用特征函数行为，其他状态用原策略），然后训练参数化价值函数 $V(s|s_e, \mathcal{C}; \beta)$ 来估计条件策略下的期望回报。

    核心思路：结果特征函数 $\tilde{v}_s^\pi(\mathcal{C})$ 需要在 $2^{|\mathcal{F}|} \times |\mathcal{S}|$ 个不同的 $(s_e, \mathcal{C})$ 对上求解独立RL问题。通过单一条件策略和参数化价值函数将所有这些问题统一，提供在策略（Eq. 14）和离策略（Eq. 15）两种训练方式。

    设计动机：结果解释是 RL 特有的挑战——需要评估"部分信息下行动的长期后果"，无法用监督学习方法直接处理。

3. **消除特征模型的单样本近似**

    功能：用单一采样的特征值直接替换预训练的特征函数模型，将特征估计嵌入 Shapley 模型训练中。

    核心思路：在 Shapley 损失中将 $\tilde{\pi}_s^a(\mathcal{C})$ 替换为 $\pi(s', a)$（其中 $s' \sim p^\pi(\cdot | s^\mathcal{C})$），得到新损失 $\mathcal{L}(\theta) = \mathbb{E}_{p^\pi(s)} \mathbb{E}_{p(\mathcal{C})} \mathbb{E}_{s' \sim p^\pi(\cdot|s^\mathcal{C})} |\pi(s',a) - \pi_{s,a}(\emptyset) - \sum_{i \in \mathcal{C}} \hat{\phi}^i(s,a;\theta)|^2$。作者证明此损失在全局最优处恢复精确无偏 Shapley 值。

    设计动机：特征模型的训练是主要计算瓶颈（约占 50% 算力），单样本近似用更高的梯度方差换取消除预训练开销和误差传播。实验中将总训练时间减半且精度更高。

### 损失函数 / 训练策略

- Shapley 模型采用 FastSHAP 的加权最小二乘目标（Eq. 10），子集 $\mathcal{C}$ 按 Shapley 权重分布采样（Eq. 6）
- 效率约束通过事后校正强制：$\phi^i \leftarrow \hat{\phi}^i + \frac{1}{|\mathcal{F}|}(\pi(s,a) - \tilde{\pi}_s^a(\emptyset) - \sum_j \hat{\phi}^j)$
- 离策略学习使用重要性采样校正分布不匹配：权重 $\frac{\pi(s_t, a_t)}{\pi_t(s_t, a_t)}$
- 持续学习通过联合更新智能体和解释模型，更新比率 10:1 或 50:1 效果最佳

## 实验关键数据

### 主实验（表格）

**大规模 Mastermind 域的收敛性（10 次运行）**：

| 领域 | 模型 | 收敛步数 | 最终损失 |
|------|------|----------|----------|
| Mastermind-443 (24特征, ≥4.3×10⁷状态) | 特征模型 | (1.10±0.11)×10⁶ | (3.83±0.02)×10⁻³ |
| | Shapley模型 | (7.31±0.68)×10⁵ | (1.30±0.04)×10⁻³ |
| Mastermind-463 (36特征, ≥2.8×10¹¹状态) | 特征模型 | (1.18±0.12)×10⁶ | (3.70±0.01)×10⁻³ |
| | Shapley模型 | (7.12±0.51)×10⁵ | (1.88±0.04)×10⁻³ |

两个关键发现：(1) 收敛稳定可靠；(2) 所需训练步数不随状态/特征数增长而增加。

### 消融实验（表格）

**单样本 vs 模型基 vs 精确特征值（Mastermind-222, 行为Shapley）**：

| 方法 | 总训练步数 | 最终Shapley MSE |
|------|-----------|----------------|
| 精确特征值 | 基准 | 最低（收敛稍慢） |
| 模型基特征（标准） | 2× 基准 | 较高（误差传播） |
| **单样本近似** | **1× 基准** | **与精确值相当** |

单样本方法在特征模型开始Shapley训练之前就已收敛，总训练时间减半。

### 关键发现

- **Hypercube 扩展性**：固定特征数时，训练开销随状态数近似多项式增长（log-log 尺度近线性）；固定状态数时，增加特征数对训练开销几乎无影响
- **离策略学习**：使用重要性采样的训练缓冲区降低了近似误差，但未达到在策略基线的水平
- **持续学习**：联合训练时更新比率 10:1 即可使解释模型跟上策略变化，避免策略大幅更新导致的误差尖峰
- **可解释性验证**：Mastermind-463 的行为解释显示最近猜测对下一步动作贡献最大，未使用槽位贡献为零（满足 Shapley 虚空性公理）

## 亮点与洞察

- 将 FastSHAP 的摊销化近似思想系统性地扩展到 RL 的三种解释目标，处理了 RL 特有的时间依赖、离策略和策略演化问题
- 单样本近似的理论贡献特别优雅——用更高方差换取消除整个计算阶段，且可直接推广到监督学习
- 为 RL 可解释性建立了完整的理论-实践框架，后续工作可以在此基础上扩展

## 局限与展望

- 实验限于离散动作空间的表格型/小规模域，未在连续控制或像素观测等高维场景验证
- 离策略重要性采样在高维或长horizon场景中可能产生高方差
- 缺少用户研究来验证解释的可理解性和实用性
- 稳态分布的近似在高维连续状态空间中仍是开放问题

## 相关工作与启发

- 建立在 SVERL (Beechey et al., 2023/2025) 的理论框架上，是其首个可扩展的实用化方案
- 与 FastSHAP (Jethani et al., 2021) 的关系：将单步预测的 Shapley 近似扩展到多步序列决策
- 启发方向：单样本近似消除特征模型的思路可直接回补到监督学习中改进 FastSHAP

## 评分

⭐⭐⭐⭐ 理论优雅、框架完整，系统性地解决了 Shapley 值在 RL 中的扩展性问题，单样本近似是亮眼贡献，但实验规模偏小。

<!-- RELATED:START -->

## 相关论文

- [Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)
- [Learning to Clean: Reinforcement Learning for Noisy Label Correction](learning_to_clean_reinforcement_learning_for_noisy_label_correction.md)
- [Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [Towards Provable Emergence of In-Context Reinforcement Learning](towards_provable_emergence_of_in-context_reinforcement_learning.md)

<!-- RELATED:END -->
